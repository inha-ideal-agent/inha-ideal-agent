"""검색 아카이브 + 자기개선 루프 — SQLite 저장/검색/라벨 export.

구현 계약:

class Archive:
    def __init__(self, db_path: str | Path = config.DB_PATH): ...
        '''스키마 자동 생성(멱등). 스레드 안전을 위해 커넥션은 호출마다 열고 닫는다.
        스키마 개요:
          inspections(record_id PK, created_at, film_id, block, weld_id, joint_type,
                      thickness_mm, quality_level, inspector, overall_passed INT,
                      defect_count INT, report_text, report_source, image_name,
                      image_w, image_h, payload_json TEXT)
          defects(id PK AUTOINC, record_id FK, defect_id, defect_type, source, status,
                  size_mm REAL NULL, passed INT NULL, clause TEXT,
                  x1 REAL, y1 REAL, x2 REAL, y2 REAL, confidence REAL)
        payload_json = InspectionRecord.to_json() 전문 (재구성용).'''

    def save(self, record: InspectionRecord) -> None:
        '''승인 기록 저장. 같은 record_id 재저장 시 교체(REPLACE).'''

    def get(self, record_id: str) -> InspectionRecord | None: ...

    def search(self, *, block: str = "", weld_id: str = "", defect_type: str = "",
               passed: bool | None = None, text: str = "",
               date_from: str = "", date_to: str = "") -> pd.DataFrame:
        '''필터 검색 → 요약 DataFrame(record_id, created_at, film_id, block, weld_id,
        quality_level, overall_passed, defect_count, defect_types(콤마 결합), inspector).
        text는 report_text/film_id/weld_id/block LIKE 검색. 빈 필터는 무시.
        최신순 정렬. ("3번 블록 수직 이음 기공 이력" 3초 검색 시나리오)'''

    def stats(self) -> dict:
        '''자기개선 루프 대시보드용:
        {'total': int, 'passed': int, 'failed': int,
         'defect_type_counts': dict[str,int],           # accepted만
         'ai_proposed': int, 'ai_accepted': int, 'ai_rejected': int,
         'human_added': int,                            # AI가 놓쳐 사람이 추가한 것 = 미탐 신호
         'acceptance_rate': float | None}'''

    def export_yolo_labels(self, out_dir: str | Path) -> dict:
        '''승인(accepted) 결함들을 YOLO 학습 라벨(txt)로 export.
        클래스 매핑은 models.DEFECT_TYPES 키 순서 기반 classes.txt 도 함께 기록.
        각 record → <image_name의 stem>.txt, 각 줄 "cls cx cy w h" (0~1 정규화,
        image_size 기준). 반환 {'records': n, 'labels': m, 'out_dir': str}.
        → 판독원의 승인 기록이 곧 라벨링 데이터가 되는 자기개선 루프의 출구.'''
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from rtworkbench import config
from rtworkbench.models import DEFECT_TYPES, InspectionRecord

# YOLO export 클래스 순서: DEFECT_TYPES 키 순서에서 'unknown' 제외
YOLO_CLASSES: tuple[str, ...] = tuple(k for k in DEFECT_TYPES if k != "unknown")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    record_id      TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    film_id        TEXT NOT NULL DEFAULT '',
    block          TEXT NOT NULL DEFAULT '',
    weld_id        TEXT NOT NULL DEFAULT '',
    joint_type     TEXT NOT NULL DEFAULT '',
    thickness_mm   REAL,
    quality_level  TEXT NOT NULL DEFAULT '',
    inspector      TEXT NOT NULL DEFAULT '',
    overall_passed INTEGER NOT NULL,
    defect_count   INTEGER NOT NULL DEFAULT 0,
    report_text    TEXT NOT NULL DEFAULT '',
    report_source  TEXT NOT NULL DEFAULT '',
    image_name     TEXT NOT NULL DEFAULT '',
    image_w        INTEGER,
    image_h        INTEGER,
    elapsed_seconds REAL,
    payload_json   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS defects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id   TEXT NOT NULL REFERENCES inspections(record_id) ON DELETE CASCADE,
    defect_id   TEXT NOT NULL,
    defect_type TEXT NOT NULL,
    source      TEXT NOT NULL,
    status      TEXT NOT NULL,
    size_mm     REAL,
    passed      INTEGER,
    clause      TEXT,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    confidence  REAL
);

CREATE INDEX IF NOT EXISTS idx_defects_record ON defects(record_id);
CREATE INDEX IF NOT EXISTS idx_defects_type ON defects(defect_type, status);
CREATE INDEX IF NOT EXISTS idx_inspections_created ON inspections(created_at);
"""


class Archive:
    """승인 기록 아카이브. 저장 = 검색 이력 + 자기개선 루프의 라벨 원천."""

    def __init__(self, db_path: str | Path = config.DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # 스키마 생성(멱등)
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)
            # 구버전 DB 마이그레이션: elapsed_seconds 컬럼이 없으면 추가
            try:
                conn.execute("ALTER TABLE inspections ADD COLUMN elapsed_seconds REAL")
            except sqlite3.OperationalError:
                pass  # 이미 존재

    # ------------------------------------------------------------------ 내부

    def _connect(self) -> sqlite3.Connection:
        """호출마다 새 커넥션 — Streamlit 재실행/멀티스레드 안전."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ------------------------------------------------------------------ 저장/조회

    def save(self, record: InspectionRecord) -> None:
        """승인 기록 저장. 같은 record_id 재저장 시 통째로 교체(중복 없음)."""
        ctx = record.context
        # 결함별 룰 판정 결과(크기/합부/조항)를 defect_id로 붙인다
        verdict_by_id = {v.defect_id: v for v in record.verdicts}
        accepted_count = sum(1 for c in record.candidates if c.status == "accepted")

        with closing(self._connect()) as conn, conn:
            # 교체: 기존 결함 행 제거 후 인스펙션 행 REPLACE
            conn.execute("DELETE FROM defects WHERE record_id = ?", (record.record_id,))
            conn.execute(
                """
                INSERT OR REPLACE INTO inspections (
                    record_id, created_at, film_id, block, weld_id, joint_type,
                    thickness_mm, quality_level, inspector, overall_passed,
                    defect_count, report_text, report_source, image_name,
                    image_w, image_h, elapsed_seconds, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    record.created_at,
                    ctx.film_id,
                    ctx.block,
                    ctx.weld_id,
                    ctx.joint_type,
                    ctx.thickness_mm,
                    ctx.quality_level,
                    ctx.inspector,
                    int(record.overall_passed),
                    accepted_count,
                    record.report_text,
                    record.report_source,
                    record.image_name,
                    int(record.image_size[0]),
                    int(record.image_size[1]),
                    record.elapsed_seconds,
                    record.to_json(),
                ),
            )
            for c in record.candidates:
                v = verdict_by_id.get(c.id)
                conn.execute(
                    """
                    INSERT INTO defects (
                        record_id, defect_id, defect_type, source, status,
                        size_mm, passed, clause, x1, y1, x2, y2, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.record_id,
                        c.id,
                        c.defect_type,
                        c.source,
                        c.status,
                        v.size_mm if v is not None else None,
                        int(v.passed) if v is not None else None,
                        v.clause if v is not None else None,
                        c.bbox[0],
                        c.bbox[1],
                        c.bbox[2],
                        c.bbox[3],
                        c.confidence,
                    ),
                )

    def get(self, record_id: str) -> InspectionRecord | None:
        """record_id로 전체 기록 재구성. 없으면 None."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT payload_json FROM inspections WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            return None
        return InspectionRecord.from_json(row[0])

    # ------------------------------------------------------------------ 검색

    def search(
        self,
        *,
        block: str = "",
        weld_id: str = "",
        defect_type: str = "",
        passed: bool | None = None,
        text: str = "",
        date_from: str = "",
        date_to: str = "",
    ) -> pd.DataFrame:
        """필터 검색 → 요약 DataFrame. 빈 필터는 무시, 최신순 정렬."""
        where: list[str] = []
        params: list = []

        if block:
            where.append("i.block LIKE ?")
            params.append(f"%{block}%")
        if weld_id:
            where.append("i.weld_id LIKE ?")
            params.append(f"%{weld_id}%")
        if defect_type:
            # 한국어 표기로 들어와도 코드 키로 역변환
            key = defect_type
            if key not in DEFECT_TYPES:
                ko_to_key = {v: k for k, v in DEFECT_TYPES.items()}
                key = ko_to_key.get(defect_type, defect_type)
            where.append(
                "EXISTS (SELECT 1 FROM defects d WHERE d.record_id = i.record_id "
                "AND d.defect_type = ? AND d.status = 'accepted')"
            )
            params.append(key)
        if passed is not None:
            where.append("i.overall_passed = ?")
            params.append(int(passed))
        if text:
            where.append(
                "(i.report_text LIKE ? OR i.film_id LIKE ? OR i.weld_id LIKE ? OR i.block LIKE ?)"
            )
            params.extend([f"%{text}%"] * 4)
        if date_from:
            where.append("i.created_at >= ?")
            params.append(date_from)
        if date_to:
            # 날짜만 주어지면 그 날의 끝까지 포함
            dt = date_to if len(date_to) > 10 else date_to + "T23:59:59"
            where.append("i.created_at <= ?")
            params.append(dt)

        sql = (
            "SELECT i.record_id, i.created_at, i.film_id, i.block, i.weld_id, "
            "i.quality_level, i.overall_passed, i.defect_count, i.inspector "
            "FROM inspections i"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY i.created_at DESC, i.record_id DESC"

        with closing(self._connect()) as conn:
            df = pd.read_sql_query(sql, conn, params=params)
            # accepted 결함 유형들(한국어, 콤마 결합) 컬럼 구성
            type_rows = conn.execute(
                "SELECT record_id, defect_type FROM defects "
                "WHERE status = 'accepted' ORDER BY id"
            ).fetchall()

        types_by_record: dict[str, list[str]] = {}
        for rid, dtype in type_rows:
            ko = DEFECT_TYPES.get(dtype, dtype)
            bucket = types_by_record.setdefault(rid, [])
            if ko not in bucket:  # 중복 유형은 1회만 표기
                bucket.append(ko)

        df["overall_passed"] = df["overall_passed"].astype(bool)
        df["defect_types"] = [
            ", ".join(types_by_record.get(rid, [])) for rid in df["record_id"]
        ]
        # 컬럼 순서 확정
        return df[
            [
                "record_id",
                "created_at",
                "film_id",
                "block",
                "weld_id",
                "quality_level",
                "overall_passed",
                "defect_count",
                "defect_types",
                "inspector",
            ]
        ]

    # ------------------------------------------------------------------ 통계

    def stats(self) -> dict:
        """자기개선 루프 대시보드용 집계."""
        with closing(self._connect()) as conn:
            total = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
            passed = conn.execute(
                "SELECT COUNT(*) FROM inspections WHERE overall_passed = 1"
            ).fetchone()[0]
            type_rows = conn.execute(
                "SELECT defect_type, COUNT(*) FROM defects "
                "WHERE status = 'accepted' GROUP BY defect_type"
            ).fetchall()
            ai_proposed = conn.execute(
                "SELECT COUNT(*) FROM defects WHERE source = 'ai'"
            ).fetchone()[0]
            ai_accepted = conn.execute(
                "SELECT COUNT(*) FROM defects WHERE source = 'ai' AND status = 'accepted'"
            ).fetchone()[0]
            ai_rejected = conn.execute(
                "SELECT COUNT(*) FROM defects WHERE source = 'ai' AND status = 'rejected'"
            ).fetchone()[0]
            human_added = conn.execute(
                "SELECT COUNT(*) FROM defects WHERE source = 'human'"
            ).fetchone()[0]
            avg_elapsed = conn.execute(
                "SELECT AVG(elapsed_seconds) FROM inspections "
                "WHERE elapsed_seconds IS NOT NULL"
            ).fetchone()[0]

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "defect_type_counts": {t: n for t, n in type_rows},
            "ai_proposed": ai_proposed,
            "ai_accepted": ai_accepted,
            "ai_rejected": ai_rejected,
            "human_added": human_added,  # AI 미탐 신호
            "acceptance_rate": (ai_accepted / ai_proposed) if ai_proposed > 0 else None,
            "avg_elapsed_seconds": avg_elapsed,  # 평균 판독 소요시간 (창출 효과 지표)
        }

    # ------------------------------------------------------------------ 백업/복원

    def export_all_json(self) -> str:
        """전체 기록을 JSON Lines(줄당 1건)로 반환 — 백업/이관용.

        클라우드 배포처럼 저장소가 휘발되는 환경에서 아카이브를 파일로
        내려받아 두었다가 import_json 으로 복원한다.
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT payload_json FROM inspections ORDER BY created_at, record_id"
            ).fetchall()
        return "\n".join(r[0] for r in rows) + ("\n" if rows else "")

    def import_json(self, text: str) -> int:
        """export_all_json 백업(JSON Lines)을 복원한다. 반환: 복원 건수.

        같은 record_id는 교체(save와 동일). 파손된 줄이 하나라도 있으면
        해당 줄에서 ValueError — 그 이전 줄들은 이미 저장되어 있다.
        """
        count = 0
        for lineno, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = InspectionRecord.from_json(line)
            except Exception as exc:
                raise ValueError(f"백업 {lineno}번째 줄 파싱 실패: {exc}") from exc
            self.save(record)
            count += 1
        return count

    # ------------------------------------------------------------------ 라벨 export

    def export_yolo_labels(self, out_dir: str | Path) -> dict:
        """승인(accepted) 결함들을 YOLO 학습 라벨(txt)로 export.

        - 클래스 인덱스: YOLO_CLASSES(models.DEFECT_TYPES 키 순서, 'unknown' 제외)
        - 각 record → <image_name stem>.txt (accepted 결함이 없으면 빈 파일 = 음성 예시)
          — 같은 stem의 record가 여러 건이면 라벨 줄을 병합 (덮어쓰기 라벨 소실 방지)
        - 각 줄 "cls cx cy w h" — image_size로 0~1 정규화, 범위 밖은 클립
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        class_index = {name: i for i, name in enumerate(YOLO_CLASSES)}

        with closing(self._connect()) as conn:
            recs = conn.execute(
                "SELECT record_id, image_name, image_w, image_h FROM inspections"
            ).fetchall()
            defect_rows = conn.execute(
                "SELECT record_id, defect_type, x1, y1, x2, y2 FROM defects "
                "WHERE status = 'accepted' ORDER BY id"
            ).fetchall()

        defects_by_record: dict[str, list[tuple]] = {}
        for rid, dtype, x1, y1, x2, y2 in defect_rows:
            defects_by_record.setdefault(rid, []).append((dtype, x1, y1, x2, y2))

        # stem 기준으로 먼저 집계 — 같은 image_name stem의 레코드가 여러 건이면
        # 라벨 줄을 병합한다(레코드별 write는 앞 레코드의 라벨을 조용히 덮어쓴다).
        n_records = 0
        lines_by_stem: dict[str, list[str]] = {}
        for rid, image_name, img_w, img_h in recs:
            stem = Path(image_name).stem if image_name else rid
            lines = lines_by_stem.setdefault(stem, [])
            for dtype, x1, y1, x2, y2 in defects_by_record.get(rid, []):
                cls = class_index.get(dtype)
                if cls is None:  # 'unknown' 등 학습 클래스가 아니면 제외
                    continue
                if not img_w or not img_h:
                    continue
                # 정규화 후 0~1 클립 → cx cy w h
                nx1 = min(max(x1 / img_w, 0.0), 1.0)
                ny1 = min(max(y1 / img_h, 0.0), 1.0)
                nx2 = min(max(x2 / img_w, 0.0), 1.0)
                ny2 = min(max(y2 / img_h, 0.0), 1.0)
                cx = (nx1 + nx2) / 2.0
                cy = (ny1 + ny2) / 2.0
                w = abs(nx2 - nx1)
                h = abs(ny2 - ny1)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            n_records += 1

        n_labels = 0
        for stem, lines in lines_by_stem.items():
            (out / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            n_labels += len(lines)

        (out / "classes.txt").write_text(
            "\n".join(YOLO_CLASSES) + "\n", encoding="utf-8"
        )
        return {"records": n_records, "labels": n_labels, "out_dir": str(out)}
