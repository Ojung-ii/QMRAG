from __future__ import annotations

import contextlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .io_utils import dump_json, ensure_dir, to_jsonable


def _iso_from_epoch(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


class TimingRecorder:
    def __init__(self, out_dir: str | Path, enabled: bool = True):
        self.out_dir = ensure_dir(out_dir)
        self.enabled = bool(enabled)
        self.events_path = self.out_dir / "timing_events.jsonl"
        self.summary_path = self.out_dir / "timing_summary.json"
        self.summary_md_path = self.out_dir / "timing_summary.md"
        self._events: list[dict[str, Any]] = []
        if self.enabled and self.events_path.exists():
            self.events_path.unlink()

    def record(
        self,
        *,
        dataset: str,
        stage: str,
        start_ts: float,
        end_ts: float,
        query_id: str | None = None,
        num_items_in: int | None = None,
        num_items_out: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        row = {
            "dataset": dataset,
            "query_id": query_id,
            "stage": stage,
            "start_ts": _iso_from_epoch(start_ts),
            "end_ts": _iso_from_epoch(end_ts),
            "duration_ms": round(max(0.0, end_ts - start_ts) * 1000.0, 6),
            "num_items_in": num_items_in,
            "num_items_out": num_items_out,
            "extra": dict(extra or {}),
        }
        self._events.append(row)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")

    def record_duration(
        self,
        *,
        dataset: str,
        stage: str,
        duration_s: float,
        query_id: str | None = None,
        num_items_in: int | None = None,
        num_items_out: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        end_ts = time.time()
        start_ts = end_ts - max(0.0, float(duration_s or 0.0))
        self.record(
            dataset=dataset,
            query_id=query_id,
            stage=stage,
            start_ts=start_ts,
            end_ts=end_ts,
            num_items_in=num_items_in,
            num_items_out=num_items_out,
            extra=extra,
        )

    @contextlib.contextmanager
    def time_block(
        self,
        *,
        dataset: str,
        stage: str,
        query_id: str | None = None,
        num_items_in: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        payload: dict[str, Any] = {"num_items_out": None}
        start = time.time()
        try:
            yield payload
        finally:
            self.record(
                dataset=dataset,
                query_id=query_id,
                stage=stage,
                start_ts=start,
                end_ts=time.time(),
                num_items_in=num_items_in,
                num_items_out=payload.get("num_items_out"),
                extra=extra,
            )

    def summary(self) -> dict[str, Any]:
        by_stage: dict[str, list[float]] = {}
        for event in self._events:
            by_stage.setdefault(str(event.get("stage")), []).append(float(event.get("duration_ms", 0.0) or 0.0))
        stages: dict[str, dict[str, Any]] = {}
        for stage, values in sorted(by_stage.items()):
            vals = sorted(values)
            if not vals:
                continue
            p95_idx = min(len(vals) - 1, int(0.95 * (len(vals) - 1)))
            stages[stage] = {
                "total_ms": round(sum(vals), 6),
                "mean_ms": round(sum(vals) / len(vals), 6),
                "p50_ms": round(statistics.median(vals), 6),
                "p95_ms": round(vals[p95_idx], 6),
                "max_ms": round(max(vals), 6),
                "count": len(vals),
            }
        return {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "event_count": len(self._events),
            "stages": stages,
        }

    def write_summary(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        summary = self.summary()
        dump_json(summary, self.summary_path)
        lines = ["# Timing Summary", "", "| stage | count | total_ms | mean_ms | p50_ms | p95_ms | max_ms |", "|---|---:|---:|---:|---:|---:|---:|"]
        for stage, row in summary.get("stages", {}).items():
            lines.append(
                f"| {stage} | {row.get('count', 0)} | {row.get('total_ms', 0):.3f} | "
                f"{row.get('mean_ms', 0):.3f} | {row.get('p50_ms', 0):.3f} | "
                f"{row.get('p95_ms', 0):.3f} | {row.get('max_ms', 0):.3f} |"
            )
        self.summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return summary
