from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from solver import (
    SolverParams,
    evaluate_schedule,
    load_drivers,
    load_orders,
    optimize_schedule,
)


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str
    low: int | float
    high: int | float
    resolution: int | float

    def midpoint(self, low: int | float, high: int | float) -> int | float | None:
        if self.kind == "int":
            low_int = int(low)
            high_int = int(high)
            if high_int - low_int <= int(self.resolution):
                return None
            candidate = low_int + (high_int - low_int) // 2
            if candidate <= low_int or candidate >= high_int:
                return None
            return candidate

        low_float = float(low)
        high_float = float(high)
        resolution = float(self.resolution)
        if high_float - low_float <= resolution:
            return None
        candidate = round((low_float + high_float) / 2.0, 4)
        if candidate <= low_float + 1e-9 or candidate >= high_float - 1e-9:
            return None
        return candidate


DEFAULT_SPECS = [
    ParamSpec("driver_lookahead", "int", 2, 8, 1),
    ParamSpec("extra_candidate_target", "int", 48, 112, 1),
    ParamSpec("cluster_candidate_limit", "int", 12, 36, 1),
    ParamSpec("candidate_pool_limit", "int", 10, 26, 1),
    ParamSpec("reload_nearest_gap_limit", "float", 8.0, 18.0, 0.5),
    ParamSpec("reload_marginal_trip_factor", "float", 0.2, 0.55, 0.02),
    ParamSpec("cluster_size_scale_base", "float", 50.0, 140.0, 2.0),
    ParamSpec("rebalance_target_pool", "int", 30, 90, 1),
]


class DirectionalBinaryTuner:
    def __init__(
        self,
        orders_path: str,
        delivers_path: str,
        output_dir: Path,
        base_params: SolverParams,
    ) -> None:
        self.orders_path = orders_path
        self.delivers_path = delivers_path
        self.orders = load_orders(orders_path)
        self.drivers = load_drivers(delivers_path)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_params = base_params
        self.cache: dict[str, dict[str, Any]] = {}
        self.trial_counter = 0
        self.best_record: dict[str, Any] | None = None
        self.best_params = base_params
        self.history_path = self.output_dir / "trial_history.jsonl"
        self.best_params_path = self.output_dir / "best_params.json"
        self.best_metrics_path = self.output_dir / "best_metrics.json"
        self.best_submission_path = self.output_dir / "best_submission.json"
        self.best_trial_path = self.output_dir / "best_trial.json"
        self.summary_path = self.output_dir / "summary.json"

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _params_key(params: SolverParams) -> str:
        return json.dumps(params.to_dict(), sort_keys=True)

    @staticmethod
    def _is_better(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return (left["makespan"], left["total_distance"]) < (
            right["makespan"],
            right["total_distance"],
        )

    def evaluate(self, params: SolverParams, label: str) -> dict[str, Any]:
        cache_key = self._params_key(params)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        self.trial_counter += 1
        started = time.perf_counter()
        schedule = optimize_schedule(self.orders, self.drivers, params=params)
        evaluation = evaluate_schedule(self.orders, self.drivers, schedule)
        wall_time = time.perf_counter() - started

        record = {
            "trial": self.trial_counter,
            "label": label,
            "timestamp_utc": self._now_iso(),
            "orders_path": self.orders_path,
            "delivers_path": self.delivers_path,
            "makespan": evaluation.makespan,
            "total_distance": evaluation.total_distance,
            "delivered_orders": evaluation.delivered_orders,
            "wall_time_sec": wall_time,
            "params": params.to_dict(),
        }

        is_new_best = self.best_record is None or self._is_better(record, self.best_record)
        record["is_best"] = is_new_best

        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.cache[cache_key] = record

        if is_new_best:
            self.best_record = record
            self.best_params = params
            self.best_params_path.write_text(
                json.dumps(params.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.best_metrics_path.write_text(
                json.dumps(
                    {
                        "orders": len(self.orders),
                        "drivers": len(self.drivers),
                        "makespan": evaluation.makespan,
                        "total_distance": evaluation.total_distance,
                        "delivered_orders": evaluation.delivered_orders,
                        "trial": self.trial_counter,
                        "label": label,
                        "wall_time_sec": wall_time,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            self.best_submission_path.write_text(
                json.dumps(schedule, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.best_trial_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return record

    def save_summary(self, elapsed_sec: float) -> None:
        payload = {
            "completed_at_utc": self._now_iso(),
            "elapsed_sec": elapsed_sec,
            "trials": self.trial_counter,
            "unique_trials": len(self.cache),
            "best_record": self.best_record,
        }
        self.summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def run(
        self,
        specs: list[ParamSpec],
        time_budget_sec: int,
        trial_limit: int | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        deadline = started + max(1, time_budget_sec)
        current_params = self.base_params
        current_record = self.evaluate(current_params, "baseline")
        search_ranges = {
            spec.name: [spec.low, spec.high]
            for spec in specs
        }

        round_index = 0
        while time.perf_counter() < deadline:
            if trial_limit is not None and self.trial_counter >= trial_limit:
                break

            round_index += 1
            improved = False
            any_open_dimension = False

            for spec in specs:
                if time.perf_counter() >= deadline:
                    break
                if trial_limit is not None and self.trial_counter >= trial_limit:
                    break

                low, high = search_ranges[spec.name]
                current_value = getattr(current_params, spec.name)
                left_value = spec.midpoint(low, current_value)
                right_value = spec.midpoint(current_value, high)
                if left_value is None and right_value is None:
                    continue

                any_open_dimension = True
                candidates: list[tuple[str, int | float, dict[str, Any]]] = []
                if left_value is not None:
                    left_params = current_params.with_updates(**{spec.name: left_value})
                    left_record = self.evaluate(
                        left_params,
                        f"round_{round_index}:{spec.name}:left:{left_value}",
                    )
                    candidates.append(("left", left_value, left_record))
                    if trial_limit is not None and self.trial_counter >= trial_limit:
                        break

                if right_value is not None and time.perf_counter() < deadline:
                    right_params = current_params.with_updates(**{spec.name: right_value})
                    right_record = self.evaluate(
                        right_params,
                        f"round_{round_index}:{spec.name}:right:{right_value}",
                    )
                    candidates.append(("right", right_value, right_record))

                better_candidates = [
                    candidate
                    for candidate in candidates
                    if self._is_better(candidate[2], current_record)
                ]

                if better_candidates:
                    direction, chosen_value, chosen_record = min(
                        better_candidates,
                        key=lambda candidate: (
                            candidate[2]["makespan"],
                            candidate[2]["total_distance"],
                        ),
                    )
                    if direction == "left":
                        search_ranges[spec.name] = [low, current_value]
                    else:
                        search_ranges[spec.name] = [current_value, high]
                    current_params = current_params.with_updates(**{spec.name: chosen_value})
                    current_record = chosen_record
                    improved = True
                    continue

                narrowed_low = left_value if left_value is not None else low
                narrowed_high = right_value if right_value is not None else high
                search_ranges[spec.name] = [narrowed_low, narrowed_high]

            if not any_open_dimension:
                break
            if not improved:
                intervals_collapsed = True
                for spec in specs:
                    low, high = search_ranges[spec.name]
                    current_value = getattr(current_params, spec.name)
                    if spec.midpoint(low, current_value) is not None:
                        intervals_collapsed = False
                        break
                    if spec.midpoint(current_value, high) is not None:
                        intervals_collapsed = False
                        break
                if intervals_collapsed:
                    break

        elapsed = time.perf_counter() - started
        self.save_summary(elapsed)
        if self.best_record is None:
            raise RuntimeError("No tuning trial completed.")
        return self.best_record


def load_base_params(path: str | None) -> SolverParams:
    if not path:
        return SolverParams()
    with Path(path).open("r", encoding="utf-8") as handle:
        return SolverParams.from_dict(json.load(handle))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune solver parameters with directional binary search.",
    )
    parser.add_argument("--orders", required=True, help="path to orders.csv")
    parser.add_argument(
        "--delivers",
        "--drivers",
        dest="delivers",
        required=True,
        help="path to delivers.csv",
    )
    parser.add_argument(
        "--time-budget-sec",
        type=int,
        default=10800,
        help="total tuning time budget in seconds (default: 10800 = 3 hours)",
    )
    parser.add_argument(
        "--output-dir",
        default="tuning",
        help="directory for tuning logs and best artifacts",
    )
    parser.add_argument(
        "--base-params",
        default=None,
        help="optional JSON file to seed the search from",
    )
    parser.add_argument(
        "--trial-limit",
        type=int,
        default=None,
        help="optional hard cap on trial count, useful for smoke tests",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_params = load_base_params(args.base_params)
    tuner = DirectionalBinaryTuner(
        orders_path=args.orders,
        delivers_path=args.delivers,
        output_dir=Path(args.output_dir),
        base_params=base_params,
    )
    best_record = tuner.run(
        specs=DEFAULT_SPECS,
        time_budget_sec=args.time_budget_sec,
        trial_limit=args.trial_limit,
    )

    print(f"Trials completed: {tuner.trial_counter}")
    print(f"Best makespan: {best_record['makespan']:.4f}")
    print(f"Best total distance: {best_record['total_distance']:.4f}")
    print(f"Best params saved to: {tuner.best_params_path.resolve()}")
    print(f"Best metrics saved to: {tuner.best_metrics_path.resolve()}")
    print(f"Best submission saved to: {tuner.best_submission_path.resolve()}")


if __name__ == "__main__":
    main()
