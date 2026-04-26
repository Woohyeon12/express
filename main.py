from __future__ import annotations

import argparse
import json
from pathlib import Path

from solver import (
    SolverParams,
    evaluate_schedule,
    load_drivers,
    load_orders,
    optimize_schedule,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tilda Express delivery optimization heuristic",
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
        "--output",
        default="submission.json",
        help="output submission JSON path",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="optional metrics JSON path",
    )
    parser.add_argument(
        "--params",
        default=None,
        help="optional solver parameter JSON path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    orders = load_orders(args.orders)
    drivers = load_drivers(args.delivers)
    params = None
    if args.params:
        with Path(args.params).open("r", encoding="utf-8") as handle:
            params = SolverParams.from_dict(json.load(handle))

    schedule = optimize_schedule(orders=orders, drivers=drivers, params=params)
    evaluation = evaluate_schedule(orders=orders, drivers=drivers, schedule=schedule)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(schedule, handle, ensure_ascii=False, indent=2)

    if args.metrics:
        metrics_path = Path(args.metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_payload = {
            "orders": len(orders),
            "drivers": len(drivers),
            "makespan": evaluation.makespan,
            "total_distance": evaluation.total_distance,
            "delivered_orders": evaluation.delivered_orders,
        }
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)

    print(f"Orders: {len(orders)}")
    print(f"Drivers: {len(drivers)}")
    print(f"Delivered orders: {evaluation.delivered_orders}")
    print(f"Makespan: {evaluation.makespan:.4f}")
    print(f"Total distance: {evaluation.total_distance:.4f}")
    if args.params:
        print(f"Params loaded from: {Path(args.params).resolve()}")
    print(f"Submission saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
