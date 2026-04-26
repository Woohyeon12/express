from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


SVG_WIDTH = 560
SVG_HEIGHT = 320
MARGIN = 36


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a lightweight HTML data report for Tilda Express inputs.",
    )
    parser.add_argument("--orders", required=True, help="Path to orders.csv")
    parser.add_argument("--delivers", required=True, help="Path to delivers.csv")
    parser.add_argument(
        "--output-dir",
        default="analysis",
        help="Directory where the report files will be written",
    )
    return parser


def scale(value: float, min_value: float, max_value: float, start: float, end: float) -> float:
    if max_value <= min_value:
        return (start + end) / 2
    ratio = (value - min_value) / (max_value - min_value)
    return start + ratio * (end - start)


def svg_bar_chart(series: pd.Series, title: str, fill: str) -> str:
    width = SVG_WIDTH
    height = SVG_HEIGHT
    plot_width = width - 2 * MARGIN
    plot_height = height - 2 * MARGIN
    max_value = max(float(series.max()), 1.0)
    bar_width = plot_width / max(len(series), 1)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart">',
        f'<text x="{MARGIN}" y="22" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{MARGIN}" y1="{height - MARGIN}" x2="{width - MARGIN}" y2="{height - MARGIN}" class="axis" />',
        f'<line x1="{MARGIN}" y1="{MARGIN}" x2="{MARGIN}" y2="{height - MARGIN}" class="axis" />',
    ]

    for index, (label, value) in enumerate(series.items()):
        bar_height = 0.0 if max_value == 0 else (float(value) / max_value) * plot_height
        x = MARGIN + index * bar_width + 6
        y = height - MARGIN - bar_height
        w = max(bar_width - 12, 8)
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar_height:.1f}" fill="{fill}" rx="4" />')
        parts.append(f'<text x="{x + w / 2:.1f}" y="{height - 14}" class="tick" text-anchor="middle">{html.escape(str(label))}</text>')
        parts.append(f'<text x="{x + w / 2:.1f}" y="{max(y - 6, 30):.1f}" class="value" text-anchor="middle">{int(value)}</text>')

    return "".join(parts) + "</svg>"


def svg_histogram(values: np.ndarray, title: str, bins: int, fill: str, threshold: float | None = None) -> str:
    width = SVG_WIDTH
    height = SVG_HEIGHT
    plot_width = width - 2 * MARGIN
    plot_height = height - 2 * MARGIN
    counts, edges = np.histogram(values, bins=bins)
    max_count = max(int(counts.max()), 1)
    bar_width = plot_width / len(counts)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart">',
        f'<text x="{MARGIN}" y="22" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{MARGIN}" y1="{height - MARGIN}" x2="{width - MARGIN}" y2="{height - MARGIN}" class="axis" />',
        f'<line x1="{MARGIN}" y1="{MARGIN}" x2="{MARGIN}" y2="{height - MARGIN}" class="axis" />',
    ]

    for index, count in enumerate(counts):
        bar_height = (int(count) / max_count) * plot_height
        x = MARGIN + index * bar_width + 1
        y = height - MARGIN - bar_height
        w = max(bar_width - 2, 2)
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar_height:.1f}" fill="{fill}" rx="2" />')

    tick_positions = np.linspace(0, len(edges) - 1, num=min(6, len(edges)), dtype=int)
    seen_ticks: set[int] = set()
    for tick_index in tick_positions:
        if tick_index in seen_ticks:
            continue
        seen_ticks.add(tick_index)
        tick_value = edges[tick_index]
        x = scale(float(tick_value), float(edges[0]), float(edges[-1]), MARGIN, width - MARGIN)
        parts.append(f'<text x="{x:.1f}" y="{height - 14}" class="tick" text-anchor="middle">{tick_value:.1f}</text>')

    if threshold is not None and float(edges[0]) <= threshold <= float(edges[-1]):
        threshold_x = scale(float(threshold), float(edges[0]), float(edges[-1]), MARGIN, width - MARGIN)
        parts.append(f'<line x1="{threshold_x:.1f}" y1="{MARGIN}" x2="{threshold_x:.1f}" y2="{height - MARGIN}" class="threshold" />')
        parts.append(f'<text x="{threshold_x + 4:.1f}" y="{MARGIN + 14}" class="threshold-label">outlier threshold {threshold:.1f}</text>')

    return "".join(parts) + "</svg>"


def svg_scatter(
    pickups: pd.DataFrame,
    deliveries: pd.DataFrame,
    drivers: pd.DataFrame,
    title: str,
) -> str:
    width = SVG_WIDTH
    height = SVG_HEIGHT
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart">',
        f'<text x="{MARGIN}" y="22" class="chart-title">{html.escape(title)}</text>',
        f'<rect x="{MARGIN}" y="{MARGIN}" width="{width - 2 * MARGIN}" height="{height - 2 * MARGIN}" class="plot-bg" />',
        f'<line x1="{MARGIN}" y1="{height - MARGIN}" x2="{width - MARGIN}" y2="{height - MARGIN}" class="axis" />',
        f'<line x1="{MARGIN}" y1="{MARGIN}" x2="{MARGIN}" y2="{height - MARGIN}" class="axis" />',
    ]

    def point(x_value: float, y_value: float) -> tuple[float, float]:
        x = scale(x_value, 1, 100, MARGIN, width - MARGIN)
        y = scale(y_value, 1, 100, height - MARGIN, MARGIN)
        return (x, y)

    for _, row in pickups.iterrows():
        x, y = point(float(row["PickupX"]), float(row["PickupY"]))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.4" fill="#1d4ed8" opacity="0.35" />')

    for _, row in deliveries.iterrows():
        x, y = point(float(row["DeliveryX"]), float(row["DeliveryY"]))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.4" fill="#16a34a" opacity="0.25" />')

    for _, row in drivers.iterrows():
        x, y = point(float(row["CurrentX"]), float(row["CurrentY"]))
        parts.append(f'<rect x="{x - 2.0:.1f}" y="{y - 2.0:.1f}" width="4" height="4" fill="#dc2626" opacity="0.9" />')

    legend_y = 34
    parts.append('<circle cx="385" cy="18" r="4" fill="#1d4ed8" opacity="0.7" />')
    parts.append(f'<text x="394" y="{legend_y}" class="legend">pickup</text>')
    parts.append('<circle cx="450" cy="18" r="4" fill="#16a34a" opacity="0.7" />')
    parts.append(f'<text x="459" y="{legend_y}" class="legend">delivery</text>')
    parts.append('<rect x="513" y="14" width="8" height="8" fill="#dc2626" opacity="0.9" />')
    parts.append(f'<text x="526" y="{legend_y}" class="legend">driver start</text>')

    return "".join(parts) + "</svg>"


def iqr_threshold(values: pd.Series) -> float:
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    return q3 + 1.5 * (q3 - q1)


def write_report(
    orders: pd.DataFrame,
    drivers: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    orders = orders.copy()
    drivers = drivers.copy()

    orders["TripDistance"] = np.hypot(
        orders["PickupX"] - orders["DeliveryX"],
        orders["PickupY"] - orders["DeliveryY"],
    )

    driver_points = drivers[["CurrentX", "CurrentY"]].to_numpy(dtype=float)
    pickup_points = orders[["PickupX", "PickupY"]].to_numpy(dtype=float)
    nearest_driver_distance = np.sqrt(((pickup_points[:, None, :] - driver_points[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    orders["NearestDriverPickupDistance"] = nearest_driver_distance

    trip_threshold = iqr_threshold(orders["TripDistance"])
    pickup_threshold = iqr_threshold(orders["NearestDriverPickupDistance"])
    orders["IsTripOutlier"] = orders["TripDistance"] > trip_threshold
    orders["IsPickupOutlier"] = orders["NearestDriverPickupDistance"] > pickup_threshold
    orders["OutlierScore"] = (
        orders["TripDistance"] / max(float(orders["TripDistance"].median()), 1.0)
        + orders["NearestDriverPickupDistance"] / max(float(orders["NearestDriverPickupDistance"].median()), 1.0)
    )

    outliers = orders.loc[
        orders["IsTripOutlier"] | orders["IsPickupOutlier"],
        [
            "OrderID",
            "PickupX",
            "PickupY",
            "DeliveryX",
            "DeliveryY",
            "OrderSize",
            "TripDistance",
            "NearestDriverPickupDistance",
            "OutlierScore",
            "IsTripOutlier",
            "IsPickupOutlier",
        ],
    ].sort_values(["OutlierScore", "TripDistance"], ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    outlier_csv = output_dir / "order_outliers.csv"
    summary_json = output_dir / "data_summary.json"
    report_html = output_dir / "data_report.html"

    outliers.to_csv(outlier_csv, index=False)

    summary = {
        "order_count": int(len(orders)),
        "driver_count": int(len(drivers)),
        "trip_distance_mean": float(orders["TripDistance"].mean()),
        "trip_distance_p95": float(orders["TripDistance"].quantile(0.95)),
        "trip_distance_outlier_threshold": trip_threshold,
        "pickup_distance_mean": float(orders["NearestDriverPickupDistance"].mean()),
        "pickup_distance_p95": float(orders["NearestDriverPickupDistance"].quantile(0.95)),
        "pickup_distance_outlier_threshold": pickup_threshold,
        "trip_outlier_count": int(orders["IsTripOutlier"].sum()),
        "pickup_outlier_count": int(orders["IsPickupOutlier"].sum()),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    order_size_svg = svg_bar_chart(
        orders["OrderSize"].value_counts().sort_index(),
        "Order Size Distribution",
        "#2563eb",
    )
    capacity_svg = svg_bar_chart(
        drivers["Capacity"].value_counts().sort_index(),
        "Driver Capacity Distribution",
        "#ea580c",
    )
    trip_svg = svg_histogram(
        orders["TripDistance"].to_numpy(dtype=float),
        "Trip Distance Distribution",
        bins=24,
        fill="#0f766e",
        threshold=trip_threshold,
    )
    pickup_svg = svg_histogram(
        orders["NearestDriverPickupDistance"].to_numpy(dtype=float),
        "Nearest Driver to Pickup Distance",
        bins=24,
        fill="#7c3aed",
        threshold=pickup_threshold,
    )
    scatter_svg = svg_scatter(orders, orders, drivers, "Spatial Distribution of Orders and Drivers")

    top_outliers = outliers.head(20).copy()
    table_rows = []
    for _, row in top_outliers.iterrows():
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['OrderID'])}</td>"
            f"<td>{int(row['OrderSize'])}</td>"
            f"<td>{row['TripDistance']:.2f}</td>"
            f"<td>{row['NearestDriverPickupDistance']:.2f}</td>"
            f"<td>{'yes' if row['IsTripOutlier'] else 'no'}</td>"
            f"<td>{'yes' if row['IsPickupOutlier'] else 'no'}</td>"
            "</tr>"
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tilda Express Data Report</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: #f7f7f5;
      color: #111827;
    }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    p {{
      color: #4b5563;
      line-height: 1.5;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 20px 0 28px;
    }}
    .card {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 8px 20px rgba(17, 24, 39, 0.04);
    }}
    .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #6b7280;
    }}
    .value-big {{
      margin-top: 8px;
      font-size: 28px;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-bottom: 22px;
    }}
    .panel {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 16px;
      padding: 12px;
      box-shadow: 0 8px 20px rgba(17, 24, 39, 0.04);
    }}
    .panel.full {{
      grid-column: 1 / -1;
    }}
    .chart {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .chart-title {{
      font-size: 14px;
      font-weight: 700;
      fill: #111827;
    }}
    .axis {{
      stroke: #94a3b8;
      stroke-width: 1;
    }}
    .plot-bg {{
      fill: #fcfcfb;
      stroke: #e5e7eb;
    }}
    .tick {{
      font-size: 11px;
      fill: #475569;
    }}
    .value {{
      font-size: 11px;
      fill: #1f2937;
    }}
    .legend {{
      font-size: 11px;
      fill: #475569;
    }}
    .threshold {{
      stroke: #dc2626;
      stroke-width: 1.5;
      stroke-dasharray: 5 4;
    }}
    .threshold-label {{
      font-size: 11px;
      fill: #dc2626;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid #e5e7eb;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #f1f5f9;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      background: #f8fafc;
    }}
    @media (max-width: 960px) {{
      .cards, .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Tilda Express Data Report</h1>
    <p>Distribution and outlier overview for the provided order and driver input files.</p>

    <div class="cards">
      <div class="card">
        <div class="label">Orders</div>
        <div class="value-big">{len(orders):,}</div>
      </div>
      <div class="card">
        <div class="label">Drivers</div>
        <div class="value-big">{len(drivers):,}</div>
      </div>
      <div class="card">
        <div class="label">Trip Outliers</div>
        <div class="value-big">{int(orders["IsTripOutlier"].sum()):,}</div>
      </div>
      <div class="card">
        <div class="label">Pickup Outliers</div>
        <div class="value-big">{int(orders["IsPickupOutlier"].sum()):,}</div>
      </div>
    </div>

    <div class="grid">
      <div class="panel">{order_size_svg}</div>
      <div class="panel">{capacity_svg}</div>
      <div class="panel">{trip_svg}</div>
      <div class="panel">{pickup_svg}</div>
      <div class="panel full">{scatter_svg}</div>
    </div>

    <h2>Top Outlier Orders</h2>
    <p>Orders flagged by trip distance, pickup remoteness from the nearest driver start, or both.</p>
    <table>
      <thead>
        <tr>
          <th>OrderID</th>
          <th>Size</th>
          <th>Trip Distance</th>
          <th>Nearest Driver Pickup Distance</th>
          <th>Trip Outlier</th>
          <th>Pickup Outlier</th>
        </tr>
      </thead>
      <tbody>
        {"".join(table_rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    report_html.write_text(html_text, encoding="utf-8")
    return (report_html, outlier_csv, summary_json)


def main() -> None:
    args = build_parser().parse_args()
    orders = pd.read_csv(args.orders)
    drivers = pd.read_csv(args.delivers)
    output_dir = Path(args.output_dir)
    report_html, outlier_csv, summary_json = write_report(orders, drivers, output_dir)
    print(f"Report saved to: {report_html.resolve()}")
    print(f"Outlier CSV saved to: {outlier_csv.resolve()}")
    print(f"Summary JSON saved to: {summary_json.resolve()}")


if __name__ == "__main__":
    main()
