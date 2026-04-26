# Tilda Express Delivery Optimization

This repository solves the Tilda Express assignment by building a feasible
pickup-and-delivery schedule for every driver and saving the result in the
required `submission.json` format.

The problem is closer to constrained routing and scheduling than classical
machine learning. The current solution therefore uses a heuristic optimizer
with exact feasibility checks inside each small batch.

## Approach

1. Pick one of the currently least-loaded drivers with a small lookahead over
   projected completion time.
2. Choose a nearby seed order that the driver can carry.
3. Build order micro-clusters from pickup and delivery coordinates, then build
   driver-region clusters by clustering those order-cluster centroids together
   with driver positions.
4. Expand the batch with nearby, same-order-cluster, and same-driver-region
   orders using exact marginal route cost rather than raw distance only.
5. Allow one conservative same-cluster reload order when part of the batch can
   be delivered first and a nearby pickup can fill the freed capacity.
6. Recompute both clustering stages whenever the remaining order pool drops by
   another `10%`.
7. Solve the action order inside each batch with exact dynamic programming
   under pickup-before-delivery and capacity constraints.
8. Rebalance the last batches of overloaded drivers to reduce makespan.

## Run The Solver

```bash
python main.py --orders orders.csv --delivers delivers.csv --output submission.json --metrics metrics.json
```

Example:

```bash
python main.py --orders "C:\Users\서우현\Desktop\tilda\orders.csv" --delivers "C:\Users\서우현\Desktop\tilda\delivers.csv" --output submission.json --metrics metrics.json
```

## Run With Custom Parameters

The solver now supports external parameter files:

```bash
python main.py --orders orders.csv --delivers delivers.csv --params tuning/best_params.json --output submission.json --metrics metrics.json
```

For the current best validated result in this repo, run:

```bash
python main.py --orders orders.csv --delivers delivers.csv --params best_params.json --output submission.json --metrics metrics.json
```

## Three-Hour Parameter Tuning

Use `tune_params.py` to search for better heuristic parameters over a fixed
time budget. The tuner uses a directional coordinate-wise binary search: for
each numeric parameter, it evaluates the left and right midpoint around the
current best setting and keeps moving only in directions that improve the
objective.

This is not a mathematically exact binary search because the routing objective
is non-convex and non-monotonic, but it gives a practical “binary-search-like”
improvement loop that fits the assignment well.

Run a full 3-hour tuning session:

```bash
python tune_params.py --orders orders.csv --delivers delivers.csv --time-budget-sec 10800 --output-dir tuning
```

Useful smoke test:

```bash
python tune_params.py --orders orders.csv --delivers delivers.csv --time-budget-sec 600 --trial-limit 2 --output-dir tuning
```

Tuning outputs:

- `tuning/trial_history.jsonl`: one JSON line per evaluated parameter set
- `tuning/best_params.json`: best parameter set found so far
- `tuning/best_metrics.json`: best makespan and total distance
- `tuning/best_submission.json`: best schedule found so far
- `tuning/best_trial.json`: metadata for the best trial
- `tuning/summary.json`: final tuning summary

## Current Clustering Logic

The solver now uses two clustering stages.

Stage 1: order micro-clusters

- Features: `(PickupX, PickupY, DeliveryX, DeliveryY, scaled OrderSize)`
- Goal: keep nearby and capacity-compatible orders together
- Cluster count: selected automatically by an elbow heuristic

Stage 2: driver-region clusters

- Inputs: order-cluster centroids from stage 1 plus current driver positions
- Goal: bind remaining order neighborhoods to the most relevant active drivers
- Cluster count: selected again by an elbow heuristic on the combined points

Both stages are recomputed whenever the remaining order pool shrinks by another
`10%`, so the assignment stays aligned with the evolving driver positions.

For the current dataset, the initial elbow pass selected:

- order micro-clusters: `k=120`
- driver-region clusters: `k=75`

Those diagnostics are saved in `analysis/elbow_curve.json`.

## Improvement Experiments

Five structural improvements were tested independently against the current
two-stage clustering baseline:

- `01_bottleneck_lns`: no improvement
- `02_window_rebalance`: worse makespan
- `03_multi_start`: improved makespan and total distance
- `04_adaptive_region_bias`: worse makespan
- `05_second_reload`: worse makespan

The first five stand-alone experiments only improved on the baseline with
`multi-start`. A follow-up combination test then paired that stronger
initialization with a second constrained reload, which improved makespan
further. The current best configuration is:

```json
{
  "multi_start_count": 3,
  "second_reload_enabled": true
}
```

The experiment summary is saved in `experiments/summary.json`.

## Visualization

Generate a simple exploratory data report with:

```bash
python visualize_data.py --orders orders.csv --delivers delivers.csv --output-dir analysis
```

Generated files:

- `analysis/data_report.html`
- `analysis/order_outliers.csv`
- `analysis/data_summary.json`
- `analysis/elbow_curve.json`

## Output Format

`submission.json` has the required structure:

```json
{
  "D001": ["P:O0001", "P:O0042", "D:O0001", "D:O0042"],
  "D002": ["P:O0002", "D:O0002"]
}
```

- `P:OrderID`: pickup action
- `D:OrderID`: delivery action
