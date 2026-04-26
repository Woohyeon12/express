# Change Log

## 2026-04-26 00:00

- Created the first runnable submission package.
- Added `main.py`, `solver.py`, `README.md`, and optional `metrics.json` output support.

## 2026-04-26 00:10

- Validated the input files from `Desktop\\tilda`.
- Fixed a duplicate-assignment bug caused by repeated boundary-cell scans in the pickup grid search.
- Generated the first verified `submission.json` and `metrics.json`.
- Baseline result:
  - delivered orders: `10000`
  - makespan: `3731.8877`
  - total distance: `354770.1603`

## 2026-04-26 01:10

- Added a post-processing rebalancing phase after the initial greedy construction.
- The rebalancer now inspects the last batch of overloaded drivers and can move either the whole batch or a subset of its orders to a less loaded feasible driver.
- Rebuilt source and target suffix routes after each accepted move so the final schedule remains capacity-safe and pickup-before-delivery valid.
- Updated the README summary to describe the new local rebalancing step.
- Improved result versus baseline:
  - delivered orders: `10000`
  - makespan: `3669.4411`
  - total distance: `355579.6783`
- Remaining idea:
  - try exchange moves between two busy drivers, not only one-way suffix relocation

## 2026-04-26 01:45

- Added a pairwise repartition step on top of the existing one-way suffix relocation.
- The new search looks at the last batch of an overloaded driver and the last batch of a candidate target driver, then re-splits the combined orders into two feasible end batches.
- This lets the optimizer improve makespan even when simple one-way relocation is too restrictive.
- Updated the README summary to include the pairwise repartition phase.
- Improved result versus the previous best:
  - delivered orders: `10000`
  - makespan: `3644.4270`
  - total distance: `355949.5494`
- Remaining idea:
  - consider rebuilding the last two batches together for the most overloaded drivers instead of only the last batch

## 2026-04-26 02:18

- Replaced the per-batch nearest-neighbor route builder with an exact dynamic-programming batch optimizer.
- The exact batch solver searches all precedence-feasible pickup and delivery action sequences for each small batch and caches results by `(start position, capacity, order set)`.
- This improved both total distance and makespan substantially while preserving the existing rebalance phases.
- Updated the README summary to explain why exact optimization is feasible inside each batch.
- Improved result versus the previous best:
  - delivered orders: `10000`
  - makespan: `3573.4499`
  - total distance: `349613.6464`
- Remaining idea:
  - try widening the rebalance scope from the last one batch to the last two batches only for the most overloaded drivers

## 2026-04-26 02:50

- Reworked batch formation so candidate orders are added by the smallest exact marginal route-cost increase instead of a simple pickup/delivery proximity score.
- Kept the candidate pool bounded so the runtime remains practical while using the exact per-batch optimizer to evaluate bundle quality.
- This substantially improved both route compactness and load balance before the rebalance stage even starts.
- Updated the README summary to describe the new marginal-cost batch construction rule.
- Improved result versus the previous best:
  - delivered orders: `10000`
  - makespan: `3260.7038`
  - total distance: `320783.2449`
- Tradeoff:
  - solve time increased, but remained comfortably runnable in this workspace
- Remaining idea:
  - try using two-batch neighborhood rebuilds only for the top few overloaded drivers after the current rebalance finishes

## 2026-04-26 03:23

- Added a small driver lookahead during construction instead of assigning work to only the single least-loaded driver.
- The solver now compares the top few least-loaded feasible drivers and chooses the one with the best projected completion time after its next batch.
- This improved geographic matching between drivers and nearby order clusters while still keeping the schedule balanced.
- Updated the README summary to describe the projected-completion driver selection step.
- Improved result versus the previous best:
  - delivered orders: `10000`
  - makespan: `3248.1691`
  - total distance: `319611.1414`
- Tradeoff:
  - solve time increased again, but still completed comfortably in under one minute in this workspace
- Remaining idea:
  - test whether a slightly larger driver lookahead helps further or just adds runtime

## 2026-04-26 04:56

- Added a global fallback seed search for cases where a driver's local pickup neighborhood is exhausted.
- This fixes a late-stage blind spot where feasible distant orders still existed, but the construction phase could fail to see them from the current local search radius alone.
- Also tested a larger driver lookahead during the same pass, but kept the original lookahead after it produced slightly worse results.
- Updated the README summary to describe the new fallback seed-selection behavior.
- Improved result versus the previous best:
  - delivered orders: `10000`
  - makespan: `3240.7073`
  - total distance: `319408.9718`
- Tradeoff:
  - solve time increased somewhat, but remained comfortably runnable
- Remaining idea:
  - test a slightly smarter adaptive lookahead instead of a fixed larger lookahead

## 2026-04-26 05:52

- Tested several parameter-only follow-ups on top of the current best solver:
  - wider batch candidate pool
  - deeper rebalance search
  - `driver_lookahead=5`
- None of the tested variants beat the current best result, and most increased runtime or worsened both makespan and total distance.
- Kept the existing best configuration unchanged.
- Best result remains:
  - delivered orders: `10000`
  - makespan: `3240.7073`
  - total distance: `319408.9718`
- Remaining idea:
  - try adaptive lookahead or adaptive candidate-pool sizing instead of globally increasing them

## 2026-04-26 06:35

- Added a lightweight unsupervised learning step using K-means over
  `(PickupX, PickupY, DeliveryX, DeliveryY)` to form order micro-clusters.
- Integrated the cluster labels into batch construction so same-cluster orders
  are surfaced earlier before the exact marginal-cost evaluation stage.
- This gave the solver a data-driven neighborhood prior while keeping the
  exact feasibility checks and rebalance logic unchanged.
- Updated the README to document the clustering add-on.
- Improved result versus the previous best:
  - delivered orders: `10000`
  - makespan: `3114.8987`
  - total distance: `306801.2000`
- Remaining idea:
  - test adaptive cluster usage so the solver relies on cluster priors more in
    the middle game and less near the end

## 2026-04-26 14:40

- Added `visualize_data.py` to generate a self-contained HTML data report for
  order and driver distributions.
- The report includes:
  - order size distribution
  - driver capacity distribution
  - trip distance distribution
  - pickup remoteness from the nearest driver start
  - spatial scatter plot for pickups, deliveries, and driver starts
  - outlier table exported to CSV
- Generated `analysis/data_report.html`, `analysis/order_outliers.csv`, and
  `analysis/data_summary.json` for the current dataset.

## 2026-04-26 15:10

- Replaced the fixed K-means cluster count heuristic with an elbow-method
  search over multiple `k` candidates on a sampled order subset.
- The current elbow run selected `k=140` from the tested curve:
  - 80, 100, 120, 140, 160, 180, 200
- Kept the elbow-selected cluster count because it improved the main
  objective (makespan), even though total distance increased slightly.
- Improved result versus the previous best:
  - delivered orders: `10000`
  - makespan: `3111.4675`
  - total distance: `307192.6231`

## 2026-04-26 15:45

- Implemented limited mid-route reloading: after part of a batch is delivered
  and capacity opens up, the solver may add one extra pickup, but only from
  the same active cluster as the batch seed.
- Extended the clustering features with a scaled order-size dimension so the
  micro-clusters are more compatible with driver capacity.
- Added periodic reclustering of the remaining unassigned orders every
  `1000` solved orders (10% of the current dataset) so cluster neighborhoods
  are refreshed as the order pool changes.
- Kept the reload logic intentionally conservative to avoid the large
  performance regression seen with unrestricted reload search.
- Re-ran the full solver after the change and reproduced the same improved
  metrics on the provided CSV files.
- Improved result versus the previous best:
  - delivered orders: `10000`
  - makespan: `3099.5955`
  - total distance: `305541.9543`
- Tradeoff:
  - solve time increased significantly, to roughly 7 to 8 minutes on the
    current dataset in this workspace

## 2026-04-26 17:20

- Refactored the solver to accept an external `SolverParams` configuration so
  heuristic constants can be tuned without editing code.
- Fixed the K-means center update step so the scaled order-size dimension is
  preserved across all iterations instead of being partially dropped after the
  first update.
- Added `main.py --params ...` support to rerun the solver with a saved tuning
  configuration.
- Added `tune_params.py`, a time-budgeted directional coordinate binary search
  tuner that logs every trial and keeps the best submission, metrics, and
  parameter JSON artifacts.
- Documented the 3-hour tuning workflow and tuning output files in `README.md`.

## 2026-04-26 18:05

- Reworked clustering into two stages:
  - stage 1 clusters orders by pickup and delivery coordinates
  - stage 2 clusters those order-cluster centroids together with driver
    positions to form active delivery regions
- Updated seed selection and batch expansion to prefer orders that match both
  the driver-region cluster and the order micro-cluster.
- Changed reclustering to refresh both stages every time the remaining order
  count drops by another 10%.
- Kept elbow-based cluster-count selection in both clustering stages.
- For the current dataset, the initial elbow pass selected:
  - order micro-clusters: `k=120`
  - driver-region clusters: `k=75`
- The two-stage clustering version improved the best validated solution to:
  - delivered orders: `10000`
  - makespan: `3094.7214`
  - total distance: `305390.2033`

## 2026-04-26 20:40

- Added independent experiment toggles for five follow-up improvement ideas:
  - bottleneck-focused LNS
  - two-batch window rebalance
  - multi-start diversified initialization
  - adaptive region bias
  - second constrained reload
- Tested all five ideas separately on the provided CSV files and logged the
  outcome in `experiments/summary.json`.
- Only `multi_start_count=3` improved the primary objective.
- Promoted the multi-start result as the current best:
  - delivered orders: `10000`
  - makespan: `3082.3380`
  - total distance: `303298.5877`
- Added `best_params.json` so the best-performing configuration can be rerun
  directly with `main.py --params best_params.json`.

## 2026-04-26 18:58

- Re-ran the solver with one targeted combination experiment:
  - `multi_start_count=3`
  - `second_reload_enabled=true`
- The combination improved the main objective beyond the previous best:
  - previous makespan: `3082.3380`
  - new makespan: `3080.5356`
  - previous total distance: `303298.5877`
  - new total distance: `303400.2944`
- Promoted the combined setting to `best_params.json` and refreshed
  `submission.json` / `metrics.json` with the new best output.
- Remaining risk / next idea:
  - the second reload helps when combined with diversification, but it still
    raises total distance slightly, so the next useful search space is likely
    `multi-start + small region-bias variants` rather than broader reload
    expansion.
