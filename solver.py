from __future__ import annotations

import csv
import heapq
import itertools
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Callable


GRID_MIN = 1
GRID_MAX = 100
SEARCH_RADII = list(range(0, 13)) + [15, 18, 22, 27, 33, 40, 50, 65, 85, 99]


@dataclass(frozen=True)
class Order:
    order_id: str
    pickup_x: int
    pickup_y: int
    delivery_x: int
    delivery_y: int
    size: int

    @property
    def pickup(self) -> tuple[int, int]:
        return (self.pickup_x, self.pickup_y)

    @property
    def delivery(self) -> tuple[int, int]:
        return (self.delivery_x, self.delivery_y)

    @property
    def trip_distance(self) -> float:
        return euclidean(self.pickup, self.delivery)


@dataclass(frozen=True)
class Driver:
    driver_id: str
    current_x: int
    current_y: int
    capacity: int

    @property
    def start(self) -> tuple[int, int]:
        return (self.current_x, self.current_y)


@dataclass(order=True)
class DriverState:
    elapsed_time: float
    tie_breaker: int
    driver_id: str = field(compare=False)
    start_x: int = field(compare=False)
    start_y: int = field(compare=False)
    x: int = field(compare=False)
    y: int = field(compare=False)
    capacity: int = field(compare=False)
    actions: list[str] = field(compare=False, default_factory=list)
    batches: list["BatchPlan"] = field(compare=False, default_factory=list)
    batches_completed: int = field(compare=False, default=0)
    orders_completed: int = field(compare=False, default=0)

    @property
    def position(self) -> tuple[int, int]:
        return (self.x, self.y)

    @property
    def start(self) -> tuple[int, int]:
        return (self.start_x, self.start_y)


@dataclass(frozen=True)
class BatchPlan:
    order_indices: tuple[int, ...]
    actions: tuple[str, ...]
    end_pos: tuple[int, int]
    distance: float
    total_size: int
    peak_load: int


@dataclass(frozen=True)
class Evaluation:
    makespan: float
    total_distance: float
    route_distances: dict[str, float]
    delivered_orders: int


@dataclass(frozen=True)
class SolverParams:
    driver_lookahead: int = 4
    extra_candidate_target: int = 72
    cluster_candidate_limit: int = 24
    candidate_pool_limit: int = 18
    seed_candidate_target: int = 40
    region_candidate_limit: int = 60
    non_cluster_pickup_gap_limit: float = 20.0
    non_cluster_delivery_gap_limit: float = 28.0
    candidate_pickup_weight: float = 0.58
    candidate_delivery_weight: float = 0.38
    candidate_size_ratio_weight: float = 4.0
    candidate_size_bonus: float = 0.2
    cluster_bonus_base: float = 4.0
    cluster_bonus_size_ratio_weight: float = 1.0
    addition_peak_weight: float = 0.12
    reload_candidate_limit: int = 6
    reload_nearest_gap_limit: float = 12.0
    reload_marginal_base: float = 16.0
    reload_marginal_trip_factor: float = 0.35
    reload_marginal_trip_offset: float = 4.0
    reload_gap_weight: float = 0.18
    reload_peak_weight: float = 0.06
    region_match_bonus_batch: float = 1.0
    region_match_bonus_seed: float = 1.5
    cluster_size_scale_base: float = 90.0
    kmeans_sample_size: int = 2000
    kmeans_iterations: int = 8
    rebalance_rounds: int = 120
    rebalance_source_pool: int = 16
    rebalance_target_pool: int = 60
    rebalance_window_depth: int = 1
    recluster_interval: int | None = None
    enable_bottleneck_lns: bool = False
    lns_rounds: int = 4
    lns_source_pool: int = 3
    lns_target_pool: int = 8
    lns_window_depth: int = 2
    multi_start_count: int = 1
    multi_start_seed_step: int = 97
    random_seed: int = 0
    shuffle_driver_order: bool = False
    adaptive_region_bias: bool = False
    second_reload_enabled: bool = False

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "SolverParams":
        valid_names = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - valid_names)
        if unknown:
            raise ValueError(f"Unknown solver parameter(s): {', '.join(unknown)}")

        int_fields = {
            "driver_lookahead",
            "extra_candidate_target",
            "cluster_candidate_limit",
            "candidate_pool_limit",
            "seed_candidate_target",
            "region_candidate_limit",
            "reload_candidate_limit",
            "kmeans_sample_size",
            "kmeans_iterations",
            "rebalance_rounds",
            "rebalance_source_pool",
            "rebalance_target_pool",
            "rebalance_window_depth",
            "recluster_interval",
            "lns_rounds",
            "lns_source_pool",
            "lns_target_pool",
            "lns_window_depth",
            "multi_start_count",
            "multi_start_seed_step",
            "random_seed",
        }
        float_fields = {
            "non_cluster_pickup_gap_limit",
            "non_cluster_delivery_gap_limit",
            "candidate_pickup_weight",
            "candidate_delivery_weight",
            "candidate_size_ratio_weight",
            "candidate_size_bonus",
            "cluster_bonus_base",
            "cluster_bonus_size_ratio_weight",
            "addition_peak_weight",
            "reload_nearest_gap_limit",
            "reload_marginal_base",
            "reload_marginal_trip_factor",
            "reload_marginal_trip_offset",
            "reload_gap_weight",
            "reload_peak_weight",
            "region_match_bonus_batch",
            "region_match_bonus_seed",
            "cluster_size_scale_base",
        }
        bool_fields = {
            "enable_bottleneck_lns",
            "shuffle_driver_order",
            "adaptive_region_bias",
            "second_reload_enabled",
        }

        normalized: dict[str, object] = {}
        for key, value in raw.items():
            if key in int_fields:
                normalized[key] = None if value is None else int(value)
            elif key in float_fields:
                normalized[key] = float(value)
            elif key in bool_fields:
                normalized[key] = bool(value)
            else:
                normalized[key] = value
        return cls(**normalized)

    def with_updates(self, **updates: int | float | None) -> "SolverParams":
        return replace(self, **updates)


def euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def squared_feature_distance(
    a: tuple[float, ...],
    b: tuple[float, ...],
) -> float:
    return sum((left - right) ** 2 for left, right in zip(a, b))


def load_orders(path: str) -> list[Order]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            Order(
                order_id=row["OrderID"].strip(),
                pickup_x=int(row["PickupX"]),
                pickup_y=int(row["PickupY"]),
                delivery_x=int(row["DeliveryX"]),
                delivery_y=int(row["DeliveryY"]),
                size=int(row["OrderSize"]),
            )
            for row in reader
        ]


def load_drivers(path: str) -> list[Driver]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            Driver(
                driver_id=row["DeliverID"].strip(),
                current_x=int(row["CurrentX"]),
                current_y=int(row["CurrentY"]),
                capacity=int(row["Capacity"]),
            )
            for row in reader
        ]


class DeliveryOptimizer:
    def __init__(
        self,
        orders: list[Order],
        drivers: list[Driver],
        params: SolverParams | None = None,
    ) -> None:
        self.orders = orders
        self.drivers = drivers
        self.params = params or SolverParams()
        self.unassigned = [True] * len(orders)
        self.remaining_orders = len(orders)
        self.pickup_grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.batch_plan_cache: dict[tuple[int, int, int, tuple[int, ...]], BatchPlan | None] = {}
        self.cluster_count = 0
        self.cluster_elbow_curve: list[tuple[int, float]] = []
        self.cluster_centers: list[tuple[float, float, float, float]] = []
        self.order_cluster_ids = [-1] * len(self.orders)
        self.cluster_members: list[list[int]] = []
        self.region_count = 0
        self.region_elbow_curve: list[tuple[int, float]] = []
        self.region_members: list[list[int]] = []
        self.order_region_ids = [-1] * len(self.orders)
        self.driver_region_ids: dict[str, int] = {}
        self.recluster_interval = self.params.recluster_interval or max(1, len(self.orders) // 10)
        self.next_recluster_at = self.recluster_interval
        for index, order in enumerate(orders):
            self.pickup_grid[order.pickup].append(index)

    def solve(self) -> dict[str, list[str]]:
        available: list[DriverState] = []
        states_by_id: dict[str, DriverState] = {}
        driver_lookahead = self.params.driver_lookahead
        driver_sequence = self.drivers.copy()
        if self.params.shuffle_driver_order:
            random.Random(self.params.random_seed + 101).shuffle(driver_sequence)
        for index, driver in enumerate(driver_sequence):
            state = DriverState(
                elapsed_time=0.0,
                tie_breaker=index,
                driver_id=driver.driver_id,
                start_x=driver.current_x,
                start_y=driver.current_y,
                x=driver.current_x,
                y=driver.current_y,
                capacity=driver.capacity,
            )
            states_by_id[driver.driver_id] = state
            heapq.heappush(available, state)

        self._refresh_active_clusters(states_by_id)

        while self.remaining_orders > 0:
            if not available:
                raise RuntimeError("No feasible driver remains for the outstanding orders.")

            sampled_states: list[tuple[float, float, int, DriverState, list[int], BatchPlan]] = []
            while available and len(sampled_states) < driver_lookahead:
                state = heapq.heappop(available)
                batch = self._select_batch(state)
                if not batch:
                    continue
                batch_plan = self._build_batch_plan(state.position, batch, state.capacity)
                projected_completion = state.elapsed_time + batch_plan.distance
                sampled_states.append(
                    (
                        projected_completion,
                        batch_plan.distance,
                        state.tie_breaker,
                        state,
                        batch,
                        batch_plan,
                    )
                )

            if not sampled_states:
                raise RuntimeError("Orders remain unassigned after all drivers retired.")

            sampled_states.sort(key=lambda item: (item[0], item[1], item[2]))
            _, _, _, state, batch, batch_plan = sampled_states[0]
            for _, _, _, other_state, _, _ in sampled_states[1:]:
                heapq.heappush(available, other_state)

            for order_index in batch:
                self.unassigned[order_index] = False
            self.remaining_orders -= len(batch)

            self._append_batch(state, batch_plan)
            state.orders_completed += len(batch)
            state.batches_completed += 1
            heapq.heappush(available, state)

            solved_orders = len(self.orders) - self.remaining_orders
            if self.remaining_orders > 0 and solved_orders >= self.next_recluster_at:
                self._refresh_active_clusters(states_by_id)
                while solved_orders >= self.next_recluster_at:
                    self.next_recluster_at += self.recluster_interval

        self._rebalance_driver_loads(states_by_id)
        if self.params.enable_bottleneck_lns:
            self._run_bottleneck_lns(states_by_id)
        return {
            driver.driver_id: states_by_id[driver.driver_id].actions
            for driver in self.drivers
        }

    def _select_batch(self, state: DriverState) -> list[int]:
        seed_index = self._find_seed_order(state)
        if seed_index is None:
            return []

        seed_order = self.orders[seed_index]
        batch = [seed_index]
        used_capacity = seed_order.size
        max_batch_orders = batch_order_limit(state.capacity)
        if max_batch_orders == 1:
            return batch

        extra_candidates = self._collect_candidates(
            center=seed_order.pickup,
            capacity=state.capacity,
            target=self.params.extra_candidate_target,
        )
        regional_candidates = self._collect_region_candidates(
            driver_id=state.driver_id,
            center=seed_order.pickup,
            capacity=state.capacity,
            limit=max(16, self.params.cluster_candidate_limit),
        )
        clustered_candidates = self._collect_cluster_candidates(
            seed_index=seed_index,
            capacity=state.capacity,
            limit=self.params.cluster_candidate_limit,
        )
        preferred_cluster_candidates = set(clustered_candidates)
        preferred_region_candidates = set(regional_candidates)
        merged_candidates: list[int] = []
        seen_candidates: set[int] = set()
        for candidate_index in clustered_candidates + regional_candidates + extra_candidates:
            if candidate_index in seen_candidates:
                continue
            merged_candidates.append(candidate_index)
            seen_candidates.add(candidate_index)

        scored_candidates: list[tuple[float, int]] = []
        for candidate_index in merged_candidates:
            if candidate_index == seed_index or not self.unassigned[candidate_index]:
                continue
            candidate = self.orders[candidate_index]

            pickup_gap = euclidean(seed_order.pickup, candidate.pickup)
            delivery_gap = euclidean(seed_order.delivery, candidate.delivery)
            in_seed_cluster = candidate_index in preferred_cluster_candidates
            in_driver_region = candidate_index in preferred_region_candidates
            if (
                not in_seed_cluster
                and pickup_gap > self.params.non_cluster_pickup_gap_limit
                and delivery_gap > self.params.non_cluster_delivery_gap_limit
            ):
                continue

            size_ratio = candidate.size / max(state.capacity, 1)
            score = (
                self.params.candidate_pickup_weight * pickup_gap
                + self.params.candidate_delivery_weight * delivery_gap
                + self.params.candidate_size_ratio_weight * size_ratio
                - self.params.candidate_size_bonus * candidate.size
            )
            if in_seed_cluster:
                score -= self.params.cluster_bonus_base + (
                    self.params.cluster_bonus_size_ratio_weight
                    * min(candidate.size, state.capacity)
                    / max(state.capacity, 1)
                )
            if in_driver_region:
                score -= self._region_match_bonus(
                    driver_id=state.driver_id,
                    order_index=candidate_index,
                    base_bonus=self.params.region_match_bonus_batch,
                )
            scored_candidates.append((score, candidate_index))

        scored_candidates.sort(key=lambda item: (item[0], self.orders[item[1]].order_id))
        candidate_pool = [
            candidate_index
            for _, candidate_index in scored_candidates[: self.params.candidate_pool_limit]
        ]
        current_plan = self._build_batch_plan(state.position, batch, state.capacity)

        while candidate_pool and len(batch) < max_batch_orders and used_capacity < state.capacity:
            best_choice: tuple[float, int, BatchPlan] | None = None
            for candidate_index in list(candidate_pool):
                candidate = self.orders[candidate_index]
                if used_capacity + candidate.size > state.capacity:
                    continue
                tentative_plan = self._try_build_batch_plan(
                    state.position,
                    batch + [candidate_index],
                    state.capacity,
                )
                if tentative_plan is None:
                    continue

                marginal_cost = tentative_plan.distance - current_plan.distance
                effective_size = max(candidate.size, 1)
                efficiency_score = (
                    marginal_cost / effective_size
                    + self.params.addition_peak_weight
                    * (tentative_plan.peak_load / max(state.capacity, 1))
                )
                choice = (efficiency_score, candidate_index, tentative_plan)
                if best_choice is None or choice < best_choice:
                    best_choice = choice

            if best_choice is None:
                break

            _, chosen_index, chosen_plan = best_choice
            batch.append(chosen_index)
            used_capacity += self.orders[chosen_index].size
            current_plan = chosen_plan
            candidate_pool = [
                candidate_index
                for candidate_index in candidate_pool
                if candidate_index != chosen_index
            ]

        max_reload_steps = 2 if self.params.second_reload_enabled else 1
        for _ in range(max_reload_steps):
            reload_total_size = sum(self.orders[index].size for index in batch)
            reload_choice: tuple[float, int, BatchPlan] | None = None
            reload_candidates = [
                candidate_index
                for candidate_index in clustered_candidates
                if candidate_index not in batch
            ][: self.params.reload_candidate_limit]
            if not reload_candidates:
                break

            delivery_anchors = [self.orders[index].delivery for index in batch]
            for candidate_index in reload_candidates:
                candidate = self.orders[candidate_index]
                if reload_total_size + candidate.size > batch_total_size_limit(state.capacity):
                    continue
                if reload_total_size + candidate.size <= state.capacity:
                    continue

                nearest_delivery_gap = min(
                    euclidean(delivery_point, candidate.pickup)
                    for delivery_point in delivery_anchors
                )
                if nearest_delivery_gap > self.params.reload_nearest_gap_limit:
                    continue

                tentative_plan = self._try_build_batch_plan(
                    state.position,
                    batch + [candidate_index],
                    state.capacity,
                )
                if tentative_plan is None:
                    continue

                marginal_cost = tentative_plan.distance - current_plan.distance
                if marginal_cost > max(
                    self.params.reload_marginal_base,
                    self.params.reload_marginal_trip_factor * candidate.trip_distance
                    + self.params.reload_marginal_trip_offset,
                ):
                    continue

                reload_score = (
                    marginal_cost / max(candidate.size, 1)
                    + self.params.reload_gap_weight * nearest_delivery_gap
                    + self.params.reload_peak_weight
                    * (tentative_plan.peak_load / max(state.capacity, 1))
                )
                choice = (reload_score, candidate_index, tentative_plan)
                if reload_choice is None or choice < reload_choice:
                    reload_choice = choice

            if reload_choice is None:
                break

            _, chosen_index, chosen_plan = reload_choice
            batch.append(chosen_index)
            current_plan = chosen_plan

        return batch

    def _build_batch_plan(
        self,
        start: tuple[int, int],
        order_indices: list[int] | tuple[int, ...],
        capacity: int,
    ) -> BatchPlan:
        batch_plan = self._try_build_batch_plan(start, order_indices, capacity)
        if batch_plan is None:
            raise ValueError("Infeasible batch under the current capacity constraints.")
        return batch_plan

    def _try_build_batch_plan(
        self,
        start: tuple[int, int],
        order_indices: list[int] | tuple[int, ...],
        capacity: int,
    ) -> BatchPlan | None:
        canonical_indices = tuple(sorted(order_indices))
        cache_key = (start[0], start[1], capacity, canonical_indices)
        if cache_key in self.batch_plan_cache:
            cached = self.batch_plan_cache[cache_key]
            return cached

        total_size = sum(self.orders[order_index].size for order_index in canonical_indices)
        if total_size > batch_total_size_limit(capacity):
            self.batch_plan_cache[cache_key] = None
            return None
        route_distance, action_steps, peak_load = self._solve_batch_route_exact(
            start=start,
            order_indices=canonical_indices,
            capacity=capacity,
        )
        if not math.isfinite(route_distance):
            self.batch_plan_cache[cache_key] = None
            return None

        position = start
        actions: list[str] = []
        for action_type, local_index in action_steps:
            order = self.orders[canonical_indices[local_index]]
            position = order.pickup if action_type == "P" else order.delivery
            actions.append(f"{action_type}:{order.order_id}")

        batch_plan = BatchPlan(
            order_indices=canonical_indices,
            actions=tuple(actions),
            end_pos=position,
            distance=route_distance,
            total_size=total_size,
            peak_load=peak_load,
        )
        self.batch_plan_cache[cache_key] = batch_plan
        return batch_plan

    @staticmethod
    def _append_batch(state: DriverState, batch_plan: BatchPlan) -> None:
        state.batches.append(batch_plan)
        state.actions.extend(batch_plan.actions)
        state.elapsed_time += batch_plan.distance
        state.x, state.y = batch_plan.end_pos

    @staticmethod
    def _pop_last_batch(state: DriverState) -> BatchPlan:
        batch_plan = state.batches.pop()
        state.elapsed_time -= batch_plan.distance
        del state.actions[-len(batch_plan.actions) :]
        if state.batches:
            state.x, state.y = state.batches[-1].end_pos
        else:
            state.x, state.y = state.start
        return batch_plan

    def _extract_tail_context(
        self,
        state: DriverState,
        window_depth: int,
    ) -> tuple[tuple[int, int], list[BatchPlan], tuple[int, ...], float]:
        depth = min(window_depth, len(state.batches))
        tail_batches = state.batches[-depth:]
        restart = state.batches[-depth - 1].end_pos if len(state.batches) > depth else state.start
        tail_indices = tuple(
            order_index
            for batch in tail_batches
            for order_index in batch.order_indices
        )
        tail_distance = sum(batch.distance for batch in tail_batches)
        return (restart, tail_batches, tail_indices, tail_distance)

    def _replace_tail_batches(
        self,
        state: DriverState,
        window_depth: int,
        new_batches: list[BatchPlan],
    ) -> None:
        removed_orders = 0
        removed_batches = min(window_depth, len(state.batches))
        for _ in range(removed_batches):
            removed_plan = self._pop_last_batch(state)
            removed_orders += len(removed_plan.order_indices)

        state.orders_completed -= removed_orders
        state.batches_completed -= removed_batches
        for batch_plan in new_batches:
            self._append_batch(state, batch_plan)
            state.orders_completed += len(batch_plan.order_indices)
            state.batches_completed += 1

    def _select_batch_from_pool(
        self,
        start: tuple[int, int],
        capacity: int,
        available_indices: list[int] | tuple[int, ...],
    ) -> list[int]:
        feasible_indices = [
            order_index
            for order_index in available_indices
            if self.orders[order_index].size <= capacity
        ]
        if not feasible_indices:
            return []

        available_set = set(feasible_indices)

        def seed_score(order_index: int) -> tuple[float, str]:
            order = self.orders[order_index]
            return (
                euclidean(start, order.pickup) + 0.35 * order.trip_distance - 0.2 * order.size,
                order.order_id,
            )

        seed_index = min(feasible_indices, key=seed_score)
        seed_order = self.orders[seed_index]
        batch = [seed_index]
        used_capacity = seed_order.size
        max_batch_orders = batch_order_limit(capacity)
        if max_batch_orders == 1:
            return batch

        candidate_indices = [
            order_index
            for order_index in feasible_indices
            if order_index != seed_index
        ]
        candidate_indices.sort(
            key=lambda order_index: (
                0.6 * euclidean(seed_order.pickup, self.orders[order_index].pickup)
                + 0.4 * euclidean(seed_order.delivery, self.orders[order_index].delivery)
                - 0.15 * self.orders[order_index].size,
                self.orders[order_index].order_id,
            )
        )
        candidate_pool = candidate_indices[: self.params.candidate_pool_limit]
        current_plan = self._build_batch_plan(start, batch, capacity)

        while candidate_pool and len(batch) < max_batch_orders and used_capacity < capacity:
            best_choice: tuple[float, int, BatchPlan] | None = None
            for candidate_index in list(candidate_pool):
                if candidate_index not in available_set:
                    continue
                candidate = self.orders[candidate_index]
                if used_capacity + candidate.size > capacity:
                    continue

                tentative_plan = self._try_build_batch_plan(
                    start,
                    batch + [candidate_index],
                    capacity,
                )
                if tentative_plan is None:
                    continue

                marginal_cost = tentative_plan.distance - current_plan.distance
                choice = (
                    marginal_cost / max(candidate.size, 1),
                    candidate_index,
                    tentative_plan,
                )
                if best_choice is None or choice < best_choice:
                    best_choice = choice

            if best_choice is None:
                break

            _, chosen_index, chosen_plan = best_choice
            batch.append(chosen_index)
            used_capacity += self.orders[chosen_index].size
            current_plan = chosen_plan
            candidate_pool = [
                candidate_index
                for candidate_index in candidate_pool
                if candidate_index != chosen_index
            ]

        return batch

    def _build_batches_from_pool(
        self,
        start: tuple[int, int],
        capacity: int,
        order_indices: tuple[int, ...],
    ) -> list[BatchPlan] | None:
        remaining = list(order_indices)
        position = start
        batch_plans: list[BatchPlan] = []
        while remaining:
            batch = self._select_batch_from_pool(position, capacity, remaining)
            if not batch:
                return None
            batch_plan = self._try_build_batch_plan(position, batch, capacity)
            if batch_plan is None:
                return None
            batch_plans.append(batch_plan)
            chosen = set(batch)
            remaining = [
                order_index
                for order_index in remaining
                if order_index not in chosen
            ]
            position = batch_plan.end_pos
        return batch_plans

    @staticmethod
    def _batch_sequence_distance(batch_plans: list[BatchPlan]) -> float:
        return sum(batch_plan.distance for batch_plan in batch_plans)

    def _iter_transfer_subsets(
        self,
        order_indices: tuple[int, ...],
        tail_batches: list[BatchPlan],
    ):
        seen: set[tuple[int, ...]] = set()
        ordered_indices = tuple(sorted(order_indices))

        def register(candidate: tuple[int, ...], *, allow_full: bool = False) -> bool:
            canonical = tuple(sorted(candidate))
            if not canonical or canonical in seen:
                return False
            if not allow_full and canonical == ordered_indices:
                return False
            seen.add(canonical)
            return True

        for batch_plan in tail_batches:
            candidate = tuple(batch_plan.order_indices)
            if register(candidate):
                yield candidate

        for order_index in ordered_indices:
            candidate = (order_index,)
            if register(candidate):
                yield candidate

        for left, right in itertools.combinations(ordered_indices, 2):
            candidate = (left, right)
            if register(candidate):
                yield candidate

        if register(ordered_indices, allow_full=True):
            yield ordered_indices

    def _repair_orders_between_states(
        self,
        source_start: tuple[int, int],
        source_capacity: int,
        source_base_time: float,
        target_start: tuple[int, int],
        target_capacity: int,
        target_base_time: float,
        order_indices: tuple[int, ...],
        other_max: float,
    ) -> tuple[list[BatchPlan], list[BatchPlan]] | None:
        remaining = list(order_indices)
        source_pos = source_start
        target_pos = target_start
        source_time = source_base_time
        target_time = target_base_time
        source_plans: list[BatchPlan] = []
        target_plans: list[BatchPlan] = []

        while remaining:
            best_choice: tuple[float, str, BatchPlan, list[int]] | None = None
            for owner, position, capacity, elapsed_time in (
                ("source", source_pos, source_capacity, source_time),
                ("target", target_pos, target_capacity, target_time),
            ):
                feasible_pool = [
                    order_index
                    for order_index in remaining
                    if self.orders[order_index].size <= capacity
                ]
                if not feasible_pool:
                    continue
                batch = self._select_batch_from_pool(position, capacity, feasible_pool)
                if not batch:
                    continue
                batch_plan = self._try_build_batch_plan(position, batch, capacity)
                if batch_plan is None:
                    continue
                projected_source = source_time + (batch_plan.distance if owner == "source" else 0.0)
                projected_target = target_time + (batch_plan.distance if owner == "target" else 0.0)
                objective = max(other_max, projected_source, projected_target)
                choice = (objective, owner, batch_plan, batch)
                if best_choice is None or choice < best_choice:
                    best_choice = choice

            if best_choice is None:
                return None

            _, owner, batch_plan, batch = best_choice
            chosen = set(batch)
            remaining = [
                order_index
                for order_index in remaining
                if order_index not in chosen
            ]
            if owner == "source":
                source_plans.append(batch_plan)
                source_pos = batch_plan.end_pos
                source_time += batch_plan.distance
            else:
                target_plans.append(batch_plan)
                target_pos = batch_plan.end_pos
                target_time += batch_plan.distance

        return (source_plans, target_plans)

    def _rebalance_driver_loads(self, states_by_id: dict[str, DriverState]) -> None:
        states = list(states_by_id.values())
        if len(states) < 2:
            return

        max_rounds = self.params.rebalance_rounds
        source_pool = self.params.rebalance_source_pool
        target_pool = self.params.rebalance_target_pool
        window_depth_limit = self.params.rebalance_window_depth
        improvement_floor = 1e-6

        for _ in range(max_rounds):
            ranked = sorted(states, key=lambda state: state.elapsed_time, reverse=True)
            current_makespan = ranked[0].elapsed_time
            best_action: tuple | None = None

            for source_rank, source in enumerate(ranked[:source_pool]):
                if not source.batches:
                    continue
                other_max = ranked[0].elapsed_time
                if source_rank == 0 and len(ranked) > 1:
                    other_max = ranked[1].elapsed_time

                batch_plan = source.batches[-1]
                if batch_plan.total_size > source.capacity:
                    continue
                if batch_plan.total_size > max(
                    target.capacity for target in states if target.driver_id != source.driver_id
                ):
                    continue

                candidate_targets = sorted(
                    (target for target in states if target.driver_id != source.driver_id),
                    key=lambda state: state.elapsed_time,
                )[:target_pool]
                source_restart = source.batches[-2].end_pos if len(source.batches) > 1 else source.start

                if window_depth_limit > 1:
                    for source_window_depth in range(2, min(window_depth_limit, len(source.batches)) + 1):
                        (
                            source_window_restart,
                            source_tail_batches,
                            source_tail_indices,
                            source_tail_distance,
                        ) = self._extract_tail_context(source, source_window_depth)
                        if len(source_tail_indices) > 12:
                            continue

                        for target in candidate_targets[: max(12, target_pool // 2)]:
                            for moved_indices in self._iter_transfer_subsets(
                                source_tail_indices,
                                source_tail_batches,
                            ):
                                moved_tail_plans = self._build_batches_from_pool(
                                    target.position,
                                    target.capacity,
                                    moved_indices,
                                )
                                if moved_tail_plans is None:
                                    continue

                                retained_indices = tuple(
                                    order_index
                                    for order_index in source_tail_indices
                                    if order_index not in set(moved_indices)
                                )
                                retained_tail_plans: list[BatchPlan] = []
                                if retained_indices:
                                    retained_tail_plans = self._build_batches_from_pool(
                                        source_window_restart,
                                        source.capacity,
                                        retained_indices,
                                    ) or []
                                    if not retained_tail_plans:
                                        continue

                                new_source_time = (
                                    source.elapsed_time - source_tail_distance
                                    + self._batch_sequence_distance(retained_tail_plans)
                                )
                                new_target_time = (
                                    target.elapsed_time
                                    + self._batch_sequence_distance(moved_tail_plans)
                                )
                                new_makespan = max(other_max, new_source_time, new_target_time)
                                if new_makespan + improvement_floor >= current_makespan:
                                    continue

                                window_move = (
                                    new_makespan,
                                    "window_move",
                                    source,
                                    target,
                                    source_window_depth,
                                    retained_tail_plans,
                                    moved_tail_plans,
                                )
                                if best_action is None or new_makespan < best_action[0]:
                                    best_action = window_move

                for target in candidate_targets:
                    if target.capacity < max(self.orders[index].size for index in batch_plan.order_indices):
                        continue

                    for moved_indices in self._iter_movable_subsets(batch_plan.order_indices):
                        moved_size = sum(self.orders[index].size for index in moved_indices)
                        if moved_size > target.capacity:
                            continue

                        retained_indices = tuple(
                            index for index in batch_plan.order_indices if index not in moved_indices
                        )
                        retained_plan = None
                        new_source_time = source.elapsed_time - batch_plan.distance
                        if retained_indices:
                            retained_plan = self._build_batch_plan(
                                source_restart,
                                retained_indices,
                                source.capacity,
                            )
                            new_source_time += retained_plan.distance

                        moved_plan = self._build_batch_plan(
                            target.position,
                            moved_indices,
                            target.capacity,
                        )
                        new_target_time = target.elapsed_time + moved_plan.distance
                        new_makespan = max(other_max, new_source_time, new_target_time)
                        if new_makespan + improvement_floor >= current_makespan:
                            continue

                        move = (
                            new_makespan,
                            "move",
                            source,
                            target,
                            moved_plan,
                            retained_plan,
                            batch_plan,
                        )
                        if best_action is None or new_makespan < best_action[0]:
                            best_action = move

                    if not target.batches:
                        continue

                    target_batch = target.batches[-1]
                    if target_batch.total_size > target.capacity:
                        continue
                    target_restart = target.batches[-2].end_pos if len(target.batches) > 1 else target.start
                    combined_indices = batch_plan.order_indices + target_batch.order_indices
                    if len(combined_indices) > 12:
                        continue

                    for source_indices in self._iter_partition_subsets(combined_indices):
                        target_indices = tuple(
                            order_index
                            for order_index in combined_indices
                            if order_index not in source_indices
                        )
                        if not source_indices or not target_indices:
                            continue

                        source_size = sum(self.orders[index].size for index in source_indices)
                        target_size = sum(self.orders[index].size for index in target_indices)
                        if source_size > source.capacity or target_size > target.capacity:
                            continue
                        if (
                            source_indices == batch_plan.order_indices
                            and target_indices == target_batch.order_indices
                        ):
                            continue

                        source_plan = self._build_batch_plan(
                            source_restart,
                            source_indices,
                            source.capacity,
                        )
                        target_plan = self._build_batch_plan(
                            target_restart,
                            target_indices,
                            target.capacity,
                        )
                        new_source_time = source.elapsed_time - batch_plan.distance + source_plan.distance
                        new_target_time = target.elapsed_time - target_batch.distance + target_plan.distance
                        new_makespan = max(other_max, new_source_time, new_target_time)
                        if new_makespan + improvement_floor >= current_makespan:
                            continue

                        repartition = (
                            new_makespan,
                            "swap",
                            source,
                            target,
                            source_plan,
                            target_plan,
                            batch_plan,
                            target_batch,
                        )
                        if best_action is None or new_makespan < best_action[0]:
                            best_action = repartition

            if best_action is None:
                return

            action_type = best_action[1]
            if action_type == "move":
                _, _, source, target, moved_plan, retained_plan, original_batch = best_action
                removed_plan = self._pop_last_batch(source)
                source.orders_completed -= len(removed_plan.order_indices)
                source.batches_completed -= 1
                if retained_plan is not None:
                    self._append_batch(source, retained_plan)
                    source.orders_completed += len(retained_plan.order_indices)
                    source.batches_completed += 1
                self._append_batch(target, moved_plan)
                target.orders_completed += len(moved_plan.order_indices)
                target.batches_completed += 1

                if removed_plan.order_indices != original_batch.order_indices:
                    raise RuntimeError("Batch rebalance applied to an unexpected source batch.")
                continue

            if action_type == "window_move":
                _, _, source, target, source_window_depth, retained_tail_plans, moved_tail_plans = best_action
                self._replace_tail_batches(source, source_window_depth, retained_tail_plans)
                for batch_plan in moved_tail_plans:
                    self._append_batch(target, batch_plan)
                    target.orders_completed += len(batch_plan.order_indices)
                    target.batches_completed += 1
                continue

            if action_type == "swap":
                _, _, source, target, source_plan, target_plan, old_source, old_target = best_action
                removed_source = self._pop_last_batch(source)
                removed_target = self._pop_last_batch(target)
                source.orders_completed -= len(removed_source.order_indices)
                source.batches_completed -= 1
                target.orders_completed -= len(removed_target.order_indices)
                target.batches_completed -= 1
                self._append_batch(source, source_plan)
                self._append_batch(target, target_plan)
                source.orders_completed += len(source_plan.order_indices)
                source.batches_completed += 1
                target.orders_completed += len(target_plan.order_indices)
                target.batches_completed += 1

                if removed_source.order_indices != old_source.order_indices:
                    raise RuntimeError("Unexpected source batch during swap rebalance.")
                if removed_target.order_indices != old_target.order_indices:
                    raise RuntimeError("Unexpected target batch during swap rebalance.")
                continue

            raise RuntimeError(f"Unknown rebalance action: {action_type}")

    def _run_bottleneck_lns(self, states_by_id: dict[str, DriverState]) -> None:
        states = list(states_by_id.values())
        if len(states) < 2:
            return

        for _ in range(self.params.lns_rounds):
            ranked = sorted(states, key=lambda state: state.elapsed_time, reverse=True)
            current_makespan = ranked[0].elapsed_time
            best_action: tuple | None = None

            for source_rank, source in enumerate(ranked[: self.params.lns_source_pool]):
                if not source.batches:
                    continue
                other_max = ranked[0].elapsed_time
                if source_rank == 0 and len(ranked) > 1:
                    other_max = ranked[1].elapsed_time

                for window_depth in range(1, min(self.params.lns_window_depth, len(source.batches)) + 1):
                    (
                        source_restart,
                        _source_tail_batches,
                        source_tail_indices,
                        source_tail_distance,
                    ) = self._extract_tail_context(source, window_depth)
                    if len(source_tail_indices) > 12:
                        continue

                    candidate_targets = sorted(
                        (target for target in states if target.driver_id != source.driver_id),
                        key=lambda state: state.elapsed_time,
                    )[: self.params.lns_target_pool]

                    for target in candidate_targets:
                        repaired = self._repair_orders_between_states(
                            source_start=source_restart,
                            source_capacity=source.capacity,
                            source_base_time=source.elapsed_time - source_tail_distance,
                            target_start=target.position,
                            target_capacity=target.capacity,
                            target_base_time=target.elapsed_time,
                            order_indices=source_tail_indices,
                            other_max=other_max,
                        )
                        if repaired is None:
                            continue

                        source_tail_plans, target_tail_plans = repaired
                        new_source_time = (
                            source.elapsed_time - source_tail_distance
                            + self._batch_sequence_distance(source_tail_plans)
                        )
                        new_target_time = (
                            target.elapsed_time
                            + self._batch_sequence_distance(target_tail_plans)
                        )
                        new_makespan = max(other_max, new_source_time, new_target_time)
                        if new_makespan >= current_makespan - 1e-6:
                            continue

                        action = (
                            new_makespan,
                            source,
                            target,
                            window_depth,
                            source_tail_plans,
                            target_tail_plans,
                        )
                        if best_action is None or action[0] < best_action[0]:
                            best_action = action

            if best_action is None:
                return

            _, source, target, window_depth, source_tail_plans, target_tail_plans = best_action
            self._replace_tail_batches(source, window_depth, source_tail_plans)
            for batch_plan in target_tail_plans:
                self._append_batch(target, batch_plan)
                target.orders_completed += len(batch_plan.order_indices)
                target.batches_completed += 1

    @staticmethod
    def _iter_movable_subsets(order_indices: tuple[int, ...]):
        total_orders = len(order_indices)
        for subset_size in range(1, total_orders + 1):
            for subset in itertools.combinations(order_indices, subset_size):
                yield subset

    @staticmethod
    def _iter_partition_subsets(order_indices: tuple[int, ...]):
        total_orders = len(order_indices)
        for subset_size in range(1, total_orders):
            for subset in itertools.combinations(order_indices, subset_size):
                yield subset

    def _refresh_active_clusters(self, states_by_id: dict[str, DriverState]) -> None:
        active_indices = [index for index, is_unassigned in enumerate(self.unassigned) if is_unassigned]
        if not active_indices:
            self.cluster_count = 0
            self.cluster_elbow_curve = []
            self.cluster_centers = []
            self.order_cluster_ids = [-1] * len(self.orders)
            self.cluster_members = []
            self.region_count = 0
            self.region_elbow_curve = []
            self.region_members = []
            self.order_region_ids = [-1] * len(self.orders)
            self.driver_region_ids = {}
            return

        (
            self.cluster_count,
            self.cluster_elbow_curve,
            self.cluster_centers,
            self.order_cluster_ids,
            self.cluster_members,
        ) = self._build_order_clusters(active_indices)
        (
            self.region_count,
            self.region_elbow_curve,
            self.order_region_ids,
            self.region_members,
            self.driver_region_ids,
        ) = self._build_driver_regions(
            cluster_members=self.cluster_members,
            cluster_centers=self.cluster_centers,
            states_by_id=states_by_id,
        )

    def _build_order_clusters(
        self,
        active_indices: list[int],
    ) -> tuple[
        int,
        list[tuple[int, float]],
        list[tuple[float, float, float, float]],
        list[int],
        list[list[int]],
    ]:
        order_count = len(active_indices)
        if order_count == 0:
            return (0, [], [], [-1] * len(self.orders), [])

        max_capacity = max((driver.capacity for driver in self.drivers), default=20)
        size_scale = self.params.cluster_size_scale_base / max(max_capacity, 1)
        features = [
            (
                float(self.orders[index].pickup_x),
                float(self.orders[index].pickup_y),
                float(self.orders[index].delivery_x),
                float(self.orders[index].delivery_y),
                float(self.orders[index].size) * size_scale,
            )
            for index in active_indices
        ]
        cluster_count, elbow_curve = self._choose_cluster_count_with_elbow(features)
        local_labels, local_members, _ = self._run_kmeans(features, cluster_count)

        cluster_centers: list[tuple[float, float, float, float]] = []
        global_labels = [-1] * len(self.orders)
        global_members = [[] for _ in range(cluster_count)]
        for local_index, cluster_id in enumerate(local_labels):
            global_index = active_indices[local_index]
            global_labels[global_index] = cluster_id
            global_members[cluster_id].append(global_index)

        for member_indices in global_members:
            if not member_indices:
                cluster_centers.append((0.0, 0.0, 0.0, 0.0))
                continue
            pickup_x = sum(self.orders[index].pickup_x for index in member_indices) / len(member_indices)
            pickup_y = sum(self.orders[index].pickup_y for index in member_indices) / len(member_indices)
            delivery_x = sum(self.orders[index].delivery_x for index in member_indices) / len(member_indices)
            delivery_y = sum(self.orders[index].delivery_y for index in member_indices) / len(member_indices)
            cluster_centers.append((pickup_x, pickup_y, delivery_x, delivery_y))

        return (cluster_count, elbow_curve, cluster_centers, global_labels, global_members)

    def _build_driver_regions(
        self,
        cluster_members: list[list[int]],
        cluster_centers: list[tuple[float, float, float, float]],
        states_by_id: dict[str, DriverState],
    ) -> tuple[int, list[tuple[int, float]], list[int], list[list[int]], dict[str, int]]:
        active_cluster_ids = [
            cluster_id
            for cluster_id, member_indices in enumerate(cluster_members)
            if member_indices
        ]
        driver_ids = sorted(states_by_id)
        if not active_cluster_ids or not driver_ids:
            return (0, [], [-1] * len(self.orders), [], {})

        combined_features = [cluster_centers[cluster_id] for cluster_id in active_cluster_ids]
        combined_features.extend(
            self._driver_feature(states_by_id[driver_id].position)
            for driver_id in driver_ids
        )

        max_region_count = min(len(combined_features), len(driver_ids))
        candidate_counts = sorted(
            {
                min(max_region_count, max(2, int(len(driver_ids) * ratio)))
                for ratio in (0.45, 0.6, 0.75, 0.9, 1.0)
            }
        )
        region_count, elbow_curve = self._choose_cluster_count_with_elbow(
            combined_features,
            candidate_counts=candidate_counts,
        )
        labels, _, _ = self._run_kmeans(combined_features, region_count)

        driver_region_ids: dict[str, int] = {}
        for offset, driver_id in enumerate(driver_ids, start=len(active_cluster_ids)):
            driver_region_ids[driver_id] = labels[offset]

        populated_driver_regions = set(driver_region_ids.values())
        cluster_region_ids: dict[int, int] = {}
        for offset, cluster_id in enumerate(active_cluster_ids):
            assigned_region = labels[offset]
            if assigned_region not in populated_driver_regions:
                assigned_region = self._nearest_driver_region(
                    cluster_center=cluster_centers[cluster_id],
                    driver_ids=driver_ids,
                    states_by_id=states_by_id,
                    driver_region_ids=driver_region_ids,
                )
            cluster_region_ids[cluster_id] = assigned_region

        order_region_ids = [-1] * len(self.orders)
        region_members = [[] for _ in range(region_count)]
        for cluster_id, member_indices in enumerate(cluster_members):
            if not member_indices:
                continue
            region_id = cluster_region_ids.get(cluster_id, -1)
            if region_id < 0:
                continue
            for order_index in member_indices:
                order_region_ids[order_index] = region_id
                region_members[region_id].append(order_index)

        return (region_count, elbow_curve, order_region_ids, region_members, driver_region_ids)

    def _choose_cluster_count_with_elbow(
        self,
        features: list[tuple[float, ...]],
        candidate_counts: list[int] | None = None,
    ) -> tuple[int, list[tuple[int, float]]]:
        point_count = len(features)
        driver_count = max(len(self.drivers), 1)
        if point_count <= 2:
            return (point_count, [(point_count, 0.0)])

        if candidate_counts is None:
            candidate_counts = sorted(
                {
                    min(point_count, max(24, int(driver_count * ratio)))
                    for ratio in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
                }
            )
        else:
            candidate_counts = sorted(
                {
                    min(point_count, max(1, int(cluster_count)))
                    for cluster_count in candidate_counts
                }
            )
        if len(candidate_counts) == 1:
            return (candidate_counts[0], [(candidate_counts[0], 0.0)])

        rng = random.Random(self.params.random_seed + 7)
        sample_size = min(point_count, self.params.kmeans_sample_size)
        sample_indices = list(range(point_count))
        if sample_size < point_count:
            sample_indices = sorted(rng.sample(sample_indices, sample_size))
        sample_features = [features[index] for index in sample_indices]

        elbow_curve: list[tuple[int, float]] = []
        for cluster_count in candidate_counts:
            _, _, inertia = self._run_kmeans(sample_features, cluster_count)
            elbow_curve.append((cluster_count, inertia))

        first_k, first_inertia = elbow_curve[0]
        last_k, last_inertia = elbow_curve[-1]
        dx = float(last_k - first_k)
        dy = float(last_inertia - first_inertia)
        denom = math.hypot(dx, dy)
        if denom == 0.0:
            return (candidate_counts[len(candidate_counts) // 2], elbow_curve)

        best_k = elbow_curve[0][0]
        best_distance = -1.0
        for cluster_count, inertia in elbow_curve[1:-1]:
            distance = abs(
                dy * cluster_count - dx * inertia + last_k * first_inertia - last_inertia * first_k
            ) / denom
            if distance > best_distance:
                best_distance = distance
                best_k = cluster_count

        return (best_k, elbow_curve)

    def _run_kmeans(
        self,
        features: list[tuple[float, ...]],
        cluster_count: int,
    ) -> tuple[list[int], list[list[int]], float]:
        order_count = len(features)
        if cluster_count >= order_count:
            labels = list(range(order_count))
            members = [[index] for index in range(order_count)]
            return (labels, members, 0.0)

        rng = random.Random(self.params.random_seed + 42)
        first_index = rng.randrange(order_count)
        centers = [features[first_index]]
        min_distances = [squared_feature_distance(point, centers[0]) for point in features]

        while len(centers) < cluster_count:
            total_distance = sum(min_distances)
            if total_distance <= 0.0:
                centers.append(features[rng.randrange(order_count)])
                continue

            threshold = rng.random() * total_distance
            cumulative = 0.0
            chosen_index = order_count - 1
            for index, distance in enumerate(min_distances):
                cumulative += distance
                if cumulative >= threshold:
                    chosen_index = index
                    break

            centers.append(features[chosen_index])
            new_center = centers[-1]
            for index, point in enumerate(features):
                candidate_distance = squared_feature_distance(point, new_center)
                if candidate_distance < min_distances[index]:
                    min_distances[index] = candidate_distance

        labels = [0] * order_count
        dimension = len(features[0])
        for _ in range(self.params.kmeans_iterations):
            changed = False
            sums = [[0.0] * dimension + [0.0] for _ in range(cluster_count)]

            for index, point in enumerate(features):
                best_cluster = 0
                best_distance = squared_feature_distance(point, centers[0])
                for cluster_id in range(1, cluster_count):
                    distance = squared_feature_distance(point, centers[cluster_id])
                    if distance < best_distance:
                        best_cluster = cluster_id
                        best_distance = distance
                if labels[index] != best_cluster:
                    labels[index] = best_cluster
                    changed = True

                slot = sums[best_cluster]
                for axis, value in enumerate(point):
                    slot[axis] += value
                slot[dimension] += 1

            new_centers: list[tuple[float, ...]] = []
            for cluster_id, slot in enumerate(sums):
                if slot[dimension] == 0:
                    new_centers.append(features[rng.randrange(order_count)])
                    continue
                count = slot[dimension]
                new_centers.append(
                    tuple(
                        slot[axis] / count
                        for axis in range(dimension)
                    )
                )

            centers = new_centers
            if not changed:
                break

        members = [[] for _ in range(cluster_count)]
        inertia = 0.0
        for index, label in enumerate(labels):
            best_cluster = 0
            best_distance = squared_feature_distance(features[index], centers[0])
            for cluster_id in range(1, cluster_count):
                distance = squared_feature_distance(features[index], centers[cluster_id])
                if distance < best_distance:
                    best_cluster = cluster_id
                    best_distance = distance
            labels[index] = best_cluster
            members[best_cluster].append(index)
            inertia += best_distance

        return (labels, members, inertia)

    def _collect_cluster_candidates(
        self,
        seed_index: int,
        capacity: int,
        limit: int,
    ) -> list[int]:
        seed_cluster = self.order_cluster_ids[seed_index]
        if seed_cluster < 0 or seed_cluster >= len(self.cluster_members):
            return []
        seed_order = self.orders[seed_index]
        scored: list[tuple[float, int]] = []

        for candidate_index in self.cluster_members[seed_cluster]:
            if candidate_index == seed_index or not self.unassigned[candidate_index]:
                continue
            candidate = self.orders[candidate_index]
            if candidate.size > capacity:
                continue

            pickup_gap = euclidean(seed_order.pickup, candidate.pickup)
            delivery_gap = euclidean(seed_order.delivery, candidate.delivery)
            size_ratio = candidate.size / max(capacity, 1)
            score = (
                0.5 * pickup_gap
                + 0.4 * delivery_gap
                + 6.0 * size_ratio
                - 0.1 * candidate.size
            )
            scored.append((score, candidate_index))

        scored.sort(key=lambda item: (item[0], self.orders[item[1]].order_id))
        return [candidate_index for _, candidate_index in scored[:limit]]

    def _collect_region_candidates(
        self,
        driver_id: str,
        center: tuple[int, int],
        capacity: int,
        limit: int,
    ) -> list[int]:
        region_id = self.driver_region_ids.get(driver_id, -1)
        if region_id < 0 or region_id >= len(self.region_members):
            return []

        scored: list[tuple[float, int]] = []
        for order_index in self.region_members[region_id]:
            if not self.unassigned[order_index]:
                continue
            order = self.orders[order_index]
            if order.size > capacity:
                continue

            pickup_gap = euclidean(center, order.pickup)
            score = pickup_gap + 0.25 * order.trip_distance - 0.15 * order.size
            scored.append((score, order_index))

        scored.sort(key=lambda item: (item[0], self.orders[item[1]].order_id))
        return [order_index for _, order_index in scored[:limit]]

    def _region_match_bonus(
        self,
        driver_id: str,
        order_index: int,
        base_bonus: float,
    ) -> float:
        driver_region = self.driver_region_ids.get(driver_id, -1)
        order_region = self.order_region_ids[order_index]
        if driver_region < 0 or driver_region != order_region:
            return 0.0
        if not self.params.adaptive_region_bias:
            return base_bonus

        populated_sizes = [
            len(member_indices)
            for member_indices in self.region_members
            if member_indices
        ]
        if not populated_sizes:
            return base_bonus
        average_size = sum(populated_sizes) / len(populated_sizes)
        if average_size <= 0:
            return base_bonus

        region_size = len(self.region_members[driver_region])
        multiplier = max(0.65, min(1.65, 0.75 + 0.5 * (region_size / average_size)))
        return base_bonus * multiplier

    @staticmethod
    def _driver_feature(position: tuple[int, int]) -> tuple[float, float, float, float]:
        return (
            float(position[0]),
            float(position[1]),
            float(position[0]),
            float(position[1]),
        )

    def _nearest_driver_region(
        self,
        cluster_center: tuple[float, float, float, float],
        driver_ids: list[str],
        states_by_id: dict[str, DriverState],
        driver_region_ids: dict[str, int],
    ) -> int:
        nearest_driver_id = min(
            driver_ids,
            key=lambda driver_id: squared_feature_distance(
                cluster_center,
                self._driver_feature(states_by_id[driver_id].position),
            ),
        )
        return driver_region_ids[nearest_driver_id]

    def _solve_batch_route_exact(
        self,
        start: tuple[int, int],
        order_indices: tuple[int, ...],
        capacity: int,
    ) -> tuple[float, tuple[tuple[str, int], ...], int]:
        orders = [self.orders[index] for index in order_indices]
        sizes = [order.size for order in orders]
        full_mask = (1 << len(order_indices)) - 1

        def load_of(picked_mask: int, delivered_mask: int) -> int:
            total = 0
            for bit, size in enumerate(sizes):
                if picked_mask & (1 << bit) and not delivered_mask & (1 << bit):
                    total += size
            return total

        def position_of(last_action: tuple[str, int] | None) -> tuple[int, int]:
            if last_action is None:
                return start
            action_type, local_index = last_action
            order = orders[local_index]
            return order.pickup if action_type == "P" else order.delivery

        memo: dict[tuple[int, int, tuple[str, int] | None], tuple[float, tuple[tuple[str, int], ...]]] = {}

        def search(
            picked_mask: int,
            delivered_mask: int,
            last_action: tuple[str, int] | None,
        ) -> tuple[float, tuple[tuple[str, int], ...]]:
            state = (picked_mask, delivered_mask, last_action)
            if state in memo:
                return memo[state]
            if delivered_mask == full_mask:
                memo[state] = (0.0, ())
                return memo[state]

            current_position = position_of(last_action)
            current_load = load_of(picked_mask, delivered_mask)
            best_cost = math.inf
            best_steps: tuple[tuple[str, int], ...] = ()

            for local_index, order in enumerate(orders):
                bit = 1 << local_index
                if not picked_mask & bit:
                    if current_load + order.size > capacity:
                        continue
                    next_action = ("P", local_index)
                    step_cost = euclidean(current_position, order.pickup)
                    tail_cost, tail_steps = search(
                        picked_mask | bit,
                        delivered_mask,
                        next_action,
                    )
                elif not delivered_mask & bit:
                    next_action = ("D", local_index)
                    step_cost = euclidean(current_position, order.delivery)
                    tail_cost, tail_steps = search(
                        picked_mask,
                        delivered_mask | bit,
                        next_action,
                    )
                else:
                    continue

                total_cost = step_cost + tail_cost
                candidate_steps = (next_action,) + tail_steps
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_steps = candidate_steps
                elif total_cost == best_cost and candidate_steps < best_steps:
                    best_steps = candidate_steps

            memo[state] = (best_cost, best_steps)
            return memo[state]

        best_cost, best_steps = search(0, 0, None)
        if not math.isfinite(best_cost):
            return (math.inf, (), 0)

        current_load = 0
        peak_load = 0
        delivered_mask = 0
        picked_mask = 0
        for action_type, local_index in best_steps:
            bit = 1 << local_index
            if action_type == "P":
                picked_mask |= bit
                current_load += sizes[local_index]
            else:
                delivered_mask |= bit
                current_load -= sizes[local_index]
            peak_load = max(peak_load, current_load)

        return (best_cost, best_steps, peak_load)

    def _find_seed_order(self, state: DriverState) -> int | None:
        nearby_candidates = self._collect_candidates(
            center=state.position,
            capacity=state.capacity,
            target=self.params.seed_candidate_target,
        )
        regional_candidates = self._collect_region_candidates(
            driver_id=state.driver_id,
            center=state.position,
            capacity=state.capacity,
            limit=self.params.region_candidate_limit,
        )
        merged_candidates: list[int] = []
        seen_candidates: set[int] = set()
        for order_index in nearby_candidates + regional_candidates:
            if order_index in seen_candidates:
                continue
            seen_candidates.add(order_index)
            merged_candidates.append(order_index)

        def score(order_index: int) -> tuple[float, str]:
            order = self.orders[order_index]
            deadhead = euclidean(state.position, order.pickup)
            trip = order.trip_distance
            region_bonus = self._region_match_bonus(
                driver_id=state.driver_id,
                order_index=order_index,
                base_bonus=self.params.region_match_bonus_seed,
            )
            return (deadhead + 0.35 * trip - 0.2 * order.size - region_bonus, order.order_id)

        if merged_candidates:
            return min(merged_candidates, key=score)

        if not nearby_candidates:
            fallback_best: int | None = None
            fallback_score: tuple[float, str] | None = None
            for order_index, is_unassigned in enumerate(self.unassigned):
                if not is_unassigned:
                    continue
                if self.orders[order_index].size > state.capacity:
                    continue
                candidate_score = score(order_index)
                if fallback_score is None or candidate_score < fallback_score:
                    fallback_best = order_index
                    fallback_score = candidate_score
            return fallback_best

        return min(nearby_candidates, key=score)

    def _collect_candidates(
        self,
        center: tuple[int, int],
        capacity: int,
        target: int,
    ) -> list[int]:
        cx = int(center[0])
        cy = int(center[1])
        candidates: list[int] = []
        seen_cells: set[tuple[int, int]] = set()
        for radius in SEARCH_RADII:
            for cell in iter_border_cells(cx, cy, radius):
                if cell in seen_cells:
                    continue
                seen_cells.add(cell)
                for order_index in self.pickup_grid.get(cell, ()):
                    if not self.unassigned[order_index]:
                        continue
                    if self.orders[order_index].size > capacity:
                        continue
                    candidates.append(order_index)
                    if len(candidates) >= target:
                        return candidates
        return candidates

    @staticmethod
    def _nearest_neighbor_route(
        start: tuple[int, int],
        order_indices: list[int],
        point_getter: Callable[[int], tuple[int, int]],
    ) -> list[int]:
        remaining = order_indices.copy()
        route: list[int] = []
        current = start
        while remaining:
            best_position = min(
                range(len(remaining)),
                key=lambda pos: (euclidean(current, point_getter(remaining[pos])), remaining[pos]),
            )
            chosen = remaining.pop(best_position)
            route.append(chosen)
            current = point_getter(chosen)
        return route


def batch_order_limit(capacity: int) -> int:
    if capacity <= 4:
        return 2
    if capacity <= 8:
        return 3
    if capacity <= 12:
        return 4
    if capacity <= 16:
        return 5
    return 6


def batch_total_size_limit(capacity: int) -> int:
    return capacity + min(8, max(3, capacity // 2))


def iter_border_cells(cx: int, cy: int, radius: int):
    if radius == 0:
        if GRID_MIN <= cx <= GRID_MAX and GRID_MIN <= cy <= GRID_MAX:
            yield (cx, cy)
        return

    x_min = max(GRID_MIN, cx - radius)
    x_max = min(GRID_MAX, cx + radius)
    y_min = max(GRID_MIN, cy - radius)
    y_max = min(GRID_MAX, cy + radius)

    for x in range(x_min, x_max + 1):
        yield (x, y_min)
    if y_max != y_min:
        for x in range(x_min, x_max + 1):
            yield (x, y_max)

    for y in range(y_min + 1, y_max):
        yield (x_min, y)
    if x_max != x_min:
        for y in range(y_min + 1, y_max):
            yield (x_max, y)


def optimize_schedule(
    orders: list[Order],
    drivers: list[Driver],
    params: SolverParams | None = None,
) -> dict[str, list[str]]:
    effective_params = params or SolverParams()
    if effective_params.multi_start_count <= 1:
        optimizer = DeliveryOptimizer(orders=orders, drivers=drivers, params=effective_params)
        return optimizer.solve()

    best_schedule: dict[str, list[str]] | None = None
    best_metrics: tuple[float, float] | None = None
    for restart_index in range(effective_params.multi_start_count):
        variant_params = effective_params.with_updates(
            random_seed=effective_params.random_seed
            + restart_index * effective_params.multi_start_seed_step,
            shuffle_driver_order=True,
        )
        optimizer = DeliveryOptimizer(orders=orders, drivers=drivers, params=variant_params)
        schedule = optimizer.solve()
        evaluation = evaluate_schedule(orders=orders, drivers=drivers, schedule=schedule)
        metrics = (evaluation.makespan, evaluation.total_distance)
        if best_metrics is None or metrics < best_metrics:
            best_metrics = metrics
            best_schedule = schedule

    if best_schedule is None:
        raise RuntimeError("Multi-start search did not produce a schedule.")
    return best_schedule


def evaluate_schedule(
    orders: list[Order],
    drivers: list[Driver],
    schedule: dict[str, list[str]],
) -> Evaluation:
    orders_by_id = {order.order_id: order for order in orders}
    drivers_by_id = {driver.driver_id: driver for driver in drivers}
    seen_pickups: set[str] = set()
    seen_deliveries: set[str] = set()
    route_distances: dict[str, float] = {}

    for driver in drivers:
        actions = schedule.get(driver.driver_id, [])
        onboard: set[str] = set()
        position = driver.start
        carried_size = 0
        route_distance = 0.0

        for action in actions:
            try:
                action_type, order_id = action.split(":", 1)
            except ValueError as exc:
                raise ValueError(f"Invalid action format: {action}") from exc

            order_id = order_id.strip()
            if order_id not in orders_by_id:
                raise ValueError(f"Unknown order referenced in schedule: {order_id}")
            order = orders_by_id[order_id]

            if action_type == "P":
                if order.size > driver.capacity:
                    raise ValueError(f"{driver.driver_id} cannot carry {order_id}.")
                if order_id in seen_pickups:
                    raise ValueError(f"Order {order_id} was picked up more than once.")
                route_distance += euclidean(position, order.pickup)
                position = order.pickup
                carried_size += order.size
                if carried_size > driver.capacity:
                    raise ValueError(f"Capacity exceeded by {driver.driver_id}.")
                onboard.add(order_id)
                seen_pickups.add(order_id)
            elif action_type == "D":
                if order_id not in onboard:
                    raise ValueError(f"Order {order_id} delivered before pickup by {driver.driver_id}.")
                route_distance += euclidean(position, order.delivery)
                position = order.delivery
                carried_size -= order.size
                onboard.remove(order_id)
                if order_id in seen_deliveries:
                    raise ValueError(f"Order {order_id} was delivered more than once.")
                seen_deliveries.add(order_id)
            else:
                raise ValueError(f"Unknown action type: {action}")

        if onboard:
            raise ValueError(f"{driver.driver_id} finished route with undelivered orders.")
        route_distances[driver.driver_id] = route_distance

    missing_pickups = set(orders_by_id) - seen_pickups
    missing_deliveries = set(orders_by_id) - seen_deliveries
    if missing_pickups or missing_deliveries:
        raise ValueError(
            "Schedule is incomplete. "
            f"Missing pickups: {len(missing_pickups)}, missing deliveries: {len(missing_deliveries)}"
        )

    makespan = max(route_distances.values(), default=0.0)
    total_distance = sum(route_distances.values())
    return Evaluation(
        makespan=makespan,
        total_distance=total_distance,
        route_distances=route_distances,
        delivered_orders=len(seen_deliveries),
    )
