from __future__ import annotations

import csv
import heapq
import itertools
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class Evaluation:
    makespan: float
    total_distance: float
    route_distances: dict[str, float]
    delivered_orders: int


def euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def squared_feature_distance(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    return (
        (a[0] - b[0]) ** 2
        + (a[1] - b[1]) ** 2
        + (a[2] - b[2]) ** 2
        + (a[3] - b[3]) ** 2
    )


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
    def __init__(self, orders: list[Order], drivers: list[Driver]) -> None:
        self.orders = orders
        self.drivers = drivers
        self.unassigned = [True] * len(orders)
        self.remaining_orders = len(orders)
        self.pickup_grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.batch_plan_cache: dict[tuple[int, int, int, tuple[int, ...]], BatchPlan] = {}
        (
            self.order_cluster_ids,
            self.cluster_members,
        ) = self._build_order_clusters()
        for index, order in enumerate(orders):
            self.pickup_grid[order.pickup].append(index)

    def solve(self) -> dict[str, list[str]]:
        available: list[DriverState] = []
        states_by_id: dict[str, DriverState] = {}
        driver_lookahead = 4
        for index, driver in enumerate(self.drivers):
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

        self._rebalance_driver_loads(states_by_id)
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
        if used_capacity >= state.capacity or max_batch_orders == 1:
            return batch

        extra_candidates = self._collect_candidates(
            center=seed_order.pickup,
            capacity=state.capacity,
            target=72,
        )
        clustered_candidates = self._collect_cluster_candidates(
            seed_index=seed_index,
            capacity=state.capacity,
            limit=24,
        )
        preferred_cluster_candidates = set(clustered_candidates)
        merged_candidates: list[int] = []
        seen_candidates: set[int] = set()
        for candidate_index in clustered_candidates + extra_candidates:
            if candidate_index in seen_candidates:
                continue
            merged_candidates.append(candidate_index)
            seen_candidates.add(candidate_index)

        scored_candidates: list[tuple[float, int]] = []
        for candidate_index in merged_candidates:
            if candidate_index == seed_index or not self.unassigned[candidate_index]:
                continue
            candidate = self.orders[candidate_index]
            if used_capacity + candidate.size > state.capacity:
                continue

            pickup_gap = euclidean(seed_order.pickup, candidate.pickup)
            delivery_gap = euclidean(seed_order.delivery, candidate.delivery)
            in_seed_cluster = candidate_index in preferred_cluster_candidates
            if not in_seed_cluster and pickup_gap > 20.0 and delivery_gap > 28.0:
                continue

            score = 0.6 * pickup_gap + 0.4 * delivery_gap - 0.25 * candidate.size
            if in_seed_cluster:
                score -= 5.0
            scored_candidates.append((score, candidate_index))

        scored_candidates.sort(key=lambda item: (item[0], self.orders[item[1]].order_id))
        candidate_pool = [candidate_index for _, candidate_index in scored_candidates[:18]]
        current_plan = self._build_batch_plan(state.position, batch, state.capacity)

        while candidate_pool and len(batch) < max_batch_orders and used_capacity < state.capacity:
            best_choice: tuple[float, int, BatchPlan] | None = None
            for candidate_index in list(candidate_pool):
                candidate = self.orders[candidate_index]
                if used_capacity + candidate.size > state.capacity:
                    continue

                tentative_batch = batch + [candidate_index]
                tentative_plan = self._build_batch_plan(
                    state.position,
                    tentative_batch,
                    state.capacity,
                )
                marginal_cost = tentative_plan.distance - current_plan.distance
                efficiency_score = marginal_cost / max(candidate.size, 1)
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

        return batch

    def _build_batch_plan(
        self,
        start: tuple[int, int],
        order_indices: list[int] | tuple[int, ...],
        capacity: int,
    ) -> BatchPlan:
        canonical_indices = tuple(sorted(order_indices))
        cache_key = (start[0], start[1], capacity, canonical_indices)
        cached = self.batch_plan_cache.get(cache_key)
        if cached is not None:
            return cached

        total_size = sum(self.orders[order_index].size for order_index in canonical_indices)
        if total_size > capacity:
            raise ValueError("Batch exceeds driver capacity.")

        route_distance, action_steps = self._solve_batch_route_exact(
            start=start,
            order_indices=canonical_indices,
            capacity=capacity,
        )

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

    def _rebalance_driver_loads(self, states_by_id: dict[str, DriverState]) -> None:
        states = list(states_by_id.values())
        if len(states) < 2:
            return

        max_rounds = 120
        target_pool = 60
        improvement_floor = 1e-6

        for _ in range(max_rounds):
            ranked = sorted(states, key=lambda state: state.elapsed_time, reverse=True)
            current_makespan = ranked[0].elapsed_time
            best_action: tuple | None = None

            for source_rank, source in enumerate(ranked[:16]):
                if not source.batches:
                    continue
                other_max = ranked[0].elapsed_time
                if source_rank == 0 and len(ranked) > 1:
                    other_max = ranked[1].elapsed_time

                batch_plan = source.batches[-1]
                if batch_plan.total_size > max(
                    target.capacity for target in states if target.driver_id != source.driver_id
                ):
                    continue

                candidate_targets = sorted(
                    (target for target in states if target.driver_id != source.driver_id),
                    key=lambda state: state.elapsed_time,
                )[:target_pool]
                source_restart = source.batches[-2].end_pos if len(source.batches) > 1 else source.start

                for target in candidate_targets:
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

    def _build_order_clusters(self) -> tuple[list[int], list[list[int]]]:
        order_count = len(self.orders)
        if order_count == 0:
            return ([], [])

        cluster_count = min(max(len(self.drivers) + len(self.drivers) // 2, 80), order_count)
        features = [
            (
                float(order.pickup_x),
                float(order.pickup_y),
                float(order.delivery_x),
                float(order.delivery_y),
            )
            for order in self.orders
        ]

        rng = random.Random(42)
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
        for _ in range(8):
            changed = False
            sums = [[0.0, 0.0, 0.0, 0.0, 0] for _ in range(cluster_count)]

            for index, point in enumerate(features):
                best_cluster = min(
                    range(cluster_count),
                    key=lambda cluster_id: squared_feature_distance(point, centers[cluster_id]),
                )
                if labels[index] != best_cluster:
                    labels[index] = best_cluster
                    changed = True

                slot = sums[best_cluster]
                slot[0] += point[0]
                slot[1] += point[1]
                slot[2] += point[2]
                slot[3] += point[3]
                slot[4] += 1

            new_centers: list[tuple[float, float, float, float]] = []
            for cluster_id, slot in enumerate(sums):
                if slot[4] == 0:
                    new_centers.append(features[rng.randrange(order_count)])
                    continue
                count = slot[4]
                new_centers.append(
                    (
                        slot[0] / count,
                        slot[1] / count,
                        slot[2] / count,
                        slot[3] / count,
                    )
                )

            centers = new_centers
            if not changed:
                break

        members = [[] for _ in range(cluster_count)]
        for index, label in enumerate(labels):
            members[label].append(index)

        return (labels, members)

    def _collect_cluster_candidates(
        self,
        seed_index: int,
        capacity: int,
        limit: int,
    ) -> list[int]:
        seed_cluster = self.order_cluster_ids[seed_index]
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
            score = 0.55 * pickup_gap + 0.45 * delivery_gap - 0.2 * candidate.size
            scored.append((score, candidate_index))

        scored.sort(key=lambda item: (item[0], self.orders[item[1]].order_id))
        return [candidate_index for _, candidate_index in scored[:limit]]

    def _solve_batch_route_exact(
        self,
        start: tuple[int, int],
        order_indices: tuple[int, ...],
        capacity: int,
    ) -> tuple[float, tuple[tuple[str, int], ...]]:
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

        return search(0, 0, None)

    def _find_seed_order(self, state: DriverState) -> int | None:
        candidates = self._collect_candidates(
            center=state.position,
            capacity=state.capacity,
            target=40,
        )

        def score(order_index: int) -> tuple[float, str]:
            order = self.orders[order_index]
            deadhead = euclidean(state.position, order.pickup)
            trip = order.trip_distance
            return (deadhead + 0.35 * trip - 0.2 * order.size, order.order_id)

        if not candidates:
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

        return min(candidates, key=score)

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


def optimize_schedule(orders: list[Order], drivers: list[Driver]) -> dict[str, list[str]]:
    optimizer = DeliveryOptimizer(orders=orders, drivers=drivers)
    return optimizer.solve()


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
