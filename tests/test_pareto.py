from dataclasses import dataclass
from typing import Optional

from openrouter_frontier.pareto import Objective, annotate_frontier, cost_quality_frontier, pareto_mask


@dataclass
class Item:
    cost: float
    speed: Optional[float]


OBJECTIVES = [Objective(lambda i: i.cost, minimize=True), Objective(lambda i: i.speed, minimize=False)]


def test_pareto_mask_basic_dominance():
    items = [Item(1.0, 10), Item(2.0, 20), Item(2.0, 5), Item(1.0, 10)]
    mask = pareto_mask(items, OBJECTIVES)
    # (2.0, 5) is beaten by (1.0, 10) on both axes; equal items do not dominate each other.
    assert mask == [True, True, False, True]


def test_missing_values_never_dominate():
    items = [Item(1.0, None), Item(1.0, 1.0)]
    assert pareto_mask(items, OBJECTIVES) == [False, True]


def test_cost_quality_frontier_and_efficient_point():
    cands = [
        {"id": "a", "cost": 1.0, "score": 10.0},
        {"id": "b", "cost": 2.0, "score": 30.0},   # big jump for little cost => efficient point
        {"id": "c", "cost": 3.0, "score": 20.0},   # dominated by b
        {"id": "d", "cost": 10.0, "score": 31.0},
    ]
    frontier, efficient = cost_quality_frontier(cands)
    assert [f["id"] for f in frontier] == ["a", "b", "d"]
    assert frontier[efficient]["id"] == "b"

    annotate_frontier(cands, frontier, efficient)
    by_id = {c["id"]: c for c in cands}
    assert by_id["c"]["on_frontier"] is False
    assert by_id["c"]["dist"] == 10.0  # best cheaper frontier score (30) minus own score (20)
    assert by_id["b"]["is_efficient"] is True
    assert by_id["a"]["dist"] == 0.0


def test_frontier_needs_three_points_for_efficient_point():
    frontier, efficient = cost_quality_frontier([{"id": "a", "cost": 1, "score": 1}, {"id": "b", "cost": 2, "score": 2}])
    assert len(frontier) == 2 and efficient is None
    assert cost_quality_frontier([]) == ([], None)
