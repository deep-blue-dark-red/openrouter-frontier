"""Multi-objective Pareto frontier utilities.

Two flavours are provided:

* :func:`pareto_mask` - general N-objective dominance test. An item ``a`` is dominated
  when some other item ``b`` is at least as good on every objective and strictly better
  on at least one. Items that are not dominated form the frontier.
* :func:`cost_quality_frontier` - the classic 2-D "minimise cost, maximise score" sweep,
  plus efficient point detection and vertical distance-to-frontier for dominated items.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Objective:
    """One axis of a Pareto comparison.

    :param key: Extracts the metric from an item. May return ``None`` for missing data.
    :param minimize: ``True`` if smaller is better (cost, latency), ``False`` if larger is
                     better (throughput, uptime, cache hit rate).
    :param missing: Value substituted when ``key`` returns ``None``. Defaults to the worst
                    possible value so items with missing data never dominate on that axis.
    """

    key: Callable[[Any], Optional[float]]
    minimize: bool = True
    missing: Optional[float] = None

    def value(self, item: Any) -> float:
        v = self.key(item)
        if v is not None:
            return float(v)
        if self.missing is not None:
            return self.missing
        return float("inf") if self.minimize else float("-inf")


def dominates(b_vals: Sequence[float], a_vals: Sequence[float], objectives: Sequence[Objective]) -> bool:
    """Return ``True`` if ``b`` weakly beats ``a`` on every objective and strictly on at least one."""
    strict = False
    for bv, av, obj in zip(b_vals, a_vals, objectives):
        if obj.minimize:
            if bv > av:
                return False
            strict = strict or bv < av
        else:
            if bv < av:
                return False
            strict = strict or bv > av
    return strict


def pareto_mask(items: Sequence[T], objectives: Sequence[Objective]) -> List[bool]:
    """Return a list of booleans, ``True`` where ``items[i]`` is Pareto-optimal.

    Runs in O(n^2 * k) for ``n`` items and ``k`` objectives, which is more than fast enough
    for the tens of providers or hundreds of models this project compares.
    """
    vals = [[obj.value(it) for obj in objectives] for it in items]
    mask = []
    for i, a in enumerate(vals):
        dominated = any(j != i and dominates(b, a, objectives) for j, b in enumerate(vals))
        mask.append(not dominated)
    return mask


def cost_quality_frontier(
    candidates: List[Dict[str, Any]],
    cost_key: str = "cost",
    score_key: str = "score",
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Compute the 2-D frontier for "minimise cost, maximise score" and locate its efficient point.

    Candidates are sorted by ascending cost (ties broken by descending score); an item is on
    the frontier if its score exceeds every cheaper item's score.

    The efficient point is the frontier point that maximises ``norm_score - norm_cost``, where both
    axes are min-max normalised to [0, 1] over the frontier. Geometrically that is the point
    furthest above the chord joining the cheapest and the highest-scoring frontier items,
    i.e. the best marginal quality gained per dollar. An efficient point requires at least 3 points.

    :return: ``(frontier_items, efficient_index_into_frontier)``
    """
    if not candidates:
        return [], None

    ordered = sorted(candidates, key=lambda x: (x[cost_key], -x[score_key]))
    frontier: List[Dict[str, Any]] = []
    best_score = float("-inf")
    for item in ordered:
        if item[score_key] > best_score:
            frontier.append(item)
            best_score = item[score_key]

    efficient_idx: Optional[int] = None
    if len(frontier) >= 3:
        min_cost, max_cost = frontier[0][cost_key], frontier[-1][cost_key]
        min_score, max_score = frontier[0][score_key], frontier[-1][score_key]
        cost_range = (max_cost - min_cost) or 1.0
        score_range = (max_score - min_score) or 1.0
        best_gain = float("-inf")
        for i, item in enumerate(frontier):
            gain = (item[score_key] - min_score) / score_range - (item[cost_key] - min_cost) / cost_range
            if gain > best_gain:
                best_gain, efficient_idx = gain, i

    return frontier, efficient_idx


def annotate_frontier(
    candidates: List[Dict[str, Any]],
    frontier: List[Dict[str, Any]],
    efficient_idx: Optional[int],
    id_key: str = "id",
    cost_key: str = "cost",
    score_key: str = "score",
) -> None:
    """Tag each candidate in place with ``on_frontier``, ``is_efficient`` and ``dist``.

    ``dist`` is the vertical gap to the frontier: the best frontier score available at or
    below the candidate's cost, minus the candidate's own score. Frontier items get 0.
    """
    frontier_ids = {f[id_key] for f in frontier}
    efficient_id = frontier[efficient_idx][id_key] if efficient_idx is not None else None
    for item in candidates:
        item["on_frontier"] = item[id_key] in frontier_ids
        item["is_efficient"] = item[id_key] == efficient_id
        if item["on_frontier"] or not frontier:
            item["dist"] = 0.0
            continue
        cheaper = [f[score_key] for f in frontier if f[cost_key] <= item[cost_key]]
        item["dist"] = (max(cheaper) - item[score_key]) if cheaper else 1.0


def frontier_sort_key(item: Dict[str, Any]) -> Tuple[bool, float]:
    """Sort frontier items first by ascending cost, then dominated items by distance."""
    return (not item["on_frontier"], item["cost"] if item["on_frontier"] else item["dist"])
