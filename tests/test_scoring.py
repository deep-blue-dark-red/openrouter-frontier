import math

from openrouter_frontier.scoring import EndpointPricing, ScoringConfig, evaluate_endpoint, shrink_hit_rate


def test_shrink_hit_rate_limits():
    assert shrink_hit_rate(0.9, total_tokens=0, prior=0.5, prior_weight_tokens=1e9) == 0.5
    # With T == W the result is the midpoint between observation and prior.
    assert math.isclose(shrink_hit_rate(0.9, 1e9, 0.5, 1e9), 0.7)
    # Huge traffic: observation dominates.
    assert math.isclose(shrink_hit_rate(0.9, 1e15, 0.5, 1e9), 0.9, rel_tol=1e-5)


def test_pure_token_cost_no_cache():
    pricing = EndpointPricing(prompt=1.0, completion=2.0)  # $/M
    cfg = ScoringConfig(prompt_tokens=1_000_000, completion_tokens=500_000, price_failures=False)
    sb = evaluate_endpoint(pricing, cache_hit_rate=0.9, total_tokens=10**12, config=cfg)
    # No cache-read price => hit rate forced to 0 and every prompt token costs the input price.
    assert sb.h_used == 0.0
    assert math.isclose(sb.token_cost_usd, 1.0 * 1.0 + 0.5 * 2.0)
    assert sb.time_cost_usd == 0.0 and sb.failure_cost_usd == 0.0
    assert math.isclose(sb.total_cost_usd, sb.token_cost_usd)


def test_cache_hit_lowers_prompt_cost():
    pricing = EndpointPricing(prompt=1.0, completion=0.0, input_cache_read=0.1)
    cfg = ScoringConfig(prompt_tokens=1_000_000, completion_tokens=0, price_failures=False, prior_weight_tokens=0)
    sb = evaluate_endpoint(pricing, cache_hit_rate=0.5, total_tokens=1, config=cfg)
    assert math.isclose(sb.h_used, 0.5)
    # 50% at $0.1/M, 50% at $1.0/M over 1M tokens => $0.55
    assert math.isclose(sb.token_cost_usd, 0.55)
    assert sb.hit_price == 0.1 and sb.miss_price == 1.0


def test_cache_write_price_is_miss_price():
    pricing = EndpointPricing(prompt=1.0, completion=0.0, input_cache_read=0.1, input_cache_write=1.25)
    sb = evaluate_endpoint(pricing, cache_hit_rate=0.0, config=ScoringConfig(price_failures=False))
    assert sb.miss_price == 1.25


def test_time_and_failure_costs():
    pricing = EndpointPricing(prompt=1.0, completion=1.0, input_cache_read=0.0)
    cfg = ScoringConfig(
        prompt_tokens=1_000_000, completion_tokens=100, time_value_usd_per_hour=3600.0,  # $1/s
        price_failures=True, prior_weight_tokens=0,
    )
    sb = evaluate_endpoint(pricing, cache_hit_rate=1.0, total_tokens=1, ttft_seconds=2.0, throughput_tps=50.0, uptime_pct=90.0, config=cfg)
    # timeCost = $1/s * (2s ttft + 100/50 s generation) = $4
    assert math.isclose(sb.time_cost_usd, 4.0)
    # failureCost = 0.1 * (1M * 1.0 * (1.0 - 0.0) / 1e6 + $1/s * 2s) = 0.1 * 3 = 0.3
    assert math.isclose(sb.failure_cost_usd, 0.3)
    assert math.isclose(sb.total_cost_usd, sb.token_cost_usd + 4.0 + 0.3)


def test_from_api_dict_scales_per_token_prices_and_applies_discount():
    p = EndpointPricing.from_api_dict(
        {"prompt": "0.000001", "completion": "0.000002", "input_cache_read": "0.0000001", "discount": 0.5}
    )
    assert math.isclose(p.prompt, 0.5)
    assert math.isclose(p.completion, 1.0)
    assert math.isclose(p.input_cache_read, 0.05)
    assert p.input_cache_write is None
    undiscounted = EndpointPricing.from_api_dict({"prompt": "0.000001", "completion": "0.000002", "discount": 0.5}, apply_discount=False)
    assert math.isclose(undiscounted.prompt, 1.0)
