import math

from openrouter_frontier.scoring import EndpointPricing, ScoringConfig, evaluate_endpoint


def test_published_hit_rate_is_used_as_observed():
    pricing = EndpointPricing(prompt=1.0, completion=0.0, input_cache_read=0.1)
    cfg = ScoringConfig(prompt_tokens=1_000_000, completion_tokens=0, price_failures=False)
    for h in (0.0, 0.3, 0.9, 1.0):
        sb = evaluate_endpoint(pricing, cache_hit_rate=h, config=cfg)
        assert sb.cache_hit_rate == h
        assert math.isclose(sb.token_cost_usd, h * 0.1 + (1 - h) * 1.0)
    # No observation => cold cache, never a made-up prior.
    assert evaluate_endpoint(pricing, cache_hit_rate=None, config=cfg).cache_hit_rate == 0.0


def test_pure_token_cost_no_cache():
    pricing = EndpointPricing(prompt=1.0, completion=2.0)  # $/M
    cfg = ScoringConfig(prompt_tokens=1_000_000, completion_tokens=500_000, price_failures=False)
    sb = evaluate_endpoint(pricing, cache_hit_rate=0.9, config=cfg)
    # No cache-read price => hit rate forced to 0 and every prompt token costs the input price.
    assert sb.cache_hit_rate == 0.0
    assert math.isclose(sb.token_cost_usd, 1.0 * 1.0 + 0.5 * 2.0)
    assert sb.time_cost_usd == 0.0 and sb.failure_cost_usd == 0.0
    assert math.isclose(sb.total_cost_usd, sb.token_cost_usd)


def test_cache_hit_lowers_prompt_cost():
    pricing = EndpointPricing(prompt=1.0, completion=0.0, input_cache_read=0.1)
    cfg = ScoringConfig(prompt_tokens=1_000_000, completion_tokens=0, price_failures=False)
    sb = evaluate_endpoint(pricing, cache_hit_rate=0.5, config=cfg)
    assert math.isclose(sb.cache_hit_rate, 0.5)
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
        price_failures=True,
    )
    sb = evaluate_endpoint(pricing, cache_hit_rate=1.0, ttft_seconds=2.0, throughput_tps=50.0, uptime_pct=90.0, config=cfg)
    # timeCost = $1/s * (2s ttft + 100/50 s generation) = $4
    assert math.isclose(sb.time_cost_usd, 4.0)
    # failureCost = 0.1 * (1M * 1.0 * (1.0 - 0.0) / 1e6 + $1/s * 2s) = 0.1 * 3 = 0.3
    assert math.isclose(sb.failure_cost_usd, 0.3)
    assert math.isclose(sb.total_cost_usd, sb.token_cost_usd + 4.0 + 0.3)


def test_from_api_dict_treats_api_prices_as_already_discounted():
    # Real payload shape: Z.ai GLM-5.3-Flash at 50% off. The API quotes the *net* price
    # (0.000000075/token = $0.075/M); list price is $0.15/M. The discount must not be
    # applied a second time.
    raw = {"prompt": "0.000000075", "completion": "0.00000025", "input_cache_read": "0.000000015", "discount": 0.5}
    p = EndpointPricing.from_api_dict(raw)
    assert math.isclose(p.prompt, 0.075)
    assert math.isclose(p.completion, 0.25)
    assert math.isclose(p.input_cache_read, 0.015)
    assert p.input_cache_write is None
    assert p.discount == 0.5

    listed = EndpointPricing.from_api_dict(raw, apply_discount=False)
    assert math.isclose(listed.prompt, 0.15)
    assert math.isclose(listed.completion, 0.50)
    assert math.isclose(listed.input_cache_read, 0.03)


def test_from_api_dict_no_discount_is_identity():
    raw = {"prompt": "0.00000015", "completion": "0.0000005", "input_cache_read": "0.00000003", "discount": 0}
    a = EndpointPricing.from_api_dict(raw)
    b = EndpointPricing.from_api_dict(raw, apply_discount=False)
    assert (a.prompt, a.completion, a.input_cache_read) == (b.prompt, b.completion, b.input_cache_read)
    assert math.isclose(a.prompt, 0.15)


def test_price_per_million_is_strict_per_token_conversion():
    from openrouter_frontier._util import price_per_million

    assert math.isclose(price_per_million("0.000000075"), 0.075)
    # A legitimately tiny per-token price must scale too, not be mistaken for per-million.
    assert math.isclose(price_per_million("0.000000000005"), 0.000005)
    assert price_per_million(None) is None
    assert price_per_million("") is None
    assert price_per_million("abc") is None
