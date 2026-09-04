import math

import pytest

from openrouter_frontier.scoring import (
    EndpointInputs, EndpointPricing, ScoringConfig, evaluate_endpoint, impute_missing,
)

GLM = EndpointPricing(prompt=0.075, completion=0.25, input_cache_read=0.015)  # $/M, no write price


def closed_form_order(h, u, a=2000, o=500, N=200, m=0.075e-6, r=0.015e-6, w=0.075e-6, c=0.25e-6):
    """Proposition 1 of docs/task_cost_model.tex with B = A cold and v = 0."""
    d = a + o; q = 1 - u
    pi = h * r + (1 - h) * m
    alpha = w * a + c * o
    S = d * N * (N - 1) / 2
    return N * ((1 - q) * alpha + q * alpha) + ((1 - q) * pi + q * w) * S + q * h * (m - r) * d * (N - 1)


def closed_form_sticky(h, u, a=2000, o=500, N=200, m=0.075e-6, r=0.015e-6, w=0.075e-6, c=0.25e-6):
    """Section 7 (sticky) with B = A."""
    d = a + o; q = 1 - u; x = 1 - q
    pi = h * r + (1 - h) * m
    alpha = w * a + c * o
    S = d * N * (N - 1) / 2
    G0 = (1 - x ** N) / (1 - x)
    G1 = x * (1 - N * x ** (N - 1) + (N - 1) * x ** N) / (1 - x) ** 2
    return (x * alpha + q * alpha - alpha) * G0 + (x * pi + q * w - pi) * d * G1 + N * alpha + pi * S


@pytest.mark.parametrize("h,u", [(0.885, 0.989), (0.469, 0.995), (0.0, 0.9), (1.0, 1.0)])
def test_matches_paper_closed_form_order(h, u):
    cfg = ScoringConfig(new_tokens_per_turn=2000, completion_tokens=500, turns=200, time_value_usd_per_hour=0, routing="order")
    sb = evaluate_endpoint(GLM, EndpointInputs(cache_hit_rate=h, uptime_pct=u * 100), cfg)
    assert math.isclose(sb.task_cost_usd, closed_form_order(h, u), rel_tol=1e-9)
    parts = (sb.fixed_cost_usd + sb.time_cost_usd + sb.read_baseline_usd + sb.miss_premium_usd
             + sb.failure_premium_usd + sb.return_penalty_usd)
    assert math.isclose(parts, sb.task_cost_usd, rel_tol=1e-9)


@pytest.mark.parametrize("h,u", [(0.885, 0.989), (0.469, 0.995)])
def test_matches_paper_closed_form_sticky(h, u):
    cfg = ScoringConfig(new_tokens_per_turn=2000, completion_tokens=500, turns=200, time_value_usd_per_hour=0, routing="sticky")
    sb = evaluate_endpoint(GLM, EndpointInputs(cache_hit_rate=h, uptime_pct=u * 100), cfg)
    assert math.isclose(sb.task_cost_usd, closed_form_sticky(h, u), rel_tol=1e-9)
    assert math.isclose(sb.migration_probability, 1 - u ** 200)
    assert sb.return_penalty_usd == 0.0


def test_worked_example_numbers():
    cfg = ScoringConfig(new_tokens_per_turn=2000, completion_tokens=500, turns=200, time_value_usd_per_hour=0, routing="order")
    z = evaluate_endpoint(GLM, EndpointInputs(cache_hit_rate=0.885, uptime_pct=98.9), cfg)
    d = evaluate_endpoint(GLM, EndpointInputs(cache_hit_rate=0.469, uptime_pct=99.5), cfg)
    assert round(z.task_cost_usd, 2) == 1.17 and round(d.task_cost_usd, 2) == 2.39
    assert round(z.sigma_proc_usd, 3) == 0.081 and round(d.sigma_proc_usd, 3) == 0.122


def test_quadratic_in_turns_and_linear_term_only_at_one_turn():
    cfg1 = ScoringConfig(turns=1, completion_tokens=500, time_value_usd_per_hour=0, routing="order", price_failures=False)
    sb1 = evaluate_endpoint(GLM, EndpointInputs(cache_hit_rate=0.5, uptime_pct=100), cfg1)
    # one turn: no prefix, so only new tokens (at write=input price) and output
    assert math.isclose(sb1.task_cost_usd, (2000 * 0.075 + 500 * 0.25) / 1e6)
    assert sb1.read_baseline_usd == 0 and sb1.miss_premium_usd == 0
    # doubling N roughly quadruples the prefix terms
    c100 = evaluate_endpoint(GLM, EndpointInputs(cache_hit_rate=0.5, uptime_pct=100), ScoringConfig(turns=100, completion_tokens=500, time_value_usd_per_hour=0, price_failures=False))
    c200 = evaluate_endpoint(GLM, EndpointInputs(cache_hit_rate=0.5, uptime_pct=100), ScoringConfig(turns=200, completion_tokens=500, time_value_usd_per_hour=0, price_failures=False))
    ratio = (c200.read_baseline_usd + c200.miss_premium_usd) / (c100.read_baseline_usd + c100.miss_premium_usd)
    assert math.isclose(ratio, 199 * 200 / (99 * 100))


def test_new_tokens_never_cached():
    """A hit rate of 1 still pays the write price on the a new tokens each turn."""
    cfg = ScoringConfig(turns=10, completion_tokens=500, time_value_usd_per_hour=0, price_failures=False)
    sb = evaluate_endpoint(GLM, EndpointInputs(cache_hit_rate=1.0, uptime_pct=100), cfg)
    assert math.isclose(sb.fixed_cost_usd, 10 * (2000 * 0.075 + 500 * 0.25) / 1e6)
    assert sb.miss_premium_usd == 0.0
    assert math.isclose(sb.perfect_cache_cost_usd, sb.task_cost_usd)


def test_three_prices_and_miss_policy():
    pr = EndpointPricing(prompt=1.0, completion=0.0, input_cache_read=0.1, input_cache_write=1.25)
    inp = EndpointInputs(cache_hit_rate=0.0, uptime_pct=100)
    rewrite = evaluate_endpoint(pr, inp, ScoringConfig(turns=2, completion_tokens=500, time_value_usd_per_hour=0, price_failures=False, miss_policy="rewrite"))
    process = evaluate_endpoint(pr, inp, ScoringConfig(turns=2, completion_tokens=500, time_value_usd_per_hour=0, price_failures=False, miss_policy="process"))
    assert rewrite.miss_price == 1.25 and process.miss_price == 1.0
    assert rewrite.write_price == 1.25 and rewrite.read_price == 0.1 and rewrite.input_price == 1.0
    # turn 2 prefix of 2500 tokens, all missed: 1.25 vs 1.0 $/M
    assert math.isclose(rewrite.miss_premium_usd + rewrite.read_baseline_usd, 2500 * 1.25 / 1e6)
    assert math.isclose(process.miss_premium_usd + process.read_baseline_usd, 2500 * 1.0 / 1e6)


def test_no_cache_endpoint_is_cold_at_input_price():
    pr = EndpointPricing(prompt=1.0, completion=2.0)
    sb = evaluate_endpoint(pr, EndpointInputs(cache_hit_rate=0.9, uptime_pct=100), ScoringConfig(turns=3, time_value_usd_per_hour=0, price_failures=False))
    assert sb.cache_hit_rate == 0.0 and sb.read_price == 1.0 and sb.miss_price == 1.0


def test_cache_modes():
    inp = EndpointInputs(cache_hit_rate=0.8, uptime_pct=100)
    agg = evaluate_endpoint(GLM, inp, ScoringConfig(turns=5, cache_mode="aggregate"))
    cold = evaluate_endpoint(GLM, inp, ScoringConfig(turns=5, cache_mode="cold"))
    asm = evaluate_endpoint(GLM, inp, ScoringConfig(turns=5, cache_mode="assumed", assumed_hit_rate=0.3))
    assert (agg.cache_hit_rate, cold.cache_hit_rate, asm.cache_hit_rate) == (0.8, 0.0, 0.3)
    assert cold.task_cost_usd > asm.task_cost_usd > agg.task_cost_usd


def test_cache_miss_costs_prefill_time():
    # $3600/hr => $1/s. Decode 50 tps, prefill 100x => 5000 tok/s. h = 0: every turn re-prefills its prefix.
    pr = EndpointPricing(prompt=0.0, completion=0.0, input_cache_read=0.0)
    inp = EndpointInputs(cache_hit_rate=0.0, uptime_pct=100, ttft_p50=0.0, tps_p50=50.0)
    cfg = ScoringConfig(new_tokens_per_turn=5000, completion_tokens=0, turns=3, time_value_usd_per_hour=3600.0,
                        price_failures=False, prefill_multiplier=100.0)
    sb = evaluate_endpoint(pr, inp, cfg)
    # new tokens 5000 each turn (1 s) + prefixes 0, 5000, 10000 tokens (0 + 1 + 2 s) => 6 s
    assert math.isclose(sb.time_cost_usd, 6.0) and math.isclose(sb.prefill_cost_usd, 6.0)
    warm = evaluate_endpoint(pr, EndpointInputs(cache_hit_rate=1.0, uptime_pct=100, ttft_p50=0.0, tps_p50=50.0), cfg)
    assert math.isclose(warm.time_cost_usd, 3.0)  # only the new tokens
    assert sb.prefill_tps == 5000.0
    # the published TTFT is not charged
    slow = evaluate_endpoint(pr, EndpointInputs(cache_hit_rate=1.0, uptime_pct=100, ttft_p50=30.0, tps_p50=50.0), cfg)
    assert math.isclose(slow.time_cost_usd, 3.0) and slow.ttft_seconds == 30.0


def test_overhead_seconds_charged_per_request():
    pr = EndpointPricing(prompt=0.0, completion=0.0, input_cache_read=0.0)
    inp = EndpointInputs(cache_hit_rate=1.0, uptime_pct=100)  # no throughput => no prefill or decode time
    cfg = ScoringConfig(new_tokens_per_turn=1, completion_tokens=0, turns=4, time_value_usd_per_hour=3600.0,
                        price_failures=False, overhead_seconds=0.5)
    sb = evaluate_endpoint(pr, inp, cfg)
    assert math.isclose(sb.ttft_cost_usd, 2.0) and math.isclose(sb.time_cost_usd, 2.0)


def test_time_cost_uses_lognormal_means_and_seconds_per_token():
    # p90 twice the median => sigma = ln2/1.2816, mean = median*exp(sigma^2/2)
    sigma = math.log(2) / 1.2815515655446004
    inp = EndpointInputs(cache_hit_rate=1.0, uptime_pct=100, ttft_p50=1.0, ttft_p90=2.0, tps_p50=50.0, tps_p90=100.0)
    assert math.isclose(inp.expected_ttft, math.exp(sigma ** 2 / 2))
    assert math.isclose(inp.expected_seconds_per_token, (1 / 50) * math.exp(sigma ** 2 / 2))
    cfg = ScoringConfig(turns=4, completion_tokens=100, new_tokens_per_turn=0, time_value_usd_per_hour=3600.0, price_failures=False)
    sb = evaluate_endpoint(EndpointPricing(prompt=0.0, completion=0.0, input_cache_read=0.0), inp, cfg)
    # no new tokens and h = 1 => no prefill; decode only, at the lognormal mean seconds/token
    assert math.isclose(sb.decode_cost_usd, 4 * 100 * inp.expected_seconds_per_token)
    assert math.isclose(sb.time_cost_usd, sb.decode_cost_usd)


def test_failure_cold_retry_and_return_penalty():
    # u = 90%, B = A cold, $1/s, 0.5 s overhead per request, no throughput published (=> no prefill/decode time).
    pr = EndpointPricing(prompt=1.0, completion=0.0, input_cache_read=0.0)
    inp = EndpointInputs(cache_hit_rate=1.0, uptime_pct=90.0)
    cfg = ScoringConfig(new_tokens_per_turn=1_000_000, completion_tokens=0, turns=2, time_value_usd_per_hour=3600.0,
                        routing="order", overhead_seconds=0.5)
    sb = evaluate_endpoint(pr, inp, cfg)
    q = 0.1
    # turn 1: S=0. ok: $1.0 new + 0.5 s; fail: wasted 0.5 s + cold $1.0 + 0.5 s.
    # turn 2: S=1M. ok: $1.0 new + $0 read + 0.5 s; fail: 0.5 s + ($1.0 + $1.0 cold prefix) + 0.5 s; no return penalty on last turn
    expect = ((1 - q) * (1.0 + 0.5) + q * (0.5 + 1.0 + 0.5)) + ((1 - q) * (1.0 + 0.5) + q * (0.5 + 2.0 + 0.5))
    expect += q * 1.0  # return penalty at turn 1 only: h (m - r) d = 1 * 1e-6 * 1e6 = $1.0 (no prefill term without throughput)
    assert math.isclose(sb.task_cost_usd, expect)
    assert math.isclose(sb.return_penalty_usd, q * 1.0)
    assert math.isclose(sb.ttft_cost_usd, (1 - q) * 0.5 * 2 + q * 1.0 * 2)


def test_impute_missing_takes_worst_observed():
    a = EndpointInputs(cache_hit_rate=0.5, uptime_pct=99.0, ttft_p50=1.0, ttft_p90=2.0, tps_p50=50.0, tps_p90=80.0)
    b = EndpointInputs(cache_hit_rate=0.5, uptime_pct=95.0, ttft_p50=3.0, ttft_p90=6.0, tps_p50=20.0, tps_p90=30.0)
    c = EndpointInputs()  # nothing published
    impute_missing([a, b, c])
    assert (c.ttft_p50, c.ttft_p90, c.tps_p50, c.tps_p90, c.uptime_pct, c.cache_hit_rate) == (3.0, 6.0, 20.0, 30.0, 95.0, 0.0)
    assert set(c.imputed) == {"latency", "throughput", "uptime", "cache_hit_rate"}
    assert a.imputed == []


def test_parameter_sigma_scales_as_n_squared():
    inp = EndpointInputs(cache_hit_rate=0.8, uptime_pct=100)
    s1 = evaluate_endpoint(GLM, inp, ScoringConfig(turns=100, completion_tokens=500, sigma_h=0.03, time_value_usd_per_hour=0, price_failures=False)).sigma_par_usd
    s2 = evaluate_endpoint(GLM, inp, ScoringConfig(turns=200, completion_tokens=500, sigma_h=0.03, time_value_usd_per_hour=0, price_failures=False)).sigma_par_usd
    assert math.isclose(s2 / s1, 199 * 200 / (99 * 100))
    # sigma_par = sigma_h (m-r) sum S_k
    assert math.isclose(s1, 0.03 * (0.075 - 0.015) / 1e6 * 2500 * 100 * 99 / 2)


def test_objective_adds_risk_terms():
    inp = EndpointInputs(cache_hit_rate=0.8, uptime_pct=99.0)
    cfg = ScoringConfig(turns=50, sigma_h=0.05, lambda_proc=1.0, lambda_par=2.0, time_value_usd_per_hour=0)
    sb = evaluate_endpoint(GLM, inp, cfg)
    assert math.isclose(sb.objective_usd, sb.task_cost_usd + sb.sigma_proc_usd + 2 * sb.sigma_par_usd)


def test_config_defaults_and_validation():
    cfg = ScoringConfig()
    assert cfg.time_value_usd_per_hour == 20.0 and cfg.task_tokens == 300_000 and cfg.output_tokens == 10_000
    # N = (300k - 10k) / 2000 = 145 turns; o = 10k / 145 = 69 per turn
    assert cfg.n_turns == 145 and cfg.completion_per_turn == 69 and cfg.routing == "sticky"
    assert ScoringConfig(completion_tokens=500).n_turns == 120
    assert ScoringConfig(turns=50).completion_per_turn == 200
    with pytest.raises(ValueError):
        ScoringConfig(new_tokens_per_turn=-1)
    with pytest.raises(ValueError):
        ScoringConfig(routing="random")
    with pytest.raises(ValueError):
        ScoringConfig(assumed_hit_rate=1.5)


def test_from_api_dict_treats_api_prices_as_already_discounted():
    raw = {"prompt": "0.000000075", "completion": "0.00000025", "input_cache_read": "0.000000015", "discount": 0.5}
    p = EndpointPricing.from_api_dict(raw)
    assert math.isclose(p.prompt, 0.075) and math.isclose(p.completion, 0.25) and math.isclose(p.input_cache_read, 0.015)
    listed = EndpointPricing.from_api_dict(raw, apply_discount=False)
    assert math.isclose(listed.prompt, 0.15) and math.isclose(listed.input_cache_read, 0.03)


def test_price_per_million_is_strict_per_token_conversion():
    from openrouter_frontier._util import price_per_million
    assert math.isclose(price_per_million("0.000000075"), 0.075)
    assert math.isclose(price_per_million("0.000000000005"), 0.000005)
    assert price_per_million(None) is None and price_per_million("abc") is None
