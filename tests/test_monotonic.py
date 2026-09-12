from src.ecl_models.monotonic import constraints_for


def test_signs():
    feats = ["leverage", "liquidity_ratio", "realgdp_growth_lag_2Q", "unemp_lag_1Q"]
    mapping = {"leverage": 1, "liquidity_ratio": -1, "realgdp_growth": -1, "unemp": 1}
    assert constraints_for(feats, mapping) == [1, -1, -1, 1]
