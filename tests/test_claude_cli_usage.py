from armmind.llm.claude_cli import usage_from_result


def test_usage_keeps_all_token_fields():
    ev = {"usage": {"input_tokens": 5, "output_tokens": 7, "cache_creation_input_tokens": 100,
                    "cache_read_input_tokens": 200}, "total_cost_usd": 0.03, "modelUsage": {"m1": {}}}
    u = usage_from_result(ev, "claude", 1.5)
    assert (u.model, u.latency_s, u.in_tokens, u.out_tokens) == ("m1", 1.5, 5, 7)
    assert (u.cache_write_tokens, u.cache_read_tokens, u.cost_usd) == (100, 200, 0.03)


def test_usage_fields_are_none_when_missing():
    u = usage_from_result({}, "claude", 0.1)
    assert u.model == "claude" and u.in_tokens is None and u.cache_write_tokens is None and u.cache_read_tokens is None
