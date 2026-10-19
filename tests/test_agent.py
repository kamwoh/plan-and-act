import asyncio

from armmind.agent.loop import Agent, AgentConfig
from armmind.llm.fake import FakeBackend
from armmind.llm.protocol import LLMError
from tests.fakes import DeafBackend, FakeEnv, HangingBackend, Recorder, bad, direct, good


async def test_success_ends_run():
    env, rec = FakeEnv(succeed_after=2), Recorder()
    backend = FakeBackend([good([{"type": "open"}, {"type": "close"}, {"type": "open"}])])
    status = await Agent(backend).run(env, direct, rec)
    assert status == "success"
    assert len(env.acts) == 2  # stops acting once the task is solved
    assert rec.kinds() == ["run_start", "observation", "ask", "reply", "action", "action", "step", "run_end"]
    assert [d for k, d in rec.events if k == "action"] == [{"step": 1, "index": 0, "of": 3},
                                                           {"step": 1, "index": 1, "of": 3}]
    step = dict(rec.events)["step"]
    assert step["success"] is True and len(step["results"]) == 2


async def test_images_go_to_the_model():
    backend = FakeBackend([good()])
    await Agent(backend, AgentConfig(max_steps=1)).run(FakeEnv(), direct, Recorder())
    msg, schema = backend.sent[0]
    assert msg.images == (b"front", b"top") and schema["type"] == "object"


async def test_one_bad_reply_then_retry_succeeds():
    env = FakeEnv()
    backend = FakeBackend([bad(), good()])
    await Agent(backend, AgentConfig(max_steps=1)).run(env, direct, Recorder())
    assert len(env.acts) == 1
    assert "not valid" in backend.sent[1][0].text


async def test_two_bad_replies_skip_the_step():
    env, rec = FakeEnv(), Recorder()
    backend = FakeBackend([bad(), bad(), good()])
    status = await Agent(backend, AgentConfig(max_steps=2)).run(env, direct, rec)
    assert "bad_reply" in rec.kinds()
    assert len(env.acts) == 1  # only step 2 acted
    assert status == "step_limit"


async def test_invalid_action_counts_as_bad_reply():
    env = FakeEnv()
    backend = FakeBackend([good([{"type": "move_to", "xyz": [1, 2]}]), good()])
    await Agent(backend, AgentConfig(max_steps=1)).run(env, direct, Recorder())
    assert len(env.acts) == 1 and "xyz" in backend.sent[1][0].text


async def test_auth_error_ends_run():
    rec = Recorder()
    backend = FakeBackend([[LLMError("auth", "please log in")]])
    status = await Agent(backend).run(FakeEnv(), direct, rec)
    assert status == "llm_error" and "llm_error" in rec.kinds()
    assert backend.closed == 1


async def test_notes_persist_when_omitted():
    backend = FakeBackend([good(notes="cube is left"), good(), good()])
    await Agent(backend, AgentConfig(max_steps=3)).run(FakeEnv(), direct, Recorder())
    assert "Your notes: cube is left" in backend.sent[2][0].text


async def test_stop_while_model_thinks():
    agent, rec = Agent(HangingBackend()), Recorder()
    task = asyncio.create_task(agent.run(FakeEnv(), direct, rec))
    await asyncio.sleep(0.05)
    await agent.stop()
    status = await asyncio.wait_for(task, 1.0)
    assert status == "stopped" and rec.kinds()[-1] == "run_end"


async def test_stop_wins_over_a_model_call_that_ignores_cancel():
    agent, rec = Agent(DeafBackend()), Recorder()
    task = asyncio.create_task(agent.run(FakeEnv(), direct, rec))
    await asyncio.sleep(0.05)
    await agent.stop()
    status = await asyncio.wait_for(task, 1.0)
    assert status == "stopped" and rec.kinds()[-1] == "run_end"


async def test_each_step_opens_and_closes_its_own_conversation():
    backend = FakeBackend([good(), good(), good()])
    await Agent(backend, AgentConfig(max_steps=3)).run(FakeEnv(), direct, Recorder())
    assert len(backend.systems) == 3 and backend.closed == 3


async def test_retry_uses_the_same_conversation():
    convs = []

    class Spy(FakeBackend):
        async def open(self, system, chain=None, resume=None):
            conv = await super().open(system)
            convs.append(conv)
            return conv

    backend = Spy([bad(), good()])
    await Agent(backend, AgentConfig(max_steps=1)).run(FakeEnv(), direct, Recorder())
    assert len(convs) == 1 and len(backend.sent) == 2 and backend.closed == 1


async def test_done_without_check_ends_run():
    env, rec = FakeEnv(has_check=False), Recorder()
    backend = FakeBackend([good(done=True), good()])
    status = await Agent(backend).run(env, direct, rec)
    assert status == "done" and len(env.acts) == 1  # the actions of the same reply still run
    events = dict(rec.events)
    assert events["step"]["done"] is True and events["run_end"]["status"] == "done"
    assert events["run_start"]["check"] is None


async def test_done_with_check_continues_and_tells_the_model():
    env, rec = FakeEnv(), Recorder()  # a check that never passes
    backend = FakeBackend([good(done=True), good()])
    status = await Agent(backend, AgentConfig(max_steps=2)).run(env, direct, rec)
    assert status == "step_limit"
    assert "The automatic check says the task is not finished yet." in backend.sent[1][0].text
    assert "not finished yet" not in backend.sent[0][0].text


async def test_run_start_names_the_check():
    rec = Recorder()
    await Agent(FakeBackend([good()]), AgentConfig(max_steps=1)).run(FakeEnv(), direct, rec)
    assert dict(rec.events)["run_start"]["check"] == "reach"


async def test_check_still_ends_with_success_when_done():
    env = FakeEnv(succeed_after=1)
    status = await Agent(FakeBackend([good(done=True)])).run(env, direct, Recorder())
    assert status == "success"


async def test_stop_during_actions_wins_over_done():
    agent = Agent(FakeBackend([good([{"type": "open"}, {"type": "close"}], done=True)]))
    env = FakeEnv(has_check=False)
    act = env.act
    env.act = lambda action: (agent._stop.set(), act(action))[1]  # Stop arrives during the first action
    status = await agent.run(env, direct, Recorder())
    assert status == "stopped" and len(env.acts) == 1


def usage(**kw):
    from armmind.llm.protocol import Usage
    return Usage("fake", kw.pop("latency_s", 1.0), kw.pop("in_tokens", 10), kw.pop("out_tokens", 5),
                 kw.pop("cost_usd", 0.01), **kw)


async def test_step_records_both_attempts_of_a_retry():
    rec = Recorder()
    first = [LLMError("schema", "answer did not match"), usage(in_tokens=100, out_tokens=7, cache_write_tokens=50,
                                                              cache_read_tokens=0, cost_usd=0.02, latency_s=2.0)]
    second = [*good()[:1], usage(in_tokens=3, out_tokens=4, cache_write_tokens=0, cache_read_tokens=40,
                                 cost_usd=0.005, latency_s=0.5)]
    backend = FakeBackend([first, second])
    await Agent(backend, AgentConfig(max_steps=1)).run(FakeEnv(), direct, rec)
    step = dict(rec.events)["step"]
    a1, a2 = step["attempts"]
    assert a1["images"] == 2 and a2["images"] == 0
    assert a1["error"] == "answer did not match" and a1["reply"] is None
    assert a2["error"] is None and '"plan": "p"' in a2["reply"]
    assert a1["message"] == step["prompt"] and "not valid" in a2["message"]
    assert a1["usage"] == {"in": 100, "cache_write": 50, "cache_read": 0, "out": 7, "cost_usd": 0.02,
                           "latency_s": 2.0}
    assert step["usage"] == {"in": 103, "cache_write": 50, "cache_read": 40, "out": 11, "cost_usd": 0.025,
                             "latency_s": 2.5}
    assert step["latency_s"] == 0.5 and step["cost_usd"] == 0.005  # kept as before: the last attempt


async def test_usage_is_null_when_no_attempt_has_usage():
    rec = Recorder()
    await Agent(FakeBackend([[*good()[:1]]]), AgentConfig(max_steps=1)).run(FakeEnv(), direct, rec)
    step = dict(rec.events)["step"]
    assert step["usage"] is None and step["attempts"][0]["usage"] is None


async def test_bad_reply_and_llm_error_carry_prompt_and_attempts():
    rec = Recorder()
    backend = FakeBackend([bad(), bad(), [LLMError("auth", "please log in")]])
    await Agent(backend, AgentConfig(max_steps=2)).run(FakeEnv(), direct, rec)
    events = dict(rec.events)
    for kind in ("bad_reply", "llm_error"):
        assert events[kind]["prompt"].startswith("Scene: ")
        assert "usage" in events[kind]
    assert len(events["bad_reply"]["attempts"]) == 2
    assert events["llm_error"]["attempts"][0]["error"] == "please log in"


async def test_run_start_has_the_system_prompt():
    from armmind.agent.prompts import SYSTEM
    rec = Recorder()
    await Agent(FakeBackend([good()]), AgentConfig(max_steps=1)).run(FakeEnv(), direct, rec)
    assert dict(rec.events)["run_start"]["system_prompt"] == SYSTEM


async def test_custom_system_prompt_reaches_the_backend():
    backend, rec = FakeBackend([good()]), Recorder()
    await Agent(backend, AgentConfig(max_steps=1), system="Be brief.").run(FakeEnv(), direct, rec)
    assert backend.systems == ["Be brief."] and dict(rec.events)["run_start"]["system_prompt"] == "Be brief."


async def test_empty_system_prompt_falls_back_to_the_default():
    from armmind.agent.prompts import SYSTEM
    backend = FakeBackend([good()])
    await Agent(backend, AgentConfig(max_steps=1), system="").run(FakeEnv(), direct, Recorder())
    assert backend.systems == [SYSTEM]


async def test_scene_text_reaches_the_model_and_the_log():
    backend, rec = FakeBackend([good()]), Recorder()
    await Agent(backend, AgentConfig(max_steps=1), system="Be brief.").run(FakeEnv(), direct, rec)
    assert backend.sent[0][0].text.startswith(f"Scene: {FakeEnv.scene_text}\n")
    assert dict(rec.events)["run_start"]["scene"] == "cubes"


async def test_image_text_of_the_env_reaches_the_model():
    backend, env = FakeBackend([good()]), FakeEnv()
    env.image_text = "Image 1: side camera. Image 2: top camera."
    await Agent(backend, AgentConfig(max_steps=1)).run(env, direct, Recorder())
    assert backend.sent[0][0].text.endswith("\nImage 1: side camera. Image 2: top camera.")


async def test_lessons_reach_the_next_step_and_the_step_event():
    backend, rec = FakeBackend([good(lessons=["Closing high slipped."]), good(lessons=["Closing high slipped."]),
                                good()]), Recorder()
    await Agent(backend, AgentConfig(max_steps=3, history_steps=1)).run(FakeEnv(), direct, rec)
    assert "Lessons: (none)" in backend.sent[0][0].text
    assert "Lessons:\n- Closing high slipped.\n" in backend.sent[1][0].text
    assert backend.sent[2][0].text.count("Closing high slipped.") == 1
    steps = [d for k, d in rec.events if k == "step"]
    assert steps[0]["lessons"] == ["Closing high slipped."] and steps[2]["lessons"] == []


async def test_run_lessons_are_not_limited_by_history_steps():
    replies = [good(lessons=[f"lesson {i}"]) for i in range(4)]
    backend = FakeBackend(replies)
    await Agent(backend, AgentConfig(max_steps=4, history_steps=1)).run(FakeEnv(), direct, Recorder())
    assert all(f"- lesson {i}" in backend.sent[3][0].text for i in range(3))


async def test_ask_comes_before_each_attempt_and_before_the_step():
    rec = Recorder()
    await Agent(FakeBackend([bad(), good()]), AgentConfig(max_steps=1)).run(FakeEnv(), direct, rec)
    assert rec.kinds() == ["run_start", "observation", "ask", "ask", "reply", "action", "step", "run_end"]
    asks = [d for k, d in rec.events if k == "ask"]
    step = dict(rec.events)["step"]
    assert [a["attempt"] for a in asks] == [1, 2] and all(a["step"] == 1 for a in asks)
    assert asks[0]["prompt"] == step["prompt"] and "not valid" in asks[1]["prompt"]
    assert set(asks[0]) == {"step", "prompt", "attempt"}


async def test_reply_event_carries_what_the_chat_shows():
    rec = Recorder()
    backend = FakeBackend([good(notes="n1", lessons=["l1"])])
    await Agent(backend, AgentConfig(max_steps=1)).run(FakeEnv(), direct, rec)
    reply = dict(rec.events)["reply"]
    assert reply["observation"] == "o" and reply["notes"] == "n1" and reply["todo"] == ["t"]
    assert reply["lessons"] == ["l1"] and reply["usage"]["latency_s"] == 0.1
