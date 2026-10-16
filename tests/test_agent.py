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
    assert rec.kinds() == ["run_start", "observation", "reply", "step", "run_end"]
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
        async def open(self, system):
            conv = await super().open(system)
            convs.append(conv)
            return conv

    backend = Spy([bad(), good()])
    await Agent(backend, AgentConfig(max_steps=1)).run(FakeEnv(), direct, Recorder())
    assert len(convs) == 1 and len(backend.sent) == 2 and backend.closed == 1
