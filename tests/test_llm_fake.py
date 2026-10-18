from armmind.llm.fake import FakeBackend
from armmind.llm.protocol import Final, LLMError, Message


async def test_fake_backend_replays_and_records():
    b = FakeBackend([[Final("hi", {"a": 1})]])
    conv = await b.open("system text")
    events = [ev async for ev in conv.send(Message("user", "q"), {"type": "object"})]
    assert events == [Final("hi", {"a": 1})]
    assert b.systems == ["system text"] and b.sent[0][0].text == "q"
    events = [ev async for ev in conv.send(Message("user", "q2"))]
    assert isinstance(events[0], LLMError)


async def test_demo_backend_replies_by_step_number():
    from armmind.agent.parse import parse_reply
    from armmind.llm.fake import DEMO_REPLIES, DemoBackend
    from armmind.llm.protocol import Final, Message
    conv = await DemoBackend(delay_s=0).open("s")
    for n in (2, 1, 2):
        events = [e async for e in conv.send(Message("user", f"Task: t\nStep {n} of 5."))]
        final = next(e for e in events if isinstance(e, Final))
        assert final.parsed == DEMO_REPLIES[n - 1]
        parse_reply(final.parsed)


async def test_demo_backend_plans_low_then_replans_with_a_lesson():
    from armmind.agent.parse import parse_plan
    from armmind.agent.prompts import PLAN_REPLY_SCHEMA
    from armmind.llm.fake import DEMO_PLAN, DEMO_PLAN_LOW, DemoBackend
    conv = await DemoBackend(delay_s=0).open("s")

    async def ask(text):
        events = [e async for e in conv.send(Message("user", text), PLAN_REPLY_SCHEMA)]
        return next(e for e in events if isinstance(e, Final)).parsed

    first = await ask("Task: t\nPlan the whole task now, in one reply.")
    assert first == DEMO_PLAN_LOW and not parse_plan(first).lessons
    assert first["keyframes"][-1]["xyz"][2] < 0.1  # the lift is too low for the pick check
    second = await ask("Task: t\nThis is a re-plan, round 2. Your last plan ran once in the physics.")
    assert {k: v for k, v in second.items() if k != "lessons"} == DEMO_PLAN
    assert len(parse_plan(second).lessons) == 1
