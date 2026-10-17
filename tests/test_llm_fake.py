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
