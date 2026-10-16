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
