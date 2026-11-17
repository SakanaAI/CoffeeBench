"""$0 test: an empty LLM response (no content, 0 tokens) is retried, not taken as a turn.

    ~/.cache/cb_hbvenv/bin/python scripts/test_empty_llm_response.py   (needs harbor installed)
"""
import asyncio
import sys
import types

sys.path.insert(0, ".")
import coffeebench.harbor_agent as H  # noqa: E402
from harbor.llms.base import LLMResponse  # noqa: E402

class U:  # usage stub
    def __init__(s, p, c): s.prompt_tokens, s.completion_tokens = p, c
assert H._is_empty_response(LLMResponse(content="", usage=U(0, 0)))
assert H._is_empty_response(LLMResponse(content="  ", usage=None))
assert not H._is_empty_response(LLMResponse(content="", usage=U(100, 5)))
assert not H._is_empty_response(LLMResponse(content='{"a":1}', usage=U(0, 0)))
assert H._is_transient_llm_error(H.EmptyLLMResponseError("x"))
# retry loop: two empty responses then a real one; chat history must hold only the final exchange
class Chat:
    def __init__(s):
        s._m = [{"role": "user", "content": "sys"}]
        s.resets = 0
    @property
    def messages(s): return s._m
    def reset_response_chain(s): s.resets+=1
calls={"n":0}
async def fake_super(self, chat, prompt, oi, session):
    calls["n"]+=1
    content = "" if calls["n"] < 3 else '{"analysis":"ok"}'
    chat._m += [{"role":"user","content":prompt},{"role":"assistant","content":content}]
    return LLMResponse(content=content, usage=U(0 if not content else 10, 0 if not content else 3))
H.Terminus2._query_llm = fake_super
agent = H.CoffeeTerminus.__new__(H.CoffeeTerminus)
agent.logger = types.SimpleNamespace(warning=lambda *a, **k: None)
agent._calibrate_tokens = lambda *a, **k: None
orig_sleep = asyncio.sleep
H.asyncio.sleep = lambda d: orig_sleep(0)
chat = Chat()
r = asyncio.run(agent._query_llm(chat, "obs"))
assert r.content and calls["n"] == 3, calls
assert [m["content"] for m in chat.messages] == ["sys", "obs", '{"analysis":"ok"}'], chat.messages
assert chat.resets == 2
print("empty-response retry: OK")
