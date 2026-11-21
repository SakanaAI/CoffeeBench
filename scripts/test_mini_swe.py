"""$0 test of the mini-swe-agent action format (CoffeeTerminus parser_name="mini_swe").

    ~/.cache/cb_hbvenv/bin/python scripts/test_mini_swe.py   (needs harbor installed)
"""
import sys

sys.path.insert(0, ".")
import coffeebench.harbor_agent as H  # noqa: E402

p = H.MiniSweParser()
r = p.parse_response("THOUGHT: join first.\n\n```bash\ncoffee start\n```")
assert r.commands[0].keystrokes == "coffee start\n" and not r.is_task_complete and not r.error and r.analysis == "join first."
r = p.parse_response("THOUGHT: msg\n```bash\ncoffee send_message - <<'EOF'\n{\"recipient\": \"farmer_A\"}\nEOF\n```")
assert r.commands[0].keystrokes.startswith("coffee send_message - <<'EOF'\n") and r.commands[0].keystrokes.endswith("EOF\n")
r = p.parse_response("THOUGHT: two\n```bash\nls\n```\n```bash\npwd\n```")
assert r.error and not r.commands
r = p.parse_response("no block here")
assert r.error and not r.commands
r = p.parse_response("THOUGHT: done\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```")
assert r.is_task_complete and not r.commands
r = p.parse_response("Note: x\n```\ncoffee view_dashboard\n```")  # unlabelled fence accepted
assert r.analysis == "x"
assert r.commands[0].keystrokes == "coffee view_dashboard\n"
tpl = H.MINI_SWE_TEMPLATE.read_text()
out = tpl.format(instruction="RULES", terminal_state="$ ")
assert "RULES" in out and "```bash" in out and "ls -la" not in out and "THOUGHT" not in out
print("mini-swe parser/template: OK")
