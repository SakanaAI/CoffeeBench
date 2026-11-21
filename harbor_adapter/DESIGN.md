# CoffeeBench on Harbor — design (2026-10-17)

All 6 agents run as Harbor trials (one per firm), driving a `coffee` CLI against one shared
CoffeeBench env server per simulation. Harbor refs are to `~/programming/harbor/src/harbor` (v0.24.0).

## 0. CoffeeTerminus (terminus-2 subclass, loaded via `import_path`)
Stock terminus-2 sends every command non-blocking with duration capped at 60 s
(`agents/terminus_2/terminus_2.py:1310,1340`), so waiting at the turn barrier costs one LLM call per poll.
`coffeebench/harbor_agent.py:CoffeeTerminus(Terminus2)` overrides:
- `_execute_commands`: keystrokes starting with `coffee` → `block=True, max_timeout_sec=7200`
  (tmux_session's `tmux wait` mode, `tmux_session.py:612-637`). Other commands unchanged.
- `_limit_output_length`: 10 KB → ~100 KB (view_offers observations reach ~51 KB).
- per episode: `tmux clear-history` + reset `_previous_buffer` (avoid capturing 10M-line scrollback).
- `_handle_llm_interaction`: reject `task_complete` while the sim is running (`WARNINGS:` feedback).
- telemetry: POST {agent, cost_so_far, tokens, analysis} to the server after each LLM call.
- prompts (2026-10-18): own template `coffeebench/harbor_templates/coffee-terminus-json.txt`
  (`_get_prompt_template_path`; JSON response spec verbatim, business-session framing, `coffee`
  blocks / duration ignored, task_complete only after SIMULATION OVER); completion confirmation and
  `_summarize` (copied from v0.24.0, prompt texts only) ask for the business state.
  instruction.md = rules-only prompt (`coffeebench/rules_prompt.py`, §1-9) + CLI guide (§10) +
  complete tool list (§11, same text as `coffee help`).

## 1. Topology
- One server per simulation, a host process: `python -m coffeebench.server --config X --seed N --port P
  --run-token T [--resume ckpt]` (starlette/uvicorn in the same asyncio loop as `env.run()`).
- 6 tasks (one per agent id); 2 concurrent `harbor run` jobs — focal (focal model × focal task) and
  background (Haiku 5.5 × 5 tasks); Harbor crosses agents × tasks (`job_plan.py:141-170`).
  `n_concurrent_trials` ≥ trials, else the registration barrier deadlocks.
- Networking (hardened 2026-10-17, after cluster pilots where a focal agent read the CLI, ran `find /`,
  `env`, `ss`, and probed /openapi.json etc.): task containers have NO network (compose
  `network_mode: none` on `main`; the public repo with the env source is unreachable). The server
  listens on host TCP 127.0.0.1:P (launcher, CoffeeTerminus — host processes) AND on a Unix socket
  `<CB_SOCK_ROOT=/tmp>/cb_XXXX/coffee.sock` (fresh per segment; run dirs are too long for the ~104-byte
  AF_UNIX limit; dir + socket 0777, tokens authenticate), bind-mounted at /run/coffee via
  `${COFFEE_SOCK_DIR}` (Harbor runs compose with the host env). CLI + verifier use
  `COFFEE_SOCKET=/run/coffee/coffee.sock` (stdlib http.client over AF_UNIX). One uvicorn server, two
  listeners (`serve(sockets=[tcp, uds])`), `server_header=False, date_header=False`.
  Harbor's `[environment] network_mode = "no-network"` is NOT used: on the docker backend it starts the
  egress-control sidecar (NET_ADMIN + nftables) — not expected to work under rootless podman.
  Local Docker Desktop (24.0.5) cannot connect to host Unix sockets through bind mounts (ECONNREFUSED,
  tested) → `--network relay`: a relay container serves the socket in a docker volume and forwards to
  host TCP; task containers are still `network_mode: none`. `--network bridge` = old TCP fallback.
- Per-run params via task.toml `${VAR}` substitution: `COFFEE_TOKEN = ${COFFEE_TOKEN_<AID>}` (static
  per task), static `COFFEE_AGENT_ID`, `COFFEE_SOCKET` (bridge: `COFFEE_URL`).
- Auth: scoped tokens, all HMAC-SHA256 of the run token: agent `HMAC(run, aid)` (container
  COFFEE_TOKEN), harness `HMAC(run, "harness:"+aid)` (`COFFEE_HARNESS_TOKEN_<AID>`, Harbor host env
  only → CoffeeTerminus telemetry / status / resume_info / resume_ack), verifier
  `HMAC(run, "verifier:"+aid)` (`[verifier.env]` only → own `/audit`), admin = run token (launcher).
  Strict allowlist: agent tokens reach only /register /act /wait /tools /workspace and a minimal own
  /status; everything else (/, /docs, /openapi.json, /health, /audit, /admin/*, wrong method, unknown
  token, out-of-scope endpoint) → uniform `404 {"error":"not found"}`. A client-sent `agent` ≠ token's
  → 403. Bodies capped (256 KB; workspace 43 MB) → 413, non-JSON-object → 400; non-sim endpoints
  rate-limited (40/s per principal → 429). Instruction §10 rule 10 (prompts.CLI_GUIDE): market only
  via `coffee`; inspecting the harness is not allowed (no network in the container).
- Audit: `harbor_adapter/audit_terminal.py` classifies every shell command per agent into
  harness_inspection / server_direct / internet / benign_workspace / other_shell (`--csv`, `--json`).

## 2. Turn semantics
`env.run()` is unchanged; `coffeebench/server/remote_agent.py:RemoteAgent` replaces `Agent`:
`add_message` buffers the wake observation; `step_query_async` flushes (previous result + observation) to
the waiting CLI request and awaits the next action (wall-clock deadline, default 30 min → no-tool
response = `wait_for_next_day`, logged `turn_timeout`); `step_apply` reuses `Agent._execute_tool`.
- `coffee <tool> args` returns the tool result plus everything until the agent's next wake. One call = one
  step = `tool_cost_minutes`. Shell commands cost no virtual time (documented change).
- Registration barrier: sim starts once all 6 called `coffee start` (30 min timeout → abort).
- Protocol (Bearer token): `POST /register`, `POST /act {agent,tool,args}` → ticket, `GET /wait?ticket&timeout=50`
  long-poll (→ result|sim_over|bankrupt|segment_end), `GET /status`, `GET /tools` (from `_collect_tools`),
  `GET /audit/{agent}`, `POST /telemetry`, `POST|GET /workspace`, `POST /admin/stop_at_next_morning`.
  Concurrent calls → 409.
- CLI `coffeebench/cli/coffee.py`, stdlib-only: `coffee <tool> --k v | '{json}'`, `coffee help [tool]`;
  outputs > 60 KB also written to `/workspace/out/N.json`.

## 3. Ending
CLI prints `SIMULATION OVER — set task_complete true` on sim end / bankruptcy / segment end.
task.toml `[agent] timeout_sec = 41400` (backstop below 12 h walltime); `max_turns` default;
`record_terminal_session: false`.

## 3b. Harness pin (litellm supply-chain incident, 2026-10-22)
litellm 1.107.0 (PyPI 2026-10-21 11:39–13:20 UTC) was a credential stealer (cook.pth / sitecustomize.py payload;
exfil to litellm[.]dev). Checked 2026-10-22: cluster harbor tool = harbor 0.24.0 + litellm 1.104.0,
local hbvenv litellm 1.92.2; project .venvs do not contain litellm; no IOC files / marker on login node or locally.
Pins in `harbor_adapter/requirements-harness.txt`; `cluster/pbs_job.sh` verifies them (dist-info dirs, no payload files)
by file inspection before `.env` is loaded and aborts on mismatch. Never `uv tool upgrade harbor` without
re-pinning. Only terminus-2 (CoffeeTerminus) is used — no third-party agent CLIs to pin.

## 4. Context & reasoning
- Harbor `Chat` keeps assistant turns as text only: thinking blocks / OpenRouter reasoning_details are
  NOT replayed for any model (uniform; stated in the paper).
- Summarization replaces history with a fresh conversation (`terminus_2.py:1050-1065`); unwind drops only
  trailing pairs → no in-place edits of earlier turns. Still smoke-test Fable 5 / Haiku 5.5 (5 turns).
- Unknown models default to a 1M context → summarization never fires. Pass
  `model_info: {max_input_tokens: 1000000, max_output_tokens: 32768}` for ALL models (uniform budget; see below).
- Context budget fix (2026-10-21). Audit of all trial logs: proactive summarization FAILED for most
  max-effort focal runs (Sonnet55/Opus55/Haiku55 ~99%, MuseW 64%, Fable5/Gemini38F/GLM53 ~40%) with
  "hit max_tokens limit" — reasoning used up the 32,768-token cap before the summary was written; terminus-2
  then just continued, so the context grew past 200K (Sonnet55 90-day run: 425K, never summarized; Claude's
  native window is 1M). Also the budget was measured with litellm's tiktoken estimate (Fable5 reached 297K
  real tokens with working summaries). Budget now (user, 2026-10-21): **uniform 1M tokens** for every model
  (all official endpoints offer >= 1,000,000), **compaction at 85%** (`proactive_summarization_threshold:
  150000`). Fix in CoffeeTerminus: summary/question/answer calls get max_tokens 65,536 (largest cap shared
  by all official endpoints) and, while the output still runs out (or an endpoint rejects a rung), retry
  down the effort ladder max → xhigh → high → medium → low from the turn's own effort, then reasoning off
  (no effort sent: API default → off; mandatory-reasoning models reject off); if summarization still fails, the oldest whole turns are dropped to 50% of the
  budget with a harness note (`context_truncations` in telemetry); the token count is the litellm
  estimate scaled by the provider-reported prompt tokens of the last turn.
  $0 test: `scripts/test_context_budget.py`. All pre-fix runs (m*, p*, h1) had unequal effective budgets.
- Per-model kwargs (`llm_call_kwargs` are merged, `lite_llm.py:323-334`):
  ```yaml
  kwargs:
    temperature: null
    model_info: {max_input_tokens: 1000000, max_output_tokens: 32768}
    record_terminal_session: false
    llm_call_kwargs:
      max_tokens: 32768
      timeout: 900
      num_retries: 2
      extra_body: {provider: {order: [<pinned>], allow_fallbacks: false}}
  ```
  `reasoning_effort` unset = API default.

## 5. Cost
litellm copies OpenRouter `usage.cost` into response cost (billed amount incl. tiers/cache). litellm has
no `claude-haiku-5-5` price → route Haiku 5.5 via `openrouter/anthropic/claude-haiku-5.5` (slug TBC),
pinned to anthropic. Per-agent cost = Σ segments `agent_result.cost_usd` (incl. summarization subagents);
cross-check with server telemetry.

## 6. Checkpoint / resume
Pickle blockers: `Marketplace.lock` (RLock), `EventLogger` file handle/lock, lambdas
(`ba._record_truth`, `defaultdict(lambda: …)`), `Agent` objects → `__getstate__/__setstate__` + module-level
functions. Snapshot at the start of each MORNING_OPEN bucket (all agents parked): env state, undelivered
agent text, events-file byte offset. Agents reset at a segment end (walltime stop / crash restart / optional fixed
`--segment-days`) for every model; the server emits `segment_end`, trials finish, the launcher relaunches (same cluster job) or self-requeues.
Resume = EXACT LLM-context resume (user, 2026-10-17; below); the deterministic env-generated digest
(position, open listings/offers/deals, AR/AP, last 15 days of own actions, DMs, price history) is only a
fallback. `/workspace` IS preserved (below).
Restarts happen only when needed (user, 2026-10-17): no planned segments by default (`--segment-days 0`).
- Walltime: launch_sim's `StopWatch` polls `/status` (admin token), measures s/simulated day, and when
  remaining walltime (CB_JOB_START + CB_WALLTIME_SEC − `--safety-sec`) < max(s/day × `--stop-margin-days`
  [3], `--stop-margin-min-sec` [1800]) POSTs `/admin/stop_at_next_morning` → the server does a segment end
  at the next MORNING_OPEN (`stop_reason: walltime`). < `--min-segment-sec` [3600] left → requeue.
- Crash: server death, or a Harbor job exiting while its agents are still in the running sim (60 s
  grace) → stop the jobs (SIGINT to the process group), freeze `latest.pkl` as `day_NNN.pkl`, restart from
  it (≤ `--max-retries`, reset on progress). The partial day is replayed (events truncated).
- Every restart → `<run_dir>/restarts.json` (+ resume.json): `{day, reason: walltime|crash|planned, segment}`.
- Workspace: at every daily checkpoint (hook is async: before pickling) the server answers each parked,
  polling CLI's `/wait` with `upload_workspace: D`; the CLI tars `/workspace` (cap 20 MB raw:
  `/workspace/out` dropped first, then largest files) and POSTs it, then keeps waiting (≤ 60 s total,
  `--workspace-timeout`). Stored as `<run_dir>/workspaces/<agent>/day_DDD.tar.gz` → also crash restarts
  get the workspace exactly as of the checkpoint. A changed over-cap drop list becomes a note in the
  agent's next observation (part of the checkpointed outbox → identical with/without restart). A resumed
  server serves the newest tarball with day ≤ the checkpoint day; the first payload carries
  `workspace_restore` (`"quiet"` under exact resume) and the CLI extracts it (safe members only).
- Exact resume (`coffeebench/server/agent_state.py`, `harbor_agent.py`): before every blocking `coffee`
  command CoffeeTerminus (host process) writes `$COFFEE_STATE_DIR/<agent>/call_<seq>.json` (Chat
  messages, remaining commands of the batch, tmux pane + incremental-output base buffer, the episode's
  task_complete/feedback, pending-completion, summarization count) and a container marker
  `/tmp/.coffee_call` `{seq, sub, resume}`; the CLI sends call id `<seq>.<sub>` with /register,/act,/wait;
  the server stores the id parked on each open ticket in every checkpoint (`pending_call`, plus
  `checkpoints/refs.json`) and continues ticket numbering. A `segment_end` output ends the old trial
  immediately (no LLM call). Resumed trial: `GET /resume_info` → snapshot → `POST /resume_ack exact`
  (server drops the digest) → restore Chat, write the snapshot pane straight to the pane tty (no command
  echo; fixed PS1 `root@coffee` so prompts match), re-send the parked command with the CLI in resume mode
  (invocation k does register+wait = the parked call's continuation; earlier invocations of a compound
  command print `skipped`), run the rest of the batch, build the observation like terminus-2, continue
  `_run_agent_loop`. Harbor's own `resume_trajectory`/`load_trajectory` do not apply (terminus-2 has no
  native resume; ATIF steps are not the raw chat). Cost/tokens restart per trial (launcher sums).
  GC at each checkpoint: snapshots/workspaces older than latest.pkl's and not referenced by a retained
  checkpoint are deleted. Fallback to the digest (`resume_mode: fallback_digest` in harbor_server.json):
  agent not parked in a CoffeeTerminus blocking call at the checkpoint (turn timeout, `coffee` inside a
  non-`coffee`-prefixed command), or no snapshot. Unavoidable differences: shell state (cwd, variables,
  background jobs) is lost; compound commands (`coffee a; coffee b`) show `skipped` for the already
  executed part.

Implemented (2026-10-17):
- `Environment.checkpoint_hook(env, day) -> bool` at the start of every MORNING_OPEN bucket (before
  the day's mechanics); truthy → `run()` returns `{"stopped_at_day": d}` without `_finish()`.
  `run(resume=True)` skips init/run_start/bootstrap. Env pickles without `agents`/hook; Marketplace
  drops its RLock; EventLogger keeps (path, offset, elapsed ms); `_record_truth` is a bound method;
  `defaultdict(_minus_one)`. Object ids come from a seeded per-run RNG (`Marketplace.seed_ids`) →
  reproducible ids, also across resume.
- `coffeebench/server/checkpoint.py`: snapshot = env + per-agent state (transcript; scripted model
  internals incl. HeuristicRoasterModel; RemoteAgent outbox/counters/usage) + events byte offset.
  Restore = `build_run(open_events=False)` → swap in the env, rebind tools to the restored
  BusinessApps, reopen events truncated to the offset.
- Server: `checkpoints/latest.pkl` daily; `--segment-days N` stops at the next multiple of N
  (`checkpoints/day_NNN.pkl`, `segment_NNN.json`, `resume_ckpt` in harbor_server.json);
  `--resume <ckpt>`. The `segment_end` event is written after the offset → dropped on resume, so the
  final run.json / events equal an uninterrupted run (smoke_local `ckpt` / `resume`).
- Digest (`coffeebench/server/digest.py`, ≤ 80k chars) is sent once in the first payload (`digest`
  field; the CLI prints it as `--- resume digest ---`); no dataset regeneration per segment.
- `launch_sim.py` loops over segments in one job; walltime check (CB_JOB_START + CB_WALLTIME_SEC vs
  max measured segment × 1.2) → `<sim>.resume.json` + `qsub ... CB_RESUME=<ckpt>` self-requeue.

## 7. Verifier / artifacts
`tests/test.sh` GETs `/audit/$AGENT` over `$COFFEE_SOCKET` (verifier token) → `/logs/verifier/reward.json` (`true_net_income`; 0 for non-final
segments). Server writes `trajectories/hb_<cfg>/seed_N/run.events.jsonl` + `run.json` as before.

## 8. Changes vs the submission harness (paper)
JSON-terminal interface; multiple `coffee` calls per LLM response; free shell compute; timeout-idle failure
mode (report rate); terminus summarization + fixed segment resets replace ContextCompactor; no reasoning
replay; persona in user instruction (no system prompt). Apply order unchanged.

## 9. Cluster (PBS, rootless podman)
`cluster/pbs_job.sh` runs one or more simulations on one node (one per seed) and self-requeues when the walltime runs
low (§6). Rootless podman (vfs, CNI, `DOCKER_BUILDKIT=0`) is set up by `cluster/rootless_podman_env.sh`. Datasets are
generated with `--network none` (default). Verify a new site with an oracle run first: solve.sh's `isolation_check.py`
lines in the oracle agent logs (NET_CURL_BLOCKED, NET_PY_BLOCKED, SERVER_ALLOWLIST_OK, COFFEE_CLI_OK, CREDENTIALS_OK)
and non-zero verifier rewards. Pre-build `cb_prewarm` (python:3.11-slim + tmux + curl + `coffee`). Start at 6
simulations/node. Pin the same Harbor version on the cluster and locally (`requirements-harness.txt`).
Submit from a fresh checkout: `qsub -P <project> -q <queue> -l select=1 -l walltime=12:00:00 -N cb_<model>_<role>_s<seed>
-v CB_CONFIG=...,CB_FOCAL=...,CB_BG=... harbor_adapter/cluster/pbs_job.sh`.

## File layout
- `coffeebench/server/{app.py, remote_agent.py, checkpoint.py, digest.py, __main__.py}`
- `coffeebench/cli/coffee.py`, `coffeebench/harbor_agent.py`
- `harbor_adapter/{generate_tasks.py, templates/job_{focal,bg}.yaml.j2, launch_sim.py, cluster/pbs_job.sh,
  cluster/rootless_podman_env.sh, aggregate_costs.py}`
- `environment.py`: MORNING_OPEN checkpoint hook, `agent.on_end(reason)`, pickle safety.

## Implementation order (local smoke tests are $0 until step 5)
1. Pickle safety + checkpoint + digest; passive/heuristic sim to day 5 → resume to 10 == uninterrupted run.
2. RemoteAgent + server + CLI without Harbor; 6 shell-loop CLIs vs in-process scripted run.
3. Hybrid: scripted agents in-process on the server.
4. Harbor `oracle` agent with scripted `solve.sh`, 6 containers, 3-day sim.
5. CoffeeTerminus + Haiku 5.5 (1 focal + 5 scripted, 2 days, ~$1); 5-turn check per focal model. (paid → ask)
6. Full 6-LLM 3-day pilot on the cluster; measure wall time/day → cost estimate → user approval → matrix.

## Open questions
1. OpenRouter slug for Haiku 5.5. 2. Uniform 200k context cap? 3. Segment length (decided: none, restart on demand).
4. Wake deadline; timeouts = idle? 5. Shell turns free of virtual time? 6. Keep `/workspace` across
segments? (decided: yes) 7. Uniform no-reasoning-replay (paper note)?

## Decisions (user, 2026-10-17)
- Reasoning: no replay, uniform across models (paper note).
- Segment length: ~~15 simulated days~~ → superseded (2026-10-17): no planned restarts; restart only
  when the walltime runs out or after a crash (see §6). `--segment-days N` remains for tests.
- Turn rules: shell commands free of virtual time; 30 min wall-clock wake deadline → treated as idle
  (`wait_for_next_day`), timeout rate reported.
- Context cap: uniform `max_input_tokens: 1000000` for all models (2026-10-21; was 200000 — equal budget,
  = every model's native window; compaction at 85%).
- Reasoning effort (user, 2026-10-17, supersedes "API default" for focal models): focal models run at
  each model's MAXIMUM effort, matching one fixed table (`launch_sim.FOCAL_EFFORT`: Astra/Fable5/
  K3/Muse13/DSv41F/GLM53 max, Gemini38F high, Qwen38F xhigh, MiMo26P none), sent as OpenRouter
  `extra_body.reasoning.effort`. Background agents stay at the API default (market held fixed).
  `--effort default` reproduces the API-default protocol (output tag `<focal>-default`).
