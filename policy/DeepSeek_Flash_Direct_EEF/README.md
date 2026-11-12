# DeepSeek Flash Direct EEF

Evaluation only, for RoboDojo `arx_x5`, one environment, `action_type=joint`.
The API chooses Cartesian grasp-point targets. The simulator's existing CuRobo
planner produces joint paths, executed using the upstream EEF agent's limits.
There is no training, dataset conversion, or local model checkpoint.

## Isolation and serving

The complete agent, conversation, image preparation and provider SDK run in
the policy process. `deploy.py` is an RPC proxy plus local simulator planning
and execution. No SDK is imported or installed in the simulator environment.
The shared WebSocket server owns plain/encoded RGB decoding. Public custom
RPCs provide a continuation for the local planner without holding the server
model lock during an API call or while waiting for simulator planning.

Windows parent-workspace launch:

```powershell
python scripts/policy_remote.py serve --server local-llm `
  --policy-name DeepSeek_Flash_Direct_EEF --task <task>
```

The client profile selects the isolated `local_python` and named User credential
`DEEPSEEK_API_KEY`. It calls the normal `XPolicyLab/setup_policy_server.py`.
Start simulation separately with `policy_remote.py client`, using the same
profile, task and policy. For a whole campaign use `scripts/evaluate_remote.py`.
The client launcher strips the configured credential from the simulator's
environment; the credential is only injected into the policy child process.

For a fresh Windows policy environment, create a **new** venv without system
site packages and install `requirements.txt` using its Python. Linux has
`install.sh <base-python> <new-venv>`; alternatively install the requirements
into a dedicated conda environment and use `setup_eval_policy_server.sh` with
the standard ten positional arguments. Never use the simulator environment.

## Contract and records

The provider is fixed to `https://api.deepseek.com` and Responses API, model
`deepseek-flash`; no Azure/AIDP gateway is used. Defaults: high reasoning effort,
65536 output tokens per request, 100 calls per episode, 600-second HTTP timeout,
one retry for transient infrastructure errors, 1800-second decision deadline.
The initial private pilot used 8192 tokens and exposed reasoning truncation;
the larger budget must be validated before the scored campaign.
Recipes, prior attempts, depth and arbitrary observation extras are disabled.
Ordinary instructions, RGB images, proprioception and embodiment documentation
form the policy input. Existing `GPT_6_Astra_Direct_EEF` supplies the EEF tool
schema, motion validation, path resampling and execution loop.

`deepseek-flash` is a provider alias, not an immutable downloadable checkpoint.
Record the date, requested/returned model, adapter/config hashes and token
usage. This does not establish reproduction of a historical website score.

Private `trace_dir` stores API durations/usage separately from agent/tool logs;
no API key or raw SDK exceptions are recorded. The simulator saves a private
session-to-result mapping in `ROBODOJO_LLM_METRICS_DIR`. API request wall time
includes Internet transport and provider scheduling/inference. Local planning
and simulation frames are separate measurements. Nothing here should be
copied into a public results website.

Synthetic plain/encoded RGB checks through the real server and provider passed;
the planner continuation/session tests passed. These are interface checks, not
benchmark scores. See the parent workspace's private campaign state for live
simulator validation and episode coverage.
