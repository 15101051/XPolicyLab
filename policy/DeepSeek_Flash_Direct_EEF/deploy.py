"""Simulator-side RPC proxy and CuRobo execution; no SDK or conversation here."""

from dataclasses import asdict, replace
import json
import os
import sys
from pathlib import Path
import time

from XPolicyLab.policy.GPT_6_Astra_Direct_EEF import deploy as eef_deploy
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect import deploy as joint_deploy
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.policy import CapabilityFailure, InfrastructureFailure
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.types import Action, ActionChunk
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.docs import eef_docs_from_joint_docs


class RemoteAgent:
    def __init__(self, *, client, action_spec, task_env, **_):
        self.client = client
        self.action_spec = replace(action_spec, docs=eef_docs_from_joint_docs(action_spec.docs))
        self.plan = eef_deploy._planner(task_env)
        self.calls = 0
        self.hindsight = None
        info = client.call("begin_episode", obs={"action_spec": asdict(self.action_spec)})
        self.session = info["session"]
        self.decision_timeout_s = float(info.get("decision_timeout_s", 600))
        self.task_env = task_env

    def prepare(self, observation):
        pass

    def transcript(self):
        # Private agent traces are saved by the serving process itself.
        return []

    def audit_config(self):
        return {"adapter": "DeepSeek_Flash_Direct_EEF", "model": "deepseek-flash",
                "session": self.session}

    def act(self, observation):
        self.client.call("agent_start", obs={
            "session": self.session,
            "vision": {name: {"color": value} for name, value in observation.images.items()},
            "state": dict(observation.state), "instruction": observation.instruction,
            "step": observation.step, "remaining_steps": observation.remaining_steps})
        deadline = time.monotonic() + self.decision_timeout_s
        while time.monotonic() < deadline:
            result = self.client.call("agent_poll", obs={"session": self.session})
            if "calls" in result:
                self.calls = result["calls"]
            state = result["state"]
            if state == "planning":
                request = result["request"]
                try:
                    reply = {"result": self.plan(arm=request["arm"], target_pose=request["target_pose"])}
                except Exception:
                    reply = {"error": "planner execution failed"}
                self.client.call("planner_result", obs=dict(reply, session=self.session, id=request["id"]))
            elif state == "ready":
                self.calls = result["calls"]
                chunk = dict(result["chunk"])
                chunk["actions"] = [Action(**item) for item in chunk["actions"]]
                return ActionChunk(**chunk)
            elif state == "error":
                failure = CapabilityFailure if result["kind"] == "capability" else InfrastructureFailure
                raise failure("DeepSeek agent: " + result["message"])
            else:
                time.sleep(0.1)
        raise InfrastructureFailure("DeepSeek decision deadline exceeded")

    def confirm_executed(self, played):
        # The inherited EEF agent's confirm_executed is a no-op. Observations
        # supply the measured state for its next decision.
        pass

    def finish(self, completed_normally):
        result = self.client.call("finish_episode", obs={"session": self.session})
        deadline = time.monotonic() + 650
        while result.get("state") == "stopping" and time.monotonic() < deadline:
            time.sleep(0.2)
            result = self.client.call("finish_episode", obs={"session": self.session})
        if result.get("state") == "stopping":
            raise InfrastructureFailure("Agent cancellation deadline exceeded")
        directory = Path(os.environ.get("ROBODOJO_LLM_METRICS_DIR", ".cache/llm/episodes"))
        directory.mkdir(parents=True, exist_ok=True)
        # Mapping to private result provenance stays on this machine.
        result.update(completed_normally=completed_normally, finished_at=time.time(),
                      run_id=os.environ.get("ROBODOJO_RUN_ID"),
                      layout_id=int(self.task_env.env_seeds[getattr(self.task_env, "env_idx", 0)]),
                      task=self.task_env.task_name)
        (directory / (self.session + ".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")


def eval_one_episode(TASK_ENV, model_client):
    proxy = None

    def factory(**kwargs):
        nonlocal proxy
        proxy = RemoteAgent(client=model_client, **kwargs)
        return proxy

    # Reuse the existing execution loop, stop/capability handling and action
    # bounds. Traces containing images are intentionally disabled here.
    try:
        joint_deploy.eval_one_episode(TASK_ENV, model_client, policy_factory=factory)
    finally:
        if proxy is not None:
            active_error = sys.exc_info()[0] is not None
            try:
                proxy.finish(not active_error)
            except Exception:
                if not active_error:
                    raise InfrastructureFailure("Failed to archive provider timing") from None
                print("[DeepSeek] Could not finalize failed episode timings", flush=True)


def eval_one_episode_batch(TASK_ENV, model_client):
    if int(getattr(TASK_ENV, "num_envs", 1)) != 1:
        raise InfrastructureFailure("DeepSeek requires num_envs=1 and eval_batch=false")
    eval_one_episode(TASK_ENV, model_client)
