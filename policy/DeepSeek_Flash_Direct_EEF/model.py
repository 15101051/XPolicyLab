"""Full agent state lives here; simulator planning uses nonblocking continuations.

The shared websocket server holds a model lock during each RPC. An API worker
therefore yields planner requests through agent_poll instead of calling back
over a second connection while that lock is held.
"""

from dataclasses import asdict
from pathlib import Path
import json
import threading
import uuid

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect import policy as shared
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.types import Observation
from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.policy import EefAgentPolicy
from .provider import ProviderClient


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.config = dict(model_cfg)
        if self.config.get("env_cfg_type") != "arx_x5" or self.config.get("action_type") != "joint":
            raise ValueError("DeepSeek EEF supports arx_x5 joint-path execution only")
        self.action_dimensions = get_robot_action_dim_info(self.config["env_cfg_type"])
        self._lock = threading.Lock()
        self._thread = None
        self._policy = None
        self._provider = None
        self._session = None
        self._decision = None
        self._pending = None
        self._reply = None
        self._event = threading.Event()

    def reset(self):
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Cannot reset an active agent; cancel it first")
        self._policy = None
        self._decision = self._pending = self._reply = None

    def begin_episode(self, payload):
        self.reset()
        spec = dict(payload["action_spec"])
        spec["labels"] = tuple(spec["labels"])
        spec["low"] = np.asarray(spec["low"], dtype=np.float64)
        spec["high"] = np.asarray(spec["high"], dtype=np.float64)
        spec["max_step"] = tuple(spec["max_step"]) if spec.get("max_step") else None
        action_spec = shared.RoboDojoActionSpec(**spec)
        self._session = uuid.uuid4().hex
        self._provider = ProviderClient(self.config, self._session)
        shared.PLANNERS["deepseek"] = shared.Planner("deepseek-flash", "responses", "")
        # Construct a fixed configuration; ambient L4_* values cannot enable a
        # task recipe, a prior attempt, or an alternative provider gateway.
        env = {"L4_INSPECT_PLANNER": "deepseek", "L4_INSPECT_MODEL": self.config.get("model", "deepseek-flash"),
               "L4_INSPECT_BASE_URL": "https://api.deepseek.com", "L4_INSPECT_API_STYLE": "responses",
               "L4_INSPECT_API_KEY_ENV": "DEEPSEEK_API_KEY", "L4_INSPECT_USE_RECIPE": "0",
               "L4_INSPECT_DEPTH": "off", "L4_INSPECT_KEEP_ALL_IMAGES": "1",
               "L4_INSPECT_REASONING_EFFORT": self.config.get("reasoning_effort", "high"),
               "L4_INSPECT_MAX_LLM_CALLS": str(self.config.get("max_llm_calls", 100))}
        self._policy = EefAgentPolicy(action_spec=action_spec, env=env,
                                      client=self._provider, planner=self._plan)
        return {"session": self._session, "server_pid": __import__("os").getpid(),
                "decision_timeout_s": float(self.config.get("decision_timeout_s", 1800)),
                "model": env["L4_INSPECT_MODEL"]}

    def _plan(self, *, arm, target_pose):
        token = uuid.uuid4().hex
        with self._lock:
            self._event.clear()
            self._reply = None
            self._pending = {"id": token, "arm": arm, "target_pose": target_pose}
        if not self._event.wait(float(self.config.get("decision_timeout_s", 600))):
            raise shared.InfrastructureFailure("Simulator planner continuation timed out")
        with self._lock:
            reply = self._reply
            self._pending = None
        if not reply or reply.get("error"):
            raise shared.InfrastructureFailure("Simulator motion planner failed")
        return reply["result"]

    def agent_start(self, payload):
        if not self._policy:
            raise RuntimeError("begin_episode is required")
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Agent already has an active decision")
        if payload["session"] != self._session:
            raise RuntimeError("Stale episode session")
        # The server has already decoded vision colors. No task/layout labels,
        # reward data or arbitrary observation extras enter the agent.
        obs = Observation(images={name: value["color"] for name, value in payload["vision"].items()},
                          state=payload["state"], instruction=payload.get("instruction"),
                          step=int(payload["step"]), remaining_steps=payload.get("remaining_steps"))
        with self._lock:
            self._decision = self._pending = None

        def decide():
            try:
                chunk = self._policy.act(obs)
                result = {"state": "ready", "chunk": asdict(chunk)}
            except shared.CapabilityFailure as error:
                result = {"state": "error", "kind": "capability", "message": str(error)}
            except BaseException as error:
                result = {"state": "error", "kind": "infrastructure", "message": type(error).__name__}
            with self._lock:
                self._decision = result

        self._thread = threading.Thread(target=decide, name="deepseek-agent", daemon=True)
        self._thread.start()
        return {"state": "pending"}

    def agent_poll(self, payload):
        if payload["session"] != self._session:
            raise RuntimeError("Stale episode session")
        with self._lock:
            if self._decision is not None:
                return dict(self._decision, calls=self._policy.calls,
                            provider_records=list(self._provider.records))
            if self._pending is not None and self._reply is None:
                return {"state": "planning", "request": self._pending}
            return {"state": "pending"}

    def planner_result(self, payload):
        with self._lock:
            if (payload["session"] != self._session or not self._pending
                    or payload["id"] != self._pending["id"] or self._reply is not None):
                raise RuntimeError("Stale planner continuation")
            self._reply = payload
            self._event.set()
        return {"ok": True}

    def finish_episode(self, payload):
        if payload["session"] != self._session:
            raise RuntimeError("Stale episode session")
        if self._thread and self._thread.is_alive():
            self._provider.cancelled.set()
            with self._lock:
                self._reply = {"error": "cancelled"}
                self._event.set()
            self._thread.join(timeout=0)
        if self._thread and self._thread.is_alive():
            return {"state": "stopping", "session": self._session}
        directory = Path(self.config.get("trace_dir", ".cache/llm/serving"))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / (self._session + "-agent.json")).write_text(
            json.dumps({"session": self._session, "config": self._policy.audit_config(),
                        "serving_config": {key: self.config.get(key) for key in (
                            "model", "reasoning_effort", "max_output_tokens", "max_llm_calls",
                            "api_timeout_s", "decision_timeout_s")},
                        "calls": self._policy.calls, "turns": self._policy.transcript()}, indent=2),
            encoding="utf-8")
        return {"state": "finished", "session": self._session, "calls": self._policy.calls,
                "provider_records": list(self._provider.records)}
