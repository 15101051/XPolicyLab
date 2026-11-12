"""Direct, bounded provider requests. Imported only by the serving process."""

import json
import os
from pathlib import Path
import threading
import time

from XPolicyLab.policy.GPT_6_Astra_Direct_EEF.inspect.policy import (
    InfrastructureFailure, classify_openai_error,
)
from XPolicyLab.utils.openai_responses import (
    ReasoningReplayStore, chat_messages_to_responses_input,
    chat_tools_to_responses_tools, responses_to_chat_completion,
)


class ProviderClient:
    def __init__(self, config, session):
        from openai import OpenAI

        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise InfrastructureFailure("DEEPSEEK_API_KEY is missing from the serving process")
        self.config = config
        self.session = session
        self.store = ReasoningReplayStore()
        self.records = []
        self.client = OpenAI(api_key=key, base_url="https://api.deepseek.com",
                             max_retries=0, timeout=float(config.get("api_timeout_s", 600)))
        directory = Path(config.get("trace_dir", ".cache/llm/serving"))
        directory.mkdir(parents=True, exist_ok=True)
        self.trace_path = directory / (session + "-api.jsonl")
        self.cancelled = threading.Event()

    def complete(self, messages, tools):
        request = dict(model=self.config.get("model", "deepseek-flash"),
                       input=chat_messages_to_responses_input(messages, reasoning_store=self.store),
                       tools=chat_tools_to_responses_tools(tools),
                       reasoning={"effort": self.config.get("reasoning_effort", "high")},
                       max_output_tokens=int(self.config.get("max_output_tokens", 65536)))
        for attempt in range(2):
            if self.cancelled.is_set():
                raise InfrastructureFailure("Episode was cancelled")
            started = time.perf_counter()
            record = dict(kind="provider_inference", session=self.session, at=time.time(),
                          index=len(self.records), attempt=attempt, requested_model=request["model"],
                          max_output_tokens=request["max_output_tokens"],
                          reasoning_effort=request["reasoning"]["effort"])
            failure = None
            try:
                response = self.client.responses.create(**request)
                record.update(ok=True, model=response.model, status=response.status,
                              usage=response.usage.model_dump() if response.usage else None)
                if response.status == "failed" or response.error is not None:
                    record["ok"] = False
                    record["error_type"] = "ProviderResponseFailed"
                    failure = InfrastructureFailure("Provider returned a failed response", retryable=True)
            except Exception as error:
                # Never persist provider bodies or exception strings: either can echo inputs or keys.
                record.update(ok=False, error_type=type(error).__name__,
                              status_code=getattr(error, "status_code", None))
                failure = classify_openai_error(error)
            finally:
                record["seconds"] = time.perf_counter() - started
                self.records.append(record)
                with self.trace_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            if failure is None:
                return responses_to_chat_completion(response, reasoning_store=self.store)
            if not isinstance(failure, InfrastructureFailure) or not failure.retryable or attempt:
                raise failure from None
            if self.cancelled.wait(min(5, max(1, failure.retry_after_s or 1))):
                raise InfrastructureFailure("Episode was cancelled")
