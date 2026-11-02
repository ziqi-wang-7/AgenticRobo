# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local OTLP/JSON receiver for timing, without prompts or credentials."""

from __future__ import annotations

import gzip
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class EventWriter:
    """Serialize small timing records to one run-owned file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8", buffering=1)
        self.lock = threading.Lock()

    def emit(self, kind: str, **fields) -> None:
        """Append a timestamped measurement."""
        event = {"kind": kind, "wall_ns": time.time_ns(), **fields}
        with self.lock:
            self.file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def close(self) -> None:
        """Close after all producers stop."""
        self.file.close()


def attributes(values: list[dict]) -> dict:
    """Retain numeric measurements and a narrow set of non-content fields."""
    safe = {
        "event.name",
        "event.kind",
        "conversation.id",
        "turn.id",
        "response.id",
        "model",
        "model_name",
        "duration_ms",
        "attempt",
        "success",
        "kind",
        "http.response.status_code",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "reasoning_output_tokens",
        "total_tokens",
        "otel.name",
        "operation.name",
        "request_id",
        "endpoint",
    }
    result = {}
    for pair in values:
        key = pair.get("key", "")
        if key not in safe and not key.endswith("_tokens") and not key.endswith("_ms"):
            continue
        value = pair.get("value", {})
        for tag in ("stringValue", "intValue", "doubleValue", "boolValue"):
            if tag in value:
                result[key] = value[tag]
                break
    return result


class CodexTelemetry:
    """Receive the episode's Codex events on loopback; never alter global config."""

    def __init__(self, writer: EventWriter) -> None:
        self.writer = writer
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length < 16 * 1024 * 1024:
                        raise ValueError("invalid payload length")
                    body = self.rfile.read(length)
                    if self.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    outer.consume(json.loads(body))
                except Exception as exc:
                    writer.emit("otel_decode_error", error_type=type(exc).__name__)
                    self.send_error(400)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        """Start receiving before launching Codex."""
        self.thread.start()

    def overrides(self) -> tuple[str, ...]:
        """Return per-process TOML overrides for the SDK configuration."""
        endpoint = f"http://127.0.0.1:{self.server.server_port}"
        return (
            f'otel.exporter={{otlp-http={{endpoint="{endpoint}/v1/logs",protocol="json"}}}}',
            f'otel.trace_exporter={{otlp-http={{endpoint="{endpoint}/v1/traces",protocol="json"}}}}',
            "otel.log_user_prompt=false",
        )

    def consume(self, payload: dict) -> None:
        """Extract timing records; discard bodies and unapproved attributes."""
        for resource in payload.get("resourceLogs", []):
            for scope in resource.get("scopeLogs", []):
                for record in scope.get("logRecords", []):
                    self.writer.emit(
                        "otel_log",
                        timestamp_ns=int(record.get("timeUnixNano", 0)),
                        attributes=attributes(record.get("attributes", [])),
                    )
        for resource in payload.get("resourceSpans", []):
            for scope in resource.get("scopeSpans", []):
                for span in scope.get("spans", []):
                    if span.get("name") not in {
                        "responses_websocket.stream_request",
                        "codex.sampling",
                        "run_sampling_request",
                        "try_run_sampling_request",
                        "stream_request",
                        "receiving_stream",
                        "model_client.stream_responses_sse",
                        "responses_sse.stream_request",
                    }:
                        continue
                    start = int(span.get("startTimeUnixNano", 0))
                    end = int(span.get("endTimeUnixNano", 0))
                    self.writer.emit(
                        "otel_span",
                        name=span.get("name"),
                        start_ns=start,
                        end_ns=end,
                        duration_ms=(end - start) / 1e6,
                        trace_id=span.get("traceId"),
                        span_id=span.get("spanId"),
                        parent_id=span.get("parentSpanId"),
                        attributes=attributes(span.get("attributes", [])),
                        status=span.get("status", {}).get("code"),
                    )

    def close(self) -> None:
        """Stop after Codex has flushed its exporters on shutdown."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
