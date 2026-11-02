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

"""Time the production Pi0.5 predictor without changing RNG or action output."""

import json
import os
import time
from pathlib import Path

from profiling.codex_telemetry import EventWriter


def main() -> None:
    """Run the production server, recording synchronized inference and CPU copy."""
    from rpent.robots.components import pi05_vla_server as production

    output = Path(os.environ["RPENT_BASELINE_SERVICES"])
    writer = EventWriter(output / "vla.jsonl")
    original = production.Pi05VLAFacade.predict

    def predict(self, *args, **kwargs):
        marker = json.loads((output / "current-episode.json").read_text())
        started = time.perf_counter_ns()
        error = None
        try:
            # The production .cpu().numpy() synchronizes the forward pass.
            return original(self, *args, **kwargs)
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            writer.emit(
                "vla_server",
                episode_id=marker["episode_id"],
                duration_ms=(time.perf_counter_ns() - started) / 1e6,
                error=error,
            )

    production.Pi05VLAFacade.predict = predict
    try:
        production.main()
    finally:
        writer.close()


if __name__ == "__main__":
    main()
