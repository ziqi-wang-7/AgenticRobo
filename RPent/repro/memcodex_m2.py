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

"""Prepare an indexed M2 evaluation paired with the frozen M0 task manifest."""

import argparse
from pathlib import Path

from repro.memcodex_m1 import prepare


def main() -> None:
    """Freeze an M2 batch without starting evaluation or changing M0/M1 results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("memory/memcodex_v1"))
    parser.add_argument(
        "--index", type=Path, default=Path("memory/memcodex_m2_v1/INDEX.json")
    )
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--action-evidence", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--suite")
    parser.add_argument("--task", type=int)
    parser.set_defaults(mode="m2")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
