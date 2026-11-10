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

"""Record or independently verify one private reset snapshot without a planner/VLA."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("record", "restore"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda-device", type=int, default=1)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    os.environ["LIBERO_TYPE"] = "pro"
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    from robots.libero.env_server import LiberoEnvFacade, make_env
    from rpent.utils.egl import configure_egl_device

    configure_egl_device(args.cuda_device)
    import torch

    torch.cuda.set_device(args.cuda_device)
    env = make_env(
        args.task,
        args.seed,
        args.suite,
        initial_scene=str(args.path.resolve()),
        scene_mode=args.mode,
    )
    facade = LiberoEnvFacade(
        env,
        meta={},
        initial_scene=str(args.path.resolve()),
        scene_mode=args.mode,
        scene_receipt=str(args.receipt.resolve()),
    )
    try:
        facade.reset()
    finally:
        facade.close()


if __name__ == "__main__":
    main()
