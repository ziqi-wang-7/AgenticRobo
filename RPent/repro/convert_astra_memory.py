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

"""Convert the pinned official Astra memory into an auditable M1 corpus."""

import argparse
import json
import tempfile
from pathlib import Path

from repro.memcodex_corpus import build, encoded, read_sources, validate


def main() -> None:
    """Publish a complete conversion without replacing populated corpora."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if (
        output == source
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError("Source and output must be separate directories")
    catalog_document = json.loads(args.catalog.read_text())
    catalog = [
        {k: row[k] for k in ("suite", "task", "language")}
        for row in catalog_document["episodes"]
    ]
    files, manifest = build(read_sources(source, args.receipt), catalog)
    if output.exists():
        old = json.loads((output / "MANIFEST.json").read_text())
        if old.get("status") == "converted":
            if old != manifest:
                raise ValueError(
                    "Existing corpus differs from the requested conversion; choose a new output directory"
                )
            print(json.dumps(validate(output, source, args.receipt), indent=2))
            return
        if old.get("entry_count") != 0 or any(
            p.is_file() and p.name != "MANIFEST.json" for p in output.rglob("*")
        ):
            raise ValueError("Output is not an empty placeholder")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=".memcodex-convert-"
    ) as temporary:
        staged = Path(temporary) / "corpus"
        staged.mkdir()
        for name, content in files.items():
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (staged / "MANIFEST.json").write_bytes(encoded(manifest))
        report = validate(staged, source, args.receipt)
        if output.exists():
            backup = output.with_name(output.name + ".empty-before-m1")
            if backup.exists():
                raise FileExistsError(backup)
            output.rename(backup)
        staged.rename(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
