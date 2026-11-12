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

"""Conservative, source-derived coverage of particular recovery problems.

These labels describe historical text. They do not verify a physical state or
predict success. Unknown failure descriptions remain explicitly uncovered.
"""

from __future__ import annotations

import re

FAILURES = {
    "wrong_drawer": r"wrong (?:drawer|handle|layer)|(?:middle|upper|top) drawer (?:rather than|instead of)|wrong.*drawer|drawer.*wrong|wrong.*layer",
    "wrong_spatial_relation": r"(?:wrong|incorrect|unmet|unsatisfied|not satisfied).{0,45}(?:left|right|side|spatial|relation|placement)|(?:left|right)[ -]side.{0,35}(?:relation|placement).{0,25}(?:unmet|wrong|incorrect|not satisfied)|(?:relation|placement).{0,35}(?:wrong side|wrong direction|not satisfied|unmet)|placed? on the wrong side",
    "object_collision": r"\bcollision\b|collid(?:e|ed|ing|es)|bump(?:ed|ing)? into|hit (?:the )?(?:cup|mug|object|obstacle)",
    "grasp_space_insufficient": r"(?:not enough|insufficient|limited|no).{0,35}(?:grasp|gripping|pickup) space|(?:grasp|gripping|pickup) space.{0,35}(?:insufficient|blocked|too small)|no room to (?:grasp|pick)|obstructed grasp",
    "object_tipped": r"\btipp?ed\b|\btilt(?:ed|ing)?\b|fell over|toppl(?:ed|ing)",
    "object_slipped": r"\bslip(?:s|ped|ping)?\b|\bdrop(?:ped|ping)?\b|lost (?:the )?(?:grasp|grip|mug|pot|object)",
    "grasp_not_closed": r"(?:finger|gripper)s?.{0,30}(?:stay|remain|still|left).{0,15}open|no clos|does not close|did not close|no lift|without (?:closing|closure)|open.finger",
    "insertion_blocked": r"(?:insert|insertion|cavity|mouth).{0,60}(?:stall|block|obstruct|jam)|(?:stall|block|obstruct|jam).{0,60}(?:insert|insertion|cavity|mouth)",
    "release_stuck": r"(?:releas|open finger|withdraw).{0,60}(?:caught|follow|hook|stuck)|(?:caught|hook|stuck).{0,60}(?:releas|withdraw)",
    "placement_not_satisfied": r"(?:placement|place|put|set).{0,45}(?:goal|predicate|target|relation).{0,25}(?:not satisfied|unmet|failed|wrong)|(?:goal|predicate).{0,35}(?:not satisfied|unmet) after (?:placement|release)",
    "contact_stalled": r"\bstall(?:ed|s|ing)?\b|no progress|(?:drawer|door).{0,35}(?:still|remain|left).{0,15}closed|(?:left|kept).{0,20}drawer.{0,10}closed|(?:missed|no).{0,15}(?:contact|engagement)",
}
_PHASES = {
    "wrong_drawer": {"approach", "articulate", "recovery"},
    "wrong_spatial_relation": {"transport", "release", "recovery"},
    "object_collision": {"approach", "grasp", "transport", "recovery"},
    "grasp_space_insufficient": {"approach", "grasp", "recovery"},
    "object_tipped": {"grasp", "transport", "insert", "recovery"},
    "object_slipped": {"grasp", "transport", "insert", "recovery"},
    "grasp_not_closed": {"approach", "grasp", "recovery"},
    "insertion_blocked": {"insert", "recovery"},
    "release_stuck": {"release", "recovery"},
    "placement_not_satisfied": {"transport", "release", "recovery"},
    "contact_stalled": set(),
}


def failure_kinds(text: str) -> list[str]:
    """Extract problem labels from an attributed issue description."""
    kinds = [
        kind
        for kind, pattern in FAILURES.items()
        if re.search(pattern, text, re.I | re.S)
    ]
    # A blocked insertion is more specific than a general motion stall.
    if "insertion_blocked" in kinds and "contact_stalled" in kinds:
        kinds.remove("contact_stalled")
    return sorted(kinds)


def object_types(text: str) -> set[str]:
    """Normalize only bounded object categories; empty means unspecified."""
    return {
        name
        for name, pattern in {
            "moka_pot": r"moka[ _-]*pot",
            "mug": r"\bmug\b|_mug\b",
            "bowl": r"\bbowl\b|_bowl\b",
            "drawer": r"drawer|cabinet",
            "microwave": r"microwave",
            "milk": r"\bmilk\b",
            "pudding_box": r"pudding(?:[ _-]*box)?|chocolate_pudding",
            "cup": r"\bcup\b|_cup\b",
            "plate": r"\bplate\b|_plate\b",
        }.items()
        if re.search(pattern, text, re.I)
    }


def recovery_cases(text: str) -> list[dict]:
    """Require both a failure symptom and a response in the same source passage.

    Table rows are independent cases. Narrative paragraphs are bounded by blank
    lines. A passing mention or a negated historical failure is not coverage.
    """
    passages = re.split(r"\n\s*\n|(?=^\|)", text, flags=re.M)
    cases = []
    for passage in passages:
        if re.search(
            r"\b(?:never|no observed|did not|didn't)\s+(?:slip|drop)", passage, re.I
        ):
            continue
        columns = (
            passage.strip().strip("|").split("|")
            if passage.lstrip().startswith("|")
            else []
        )
        symptom = columns[0] if len(columns) >= 3 else passage
        response = columns[-1] if len(columns) >= 3 else passage
        if not re.search(
            r"re-?locali[sz]|retry|re-?grasp|re-?pick|adjust|retreat|withdraw|re-?clamp|explicit.{0,15}clos|close the|reopen|switch|reverse|tilt|preposition|reposition|lower|stop|check|inspect|verify|push|move aside|clear (?:the )?(?:obstacle|space|object)|create (?:more )?space|make room|rotate|yaw|change orientation|realign|correct (?:the )?(?:side|position)",
            response,
            re.I,
        ):
            continue
        for kind in failure_kinds(symptom):
            case = {
                "kind": kind,
                "object_types": sorted(object_types(symptom)),
                "phases": sorted(_PHASES[kind]),
            }
            if case not in cases:
                cases.append(case)
    return sorted(cases, key=lambda x: (x["kind"], x["object_types"], x["phases"]))


def requirements(state: dict) -> tuple[str, ...]:
    """Preserve a structured failure, otherwise infer bounded issue labels."""
    explicit = state.get("failure_kind")
    if explicit:
        if explicit not in FAILURES:
            raise ValueError(f"Unknown recovery failure kind: {explicit}")
        return (explicit,)
    issue = str(state.get("issue", "")).strip()
    return tuple(failure_kinds(issue))


def covers(case: dict, kind: str, state: dict) -> bool:
    """Check historical problem, object scope and phase without asserting facts."""
    if case["kind"] != kind:
        return False
    current = object_types(str(state.get("failure_object", "")))
    scope = set(case["object_types"])
    phase = state.get("phase")
    return (not current or not scope or bool(current & scope)) and (
        not phase or not case["phases"] or phase in case["phases"]
    )
