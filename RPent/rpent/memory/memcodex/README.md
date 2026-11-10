# Budgeted Memory Controller

[中文说明](README_zh.md)

`MemCodexController` retrieves task experience through four layers under an
explicit memory-context budget. It is an opt-in module. The original [`MemoryManager`](../manager.py)
remains the default; LIBERO Codex evaluation can select fixed M1 or indexed M2 below. This implementation borrows progressive retrieval from MemCodex without
importing the sibling project or calling an LLM.

## Run a Local Example

From the RPent repository root, use the existing Python 3.11+ environment. No
additional dependencies, GPU, model weights, or credentials are required.
The [demo corpus](../../../tests/fixtures/memcodex_demo) is synthetic; it is
not experience from a LIBERO rollout.

```bash
.venv/bin/python - <<'PY'
from rpent.memory.memcodex import MemCodexController

controller = MemCodexController("tests/fixtures/memcodex_demo", token_budget=2048)
observation = {"localized_objects": ["mug", "plate"]}
state = {
    "required_objects": ["mug", "plate"],
    "required_skills": ["pick", "place"],
    "available_skills": ["pick", "place"],
    "conditions": {"graspable": True},
}
context = controller.retrieve("demo/mug-to-plate", observation, state)
print(context)
trace = controller.last_result
assert trace is not None and trace.sufficient
assert trace.token_count <= controller.token_budget
print(trace.layers_visited, trace.token_count, trace.counter_name)

recovery = controller.retrieve_with_trace(
    "demo/mug-to-plate",
    observation,
    {**state, "anomaly": True, "needs_recovery": True,
     "conditions": {"graspable": True, "failure": "missed_grasp"}},
)
assert recovery.sufficient and recovery.layers_visited[-1] == "raw"
print(recovery.selected_ids)
PY
```

The first request stops after `summary`, `relational`, and `skills`; recovery
adds `raw`. `retrieve()` returns a string. `retrieve_with_trace()` returns an
immutable result containing the same context, budget measurements, selected and
omitted entry IDs, files read, missing evidence, corpus revision, cache status,
and elapsed retrieval time in milliseconds.

## Corpus and Query Contracts

Place one UTF-8 JSON object per file under `summary/`, `relational/`, `skills/`,
or `raw/`. Nested directories are supported. The containing directory determines
the layer; IDs must be unique within a layer. Every entry needs an explicit
`applies_when` mapping, including `{}` for unconditional applicability.

```json
{
  "id": "pick-place",
  "task_ids": ["demo/mug-to-plate"],
  "text": "Use pick then place. Confirm grasp, carry closed, release on support.",
  "applies_when": {"graspable": true},
  "provides": ["skill"],
  "skill_ids": ["pick", "place"],
  "objects": ["mug", "plate"]
}
```

`id`, `task_ids`, `text`, and `applies_when` are required. The other three fields
are optional lists. `condition_text` optionally preserves a natural-language
applicability condition, and `source_refs` is an optional list of source line
references. Progressive routing conservatively rejects entries with unresolved
`condition_text`; fixed mode supplies these conditions to the planner unchanged. Unknown fields and malformed entries raise a `ValueError`
with the filename. Each file is limited to 1 MiB. Symlinks inside layers are
rejected. The controller reads JSON entries, not the original Markdown corpus.

`provides` describes evidence actually present in the text: `task` for the goal
and task strategy, `skill` for actionable skill instructions, `anomaly` for
handling the indicated anomaly, and `recovery` for recovery guidance. Merely
naming a skill in `skill_ids` does not provide its instructions. Metadata must
be reviewed during corpus conversion; the controller cannot verify its truth.

| Input | Meaning |
| --- | --- |
| `task` | Canonical task ID or exact instruction, matched exactly against `task_ids`. `*` permits shared entries but cannot alone establish task-specific evidence. |
| `observation.localized_objects` | Names of objects located by current perception. Images alone do not establish localization. |
| `state.required_objects` | Objects that must be located. If omitted, use object names in retained entries. |
| `state.objects_localized` | Optional authoritative boolean. `False` overrides old localization evidence; omit when unknown. |
| `state.required_skills` | All required skill IDs must have applicable guidance in the retained context. |
| `state.available_skills` | Currently executable skill IDs. Supply the actual runtime list; if omitted, only memory-declared skills are checked. |
| `state.anomaly`, `state.needs_recovery` | Optional booleans, default `False`. An anomaly requires both anomaly and recovery guidance. |
| `observation.conditions`, `state.conditions` | Flat mappings of names to finite JSON scalars. State values override observation values. Matching requires identical types and values; unknown conditions reject an entry. |

Condition facts are supplied explicitly; flags such as `needs_recovery` are not
automatically copied into the condition map. Unrecognized observation/state
fields are ignored. Object locations and geometric relations must come from
the current environment; memory text does not turn them into observed facts.

After each layer, the controller checks **only the entries that fit in the
returned context**. It stops when task match, localization, applicable skills,
and any required recovery evidence are present. Otherwise it escalates through
the four layers and returns `sufficient=False` with `missing` reasons. Budget
zero returns empty context without reading entry contents. The caller must
decide whether to obtain a new observation, use a fallback, or proceed; the
controller does not drive a robot or predict native task success.

## Budget, Compression, and Caching

By default, `counter_name="utf8_byte_upper_bound"` counts UTF-8 bytes as a
conservative upper bound for byte-based tokenizers. This is **not a measured
planner token count**. For model-specific measurements, inject an already loaded
local tokenizer, excluding special tokens:

```python
controller = MemCodexController(
    "memory/memcodex_v1",
    token_budget=1024,
    token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
    counter_name="my-tokenizer-pinned-revision",
)
```

The counter must be deterministic, return zero for empty text, and a positive
integer otherwise. The budget includes rendered entry headers, applicability
conditions, and separators. It covers the memory string only, not the complete
planner prompt, image tokens, history, or tool descriptions.

Compression filters and deduplicates complete entries. It never truncates an
instruction or its applicability conditions. Active recovery takes priority,
then layer order and retrieval rank. Retrieval retains at most eight candidates
per layer by default (`max_per_layer`). This is a greedy heuristic, not a global
minimum-token optimizer; a large early entry can crowd out later evidence.
Insufficient output remains explicitly marked even if a different combination
could have fit. Write small, self-contained entries and compare budgets using
native episode outcomes.

Use one controller per episode/thread. The in-memory LRU holds 128 results by
default (`cache_size=0` disables it). Keys include every normalized routing fact,
budget, counter name, and a filesystem metadata revision. Normal corpus edits,
additions, and deletions invalidate results; concurrent edits during retrieval
raise an error. Freeze the corpus for evaluation. Cache hits avoid JSON reads,
but still scan file metadata. They save retrieval work, not the tokens of any
context sent again to the planner. Do not mutate controller components or the
tokenizer configuration during use; construct a new controller when changing
configuration.

## Record Cost and Native Success

The repository-local [`profiling`](../../../profiling) package records JSONL
without copying memory text. Run it from the repository root; it is not included
in the installed `rpent` package. For each context actually injected into the
planner, call:

```python
from profiling.profiler import MemoryProfiler

profiler = MemoryProfiler("logs/memcodex-trial/profile.jsonl", variant="budget-1024")
result = controller.retrieve_with_trace(task_id, observation, state)
profiler.record(episode_id, result)
# After the episode, record the environment's actual boolean outcome once:
profiler.outcome(episode_id, native_success)
```

Use the same task/seed episode IDs across variants, with one variant per fixed
budget, tokenizer, corpus, and controller configuration. Record every injected
context, including cache hits. The measured cost is the sum of memory-context
tokens over the episode. Pass the chosen success threshold explicitly:

```bash
.venv/bin/python -m profiling.analyze logs/memcodex-trial/profile.jsonl --tau 0.95
```

The analyzer reports success among completed episodes, coverage, mean memory
tokens per episode, retrieval latency, and cache hits. It selects the cheapest
variant meeting observed τ only when all variants have identical, complete
episode cohorts and matching token counters. Missing outcomes do not count as
failures; incomplete or unpaired cohorts cannot produce a winner. Small-sample
success rates do not guarantee future success. Compare frozen configurations
on a separate evaluation set before concluding that token reduction preserves
task performance.

## M1: Convert, Validate, and Evaluate Fixed Memory

M1 uses the official `GPT_6_astra_low` corpus from `RLinf/RPent-memory`, revision
`47f301ef33bf8d0e8c770342f16b3c39c6a0e8c1`. The source is
`memory/libero/GPT_6_astra_low/`; `memory/original/` is the older GPT-5.5 corpus
and is **not** the M1 source. Install the optional local counter with
`uv pip install -e ".[memcodex]"`. Conversion itself requires no LLM or GPU.
Run from the repository root:

```bash
.venv/bin/python -m repro.convert_astra_memory \
  --source memory/libero/GPT_6_astra_low \
  --receipt memory/libero/GPT_6_astra_low.download-receipt.json \
  --catalog logs/liberopro-baseline-astra-low-s1-20261030-tmux/plan.json \
  --output memory/memcodex_v1

.venv/bin/python -m repro.validate_memcodex_corpus memory/memcodex_v1 \
  --source memory/libero/GPT_6_astra_low \
  --receipt memory/libero/GPT_6_astra_low.download-receipt.json
```

The converter preserves all 318 source files verbatim inside `raw` entries,
then extracts 1,109 entries across all four layers. It maps the 80 task IDs and
checks task-language identity. Summary holds task applicability; relational
holds scene localization and perception guidance; skills holds complete strategy,
parameter, failure and recovery sections and historical command order. Only
exact duplicates are merged. Recipe position payloads are omitted online;
parameters and Markdown retain their original qualifications and historical
warnings. Every entry has source references; `MANIFEST.json` records hashes,
transformations and static task/global links. The validator independently
rebuilds every entry from the pinned source, so edited content with an updated
hash still fails validation. It verifies transformation fidelity, not whether
historical advice is semantically correct for a new scene.

`libero_10_swap/t6` has no task-family or task-specific source. Its fixed
fallback contains all global leaves with an explicit missing-source notice.
This is shared guidance, not an invented successful task recipe. Raw archives
are available for offline audit; online integrity checks hash them, but fixed
retrieval never supplies their contents to the planner.

`MemCodexController(..., mode="fixed")` returns all task-matched summary,
relational and skill entries in stable order. It ignores observation/state,
top-k, early stopping and cache. Natural-language conditions remain visible for
scene verification. `sufficient=False` means sufficiency was not assessed in
this mode. The online default is 32,768 memory tokens, estimated with
`tiktoken/o200k_base`, including entry headers, conditions and the introduction.
Overflow is an error: M1 never silently truncates or drops instructions. These
are local estimates, not the planner's authoritative token usage.

To run one development episode without reusing evaluation seed 1:

```bash
.venv/bin/python -m repro.memcodex_m1 \
  --baseline logs/liberopro-baseline-astra-low-s1-20261030-tmux \
  --output logs/m1-development-goal-t0-s2 \
  --suite libero_goal_task --task 0 --seed 2
.venv/bin/python -m repro.liberopro_baseline run logs/m1-development-goal-t0-s2
```

For the paired 80-task seed-1 evaluation, omit `--suite`, `--task`, and `--seed`
and choose a fresh output directory. Preparation freezes current code hashes,
corpus hashes and the baseline's task assets, weights and execution limits.
`run` owns the two GPU services and cleans them up on completion; it preserves
infrastructure failures for diagnosis. It never overwrites the M0 batch.
The same production CLI also accepts `--memcodex-root memory/memcodex_v1
--memory-token-budget 32768` with `--planner codex`, without `--memory-repo`,
`--memory-dir` or `--explore`. The tested integration is CLI Codex evaluation.

Per episode, `m1-context.json` records the supplied text, selected IDs, budget,
source revision and preparation latency; `memory-view/MEMORY.md` is the read-only
fixed view. The profiler records its token count when the actual planner prompt
is submitted, plus any later memory reads. Preparation latency includes source
integrity checks and tokenization. Actual model input/output/reasoning tokens
and request timing retain their existing SDK/telemetry sources. Memory Tokens
count supplied memory text once per delivery, without multiplying by subsequent
history reuse; do not equate this with billed input tokens.

The planner can read this fixed view, robot guides and current artifacts through
MCP. Original experience, raw archives and previous runs are denied there.
Codex native shell tools are disabled for M1 using the documented
[`features.shell_tool`](https://learn.chatgpt.com/docs/config-file/config-reference)
setting. This is a tool-access control for the experiment, not an OS sandbox.
Other native tool capabilities are not covered by the file manager. Frozen
code/corpus checks and trace auditing remain necessary for evaluation.

M1 changes organization **and** fixed delivery/file access. Comparing it with
M0 measures that combined configuration; attributing gains solely to hierarchy
would require an additional control with the same delivery and access rules.
M0's mixed native-command outputs have incomplete memory-token attribution;
report coverage and use only paired attributable episodes for that metric.
Neither this implementation nor a successful smoke proves the success-rate
constraint τ. M2 progressive routing and M3 adaptive routing/cache are not
active in M1.

Run the focused offline contracts:

```bash
.venv/bin/python -m pytest tests/unit_tests/rpent/memory \
  tests/unit_tests/rpent/planner/test_codex_contracts.py \
  tests/unit_tests/robots/libero/test_memory_versions.py \
  tests/unit_tests/robots/libero/test_baseline_metrics_contracts.py -q
```

The generated corpus, validation reports, logs and weights are local ignored
artifacts. No deleted self-generated seed-0 experience is used.

## M2: indexed coarse-to-fine retrieval

M2 is opt-in (`mode="m2"`, required `index_path`). It reuses the unchanged M1
corpus and a separate deterministic metadata index. Build and validate it once:

```bash
.venv/bin/python -m repro.index_memcodex_memory memory/memcodex_v1 \
  --source memory/libero/GPT_6_astra_low \
  --receipt memory/libero/GPT_6_astra_low.download-receipt.json \
  --output memory/memcodex_m2_v2/INDEX.json
```

Online retrieval opens at most eight candidates per visited layer, selected from
task postings before reading bodies. It visits summary, relational, skills, raw
in that order. Complete entries are packed under the token budget; omitted
entries cannot satisfy coverage rules. Index/manifest changes and changed
selected bodies are rejected. Full source validation happens during preparation;
online requests do not rescan every layer. Unselected bodies are checked when read.

The LIBERO `retrieve_memory` tool checks the live `task_language`, latest step,
available primitives and failed-grasp status. `purpose="overview"` returns only
the summary; its sufficiency verdict covers this limited purpose. Execution
queries also require object localization and applicable skill guidance; recovery
queries require recovery guidance. Missing perception does not stop skill or raw
reads. The response separates `memory_missing` / `memory_read_complete` from
`state_missing` / `state_evidence_complete`. The latter only covers requested
evidence checks, not physical success or execution permission. `back_project` and
`segment` attach evidence IDs to successful current-step results. The planner
binds task object IDs to those IDs after inspecting images; motion invalidates
old evidence. This verifies evidence provenance, not the semantic identity of
the selected pixel or mask. No simulator object poses enter the controller.

Natural-language conditions are returned for the existing planner to review;
they do not count as applicable until the planner confirms a previously returned
entry at the current step. Generic historical warnings remain visible. Explicit
primitive failures and planner-reported visual issues trigger recovery coverage.
No extra judge LLM or result cache is used. Metadata remains loaded in memory;
this is not the M3 result cache. Sufficiency is not a success-rate prediction.

For the one task lacking its own family (`libero_10_swap/t6`), the index includes
two static same-family references ranked by task-entity overlap. These are
marked as analogies, require condition review, and never replace the current
instruction. Raw sources can be delivered on escalation; their old coordinates
and reset commands remain non-executable historical evidence.

```bash
.venv/bin/python -m repro.memcodex_m2 \
  --baseline logs/liberopro-baseline-astra-low-s1-20261030-tmux \
  --output logs/m2-development-object-t0-s2 \
  --index memory/memcodex_m2_v2/INDEX.json \
  --suite libero_object_task --task 0 --seed 2
.venv/bin/python -m repro.liberopro_baseline run logs/m2-development-object-t0-s2
```

For 80 paired seed-1 episodes omit suite/task/seed and choose a fresh output.
The production CLI accepts `--memcodex-root memory/memcodex_v1 --memcodex-mode m2
--memcodex-index memory/memcodex_m2_v2/INDEX.json --memory-token-budget 32768`.
Use Codex evaluation and a validated index; do not combine with exploration,
`--memory-dir`, or `--memory-repo`. M0 and default fixed M1 remain separate modes.

`m2-config.json` freezes the corpus/index identity. `m2-retrievals.jsonl` records
each query's layers, opened files, selected/omitted IDs, sufficiency gaps,
perception evidence, condition confirmations, token count and latency. These
sidecars are private to measurement. Memory Tokens count the entire serialized
tool response, including headers and JSON envelope, once per delivery. The
32,768-token limit is per retrieval, not per episode; repeated deliveries count
again. A 60,000-byte transport ceiling also applies, without cutting entries.
Retrieval latency excludes startup validation (included in end-to-end latency).
Task success, LLM calls and all original telemetry remain separately measured.

M2 changes retrieval, tool delivery, state checks and available fallback evidence.
Evaluate success and total episode memory tokens together against paired M1;
reduced context per call does not guarantee fewer calls or lower total tokens.

## Action-conditioned evidence baseline

For indexed M2, add `--memcodex-action-evidence` to the production CLI command
above. This selects a planner workflow that obtains geometry for the proposed
action's target or destination; later geometry can be deferred. Omit the flag
to retain the original M2 prompt. The optional `candidate` tool parameter is
available in either M2 workflow. M1 and non-indexed modes do not accept the flag.

For a reproducible batch, add `--action-evidence` to `python -m
repro.memcodex_m2`. The plan records `action_evidence`, hashes the robot evidence
adapter and core policy sources, and forwards the production flag to each run.
The CLI uses `manual_route`; an offline evolution driver and custom-policy file
loader are not included.

Use the object IDs returned by overview and current evidence IDs returned by
`back_project` or `segment`. For example, after observing the milk at step 0:

```json
{
  "step": 0,
  "purpose": "execute",
  "candidate": {
    "tool": "pi0_pick",
    "args": {"prompt": "pick up the milk"},
    "roles": {"target_object": "milk"},
    "expected_effect": "holding the milk"
  },
  "localizations": {"milk": "loc-0001"}
}
```

This is an example `retrieve_memory` argument object; replace the step and
evidence ID with actual outputs. The query does not require basket localization
or guidance for a later release. It validates the proposed tool's registered
argument schema, applies its defaults, and never executes the candidate.

| Candidate tool | Required roles | Additional contract |
| --- | --- | --- |
| `pi0_pick` | `target_object` | Current target geometry; identity remains a planner interpretation. |
| `move_to` | `held_object`, `destination` | Distinct task objects, current destination geometry, `gripper=1.0`, and a holding check. Only transport is supported. |
| `release` | `held_object`, `destination` | Distinct task objects, current destination geometry, holding and placement-readiness checks. |

Unsupported primitives use ordinary retrieval without `candidate` when needed.
An empty-gripper approach to an identified pick target reuses that target's
geometry and pick guidance, with the guides' motion checks; it does not force
localization of future destinations. `expected_effect` is an optional description; it
cannot override the tool's parameters or establish an observed fact.

Optional `visual_checks` records positive planner interpretations, for example
`{"fact":"holding","object_id":"milk","step":3,"artifact":"wrist_high.png"}`.
For release, `placement_ready` names the destination. Each check must name the
appropriate candidate object and a camera PNG recorded at the current step.
These checks remain `claimed`; pointing to an image does not independently
verify its contents. Geometric localization is `verified` only as a tool-derived
coordinate with planner-provided semantic binding. Identity is a separate claim.
The existing pick heuristic never establishes holding.

The response header and private retrieval trace separate:

- `memory_read_complete`: retained entries cover the candidate skill and any
  required recovery guidance.
- `action_preconditions_met`: every fixed action fact is independently verified.
- `action_missing_facts` and `action_unverified_facts`: absent evidence versus
  existing planner claims requiring visual judgment.
- `routing_decision`: `STOP`, `NARROW`, `DESCEND`, `OBSERVE`, or `REVIEW`, with a reason.

Because this adapter has no independent identity/holding/placement verifier,
normal candidate queries retain unverified facts and report
`action_preconditions_met=false`, even after visual checks. `sufficient` requires
both completeness fields to be true. Do not repeat retrieval to turn a claim
into verification: inspect observations and use the existing planner-controlled
primitive workflow. These fields grant no execution permission. `OBSERVE`
reports a gap; it does not capture a frame or return extra images. Automatic
primitive image returns, VLA inputs and planner scheduling are unchanged.

Use `applicable_entries` and `rejected_entries` for previously returned memory
entries. The lists must be disjoint and are scoped to the current step and
normalized candidate fingerprint. Changing a tool parameter or role invalidates
the prior review scope. Explicitly rejected candidates are skipped so retrieval
can proceed to raw experience. A query without a candidate has a separate scope.

### Replacing the pure routing function

[`policy.py`](policy.py) defines frozen `ActionQuery`, `RoutingContext` and
`RoutingDecision` objects. A function receives only immutable candidate and
coverage data and returns a read decision. It must perform no I/O, model calls,
state mutation or primitive execution. The controller does not sandbox custom
Python code; load only reviewed functions. Supply one through the Python API:

```python
from rpent.memory.memcodex import MemCodexController, manual_route

controller = MemCodexController(
    "memory/memcodex_v1",
    token_budget=32768,
    mode="m2",
    index_path="memory/memcodex_m2_v2/INDEX.json",
    route=manual_route,  # Replace with a reviewed pure function of the same signature.
    route_name="manual-v1",
)
```

Pass a robot-adapter-validated `action_query` to `retrieve_with_trace`. The fixed
runtime computes coverage from retained entries and validated facts, outside
the routing function. An unsupported `STOP` cannot mark missing or claimed facts
as verified. Nor can `OBSERVE` bypass missing memory before raw is exhausted.
`NARROW` selects the runtime-proposed `narrow_ids` for the next layer; a custom
policy cannot inject entry IDs or change applicability. The same budget and
whole-entry packing rules apply. Use identical action workflows and budgets
when comparing manual and replacement policies; measure complete episode token
usage, latency and native success separately from retrieval coverage.

### Source-linked descent and narrowing

New indexes use schema 3 and store `source_links`, bounded `issue_tags`, and
source-derived `recovery_cases`.
Links connect entries citing the same source file to entries in deeper layers;
line references remain available for audit. These are provenance links, not a
semantic relation graph or proof of applicability. The reader verifies the
links against entry provenance. Frozen schema-1 and schema-2 indexes remain readable and
validatable: links are derived from metadata without rewriting the corpus, but
Failure-topic tags are available in schema 2 and 3; the current rebuild emits
schema 3, including the structured recovery cases. Use a new output path as above;
the builder refuses to overwrite a different existing index.

Before descending, `NARROW` can restrict the next layer to source-linked entries
and then current primitive/recovery topics. Topic hints come from the reported
issue and offline-indexed text; they require no model call and establish no
current facts. If retained evidence is insufficient, the reader widens to other
eligible task candidates in the same layer, then descends as needed. Rejected
entries, unmet structured conditions and task boundaries remain enforced.
A full skill awaiting applicability review may return `REVIEW`; it is not
counted as usable merely because its source is reachable.

The private `routing_steps` trace records `NARROW`, `WIDEN`, descent and final
decisions, including selected candidate IDs and missing memory/state evidence.
The eight-file cap applies to both passes together; `LIMIT` records unread
candidates when it binds. Budget-omitted bodies never count as coverage. Raw is
a fallback, not an unconditional read, and missing current localization never
becomes valid from historical coordinates. Once guidance is complete, state
gaps request observation rather than more history. This does not implement an
evolution loop, execute a skill, or update the frozen experience package.

Offline acceptance checks (no GPT, VLA or simulator):

```bash
.venv/bin/python -m pytest tests/unit_tests/rpent/memory \
  tests/unit_tests/robots/libero/test_action_evidence_contracts.py -q
```


### Specific failures and drawer evidence

Recovery sufficiency checks the particular failure (`object_slipped`,
`grasp_not_closed`, `wrong_drawer`, `contact_stalled`, `insertion_blocked`, or
`release_stuck`), with object category and phase when supplied. A source passage
must describe both the symptom and a response; a generic recovery tag is not
coverage. These conservative text rules can miss useful guidance. Unrecognized
issues are reported as uncovered. Insufficient evidence widens and descends to
raw within the existing file/token limits, then reports any remaining gap.
Bodies and conditions remain unchanged; no extra LLM judge or online update is
used. Current state, historical coverage, and native task success stay separate.

`retrieve_memory` accepts `failure={kind, object_id, phase, step, artifact}`.
It rejects stale/unrecorded images and unknown objects. This is an attributed
planner observation, not a simulator assertion. Servo residual alerts compare
against that command's tolerance and budget; contact `success=false` alone does
not establish intermediate failure.

Drawer candidates use `pi0_doubled`, `move_pose`, or `move_to`,
`roles={handle: "bottom_drawer_handle"}`, and
`drawer={target_level: "bottom", operation: "open", stage: "actuate",
direction: "outward"}`. Other supported levels are `top` and `middle`.
Use `context_objects` from overview; `drawer_handle` aliases only the task's
own goal level. The contract requires fresh handle geometry, level and access
claims; an actuating push/pull additionally requires a contact claim.
`visual_checks` uses `handle_level` with the level as `value`, or `access_clear`
and `contact_ready` with `value=true`. Claims remain unverified by design.

`drawer_progress` compares consecutive saved images using `previous_step`,
`step`, `artifact`, `target_level`, `observed_level`, `expected_direction`, and
`observed_direction`. A wrong level, reversed direction, or reported no movement
requests recovery. Unknown evidence never certifies progress. This does not
supply a grasp controller or automatically interpret camera pixels.

### Frozen initial scenes for paired runs

`seed` and a pruned state alone do not restore fixed fixture poses. Record a
private snapshot and verify it in an independent environment process before
running the planner (requires the installed LIBERO-PRO/EGL environment):

```bash
python -m repro.freeze_libero_scene record /absolute/scenes/goal-t0.npz \
  --suite libero_goal_task --task 0 --seed 1 --cuda-device 1 \
  --receipt /absolute/scenes/goal-t0-record.json
python -m repro.freeze_libero_scene restore /absolute/scenes/goal-t0.npz \
  --suite libero_goal_task --task 0 --seed 1 --cuda-device 1 \
  --receipt /absolute/scenes/goal-t0-verified.json
```

Pass `--initial-scene /absolute/scenes/goal-t0.npz` to each evaluation arm.
This opt-in adapter seeds global samplers in the worker, freezes MuJoCo
integration state and fixed-body transforms before ordinary settling, and
rejects changed task/native init state, simulator version or other model arrays.
After settling it checks recorded raw observations, both RGB-D cameras and
calibration before planner control. Physics, depth and calibration must match
exactly. RGB permits at most a one-level uint8 rounding difference in 0.01% of
color channels; the receipt records any such difference. Snapshots remain outside the planner's file
boundary. The run writes only a hash receipt to `initial-scene-validation.json`.
Recording cannot overwrite an existing snapshot. Use the same validated snapshot
and simulator dependencies for both arms; equal initial state does not make an
online planner's decisions deterministic.
