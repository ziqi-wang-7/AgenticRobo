# 带预算的 Memory Controller

[English](README.md)

`MemCodexController` 在明确的记忆上下文预算内，逐层检索任务经验。它是独立的可选模块；默认沿用现有 [`MemoryManager`](../manager.py)，LIBERO 的 Codex 评测可显式启用下文的 M1 固定模式或 M2 索引模式。此实现借鉴 MemCodex 的逐层检索思路，不依赖旁边的 `memcodex` 项目，也不调用 LLM。

## 运行本地示例

在 RPent 根目录使用现有的 Python 3.11+ 环境。无需安装新依赖，不需要 GPU、模型权重或凭据。[示例语料](../../../tests/fixtures/memcodex_demo) 是人工构造的测试数据，不是 LIBERO 轨迹经验。

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

首次检索依次访问 `summary`、`relational`、`skills` 后停止；异常恢复会进一步读取 `raw`。`retrieve()` 返回字符串，`retrieve_with_trace()` 返回不可变的结果对象，其中包含相同的上下文、预算统计、选中和省略的条目 ID、读取文件数、缺失信息、语料版本、缓存状态和检索耗时（毫秒）。

## 记忆与输入格式

在 `summary/`、`relational/`、`skills/`、`raw/` 下存放 UTF-8 JSON 文件，每个文件一个条目，允许子目录。文件所属目录决定层级；同一层内的 ID 必须唯一。每条经验必须显式提供 `applies_when`；无条件适用时填写 `{}`。

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

必填字段是 `id`、`task_ids`、`text` 和 `applies_when`，另外三个字段是可选列表。`condition_text` 可保留自然语言适用条件，`source_refs` 可记录原文行号引用。逐层路由保守地拒绝带有未解析 `condition_text` 的条目；固定模式将这些条件完整交给规划器核对。未知字段、格式错误会产生带文件路径的 `ValueError`。单文件最多 1 MiB，层级目录内不接受符号链接。控制器读取此 JSON 格式，不能直接读取原始 Markdown 经验包。

`provides` 必须反映正文实际包含的内容：`task` 表示任务目标和策略，`skill` 表示可执行的技能指导，`anomaly` 表示对应异常的处理依据，`recovery` 表示恢复指导。仅在 `skill_ids` 中提到技能名不算提供技能指导。转换语料时需要核对这些元数据，控制器无法验证经验内容本身是否正确。

| 输入 | 含义 |
| --- | --- |
| `task` | 规范任务 ID 或完整指令，与 `task_ids` 精确匹配。`*` 允许共享经验，但不能单独满足当前任务的专属经验要求。 |
| `observation.localized_objects` | 当前感知已经定位的物体名称；只传图像不会被视为已经定位。 |
| `state.required_objects` | 需要定位的物体；省略时采用保留条目中的物体名称。 |
| `state.objects_localized` | 可选的权威布尔值；`False` 覆盖旧定位信息，未知时省略。 |
| `state.required_skills` | 保留的上下文必须覆盖这里要求的所有技能。 |
| `state.available_skills` | 运行时可执行的技能 ID。建议传入实际列表；省略时只检查经验声明的技能。 |
| `state.anomaly`、`state.needs_recovery` | 可选布尔值，默认 `False`；发生异常时同时要求异常处理和恢复指导。 |
| `observation.conditions`、`state.conditions` | 扁平条件映射，值为有限的 JSON 标量；state 覆盖 observation。匹配要求类型和值都相同，条件未知则该条经验不适用。 |

条件需要显式传入；`needs_recovery` 等标志不会自动写入条件映射。未识别的 observation/state 字段会被忽略。物体位置和几何关系应来自当前环境，经验中的描述不能充当当前观测事实。

每访问一层，都只对**压缩后实际保留的条目**判断是否足够。任务匹配、物体定位、适用技能及所需恢复依据齐全时提前停止；否则继续下一层。遍历后仍不足时，返回 `sufficient=False` 和 `missing` 原因。预算为零时返回空上下文，不读取条目内容。调用方据此决定补充观测、采用回退方案或继续执行；控制器本身不控制机器人，也不预测环境是否成功。

## 预算、压缩和缓存

默认的 `counter_name="utf8_byte_upper_bound"` 用 UTF-8 字节数，作为基于字节的 tokenizer 的保守上界。它**不是规划模型的真实 Token 测量值**。若要精确计数，请注入已加载的本地 tokenizer，并排除特殊 Token：

```python
controller = MemCodexController(
    "memory/memcodex_v1",
    token_budget=1024,
    token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
    counter_name="my-tokenizer-pinned-revision",
)
```

计数函数必须是确定性的，空字符串计数为零，其他字符串返回正整数。预算包含条目标题、适用条件和分隔符；只约束返回的记忆字符串，不包括完整规划器提示、图像、历史对话和工具说明。

压缩方式是筛选、去重和保留完整条目，不截断操作指令或适用条件。当前异常恢复内容优先，然后按层级和检索排名填入预算。每层默认最多保留八个候选（`max_per_layer`）。这是贪心启发式，并非全局最少 Token 求解器：较大的早期条目可能挤掉后续信息，即使其他组合能够放入预算，也会如实标为不足。建议将经验整理成小而完整的条目，再用真实回合结果比较不同预算。

每个回合或线程使用独立控制器。内存 LRU 缓存默认保留 128 个结果，`cache_size=0` 可关闭。键包含归一化后的全部路由条件、预算、计数器名称和文件元数据版本。正常的语料修改、增删会使缓存失效；检索过程中修改语料会报错。评测时应固定语料。命中缓存会省去 JSON 读取，但仍扫描文件元数据；重复发给规划器的上下文仍消耗 Token。使用过程中不要修改控制器组件或 tokenizer 配置；需要更换配置时创建新控制器。

## 统计成本与真实成功率

仓库内的 [`profiling`](../../../profiling) 模块将统计写入 JSONL，不复制记忆正文。请从仓库根目录调用；该模块不随 `rpent` 安装包分发。每次真正向规划器注入上下文时记录：

```python
from profiling.profiler import MemoryProfiler

profiler = MemoryProfiler("logs/memcodex-trial/profile.jsonl", variant="budget-1024")
result = controller.retrieve_with_trace(task_id, observation, state)
profiler.record(episode_id, result)
# 回合结束后，仅记录一次环境给出的真实布尔结果：
profiler.outcome(episode_id, native_success)
```

不同方案采用相同任务和 seed 的回合 ID；每个 variant 固定预算、tokenizer、语料与控制器配置。所有实际注入的上下文都计入，包括缓存命中。回合成本是各次注入的记忆 Token 总和。分析时显式传入成功率阈值，例如：

```bash
.venv/bin/python -m profiling.analyze logs/memcodex-trial/profile.jsonl --tau 0.95
```

分析结果包含已完成回合的成功率、完成覆盖率、每回合平均记忆 Token、检索耗时和缓存命中数。只有各方案的回合集合完全一致、全部完成且 Token 计数口径一致时，才从达到观测阈值 τ 的方案中选择记忆成本最低者。缺失结果不会算作失败；缺失或未配对的数据不能选出最优方案。小样本成功率不能保证后续成功率；应在独立评测集上比较固定配置，才能判断 Token 节省是否保持了任务表现。

## M1：转换、校验、固定读取与评测

M1 使用官方 `RLinf/RPent-memory` 的 `GPT_6_astra_low`，固定版本
`47f301ef33bf8d0e8c770342f16b3c39c6a0e8c1`。源目录是
`memory/libero/GPT_6_astra_low/`；`memory/original/` 是旧的 GPT-5.5 经验，
**不是本次转换来源**。可用 `uv pip install -e ".[memcodex]"` 安装可选的本地
Token 计数器。转换本身不调用 LLM，也不需要 GPU。在仓库根目录运行：

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

转换器在 `raw` 条目中逐字保留全部 318 个源文件，四层合计生成 1,109 条记录，
映射 80 个任务并核对任务语言。summary 存任务适用模式；relational 存场景定位和
感知经验；skills 保留完整的策略、参数、失败、恢复段落和历史动作顺序。
只合并完全相同的内容。在线 recipe 省略位置参数；其他参数和 Markdown 保留原有
限定条件及历史经验提示。每条记录带原文引用，`MANIFEST.json` 保存哈希、转换规则
和固定任务关联。校验器从固定源包独立重建每个条目；即使修改正文后重新填写哈希，
也不能通过校验。它验证转换是否忠实，不能保证旧经验适用于新场景。

官方源包缺少 `libero_10_swap/t6` 的任务族和任务专属经验。该任务固定使用全部
global 条目，并显式提示缺少专属经验；不会伪造成功案例。raw 只用于离线审计，
在线完整性检查会计算其哈希，但固定检索不会把正文交给规划器。

`MemCodexController(..., mode="fixed")` 按稳定顺序返回该任务全部匹配的
summary、relational、skills 条目，不使用 observation/state、top-k、提前停止或
缓存。自然语言条件完整保留，由规划器结合场景检查。此模式的 `sufficient=False`
表示未执行充分性判断。在线默认预算为 **32,768 个记忆 Token**，用
`tiktoken/o200k_base` 估算，包含标题、条件和引导文字。超出预算直接报错，不静默
截断或删除指令。这是本地估算值，不是规划模型返回的真实用量。

先运行一个开发回合，避免使用正式评测的 seed 1：

```bash
.venv/bin/python -m repro.memcodex_m1 \
  --baseline logs/liberopro-baseline-astra-low-s1-20261030-tmux \
  --output logs/m1-development-goal-t0-s2 \
  --suite libero_goal_task --task 0 --seed 2
.venv/bin/python -m repro.liberopro_baseline run logs/m1-development-goal-t0-s2
```

要准备 80 个任务 × seed 1 的配对评测，省略 `--suite`、`--task` 和 `--seed`，
并选择新的输出目录。准备阶段固定当前代码与语料哈希，复用 M0 的任务资源、权重
和执行上限。`run` 管理两张 GPU 上的服务，并在结束时清理自己的进程；基础设施
错误保留现场，等待诊断，不覆盖 M0。也可在原生产 CLI 使用
`--memcodex-root memory/memcodex_v1 --memory-token-budget 32768`，规划器选择
`--planner codex`，不传 `--memory-repo`、`--memory-dir` 或 `--explore`。
本次验证范围是 CLI 的 Codex 评测。

每回合的 `m1-context.json` 保存实际提供的正文、条目 ID、预算、来源和准备耗时；
`memory-view/MEMORY.md` 是只读固定视图。Profiler 在上下文真正提交给规划器时
计数，随后重复读取记忆也计入。准备耗时包含语料完整性检查与分词；模型真实的
输入、输出、推理 Token 和请求耗时仍取自 SDK/遥测。Memory Tokens 按每次提供的
记忆正文计数，不乘以后续请求复用历史的次数，因此不能直接等同于计费输入 Token。

规划器通过 MCP 只能读取固定视图、机器人指南和本回合产物；原经验包、raw 和旧
运行目录在此入口被拒绝。M1 用 Codex 文档中的
[`features.shell_tool`](https://learn.chatgpt.com/docs/config-file/config-reference)
关闭原生命令工具。这是实验的工具访问控制，不是操作系统级沙箱，文件管理器也
不覆盖其他原生工具能力；评测仍需校验固定代码、语料并审计调用轨迹。

M1 同时改变了记忆组织、固定提供方式和文件访问方式；与 M0 对比测到的是这套
组合配置。若要把收益只归因于分层，需要增加提供方式和访问规则一致的对照。
M0 部分原生命令输出混合了记忆与其他内容，无法完整归因 Memory Tokens；该指标
应报告覆盖率，只在双方可归因的配对回合上比较。实现完成或冒烟成功都不能证明
成功率约束 τ 已达标。M2 逐层检索与 M3 自适应路由、缓存均未在 M1 中启用。

运行离线回归测试：

```bash
.venv/bin/python -m pytest tests/unit_tests/rpent/memory \
  tests/unit_tests/rpent/planner/test_codex_contracts.py \
  tests/unit_tests/robots/libero/test_memory_versions.py \
  tests/unit_tests/robots/libero/test_baseline_metrics_contracts.py -q
```

转换语料、校验报告、日志和权重是被 Git 忽略的本地产物。本次不使用此前已删除
的自建 seed 0 经验。

## M2：索引驱动的逐层检索

显式选择 `mode="m2"` 并提供 `index_path`。M2 复用未改动的 M1 语料，另建确定性的
元数据索引；先运行转换校验和索引校验：

```bash
.venv/bin/python -m repro.index_memcodex_memory memory/memcodex_v1 \
  --source memory/libero/GPT_6_astra_low \
  --receipt memory/libero/GPT_6_astra_low.download-receipt.json \
  --output memory/memcodex_m2_v1/INDEX.json
```

在线先查任务索引，再打开候选正文，每层最多读取 8 条，顺序为摘要→关系→技能→原始
经验。只按完整条目装入预算，被省略的条目不能参与充分性判断。索引、清单变化和选中
文件的哈希变化都会报错。准备阶段完整校验来源；在线不扫描整层，未选中的正文在实际
读取时校验。

LIBERO 的 `retrieve_memory` 工具读取本回合真实的任务指令、最新步骤、可用原语和抓取
失败信息。`purpose="overview"` 只请求摘要，其充分性也仅针对摘要；执行请求还检查
物体定位和技能指导，恢复请求还检查恢复指导。缺少观测时在关系层停止，要求补观测。
成功的当前步骤 `back_project`／`segment` 输出附带定位凭据，规划器观察图像后将任务
物体 ID 绑定到凭据；运动后旧凭据失效。这验证了定位数据的来源，不能证明选中的像素
或掩码就是正确物体。控制器不读取仿真器的物体真值坐标。

自然语言适用条件交给当前规划器核对；只有当前步骤已经返回、随后被确认适用的条目
才参与充分性判断。通用历史经验警告仍然保留。原语显式失败和规划器报告的视觉异常
触发恢复检查，不增加专门判断充分性的 LLM 调用，不使用结果缓存。元数据常驻内存
不等于 M3 的检索结果缓存；充分性规则也不预测任务成功率。

官方缺少专属经验的 `libero_10_swap/t6` 索引含两个按任务实体重合度选出的同类参考。
它们显式标为不同任务的类比，需要核对适用条件，不能替代当前指令。升级到 raw 层时
可以返回原始经验，但旧坐标与 reset 指令只作历史证据，禁止照搬执行。

```bash
.venv/bin/python -m repro.memcodex_m2 \
  --baseline logs/liberopro-baseline-astra-low-s1-20261030-tmux \
  --output logs/m2-development-object-t0-s2 \
  --suite libero_object_task --task 0 --seed 2
.venv/bin/python -m repro.liberopro_baseline run logs/m2-development-object-t0-s2
```

准备 80 回合 seed 1 配对评测时，省略 suite/task/seed 并选新目录。生产 CLI 对应参数是
`--memcodex-root memory/memcodex_v1 --memcodex-mode m2 --memcodex-index
memory/memcodex_m2_v1/INDEX.json --memory-token-budget 32768`。使用 Codex 评测和已
校验的索引，不与探索、`--memory-dir`、`--memory-repo` 混用。M0 和默认的固定 M1
仍是独立模式。

`m2-config.json` 固定语料与索引身份；`m2-retrievals.jsonl` 记录访问层、实际打开的
文件、选中与省略条目、充分性缺项、定位凭据、条件确认、Token 数和耗时。这两份测量
文件不向规划器开放。Memory Tokens 按完整序列化工具响应计数，包含标题和 JSON
外壳；32,768 是每次检索的预算，不是整回合预算，重复提供的内容重复计数。另有
60,000 字节传输上限，均不截断条目。检索耗时不含启动校验，后者计入端到端耗时。
任务成功、LLM 调用次数和原有指标继续独立采集。

M2 同时改变检索、工具提供方式、状态检查和可用的补充经验。需与 M1 配对测量成功率
及整回合记忆 Token；单次上下文变短不保证调用更少，也不保证总 Token 更低。
