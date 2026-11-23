# Flash-Evolve：快慢双系统程序记忆自进化

`--planner flash_evolve` 将 **System-1 程序记忆回放（Flash）** 与 **System-2 慢系统大模型规划器（Codex / API / Claude Code）** 结合为单回合（Single-Attempt、无 Reset）闭环自进化架构：

1. **自动冷启动与引导（Bootstrap）**：
   * 优先从 `--flash-memory-dir` 加载该任务的 `*_plan.json` 与 `*_anchors.json`。本地语料默认使用 `<memory_dir>/flash_evolved`。HF 语料默认使用所选快照中版本目录旁的 `<version>.flash_evolved`，按语料版本和修订号隔离。
   * 若缺少计划和锚点，先查找语料的 `flash/` 和 `task_card/`。若仍缺失，再从 `task-specific/` 中 seed 0 的审计文件及对应配方编译。使用 `--no-flash-evolve` 可冻结持久记忆，此时初始编译只使用临时目录。
   * 若完全没有 seed-0 专属经验（例如 `libero_10_swap/t6`），自动回退到慢系统求解，并在求解成功后将轨迹编译为新的 Flash 程序记忆。
2. **零 Token 快路径与已学恢复分支（System-1 Fast Execution）**：
   * 使用锚点相对偏移（`anchor + offset`）执行操作；支持 **Molmo + SAM3** 双定位，也在不传 `--molmo-endpoint` 时无缝降级为 **SAM3-only** 定位，直接兼容现有 `GPT_6_astra_low` 的两卡部署。
   * 实时检测五类物理异常：`anchor_missing`、`pick_failed`、`motion_stalled`（复用 `motion_alert` 伺服残差检测）、`action_error`、`plan_incomplete`。
   * 触发异常时，按异常类型、步骤、锚点和空间范围筛选已学的 `recovery_branches`，优先尝试平滑成功率较高的分支。分支累计尝试至少 3 次且从未成功时，后续回放跳过该分支。环境耗尽步数时，回合以 `flash_truncated` 结束。
3. **原地慢系统接管与程序修补（Just-in-Time Handover & Procedural Patching）**：
   * 若快路径及已有恢复分支未能使环境终止，**不重置仿真环境（No Reset）**，将当前实时步骤、定位锚点表、已执行步骤和失败诊断打包为接管提示，交给慢系统（如 Codex，完整保留 M0 原生经验包读取能力）原地救场。
   * 在原触发位置接管并成功后，系统先通过空事件折叠过滤器（`fold_null_events`）剔除报错/中断指令、失败空抓及其无效空夹爪前摇、被替代的伺服卡死步、微幅抖动（< 1.5 cm）和空夹爪往返绕路（同时保留接触型 `pi0_pick` 与有效避障路点），再用实时锚点计算偏移并在文件锁内将恢复分支写入 `*_plan.json`。若已尝试的分支改变了现场，回合仍按环境的实际结果计分，后续救援动作只适用于改变后的状态。

---

## 快速运行命令

### 1. 单任务开发验证（直接复用现有 pi05 + sam3 环境，无需 Molmo）

```bash
.venv/bin/python -m rpent.cli.main \
  --robot libero --libero-type pro \
  --suite libero_spatial_task --task 0 --seed 2 \
  --planner flash_evolve --flash-slow-planner codex \
  --model gpt-6-astra --reasoning-effort low \
  --memory-profile hf --memory-version GPT_6_astra_low \
  --flash-memory-dir memory/libero/flash_evolved_dev \
  --no-auto-merge-memory
```

若部署了 Molmo 服务，可追加 `--molmo-endpoint http://127.0.0.1:<port>` 开启双阶段视差精细定位。

### 2. LIBERO-PRO 80 任务配对评测（或跨 Seed 连续进化流评测）

```bash
# 单 seed 配对评测（对齐 M0 seed 1）
.venv/bin/python -m repro.flash_evolve prepare \
  --baseline logs/liberopro-baseline-astra-low-s1-20261030-tmux \
  --output logs/flash-evolve-s1 \
  --seeds 1

# 多 seed 流式进化评测（seed 1 -> seed 2 -> seed 3，观察零 Token 分支命中率与成功率随轮次提升）
.venv/bin/python -m repro.flash_evolve prepare \
  --baseline logs/liberopro-baseline-astra-low-s1-20261030-tmux \
  --output logs/flash-evolve-stream-s1-3 \
  --seeds 1 2 3

# 复用原有基线调度器运行
.venv/bin/python -m repro.liberopro_baseline run logs/flash-evolve-s1

# 查看各执行模式、各 seed 的原生成功率及实际输入输出 token 数
.venv/bin/python -m repro.flash_evolve summarize logs/flash-evolve-s1
```

### 3. 离线单元测试

```bash
.venv/bin/python -m pytest tests/unit_tests/robots/libero/test_flash_evolve*.py -q
```
