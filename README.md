# AgenticRobo

基于 RPent / HarnessVLA 的机器人记忆检索实验。当前代码包含 M1 固定分层记忆和
M2 按任务索引逐层检索，以及 LIBERO / LIBERO-PRO 的在线接入、测量和配对评测脚本。

## 代码入口

- [M1 / M2 中文说明](RPent/rpent/memory/memcodex/README_zh.md)
  / [English](RPent/rpent/memory/memcodex/README.md)
- [Memory Controller](RPent/rpent/memory/memcodex/controller.py)：预算和检索模式。
- [M2 逐层检索](RPent/rpent/memory/memcodex/coarse.py)：摘要 → 关系 → 技能 → 原始经验。
- [LIBERO 在线接入](RPent/robots/libero/memcodex_online.py)：当前步骤的定位凭据、条件核对和恢复状态。
- [语料转换](RPent/repro/convert_astra_memory.py)、[校验](RPent/repro/validate_memcodex_corpus.py)
  和[索引构建](RPent/repro/index_memcodex_memory.py)。
- [M2 配对清单准备](RPent/repro/memcodex_m2.py)和[指标采集](RPent/profiling)。

M2 默认每次检索最多提供 32,768 个估算 Token，并受 60,000 字节传输上限约束。
它使用规则检查信息覆盖，不增加判断用的 LLM 调用，也不启用 M3 结果缓存。
是否保持成功率、降低整回合 Token 和耗时，需要完成配对评测后判断。

## 本地检查

在 Python 3.11 环境中，从 `RPent/` 目录安装和运行：

```bash
cd RPent
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,memcodex]'
python -m pytest tests/unit_tests/rpent/memory -q
```

完整离线回归命令为 `python -m pytest tests/unit_tests -v`。本次同步前使用已有
Python 3.11 环境验证：**1454 passed, 11 skipped**；`pre-commit run --all-files`
通过。未重新启动 GPU 评测，也未验证完整 Python 版本矩阵。

实际运行 LIBERO-PRO 还需安装对应模拟器和模型依赖、准备权重与任务资源、配置规划器
登录，见 [RPent 安装说明](RPent/docs/source-zh/rst_source/get_started/quickstart.rst)
及 [LIBERO 指南](RPent/docs/source-zh/rst_source/simulators/libero.rst)。
配对脚本还依赖已有 M0 的 `plan.json` 和 `METRICS.md`；文档中的历史日志路径需替换为
实际准备的基线目录。经验包、转换语料、模型权重、运行日志和评测结果均需单独准备。

## 版本与来源

`RPent/` 保留当前实验所用的 RPent 源码基础，源仓库为
[RLinf/RPent](https://github.com/RLinf/RPent)，本地基础提交为
`034cbdae1c2c976bf4c7e6a79cd886395b3db086`，叠加截至 2026-11-02 的 M1 / M2 实现。
离线测试使用确定性的 tokenizer 替身，避免测试时联网下载编码表；生产计数逻辑保持原样。
这里只同步源码、测试和文档，不包含原仓库 Git 历史。

转换器对应官方 `RLinf/RPent-memory` 的 `GPT_6_astra_low`，固定版本为
`47f301ef33bf8d0e8c770342f16b3c39c6a0e8c1`。仓库仅附带人工构造的测试语料，
不附带官方经验包或此前自建的 seed 0 经验。

RPent 源码遵循 [Apache-2.0](RPent/LICENSE)，保留原版权和贡献说明。
