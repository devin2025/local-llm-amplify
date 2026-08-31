# Local LLM Amplifier · 本地模型潜能榨取

> **Goal:** Not "beat GPT-5.6" — that is decided by model weights. The real goal is: **raise the effective usable intelligence of the same local model 2–4x** with context engineering, tuned sampling parameters, test-time compute, and tooling. Same machine, same model, different results.
>
> **目标不是让本地模型超过 GPT-5.6——那是模型权重决定的。真正的目标是：把同一个本地模型的有效可用智能拉高 2–4 倍**——靠上下文工程、采样参数调优、推理时计算扩展和工具链。同一台机器、同一个模型，表现可以差一个量级。

A skill (toolchain) for squeezing maximum capability out of locally deployed LLMs (llama.cpp / ik_llama.cpp / LM Studio / Ollama / vLLM). Designed to be consumed by **agents** (Hermes, deepseek-harness, WorkBuddy) as well as humans.

面向本地部署模型的潜能榨取工具链。设计为可被 **agent**（Hermes、deepseek-harness、WorkBuddy）和人类共同使用。

---

## Features · 特性

| EN | CN |
|---|---|
| 11 task-tuned sampling profiles (code / math / writing / scripts / judge / thinking models) | 11 个按任务调优的采样参数档位（代码/数学/写作/脚本/裁判/推理模型） |
| Self-consistency voting (`vote.py`) with confidence output | 自洽投票（`vote.py`），带置信度输出 |
| 4-stage pipeline: plan → solve → critique → revise (`pipeline.py`) | 四阶段流水线：规划→求解→批判→修订（`pipeline.py`） |
| Benchmark harness (`bench.py`) with CSV/JSON/HTML reports | 参数/模型基准评测（`bench.py`），出 CSV/JSON/HTML 报告 |
| Cross-backend capability degradation (LM Studio auto-drops unsupported params) | 跨后端能力降级（LM Studio 自动丢弃不支持的参数） |
| Hallucination control playbook | 幻觉抑制手册 |
| Story / emotion video-script methodology (志怪/历史/叙事) | 故事/情绪类视频脚本方法论 |
| **Zero third-party dependencies** — stdlib only, works offline | **零第三方依赖**——纯标准库，可离线运行 |

---

## Quick Start · 快速开始

```bash
cd scripts
# 1. Edit config.json — set your endpoint port + model name
# 2. Test connectivity
python llm.py --user "1+1=? 只输出数字" --profile math_logic

# 3. Use it
python vote.py --task-file q.txt --n 8 --extract boxed          # voting
python pipeline.py --mode code --task-file req.txt --verify tests.py  # 4-stage pipeline
python bench.py --taskset tasksets/math_basic.json --sweep temperature=0.2,0.4,0.6  # benchmark
```

```bash
cd scripts
# 1. 改 config.json——填你的端点端口和模型名
# 2. 连通性测试
python llm.py --user "1+1=? 只输出数字" --profile math_logic

# 3. 使用
python vote.py --task-file q.txt --n 8 --extract boxed          # 投票
python pipeline.py --mode code --task-file req.txt --verify tests.py  # 四阶段流水线
python bench.py --taskset tasksets/math_basic.json --sweep temperature=0.2,0.4,0.6  # 基准评测
```

**Python ≥ 3.10. No `pip install` required.**

**要求 Python ≥ 3.10，无需安装任何第三方包。**

---

## Toolbox · 工具

| Script | Purpose · 用途 |
|---|---|
| `llm.py` | Unified client — single call, N samples, profile presets. 统一客户端——单次调用、多候选采样、档位预设 |
| `vote.py` | Self-consistency voting with optional code verification. 自洽投票，可选代码真跑验证 |
| `pipeline.py` | plan → solve → critique → revise. 四阶段推理流水线 |
| `bench.py` | Sweep params / compare models, HTML report. 参数扫描 / 模型对比，HTML 报告 |
| `selftest.py` | Offline regression tests (no server needed). 离线回归测试（不需要模型服务） |

All scripts: CLI-only, `--json` / `--quiet`, diagnostics on stderr, exit codes `0/1/2`, stdlib-only.

所有脚本：纯 CLI、`--json` / `--quiet`、诊断走 stderr、退出码 `0/1/2`、仅标准库。

---

## Sampling Profiles · 参数档位速查

| Profile | temp | top_p | top_k | min_p | repeat | Use · 用途 |
|---|---|---|---|---|---|---|
| `code_generate` | 0.25 | 0.90 | 40 | 0.02 | 1.05 | Write code 写代码 |
| `code_debug` | 0.12 | 0.85 | 20 | 0.02 | 1.03 | Fix bugs 改 bug |
| `math_logic` | 0.30 | 0.90 | 30 | 0.03 | **1.00** | Reasoning (single) 推理（单次） |
| `math_logic_vote` | 0.85 | 0.95 | 50 | 0.02 | **1.00** | Reasoning (voting) 推理（投票采样） |
| `creative_writing` | 0.85 | 0.92 | 45 | 0.05 | 1.08 | Long-form writing 长文写作 |
| `video_script_story` | 0.92 | 0.93 | 50 | 0.05 | 1.12 | Story video scripts 故事类脚本 |
| `translation` | 0.35 | 0.85 | 25 | 0.03 | 1.05 | Translation 翻译 |
| `summarize` | 0.20 | 0.80 | 20 | 0.05 | 1.05 | Summarization 总结 |
| `thinking_model` | 0.60 | 0.95 | 20 | 0.00 | 1.00 | QwQ / R1 / Qwen3-thinking 推理模型专用 |
| `judge` | 0.10 | 0.70 | 10 | 0.02 | 1.02 | Critic / judge 裁判 / 批判 |
| `chat` | 0.70 | 0.80 | 20 | 0.00 | 1.05 | Daily chat 日常对话 |

**Golden rules · 硬规则：** `repeat_penalty` must be `1.0` for math/code. Never use CoT templates on reasoning models. High temperature (0.7–0.9) for voting samples, never low. Confidence < 0.5 → don't use the answer. If a task can be verified objectively (tests, comparisons), trust the verifier, not the model.

数学/代码的 `repeat_penalty` 必须 1.0；推理模型绝不套 CoT 模板；投票采样必须高温（0.7–0.9）；置信度 < 0.5 的答案不要用；能客观验证的任务，验证器永远比模型自评可靠。

---

## Docs · 文档

| File · 文件 | Content · 内容 |
|---|---|
| `references/sampling-params.md` | Every sampling param, Qwen specifics, common failure combos 采样参数详解、Qwen 特性、常见翻车组合 |
| `references/backend-tuning.md` | Quantization, KV cache, flash attention, DRY, speculative decoding 量化/KV cache/FA/DRY/投机解码 |
| `references/context-engineering.md` | Lost-in-the-middle, few-shot, format constraints 上下文工程 |
| `references/test-time-compute.md` | Voting, best-of-N, pipeline, MoA 推理时计算扩展 |
| `references/prompt-library.md` | Copy-paste templates for all 4 scenarios 四场景提示词模板 |
| `references/video-script-playbook.md` | Story/emotion video script methodology 故事类脚本方法论 |
| `references/hallucination-control.md` | Six hard rules + RAG controls 幻觉抑制六条硬规则 |
| `references/agent-integration.md` | Integration guide for Hermes / deepseek-harness / WorkBuddy agent 集成指南 |

---

## Related · 关联

- **`zhiguai-director`** — full AIGC short-film methodology (志怪/历史 narrative films: research → concept → script → storyboard → voice → publish). This skill complements it: the playbook from `zhiguai-director` feeds into `pipeline.py --mode script` as input, while this skill makes the local model running that pipeline stronger.

## License · 许可

MIT
