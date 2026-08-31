---
name: local-llm-amplify
description: 本地模型潜能榨取工具链。当用户使用本地部署的模型（llama.cpp / ik_llama.cpp / LM Studio / Ollama）完成任务，需要提升输出质量、调优采样参数、抑制幻觉，或需要用投票与流水线放大模型推理能力时使用。覆盖写代码、数学推理、长文写作、故事类视频脚本四类场景，含参数档位表、四阶段流水线、自洽投票和基准评测工具。触发词：本地模型、llama.cpp、LM Studio、Ollama、本地部署、模型变聪明、调参、采样参数、量化、本地大模型、self-consistency。
display_name: 本地模型潜能榨取
display_name_en: Local LLM Amplifier
description_zh: 让本地小模型发挥数倍效能：参数档位、上下文工程、投票与流水线、幻觉抑制、基准评测
description_en: Amplify local LLM capability with tuned sampling profiles, test-time compute pipelines, voting, and benchmarking
category: tools
version: 1.0.0
author: WorkBuddy
agent_created: true
---

# 本地模型潜能榨取

**目标不是让本地模型超过 GPT-5.6——那是权重决定的，做不到。目标是：把同一个本地模型的有效可用智能拉高 2–4 倍。**

四层杠杆：上下文工程 → 采样参数 → 推理时计算扩展 → 工具外挂与多模型编排。

---

## 硬规则（先读这 12 条，其余按需）

1. **场景决定参数。** 用一个档位打天下是最常见的错误。走下面路由表。
2. **数学/代码任务的 `repeat_penalty` 必须是 1.0。** 默认值 1.1 会显著拉低正确率（惩罚变量名和符号复用）。
3. **`min_p` 0.02–0.05 是本地模型性价比最高的单参数。** 砍掉长尾 token，很多胡言乱语和幻觉就藏在长尾里。llama.cpp 支持；LM Studio 不支持。
4. **推理模型（QwQ / DeepSeek-R1 蒸馏 / Qwen3 思考模式）不要套 CoT 模板。** 用 `thinking_model` 档（temp 0.6 / top_p 0.95 / top_k 20 / min_p 0）。套"让我们一步步思考"会造成双重思考和输出爆炸。
5. **能客观验证的任务，永远用验证器，不用模型自评。** 代码跑测试、数学比对答案。执行结果 > 任何裁判。
6. **有唯一正确答案 → 投票；创意任务 → 绝不投票。** 投票会把最平庸的创意方案选出来。
7. **投票采样必须高温（0.7–0.9）。** 低温投票 = 同一个答案抄 N 遍 = 白烧算力。这就是 `math_logic` 和 `math_logic_vote` 分成两个档位的原因。
8. **置信度 < 0.5 的答案不要用。** 换策略，别硬用。
9. **few-shot 给 2–4 个示例，是收益最高的单项改动。** 格式必须和期望输出完全一致。
10. **关键约束写两遍**：开头一遍，prompt 末尾再重申一遍。模型对长上下文的注意力是 U 型的，中间大段会被稀释。
11. **不要让模型心算。** 让它写 Python，你来执行。能外挂工具的计算一律外挂。
12. **LM Studio 不支持 `min_p` / `dry_penalty`，`top_k` 可能被忽略。** 脚本会按 `config.json` 的 `capabilities` 自动丢弃，不会报错——但效果确实打折。要榨性能用 llama.cpp。

---

## 场景路由表

按关键词机械匹配。匹配到就走对应行，不要自由发挥。

| 输入含 | 模式 | 参数档位 | 用什么工具 |
|---|---|---|---|
| `def` `class` `import` `函数` `实现` `写代码` `改 bug` `重构` `报错` `traceback` | code | `code_generate` / `code_debug` | `pipeline.py --mode code --verify tests.py` |
| `计算` `求解` `证明` `概率` `等于多少` `多少种` `逻辑` `\boxed` | reason | `math_logic` | `vote.py --extract boxed --n 5` |
| `脚本` `分镜` `旁白` `短视频` `口播` `视频号` `抖音` | script | `video_script_story` | `pipeline.py --mode script --rounds 1` |
| `写` `文章` `散文` `翻译` `总结` `改写` `标题` | write | `creative_writing` / `translation` / `summarize` | `pipeline.py --mode write` |
| 推理模型（QwQ / R1 / 思考模式） | — | `thinking_model` | 直接 `llm.py`，不套 CoT，不做流水线 |
| 其他 | — | `chat` | `llm.py` |

**判断顺序：先问"能不能客观验证"。**

```
有唯一正确答案吗？
├─ 是 → 能验证吗？
│        ├─ 能（测试/比对）→ pipeline --verify 或 vote --verify   ← 最强
│        └─ 不能            → vote.py（N=5~8，检查置信度）
└─ 否 → 多步复杂任务吗？
         ├─ 是 → pipeline.py（--mode write/script，1–2 轮）
         └─ 否 → few-shot + 参数调优就够，别过度工程
```

---

## 首次使用：配置（必做）

1. 改 `scripts/config.json`：`base_url` 端口、`model` 名
2. 查模型名：`python llm.py --list-models`
3. 连通性测试：`python llm.py --user "1+1=? 只输出数字" --profile math_logic`

llama.cpp 默认 `http://127.0.0.1:8080/v1`，LM Studio 默认 `1234`。

**推荐双实例**（24G 卡）：8080 跑 Qwen3-30B-A3B（通用），8081 跑 Qwen2.5-Coder-14B（代码）。
`config.json` 里 `llama_cpp_alt` 就是给这个留的位。
交叉批判比自我批判有效得多——同一个模型很难发现自己的思维盲点。

---

## 工具（全部仅依赖 Python 标准库，无需 pip install）

工作目录：`scripts/`

### llm.py — 单次调用

```bash
python llm.py --user "写一个快排" --profile code_generate
python llm.py --task-file req.txt --profile creative_writing --set temperature=0.95
python llm.py --user "..." --n 3 --seed 100        # 多候选，人工挑
python llm.py --user "..." --json                  # 机器可读
```

### vote.py — 自洽投票（有唯一答案的任务）

```bash
python vote.py --task-file q.txt --n 8 --profile math_logic_vote --extract boxed
python vote.py --task-file req.txt --n 6 --extract code --verify tests.py
python vote.py --task "..." --n 5 --judge --quiet
```

`--verify` 让代码任务质变：先用测试过滤掉跑不通的候选，再在通过者里投票。

**看 `confidence`**：≥0.7 采纳 / 0.5–0.7 采纳但标注低置信 / <0.5 别用。

### pipeline.py — 四阶段流水线（规划→求解→批判→修订）

```bash
python pipeline.py --mode code   --task-file req.txt --verify tests.py --rounds 2
python pipeline.py --mode script --task "志怪短片：..." --rounds 2 --out-dir out/s01
python pipeline.py --mode reason --task-file q.txt --rounds 1
```

批判阶段用低温裁判档（temp 0.1）——这是设计的关键，用高温让模型批判自己会变成自我辩护。
`--rounds` 1–2 足够，超过 3 轮模型会"为了改而改"。

### bench.py — 参数基准评测

```bash
python bench.py --taskset tasksets/math_basic.json --profile math_logic \
    --sweep temperature=0.2,0.4,0.6,0.8 --repeats 3
python bench.py --taskset tasksets/code_py_basic.json --profile code_generate \
    --compare-endpoint llama_cpp,llama_cpp_alt
```

输出 CSV + JSON + HTML 报告到 `scripts/results/`。
**别靠感觉调参。** 换模型/换量化值不值，用 `--compare-endpoint` 测出来。

### 退出码

`0` 成功 / `1` 配置或参数错误（查端点、查档位名）/ `2` 采样或验证全部失败。
`--quiet`：stdout 只有答案。`--json`：完整结果。诊断信息一律走 stderr。

---

## 参考文档（按需加载，一次只读一份）

一次读多份 = 主动制造 Lost in the Middle，等于都没读进去。

| 遇到什么问题 | 读这一份 |
|---|---|
| 输出质量差、想调参 | `references/sampling-params.md` |
| 上下文很长、模型顾此失彼 | `references/context-engineering.md` |
| 要不要投票、要不要流水线 | `references/test-time-compute.md` |
| 要复制提示词模板 | `references/prompt-library.md` |
| 写故事/情绪类视频脚本 | `references/video-script-playbook.md` |
| 模型在编造 | `references/hallucination-control.md` |
| 后端慢、显存不够、选量化 | `references/backend-tuning.md` |
| Hermes / deepseek-harness 集成 | `references/agent-integration.md` |

**上下文预算紧张（≤8k）时：一份都别读。** 上面的 12 条硬规则已经包含了这些文档的结论。

---

## 参数档位速查

| 档位 | temp | top_p | top_k | min_p | repeat | 用途 |
|---|---|---|---|---|---|---|
| `code_generate` | 0.25 | 0.90 | 40 | 0.02 | 1.05 | 写代码 |
| `code_debug` | 0.12 | 0.85 | 20 | 0.02 | 1.03 | 改 bug |
| `math_logic` | 0.30 | 0.90 | 30 | 0.03 | **1.00** | 推理（单次） |
| `math_logic_vote` | 0.85 | 0.95 | 50 | 0.02 | **1.00** | 推理（投票采样） |
| `creative_writing` | 0.85 | 0.92 | 45 | 0.05 | 1.08 | 长文写作 |
| `video_script_story` | 0.92 | 0.93 | 50 | 0.05 | 1.12 | 故事类脚本 |
| `translation` | 0.35 | 0.85 | 25 | 0.03 | 1.05 | 翻译 |
| `summarize` | 0.20 | 0.80 | 20 | 0.05 | 1.05 | 总结 |
| `thinking_model` | 0.60 | 0.95 | 20 | 0.00 | 1.00 | 推理模型专用 |
| `judge` | 0.10 | 0.70 | 10 | 0.02 | 1.02 | 裁判/批判 |

写作类档位额外开了 `presence_penalty` / `frequency_penalty`——中文长文对重复最敏感，两个都要，只开一个效果差一半。

---

## 交叉引用

- **志怪 / 历史 AIGC 短片全流程**（选题考据→高概念→念白剧本→分镜→配音→生成提示词→发布）→ 优先加载 `zhiguai-director`，它是那件事的完整方法论。本 skill 负责的是另一层：让跑那套流程的本地模型本身发挥得更好。两者叠加：用 `zhiguai-director` 的剧本骨架作为 `pipeline.py --mode script` 的输入。

---

## 再次确认（关键约束复述）

1. **场景路由走对**——用 `video_script_story` 的参数跑数学是灾难。
2. **repeat_penalty 在数学/代码上必须是 1.0。**
3. **能客观验证就别用模型自评。**
4. **创意任务不要投票**，要多采样人工挑。
5. **置信度 < 0.5 的答案不要用。**
6. **参考文档一次只读一份**，预算紧张时一份都不读。
