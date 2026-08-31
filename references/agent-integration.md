# Agent 集成指南

> 给 Hermes、deepseek-harness、WorkBuddy 这三类的宿主用。
> 人读的话看 SKILL.md 就够了。

---

## 一、核心认知：读这份 skill 的可能就是它要优化的那个模型

如果宿主的推理后端是本地模型，**那么 skill 的每一句话都在消耗
它自己的注意力预算**。这是自指的。

由此推出四条约束，本 skill 的写法已经遵守，你自己扩展时也要遵守：

1. **SKILL.md 必须短且硬规则化**。散文式长文档塞进 system，
   会稀释 agent 对真正任务的注意力。
2. **参考文档按需加载**。一次全读 = 主动制造 Lost in the Middle。
3. **路由表要能被机械执行**。"根据情况选择合适的档位"这种话对小模型
   等于没说。要写"含 def/class/报错 → code"。
4. **脚本输出要机器可读**。JSON + 退出码，不要人看的格式化表格
   （`--quiet` 和 `--json` 就是为此存在）。

---

## 二、加载策略：按上下文预算分级

```
预算充足（≥32k，宿主是云端大模型）
  └─ 读 SKILL.md + 当前任务对应的 1-2 份 references/

预算中等（8k-32k）
  └─ 只读 SKILL.md。需要时再单独 Read 某一篇 reference。

预算紧张（≤8k，宿主就是那个 14B 本地模型）
  └─ 只读 SKILL.md 的「硬规则」和「路由表」两节。
     其余一概不读。参考文档的结论已经浓缩进规则里了。
```

### 路由到文档的映射

| 当前任务 | 只读这一份 |
|---|---|
| 调参、输出质量差 | `sampling-params.md` |
| 上下文很长、模型顾此失彼 | `context-engineering.md` |
| 要不要投票 / 要不要流水线 | `test-time-compute.md` |
| 复制提示词模板 | `prompt-library.md` |
| 视频脚本 | `video-script-playbook.md` |
| 模型在编造 | `hallucination-control.md` |
| 后端慢 / 显存不够 | `backend-tuning.md` |

**一次只读一份。** 同时读三份等于三份都没读进去。

---

## 三、配置

首次使用必须让宿主引导用户填 `scripts/config.json`：

| 字段 | 说明 |
|---|---|
| `endpoints.*.base_url` | llama.cpp 默认 `http://127.0.0.1:8080/v1`，LM Studio 默认 `1234` |
| `endpoints.*.model` | 用 `python llm.py --list-models` 拿到 |
| `endpoints.*.capabilities` | 该后端支持哪些采样参数。LM Studio 要删掉 `top_k`/`min_p` |
| `active_endpoint` | 默认用哪个 |
| `defaults.timeout` | 长任务建议 600 |
| `defaults.max_concurrency` | 本地单卡建议 2-4。设太高只是排队，不会更快 |

**关键**：`capabilities` 字段是本 skill 能在混合后端环境下工作的原因。
脚本会按它自动丢弃不支持的参数，不会因为传了 `min_p` 给 LM Studio 就报错。

---

## 四、脚本调用约定

三个脚本都遵守同一套约定，方便程序化调用：

| 约定 | 说明 |
|---|---|
| 输入 | 全部 CLI 参数，无交互式提示 |
| 题目/长文本 | 走 `--task-file` / `--file`，不要塞在命令行参数里（转义会出问题） |
| 机器可读输出 | `--json` 输出到 stdout |
| 只要结果 | `--quiet`，stdout 只有最终答案 |
| 诊断信息 | 一律走 stderr，不污染 stdout |
| 退出码 | 0 成功 / 1 配置或参数错误 / 2 采样或验证全部失败 |
| 依赖 | **仅 Python 标准库**，不需要 pip install |

### 程序化调用示例

```bash
# 拿答案（只取 stdout）
ANS=$(python vote.py --task-file q.txt --n 8 --extract boxed --quiet)

# 拿完整结果做判断
python vote.py --task-file q.txt --n 8 --out result.json --quiet
python -c "import json;d=json.load(open('result.json'));print(d['confidence'])"
```

**置信度是给 agent 的决策信号**：

```python
if d["confidence"] >= 0.7:   → 采纳
elif d["confidence"] >= 0.5: → 采纳但标注"低置信"
else:                        → 不要采纳。换策略或上报用户
```

### 退出码处理

```bash
python vote.py ... ; echo $?
# 1 → 配置问题（端点连不上、档位名错）。检查 config.json。
# 2 → 所有候选采样失败，或代码验证全不通过。任务本身可能有问题。
```

---

## 五、宿主差异

### WorkBuddy

- 可以直接 `Bash` 调用脚本，`present_files` 展示 `results/report_*.html`
- 建议把常用档位固化到 `.workbuddy/memory/MEMORY.md`，下次不用重新问

### Hermes

- 通过 shell 工具调用脚本
- **注意上下文累积**：Hermes 的长程任务里，每轮工具返回结果都会进上下文。
  用 `--quiet` 只取答案，不要回传完整 JSON
- 失败时改措辞重试，别重复提交同一个 prompt

### deepseek-harness

- 同样的 CLI 约定
- 如果 harness 自己就是本地模型驱动，优先用 `--quiet` + 短档位名，
  减少 agent 侧解析负担
- 代码任务坚持用 `--verify`，别让模型自评

---

## 六、Agent 特有的三个坑

### 1. 错误累积

第一轮的幻觉会成为第二轮的输入事实，到第五轮整个任务建立在虚构前提上，
而且模型会越来越"确信"。

对策：每 3–5 步回溯验证一次核心前提；工具结果优先于模型记忆。

### 2. 上下文膨胀

工具返回的大文件、完整 traceback、多轮历史，会快速吃掉预算，
然后 agent 开始遗忘早期指令。

对策：大文件先裁剪；traceback 只提取关键行；长任务开新上下文。

### 3. 同质重试

同一个 prompt 在同一个上下文里重复提交，模型会沿同一条错误路径走。

对策：**改措辞** > 改参数 > 清空历史重来。不要原样重试。

---

## 七、最小集成示例

一个 agent 接到任务的完整决策路径：

```
1. 判断场景（关键词路由，见 SKILL.md 路由表）
2. 能客观验证吗？
   ├─ 代码 → pipeline.py --mode code --verify tests.py
   └─ 数学 → vote.py --extract boxed --n 5
3. 不能客观验证？
   ├─ 创意/脚本 → pipeline.py --mode script --rounds 1
   └─ 一般问答 → llm.py --profile <档位>  + few-shot
4. 检查置信度 / VERDICT
   ├─ 低 → 换策略，别硬用
   └─ 高 → 交付
```

**最容易被跳过、也最不该跳过的**：先想清楚"这个任务能不能被客观验证"。
能验证的任务，验证器的可靠性永远高于模型自评。
