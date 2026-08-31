# 后端调优：llama.cpp / ik_llama.cpp / LM Studio

> 面向 12–24 GB 显存（14B–32B 级别）这一档。别的档位思路相通，数值要调。

---

## 一、先说结论：这一档的最优解

你有 12–24 GB，大概率是 16G 或 24G 的卡。**首选 MoE 架构模型**：

| 模型 | 量化 | 体积 | 24G 卡表现 | 16G 卡表现 |
|---|---|---|---|---|
| Qwen3-30B-A3B | Q4_K_M | ~17 GB | 舒服，KV cache 还剩 6G | 紧，需 Q4_K_S 或降 ctx |
| Qwen3-30B-A3B | Q5_K_M | ~21 GB | 刚好 | 不行 |
| Qwen3-30B-A3B | Q6_K | ~24 GB | 勉强，ctx 要压到 16k | 不行 |
| Qwen2.5-14B / Qwen3-14B | Q6_K / Q8_0 | ~12–16 GB | 很宽松，可上 64k ctx | 舒服 |
| Qwen2.5-Coder-14B | Q6_K | ~12 GB | 代码任务用它 | 代码任务用它 |
| Qwen3-32B（dense） | Q4_K_M | ~19 GB | 可以，但比 30B-A3B 慢 | 不行 |

**为什么推 MoE**：Qwen3-30B-A3B 只激活 3B 参数，速度接近 7B 模型，
但知识容量和推理质量接近同尺寸 dense 模型。在 24G 这一档，
它的"每秒有效智能"明显高于任何你能塞进去的 dense 模型。

**双模型策略（推荐）**：24G 卡上同时跑两个服务实例更划算——
- `8080` → Qwen3-30B-A3B（通用 / 推理 / 写作）
- `8081` → Qwen2.5-Coder-14B Q6_K（代码专用）

本 skill 的 `config.json` 里 `llama_cpp_alt` 就是给这个留的位。
不同任务用不同模型，比一个模型打天下强得多。

---

## 二、量化怎么选

量化伤害程度按任务排序：

```
数学 / 代码  >  长文写作  >  摘要 / 分类 / 抽取
```

| 量化 | 相对体积 | 质量损失 | 建议 |
|---|---|---|---|
| Q8_0 | 大 | 极小 | 显存够就用，尤其是代码和数学 |
| Q6_K | 中大 | 很小 | 甜点。质量/体积比最优 |
| Q5_K_M | 中 | 小 | 通用甜点 |
| Q4_K_M | 中小 | 可接受 | 显存紧张时的默认选择 |
| Q4_K_S / IQ4_XS | 小 | 明显 | 只在必须塞下时用，别拿来做数学 |
| Q3_K / Q2_K | 很小 | 严重 | 不推荐。输出的流畅度都会掉 |

**经验法则**：让模型完整进显存（`-ngl 99`）比用更高量化更重要。
一个全在 GPU 上的 Q4_K_M，远快于一个 Q8_0 但要 spill 到内存的。
速度差距是 5–10 倍，质量差距远小于这个。

**验证**：用 `llama-perplexity` 或 `llama-bench` 对比，别猜。

---

## 三、llama-server 启动参数

基础命令：

```bash
llama-server \
  -m /path/to/Qwen3-30B-A3B-Q4_K_M.gguf \
  -c 32768 \
  -ngl 99 \
  -fa \
  -b 2048 -ub 512 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --host 127.0.0.1 --port 8080 \
  -np 2
```

### 逐项说明

| 参数 | 作用 | 建议 |
|---|---|---|
| `-c` | 上下文长度 | 24G 卡配 Q4_K_M 30B：32768 够用。设太大会挤占 KV cache，反而让其他任务变慢 |
| `-ngl 99` | GPU offload 层数 | 全 offload。显存不够时逐步降到 40/30/20 |
| `-fa` | Flash Attention | **必开**。省显存且提速，长上下文效果尤其明显 |
| `-b` / `-ub` | 批处理大小 | `-b 2048 -ub 512`。prompt 处理阶段的速度主要看这个 |
| `--cache-type-k/v` | KV cache 量化 | `q8_0` 质量几乎无损；显存紧张用 `q4_0`。长 ctx 时这项能省几个 G |
| `-np` | 并行 slots | 设 2–4 可让并发请求真正并行。本 skill 的投票采样会受益 |
| `--rope-scaling yarn` | 长上下文外推 | ctx 超过模型原生长度时才用，会有质量损失 |

### 长上下文专用

```bash
# 需要 64k+ 时
-c 65536 --cache-type-k q4_0 --cache-type-v q4_0 -fa --rope-scaling yarn --rope-scale 4
```

注意：Qwen 系列原生支持 32k（部分版本 128k），不要一上来就外推。

### 投机解码

```bash
llama-server -m big.gguf -md small_draft.gguf -ngl 99
```

小模型起草、大模型验证。**对 MoE 模型效果一般**（因为 MoE 本身解码就快），
对 dense 大模型提速 1.5–3 倍。你的 30B-A3B 上意义不大，14B dense 上值得试。

### 抑制长文重复（写脚本/长文时加）

```bash
--dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2 --dry-penalty-last-n 1024
```

---

## 四、ik_llama.cpp

是 llama.cpp 的一个分支，主打更好的量化方案和 MoE 优化。差异点：

- 支持 `IQ` 系列量化，同体积下质量更好（尤其是 IQ4_KSS / IQ3_M）
- MoE 模型的 CPU/GPU 混合推理更成熟，显存不够时部分 expert 放内存的调度更聪明
- 提供 `repulsor` sampler，抑制重复比 DRY 更激进

**什么时候切**：当你在 16G 卡上想塞 30B-A3B 时，ik_llama.cpp 的量化方案能帮你多挤出
1–2 GB，或者同样的体积换更好的质量。其余时候 llama.cpp 够用。

它的 HTTP server 参数与 llama.cpp 基本兼容，本 skill 的脚本可以直接用。

---

## 五、LM Studio 的限制与对策

LM Studio 提供 OpenAI 兼容 API（默认 `http://127.0.0.1:1234/v1`），
但**参数可控性明显弱于 llama.cpp**：

| 参数 | LM Studio | 对策 |
|---|---|---|
| `temperature` / `top_p` | ✅ 支持 | |
| `top_k` | ⚠️ 部分支持，可能被忽略 | |
| `min_p` | ❌ 基本不支持 | 只能用 top_p 近似替代 |
| `repeat_penalty` | ⚠️ 通过 preset 设置，API 层面不稳定 | 在 GUI 的 preset 里设好 |
| `dry_penalty` | ❌ | 用更高的 presence_penalty 近似 |
| `seed` | ⚠️ 不保证可复现 | 会影响投票采样的可复现性 |

本 skill 的 `config.json` 里给 LM Studio 端点标注了 `capabilities`，
脚本会自动丢弃不支持的参数，不会报错——但这也意味着**LM Studio 上你拿不到
min_p 和 DRY 的收益**。

**结论**：要榨性能，用 llama.cpp 的 server。LM Studio 适合日常对话和快速试模型。

---

## 六、显存不够时的降级顺序

按这个顺序砍，每砍一步验证一次：

1. 降量化：Q6_K → Q5_K_M → Q4_K_M
2. 降 KV cache 精度：`q8_0` → `q4_0`
3. 降上下文：`32768` → `16384`
4. 部分 offload：`-ngl 99` → 手工调（`-ngl 40` 之类）
5. 换 MoE 模型：dense 32B → Qwen3-30B-A3B
6. 上 ik_llama.cpp 的 IQ 量化

**不要第一时间降上下文**。很多人先砍 `-c`，实际上 KV cache 量化省得更多、代价更小。

---

## 七、测速与验证

```bash
# 纯速度
llama-bench -m model.gguf -ngl 99 -p 512 -n 128

# 困惑度（衡量量化损失）
llama-perplexity -m model.gguf -f wiki.test.raw -ngl 99

# 端到端（本 skill 的）
python bench.py --taskset tasksets/math_basic.json --profile math_logic \
    --sweep temperature=0.3,0.6 --compare-endpoint llama_cpp,llama_cpp_alt
```

用 `--compare-endpoint` 直接对比两个服务实例，是判断"换模型/换量化值不值"最诚实的方法。

---

## 八、健康自检清单

跑之前确认这七条，能省掉 90% 的排查时间：

- [ ] `-fa` 开了吗？（性能影响最大）
- [ ] `-ngl` 够不够让模型完整进显存？用 nvidia-smi 看
- [ ] `-c` 设的上下文，实际能不能撑住？跑长文时看会不会崩
- [ ] KV cache 量化开了吗？
- [ ] 服务监听在 `127.0.0.1` 还是 `0.0.0.0`？端口和 `config.json` 一致吗？
- [ ] 并发请求时 `-np` 够吗？（投票采样会同时打 N 个请求）
- [ ] 模型是哪个量化？别把 Q4_K_S 当 Q4_K_M 用
