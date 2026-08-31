# 测试题集说明

## 题集文件

| 文件 | 校验方式 | 题数 | 用途 |
|---|---|---|---|
| `math_basic.json` | `boxed_number` | 10 | 数学与逻辑。含 4 道陷阱题（m3 容斥、m7 别逐段求和、m9 必然性、m10 涨跌不对称） |
| `code_py_basic.json` | `py_exec` | 5 | Python 代码。生成代码 + `tests` 拼接后真跑 |
| `writing_zh.json` | `llm_judge` | 4 | 中文写作 / 翻译。模型互评 1–10 分 |
| `video_script.json` | `llm_judge` | 3 | 故事 / 情绪类视频脚本。模型互评 |
| `smoke.json` | 混合 | 2 | 冒烟测试，确认端点连通 |

## 题集格式

```jsonc
{
  "name": "my_taskset",
  "validator": "boxed_number",   // 默认校验器
  "extract": "boxed",            // 投票模式下用哪个抽取器
  "system": "",                  // 可选，全局系统提示词
  "tasks": [
    {
      "id": "t1",
      "prompt": "题目正文",
      "answer": "96",             // boxed_number / exact / contains / regex / last_number 用
      "tests": "assert f(1) == 1" // py_exec 用，会拼到生成代码后面执行
      "rubric": "评分标准",        // llm_judge 用
      "system": "",               // 可选，覆盖全局
      "timeout": 15               // py_exec 专用，秒
    }
  ]
}
```

## 校验器一览

| 校验器 | 比较方式 | 适合场景 |
|---|---|---|
| `exact` | 归一化后全等 | 封闭式问答 |
| `contains` | 输出包含目标串 | 关键词必须出现 |
| `regex` | 正则匹配 | 格式校验 |
| `boxed_number` | 抽取 `\boxed{}` 后比较数值 | 数学推理 |
| `last_number` | 抽取最后一个数字后比较 | 数学（模型不会用 boxed 时） |
| `py_exec` | 真跑 Python 测试 | 代码生成 |
| `llm_judge` | 裁判模型打 1–10 分，归一化为 0–1 | 写作、脚本等主观题 |

## 冒烟题集怎么跑

`smoke.json` 里两道题用了不同校验器，`bench.py` 目前按题集整体取一个校验器，
所以分开跑：

```bash
# 数学那道
python bench.py --taskset tasksets/smoke.json --profile math_logic --repeats 1 --validator boxed_number

# 更省事：直接用 llm.py 打一发，确认连通
python llm.py --user "1+1=? 只输出数字" --profile math_logic
```

## 自己出题的原则

1. **要有陷阱**。全是送分题，测不出参数差异，所有组合都是 100%。
2. **答案要唯一可判定**。别出「谈谈你的看法」这种，除非用 `llm_judge`。
3. **断言写宽松点**。`py_exec` 里用 `set(...)` / `len(...)` / `in` 而不是严格等序，
   否则会把正确实现判成错。
4. **至少 8 道题**。低于这个数，随机性会淹没参数差异。
5. **llm_judge 只能横比不能纵比**。裁判会偏袒同族模型；跨模型对比时把它的分数当参考，
   不当硬指标。真要严谨，人工抽 10 份复评一次做校准。
