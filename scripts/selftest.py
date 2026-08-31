# -*- coding: utf-8 -*-
"""离线冒烟测试：不需要模型服务，只验证纯逻辑与题集正确性。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import load_config, load_profiles, resolve_endpoint, build_sampling
from vote import extract_answer, normalize, run_python_check

PY = "C:/Users/Administrator/.workbuddy/binaries/python/versions/3.13.12/python.exe"
fail = []
def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  {extra}" if extra and not cond else ""))
    if not cond:
        fail.append(name)

print("=== 1. 答案抽取器 ===")
t1 = "经过计算，x^2+1/x^2 = 9 - 2 = 7。所以答案是 \\boxed{7}"
check("boxed 抽取", extract_answer(t1, "boxed") == "7", extract_answer(t1, "boxed"))
check("auto 优先 boxed", extract_answer(t1, "auto") == "7", extract_answer(t1, "auto"))

t2 = "解法如下：\n```python\ndef f(x):\n    return x+1\n```\n这样就完成了。"
check("code 抽取", extract_answer(t2, "code").startswith("def f"), repr(extract_answer(t2, "code")))

t3 = "比值是 0. 96，即 96%。最终答案：96"
check("last_number 抽取", extract_answer(t3, "last_number") == "96", extract_answer(t3, "last_number"))

t4 = '结果：{"answer": "yes", "confidence": "high"}'
check("json 抽取", json.loads(extract_answer(t4, "json"))["answer"] == "yes")

print("\n=== 2. 归一化（聚类正确性）===")
pairs = [("42", "42.0", True), ("96", "96%", True), ("答案是 7。", "7", True),
         ("b", "B", True), ("1/6", "1 / 6", True), (" 53 ", "53", True),
         ("结果为 42", "42", True), ("final answer: 9", "9", True),
         ("96", "0.96", False), ("7", "8", False)]
for a, b, want in pairs:
    got = normalize(a) == normalize(b)
    check(f"normalize({a!r}) == normalize({b!r}) -> {want}", got == want,
          f"实际 {got}, 得到 {normalize(a)!r} vs {normalize(b)!r}")

print("\n=== 3. 能力降级（LM Studio 不支持 min_p / dry_penalty）===")
cfg, gov = load_config()
profiles = load_profiles()
_, ll = resolve_endpoint(cfg, "llama_cpp")
_, lm = resolve_endpoint(cfg, "lm_studio")
s_ll, d_ll = build_sampling("math_logic", profiles, {}, ll["capabilities"])
s_lm, d_lm = build_sampling("math_logic", profiles, {}, lm["capabilities"])
check("llama.cpp 保留 min_p", s_ll.get("min_p") == 0.03, str(s_ll))
check("llama.cpp 保留 top_k", s_ll.get("top_k") == 30, str(s_ll))
check("LM Studio 丢弃 min_p", "min_p" not in s_lm, str(s_lm))
check("LM Studio 丢弃提示非空", "min_p" in d_lm, str(d_lm))
check("LM Studio 保留 temperature", s_lm.get("temperature") == 0.3)

print("\n=== 4. 全局覆盖与 null 值 ===")
check("config 里的 null 覆盖被剔除", gov == {}, str(gov))
s2, _ = build_sampling("math_logic", profiles, {"temperature": 0.55}, ll["capabilities"])
check("命令行覆盖生效", s2["temperature"] == 0.55)

print("\n=== 5. 题集：数学答案正确性（用 Python 独立验算）===")
ts = json.loads(Path("tasksets/math_basic.json").read_text(encoding="utf-8"))
calc = {
    "m1": 1 / (1 / 6 + 1 / 4 - 1 / 12),
    "m2": 6 * 7,
    "m3": 100 - (100 // 3 + 100 // 5 - 100 // 15),
    "m4": 40 - (25 + 20 - 10),
    "m5": 3 ** 2 - 2,
    "m6": next(n for n in range(1, 200) if n % 3 == 2 and n % 5 == 3 and n % 7 == 2),
    "m7": 300 / (4 + 6) * 10,
    "m10": 1.2 * 0.8 * 100,
}
for t in ts["tasks"]:
    tid, ans = t["id"], t["answer"]
    if tid in calc:
        want = calc[tid]
        ok = abs(float(ans) - float(want)) < 1e-9
        check(f"{tid} 答案 {ans}", ok, f"应为 {want}")
    elif tid == "m8":
        check("m8 = 1/6", str(ans) == "1/6")
    else:
        # 逻辑题：选项标签是大写 A./B./C./D.，答案与标签一致即可
        check(f"{tid} 逻辑题答案 {ans}", str(ans).strip().lower() == "b")

print("\n=== 6. 题集：代码题用参考实现跑测试 ===")
ref = '''
def merge_intervals(intervals):
    if not intervals: return []
    s = sorted(intervals)
    out = [list(s[0])]
    for a, b in s[1:]:
        if a <= out[-1][1]: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out

def flatten_dict(d, sep='.', prefix=''):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict): out.update(flatten_dict(v, sep, key))
        else: out[key] = v
    return out

def is_valid_brackets(s):
    m = {')':'(', ']':'[', '}':'{'}
    st = []
    for c in s:
        if c in '([{': st.append(c)
        elif c in m:
            if not st or st.pop() != m[c]: return False
    return not st

def top_k_frequent(nums, k):
    from collections import Counter
    return [x for x, _ in Counter(nums).most_common(k)]

def longest_common_prefix(strs):
    if not strs: return ""
    p = strs[0]
    for s in strs[1:]:
        while not s.startswith(p):
            p = p[:-1]
            if not p: return ""
    return p
'''
cts = json.loads(Path("tasksets/code_py_basic.json").read_text(encoding="utf-8"))
for t in cts["tasks"]:
    ok, log = run_python_check(ref, t["tests"])
    check(f"{t['id']} 参考实现通过测试", ok, log[-300:])

print("\n=== 7. 题集：错误实现应当被拦下（防止测试形同虚设）===")
bad = "def merge_intervals(intervals):\n    return intervals\n"
ok, _ = run_python_check(bad, cts["tasks"][0]["tests"])
check("错误实现被判失败", not ok)
bad2 = "def is_valid_brackets(s):\n    return True\n"
ok2, _ = run_python_check(bad2, cts["tasks"][2]["tests"])
check("恒 True 被判失败", not ok2)

print("\n=== 8. 题集完整性 ===")
for f in ["math_basic", "code_py_basic", "writing_zh", "video_script"]:
    d = json.loads(Path(f"tasksets/{f}.json").read_text(encoding="utf-8"))
    need = "answer" if d["validator"] in ("boxed_number", "exact", "contains", "regex", "last_number") \
        else ("tests" if d["validator"] == "py_exec" else "rubric")
    miss = [t["id"] for t in d["tasks"] if need not in t]
    check(f"{f}.json 每道题都有 {need}", not miss, str(miss))
    check(f"{f}.json 题数 >= 3", len(d["tasks"]) >= 3, str(len(d["tasks"])))

print("\n" + "=" * 50)
print(f"结果：{'全部通过' if not fail else f'{len(fail)} 项失败 -> {fail}'}")
sys.exit(1 if fail else 0)
