#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-consistency 自洽投票 —— 本地模型性价比最高的一招。

原理：同一个问题用高温采样 N 次得到 N 个候选，把答案抽取出来归一化后聚类，
多数派胜出。对数学 / 逻辑 / 事实判断类任务提升最明显；对创意写作无效
（创意任务要的是多样性，不是收敛，别用这个）。

用法：
  python vote.py --task "一个篮子里..." --n 8 --profile math_logic_vote --extract boxed
  python vote.py --task-file q.txt --n 6 --extract code --verify tests.py
  python vote.py --task "..." --n 5 --extract auto --judge      # 无多数时让模型当裁判

抽取器 extract：
  auto        自动识别（先试 boxed，再试 code 块，再试最后一个数字，最后整段）
  boxed       \\boxed{...} 里的东西
  code        ```...``` 代码块
  json        第一个 JSON 对象
  last_number 文本里最后一个数字
  last_line   最后一行
  none        整段输出归一化
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import (  # noqa: E402
    load_config, load_profiles, resolve_endpoint, build_sampling,
    sample_many, extract_text, usage_of,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------- 答案抽取 ----------------

def _last_boxed(s):
    idx = s.rfind("\\boxed")
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right = None
    depth = 0
    started = False
    while i < len(s):
        if s[i] == "{":
            depth += 1
            started = True
        elif s[i] == "}":
            depth -= 1
            if depth == 0 and started:
                right = i
                break
        i += 1
    if right is None:
        return None
    inner = s[idx:right + 1]
    m = re.match(r"\\boxed\{(.*)\}$", inner, re.S) or re.match(r"\\fbox\{(.*)\}$", inner, re.S)
    return m.group(1) if m else inner


CODE_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.S)


def extract_answer(text, mode):
    if mode == "none":
        return text.strip()
    if mode == "boxed":
        v = _last_boxed(text)
        return v.strip() if v else text.strip()
    if mode == "code":
        blocks = CODE_RE.findall(text)
        return blocks[-1].strip() if blocks else text.strip()
    if mode == "json":
        m = re.search(r"\{.*\}", text, re.S)
        return m.group(0) if m else text.strip()
    if mode == "last_number":
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        return nums[-1] if nums else text.strip()
    if mode == "last_line":
        lines = [l for l in text.strip().splitlines() if l.strip()]
        return lines[-1].strip() if lines else ""
    if mode == "auto":
        v = _last_boxed(text)
        if v:
            return v.strip()
        blocks = CODE_RE.findall(text)
        if blocks:
            return blocks[-1].strip()
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        if nums and len(text) < 4000:
            return nums[-1]
        return text.strip()
    raise ValueError(f"未知抽取器：{mode}")


def normalize(ans):
    """归一化：让「42」「42.0」「答案是 42。」「96%」「96」算同一个答案。

    这里是投票聚类的核心。归一化做得不够，同一个答案会被拆成多个簇，
    置信度被人为拉低，投票等于失效。中文模型的引导语（"答案是…"）
    和百分号修饰是最常见的两种干扰，必须处理。
    """
    a = str(ans).strip()
    # 去掉引导语：答案是 / 结果为 / answer: / result 是 ...
    a = re.sub(r"^\s*(答案|结果|result|answer|final\s+answer)\s*(?:是|为|[:：])\s*",
               "", a, flags=re.I)
    a = a.strip("。．. \t\r\n")
    # 百分号只是修饰，不参与数值比较（"96%" 应等于 "96"）
    if a.endswith("%") or a.endswith("％"):
        a = a[:-1].strip()
    m = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", a.replace(",", ""))
    if m:
        f = float(m.group(0))
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return f"{f:.6f}"
    a = re.sub(r"\\[a-zA-Z]+", "", a)
    a = re.sub(r"[\s，,、；;：:。．!！?？\"'`\*_—－\-]+", "", a)
    return a.lower()


# ---------------- 代码验证 ----------------

def run_python_check(code, test_code, timeout=15):
    src = f"{code}\n\n# ---- tests ----\n{test_code}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(src)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
        return r.returncode == 0, (r.stdout + r.stderr)[-800:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


# ---------------- 裁判 ----------------

JUDGE_SYS = """你是一个严格的评审。你会看到同一个问题的多个候选答案。
任务：选出最正确、最完整、最忠实于问题要求的那一个。

规则：
- 只输出选中答案的编号，格式严格为：BEST: <编号>
- 下一行给出不超过 60 字的理由，格式：WHY: <理由>
- 如果几个答案质量相同，选编号最小的。
- 不要复述答案内容。"""


def judge_pick(question, candidates, ep, model, profiles, cfg, timeout):
    from llm import chat_once
    opts = "\n\n".join(f"[候选 {i}]\n{c}" for i, c in enumerate(candidates, 1))
    msgs = [
        {"role": "system", "content": JUDGE_SYS},
        {"role": "user", "content": f"问题：\n{question}\n\n候选答案：\n{opts}"},
    ]
    sampling, _ = build_sampling("judge", profiles, {}, ep.get("capabilities"))
    d = cfg.get("defaults", {})
    resp = chat_once(ep["base_url"], ep.get("api_key", ""), model, msgs, sampling,
                     timeout or d.get("timeout", 600), d.get("retries", 2),
                     d.get("retry_backoff_sec", 2))
    out = extract_text(resp)
    m = re.search(r"BEST\s*:\s*(\d+)", out)
    why = ""
    mw = re.search(r"WHY\s*:\s*(.+)", out)
    if mw:
        why = mw.group(1).strip()
    idx = int(m.group(1)) if m else None
    if idx is None or not (1 <= idx <= len(candidates)):
        return None, why or out.strip()
    return idx - 1, why


# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser(description="自洽投票（self-consistency）")
    ap.add_argument("--task", "-t", help="问题文本")
    ap.add_argument("--task-file", help="问题文件")
    ap.add_argument("--sys", dest="system", help="系统提示词")
    ap.add_argument("--n", type=int, default=8, help="采样次数，默认 8")
    ap.add_argument("--profile", "-p", default="math_logic_vote")
    ap.add_argument("--endpoint", "-e")
    ap.add_argument("--model", "-m")
    ap.add_argument("--set", action="append", metavar="K=V")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--extract", default="auto",
                    choices=["auto", "boxed", "code", "json", "last_number", "last_line", "none"])
    ap.add_argument("--verify", help="Python 测试文件路径：把抽取出的代码 + 测试一起执行，只统计通过的候选")
    ap.add_argument("--judge", action="store_true", help="无多数派时用裁判模型挑最佳")
    ap.add_argument("--timeout", type=int)
    ap.add_argument("--out", help="把完整结果存成 JSON")
    ap.add_argument("--quiet", "-q", action="store_true", help="只输出最终答案")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from llm import parse_set

    task = args.task
    if args.task_file:
        p = Path(args.task_file)
        if not p.exists():
            sys.exit(f"[错误] 找不到问题文件：{p}")
        task = p.read_text(encoding="utf-8")
    if not task:
        sys.exit("[错误] 需要 --task 或 --task-file")

    cfg, global_ov = load_config()
    profiles = load_profiles()
    d = cfg.get("defaults", {})
    ep_name, ep = resolve_endpoint(cfg, args.endpoint)
    model = args.model or ep.get("model")

    ov = dict(global_ov)
    ov.update(parse_set(args.set))
    sampling, dropped = build_sampling(args.profile, profiles, ov, ep.get("capabilities"))
    if dropped and not args.quiet:
        print(f"[提示] 端点不支持 {dropped}，已丢弃。", file=sys.stderr)

    if "max_tokens" not in sampling:
        sampling["max_tokens"] = 4096

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": task})

    test_code = None
    if args.verify:
        vp = Path(args.verify)
        if not vp.exists():
            sys.exit(f"[错误] 找不到测试文件：{vp}")
        test_code = vp.read_text(encoding="utf-8")

    t0 = time.time()
    if not args.quiet:
        print(f"[采样] n={args.n}  档位={args.profile}  端点={ep_name}  模型={model}", file=sys.stderr)

    resps = sample_many(ep["base_url"], ep.get("api_key", ""), model, messages, sampling,
                        args.n, args.timeout or d.get("timeout", 600),
                        d.get("retries", 2), d.get("retry_backoff_sec", 2),
                        d.get("max_concurrency", 4), args.seed)

    raw = []
    for r in resps:
        if r is None or "_error" in r:
            raw.append({"ok": False, "error": (r or {}).get("_error", "unknown"), "text": ""})
        else:
            raw.append({"ok": True, "text": extract_text(r), "usage": usage_of(r)})

    for i, c in enumerate(raw, 1):
        c["index"] = i
        if not c["ok"]:
            c["answer"] = ""
            c["norm"] = ""
            c["passed"] = False
            continue
        c["answer"] = extract_answer(c["text"], args.extract)
        c["norm"] = normalize(c["answer"])
        c["passed"] = True
        if test_code:
            ok, log = run_python_check(c["answer"], test_code)
            c["passed"] = ok
            c["verify_log"] = log

    usable = [c for c in raw if c["ok"] and c["passed"]]
    if not usable:
        msg = "所有候选都未能通过验证。" if test_code else "所有候选采样都失败了。"
        print(f"[错误] {msg}", file=sys.stderr)
        if args.out:
            Path(args.out).write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        sys.exit(2)

    groups = defaultdict(list)
    for c in usable:
        groups[c["norm"]].append(c)
    counts = Counter({k: len(v) for k, v in groups.items()})
    top = counts.most_common()
    best_norm, best_votes = top[0]
    tie = [k for k, v in top if v == best_votes]

    winner = groups[best_norm][0]
    method = "majority"
    note = ""

    if len(tie) > 1:
        method = "tie"
        note = f"平票（{len(tie)} 个答案各 {best_votes} 票）"
        if args.judge:
            cands = [groups[k][0]["answer"] for k in tie]
            idx, why = judge_pick(task, cands, ep, model, profiles, cfg, args.timeout)
            if idx is not None:
                winner = groups[tie[idx]][0]
                method = "judge"
                note = f"平票后由裁判选定：{why}"
            else:
                note += "；裁判失效，取第一个"
        else:
            note += "；建议加 --judge 让模型裁决，或提高 --n"

    confidence = best_votes / len(usable)

    payload = {
        "task": task,
        "n": args.n,
        "usable": len(usable),
        "profile": args.profile,
        "endpoint": ep_name,
        "model": model,
        "extract": args.extract,
        "method": method,
        "confidence": round(confidence, 4),
        "votes": best_votes,
        "tie_count": len(tie),
        "note": note,
        "winner_answer": winner["answer"],
        "winner_full_text": winner["text"],
        "elapsed_sec": round(time.time() - t0, 1),
        "distribution": [{"answer": groups[k][0]["answer"], "votes": v} for k, v in top],
        "candidates": raw,
    }

    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.quiet:
        print(winner["answer"])
        return

    print("\n" + "=" * 60)
    print("自洽投票结果")
    print("=" * 60)
    print(f"有效候选   : {len(usable)}/{args.n}" + (f"（代码验证过滤后）" if test_code else ""))
    print(f"决胜方式   : {method}  置信度 {confidence:.0%}  ({best_votes}/{len(usable)})")
    if note:
        print(f"备注       : {note}")
    print(f"耗时       : {payload['elapsed_sec']}s")
    print("-" * 60)
    print("答案分布：")
    for k, v in top:
        preview = groups[k][0]["answer"].replace("\n", " ⏎ ")
        if len(preview) > 70:
            preview = preview[:70] + "…"
        bar = "█" * v
        print(f"  {v:>2} 票 {bar:<12} {preview}")
    print("-" * 60)
    print("最终答案：")
    print(winner["answer"])
    print("=" * 60)
    if confidence < 0.5 and not args.judge:
        print(f"[警告] 置信度仅 {confidence:.0%}，答案不可靠。提高 --n 或加 --judge。", file=sys.stderr)


if __name__ == "__main__":
    main()
