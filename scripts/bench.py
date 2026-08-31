#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数 / 模型基准评测 —— 把"调参靠感觉"变成"调参看数字"。

做法：拿一份测试题集，遍历若干参数组合（或若干模型 × 端点），
每组跑 R 次，用校验器打分，最后出排名 + CSV + HTML 报告。

用法：
  # 扫温度
  python bench.py --taskset tasksets/math_basic.json --profile math_logic \
      --sweep temperature=0.2,0.4,0.6,0.8 --repeats 3

  # 对比"单次求解" vs "8 次投票"
  python bench.py --taskset tasksets/math_basic.json --profile math_logic --mode single
  python bench.py --taskset tasksets/math_basic.json --profile math_logic_vote --mode vote --vote-n 8

  # 对比两个后端/模型
  python bench.py --taskset tasksets/code_py_basic.json --profile code_generate \
      --compare-endpoint llama_cpp,llama_cpp_alt

题集格式见 tasksets/README.md。
"""

import argparse
import csv
import itertools
import json
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import (  # noqa: E402
    load_config, load_profiles, resolve_endpoint, build_sampling,
    chat_once, extract_text, usage_of, parse_set, sample_many,
)
from vote import extract_answer, normalize, run_python_check  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OUT_DIR = Path(__file__).resolve().parent / "results"


# ---------------- 校验器 ----------------

JUDGE_SYS = """你是严格的阅卷人。根据评分标准给下面的回答打分。
严格按格式输出两行：
SCORE: <1-10 的整数>
REASON: <不超过 40 字的理由>
不要输出其他内容。"""


def check(task, output, validator, ep=None, model=None, profiles=None, cfg=None, timeout=None):
    """返回 (score: 0.0-1.0, detail: str)"""
    v = validator
    expected = task.get("answer")
    if v == "exact":
        ok = normalize(expected) == normalize(output)
        return (1.0 if ok else 0.0), ("match" if ok else f"期望 {expected!r}")
    if v == "contains":
        ok = str(expected) in output
        return (1.0 if ok else 0.0), ("match" if ok else "未包含目标串")
    if v == "regex":
        ok = re.search(str(expected), output, re.S) is not None
        return (1.0 if ok else 0.0), ("match" if ok else "正则不匹配")
    if v == "boxed_number":
        got = extract_answer(output, "boxed")
        ok = normalize(got) == normalize(expected)
        return (1.0 if ok else 0.0), (f"得到 {got!r} / 期望 {expected!r}")
    if v == "last_number":
        got = extract_answer(output, "last_number")
        ok = normalize(got) == normalize(expected)
        return (1.0 if ok else 0.0), (f"得到 {got!r} / 期望 {expected!r}")
    if v == "py_exec":
        code = extract_answer(output, "code")
        ok, log = run_python_check(code, task.get("tests", ""), timeout=task.get("timeout", 15))
        return (1.0 if ok else 0.0), ("测试通过" if ok else log[-300:])
    if v == "llm_judge":
        msgs = [
            {"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content":
                f"题目：\n{task['prompt']}\n\n评分标准：\n{task.get('rubric','')}\n\n"
                f"回答：\n{output}\n\n请打分。"},
        ]
        sampling, _ = build_sampling("judge", profiles, {}, (ep or {}).get("capabilities"))
        d = (cfg or {}).get("defaults", {})
        resp = chat_once(ep["base_url"], ep.get("api_key", ""), model, msgs, sampling,
                         timeout or d.get("timeout", 600), d.get("retries", 2),
                         d.get("retry_backoff_sec", 2))
        txt = extract_text(resp)
        m = re.search(r"SCORE\s*:\s*(\d+)", txt)
        if not m:
            return 0.0, f"裁判输出无法解析：{txt[:120]}"
        sc = max(1, min(10, int(m.group(1))))
        return (sc - 1) / 9.0, f"{sc}/10"
    return 0.0, f"未知校验器 {v}"


# ---------------- 执行单元 ----------------

def run_one(job, cfg, profiles):
    """跑一道题的一次（可能是投票模式，内部采样 n 次）。"""
    ep = job["ep"]
    model = job["model"]
    task = job["task"]
    sampling = job["sampling"]
    d = cfg.get("defaults", {})
    messages = []
    if job.get("system"):
        messages.append({"role": "system", "content": job["system"]})
    if task.get("system"):
        messages.append({"role": "system", "content": task["system"]})
    messages.append({"role": "user", "content": task["prompt"]})

    t0 = time.time()
    if job["mode"] == "vote":
        resps = sample_many(ep["base_url"], ep.get("api_key", ""), model, messages, sampling,
                            job["vote_n"], job.get("timeout") or d.get("timeout", 600),
                            d.get("retries", 2), d.get("retry_backoff_sec", 2),
                            d.get("max_concurrency", 4), job.get("seed", 1))
        texts = [extract_text(r) for r in resps if r and "_error" not in r]
        from collections import Counter, defaultdict
        groups = defaultdict(list)
        for t in texts:
            a = extract_answer(t, task.get("extract", job["extract"]))
            groups[normalize(a)].append(a)
        if not groups:
            return {"score": 0.0, "detail": "采样全失败", "sec": time.time() - t0, "gen_tokens": 0}
        best = max(groups.values(), key=len)
        output = best[0]
        votes = len(best)
        if len([1 for g in groups.values() if len(g) == votes]) > 1 and job.get("tie_break_judge"):
            output = max(groups.values(), key=len)[0]
        gen_tokens = sum((usage_of(r) or {}).get("completion_tokens", 0) for r in resps if r)
        sec = time.time() - t0
        score, detail = check(task, output, job["validator"],
                              ep=ep, model=model, profiles=profiles, cfg=cfg,
                              timeout=job.get("timeout"))
        return {"score": score, "detail": detail, "sec": sec, "gen_tokens": gen_tokens,
                "confidence": votes / max(1, len(texts)), "output": output,
                "candidates": len(texts)}
    else:
        s = dict(sampling)
        if job.get("seed") is not None:
            s["seed"] = job["seed"] + job["repeat_idx"]
        resp = chat_once(ep["base_url"], ep.get("api_key", ""), model, messages, s,
                         job.get("timeout") or d.get("timeout", 600),
                         d.get("retries", 2), d.get("retry_backoff_sec", 2))
        sec = time.time() - t0
        output = extract_text(resp)
        gen_tokens = (usage_of(resp) or {}).get("completion_tokens", 0) or 0
        score, detail = check(task, output, job["validator"],
                              ep=ep, model=model, profiles=profiles, cfg=cfg,
                              timeout=job.get("timeout"))
        return {"score": score, "detail": detail, "sec": sec, "gen_tokens": gen_tokens,
                "output": output, "confidence": None, "candidates": 1}


# ---------------- 报告 ----------------

def build_html(report, path):
    rows = report["combos"]
    best = rows[0] if rows else None

    def cell_bar(v):
        try:
            pct = max(0.0, min(1.0, float(v))) * 100
        except Exception:
            pct = 0.0
        return f'<div class="bar"><span style="width:{pct:.1f}%"></span></div>'

    trs = []
    for i, r in enumerate(rows, 1):
        params = " ".join(f"{k}={v}" for k, v in r["params"].items())
        trs.append(
            f"<tr class='{'top' if i == 1 else ''}'>"
            f"<td class='rank'>{i}</td>"
            f"<td><code>{r['label']}</code></td>"
            f"<td><code>{params or '(档位默认)'}</code></td>"
            f"<td class='num'>{r['accuracy'] * 100:.1f}%</td>"
            f"<td>{cell_bar(r['accuracy'])}</td>"
            f"<td class='num'>{r['avg_sec']:.1f}s</td>"
            f"<td class='num'>{r['tok_per_sec']:.1f}</td>"
            f"<td class='num'>{r['runs']}</td>"
            f"</tr>"
        )

    per_task = ""
    if report.get("per_task"):
        pt_rows = []
        for t in report["per_task"]:
            pt_rows.append(
                f"<tr><td class='taskq' title='{t['prompt']}'>{t['id']}</td>"
                + "".join(f"<td class='num'>{v * 100:.0f}</td>" for v in t["scores"])
                + "</tr>")
        head = "".join(f"<th>{r['label']}<br><small>{' '.join(f'{k}={v}' for k, v in r['params'].items()) or '默认'}</small></th>"
                       for r in rows)
        per_task = (f"<h2>逐题对比</h2><table class='pt'><thead><tr><th>题目</th>{head}</tr></thead>"
                    f"<tbody>{''.join(pt_rows)}</tbody></table>")

    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>本地模型基准报告</title>
<style>
:root{{color-scheme:dark}}
body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;margin:0;padding:32px;
background:#16161a;color:#e8e8ea;line-height:1.6}}
h1{{font-size:20px;font-weight:500;margin:0 0 4px}}
h2{{font-size:15px;font-weight:500;margin:32px 0 12px}}
.meta{{color:#8b8b93;font-size:13px;margin-bottom:24px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th{{text-align:left;font-weight:500;color:#8b8b93;border-bottom:1px solid #2e2e35;padding:8px 10px}}
td{{border-bottom:1px solid #232329;padding:8px 10px;vertical-align:middle}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
td.rank{{width:36px;color:#8b8b93}}
tr.top td{{background:#1b2430}}
code{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;color:#b5d4f4}}
.bar{{background:#232329;border-radius:3px;height:8px;width:120px;overflow:hidden}}
.bar span{{display:block;height:100%;background:#378ADD}}
.taskq{{max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.pt th small{{color:#5f5e5a;font-weight:400}}
.verdict{{background:#1b2430;border-left:3px solid #378ADD;padding:12px 16px;border-radius:0 8px 8px 0;margin:24px 0}}
</style></head><body>
<h1>本地模型基准评测报告</h1>
<div class="meta">题集 {report['taskset']} · 模式 {report['mode']} · 重复 {report['repeats']} 次 ·
生成于 {report['generated_at']}</div>
{f'<div class="verdict"><strong>最优组合：{best["label"]} {" ".join(f"{k}={v}" for k, v in best["params"].items())}</strong><br>准确率 {best["accuracy"] * 100:.1f}% · 平均 {best["avg_sec"]:.1f}s · {best["tok_per_sec"]:.1f} tok/s</div>' if best else ''}
<h2>组合排名</h2>
<table><thead><tr><th>#</th><th>后端 / 模型</th><th>参数</th><th>准确率</th><th></th>
<th>平均耗时</th><th>tok/s</th><th>样本数</th></tr></thead>
<tbody>{''.join(trs)}</tbody></table>
{per_task}
</body></html>"""
    path.write_text(html, encoding="utf-8")


# ---------------- 主流程 ----------------

def parse_sweep(specs):
    grid = {}
    for s in specs or []:
        if "=" not in s:
            sys.exit(f"[错误] --sweep 格式应为 key=v1,v2，收到：{s}")
        k, vs = s.split("=", 1)
        vals = []
        for v in vs.split(","):
            v = v.strip()
            try:
                vals.append(float(v) if "." in v else int(v))
            except ValueError:
                vals.append(v)
        grid[k.strip()] = vals
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])] if keys else [{}]
    return combos


def main():
    ap = argparse.ArgumentParser(description="本地模型参数/模型基准评测")
    ap.add_argument("--taskset", required=True, help="题集 JSON 路径")
    ap.add_argument("--profile", default="math_logic")
    ap.add_argument("--sweep", action="append", metavar="K=V1,V2", help="参数扫描（可重复）")
    ap.add_argument("--compare-endpoint", help="对比多个端点，逗号分隔")
    ap.add_argument("--endpoint", "-e")
    ap.add_argument("--model", "-m")
    ap.add_argument("--set", action="append", metavar="K=V")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--mode", choices=["single", "vote"], default="single")
    ap.add_argument("--vote-n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, help="只跑前 N 道题")
    ap.add_argument("--timeout", type=int)
    ap.add_argument("--tag", help="结果文件名前缀")
    ap.add_argument("--no-html", action="store_true")
    args = ap.parse_args()

    cfg, global_ov = load_config()
    profiles = load_profiles()
    ov = dict(global_ov)
    ov.update(parse_set(args.set))

    tsp = Path(args.taskset)
    if not tsp.exists():
        sys.exit(f"[错误] 找不到题集：{tsp}")
    ts = json.loads(tsp.read_text(encoding="utf-8"))
    tasks = ts.get("tasks", [])
    if args.limit:
        tasks = tasks[:args.limit]
    if not tasks:
        sys.exit("[错误] 题集为空")
    validator = ts.get("validator", "boxed_number")

    ep_names = ([n.strip() for n in args.compare_endpoint.split(",")]
                if args.compare_endpoint else [args.endpoint or cfg.get("active_endpoint")])

    param_combos = parse_sweep(args.sweep)

    # 展开成"组合"
    combos = []
    for ep_name in ep_names:
        ep_n, ep = resolve_endpoint(cfg, ep_name)
        model = args.model or ep.get("model")
        for pc in param_combos:
            sampling, dropped = build_sampling(args.profile, profiles, {**ov, **pc}, ep.get("capabilities"))
            if "max_tokens" not in sampling:
                sampling["max_tokens"] = 4096
            combos.append({
                "label": f"{ep_n} · {model}",
                "ep_name": ep_n, "ep": ep, "model": model,
                "params": pc, "sampling": sampling, "dropped": dropped,
            })

    print(f"[评测] 题集={tsp.name}  题数={len(tasks)}  组合数={len(combos)}  "
          f"重复={args.repeats}  模式={args.mode}" + (f" n={args.vote_n}" if args.mode == "vote" else ""),
          file=sys.stderr)
    if combos[0]["dropped"]:
        print(f"[提示] 端点不支持 {combos[0]['dropped']}，已丢弃。", file=sys.stderr)

    jobs = []
    ci = 0
    for c in combos:
        for t in tasks:
            for r in range(args.repeats):
                ci_key = f"{combos.index(c)}|{t.get('id')}"
                jobs.append({
                    "combo_idx": combos.index(c),
                    "task": t, "ep": c["ep"], "model": c["model"],
                    "sampling": c["sampling"], "validator": validator,
                    "mode": args.mode, "vote_n": args.vote_n,
                    "extract": ts.get("extract", "auto"),
                    "timeout": args.timeout, "seed": args.seed,
                    "repeat_idx": r, "system": ts.get("system"),
                })

    results = [None] * len(jobs)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=cfg.get("defaults", {}).get("max_concurrency", 4)) as ex:
        futs = {ex.submit(run_one, j, cfg, profiles): i for i, j in enumerate(jobs)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except SystemExit:
                raise
            except Exception as e:  # noqa
                results[i] = {"score": 0.0, "detail": f"异常：{e}", "sec": 0.0, "gen_tokens": 0}
            done += 1
            print(f"\r  进度 {done}/{len(jobs)}", end="", file=sys.stderr)
    print("", file=sys.stderr)

    # 汇总
    rows = []
    per_task_map = {}
    for ci_, c in enumerate(combos):
        rs = [r for i, r in enumerate(results) if jobs[i]["combo_idx"] == ci_ and r]
        if not rs:
            continue
        acc = sum(r["score"] for r in rs) / len(rs)
        secs = [r["sec"] for r in rs]
        toks = sum(r.get("gen_tokens", 0) for r in rs)
        rows.append({
            "label": c["label"], "params": c["params"],
            "accuracy": round(acc, 4),
            "avg_sec": round(sum(secs) / len(secs), 2),
            "tok_per_sec": round(toks / max(0.001, sum(secs)), 1),
            "runs": len(rs),
            "_map": {},
        })
        for i, r in enumerate(results):
            if jobs[i]["combo_idx"] == ci_:
                tid = jobs[i]["task"].get("id", f"t{i}")
                rows[-1]["_map"].setdefault(tid, []).append(r["score"])

    rows.sort(key=lambda x: (-x["accuracy"], x["avg_sec"]))
    for r in rows:
        r.pop("_map", None)

    if args.mode == "single" and args.repeats > 1:
        for ci_, c in enumerate(combos):
            m = {}
            for i, r in enumerate(results):
                if jobs[i]["combo_idx"] == ci_:
                    m.setdefault(jobs[i]["task"].get("id", f"t{i}"), []).append(r["score"])
            if m:
                per_task_map[ci_] = {k: sum(v) / len(v) for k, v in m.items()}

    task_ids = [t.get("id", i) for i, t in enumerate(tasks)]
    per_task = []
    for tid in task_ids:
        scores = []
        for ci_ in range(len(combos)):
            m = per_task_map.get(ci_, {})
            scores.append(m.get(tid, 0.0))
        prompt = next((t["prompt"] for t in tasks if t.get("id") == tid), "")
        per_task.append({"id": tid, "scores": scores, "prompt": prompt.replace('"', "&quot;")})

    report = {
        "taskset": tsp.name,
        "mode": args.mode,
        "repeats": args.repeats,
        "vote_n": args.vote_n if args.mode == "vote" else None,
        "profile": args.profile,
        "validator": validator,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
        "combos": rows,
        "per_task": per_task if args.mode == "single" else [],
    }

    OUT_DIR.mkdir(exist_ok=True)
    tag = args.tag or (tsp.stem + "_" + time.strftime("%m%d_%H%M"))
    json_p = OUT_DIR / f"bench_{tag}.json"
    csv_p = OUT_DIR / f"bench_{tag}.csv"
    html_p = OUT_DIR / f"report_{tag}.html"

    json_p.write_text(json.dumps({"report": report,
                                  "details": [{"task": jobs[i]["task"].get("id"),
                                               "combo": combos[jobs[i]["combo_idx"]]["label"],
                                               "params": combos[jobs[i]["combo_idx"]]["params"],
                                               "repeat": jobs[i]["repeat_idx"],
                                               **r} for i, r in enumerate(results) if r]},
                                 ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["task", "combo", "params", "repeat", "score", "sec", "gen_tokens", "detail"])
        for i, r in enumerate(results):
            if not r:
                continue
            c = combos[jobs[i]["combo_idx"]]
            w.writerow([jobs[i]["task"].get("id", i), c["label"],
                        json.dumps(c["params"], ensure_ascii=False),
                        jobs[i]["repeat_idx"], r["score"], round(r["sec"], 2),
                        r.get("gen_tokens", 0), str(r.get("detail", ""))[:200]])

    if not args.no_html:
        build_html(report, html_p)

    print("\n" + "=" * 64)
    print(f"{'#':<3}{'后端 / 模型':<34}{'参数':<26}{'准确率':>8}{'耗时':>9}{'tok/s':>8}")
    print("-" * 64)
    for i, r in enumerate(rows, 1):
        p = " ".join(f"{k}={v}" for k, v in r["params"].items()) or "(档位默认)"
        print(f"{i:<3}{r['label'][:33]:<34}{p[:25]:<26}{r['accuracy'] * 100:>7.1f}%{r['avg_sec']:>8.1f}s{r['tok_per_sec']:>8.1f}")
    print("=" * 64)
    print(f"总耗时 {report['elapsed_sec']}s")
    print(f"JSON  {json_p}")
    print(f"CSV   {csv_p}")
    if not args.no_html:
        print(f"HTML  {html_p}")


if __name__ == "__main__":
    main()
