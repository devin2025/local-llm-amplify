#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四阶段推理流水线：规划 -> 求解 -> 批判 -> 修订。

为什么有效：本地小模型一次性从头生成到尾时，前面犯的错会一路带到底，
而且它没有余力回头检查。拆成四步之后：
  1. 每一步上下文更短，注意力不被稀释；
  2. 批判阶段用低温裁判档，等于给模型一次"跳出自己"的机会；
  3. 修订阶段只看方案 + 意见，不看原始错误输出，避免被自己带偏。

实测对"写代码"和"复杂推理"提升最大，对短问答是负优化（过度工程，别用）。

用法：
  python pipeline.py --mode code   --task-file req.txt --verify tests.py
  python pipeline.py --mode reason --task " ... " --rounds 2
  python pipeline.py --mode write  --task-file brief.txt
  python pipeline.py --mode script --task "志怪题材：..." --rounds 2
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm import (  # noqa: E402
    load_config, load_profiles, resolve_endpoint, build_sampling,
    chat_once, extract_text, usage_of, parse_set,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MODES = {
    "code": {
        "label": "写代码 / 改 bug",
        "plan": {
            "profile": "code_debug",
            "system": ("你是资深软件架构师。用户会给出一个需求。\n"
                       "只输出实现方案，不要写完整代码，不要客套。\n"
                       "结构：\n"
                       "1. 模块/函数划分\n"
                       "2. 关键数据结构\n"
                       "3. 边界条件与易错点（这是重点，至少列 3 条）\n"
                       "4. 实现顺序\n"
                       "控制在 500 字以内。")},
        "solve": {
            "profile": "code_generate",
            "system": ("按给定方案实现。\n"
                       "要求：\n"
                       "- 输出完整可运行代码，用 ``` 包裹\n"
                       "- 代码之外不要写解释、不要写总结\n"
                       "- 显式处理方案里列出的每一个边界条件")},
        "critique": {
            "profile": "judge",
            "system": ("你是严格的代码审查者，默认这段代码有 bug。\n"
                       "逐条检查：正确性、边界条件、异常路径、命名、性能陷阱。\n"
                       "严格按此格式输出：\n"
                       "ISSUES:\n"
                       "- <具体问题描述>（严重度: 高/中/低）\n"
                       "VERDICT: PASS 或 FAIL")},
        "revise": {
            "profile": "code_generate",
            "system": ("根据审查意见修订代码。\n"
                       "只输出修订后的完整代码，用 ``` 包裹，不要解释改了什么。")},
        "extract": "code",
    },

    "reason": {
        "label": "数学 / 逻辑推理",
        "plan": {
            "profile": "math_logic",
            "system": ("你是数学与逻辑专家。面对下面的问题：\n"
                       "1. 列出已知量、未知量、约束条件\n"
                       "2. 列出所有可能适用的方法（至少 2 种）\n"
                       "3. 选出最可靠的一种，并说明为什么排除其他的\n"
                       "不要计算最终答案，只做路径选择。")},
        "solve": {
            "profile": "math_logic",
            "system": ("按选定路径逐步求解。\n"
                       "要求：每一步写出依据，不要跳步。\n"
                       "最后一行必须用 \\boxed{答案} 给出最终结果。")},
        "critique": {
            "profile": "judge",
            "system": ("逐步检查下面的解答。对每一步判断三件事：\n"
                       "推理是否成立、计算是否正确、是否遗漏分支或特殊情况。\n"
                       "严格按此格式输出：\n"
                       "ERRORS:\n"
                       "- <第几步，错在哪>（无错误则写：无）\n"
                       "VERDICT: PASS 或 FAIL")},
        "revise": {
            "profile": "math_logic",
            "system": ("根据检查意见给出修正后的完整解答。\n"
                       "保留正确的部分，逐条修复错误。\n"
                       "最后一行必须用 \\boxed{答案} 给出最终结果。")},
        "extract": "boxed",
    },

    "write": {
        "label": "长文写作 / 翻译 / 总结",
        "plan": {
            "profile": "summarize",
            "system": ("你是资深编辑。为下面的写作任务制定提纲，不要写正文。\n"
                       "输出：\n"
                       "1. 核心立意（一句话）\n"
                       "2. 结构分段，每段标注要达成的情绪或信息目标\n"
                       "3. 这个题材最容易踩的俗套，明确列出要避开的东西\n"
                       "4. 语言基调（句式长短、人称、用词域）")},
        "solve": {
            "profile": "creative_writing",
            "system": ("严格按提纲写正文。\n"
                       "硬要求：\n"
                       "- 不要排比句堆砌，不要'不仅...而且...'式套话\n"
                       "- 不要空洞形容词（'深深地''无比''令人深思'）\n"
                       "- 每段必须有具体信息：人名、数字、动作、物件\n"
                       "- 克制使用破折号\n"
                       "只输出正文。")},
        "critique": {
            "profile": "judge",
            "system": ("你是苛刻的审稿人。检查下面的文章：\n"
                       "1. AI 腔：排比堆砌、空洞抒情、三段式、破折号泛滥、\n"
                       "   '不是A，而是B'式否定排比\n"
                       "2. 重复：同一个句式/意象/词是否出现两次以上\n"
                       "3. 信息密度：有没有实质内容，还是全是情绪词\n"
                       "4. 是否跑题\n"
                       "严格按此格式输出：\n"
                       "ISSUES:\n"
                       "- <问题>\n"
                       "SCORE: <1-10>\n"
                       "VERDICT: PASS 或 FAIL")},
        "revise": {
            "profile": "creative_writing",
            "system": ("根据审稿意见重写。\n"
                       "保留写得好的部分，逐条修复列出的问题。\n"
                       "只输出正文，不要说明改了什么。")},
        "extract": "none",
    },

    "script": {
        "label": "故事 / 情绪类视频脚本",
        "plan": {
            "profile": "summarize",
            "system": ("你是短片编剧，擅长志怪、历史、叙事类题材。\n"
                       "为下面的题材做前期设计，不要写正文。输出：\n"
                       "1. 高概念（一句话，说清'什么人被置于什么处境'）\n"
                       "2. 情感弧线：起 / 承 / 转 / 合 各自对应的情绪\n"
                       "3. 意象系统：2-3 个贯穿全片、可反复出现的具体物象\n"
                       "4. 开场 3 秒钩子、结尾落点（最后一句台词或画面）\n"
                       "5. 旁白声口：人称、年代感、语速、句长")},
        "solve": {
            "profile": "video_script_story",
            "system": ("按前期设计写完整脚本。\n"
                       "格式：[画面] 描述具体可见的东西；[旁白] 写念出来顺口的短句。\n"
                       "硬要求：\n"
                       "- 每一句旁白都要能配上一个具体画面，抽象词必须换成实物\n"
                       "- 禁止空泛抒情（'岁月沧桑''令人唏嘘'）\n"
                       "- 句式长短交替，不要全是整句\n"
                       "- 情绪靠细节推进，不靠形容词加码")},
        "critique": {
            "profile": "judge",
            "system": ("你是短视频平台的资深编审，只关心观众会不会划走。\n"
                       "检查：\n"
                       "1. 前 3 秒有没有钩子，还是平铺直叙\n"
                       "2. 全片有没有一个能被记住的画面\n"
                       "3. 旁白念出来像不像人话（有没有书面语、长从句）\n"
                       "4. 情绪是递进的还是从头平到尾\n"
                       "5. AI 腔检测\n"
                       "严格按此格式输出：\n"
                       "ISSUES:\n"
                       "- <问题>\n"
                       "SCORE: <1-10>\n"
                       "VERDICT: PASS 或 FAIL")},
        "revise": {
            "profile": "video_script_story",
            "system": ("根据编审意见重写脚本。\n"
                       "保留最好的那个画面和最好的那句旁白，其余逐条修复。\n"
                       "只输出新脚本，格式保持 [画面] / [旁白]。")},
        "extract": "none",
    },
}


def detect_mode(task):
    """关键词路由。宁可漏判落到默认 write（人工可纠正），不要错判进 code/script。"""
    t = task.lower()
    # 守卫：bash/shell 脚本是代码，不是视频脚本
    if re.search(r"(bash|shell|powershell|python|批处理|bat)\s*脚本", t):
        return "code"
    code_kw = ["def ", "class ", "import ", "```", "函数", "实现", "写代码", "改 bug",
               "重构", "debug", "报错", "traceback"]
    script_kw = ["分镜", "旁白", "短视频", "口播", "视频号", "抖音", "短片", "预告片", "脚本"]
    reason_kw = ["计算", "求解", "证明", "概率", "推理", "逻辑", "方程", "数列", "几何",
                 "百分比", "分数", "等于多少", "多少", "几小时", "几分钟", "几个人",
                 "几岁", "几分之", "速度", "相遇", "方案数"]
    if any(k in t for k in code_kw):
        return "code"
    if any(k in t for k in script_kw):
        return "script"
    if any(k in t for k in reason_kw):
        return "reason"
    return "write"


def read_task(args):
    if args.task_file:
        p = Path(args.task_file)
        if not p.exists():
            sys.exit(f"[错误] 找不到任务文件：{p}")
        t = p.read_text(encoding="utf-8")
    elif args.task:
        t = args.task
    else:
        sys.exit("[错误] 需要 --task 或 --task-file")
    if args.context:
        cp = Path(args.context)
        if not cp.exists():
            sys.exit(f"[错误] 找不到背景文件：{cp}")
        t = f"{t}\n\n===== 背景资料 =====\n{cp.read_text(encoding='utf-8')}"
    return t


def call_stage(stage_name, task, prior, mode_cfg, ep, model, profiles, cfg, args, overrides):
    scfg = mode_cfg[stage_name]
    sampling, dropped = build_sampling(scfg["profile"], profiles, overrides, ep.get("capabilities"))
    if "max_tokens" not in sampling:
        sampling["max_tokens"] = 4096

    parts = []
    if stage_name == "solve":
        parts.append(f"【任务】\n{task}")
        parts.append(f"【已确定的方案】\n{prior['plan']}")
    elif stage_name == "critique":
        parts.append(f"【原始任务】\n{task}")
        parts.append(f"【待审查的产出】\n{prior['solve']}")
    elif stage_name == "revise":
        parts.append(f"【原始任务】\n{task}")
        parts.append(f"【当前产出】\n{prior['solve']}")
        parts.append(f"【审查意见】\n{prior['critique']}")
    else:
        parts.append(task)

    messages = [
        {"role": "system", "content": scfg["system"]},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    d = cfg.get("defaults", {})
    if args.seed is not None:
        sampling["seed"] = args.seed
    t0 = time.time()
    resp = chat_once(ep["base_url"], ep.get("api_key", ""), model, messages, sampling,
                     args.timeout or d.get("timeout", 600), d.get("retries", 2),
                     d.get("retry_backoff_sec", 2))
    text = extract_text(resp)
    return text, time.time() - t0, dropped


def verdict_of(text):
    m = re.search(r"VERDICT\s*:\s*(PASS|FAIL)", text, re.I)
    return m.group(1).upper() if m else None


def main():
    ap = argparse.ArgumentParser(description="四阶段推理流水线")
    ap.add_argument("--task", "-t")
    ap.add_argument("--task-file")
    ap.add_argument("--context", help="背景资料文件，会拼进任务里")
    ap.add_argument("--mode", choices=list(MODES) + ["auto"], default="auto")
    ap.add_argument("--rounds", type=int, default=1, help="批判-修订循环轮数，默认 1")
    ap.add_argument("--endpoint", "-e")
    ap.add_argument("--model", "-m")
    ap.add_argument("--set", action="append", metavar="K=V")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--timeout", type=int)
    ap.add_argument("--verify", help="code 模式：Python 测试文件，每轮修订后跑一遍")
    ap.add_argument("--verify-cmd", help="code 模式：shell 验证命令，输出需匹配 --expect")
    ap.add_argument("--expect", help="配合 --verify-cmd，stdout 需包含该字符串")
    ap.add_argument("--out", help="结果存成 JSON")
    ap.add_argument("--out-dir", help="每一阶段单独存文件的目录")
    ap.add_argument("--quiet", "-q", action="store_true", help="只打印最终产出")
    args = ap.parse_args()

    task = read_task(args)
    mode = args.mode
    if mode == "auto":
        mode = detect_mode(task)
        print(f"[自动路由] 判定为 '{mode}'（{MODES[mode]['label']}）。"
              f"判错的话用 --mode 手动指定。", file=sys.stderr)

    cfg, global_ov = load_config()
    profiles = load_profiles()
    ep_name, ep = resolve_endpoint(cfg, args.endpoint)
    model = args.model or ep.get("model")
    mode_cfg = MODES[mode]

    ov = dict(global_ov)
    ov.update(parse_set(args.set))

    test_code = None
    if args.verify:
        vp = Path(args.verify)
        if not vp.exists():
            sys.exit(f"[错误] 找不到测试文件：{vp}")
        test_code = vp.read_text(encoding="utf-8")

    prior = {}
    log = []
    total_t0 = time.time()

    def show(title, text, sec):
        if args.quiet:
            return
        print(f"\n{'=' * 60}\n{title}   ({sec:.1f}s)\n{'=' * 60}")
        print(text)

    text, sec, _ = call_stage("plan", task, prior, mode_cfg, ep, model, profiles, cfg, args, ov)
    prior["plan"] = text
    log.append({"stage": "plan", "sec": round(sec, 1), "text": text})
    show("第 1 步 · 规划", text, sec)

    text, sec, _ = call_stage("solve", task, prior, mode_cfg, ep, model, profiles, cfg, args, ov)
    prior["solve"] = text
    log.append({"stage": "solve", "sec": round(sec, 1), "text": text})
    show("第 2 步 · 求解", text, sec)

    final = prior["solve"]

    for r in range(1, args.rounds + 1):
        text, sec, _ = call_stage("critique", task, prior, mode_cfg, ep, model, profiles, cfg, args, ov)
        prior["critique"] = text
        log.append({"stage": f"critique#{r}", "sec": round(sec, 1), "text": text})
        show(f"第 {2 + r} 步 · 批判（第 {r} 轮）", text, sec)

        v = verdict_of(text)
        if v == "PASS" and r == args.rounds:
            print("[跳过修订] 审查结论为 PASS。", file=sys.stderr)
            break

        text, sec, _ = call_stage("revise", task, prior, mode_cfg, ep, model, profiles, cfg, args, ov)
        prior["solve"] = text
        log.append({"stage": f"revise#{r}", "sec": round(sec, 1), "text": text})
        show(f"第 {3 + r} 步 · 修订（第 {r} 轮）", text, sec)
        final = text

        if test_code:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from vote import extract_answer, run_python_check
            code = extract_answer(text, "code")
            ok, out = run_python_check(code, test_code)
            tag = "通过" if ok else "失败"
            print(f"\n[代码验证] {tag}")
            if not ok:
                print(out[-600:], file=sys.stderr)
            log.append({"stage": f"verify#{r}", "passed": ok, "log": out[-800:]})

    if args.verify_cmd:
        import subprocess
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from vote import extract_answer
        code = extract_answer(final, "code")
        with open("_pipeline_verify.py", "w", encoding="utf-8") as f:
            f.write(code)
        try:
            r = subprocess.run(args.verify_cmd, shell=True, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=120)
            out = r.stdout + r.stderr
            ok = (args.expect in out) if args.expect else (r.returncode == 0)
            print(f"\n[命令验证] {'通过' if ok else '失败'}")
            if not ok:
                print(out[-600:], file=sys.stderr)
            log.append({"stage": "verify_cmd", "passed": ok, "log": out[-800:]})
        finally:
            Path("_pipeline_verify.py").unlink(missing_ok=True)

    payload = {
        "mode": mode,
        "task": task,
        "endpoint": ep_name,
        "model": model,
        "rounds": args.rounds,
        "elapsed_sec": round(time.time() - total_t0, 1),
        "stages": log,
        "final": final,
    }

    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.out_dir:
        od = Path(args.out_dir)
        od.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(log, 1):
            (od / f"{i:02d}_{s['stage'].replace('#', '_')}.md").write_text(s["text"], encoding="utf-8")
        (od / "final.md").write_text(final, encoding="utf-8")
        (od / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[已保存] {od}", file=sys.stderr)

    if args.quiet:
        print(final)
    else:
        print(f"\n[总耗时] {payload['elapsed_sec']}s  模式={mode}  端点={ep_name}  模型={model}", file=sys.stderr)


if __name__ == "__main__":
    main()
