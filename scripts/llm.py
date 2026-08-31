#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地模型统一客户端 —— 只依赖 Python 标准库，开箱即用。

设计要点：
  1. 面向 llama.cpp / ik_llama.cpp / LM Studio / vLLM 的 OpenAI 兼容端点。
  2. 参数档位从 profiles.json 读，后端不支持的参数按 config.json 的
     capabilities 自动降级丢弃（LM Studio 不支持 top_k / min_p，不会报错）。
  3. n>1 时用并发请求 + 不同 seed 实现，而不是服务端 n 参数
     —— 各后端对 chat/completions 的 n 支持不一致，这条路最稳。

命令行用法：
  python llm.py --user "写一个快排" --profile code_generate
  python llm.py --user "..." --profile creative_writing --set temperature=0.95
  python llm.py --user "..." --endpoint lm_studio --dump
  python llm.py --list-models
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
PROFILES_PATH = ROOT / "profiles.json"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# llama.cpp /v1/chat/completions 接受的采样参数（顶层）
SAMPLING_KEYS = [
    "temperature", "top_p", "top_k", "min_p", "typical_p",
    "repeat_penalty", "presence_penalty", "frequency_penalty",
    "dry_penalty", "xtc_threshold", "mirostat", "mirostat_tau", "mirostat_eta",
    "max_tokens", "seed", "stop", "n",
]


def die(msg, code=1):
    print(f"[错误] {msg}", file=sys.stderr)
    sys.exit(code)


def load_json(path, what):
    if not path.exists():
        die(f"找不到 {what}：{path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        die(f"{what} 不是合法 JSON：{path}\n  {e}")


def load_config():
    cfg = load_json(CONFIG_PATH, "配置文件")
    ov = cfg.get("global_sampling_overrides") or {}
    return cfg, {k: v for k, v in ov.items() if not k.startswith("_") and v is not None}


def load_profiles():
    data = load_json(PROFILES_PATH, "参数档位表")
    return data.get("profiles", {})


def resolve_endpoint(cfg, name=None):
    eps = cfg.get("endpoints", {})
    name = name or cfg.get("active_endpoint")
    if name not in eps:
        die(f"端点 '{name}' 不存在。可用：{', '.join(eps)}")
    return name, eps[name]


def build_sampling(profile_name, profiles, overrides, capability_filter=None):
    """拼装采样参数：档位 -> 全局覆盖 -> 命令行覆盖 -> 能力过滤"""
    params = {}
    if profile_name:
        if profile_name not in profiles:
            die(f"档位 '{profile_name}' 不存在。可用：{', '.join(profiles)}")
        src = profiles[profile_name]
        params = {k: v for k, v in src.items() if not k.startswith("_")}
    params.update(overrides or {})

    if capability_filter is not None:
        dropped = [k for k in params if k in SAMPLING_KEYS and k not in capability_filter]
        params = {k: v for k, v in params.items() if k not in dropped}
        return params, dropped
    return params, []


def http_post(url, payload, headers, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat_once(base_url, api_key, model, messages, sampling, timeout, retries, backoff, dump=False):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, "stream": False}
    payload.update(sampling)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if dump:
        print("---- REQUEST ----", file=sys.stderr)
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        print("-----------------", file=sys.stderr)

    last_err = None
    for attempt in range(retries + 1):
        try:
            return http_post(url, payload, headers, timeout)
        except urllib.error.URLError as e:
            last_err = e
            hint = ""
            if isinstance(e, urllib.error.HTTPError):
                try:
                    hint = e.read().decode("utf-8", "ignore")[:400]
                except Exception:
                    pass
            else:
                hint = ("连不上服务。检查：1) llama-server / LM Studio 是否已启动；"
                        "2) config.json 里的 base_url 端口是否正确；"
                        "3) 本机防火墙。")
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
            else:
                die(f"请求失败（第 {attempt + 1} 次）：{e}\n  {hint}")
        except Exception as e:  # noqa
            last_err = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
            else:
                die(f"请求异常：{e}")
    die(f"请求失败：{last_err}")


def extract_text(resp):
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def usage_of(resp):
    return (resp or {}).get("usage") or {}


def sample_many(base_url, api_key, model, messages, sampling, n, timeout, retries, backoff, concurrency, base_seed=None):
    """并发采样 n 次，seed 不同以保证多样性。"""
    seeds = []
    if base_seed is not None:
        seeds = [base_seed + i for i in range(n)]
    results = [None] * n
    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, n))) as ex:
        futs = {}
        for i in range(n):
            s = dict(sampling)
            if seeds:
                s["seed"] = seeds[i]
            futs[ex.submit(chat_once, base_url, api_key, model, messages,
                           s, timeout, retries, backoff)] = i
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except SystemExit:
                raise
            except Exception as e:  # noqa
                results[i] = {"_error": str(e)}
    return results


def parse_set(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            die(f"--set 格式应为 key=value，收到：{p}")
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            if v.lower() in ("true", "false"):
                v = v.lower() == "true"
            elif k in ("stop",):
                v = [x for x in v.split(",")]
            else:
                v = float(v) if ("." in v or "e" in v.lower()) else int(v)
        except ValueError:
            pass  # 保持字符串，比如 stop
        out[k] = v
    return out


def build_messages(system, user, files):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    elif True:
        pass
    body = user or ""
    for f in files or []:
        p = Path(f)
        if not p.exists():
            die(f"找不到文件：{p}")
        body = body + ("\n" if body else "") + p.read_text(encoding="utf-8", errors="replace")
    msgs.append({"role": "user", "content": body})
    return msgs


def main():
    ap = argparse.ArgumentParser(description="本地模型统一客户端")
    ap.add_argument("--user", "-u", help="用户消息")
    ap.add_argument("--sys", dest="system", help="系统提示词")
    ap.add_argument("--sys-file", help="系统提示词文件")
    ap.add_argument("--file", "-f", action="append", help="把文件内容追加到用户消息（可重复）")
    ap.add_argument("--profile", "-p", default="chat", help="参数档位名")
    ap.add_argument("--endpoint", "-e", help="端点名，默认用 config.json 的 active_endpoint")
    ap.add_argument("--model", "-m", help="覆盖模型名")
    ap.add_argument("--set", action="append", metavar="K=V", help="覆盖任意采样参数（可重复）")
    ap.add_argument("--n", type=int, default=1, help="采样次数（并发）")
    ap.add_argument("--seed", type=int, help="随机种子基准；配合 --n 时第 i 次用 seed+i")
    ap.add_argument("--concurrency", type=int, help="并发数，默认取 config.json")
    ap.add_argument("--timeout", type=int, help="超时秒数")
    ap.add_argument("--dump", action="store_true", help="打印请求体后退出前的调试信息")
    ap.add_argument("--json", action="store_true", help="输出原始 JSON")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--list-profiles", action="store_true")
    args = ap.parse_args()

    cfg, global_ov = load_config()
    profiles = load_profiles()
    d = cfg.get("defaults", {})

    if args.list_profiles:
        for name, p in profiles.items():
            label = p.get("_label", "")
            core = {k: v for k, v in p.items() if not k.startswith("_")}
            print(f"{name:<20} {label}")
            print(f"{'':<20} {core}")
        return

    ep_name, ep = resolve_endpoint(cfg, args.endpoint)

    if args.list_models:
        url = ep["base_url"].rstrip("/") + "/models"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {ep.get('api_key','')}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read().decode("utf-8"))
            for m in data.get("data", []):
                print(m.get("id"))
        except Exception as e:
            die(f"拉取模型列表失败：{e}")
        return

    if not args.user and not args.file:
        die("需要 --user 或 --file。用 --help 看用法。")

    sys_text = args.system
    if args.sys_file:
        sp = Path(args.sys_file)
        if not sp.exists():
            die(f"找不到系统提示词文件：{sp}")
        sys_text = sp.read_text(encoding="utf-8")

    overrides = dict(global_ov)
    overrides.update(parse_set(args.set))

    caps = ep.get("capabilities")
    sampling, dropped = build_sampling(args.profile, profiles, overrides, caps)
    if dropped:
        print(f"[提示] 端点 '{ep_name}' 不支持 {dropped}，已自动丢弃。", file=sys.stderr)

    model = args.model or ep.get("model")
    messages = build_messages(sys_text, args.user, args.file)

    timeout = args.timeout or d.get("timeout", 600)
    retries = d.get("retries", 2)
    backoff = d.get("retry_backoff_sec", 2)
    concurrency = args.concurrency or d.get("max_concurrency", 4)

    if "max_tokens" not in sampling:
        sampling["max_tokens"] = 2048

    t0 = time.time()

    if args.n > 1:
        resps = sample_many(ep["base_url"], ep.get("api_key", ""), model, messages,
                            sampling, args.n, timeout, retries, backoff,
                            concurrency, args.seed)
        texts = [extract_text(r) for r in resps]
        if args.json:
            print(json.dumps(resps, ensure_ascii=False, indent=2))
        else:
            for i, t in enumerate(texts, 1):
                print(f"\n===== 候选 {i} =====")
                print(t)
        print(f"\n[耗时] {time.time() - t0:.1f}s  端点={ep_name}  模型={model}  档位={args.profile}", file=sys.stderr)
    else:
        if args.seed is not None:
            sampling["seed"] = args.seed
        resp = chat_once(ep["base_url"], ep.get("api_key", ""), model, messages,
                         sampling, timeout, retries, backoff, dump=args.dump)
        if args.json:
            print(json.dumps(resp, ensure_ascii=False, indent=2))
        else:
            print(extract_text(resp))
            u = usage_of(resp)
            if u:
                tk = u.get("completion_tokens") or u.get("eval_count")
                if tk and u.get("total_tokens") is None:
                    pass
            print(f"\n[耗时] {time.time() - t0:.1f}s  端点={ep_name}  模型={model}  档位={args.profile}", file=sys.stderr)


if __name__ == "__main__":
    main()
