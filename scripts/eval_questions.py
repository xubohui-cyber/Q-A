"""10 道评测题：跑混合检索，记录召回的块、是否命中关键证据，并导出原始 JSON。

用法：
    python eval_questions.py dump     # 打印每题的召回证据（人工核对用）
    python eval_questions.py build    # 合并人工判定，输出 web/data/eval.json
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieve import Retriever  # noqa: E402
from panorama import metric_of, fmt_yi  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "out", "eval_raw.json")
CURATED = os.path.join(ROOT, "eval", "annotations.json")
WEB_OUT = os.path.join(ROOT, "web", "data", "eval.json")

# gold: 关键证据必须包含的片段（用于自动判定“有没有召回对”）
QUESTIONS = [
    {
        "id": "Q1",
        "question": "贵州茅台2025年营业收入和归母净利润是多少？",
        "opts": {"code": "600519", "period": "2025A", "top_k": 8, "auto_filter": False},
        "gold": {"code": "600519", "period": "2025A", "facts": [
            {"name": "2025年营业收入 1688.38 亿元", "any_of": ["168,838,102,514.79", "16,883,810.25"]},
            {"name": "2025年归母净利润 823.20 亿元", "any_of": ["82,320,067,101.68"]},
            {"name": "同比增速 -1.21% / -4.53%", "any_of": ["-1.21", "-4.53"]},
        ]},
        "panorama": False,
    },
    {
        "id": "Q2",
        "question": "五粮液2025年酒类产品的毛利率是多少？",
        "opts": {"code": "000858", "period": "2025A", "top_k": 8, "auto_filter": False},
        "gold": {"code": "000858", "period": "2025A", "facts": [
            {"name": "酒类毛利率（分产品表）", "any_of": ["毛利率"]},
        ]},
        "panorama": False,
    },
    {
        "id": "Q3",
        "question": "山西汾酒2025年省外市场收入是多少？",
        "opts": {"code": "600809", "period": "2025A", "top_k": 8, "auto_filter": False},
        "gold": {"code": "600809", "period": "2025A", "facts": [
            {"name": "合并口径省外收入 252.02 亿元", "any_of": ["25,202,184,766.32"]},
            {"name": "分地区（省内/省外）表", "any_of": ["省外"]},
        ]},
        "panorama": False,
    },
    {
        "id": "Q4",
        "question": "泸州老窖2026年上半年营业收入和归母净利润同比变化？",
        "opts": {"code": "000568", "period": "2026H1", "top_k": 8, "auto_filter": False},
        "gold": {"code": "000568", "period": "2026H1", "facts": [
            {"name": "H1 营业收入 104.72 亿元（-36.35%）", "any_of": ["10,472,224,575.78"]},
            {"name": "H1 归母净利润 43.39 亿元（-43.37%）", "any_of": ["4,339,264,086.33"]},
        ]},
        "panorama": False,
    },
    {
        "id": "Q5",
        "question": "洋河股份2025年末合同负债余额是多少？",
        "opts": {"code": "002304", "period": "2025A", "top_k": 8, "auto_filter": False},
        "gold": {"code": "002304", "period": "2025A", "facts": [
            {"name": "合同负债期末余额 75.29 亿元", "any_of": ["7,529,047,335.12"]},
        ]},
        "panorama": False,
    },
    {
        "id": "Q6",
        "question": "今世缘2025年度利润分配方案是什么？",
        "opts": {"code": "603369", "period": "2025A", "top_k": 8, "auto_filter": False},
        "gold": {"code": "603369", "period": "2025A", "facts": [
            {"name": "每 10 股派息 12 元 / 合计 14.96 亿元", "any_of": ["1,496,160,044.40"]},
        ]},
        "panorama": False,
    },
    {
        "id": "Q7",
        "question": "口子窖2025年年报披露了哪些经营风险？",
        "opts": {"code": "603589", "period": "2025A", "top_k": 8, "auto_filter": False},
        "gold": {"code": "603589", "period": "2025A", "facts": [
            {"name": "第三节「可能面对的风险」", "any_of": ["可能面对的风险", "可能面临的风险"]},
        ]},
        "panorama": False,
    },
    {
        "id": "Q8",
        "question": "全景：2025年营业收入最高的5家白酒公司是谁？",
        "opts": {"period": "2025A", "top_k": 3, "auto_filter": False},
        "panorama": True,
        "gold": {"period": "2025A", "facts": [
            {"name": "各公司 2025 年营业收入", "any_of": ["营业收入"]},
        ]},
    },
    {
        "id": "Q9",
        "question": "全景：2025年归母净利润同比增速最高的3家公司？",
        "opts": {"period": "2025A", "top_k": 3, "auto_filter": False},
        "panorama": True,
        "gold": {"period": "2025A", "facts": [
            {"name": "各公司 2025 年归母净利润及同比", "any_of": ["归属于上市公司股东的净利润"]},
        ]},
    },
    {
        "id": "Q10",
        "question": "全景：2026年上半年哪些白酒公司营业收入同比下滑？",
        "opts": {"period": "2026H1", "top_k": 3, "auto_filter": False},
        "panorama": True,
        "gold": {"period": "2026H1", "facts": [
            {"name": "各公司 2026H1 营业收入及同比", "any_of": ["营业收入"]},
        ]},
    },
]

# 附加：语料里没有答案的问题，用来检验会不会“编造”
EXTRA = {
    "id": "Q11",
    "question": "贵州茅台2026年第三季度单季营业收入是多少？",
    "opts": {"code": "600519", "top_k": 5, "auto_filter": False},
    "gold": {"code": "600519", "facts": [
        {"name": "2026 年第三季度数据（语料中不存在）", "any_of": ["2026年第三季度", "2026 年第三季度"]},
    ]},
    "panorama": False,
    "extra": True,
}


def run(mode="hybrid"):
    r = Retriever()
    out = []
    raw_path = RAW if mode == "hybrid" else RAW.replace(".json", f"_{mode}.json")
    for spec in QUESTIONS + [EXTRA]:
        if spec["panorama"]:
            res = r.search_grouped(spec["question"], per_company=spec["opts"]["top_k"],
                                   period=spec["opts"].get("period"))
            items = []
            for code, g in res["groups"].items():
                for c in g["results"]:
                    items.append({**c, "rank": len(items) + 1})
        else:
            res = r.search(spec["question"], top_k=spec["opts"]["top_k"],
                           code=spec["opts"].get("code"), period=spec["opts"].get("period"),
                           auto_filter=spec["opts"].get("auto_filter", True), mode=mode)
            items = res["results"]
        gold = spec["gold"]
        facts = [dict(f) for f in gold["facts"]]
        if spec["panorama"]:
            groups = res["groups"]
            for f in facts:
                hit = [
                    code for code, g in groups.items()
                    if any(any(s in c["text"] for s in f["any_of"]) for c in g["results"])
                ]
                f["hit_rank"] = None
                f["companies_hit"] = len(hit)
                f["companies_total"] = len(groups)
                f["recalled"] = len(hit) == len(groups)
        else:
            for f in facts:
                f["hit_rank"], f["recalled"] = None, False
                for i, c in enumerate(items, 1):
                    if gold.get("code") and c["code"] != gold["code"]:
                        continue
                    if gold.get("period") and gold["period"] not in c["period"]:
                        continue
                    if any(s in c["text"] for s in f["any_of"]):
                        f["hit_rank"], f["recalled"] = i, True
                        break
        recalled = all(f["recalled"] for f in facts)
        gold_rank = min((f["hit_rank"] for f in facts if f.get("hit_rank")), default=None)
        out.append({
            "id": spec["id"],
            "question": spec["question"],
            "panorama": spec["panorama"],
            "extra": spec.get("extra", False),
            "opts": spec["opts"],
            "filter": res.get("filter", {}),
            "gold": gold,
            "facts": facts,
            "n_facts_recalled": sum(1 for f in facts if f["recalled"]),
            "gold_recalled": recalled,
            "gold_rank": gold_rank,
            "n_companies": len(res.get("groups", {})) if spec["panorama"] else None,
            "panorama_table": (
                panorama_table(r, res, spec["opts"].get("period")) if spec["panorama"] else None
            ),
            "retrieved": [
                {
                    "rank": i,
                    "idx": c["idx"],
                    "company": c["name"],
                    "code": c["code"],
                    "period": c["period"],
                    "report": c["title"],
                    "section": c["section"],
                    "subsection": c["subsection"],
                    "page": c["page_start"],
                    "type": c["type"],
                    "score": round(c["score"], 4),
                    "bm25": float(c["bm25"]),
                    "cos": float(c["cos"]) if c["cos"] is not None else None,
                    "bm25_rank": (res.get("bm25_top") or []).index(c["idx"]) + 1 if c["idx"] in (res.get("bm25_top") or []) else None,
                    "dense_rank": (res.get("dense_top") or []).index(c["idx"]) + 1 if c["idx"] in (res.get("dense_top") or []) else None,
                    "gold": bool(
                        (not gold.get("code") or c["code"] == gold["code"])
                        and (not gold.get("period") or gold["period"] in c["period"])
                        and any(s in c["text"] for f in facts for s in f["any_of"])
                    ),
                    "text": c["text"][:2500],
                }
                for i, c in enumerate(items, 1)
            ],
        })
    json.dump({"questions": out, "mode": mode}, open(raw_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print("wrote", raw_path)


def panorama_table(r, res, period, per_company=3):
    """把每家公司的「主要会计数据」表解析成可比较的数字。

    营业收入与归母净利润各自用指标化的查询再召回一次，避免一张表被 PDF 拆到不同块里。
    """
    metrics = {
        "revenue": "主要会计数据 营业收入 上年同期 增减",
        "profit": "主要会计数据 归属于上市公司股东的净利润 上年同期 增减",
    }
    fetched = {
        k: r.search_grouped(q, per_company=per_company, period=period)
        for k, q in metrics.items()
    }
    rows = []
    for code, g in sorted(res["groups"].items()):
        row = {"company": g["name"], "code": code,
               "page": g["results"][0]["page_start"], "url": g["results"][0]["url"]}
        for key in ("revenue", "profit"):
            got = None
            grp = fetched[key]["groups"].get(code)
            if grp:
                merged = "\n".join(c["text"] for c in grp["results"])
                got = metric_of(merged, key)
                if got:
                    row[f"{key}_page"] = grp["results"][0]["page_start"]
            row[key] = float(got[0]) if got else None
            row[f"{key}_prev"] = float(got[1]) if got else None
            row[f"{key}_yoy"] = got[2] if got else None
        rows.append(row)
    return rows


def dump():
    data = json.load(open(RAW, encoding="utf-8"))
    for q in data["questions"]:
        print("=" * 100)
        print(f"{q['id']}  {q['question']}   [filter={q['filter']}]")
        for f in q.get("facts", []):
            if q["panorama"]:
                print(f"  事实「{f['name']}」: 命中公司 {f.get('companies_hit')}/{f.get('companies_total')}")
            else:
                print(f"  事实「{f['name']}」: 召回={f['recalled']} 最高名次={f['hit_rank']}")
        print(f"  全部关键事实召回: {q['gold_recalled']}  最高名次 {q['gold_rank']}")
        limit = 16 if q["panorama"] else 8
        width = 600 if q["panorama"] else 300
        for r in q["retrieved"][:limit]:
            print(f"  #{r['rank']} {r['company']} {r['period']} | {r['section']} / {r['subsection']} | P{r['page']} "
                  f"| {r['type']} | score={r['score']} bm25#{r['bm25_rank']} vec#{r['dense_rank']} {'★GOLD' if r['gold'] else ''}")
            print("      " + r["text"][:width].replace("\n", " ⏎ "))


def build():
    raw = json.load(open(RAW, encoding="utf-8"))
    curated = json.load(open(CURATED, encoding="utf-8"))
    cq = {c["id"]: c for c in curated["questions"]}
    out = []
    for q in raw["questions"]:
        c = cq.get(q["id"], {})
        verdict = c.get("verdict", "未判定")
        if verdict.startswith("正确"):
            vclass = "ok"
        elif verdict.startswith("部分"):
            vclass = "part"
        else:
            vclass = "bad"
        out.append({
            "id": q["id"],
            "question": q["question"],
            "panorama": q["panorama"],
            "extra": q["extra"],
            "expected": c.get("expected", ""),
            "answer": c.get("answer", ""),
            "verdict": verdict,
            "verdict_class": vclass,
            "analysis": c.get("analysis", ""),
            "gold_recalled": q["gold_recalled"],
            "gold_rank": q["gold_rank"],
            "facts": q.get("facts", []),
            "retrieved": [
                {k: v for k, v in r.items() if k != "text"}
                for r in q["retrieved"]
            ],
            "panorama_table": q.get("panorama_table"),
        })
    payload = {
        "retriever": "BM25(字符 bigram) + bge-small-zh-v1.5 向量，α=0.5，Top-8",
        "questions": out,
    }
    os.makedirs(os.path.dirname(WEB_OUT), exist_ok=True)
    json.dump(payload, open(WEB_OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=float)
    print("wrote", WEB_OUT, len(out), "题")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "dump":
        dump()
    elif cmd == "build":
        build()
    else:
        run("bm25" if cmd == "bm25" else "hybrid")
