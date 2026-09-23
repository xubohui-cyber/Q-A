"""由评测结果生成 Markdown 报告 eval/rag_eval.md（10 题逐题记录）。"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    data = json.load(open(os.path.join(ROOT, "web", "data", "eval.json"), encoding="utf-8"))
    questions = data["questions"]
    ok = [q for q in questions if q["verdict_class"] == "ok"]
    part = [q for q in questions if q["verdict_class"] == "part"]
    bad = [q for q in questions if q["verdict_class"] == "bad"]
    recalled = [q for q in questions if q["gold_recalled"]]
    panorama = [q for q in questions if q["panorama"]]

    L = []
    L.append("# 白酒行业年报问答：10 题逐题评测记录\n")
    L.append(f"检索配置：{data['retriever']}；语料为 16 家白酒上市公司 48 份年报/半年报"
             "（8,254 页，12,241 张表，16,709 个文本块）。\n")
    L.append("| 指标 | 结果 |\n| --- | --- |")
    L.append(f"| 题量 | {len(questions)} 题（含 3 道跨公司全景题、1 道语料外拒答题） |")
    L.append(f"| 关键证据被召回 | {len(recalled)}/{len(questions)} |")
    L.append(f"| 回答完全正确 | {len(ok)} |")
    L.append(f"| 部分正确 | {len(part)} |")
    L.append(f"| 错误 | {len(bad)} |")
    L.append("")
    L.append("## 结果一览\n")
    L.append("| # | 题目 | 类型 | 关键证据最高名次 | 判定 |\n| --- | --- | --- | --- | --- |")
    for i, q in enumerate(questions, 1):
        kind = "跨公司全景" if q["panorama"] else ("语料外" if q["extra"] else "单公司")
        rank = "—" if q["gold_rank"] is None else f"第 {q['gold_rank']} 位"
        L.append(f"| Q{i} | {q['question']} | {kind} | {rank} | {q['verdict']} |")
    L.append("")
    L.append("## 逐题记录\n")

    for i, q in enumerate(questions, 1):
        L.append(f"### Q{i}　{q['question']}\n")
        L.append(f"- **标准答案**：{q['expected']}")
        L.append(f"- **系统回答**：{q['answer']}")
        L.append(f"- **判定**：{q['verdict']}")
        if q.get("facts"):
            bits = []
            for f in q["facts"]:
                if q["panorama"]:
                    bits.append(f"{f['name']}（{f.get('companies_hit')}/{f.get('companies_total')} 家）")
                else:
                    bits.append(f"{f['name']}（{'第 %s 位' % f['hit_rank'] if f['recalled'] else '未召回'}）")
            L.append(f"- **关键事实召回**：" + "；".join(bits))
        L.append(f"- **错在哪 / 为什么**：{q['analysis']}")
        L.append("")
        if q["retrieved"]:
            L.append("召回的块：\n")
            L.append("| 名次 | 公司 | 报告期 | 章节 | 页码 | 类型 | BM25 名次 | 向量名次 | 关键证据 |")
            L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for r in q["retrieved"][:8]:
                L.append(
                    f"| #{r['rank']} | {r['company']} | {r['period']} | {r['section'][:22]}"
                    f"{(' / ' + r['subsection'][:16]) if r.get('subsection') else ''} | P{r['page']} | "
                    f"{'表格' if r['type'] == 'table' else '正文'} | "
                    f"{r['bm25_rank'] or '—'} | {r['dense_rank'] or '—'} | {'✔' if r['gold'] else ''} |"
                )
            L.append("")
        if q.get("panorama_table"):
            L.append("全景指标解析结果（自动从「主要会计数据」表解析）：\n")
            L.append("| 公司 | 营业收入 | 营收同比 | 归母净利润 | 净利同比 | 出处页 |")
            L.append("| --- | --- | --- | --- | --- | --- |")
            for r in sorted(q["panorama_table"], key=lambda r: -(r["revenue"] or 0)):
                f = lambda v: "—" if v is None else f"{v/1e8:,.2f} 亿元"
                L.append(
                    f"| {r['company']} | {f(r['revenue'])} | "
                    f"{'—' if r['revenue_yoy'] is None else str(r['revenue_yoy']) + '%'} | "
                    f"{f(r['profit'])} | "
                    f"{'—' if r['profit_yoy'] is None else str(r['profit_yoy']) + '%'} | "
                    f"P{r.get('revenue_page') or r.get('page')} |"
                )
            L.append("")

    path = os.path.join(ROOT, "eval", "rag_eval.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write("\n".join(L))
    json.dump(data, open(os.path.join(ROOT, "eval", "eval.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("wrote", path, f"({len(questions)} 题)")


if __name__ == "__main__":
    sys.exit(main())
