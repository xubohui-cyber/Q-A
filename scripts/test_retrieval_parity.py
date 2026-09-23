"""导出 Python 端 BM25 top-10，供浏览器端实现做逐条对比。"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieve import Retriever  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QUERIES = [
    "贵州茅台2025年营业收入是多少",
    "五粮液的毛利率",
    "合同负债",
    "2026年上半年归母净利润",
    "存货跌价准备",
    "经销商数量",
    "省外市场收入占比",
    "每10股派发现金红利",
    "销售费用率",
    "经营活动产生的现金流量净额",
]


def main():
    r = Retriever()
    out = []
    for q in QUERIES:
        scores = r.bm25(q)
        top = sorted(scores, key=lambda i: (-scores[i], i))[:10]
        out.append({"query": q, "python_top": top})
    path = os.path.join(ROOT, "data", "out", "parity_ref.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    sys.exit(main())
