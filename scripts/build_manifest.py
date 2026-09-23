"""生成 data/manifest.json：列出全部报告的公司、报告期、原始 PDF 链接、页数、表格数、块数。

这份清单是「数据从哪来」的证据，网页与评测结论都可以据此回溯到巨潮资讯的原文。
"""

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    meta = json.load(open(os.path.join(ROOT, "data", "reports_meta.json"), encoding="utf-8"))
    chunk_counts = {}
    with open(os.path.join(ROOT, "data", "out", "chunks.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            key = f"{c['code']}_{c['period']}"
            chunk_counts[key] = chunk_counts.get(key, 0) + 1

    reports = []
    for key, m in sorted(meta.items(), key=lambda kv: (kv[1]["code"], kv[1]["period"])):
        text_path = os.path.join(ROOT, "data", "text", key + ".json")
        pdf_path = os.path.join(ROOT, m["file"].replace("data/raw_pdf", "data/raw_pdf"))
        info = {
            "key": key,
            "code": m["code"],
            "name": m["name"],
            "exchange": "上海证券交易所" if m["exchange"] == "sse" else "深圳证券交易所",
            "period": m["period"],
            "kind": "年度报告" if m["kind"] == "annual" else "半年度报告",
            "title": m["title"],
            "announcement_date": m["announcement_date"],
            "source": m["source"],
            "pdf_url": m["url"],
            "chunks": chunk_counts.get(key, 0),
        }
        if os.path.exists(text_path):
            t = json.load(open(text_path, encoding="utf-8"))
            info["pages"] = t["num_pages"]
            info["tables"] = t["num_tables"]
            info["chars"] = sum(len(p["text"]) for p in t["pages"])
        if os.path.exists(pdf_path):
            info["pdf_bytes"] = os.path.getsize(pdf_path)
            info["pdf_sha256"] = sha256(pdf_path)[:16]
        reports.append(info)

    out = {
        "generated": "2026-09-23",
        "source": "巨潮资讯网（cninfo.com.cn）年度报告 / 半年度报告全文",
        "note": "PDF 原文按本清单的 pdf_url 可重新下载（scripts/download_reports.py）；仓库不存放 PDF 以控制体积。",
        "stats": {
            "companies": len({r["code"] for r in reports}),
            "reports": len(reports),
            "pages": sum(r.get("pages", 0) for r in reports),
            "tables": sum(r.get("tables", 0) for r in reports),
            "chars": sum(r.get("chars", 0) for r in reports),
            "chunks": sum(r["chunks"] for r in reports),
        },
        "reports": reports,
    }
    path = os.path.join(ROOT, "data", "manifest.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", path, out["stats"])


if __name__ == "__main__":
    sys.exit(main())
