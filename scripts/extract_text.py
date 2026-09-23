"""把年报 / 半年报 PDF 解析成「保留行列结构的文本」。

策略：
1. PyMuPDF 读取每个 PDF 的文字块（blocks）与表格（find_tables）；
2. 表格用 find_tables() 还原行列并转成 Markdown，正文块按纵坐标与表格合并成阅读顺序；
3. 落入表格区域内的正文块会被丢弃，避免同一内容重复出现；
4. 输出每份报告一个 JSON：页号 + 页面文本 + 表格清单（含行列数）。
"""

import json
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(ROOT, "data", "reports_meta.json")
TEXT_DIR = os.path.join(ROOT, "data", "text")

SECTION = re.compile(
    r"^\s*第\s*[一二三四五六七八九十百]+\s*节\s*\S*"
    r"|^\s*第\s*[一二三四五六七八九十百]+\s*[章部分]"
)


def table_to_markdown(table):
    """把 PyMuPDF 表格对象转成 Markdown（第一行作表头）。"""
    try:
        rows = table.extract()
    except Exception:  # noqa: BLE001
        return None, 0, 0
    rows = [
        ["" if c is None else re.sub(r"\s+", " ", str(c)).strip() for c in row]
        for row in rows
    ]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return None, 0, 0
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    head, body = rows[0], rows[1:]
    # 表头全空时用第一行数据兜底
    if not any(head):
        head, body = body[0], body[1:]
    out = ["| " + " | ".join(head) + " |", "|" + " --- |" * ncol]
    for r in body:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out), len(rows), ncol


def inside(block, bboxes):
    x0, y0, x1, y1 = block[:4]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    for bx0, by0, bx1, by1 in bboxes:
        if bx0 - 2 <= cx <= bx1 + 2 and by0 - 2 <= cy <= by1 + 2:
            return True
    return False


def parse_pdf(job):
    meta, pdf_path, text_path = job
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001
        return meta["code"] + "_" + meta["period"], f"open failed: {exc}", 0

    pages, all_tables = [], []
    n_tables = 0
    for pno in range(doc.page_count):
        page = doc[pno]
        tables = []
        try:
            finder = page.find_tables()
            for t in finder.tables:
                md, nrow, ncol = table_to_markdown(t)
                if md:
                    tables.append(
                        {
                            "page": pno + 1,
                            "index": len(tables) + 1,
                            "markdown": md,
                            "rows": nrow,
                            "cols": ncol,
                            "bbox": [round(v, 1) for v in t.bbox],
                        }
                    )
        except Exception:  # noqa: BLE001
            tables = []

        bboxes = [t["bbox"] for t in tables]
        items = []
        try:
            for b in page.get_text("blocks"):
                if len(b) >= 7 and b[6] != 0:  # 图片块
                    continue
                txt = (b[4] or "").strip()
                if not txt or inside(b, bboxes):
                    continue
                items.append((b[1], "text", txt))
        except Exception:  # noqa: BLE001
            pass
        for t in tables:
            items.append((t["bbox"][1], "table", t["markdown"]))
        items.sort(key=lambda x: x[0])

        parts = []
        for _, kind, content in items:
            if kind == "table":
                parts.append("\n[表格开始]\n" + content + "\n[表格结束]\n")
            else:
                parts.append(content)
        text = "\n".join(parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        pages.append({"page": pno + 1, "text": text})
        for t in tables:
            t.pop("bbox", None)
            all_tables.append(t)
        n_tables += len(tables)

    out = {
        **{k: v for k, v in meta.items() if k != "bytes"},
        "num_pages": doc.page_count,
        "num_tables": n_tables,
        "pages": pages,
        "tables": all_tables,
    }
    # 章节起始页
    sections = []
    stream = [(p["page"], ln.strip()) for p in pages for ln in p["text"].split("\n")]
    for i, (pg, line) in enumerate(stream):
        if not (SECTION.match(line) and len(line) < 60):
            continue
        title = line
        if len(re.sub(r"^第\s*[一二三四五六七八九十百]+\s*节", "", line).strip()) < 2:
            # 形如「第三节」单独一行，标题在下一行
            for j in range(i + 1, min(i + 4, len(stream))):
                nxt = stream[j][1]
                if not nxt:
                    continue
                if len(nxt) < 40 and not SECTION.match(nxt):
                    title = f"{line} {nxt}"
                break
        sections.append({"page": pg, "title": title})
    out["sections"] = sections
    doc.close()
    with open(text_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return meta["code"] + "_" + meta["period"], f"pages={out['num_pages']} tables={n_tables}", len(pages)


def main(only=None, workers=6):
    os.makedirs(TEXT_DIR, exist_ok=True)
    meta = json.load(open(META, encoding="utf-8"))
    jobs = []
    for key, m in meta.items():
        if only and key not in only:
            continue
        pdf = os.path.join(ROOT, m["file"])
        if not os.path.exists(pdf):
            continue
        jobs.append((m, pdf, os.path.join(TEXT_DIR, key + ".json")))
    print(f"待解析 {len(jobs)} 份 PDF", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for key, note, _ in pool.map(parse_pdf, jobs):
            print(f"  {key}: {note}", flush=True)


if __name__ == "__main__":
    try:
        main(set(sys.argv[1:]) or None)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
