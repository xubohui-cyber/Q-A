"""把解析后的报告切成语义块，并带上「公司 / 报告期 / 章节 / 页码」元数据。

切块规则：
* 表格块：一张表 = 一个块（行列结构完整保留，不做切分）；
* 文字块：按段落聚合到接近 TARGET 字再切，最少 MIN_C，最多 MAX_C；相邻块留 1 段重叠；
* 表格前若只有很短的引导语（如「（二）报告期内审计委员会召开六次会议」），
  直接并入该表格块，避免产生碎片块；
* 每个块记录：公司代码/简称、报告期、章节、小节、起始页、结束页、类型。
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_DIR = os.path.join(ROOT, "data", "text")
OUT = os.path.join(ROOT, "data", "out", "chunks.jsonl")

SEC_RE = re.compile(r"^\s*第\s*[一二三四五六七八九十百]+\s*节")
SUB_RE = re.compile(
    r"^\s*(?:[一二三四五六七八九十]+\s*[、.．]|（[一二三四五六七八九十]+）|"
    r"\d+\s*[、.．]\s*\S|\d+\.\d+\s*\S)"
)
TABLE_RE = re.compile(r"\[表格开始\]\n(.*?)\n\[表格结束\]", re.S)

TARGET, MIN_C, MAX_C = 700, 220, 1100
HEADER_FREQ = 0.3  # 出现在 30% 以上页面的短行视为页眉/页脚


def split_segments(page_text):
    """把页面文本拆成 text / table 片段。"""
    segs, pos = [], 0
    for m in TABLE_RE.finditer(page_text):
        if m.start() > pos:
            segs.append(("text", page_text[pos : m.start()]))
        segs.append(("table", m.group(1).strip()))
        pos = m.end()
    if pos < len(page_text):
        segs.append(("text", page_text[pos:]))
    return segs


def build_stream(doc):
    """把报告变成 ('line'|'blank'|'table', 页码, 内容) 的线性流。"""
    # 先统计跨页重复的短行（页眉、页脚、页码），统一剔除
    line_pages = {}
    for page in doc["pages"]:
        for line in set(l.strip() for l in page["text"].split("\n")):
            if line:
                line_pages.setdefault(line, set()).add(page["page"])
    npage = max(doc["num_pages"], 1)
    boiler = {
        ln
        for ln, pgs in line_pages.items()
        if len(pgs) > npage * HEADER_FREQ and len(ln) < 40
    }

    stream = []
    for page in doc["pages"]:
        pno = page["page"]
        for kind, content in split_segments(page["text"]):
            if kind == "table":
                stream.append(("table", pno, content))
            else:
                for line in content.split("\n"):
                    s = line.strip()
                    if not s:
                        stream.append(("blank", pno, ""))
                        continue
                    if s in boiler or is_page_number(s, pno):
                        continue
                    if s:
                        stream.append(("line", pno, s))
    return stream


PAGENO = re.compile(r"^(\d{1,4})(?:\s*/\s*(\d{1,4}))?$")


def is_page_number(s, pno):
    """剔除页脚页码（形如 53 或 53 / 143）。"""
    m = PAGENO.match(s)
    if not m:
        return False
    n = int(m.group(1))
    return abs(n - pno) <= 1 or (m.group(2) and abs(int(m.group(2)) - pno) <= 1)


def table_title(prev_lines):
    for line in reversed(prev_lines[-4:]):
        s = line.strip()
        if 3 <= len(s) <= 40 and not s.endswith(("。", "，", "；")):
            return s
    return ""


def chunk_report(doc, key):
    chunks = []
    section, subsection = "", ""
    buf, buf_pages, last_para, recent = [], [], None, []
    counter = [0]

    def meta(page_start, page_end, ctype, extra=None):
        counter[0] += 1
        c = {
            "id": f"{key}#{counter[0]:04d}",
            "code": doc["code"],
            "name": doc["name"],
            "exchange": doc["exchange"],
            "period": doc["period"],
            "kind": doc["kind"],
            "title": doc["title"],
            "url": doc["url"],
            "announcement_date": doc["announcement_date"],
            "section": section,
            "subsection": subsection,
            "page_start": page_start,
            "page_end": page_end,
            "type": ctype,
        }
        if extra:
            c.update(extra)
        return c

    def flush():
        nonlocal buf, buf_pages, last_para
        if not buf:
            return
        body = "\n".join(buf).strip()
        if body:
            chunks.append(
                {**meta(buf_pages[0], buf_pages[-1], "text"), "text": body,
                 "n_chars": len(body)}
            )
            # 段间重叠：带上一段结尾，帮助跨块问题
            last_para = buf[-1]
            buf, buf_pages = [], []

    stream = build_stream(doc)
    skip = set()

    def heading_at(i):
        """返回 (标题, 消耗到的下标)，识别「第X节」与小节标题。"""
        kind, pno, line = stream[i]
        if kind != "line" or len(line) > 60:
            return None
        if not SEC_RE.match(line):
            return None
        title = line
        j = i + 1
        body = SEC_RE.sub("", line).strip()
        if len(body) < 2:  # 「第三节」单独成行
            while j < len(stream) and stream[j][0] == "blank":
                j += 1
            if j < len(stream) and stream[j][0] == "line" and len(stream[j][2]) < 40:
                title = f"{line} {stream[j][2]}"
                return title, j
        return title, i

    for i, (kind, pno, content) in enumerate(stream):
        if i in skip:
            continue
        if kind == "blank":
            size = sum(len(b) for b in buf)
            if buf and (size >= TARGET or size >= MAX_C):
                flush()
            continue
        if kind == "table":
            size = sum(len(b) for b in buf)
            lead = "\n".join(buf).strip()
            if size >= MIN_C:
                flush()
                lead = ""
            elif size and size < 200:
                buf, buf_pages = [], []  # 短引导语并入表格块
            elif buf:
                flush()
                lead = ""
            nrow = content.count("\n") + 1
            chunks.append(
                {
                    **meta(pno, pno, "table",
                           {"table_rows": nrow, "table_caption": table_title(recent)}),
                    "text": (lead + "\n" + content) if lead else content,
                    "n_chars": len(content) + (len(lead) + 1 if lead else 0),
                }
            )
            continue

        head = heading_at(i)
        if head:
            title, end = head
            if sum(len(b) for b in buf) >= MIN_C:
                flush()
            elif buf:
                buf, buf_pages = [], []  # 丢弃残留的极短片段，标题本身会进入 recent
            section, subsection = title, ""
            recent = [title]
            if end > i:
                skip.add(end)
            continue
        if SUB_RE.match(content) and len(content) < 50:
            if sum(len(b) for b in buf) >= MIN_C:
                flush()
            subsection = content
            recent.append(content)
            continue

        buf.append(content)
        buf_pages.append(pno)
        recent.append(content)
        if sum(len(b) for b in buf) >= MAX_C:
            flush()
            if last_para and len(last_para) < 300:
                buf, buf_pages = [last_para], [pno]
    if sum(len(b) for b in buf) >= 120:
        flush()
    return chunks


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    files = sorted(os.listdir(TEXT_DIR))
    all_chunks, n_docs = [], 0
    for fn in files:
        if not fn.endswith(".json"):
            continue
        doc = json.load(open(os.path.join(TEXT_DIR, fn), encoding="utf-8"))
        key = fn[:-5]
        cs = chunk_report(doc, key)
        all_chunks.extend(cs)
        n_docs += 1
    with open(OUT, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tot = sum(c["n_chars"] for c in all_chunks)
    print(f"报告 {n_docs} 份，块 {len(all_chunks)} 个，平均 {tot/max(len(all_chunks),1):.0f} 字")
    print("其中表格块", sum(1 for c in all_chunks if c["type"] == "table"))


if __name__ == "__main__":
    main()
