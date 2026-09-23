"""从「主要会计数据」表格块里抽出指标数值，用于跨公司对比。

必须与网页端 js/rag.js 中的 extractMetric 保持一致。
"""

import re

NUM = re.compile(r"^-?[\d,]+(?:\.\d+)?%?$")



def _clean(cell):
    return cell.replace(" ", "").replace("\u3000", "")


def _to_num(cell):
    s = _clean(cell).rstrip("%")
    if not NUM.match(_clean(cell)):
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _merge_split(cells):
    """把被 PDF 拆开的数字重新拼起来，例如 "10,472,224,575." + "78" """
    out, i = [], 0
    while i < len(cells):
        cur = _clean(cells[i])
        if (
            i + 1 < len(cells)
            and re.fullmatch(r"-?[\d,]+\.", cur)
            and re.fullmatch(r"\d{1,2}", _clean(cells[i + 1]))
        ):
            out.append(cur + _clean(cells[i + 1]))
            i += 2
            continue
        out.append(cells[i])
        i += 1
    return out


def _tables(text):
    """把文本里的连续 Markdown 表格行拆成一张张表（行 -> 单元格列表）。"""
    tables, cur = [], []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("|"):
            if set(s) <= set("|-: "):  # 分隔行
                continue
            cur.append(_merge_split([c.strip() for c in s.strip("|").split("|")]))
        elif cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    return tables


def _label_match(label, next_label, metric):
    if metric == "revenue":
        if not any(k in label for k in ("营业收入", "营业总收入")):
            return False
        return not any(x in label for x in ("成本", "构成", "比重", "占比", "明细", "分产品"))
    if metric == "profit":
        if "归属于上市公司股东" not in label:
            return False
        if "扣除非经常性损益" in label:
            return False
        if "净利润" in label:
            return True
        # 标签被 PDF 拆成两行（如「归属于上市公司股东」+「的净利润（元）」）
        return "净利润" in next_label and "扣除非经常性损益" not in next_label
    raise ValueError(metric)


def extract_metric(text, metric):
    """返回 (本期值, 上期值, 同比%)；单位统一到「元」。"""
    unit = 1.0
    if "单位：万元" in text or "单位:万元" in text or "单位：人民币万元" in text:
        unit = 1e4
    candidates = []
    for rows in _tables(text):
        header = "".join(_clean(c) for r in rows[:3] for c in r)
        if "季度" in header:  # 分季度表里是单季数据，不能当全年/半年值
            continue
        score = (2 if "主要会计数据" in header else 0) + (1 if "上年同期" in header else 0)
        for i, cells in enumerate(rows):
            if len(cells) < 3:
                continue
            label = "".join(_clean(c) for c in cells)
            next_label = (
                "".join(_clean(c) for c in rows[i + 1]) if i + 1 < len(rows) else ""
            )
            if not _label_match(label, next_label, metric):
                continue
            values = []
            for c in cells:
                v = _to_num(c)
                if v is not None:
                    values.append((v, c.strip().endswith("%")))
            # 金额（万元/元级）与同比（百分比级）按大小区分：
            # 全年/半年行是「本期 上期 (同比) 上年」，分季度行是 4 个季度金额
            amounts = [v for v, _ in values if abs(v) > 1000]
            smalls = [v for v, _ in values if abs(v) <= 1000]
            if len(amounts) < 2 or len(amounts) > 3:
                continue
            now, prev = amounts[0], amounts[1]
            pct = next((v for v, is_pct in values if is_pct and abs(v) <= 10000), None)
            if pct is None:
                pct = next((v for v in smalls if abs(v) != abs(now)), None)
            candidates.append((score, now * unit, prev * unit, pct))
    if not candidates:
        return None
    best = max(candidates, key=lambda c: (c[0], abs(c[1])))
    return best[1], best[2], best[3]


def metric_of(text, metric):
    """metric ∈ {revenue, profit}"""
    return extract_metric(text, metric)


def fmt_yi(v):
    return f"{v/1e8:,.2f} 亿元" if v is not None else "—"
