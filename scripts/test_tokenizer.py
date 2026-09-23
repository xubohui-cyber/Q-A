"""生成 Python 端分词结果，用于和浏览器端 tokenizer.js 对齐校验。"""

import json
import os
import sys

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = [
    "贵州茅台2025年营业收入是多少？",
    "2025年年度报告中，五粮液的毛利率为 77.05%，同比+0.5pct。",
    "山西汾酒(600809) 省外市场收入 3,412,345,678.90 元",
    "泸州老窖 2026 上半年归母净利润同比 -12.3%",
    "合同负债/存货 turnover   ratio",
    "ROE、EPS、分红率：每股派息27.993元（含税）",
    "公司简称：贵州茅台；股票代码：600519。",
    "（一）主要经营情况【重大事项】《提示公告》",
    "ａｂｃ　全角空格　ＡＢＣ１２３",
    "茅台+五粮液=？ 100%＆*（",
    "A股  B股；H股",
    "",
    "   ",
    "毛利率\t营业收入\n净利润",
    "3.14159, 1,234,567.89, -0.5%, +15%",
    "洋河股份的\"合同负债\"是？",
    "①②③ Ⅰ Ⅱ Ⅲ 一、二、三",
    "2026年半年度报告（未经审计）",
    "www.cninfo.com.cn 巨潮资讯",
    "白酒行业: 茅台 / 五粮液 / 泸州老窖 / 山西汾酒 / 洋河股份",
]


def main():
    tok = Tokenizer.from_file(os.path.join(ROOT, "web", "model", "tokenizer.json"))
    tok.enable_truncation(max_length=512)
    out = []
    for t in TESTS:
        enc = tok.encode(t)
        out.append({"text": t, "ids": enc.ids, "tokens": enc.tokens})
    path = os.path.join(ROOT, "data", "out", "tokenizer_ref.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", path, len(out))


if __name__ == "__main__":
    sys.exit(main())
