"""中英文混合分词器 —— 必须与网页端 js/tokenizer.js 完全一致。

规则：
1. NFKC 归一化 + 转小写；
2. 连续 ASCII 字母/数字/百分点（含 . % ）作为一个整词；
3. 连续 CJK 字符：输出每个单字 + 相邻两字组成的 bigram（中文 BM25 的常用做法，
   不依赖词典，索引与查询天然对齐）。
"""

import re
import unicodedata

CJK = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
RUN_RE = re.compile(rf"[{CJK}]+|[a-z0-9][a-z0-9.%,]*|[\u00c0-\u024f]+")


def tokenize(text):
    text = unicodedata.normalize("NFKC", text or "").lower()
    out = []
    for m in RUN_RE.finditer(text):
        s = m.group(0)
        is_cjk = "\u4e00" <= s[0] <= "\u9fff" or "\u3400" <= s[0] <= "\u4dbf"
        if not is_cjk:
            out.append(s)
            continue
        for ch in s:
            out.append(ch)
        for i in range(len(s) - 1):
            out.append(s[i : i + 2])
    return out


if __name__ == "__main__":
    print(tokenize("贵州茅台2025年营业收入1,234.56亿元，同比+15.7%"))
