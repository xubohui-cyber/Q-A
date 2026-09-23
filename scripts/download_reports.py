"""从巨潮资讯（cninfo）批量下载白酒行业上市公司年报 / 半年报全文 PDF。

用法:
    python download_reports.py            # 下载全部公司
    python download_reports.py 600519     # 只下某几家
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cninfo import download, full_url, org_id, query_announcements  # noqa: E402
from companies import COMPANIES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(ROOT, "data", "raw_pdf")
META = os.path.join(ROOT, "data", "reports_meta.json")

SE_DATE = "2023-01-01~2026-09-30"
CATEGORIES = {
    "annual": "category_ndbg_szsh",
    "semi": "category_bndbg_szsh",
}

BAD_TITLE = re.compile(r"摘要|英文|英文版|意见|说明|专项|补充|更正|公告|H股|港股|海外")
PERIOD = re.compile(r"(20\d{2})\s*年(?:年度|半年度)?报告")


def period_of(title):
    m = PERIOD.search(title)
    return m.group(1) if m else None


def pick(anns, kind, keep):
    """筛选出正式全文报告，按时间倒序取前 keep 份。"""
    seen, picked = set(), []
    for a in anns:
        title = a["announcementTitle"]
        if BAD_TITLE.search(title):
            continue
        year = period_of(title)
        if not year:
            continue
        if "年半年度报告" in title or "半年度报告" in title:
            if kind != "semi":
                continue
            tag = f"{year}H1"
        else:
            if kind != "annual":
                continue
            tag = f"{year}A"
        if tag in seen:
            continue
        seen.add(tag)
        picked.append((tag, title, a))
        if len(picked) >= keep:
            break
    return picked


KEEP = {"annual": 2, "semi": 1}


def collect(codes):
    """枚举要下载的报告（只做元数据查询，很快）。"""
    jobs = []
    for code, name, column in COMPANIES:
        if codes and code not in codes:
            continue
        try:
            oid, zwjc = org_id(code, column)
        except Exception as exc:  # noqa: BLE001
            print(f"[{code}] orgId failed: {exc}", flush=True)
            continue
        for kind, cat in CATEGORIES.items():
            try:
                anns = query_announcements(code, column, oid, cat, SE_DATE, pages=2)
            except Exception as exc:  # noqa: BLE001
                print(f"[{code}] {kind} query failed: {exc}", flush=True)
                continue
            for tag, title, a in pick(anns, kind, keep=KEEP[kind]):
                jobs.append(
                    {
                        "key": f"{code}_{tag}",
                        "code": code,
                        "name": name,
                        "exchange": column,
                        "period": tag,
                        "kind": kind,
                        "title": title,
                        "url": full_url(a["adjunctUrl"]),
                        "announcement_date": time.strftime(
                            "%Y-%m-%d", time.localtime(a["announcementTime"] / 1000)
                        ),
                    }
                )
    return jobs


def main(codes, jobs_n=8):
    os.makedirs(PDF_DIR, exist_ok=True)
    meta = {}
    if os.path.exists(META):
        meta = json.load(open(META, encoding="utf-8"))

    jobs = collect(codes)
    print(f"待下载 {len(jobs)} 份报告", flush=True)

    def work(job):
        fname = f"{job['code']}_{job['period']}.pdf"
        dest = os.path.join(PDF_DIR, fname)
        try:
            size = download(job["url"], dest)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {fname}: {exc}", flush=True)
            return
        print(f"  ok {fname} {size/1e6:.2f} MB  {job['title']}", flush=True)
        return {**job, "file": f"data/raw_pdf/{fname}", "bytes": size, "source": "巨潮资讯 cninfo.com.cn"}

    with ThreadPoolExecutor(max_workers=jobs_n) as pool:
        for res in pool.map(work, jobs):
            if res:
                meta[res["key"]] = {k: v for k, v in res.items() if k != "key"}
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n共 {len(meta)} 份报告 -> {META}", flush=True)


if __name__ == "__main__":
    main(set(sys.argv[1:]))
