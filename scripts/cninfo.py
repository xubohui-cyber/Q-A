"""巨潮资讯（cninfo）公告检索与下载的轻量客户端。"""

import json
import os
import time

import requests

BASE = "http://www.cninfo.com.cn"
STATIC = "http://static.cninfo.com.cn"
QUERY = BASE + "/new/hisAnnouncement/query"
TOPSEARCH = BASE + "/new/information/topSearch/query"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Referer": BASE + "/new/commonUrl?url=disclosure/list/notice",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}


def org_id(code, column):
    """通过 topSearch 接口拿到 cninfo 的 orgId。"""
    r = requests.post(
        TOPSEARCH,
        data={"keyWord": code, "maxNum": 10},
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    for item in r.json():
        if item.get("code") == code:
            return item["orgId"], item.get("zwjc", "")
    raise RuntimeError(f"orgId not found for {code}")


def query_announcements(code, column, orgid, category, se_date, pages=3, page_size=30):
    out = []
    for page in range(1, pages + 1):
        data = {
            "pageNum": page,
            "pageSize": page_size,
            "column": column,
            "tabName": "fulltext",
            "plate": "",
            "stock": f"{code},{orgid}",
            "searchkey": "",
            "secid": "",
            "category": category,
            "trade": "",
            "seDate": se_date,
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        r = requests.post(QUERY, data=data, headers=HEADERS, timeout=40)
        r.raise_for_status()
        js = r.json()
        anns = js.get("announcements") or []
        out.extend(anns)
        if not js.get("hasMore"):
            break
        time.sleep(0.4)
    return out


def download(url, dest, retries=3):
    if os.path.exists(dest) and os.path.getsize(dest) > 20000:
        return os.path.getsize(dest)
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=120, stream=True)
            r.raise_for_status()
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
            os.replace(tmp, dest)
            return os.path.getsize(dest)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raise
            if i == retries - 1:
                raise
            print("  retry", i + 1, exc)
            time.sleep(2 * (i + 1))
        except Exception as exc:  # noqa: BLE001
            if i == retries - 1:
                raise
            print("  retry", i + 1, exc)
            time.sleep(2 * (i + 1))
    return 0


def full_url(adjunct_url):
    path = adjunct_url.split("//", 1)[-1]
    if "/" in path and "." in path.split("/", 1)[0]:
        path = path.split("/", 1)[1]  # 去掉原域名
    return STATIC + "/" + path.lstrip("/")


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
