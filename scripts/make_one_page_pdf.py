"""生成「一页结论」PDF：三指标 + 真实问答截图 + 全景取证 + 结论与链接。

依赖：reportlab、Pillow（workspace 运行时自带）。
字体：macOS 自带 Hiragino Sans GB。
"""

import json
import os
import sys

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Frame, Image as RLImage, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, BaseDocTemplate)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOTS = os.path.join(ROOT, "docs", "screenshots")
OUT = os.path.join(ROOT, "docs", "一页结论.pdf")
FONT = "/System/Library/Fonts/STHeiti Light.ttc"
FONT_BOLD = "/System/Library/Fonts/STHeiti Medium.ttc"

pdfmetrics.registerFont(TTFont("CN", FONT, subfontIndex=0))
pdfmetrics.registerFont(TTFont("CNB", FONT_BOLD, subfontIndex=0))
registerFontFamily("CN", normal="CN", bold="CNB", italic="CN", boldItalic="CNB")

RED = colors.HexColor("#9c2b1b")
INK = colors.HexColor("#1d1a17")
MUTED = colors.HexColor("#6b6259")
LINE = colors.HexColor("#e6ded2")
BG = colors.HexColor("#faf7f2")


def crop(src, box, out):
    im = Image.open(os.path.join(SHOTS, src))
    im.crop(box).save(os.path.join(SHOTS, out))
    return os.path.join(SHOTS, out)


def main():
    data = json.load(open(os.path.join(ROOT, "data", "eval.json"), encoding="utf-8"))
    s = data["summary"]
    rec, cov = s["recall"], s["coverage"]

    # 截取截图里最有信息量的区域（去掉浏览器无关留白）
    qa = crop("qa_answer.png", (115, 0, 1130, 640), "qa_answer_crop.png")
    pano = crop("panorama.png", (115, 0, 1130, 640), "panorama_crop.png")

    styles = {
        "h1": ParagraphStyle("h1", fontName="CN", fontSize=17, leading=22, textColor=colors.white),
        "sub": ParagraphStyle("sub", fontName="CN", fontSize=8.3, leading=11.5, textColor=colors.HexColor("#f3e6df")),
        "h2": ParagraphStyle("h2", fontName="CN", fontSize=10.5, leading=14, textColor=RED, spaceBefore=2, spaceAfter=2),
        "body": ParagraphStyle("body", fontName="CN", fontSize=8.2, leading=11.6, textColor=INK),
        "small": ParagraphStyle("small", fontName="CN", fontSize=7.2, leading=9.6, textColor=MUTED),
        "cap": ParagraphStyle("cap", fontName="CN", fontSize=7.2, leading=9.4, textColor=MUTED, alignment=1),
        "cell": ParagraphStyle("cell", fontName="CN", fontSize=8, leading=10.5, textColor=INK),
        "cellc": ParagraphStyle("cellc", fontName="CN", fontSize=8, leading=10.5, textColor=INK, alignment=1),
    }

    story = []
    # 标题条
    header = Table(
        [[Paragraph("白酒行业年报问答（RAG）· 一页结论", styles["h1"])],
         [Paragraph("16 家白酒上市公司 × 3 期 = 48 份年报/半年报全文 · 8,254 页 · 12,241 张表 · 16,709 个文本块", styles["sub"])]],
        colWidths=[180 * mm],
    )
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), RED),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (0, 0), 7), ("BOTTOMPADDING", (0, 0), (0, 0), 1),
        ("TOPPADDING", (0, 1), (0, 1), 0), ("BOTTOMPADDING", (0, 1), (0, 1), 7),
    ]))
    story += [header, Spacer(1, 6)]

    # 结论
    story.append(Paragraph("结论", styles["h2"]))
    story.append(Paragraph(
        "用<b>建库检索</b>而不是微调：年报数字与页码必须可核对、每年更新、且要能指出「哪家公司、哪一节、第几页」。"
        "纯静态页面在浏览器内完成 BM25（字符 bigram）+ bge-small-zh-v1.5 向量混合检索，答案逐句带 [编号] 出处。"
        "10 道题中 7 题完全正确、1 题部分正确、2 题错误（另加 1 道语料外拒答测试未拒答）。", styles["body"]))
    story.append(Spacer(1, 4))

    # 三指标表
    metrics = Table([
        [Paragraph("评价维度", styles["cell"]), Paragraph("结果", styles["cellc"]),
         Paragraph("口径说明", styles["cell"]), Paragraph("答案正确性", styles["cellc"])],
        [Paragraph("召回 Recall@8", styles["cell"]),
         Paragraph(f"<b>{rec['hits']}/{rec['total']}</b>（{rec['rate']*100:.0f}%）", styles["cellc"]),
         Paragraph("关键事实是否进入 Top-8；全景题按 16 家公司覆盖计", styles["cell"]),
         Paragraph(f"完全正确 <b>{s['correct']}</b>", styles["cellc"])],
        [Paragraph("覆盖 Coverage", styles["cell"]),
         Paragraph(f"<b>{cov['hits']}/{cov['total']}</b>（{cov['rate']*100:.0f}%）", styles["cellc"]),
         Paragraph("真正用于作答的证据（单公司前 3 条 / 全景各公司解析值）覆盖的事实数", styles["cell"]),
         Paragraph(f"部分正确 <b>{s['partial']}</b>", styles["cellc"])],
        [Paragraph("证据质量", styles["cell"]),
         Paragraph("16/16 家", styles["cellc"]),
         Paragraph("全景题 16 家公司均解析出营业收入与归母净利润，出处可回溯到报告页码", styles["cell"]),
         Paragraph(f"错误 <b>{s['wrong']}</b>", styles["cellc"])],
    ], colWidths=[27 * mm, 30 * mm, 88 * mm, 35 * mm])
    metrics.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#faf3e7")),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story += [metrics, Spacer(1, 5)]

    # 逐题三指标（紧凑表）
    short = {
        "Q1": "茅台 2025 营收与归母净利", "Q2": "五粮液 2025 酒类毛利率",
        "Q3": "汾酒 2025 省外收入", "Q4": "泸州老窖 2026H1 营收与净利",
        "Q5": "洋河 2025 合同负债", "Q6": "今世缘 2025 利润分配",
        "Q7": "口子窖 2025 经营风险", "Q8": "全景：2025 营收前五",
        "Q9": "全景：2025 归母净利增速前三", "Q10": "全景：2026H1 营收下滑公司",
        "Q11": "语料外：2026Q3 营收（拒答测试）",
    }
    cell_s = ParagraphStyle("cell_s", fontName="CN", fontSize=6.6, leading=8.2, textColor=INK)
    cell_sc = ParagraphStyle("cell_sc", parent=cell_s, alignment=1)
    head_s = ParagraphStyle("head_s", parent=cell_s, textColor=colors.HexColor("#6b4a12"))
    rows = [[Paragraph("题", head_s), Paragraph("题目", head_s), Paragraph("召回", head_s),
             Paragraph("覆盖", head_s), Paragraph("答案", head_s)]]
    for q in data["questions"]:
        rows.append([
            Paragraph(q["id"], cell_sc),
            Paragraph(short.get(q["id"], q["question"][:18]), cell_s),
            Paragraph(f"{q['recall']['hits']}/{q['recall']['total']}", cell_sc),
            Paragraph(f"{q['coverage']['hits']}/{q['coverage']['total']}", cell_sc),
            Paragraph(q["verdict"], cell_sc),
        ])
    qt = Table(rows, colWidths=[10 * mm, 84 * mm, 20 * mm, 20 * mm, 46 * mm], repeatRows=1)
    qt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#faf3e7")),
        ("GRID", (0, 0), (-1, -1), 0.3, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
    ]))
    story += [Paragraph("逐题三指标（召回、覆盖、答案正确分别评价）", styles["h2"]), qt, Spacer(1, 5)]

    # 方法一行
    story.append(Paragraph("管线", styles["h2"]))
    story.append(Paragraph(
        "巨潮资讯下载 → PyMuPDF 提取文字 + <code>find_tables()</code> 按行列还原表格（剔除页眉页脚）→ "
        "按「公司 / 章节 / 页码」切块（表格整表成块）→ BM25(91,055 词项) + 512 维 int8 向量 → "
        "浏览器内融合排序（α=0.5）→ 抽取式作答 / 可选大模型。", styles["body"]))
    story.append(Spacer(1, 4))

    # 截图两栏
    shot_w = 87 * mm
    shot_h = shot_w * (640 - 0) / (1130 - 115)
    shots = Table([[
        RLImage(qa, width=shot_w, height=shot_h),
        RLImage(pano, width=shot_w, height=shot_h),
    ], [
        Paragraph("① 单公司：答案要点 + [1] 出处；证据卡给出公司/报告/第 6 页/章节，可直接打开原文 PDF", styles["cap"]),
        Paragraph("② 全景：16 家公司按营收降序，逐家给出营收、同比、归母净利润与出处页码", styles["cap"]),
    ]], colWidths=[shot_w + 3 * mm, shot_w + 3 * mm])
    shots.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, 1), 0), ("BOTTOMPADDING", (0, 1), (-1, 1), 2),
        ("TOPPADDING", (0, 1), (-1, 1), 2),
        ("BOX", (0, 0), (0, 0), 0.4, LINE), ("BOX", (1, 0), (1, 0), 0.4, LINE),
    ]))
    story += [shots, Spacer(1, 4)]

    # 关键发现
    story.append(Paragraph("关键发现与错因（单公司先核口径与出处，全景按公司取证）", styles["h2"]))
    finds = [
        "<b>口径混淆：</b>汾酒「省外收入」合并口径 252.02 亿元（P128）排第 4，第 2 位是母公司口径表的 21.8 万元，抽取式答案取错 → 同类指标必须核对合并/母公司口径。",
        "<b>章节被淹没：</b>口子窖「可能面对的风险」被审计报告的「重大错报风险」等表述挤出 Top-3（落到第 6 位）→ 章节性问题需章节意图或二次排序。",
        "<b>不会拒答：</b>问「2026 年三季度」时语料里没有该期数据，系统仍拼出无关要点 → 需加证据不足的拒答阈值。",
    ]
    for t in finds:
        story.append(Paragraph("• " + t, styles["body"]))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "<b>工程校验：</b>浏览器分词器与 Python 逐 token 一致（20/20）；BM25 排序逐条一致（10/10）；"
        "网页检索与离线 Top-K 一致（8/8）；浏览器内 ONNX 查询向量与 Python 逐位相同（余弦 1.000000）。", styles["small"]))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "在线页面：https://xubohui-cyber.github.io/Q-A/　｜　仓库：https://github.com/xubohui-cyber/Q-A　｜　"
        "复现：<code>scripts/download_reports.py → extract_text.py → chunk_reports.py → build_index.py → export_web.py</code>", styles["small"]))

    def footer(canv, doc):
        canv.saveState()
        canv.setFont("CN", 7)
        canv.setFillColor(MUTED)
        canv.drawString(14 * mm, 9 * mm, "数据来源：巨潮资讯网（cninfo.com.cn）年报/半年报全文；仅用于课程作业与技术演示，不构成投资建议。")
        canv.drawRightString(A4[0] - 14 * mm, 9 * mm, "2026-11-18")
        canv.restoreState()

    doc = BaseDocTemplate(OUT, pagesize=A4,
                          leftMargin=14 * mm, rightMargin=14 * mm,
                          topMargin=12 * mm, bottomMargin=14 * mm,
                          title="白酒行业年报问答（RAG）一页结论", author="xubohui-cyber")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=footer)])
    doc.build(story)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    sys.exit(main())
