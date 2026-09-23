/* 检索 + 融合 + 抽取式答案组装（全部在浏览器本地完成） */

import { tokenize, BM25Index } from "./bm25.js";
import { DenseEncoder, dotInt8 } from "./dense.js";
export { queryTerms, keyLines, markdownTableToHtml, escapeHtml, highlight, composeExtractive, extractMetric, fmtYi } from "./answer.js";

const DEFAULT_DATA = new URL("../data/", import.meta.url);

/** 下载（可选 gzip 解压）并上报进度。
 * 先把响应收完再解压：避免在部分实现（Node undici 等）上直接 pipe 网络流导致的不稳定。 */
async function fetchStream(url, { gunzip = false, onProgress } = {}) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`加载失败：${res.status} ${url}`);
  const total = Number(res.headers.get("content-length") || 0);
  if (!res.body) return res.arrayBuffer();
  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    onProgress?.(total ? Math.min(got / total, 1) : 0, got);
  }
  const bytes = new Uint8Array(got);
  let off = 0;
  for (const p of parts) {
    bytes.set(p, off);
    off += p.length;
  }
  if (!gunzip) return bytes.buffer;
  if (typeof DecompressionStream === "undefined") {
    throw new Error("当前浏览器不支持 DecompressionStream，请使用最新版 Chrome / Edge / Safari");
  }
  const decompressed = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(decompressed).arrayBuffer();
}

function minmax(map) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of map.values()) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const out = new Map();
  if (!map.size) return out;
  if (hi <= lo) {
    for (const k of map.keys()) out.set(k, 1);
    return out;
  }
  for (const [k, v] of map) out.set(k, (v - lo) / (hi - lo));
  return out;
}

const ALIAS = {
  茅台: "600519", 五粮液: "000858", 老窖: "000568", 泸州老窖: "000568",
  洋河: "002304", 汾酒: "600809", 山西汾酒: "600809", 古井: "000596",
  今世缘: "603369", 迎驾: "603198", 口子: "603589", 老白干: "600559",
  舍得: "600702", 水井坊: "600779", 金徽: "603919", 酒鬼: "000799",
  天佑德: "002646", 金种子: "600199",
};

/* 问句改写：口语化财务说法 → 年报里的规范表述（只作用于 BM25 通道，
 * 与离线端 work/scripts/query_rewrite.py 保持一致） */
const EXPANSION = {
  归母净利润: "归属于上市公司股东的净利润",
  归母净利: "归属于上市公司股东的净利润",
  归母: "归属于上市公司股东的净利润",
  净利润: "归属于上市公司股东的净利润 利润总额",
  营收: "营业收入",
  营业总收入: "营业收入",
  同比: "比上年同期增减",
  增长率: "比上年同期增减",
  分红: "利润分配 派发现金红利 每10股",
  毛利率: "毛利率 分产品 营业成本",
  费用率: "销售费用 管理费用 占营业收入比重",
  现金流: "经营活动产生的现金流量净额",
  库存: "存货 存货跌价准备",
  存货: "存货 存货跌价准备",
  应收账款: "应收账款 坏账准备",
  合同负债: "合同负债 预收款项",
  经销商: "经销商 经销商数量 增加减少",
  产能: "产能 产量 销量 库存量",
  省外: "省外 其他地区 分地区",
  省内: "省内 山西省内 分地区",
  分红率: "现金分红 占合并报表中归属于上市公司普通股股东的净利润的比率",
  存货周转: "存货周转率 存货周转天数",
  毛利率下滑: "毛利率比上年同期增减",
  高管薪酬: "董事、监事和高级管理人员报酬",
  研发: "研发投入 研发人员",
  风险: "可能面对的风险 风险因素",
  经营计划: "公司未来发展的展望 经营计划",
};

export function expandQuery(query) {
  const extra = Object.entries(EXPANSION)
    .filter(([k]) => query.includes(k))
    .map(([, v]) => v);
  return extra.length ? `${query} ${extra.join(" ")}` : query;
}

export class Rag {
  constructor(opts = {}) {
    this.dataBase = opts.dataBase || DEFAULT_DATA;
    this.chunks = [];
    this.loaded = { base: false, bm25: false };
    this.dense = new DenseEncoder("local", opts.denseOverride || null);
    this.denseLoaded = false;
  }

  url(name) {
    return new URL(name, this.dataBase);
  }

  async loadBase(onStep = () => {}) {
    onStep({ stage: "meta" });
    const meta = await (await fetch(this.url("meta.json"))).json();
    this.meta = meta;
    onStep({ stage: "chunks", ratio: 0 });
    const chunksBuf = await fetchStream(this.url("chunks.json"), {
      onProgress: (r) => onStep({ stage: "chunks", ratio: r }),
    });
    this.chunks = JSON.parse(new TextDecoder().decode(chunksBuf));
    onStep({ stage: "bm25", ratio: 0 });
    const bmBytes = await fetchStream(this.url("bm25.bin.gz"), {
      gunzip: true,
      onProgress: (r) => onStep({ stage: "bm25", ratio: r }),
    });
    this.bm25 = new BM25Index(bmBytes);
    const lenBuf = await (await fetch(this.url("doclen.bin"))).arrayBuffer();
    this.lens = new Uint32Array(lenBuf);
    this.loaded.base = true;
    this.loaded.bm25 = true;
    return meta;
  }

  async loadDense(onProgress) {
    onProgress?.({ stage: "vectors", ratio: 0 });
    const vecBytes = await fetchStream(this.url("vec.i8.bin.gz"), {
      gunzip: true,
      onProgress: (ratio) => onProgress?.({ stage: "vectors", ratio }),
    });
    this.vectors = new Int8Array(vecBytes);
    this.dim = this.vectors.length / this.chunks.length;
    await this.dense.load(onProgress);
    this.denseLoaded = true;
  }

  chunk(i) {
    const r = this.chunks[i];
    return {
      idx: i,
      code: this.meta.companies[r[0]].code,
      name: this.meta.companies[r[0]].name,
      exchange: this.meta.companies[r[0]].exchange,
      report: this.meta.reports[r[1]],
      section: this.meta.sections[r[2]] || "",
      subsection: this.meta.subs[r[3]] || "",
      pageStart: r[4],
      pageEnd: r[5],
      type: r[6] ? "table" : "text",
      text: r[7],
    };
  }

  detectFilter(query, code = null, period = null) {
    if (!code) {
      for (const c of this.meta.companies) {
        if (query.includes(c.name) || query.includes(c.code)) {
          code = c.code;
          break;
        }
      }
    }
    if (!code) {
      for (const [k, v] of Object.entries(ALIAS)) {
        if (query.includes(k)) {
          code = v;
          break;
        }
      }
    }
    if (!period) {
      if (/半年|中报|上半年/.test(query)) period = "H1";
      else if (query.includes("2024")) period = "2024A";
      else if (query.includes("2025")) period = "2025A";
    }
    return { code, period };
  }

  allowed(code, period) {
    if (!code && !period) return null;
    const set = new Set();
    for (let i = 0; i < this.chunks.length; i++) {
      const r = this.chunks[i];
      if (code && this.meta.companies[r[0]].code !== code) continue;
      if (period && !this.meta.reports[r[1]].period.includes(period)) continue;
      set.add(i);
    }
    return set.size ? set : null;
  }

  async search(query, opts = {}) {
    const { topK = 8, alpha = 0.5, mode = "hybrid", code = null, period = null,
            autoFilter = true } = opts;
    if (!this.loaded.bm25) throw new Error("索引尚未加载完成");
    let useCode = code;
    let usePeriod = period;
    if (autoFilter && !code && !period) {
      const f = this.detectFilter(query);
      useCode = f.code;
      usePeriod = f.period;
    }
    const allowed = this.allowed(useCode, usePeriod);
    const bm = this.loaded.bm25 && mode !== "dense"
      ? this.bm25.score(expandQuery(query), allowed, this.lens)
      : new Map();

    let denseScores = null;
    if (mode !== "bm25" && this.denseLoaded) {
      const [qv] = await this.dense.embed([query]);
      const all = dotInt8(qv, this.vectors, this.dim);
      denseScores = new Map();
      for (let i = 0; i < all.length; i++) {
        if (allowed && !allowed.has(i)) continue;
        denseScores.set(i, all[i]);
      }
    }
    const bn = minmax(bm);
    const dn = denseScores ? minmax(denseScores) : new Map();
    const fused = new Map();
    if (mode === "bm25") {
      for (const [k, v] of bn) fused.set(k, v);
    } else if (mode === "dense") {
      for (const [k, v] of dn) fused.set(k, v);
    } else {
      for (const k of new Set([...bn.keys(), ...dn.keys()])) {
        fused.set(k, alpha * (dn.get(k) || 0) + (1 - alpha) * (bn.get(k) || 0));
      }
    }
    const order = [...fused.entries()]
      .sort((a, b) => b[1] - a[1] || a[0] - b[0]) // 并列时按文档编号，保证与离线端一致
      .slice(0, topK);
    const rankOf = (map, i) =>
      [...map.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]).findIndex(([k]) => k === i) + 1;
    const results = order.map(([i, s], n) => {
      const c = this.chunk(i);
      c.rank = n + 1;
      c.score = s;
      c.bm25 = bm.get(i) ?? 0;
      c.cos = denseScores ? denseScores.get(i) ?? 0 : null;
      c.bm25Rank = bn.size ? (rankOf(bn, i) || null) : null;
      c.denseRank = denseScores ? (rankOf(denseScores, i) || null) : null;
      return c;
    });
    return {
      query,
      filter: { code: useCode, period: usePeriod, mode },
      results,
      alpha,
      denseUsed: !!denseScores,
    };
  }

  /** 全景题：每家公司各取 top-n */
  metricQuery(question) {
    const q = question.replace(/^\s*全景\s*[:：]?\s*/, "");
    if (q.includes("营业收入") || q.includes("营收")) {
      return "主要会计数据 营业收入 上年同期 增减";
    }
    if (q.includes("净利润")) {
      return "主要会计数据 归属于上市公司股东的净利润 上年同期 增减";
    }
    if (q.includes("毛利率")) return "主营业务 分产品 毛利率 比上年同期增减";
    return q;
  }

  async searchGrouped(query, opts = {}) {
    const { perCompany = 1, alpha = 0.5, period = null } = opts;
    const q = this.metricQuery(query);
    const wide = Math.max(perCompany * this.meta.companies.length * 4, 64);
    const res = await this.search(q, { topK: wide, alpha, period, autoFilter: false });
    const grouped = new Map();
    for (const c of res.results) {
      const arr = grouped.get(c.code) || [];
      if (arr.length < perCompany) {
        arr.push(c);
        grouped.set(c.code, arr);
      }
    }
    for (const c of this.meta.companies) {
      if (grouped.has(c.code)) continue;
      const one = await this.search(q, {
        topK: perCompany, alpha, code: c.code, period, autoFilter: false,
      });
      if (one.results.length) grouped.set(c.code, one.results);
    }
    const groups = this.meta.companies
      .filter((c) => grouped.has(c.code))
      .map((c) => ({ company: c, results: grouped.get(c.code) }));
    return { query: q, origQuery: query, groups, perCompany, ranks: res };
  }
}

/* ---------------- 文本与答案组装（实现见 answer.js，便于离线评测复用） ---------------- */

/** 调 OpenAI 兼容接口，用检索到的证据生成带引用的答案 */
export async function llmAnswer(query, results, cfg) {
  const ctx = results
    .map((c, i) => {
      const head = `[${i + 1}] ${c.name}｜${c.report.title}｜${c.section}${c.subsection ? " / " + c.subsection : ""}｜第 ${c.pageStart}${c.pageEnd > c.pageStart ? "-" + c.pageEnd : ""} 页`;
      return `${head}\n${c.text.slice(0, 1800)}`;
    })
    .join("\n\n---\n\n");
  const sys =
    "你是白酒行业年报分析助手。只能依据【证据】回答，禁止编造数字；" +
    "每个结论后面用 [编号] 标注出处；证据不足以回答时明确说“年报中没有找到相关数据”。" +
    "涉及金额时保留原文单位并换算为亿元（如原文为万元）。用简洁中文回答。";
  const res = await fetch(`${cfg.base.replace(/\/$/, "")}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${cfg.key}` },
    body: JSON.stringify({
      model: cfg.model,
      temperature: 0.1,
      messages: [
        { role: "system", content: sys },
        { role: "user", content: `【证据】\n${ctx}\n\n【问题】${query}` },
      ],
    }),
  });
  if (!res.ok) throw new Error(`大模型接口返回 ${res.status}：${(await res.text()).slice(0, 200)}`);
  const js = await res.json();
  return js.choices?.[0]?.message?.content || "(空回答)";
}
