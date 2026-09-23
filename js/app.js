import { Rag, composeExtractive, highlight, queryTerms, markdownTableToHtml, escapeHtml, llmAnswer, extractMetric, fmtYi } from "./rag.js";
import { BM25Index } from "./bm25.js";
import { METHOD_HTML } from "./method.js";

const EXAMPLES = [
  "贵州茅台2025年营业收入和归母净利润是多少？",
  "五粮液2025年酒类产品的毛利率是多少？",
  "山西汾酒2025年省外市场收入是多少？",
  "泸州老窖2026年上半年营业收入和归母净利润同比变化？",
  "洋河股份2025年末合同负债余额是多少？",
  "今世缘2025年度利润分配方案是什么？",
  "口子窖2025年报披露了哪些经营风险？",
  "全景：2025年营业收入最高的5家白酒公司是谁？",
  "全景：2025年归母净利润同比增速最高的3家公司？",
  "全景：2026年上半年哪些白酒公司营业收入同比下滑？",
];

const $ = (s) => document.querySelector(s);
const rag = new Rag();
let stats = null;

/* ---------- 加载进度 ---------- */
function progress(text, ratio) {
  $("#loadbar").classList.add("show");
  $("#loadtext").textContent = text;
  if (ratio != null) $("#barfill").style.width = `${Math.round(ratio * 100)}%`;
}
function progressDone() {
  $("#barfill").style.width = "100%";
  setTimeout(() => $("#loadbar").classList.remove("show"), 500);
}

/* ---------- 初始化 ---------- */
async function boot() {
  try {
    const meta = await rag.loadBase(({ stage, ratio }) => {
      const r = ratio == null ? 0 : ratio;
      if (stage === "meta") progress("加载元数据…", 0.02);
      else if (stage === "chunks") progress(`加载文本块索引… ${(r * 100).toFixed(0)}%`, 0.05 + r * 0.75);
      else progress(`加载 BM25 倒排索引… ${(r * 100).toFixed(0)}%`, 0.8 + r * 0.2);
    });
    stats = meta.stats;
    renderStats(stats);
    fillCompanies(meta);
    progressDone();
    $("#status").innerHTML =
      `索引就绪：<b>${stats.chunks}</b> 个块，BM25 词项 <b>${stats.bm25_terms}</b> 个。` +
      `当前为 <b>BM25 关键词检索</b>模式，` +
      `<button class="ghost" id="enableDense">启用语义检索（下载约 43MB 模型）</button>`;
    $("#enableDense").onclick = enableDense;
    const q = new URLSearchParams(location.search).get("q");
    if (q) {
      $("#q").value = q;
      runSearch();
    }
  } catch (err) {
    $("#status").innerHTML = `<span class="err">初始化失败：${escapeHtml(err.message)}</span>`;
    progressDone();
  }
}

function renderStats(s) {
  $("#s-companies").textContent = s.companies;
  $("#s-reports").textContent = s.reports;
  $("#s-pages").textContent = s.pages.toLocaleString();
  $("#s-chunks").textContent = s.chunks.toLocaleString();
  $("#s-tables").textContent = s.tables.toLocaleString();
}

function fillCompanies(meta) {
  const sel = $("#company");
  for (const c of meta.companies) {
    const o = document.createElement("option");
    o.value = c.code;
    o.textContent = `${c.name}（${c.code}）`;
    sel.appendChild(o);
  }
  const box = $("#examples");
  for (const q of EXAMPLES) {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = q;
    b.onclick = () => {
      $("#q").value = q;
      $("#panorama").checked = q.startsWith("全景");
      runSearch();
    };
    box.appendChild(b);
  }
}

async function enableDense() {
  const btn = $("#enableDense");
  btn.disabled = true;
  try {
    const customBase = ($("#modelBase")?.value || "").trim().replace(/\/$/, "");
    if (customBase) {
      rag.dense.override = {
        label: "自定义",
        onnx: `${customBase}/onnx/model_quantized.onnx`,
        tokenizer: `${customBase}/tokenizer.json`,
        wasmPaths: new URL("../vendor/ort/", import.meta.url).href,
      };
    }
    await rag.loadDense(({ stage, ratio }) => {
      if (stage === "vectors") progress("加载向量索引（8.6MB）…", 0.1 + (ratio || 0) * 0.2);
      else if (stage === "model") progress(`加载语义模型（约 24MB）… ${Math.round((ratio || 0) * 100)}%`, 0.3 + (ratio || 0) * 0.7);
      else progress("加载分词器…", 0.05);
    });
    progressDone();
    $("#status").innerHTML =
      `<b>语义检索已启用</b>：现在可以在检索设置里选择「混合（BM25 + 向量）」或「仅向量语义」模式。` +
      `（浏览器端模型：bge-small-zh-v1.5 int8，512 维）`;
  } catch (err) {
    progressDone();
    $("#status").innerHTML = `<span class="err">语义模型加载失败：${escapeHtml(err.message)}</span>` +
      `<br>可继续使用 BM25 关键词检索；若网络受限，可在 <code>web/js/dense.js</code> 里改用 jsDelivr CDN 源。`;
    btn.disabled = false;
  }
}

/* ---------- 检索 ---------- */
async function runSearch() {
  const query = $("#q").value.trim();
  if (!query) return;
  const opts = {
    topK: Number($("#topk").value) || 8,
    alpha: Number($("#alpha").value),
    mode: $("#mode").value,
    code: $("#company").value || null,
    period: $("#period").value || null,
  };
  if (opts.mode !== "bm25" && !rag.denseLoaded) {
    opts.mode = "bm25";
    $("#status").innerHTML = `检索模式已临时回退为 <b>BM25</b>（语义模型未加载，点击上方按钮可启用向量检索）。`;
  }
  $("#answerArea").innerHTML = `<div class="card"><span class="muted">检索中…</span></div>`;
  $("#evidence").innerHTML = "";
  const t0 = performance.now();
  let out;
  if ($("#panorama").checked) {
    // 全景模式必须锁定报告期，否则会把不同报告期的块混在一起对比
    const panoPeriod = opts.period || rag.detectFilter(query).period;
    out = await rag.searchGrouped(query, { perCompany: 3, alpha: opts.alpha, period: panoPeriod });
    out.period = panoPeriod;
    renderPanorama(query, out, performance.now() - t0);
  } else {
    out = await rag.search(query, opts);
    renderAnswer(query, out);
    renderEvidence(out);
  }
  renderStatusLine(out, performance.now() - t0);
}

function renderStatusLine(out, ms) {
  const f = out.filter || {};
  const parts = [];
  if (f.code) parts.push(`公司=${f.code}`);
  if (f.period) parts.push(`报告期=${f.period}`);
  const denseTag = out.denseUsed ? "BM25 + 向量" : "BM25";
  $("#status").innerHTML =
    `检索完成：${out.results ? out.results.length : out.groups.length} 条证据，${denseTag}，` +
    `${ms.toFixed(0)} ms` + (parts.length ? `，自动过滤：${parts.join("，")}` : "，未触发过滤（全库检索）");
}

function renderAnswer(query, out) {
  const bullets = composeExtractive(query, out.results, 3);
  const terms = queryTerms(query);
  const llm = llmEnabled();
  const body = bullets
    .map(
      (b) =>
        `<li>${highlight(b.text, terms)}<a class="cite" href="#ev-${b.citation}">[${b.citation}]</a></li>`
    )
    .join("");
  $("#answerArea").innerHTML = `
    <div class="card ans">
      <div class="anshead">
        <h2>答案要点</h2>
        <span class="tag">${llm ? "大模型生成中…" : "抽取式（直接引用原文）"}</span>
      </div>
      <ul>${body || "<li>没有检索到相关段落。</li>"}</ul>
      <div class="hint">条目直接来自下面命中的年报原文，点击 [编号] 可跳到证据卡片；每条证据都能点开巨潮资讯的原始 PDF 核对页码。</div>
      ${llm ? `<div id="llmBox" class="llmout"></div>` : ""}
    </div>`;
  if (llm) {
    llmAnswer(query, out.results, llm)
      .then((text) => {
        const box = $("#llmBox");
        if (box) box.innerHTML = `<hr>${escapeHtml(text).replace(/\[(\d+)\]/g, '<a class="cite" href="#ev-$1">[$1]</a>')}`;
        document.querySelector(".ans .tag").textContent = `大模型生成（${llm.model}）`;
      })
      .catch((err) => {
        const box = $("#llmBox");
        if (box) box.innerHTML = `<hr><span class="err">大模型调用失败：${escapeHtml(err.message)}</span>`;
      });
  }
}

function renderEvidence(out) {
  const terms = queryTerms(out.query);
  const cards = out.results.map((c) => evidenceCard(c, terms)).join("");
  $("#evidence").innerHTML = `<h2 class="muted" style="margin:18px 0 10px">检索到的原文块（${out.results.length}）</h2>${cards}`;
}

function evidenceCard(c, terms) {
  const scoreBits = [`融合 ${c.score.toFixed(3)}`];
  if (c.bm25Rank) scoreBits.push(`BM25 #${c.bm25Rank}`);
  if (c.denseRank) scoreBits.push(`向量 #${c.denseRank}`);
  const pages = c.pageEnd > c.pageStart ? `${c.pageStart}–${c.pageEnd}` : `${c.pageStart}`;
  return `
  <article class="ev" id="ev-${c.rank}">
    <div class="evhead">
      <span class="rank">${c.rank}</span>
      <span class="co">${escapeHtml(c.name)} <span class="muted">${c.code}</span></span>
      <span class="rep">${escapeHtml(c.report.title)}</span>
      <span class="badge ${c.type}">${c.type === "table" ? "表格" : "正文"}</span>
      <span class="pages">第 ${pages} 页</span>
      <span class="scores">${scoreBits.join(" · ")}</span>
    </div>
    <div class="sec">${escapeHtml(c.section)}${c.subsection ? " ／ " + escapeHtml(c.subsection) : ""}</div>
    <div class="evbody">${renderChunkBody(c.text, terms)}</div>
    <div class="evfoot">
      <a href="${c.report.url}" target="_blank" rel="noopener">在巨潮资讯查看原文 PDF ↗</a>
      <span class="muted">（${escapeHtml(c.report.period)}，公告日 ${escapeHtml(c.report.date)}）</span>
    </div>
  </article>`;
}

/** 把块文本渲染成 HTML：连续以 | 开头的行当作表格 */
function renderChunkBody(text, terms) {
  const lines = text.split("\n");
  const out = [];
  let tbl = [];
  const flushTable = () => {
    if (tbl.length >= 2) {
      const html = markdownTableToHtml(tbl.join("\n"));
      out.push(html || `<p>${highlight(tbl.join(" "), terms)}</p>`);
    } else if (tbl.length) {
      out.push(`<p>${highlight(tbl.join(" "), terms)}</p>`);
    }
    tbl = [];
  };
  for (const line of lines) {
    if (line.trim().startsWith("|")) {
      tbl.push(line.trim());
    } else {
      flushTable();
      if (line.trim()) out.push(`<p>${highlight(line.trim(), terms)}</p>`);
    }
  }
  flushTable();
  return out.join("");
}

function renderPanorama(query, out, ms) {
  const terms = queryTerms(query);
  const wantsProfit = query.includes("净利润");
  const wantsGrowth = /增速|增长|下滑|下降|最高/.test(query);
  const rows = out.groups.map((g) => {
    const text = g.results.map((c) => c.text).join("\n");
    const rev = extractMetric(text, "revenue");
    const pro = extractMetric(text, "profit");
    const pick = wantsProfit ? pro : rev;
    const other = wantsProfit ? rev : pro;
    const yoy = pick?.yoy ?? (pick && pick.prev ? (pick.now / pick.prev - 1) * 100 : null);
    const best = g.results[0];
    const src = best;
    return {
      company: g.company,
      value: pick ? pick.now : null,
      yoy,
      other,
      src,
      text,
    };
  });
  const sorted = rows.slice().sort((a, b) => {
    if (wantsGrowth) {
      const ay = a.yoy ?? -Infinity;
      const by = b.yoy ?? -Infinity;
      if (by !== ay) return by - ay;
    }
    return (b.value ?? -Infinity) - (a.value ?? -Infinity);
  });
  const metricName = wantsProfit ? "归母净利润" : "营业收入";
  const body = sorted
    .map(
      (r, i) => `<tr>
        <td class="c">${i + 1}. ${escapeHtml(r.company.name)}<br><span class="muted">${r.company.code}</span></td>
        <td class="num">${fmtYi(r.value)}</td>
        <td class="num ${r.yoy === null ? "" : r.yoy < 0 ? "down" : "up"}">${r.yoy === null ? "—" : (r.yoy > 0 ? "+" : "") + r.yoy.toFixed(2) + "%"}</td>
        <td class="num">${fmtYi(r.other?.now ?? null)}</td>
        <td class="muted">${escapeHtml(r.src.report.title)}｜${escapeHtml(r.src.section)}｜第 ${r.src.pageStart} 页
          · <a href="${r.src.report.url}" target="_blank" rel="noopener">原文</a></td>
      </tr>`
    )
    .join("");
  $("#answerArea").innerHTML = `
    <div class="card">
      <div class="anshead">
        <h2>跨公司全景对比（每家公司各取 3 条证据，自动解析「主要会计数据」表）</h2>
        <span class="tag">${out.groups.length} 家覆盖</span>
      </div>
      <p class="muted">指标：<b>${metricName}</b>${wantsGrowth ? "（按同比增速排序）" : "（按金额降序）"}。
      报告期：<b>${escapeHtml(out.period || "未指定（可能混合多个报告期）")}</b>。
      数字由检索到的原文表格自动解析，单位统一为亿元；点「原文」可核对页码。解析失败会显示 —，请以原文为准。</p>
      <table class="panotable">
        <thead><tr><th>公司</th><th>${metricName}</th><th>同比</th><th>${wantsProfit ? "营业收入" : "归母净利润"}</th><th>出处</th></tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>`;
  $("#evidence").innerHTML = "";
  $("#status").innerHTML = `全景检索完成：覆盖 ${out.groups.length} 家公司，指标查询「${escapeHtml(out.query)}」，用时 ${ms.toFixed(0)} ms`;
}

/* ---------- 大模型设置 ---------- */
function llmCfg() {
  return JSON.parse(localStorage.getItem("llmCfg") || '{"base":"","key":"","model":"","on":false}');
}
function llmEnabled() {
  const c = llmCfg();
  return c.on && c.base && c.key && c.model ? c : null;
}
function saveLlmCfg(cfg) {
  localStorage.setItem("llmCfg", JSON.stringify(cfg));
}

/* ---------- 评测 tab ---------- */
async function renderEval() {
  const host = $("#evalList");
  if (host.dataset.done) return;
  try {
    const data = await (await fetch(new URL("../data/eval.json", import.meta.url))).json();
    const ok = data.questions.filter((q) => q.verdict === "正确").length;
    const part = data.questions.filter((q) => q.verdict === "部分正确").length;
    const recalled = data.questions.filter((q) => q.gold_recalled).length;
    $("#evalSummary").innerHTML = `
      <div class="m"><b>${data.questions.length}</b><span>道题</span></div>
      <div class="m"><b>${recalled}/${data.questions.length}</b><span>关键证据被召回</span></div>
      <div class="m"><b>${ok}</b><span>完全正确</span></div>
      <div class="m"><b>${part}</b><span>部分正确</span></div>
      <div class="m"><b>${data.questions.filter((q) => q.panorama).length}</b><span>跨公司全景题</span></div>
      <div class="m"><b>${data.retriever || "BM25 + 向量"}</b><span>检索配置</span></div>`;
    host.innerHTML = data.questions
      .map((q, i) => {
        const cls = q.verdict_class || (q.verdict.startsWith("正确") ? "ok" : q.verdict.startsWith("部分") ? "part" : "bad");
        const rows = q.retrieved
          .map(
            (r) => `<tr class="${r.gold ? "gold" : ""}">
              <td>#${r.rank}</td><td>${escapeHtml(r.company)}</td><td>${escapeHtml(r.period)}</td>
              <td>${escapeHtml(r.section)}</td><td>第 ${r.page} 页</td>
              <td>${r.type === "table" ? "表格" : "正文"}</td>
              <td>${r.bm25_rank ?? "–"} / ${r.dense_rank ?? "–"}</td>
              <td>${r.gold ? "✔ 关键证据" : ""}</td></tr>`
          )
          .join("");
        return `<div class="card evalitem">
          <h3>Q${i + 1}　<span class="qline">${escapeHtml(q.question)}</span>
            <span class="verdict ${cls}">${q.verdict}</span>
            ${q.panorama ? '<span class="verdict part">跨公司</span>' : ""}</h3>
          <div class="meta"><b>标准答案：</b>${escapeHtml(q.expected)}</div>
          <div class="meta"><b>系统回答：</b>${escapeHtml(q.answer)}</div>
          <div class="meta"><b>判定：</b>${escapeHtml(q.analysis)}</div>
          ${q.facts && q.facts.length ? `<div class="meta"><b>关键事实召回：</b>${q.facts
            .map((f) =>
              q.panorama
                ? `${escapeHtml(f.name)} → ${f.companies_hit}/${f.companies_total} 家${f.recalled ? " ✔" : " ✘"}`
                : `${escapeHtml(f.name)} → ${f.recalled ? `第 ${f.hit_rank} 位 ✔` : "未召回 ✘"}`
            )
            .join("；")}</div>` : ""}
          <table class="recall">
            <thead><tr><th>排名</th><th>公司</th><th>报告期</th><th>章节</th><th>页码</th><th>类型</th><th>BM25/向量名次</th><th></th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>`;
      })
      .join("");
    host.dataset.done = "1";
  } catch (err) {
    host.innerHTML = `<div class="card"><span class="err">评测数据加载失败：${escapeHtml(err.message)}</span></div>`;
  }
}

/* ---------- 事件绑定 ---------- */
document.querySelectorAll(".tab").forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    $(`#tab-${tab.dataset.tab}`).classList.add("active");
    if (tab.dataset.tab === "eval") renderEval();
  };
});
$("#go").onclick = runSearch;
$("#q").addEventListener("keydown", (e) => {
  if (e.key === "Enter") runSearch();
});
$("#alpha").addEventListener("input", (e) => {
  $("#alphaval").textContent = Number(e.target.value).toFixed(2);
});
$("#methodBody").innerHTML = METHOD_HTML;

// 大模型设置面板（挂在检索设置里，避免首页太复杂）
const llmPanel = document.createElement("div");
llmPanel.className = "advgrid";
llmPanel.style.marginTop = "10px";
llmPanel.innerHTML = `
  <label class="check"><input id="llmOn" type="checkbox"> 用大模型生成答案</label>
  <label>API Base<input id="llmBase" placeholder="https://api.openai.com/v1"></label>
  <label>API Key<input id="llmKey" type="password" placeholder="sk-..."></label>
  <label>模型<input id="llmModel" placeholder="gpt-4o-mini / deepseek-chat ..."></label>
  <label>语义模型基址（可选，留空用仓库自带同源文件）
    <input id="modelBase" placeholder="https://你的镜像/...  需含 tokenizer.json 与 onnx/model_quantized.onnx"></label>`;
$(".adv").appendChild(llmPanel);
const cfg = llmCfg();
$("#llmOn").checked = cfg.on;
$("#llmBase").value = cfg.base;
$("#llmKey").value = cfg.key;
$("#llmModel").value = cfg.model;
["llmOn", "llmBase", "llmKey", "llmModel"].forEach((id) => {
  $(`#${id}`).addEventListener("change", () =>
    saveLlmCfg({
      on: $("#llmOn").checked,
      base: $("#llmBase").value.trim(),
      key: $("#llmKey").value.trim(),
      model: $("#llmModel").value.trim(),
    })
  );
});

boot();
