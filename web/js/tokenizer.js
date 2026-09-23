/* BERT WordPiece 分词器（与 Python 端 HuggingFace tokenizers 对齐）
 * 用于把查询编码成 bge-small-zh-v1.5 的输入，保证浏览器与离线索引完全一致。
 */

function isChineseChar(cp) {
  return (
    (cp >= 0x4e00 && cp <= 0x9fff) ||
    (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0x20000 && cp <= 0x2a6df) ||
    (cp >= 0x2a700 && cp <= 0x2b73f) ||
    (cp >= 0x2b740 && cp <= 0x2b81f) ||
    (cp >= 0x2b820 && cp <= 0x2ceaf) ||
    (cp >= 0xf900 && cp <= 0xfaff) ||
    (cp >= 0x2f800 && cp <= 0x2fa1f)
  );
}

const PUNCT_ASCII = [
  [0x21, 0x2f],
  [0x3a, 0x40],
  [0x5b, 0x60],
  [0x7b, 0x7e],
];

function isPunctuation(cp) {
  for (const [a, b] of PUNCT_ASCII) if (cp >= a && cp <= b) return true;
  // Unicode 类别 P*（全角标点等）
  const ch = String.fromCodePoint(cp);
  return /\p{P}/u.test(ch);
}

function isWhitespace(cp) {
  if (cp === 0x20 || cp === 0x09 || cp === 0x0a || cp === 0x0d) return true;
  return /\s/u.test(String.fromCodePoint(cp));
}

function isControl(cp) {
  const ch = String.fromCodePoint(cp);
  return /\p{Cc}|\p{Cf}/u.test(ch);
}

/** 与 HF BertNormalizer(clean_text, handle_chinese_chars, lowercase=false) 一致 */
export function normalizeBert(text) {
  let out = "";
  for (const ch of text) {
    const cp = ch.codePointAt(0);
    if (cp === 0 || cp === 0xfffd || isControl(cp)) continue;
    if (isWhitespace(cp)) {
      out += " ";
    } else if (isChineseChar(cp)) {
      out += " " + ch + " ";
    } else {
      out += ch;
    }
  }
  return out;
}

/** BertPreTokenizer：按空白切分，并把标点单独切出来 */
export function preTokenize(text) {
  const tokens = [];
  let buf = "";
  for (const ch of text) {
    const cp = ch.codePointAt(0);
    if (isWhitespace(cp)) {
      if (buf) tokens.push(buf);
      buf = "";
    } else if (isPunctuation(cp)) {
      if (buf) tokens.push(buf);
      buf = "";
      tokens.push(ch);
    } else {
      buf += ch;
    }
  }
  if (buf) tokens.push(buf);
  return tokens;
}

export class BertTokenizer {
  constructor(tokenizerJson) {
    this.vocab = tokenizerJson.model.vocab; // token -> id
    this.unkToken = tokenizerJson.model.unk_token || "[UNK]";
    this.maxInputChars = tokenizerJson.model.max_input_chars_per_word || 100;
    this.clsId = this.vocab["[CLS]"];
    this.sepId = this.vocab["[SEP]"];
    this.padId = this.vocab["[PAD]"] ?? 0;
    this.unkId = this.vocab[this.unkToken];
    this.prefix = tokenizerJson.model.continuing_subword_prefix || "##";
    this.cache = new Map();
  }

  wordpiece(token) {
    const cached = this.cache.get(token);
    if (cached) return cached;
    const chars = Array.from(token);
    const ids = [];
    let start = 0;
    let ok = true;
    if (chars.length > this.maxInputChars) {
      ok = false;
    } else {
      while (start < chars.length) {
        let end = chars.length;
        let cur = null;
        while (start < end) {
          let sub = chars.slice(start, end).join("");
          if (start > 0) sub = this.prefix + sub;
          if (Object.prototype.hasOwnProperty.call(this.vocab, sub)) {
            cur = this.vocab[sub];
            break;
          }
          end -= 1;
        }
        if (cur === null) {
          ok = false;
          break;
        }
        ids.push(cur);
        start = end;
      }
    }
    const result = ok ? ids : [this.unkId];
    this.cache.set(token, result);
    return result;
  }

  /** 返回 {ids, attentionMask}；已加 [CLS]/[SEP] 并按 maxLen 截断 */
  encode(text, maxLen = 512) {
    const ids = [this.clsId];
    for (const tok of preTokenize(normalizeBert(text))) {
      for (const id of this.wordpiece(tok)) ids.push(id);
    }
    ids.push(this.sepId);
    const trunc = ids.slice(0, maxLen);
    trunc[trunc.length - 1] = this.sepId;
    return { ids: trunc, attentionMask: trunc.map(() => 1) };
  }
}
