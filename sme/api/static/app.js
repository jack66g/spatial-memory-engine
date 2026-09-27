/* ============================================================
 * SME 配置中心 - 前端逻辑（纯原生 JS，无依赖、离线可用）
 *
 * 分段：
 *   1. API 基础（fetch 封装 / 鉴权重试）
 *   2. 全局状态
 *   3. 工具函数（转义 / 值渲染 / 类型描述）
 *   4. 渲染：分组导航 + 配置卡片 + 值控件
 *   5. 详情弹窗
 *   6. 保存 / 脏状态
 *   7. 预设 / 重置
 *   8. 测试连接面板
 *   9. 引擎状态摘要 / toast / 横幅
 *  10. 初始化
 * ============================================================ */

"use strict";

/* ---------------- 1. API 基础 ---------------- */

// 服务启用 Bearer 鉴权时，首次 401 会弹窗询问令牌并自动重试（存 sessionStorage）。
let authToken = sessionStorage.getItem("sme-auth-token") || "";

async function apiFetch(path, options) {
  const opts = Object.assign({}, options || {});
  opts.headers = Object.assign({ "Content-Type": "application/json" }, opts.headers || {});
  if (authToken) opts.headers["Authorization"] = "Bearer " + authToken;
  let res = await fetch(path, opts);
  if (res.status === 401) {
    const token = window.prompt("该服务已启用 Bearer 鉴权，请输入 API 访问令牌：");
    if (token && token.trim()) {
      authToken = token.trim();
      sessionStorage.setItem("sme-auth-token", authToken);
      opts.headers["Authorization"] = "Bearer " + authToken;
      res = await fetch(path, opts);
    }
  }
  return res;
}

async function apiJson(path, options) {
  let res;
  try {
    res = await apiFetch(path, options);
  } catch (err) {
    throw new Error("无法连接服务：" + err.message);
  }
  let data = null;
  try {
    data = await res.json();
  } catch (err) {
    /* 非 JSON 响应（如空体），保留 null */
  }
  if (!res.ok) {
    const detail = data && (data.detail || data.error);
    throw new Error(detail ? String(detail) : "请求失败（HTTP " + res.status + "）");
  }
  return data;
}

/* ---------------- 2. 全局状态 ---------------- */

const state = {
  config: null,          // GET /config 的完整响应
  edits: new Map(),      // path -> 编辑中的字符串值（未保存）
  errors: new Map(),     // path -> 最近一次保存失败的错误信息；"_global" 走横幅
  cards: new Map(),      // path -> { card, errorEl, control, item }
};

const $ = (id) => document.getElementById(id);

/* ---------------- 3. 工具函数 ---------------- */

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

// 控件可编辑字符串：bool -> "true"/"false"；空值 -> ""；
// interval 的关闭哨兵值（OFF_PERIOD = 1e9，见 config_items.py）-> "off"
function toEditableString(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number" && value >= 1e9) return "off";
  return String(value);
}

// 人类可读值：用于详情弹窗 / 默认值提示
function renderValue(value) {
  if (value === null || value === undefined) return "（未设置）";
  if (typeof value === "boolean") return value ? "开" : "关";
  if (typeof value === "number" && value >= 1e9) return "关闭";
  if (value === "") return "（空）";
  return String(value);
}

// 类型描述（与后端 type_hint 口径一致）
function describeType(item) {
  const range = (suffix) => {
    const lo = item.minimum !== null && item.minimum !== undefined ? " ≥" + item.minimum : "";
    const hi = item.maximum !== null && item.maximum !== undefined ? " ≤" + item.maximum : "";
    return suffix + "（" + lo + hi + "）";
  };
  switch (item.type) {
    case "bool": return "开/关（true/false）";
    case "enum": return "可选：" + (item.choices || []).map((c) => c || "空").join(" / ");
    case "interval": return "整数，或 off（关闭）";
    case "int": return range("整数");
    case "float": return range("数字");
    default: return "字符串";
  }
}

const SOURCE_LABELS = { defaults: "内置默认", file: "配置文件", env: "环境变量" };

/* ---------------- 4. 渲染：导航 + 卡片 + 控件 ---------------- */

function renderNav(groups) {
  const nav = $("group-nav");
  nav.innerHTML = "";
  for (const group of groups) {
    const link = document.createElement("a");
    link.href = "#group-" + group.key;
    link.dataset.target = "group-" + group.key;
    link.innerHTML =
      "<span>" + escapeHtml(group.name) + "</span>" +
      '<span class="count">' + group.items.length + "</span>";
    link.addEventListener("click", (event) => {
      event.preventDefault();
      setActiveNav(link);
      document.getElementById("group-" + group.key)
        .scrollIntoView({ behavior: "smooth", block: "start" });
    });
    nav.appendChild(link);
  }
}

function setActiveNav(activeLink) {
  document.querySelectorAll("#group-nav a").forEach((a) => {
    a.classList.toggle("active", a === activeLink);
  });
}

// 滚动时同步左侧导航高亮
window.addEventListener("scroll", () => {
  if (!state.config) return;
  let current = null;
  for (const group of state.config.groups) {
    const el = document.getElementById("group-" + group.key);
    if (el && el.getBoundingClientRect().top <= 120) current = el.id;
  }
  if (current) {
    const link = document.querySelector('#group-nav a[data-target="' + current + '"]');
    if (link) setActiveNav(link);
  }
}, { passive: true });

function renderGroups(groups) {
  state.cards.clear();
  const wrap = $("groups");
  wrap.innerHTML = "";
  for (const group of groups) {
    const section = document.createElement("section");
    section.id = "group-" + group.key;
    section.className = "group";
    section.innerHTML =
      '<div class="group-head"><h2>' + escapeHtml(group.name) + "</h2>" +
      '<p class="group-desc">' + escapeHtml(group.description || "") + "</p></div>";

    const cards = document.createElement("div");
    cards.className = "cards";
    for (const item of group.items) cards.appendChild(buildCard(item));
    section.appendChild(cards);
    wrap.appendChild(section);
  }
}

function buildCard(item) {
  const card = document.createElement("div");
  card.className = "card";

  const head = document.createElement("div");
  head.className = "card-row";
  head.innerHTML =
    '<span class="card-name">' + escapeHtml(item.name) + "</span>" +
    '<code class="card-path">' + escapeHtml(item.path) + "</code>";
  const detailBtn = document.createElement("button");
  detailBtn.type = "button";
  detailBtn.className = "card-detail-link";
  detailBtn.textContent = "详情";
  detailBtn.addEventListener("click", () => openDetail(item));
  head.appendChild(detailBtn);
  card.appendChild(head);

  const controlRow = document.createElement("div");
  controlRow.className = "control-row";
  const control = buildControl(item);
  controlRow.appendChild(control.element);
  card.appendChild(controlRow);

  const errorEl = document.createElement("div");
  errorEl.className = "card-error";
  errorEl.hidden = true;
  card.appendChild(errorEl);

  state.cards.set(item.path, { card, errorEl, control, item });
  refreshCardState(item.path);
  return card;
}

// 按类型构建值控件：bool 开关 / enum 下拉 / int+float 数字 / interval、str 文本
function buildControl(item) {
  const wrap = document.createElement("div");
  wrap.style.cssText = "display:flex;align-items:center;gap:8px;flex:1;min-width:0;";

  const commit = (rawString) => {
    if (rawString === toEditableString(item.value)) state.edits.delete(item.path);
    else state.edits.set(item.path, rawString);
    state.errors.delete(item.path);
    refreshCardState(item.path);
    updateSaveButton();
  };

  if (item.type === "bool") {
    const label = document.createElement("label");
    label.className = "switch";
    const input = document.createElement("input");
    input.type = "checkbox";
    const slider = document.createElement("span");
    slider.className = "slider";
    label.append(input, slider);
    const text = document.createElement("span");
    text.className = "switch-label";
    const sync = () => {
      input.checked = currentRaw(item) === "true";
      text.textContent = input.checked ? "开" : "关";
    };
    input.addEventListener("change", () => commit(input.checked ? "true" : "false"));
    wrap.append(label, text);
    return { element: wrap, syncFromEdits: sync };
  }

  let input;
  if (item.type === "enum") {
    input = document.createElement("select");
    for (const choice of item.choices || []) {
      const option = document.createElement("option");
      option.value = choice;
      option.textContent = choice === "" ? "（空）" : choice;
      input.appendChild(option);
    }
    input.addEventListener("change", () => commit(input.value));
  } else {
    input = document.createElement("input");
    if (item.type === "int") {
      input.type = "number";
      input.step = "1";
    } else if (item.type === "float") {
      input.type = "number";
      input.step = "any";
    } else {
      input.type = "text";
    }
    if (item.type === "interval") input.placeholder = "整数或 off";
    if (item.minimum !== null && item.minimum !== undefined) input.min = item.minimum;
    if (item.maximum !== null && item.maximum !== undefined) input.max = item.maximum;
    input.addEventListener("input", () => commit(input.value));
  }
  input.style.flex = "1";
  input.style.minWidth = "0";
  wrap.appendChild(input);
  const sync = () => { input.value = currentRaw(item); };
  return { element: wrap, setEnabled: sync, syncFromEdits: sync };
}

// 当前应显示的原始字符串（优先取未保存编辑）
function currentRaw(item) {
  return state.edits.has(item.path) ? state.edits.get(item.path) : toEditableString(item.value);
}

// 刷新卡片“已修改”标记、默认值提示与错误红字
function refreshCardState(path) {
  const entry = state.cards.get(path);
  if (!entry) return;
  const { card, errorEl, control, item } = entry;

  const modified = isModified(item);
  card.classList.toggle("modified", modified);

  let chip = card.querySelector(".default-chip");
  if (modified) {
    if (!chip) {
      chip = document.createElement("span");
      chip.className = "default-chip";
      card.insertBefore(chip, errorEl);
    }
    chip.textContent = "默认：" + renderValue(item.default);
    const dot = card.querySelector(".card-name");
    if (dot && !dot.querySelector(".modified-dot")) {
      const marker = document.createElement("span");
      marker.className = "modified-dot";
      marker.title = "与默认值不同";
      dot.insertBefore(marker, dot.firstChild);
    }
  } else if (chip) {
    chip.remove();
    const marker = card.querySelector(".modified-dot");
    if (marker) marker.remove();
  }

  control.syncFromEdits();

  const error = state.errors.get(path);
  errorEl.hidden = !error;
  errorEl.textContent = error || "";
}

/* ---------------- 5. 详情弹窗 ---------------- */

function openDetail(item) {
  // 密钥类配置只回显掩码：详情弹窗展示"是否已设置"，不展示掩码原文
  const isApiKey = typeof item.path === "string" && item.path.endsWith(".api_key");
  const currentText = isApiKey
    ? (item.set ? "已设置（隐藏显示）" : "（空）")
    : renderValue(currentDisplayValue(item));
  const rows = [
    ["当前值", currentText],
    ["默认值", renderValue(item.default)],
    ["类型", describeType(item)],
    ["状态", isModified(item) ? "已修改（未保存则仍用当前值生效）" : "未修改"],
  ];
  if (item.choices && item.choices.length) {
    rows.push(["可选值", item.choices.map((c) => c || "（空）").join(" / ")]);
  }
  if (item.minimum !== null && item.minimum !== undefined) rows.push(["最小值", String(item.minimum)]);
  if (item.maximum !== null && item.maximum !== undefined) rows.push(["最大值", String(item.maximum)]);

  const grid = rows.map(([k, v]) =>
    "<dt>" + escapeHtml(k) + "</dt><dd>" + escapeHtml(v) + "</dd>").join("");

  $("detail-body").innerHTML =
    '<div class="detail-path">' + escapeHtml(item.path) + "</div>" +
    '<div class="detail-help">' + escapeHtml(item.help || "（暂无说明）") + "</div>" +
    '<dl class="detail-grid">' + grid + "</dl>" +
    '<div class="detail-hint">修改提示：保存后引擎立即以新配置重建，现有记忆自动迁移；' +
    "仅修改不保存时，新配置只在本次运行内生效。</div>";
  $("modal-backdrop").hidden = false;
}

// 该项是否处于“已修改”状态（编辑值与当前生效值不同，或服务端标记 modified）
function isModified(item) {
  if (state.edits.has(item.path)) {
    return state.edits.get(item.path) !== toEditableString(item.value);
  }
  return Boolean(item.modified);
}

function currentDisplayValue(item) {
  if (state.edits.has(item.path)) return state.edits.get(item.path);
  if (item.value === null || item.value === undefined) return "";
  return item.value;
}

function closeModals() {
  $("modal-backdrop").hidden = true;
  $("check-backdrop").hidden = true;
}

/* ---------------- 6. 保存 / 脏状态 ---------------- */

function updateSaveButton() {
  const btn = $("btn-save");
  const count = state.edits.size;
  btn.textContent = count > 0 ? "保存（" + count + "）" : "保存";
  btn.disabled = count === 0;
}

async function saveEdits() {
  if (state.edits.size === 0) return;
  const values = {};
  for (const [path, raw] of state.edits) values[path] = raw;
  try {
    const result = await apiJson("/config", {
      method: "PUT",
      body: JSON.stringify({ values: values, save: true }),
    });
    if (!result.ok) {
      showApplyErrors(result.errors || {});
      return;
    }
    state.edits.clear();
    state.errors.clear();
    hideBanner();
    await loadConfig();
    await refreshStats();
    showToast("已保存并生效 " + result.applied + " 项，引擎已重建" +
      (result.config_file ? "，配置已写入 " + result.config_file : ""));
  } catch (err) {
    showBanner(String(err.message || err));
  }
}

// 保存/预设失败：逐项红字（含 _global 横幅）；对应不到卡片的 path（如
// "未知的配置项"）并入顶部横幅展示
function showApplyErrors(errors) {
  state.errors.clear();
  document.querySelectorAll(".card-error").forEach((el) => { el.hidden = true; });
  let globalMsg = "";
  const unmatched = [];
  let count = 0;
  for (const [path, message] of Object.entries(errors)) {
    if (path === "_global") { globalMsg = message; continue; }
    if (!state.cards.has(path)) {
      unmatched.push(path + "：" + message);
      continue;
    }
    state.errors.set(path, message);
    refreshCardState(path);
    count += 1;
  }
  const bannerParts = [];
  if (globalMsg) bannerParts.push(globalMsg);
  if (unmatched.length) bannerParts.push(unmatched.join("；"));
  if (bannerParts.length) showBanner(bannerParts.join(" "));
  else if (count > 0) showBanner(count + " 项配置未通过校验，请按红字提示修正后重试。");
  else hideBanner();
}

// 有未保存修改时，离开页面前提醒
window.addEventListener("beforeunload", (event) => {
  if (state.edits.size > 0) {
    event.preventDefault();
    event.returnValue = "";
  }
});

/* ---------------- 7. 预设 / 重置 ---------------- */

function fillPresetSelect(presets) {
  const select = $("preset-select");
  select.innerHTML = '<option value="">套用预设…</option>';
  for (const preset of presets) {
    const option = document.createElement("option");
    option.value = preset.key;
    option.textContent = preset.name;
    select.appendChild(option);
  }
}

async function applyPreset(key) {
  const preset = (state.config.presets || []).find((p) => p.key === key);
  const count = preset && preset.count ? preset.count : "";
  const name = preset ? preset.name : key;
  const message = "套用预设「" + name + "」将覆盖 " + count + " 项并重建引擎，现有记忆自动迁移。确定继续？";
  if (!window.confirm(message)) return;
  try {
    const result = await apiJson("/config/preset", {
      method: "POST",
      body: JSON.stringify({ key: key }),
    });
    if (!result.ok) {
      showApplyErrors(result.errors || {});
      return;
    }
    state.edits.clear();
    state.errors.clear();
    hideBanner();
    await loadConfig();
    await refreshStats();
    showToast("预设「" + name + "」已生效（" + result.applied.length + " 项）");
  } catch (err) {
    showBanner(String(err.message || err));
  }
}

async function resetConfig() {
  const file = state.config ? state.config.config_file : "data/sme.config.json";
  if (!window.confirm("重置为默认将删除配置文件 " + file +
      "，并以默认配置重建引擎（现有记忆自动迁移）。确定继续？")) return;
  try {
    const result = await apiJson("/config/reset", { method: "POST", body: "{}" });
    if (!result.ok) {
      showApplyErrors(result.errors || {});
      return;
    }
    state.edits.clear();
    state.errors.clear();
    hideBanner();
    await loadConfig();
    await refreshStats();
    showToast("已重置为默认配置，引擎已重建");
  } catch (err) {
    showBanner(String(err.message || err));
  }
}

/* ---------------- 8. 测试连接面板 ---------------- */

function probeRow(title, probe) {
  let status;
  let cls;
  if (!probe.configured) {
    status = probe.ok ? "未配置" : "异常";
    cls = probe.ok ? "status-idle" : "status-bad";
  } else {
    status = probe.ok ? "正常" : "失败";
    cls = probe.ok ? "status-ok" : "status-bad";
  }
  return '<div class="probe"><span class="' + cls + '">' + escapeHtml(status) + "</span>" +
    '<span><strong>' + escapeHtml(title) + "</strong>" +
    escapeHtml(probe.detail ? "：" + probe.detail : "") + "</span></div>";
}

async function runCheck() {
  const box = $("check-result");
  const ping = $("check-ping").checked;
  box.textContent = "检测中…";
  try {
    const result = await apiJson("/config/check", {
      method: "POST",
      body: JSON.stringify({ ping: ping }),
    });
    let html = result.config_valid
      ? '<div class="check-valid-ok">配置可用（config_valid = true）</div>'
      : '<div class="check-valid-bad">配置存在问题（config_valid = false）</div>';
    if (result.errors && result.errors.length) {
      html += '<ul class="errors">' +
        result.errors.map((e) => "<li>" + escapeHtml(e) + "</li>").join("") + "</ul>";
    }
    html += probeRow("LLM", result.llm || {});
    html += probeRow("Embedding", result.embedding || {});
    if (!result.engine_build_ok) {
      html += '<div class="check-valid-bad">引擎构建失败，无法探测连通性。</div>';
    }
    box.innerHTML = html;
  } catch (err) {
    box.textContent = "检测失败：" + (err.message || err);
  }
}

/* ---------------- 9. 状态摘要 / toast / 横幅 ---------------- */

async function refreshStats() {
  const el = $("engine-summary");
  try {
    const stats = await apiJson("/stats");
    const memories = (stats.memories || {}).total || 0;
    const active = (stats.memories || {}).active || 0;
    const regions = (stats.regions || {}).count || 0;
    el.textContent = "记忆 " + memories + "（活跃 " + active + "）· Region " + regions;
  } catch (err) {
    el.textContent = "引擎状态获取失败";
  }
}

function updateSourceBadge() {
  const badge = $("source-badge");
  const source = state.config ? state.config.source : "defaults";
  badge.textContent = "来源：" + (SOURCE_LABELS[source] || source);
  badge.className = "badge " + source;
  badge.title = "配置文件：" + (state.config ? state.config.config_file : "-");
}

let toastTimer = null;
function showToast(message, isError) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.toggle("error", Boolean(isError));
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 3200);
}

function showBanner(message) {
  $("banner-text").textContent = message;
  $("banner").hidden = false;
}

function hideBanner() {
  $("banner").hidden = true;
}

/* ---------------- 10. 初始化 ---------------- */

async function loadConfig() {
  const config = await apiJson("/config");
  state.config = config;
  // 保留编辑值：仅保留仍存在的配置路径
  const known = new Set();
  for (const group of config.groups) for (const item of group.items) known.add(item.path);
  for (const path of Array.from(state.edits.keys())) {
    if (!known.has(path)) state.edits.delete(path);
  }
  updateSourceBadge();
  fillPresetSelect(config.presets || []);
  renderNav(config.groups);
  renderGroups(config.groups);
  updateSaveButton();
}

async function init() {
  $("btn-save").addEventListener("click", saveEdits);
  $("btn-check").addEventListener("click", () => { $("check-backdrop").hidden = false; });
  $("btn-reset").addEventListener("click", resetConfig);
  $("preset-select").addEventListener("change", (event) => {
    const key = event.target.value;
    event.target.value = "";
    if (key) applyPreset(key);
  });
  $("check-run").addEventListener("click", runCheck);
  $("detail-close").addEventListener("click", closeModals);
  $("check-close").addEventListener("click", closeModals);
  $("banner-close").addEventListener("click", hideBanner);
  $("modal-backdrop").addEventListener("click", (event) => {
    if (event.target === event.currentTarget) closeModals();
  });
  $("check-backdrop").addEventListener("click", (event) => {
    if (event.target === event.currentTarget) closeModals();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeModals();
  });

  try {
    await loadConfig();
    await refreshStats();
  } catch (err) {
    showBanner("加载配置失败：" + (err.message || err));
  }
}

init();
