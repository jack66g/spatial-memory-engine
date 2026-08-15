// ============================================================================
// mem3d — DeepSeek Harness 持久化插件：Host 半区（普通 Cordis 插件版）
// ============================================================================
// 与 host-plugin.js（动态插件沙箱版，使用 harness）不同，本文件供 agent
// preset 的 agent.cordis.yml 以普通插件行加载：
//
//     - id: mem3d-host
//       name: 'C:/Users/黄小乐/Desktop/mem/mem3d/host-preset.js'
//
// 普通插件环境没有动态沙箱的 `harness`/`styles`/`btoa` 等注入符号，因此：
//   - 记忆工具改用 ctx.get('tools').register(...) 注册（标准 ToolDefinition）；
//   - 不提供 Client RPC（浏览器侧 3D 浮窗直连 bridge HTTP，见 README §5）；
//   - 其余（边车懒启动/健康复用、会话事件自动写入、thinking 过滤、去重、
//     workspace-write 受管 grant 沙箱策略）与动态版一致。
// ============================================================================
const fs = require('node:fs')
const path = require('node:path')
const BOOT_MARK = path.join(__dirname, '..', 'mem3d', 'data', 'host-boot.json')
module.exports = {
  apply(ctx) {
    try {
      fs.mkdirSync(path.dirname(BOOT_MARK), { recursive: true })
      const toolsInfo = { present: false }
      try {
        const t = ctx.get('tools')
        toolsInfo.present = !!t
        toolsInfo.register = t && typeof t.register === 'function' ? 'fn' : String(t && typeof t.register)
        toolsInfo.schemas = t && typeof t.schemas === 'function' ? (Array.isArray(t.schemas()) ? t.schemas().length : 'n/a') : 'no-schemas'
      } catch (e) { toolsInfo.error = String(e && e.message || e) }
      fs.writeFileSync(BOOT_MARK, JSON.stringify({ ok: true, at: new Date().toISOString(), ctx: typeof ctx, tools: toolsInfo }))
    } catch (e) {
      try { fs.writeFileSync(BOOT_MARK, JSON.stringify({ ok: false, at: new Date().toISOString(), error: String(e && e.message || e) })) } catch (_) {}
    }
    try {
    const shell = ctx.get('shell')
    const timer = ctx.get('timer')
    const tools = ctx.get('tools')
    const state = {
      port: 8756,
      root: null,
      proc: null,
      spawning: null,
      sessionId: null,
      lastError: '',
      recentHashes: [],
      hashSet: new Set(),
      stats: { writes: 0, errors: 0 },
    }
    const CANDIDATES = [
      'C:\\Users\\黄小乐\\Desktop\\mem',
      'C:\\Users\\黄小乐\\Desktop\\记忆插件',
    ]
    // 带 sessionId 的 policy 走沙箱 seam 的受管 grant 路径（与工具调用一致，
    // workspace ACE 由 seam 物化并长期站立）；无 sessionId 时 runner 自管
    // grant，进程退出会撤销 ACE，与其他受限进程互相干扰。
    function policyFor() {
      return {
        mode: 'workspace-write',
        workspaceRoot: state.root || CANDIDATES[0],
        sessionId: state.sessionId || 'mem3d-bridge',
      }
    }

    function sleep(ms) {
      if (timer && timer.timeout) return timer.timeout(ms)
      return new Promise((resolve) => {
        if (typeof setTimeout === 'function') setTimeout(resolve, ms)
        else resolve()
      })
    }

    async function pwshJson(command, timeoutMs) {
      if (!shell) return { error: 'shell service unavailable' }
      try {
        const spec = shell.resolve({
          command,
          workdir: state.root || CANDIDATES[0],
          timeoutMs: timeoutMs || 8000,
          stdoutMaxBytes: 8 * 1024 * 1024,
          sandboxPolicy: policyFor(),
        })
        const res = await shell.run(spec)
        const out = ((res.stdout && res.stdout.text) || '').trim()
        if (res.exitCode !== 0) {
          const errText = ((res.stderr && res.stderr.text) || '').trim().slice(0, 300)
          return { error: errText || ('exit ' + res.exitCode) }
        }
        if (!out) return null
        try {
          return JSON.parse(out)
        } catch (e) {
          return { error: 'bad json: ' + out.slice(0, 160) }
        }
      } catch (e) {
        return { error: String((e && e.message) || e) }
      }
    }

    async function findRoot() {
      for (const cand of CANDIDATES) {
        const r = await pwshJson(
          "if (Test-Path '" + cand + "\\mem3d\\bridge.py') { 'true' } else { 'false' }",
          6000,
        )
        if (r === 'true') {
          state.root = cand
          return cand
        }
      }
      state.root = CANDIDATES[0]
      return state.root
    }

    function baseUrl() {
      return 'http://127.0.0.1:' + state.port
    }

    function jsonCmd(method, path, body) {
      const url = baseUrl() + path
      if (method === 'GET') {
        return "(Invoke-RestMethod -Uri '" + url + "' -TimeoutSec 8) | ConvertTo-Json -Compress -Depth 40"
      }
      const b = JSON.stringify(body || {}).replace(/'/g, "''")
      return "(Invoke-RestMethod -Uri '" + url + "' -Method Post -Body '" + b +
        "' -ContentType 'application/json; charset=utf-8' -TimeoutSec 8) | ConvertTo-Json -Compress -Depth 40"
    }

    async function health() {
      const h = await pwshJson(jsonCmd('GET', '/health'), 4000)
      return h && h.ok ? h : null
    }

    async function ensureBridge() {
      const h = await health()
      if (h) return true
      if (state.spawning) return state.spawning
      state.spawning = (async () => {
        try {
          if (!shell) {
            state.lastError = 'shell service unavailable'
            return false
          }
          await findRoot()
          const cmd = "python '" + state.root + "\\mem3d\\bridge.py' --host 127.0.0.1 --port " + state.port
          const spec = shell.resolve({
            command: cmd,
            workdir: state.root,
            timeoutMs: 20000,
            stdoutMaxBytes: 65536,
            sandboxPolicy: policyFor(),
          })
          state.proc = shell.start(spec)
          for (let i = 0; i < 30; i++) {
            await sleep(500)
            const h2 = await health()
            if (h2) return true
          }
          let diag = ''
          if (state.proc && typeof state.proc.readOutput === 'function') {
            const out = state.proc.readOutput()
            diag = (out && out.delta ? out.delta : '').slice(0, 300)
          }
          state.lastError = 'bridge failed to start: ' + diag
          return false
        } catch (e) {
          state.lastError = String((e && e.message) || e)
          return false
        } finally {
          state.spawning = null
        }
      })()
      return state.spawning
    }

    function fnv1a(s) {
      let h = 0x811c9dc5
      for (let i = 0; i < s.length; i++) {
        h ^= s.charCodeAt(i)
        h = (h * 0x01000193) >>> 0
      }
      return h
    }

    function seen(text) {
      const h = fnv1a(text)
      if (state.hashSet.has(h)) return true
      state.hashSet.add(h)
      state.recentHashes.push(h)
      if (state.recentHashes.length > 400) state.hashSet.delete(state.recentHashes.shift())
      return false
    }

    // 只读叶子字段提取文本；不递归、不序列化 live data；跳过 thinking 块
    function findText(obj) {
      if (obj == null) return null
      if (typeof obj === 'string') return obj.slice(0, 2000)
      if (Array.isArray(obj)) {
        let text = null
        for (let i = 0; i < obj.length && i < 20; i++) {
          const it = obj[i]
          if (typeof it === 'string') {
            text = it.slice(0, 2000)
            break
          }
          if (it && typeof it === 'object' && typeof it.text === 'string') {
            const bt = typeof it.type === 'string' ? it.type.toLowerCase() : ''
            if (bt.indexOf('think') >= 0 || bt.indexOf('reason') >= 0) continue
            text = it.text.slice(0, 2000)
            break
          }
        }
        return text
      }
      if (typeof obj !== 'object') return null
      const ot = typeof obj.type === 'string' ? obj.type.toLowerCase() : ''
      if (ot.indexOf('think') >= 0 || ot.indexOf('reason') >= 0) return null
      for (const k of ['text', 'delta', 'content']) {
        const v = obj[k]
        if (typeof v === 'string' && v.trim().length >= 2) return v.slice(0, 2000)
      }
      const msg = obj.message
      if (msg && typeof msg === 'object') {
        const c = msg.content
        if (typeof c === 'string') return c.slice(0, 2000)
        if (Array.isArray(c)) {
          for (let i = 0; i < c.length && i < 20; i++) {
            const it = c[i]
            if (!it || typeof it !== 'object' || typeof it.text !== 'string') continue
            const bt = typeof it.type === 'string' ? it.type.toLowerCase() : ''
            if (bt.indexOf('think') >= 0 || bt.indexOf('reason') >= 0) continue
            return it.text.slice(0, 2000)
          }
        }
        const mt = msg.text
        if (typeof mt === 'string' && mt.trim().length >= 2) return mt.slice(0, 2000)
      }
      return null
    }

    function extractFromSessionEvent(event) {
      const t = event && typeof event.type === 'string' ? event.type : ''
      if (!t) return null
      if (!/message|user|assistant/.test(t)) return null
      if (t.indexOf('chunk') >= 0 || t.indexOf('step') >= 0 || t.indexOf('tool') >= 0 || t.indexOf('request') >= 0) return null
      const d = event.data
      if (!d || typeof d !== 'object') return null
      // 角色判定要穿透 data.message.role（assistant 消息的 role 常藏在这里），
      // 否则助手回复会被误判为 user 写入记忆。
      let role = typeof d.role === 'string' ? d.role : null
      if (!role && d.message && typeof d.message.role === 'string') role = d.message.role
      if (!role && d.message && d.message.message && typeof d.message.message.role === 'string') role = d.message.message.role
      if (!role) role = t.indexOf('assistant') >= 0 ? 'assistant' : (t.indexOf('user') >= 0 ? 'user' : 'user')
      // 助手消息不自动入库：内部思考与回复混在同一事件流中，逐块过滤无法
      // 完全防污染；助手侧重要结论由 sme3d_remember 显式记录。
      if (role === 'assistant') return null
      const text = findText(d)
      if (!text || text.trim().length < 2) return null
      return { text: text.trim(), role }
    }

    async function autoRemember(text, role) {
      try {
        if (!text || seen(text)) return
        const ok = await ensureBridge()
        if (!ok) return
        const res = await pwshJson(jsonCmd('POST', '/memory', { text, role: role || 'user', source: 'auto' }), 8000)
        if (res && res.error) {
          state.stats.errors++
          state.lastError = String(res.error).slice(0, 200)
        } else if (res && res.id) {
          state.stats.writes++
        }
      } catch (e) {
        state.stats.errors++
      }
    }

    ctx.on('agent/inbox/inserted', (payload) => {
      const m = payload && payload.message
      if (!m) return
      const role = typeof m.role === 'string' ? m.role : 'user'
      const text = findText(m)
      if (text && text.trim().length >= 2) autoRemember(text.trim(), role)
    })

    ctx.on('session/event', (session, event) => {
      try {
        if (session && typeof session.id === 'string') state.sessionId = session.id
        const got = extractFromSessionEvent(event)
        if (got) autoRemember(got.text, got.role)
      } catch (e) {
        /* 事件形状探测失败不致命 */
      }
    })

    // ---------------- 模型可见的记忆工具（标准 ToolDefinition） ----------------
    function makeJsonTool(name, description, parameters, execute) {
      return {
        name,
        description,
        parameters,
        output: {
          schema: { description: '任意 JSON 结果（无约束）' },
          render: (args, value) => [{ type: 'text', text: JSON.stringify(value) }],
        },
        execute,
      }
    }

    if (tools && typeof tools.register === 'function') {
      const rememberTool = makeJsonTool(
        'sme3d_remember',
        '把一条信息写入 3D 空间记忆引擎（当前模式）。role 默认 user。写入后 3D 浮窗会实时出现新的记忆点。',
        {
          type: 'object',
          properties: {
            text: { type: 'string', description: '要写入记忆的文本内容' },
            role: { type: 'string', description: '来源角色：user / assistant / system，默认 user' },
          },
          required: ['text'],
        },
        async (args) => {
          const ok = await ensureBridge()
          if (!ok) return { error: state.lastError || 'bridge unavailable' }
          const r = await pwshJson(jsonCmd('POST', '/memory', {
            text: String(args.text).trim(),
            role: typeof args.role === 'string' ? args.role : 'user',
            source: 'tool',
          }), 10000)
          if (r && r.id) state.stats.writes++
          return r || { error: 'no response' }
        },
      )
      tools.register(rememberTool)

      const recallTool = makeJsonTool(
        'sme3d_recall',
        '从 3D 空间记忆引擎检索相关记忆（当前模式），返回带相关度分数的命中列表。',
        {
          type: 'object',
          properties: {
            text: { type: 'string', description: '查询文本' },
            top_k: { type: 'number', description: '返回条数，默认 6' },
          },
          required: ['text'],
        },
        async (args) => {
          const ok = await ensureBridge()
          if (!ok) return { error: state.lastError || 'bridge unavailable', hits: [] }
          const r = await pwshJson(jsonCmd('POST', '/recall', {
            text: String(args.text).trim(),
            top_k: typeof args.top_k === 'number' ? args.top_k : 6,
          }), 10000)
          return r || { hits: [] }
        },
      )
      tools.register(recallTool)

      const modeTool = makeJsonTool(
        'sme3d_mode',
        '查看或切换 3D 记忆模式（共 8 种：chat 对话记忆 / semantic 语义记忆 / focus 只记事实 / v2 深度对话 / kb_dynamic 知识库动态 / kb_static 知识库静态 / robot 具身机器人 / minimal 裸向量库）。不传 id 列出全部模式；传 id 则切换。注意：每种模式的存储完全独立，切换到的模式若从未写过即为空白记忆。',
        {
          type: 'object',
          properties: {
            id: { type: 'string', description: '可选：要切换到的模式 id（chat/kb/minimal/semantic）' },
          },
        },
        async (args) => {
          const ok = await ensureBridge()
          if (!ok) return { error: state.lastError || 'bridge unavailable' }
          if (args && typeof args.id === 'string') {
            const r = await pwshJson(jsonCmd('POST', '/mode', { id: args.id }))
            return r || { error: 'no response' }
          }
          const m = await pwshJson(jsonCmd('GET', '/modes'))
          return m || { modes: [] }
        },
      )
      tools.register(modeTool)

      const statusTool = makeJsonTool(
        'sme3d_status',
        '查看 3D 空间记忆引擎运行状态：当前模式、记忆条数、Region 数、自动写入统计。',
        { type: 'object', properties: {} },
        async () => {
          const h = await health()
          const alive = !!(h || await ensureBridge())
          const s = alive ? await pwshJson(jsonCmd('GET', '/stats')) : null
          return {
            alive,
            mode: s && s.mode ? s.mode : null,
            memories: s && s.memory ? s.memory.total : 0,
            regions: s && s.region ? s.region.count : 0,
            provider: s && s.provider ? s.provider : null,
            seq: s && typeof s.seq === 'number' ? s.seq : 0,
            autoWrites: state.stats.writes,
            errors: state.stats.errors,
            lastError: state.lastError,
          }
        },
      )
      tools.register(statusTool)
    }
    } catch (e) {
      try { fs.writeFileSync(BOOT_MARK, JSON.stringify({ ok: false, at: new Date().toISOString(), error: String(e && e.message || e) })) } catch (_) {}
    }
  },
}
