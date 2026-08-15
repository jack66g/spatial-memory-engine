// ============================================================================
// mem3d client bundle —— DeepSeek Harness 浏览器持久化半区（手工构建）
// ============================================================================
// 由 dsh-client-modules 扫描本包（package.json 的 dsh.client + exports["./client"]）
// 后注入 window.__DSH_BOOT__ 图，页面启动即自动加载——DSH 重启后按钮/浮窗/
// 设置分区自动出现，无需重新激活。
//
// 注意：注册 id 必须与 boot 图 entry id 一致（即 entry 名 mem3d-plugin，
// 也就是本包 package.json 的 name）。写路径/其他字符串会在浏览器端报
// "bundle ... loaded without registering \"mem3d-plugin\" via
// __ModuleLoader__.load"（client.js 的 arrive() 按 row.id 校验）。
//
// 与动态插件版（client-plugin.js，走 host.call RPC）不同，本 bundle 运行在
// 完整浏览器环境：React 来自 __ModuleLoader__ 的静态模块表（require("react")），
// 数据直连 mem3d bridge（http://127.0.0.1:8756，CORS 已开），计时器用原生
// window.setInterval/setTimeout，样式用原生 DOM 注入。
// ============================================================================
window.__ModuleLoader__.load({
  id: 'mem3d-plugin',
  factory: (require) => {
    const React = require('react')
    var module = { exports: {} }
    var exports = module.exports

    // ---- bridge 直连 ----
    function fetchBridge(path, method, body) {
      const url = 'http://127.0.0.1:8756' + path
      const opts = { method: method || 'GET' }
      if (body !== undefined) {
        opts.headers = { 'Content-Type': 'application/json' }
        opts.body = JSON.stringify(body)
      }
      return fetch(url, opts)
        .then((r) => r.json())
        .catch((e) => ({ error: String((e && e.message) || e) }))
    }

    function apiStatus() {
      return fetchBridge('/stats', 'GET').then((s) => ({
        alive: !!(s && s.mode),
        mode: s && s.mode ? s.mode : null,
        memories: s && s.memory ? s.memory.total : 0,
        regions: s && s.region ? s.region.count : 0,
        provider: s && s.provider ? s.provider : null,
        seq: s && typeof s.seq === 'number' ? s.seq : 0,
        writes: 0,
        errors: 0,
        lastError: s && s.last_error ? s.last_error : '',
      }))
    }

    function apiPages() {
      return fetchBridge('/auto', 'GET').then((r) => ({ pages: [], auto: !!(r && r.auto) }))
    }

    function apiModes() {
      return fetchBridge('/modes', 'GET')
    }

    function apiSwitchMode(id) {
      return fetchBridge('/mode', 'POST', { id })
    }

    function apiScene() {
      return fetchBridge('/scene', 'GET')
    }

    function apiSetAuto(on) {
      return fetchBridge('/auto', 'POST', { on: !!on })
    }

    function apiClearMode() {
      return fetchBridge('/clear', 'POST', {})
    }

    const interval = (fn, ms) => {
      if (typeof window !== 'undefined' && window.setInterval) {
        const id = window.setInterval(fn, ms)
        return () => window.clearInterval(id)
      }
      return () => {}
    }
    const timeout = (fn, ms) => {
      if (typeof window !== 'undefined' && window.setTimeout) {
        const id = window.setTimeout(fn, ms)
        return () => window.clearTimeout(id)
      }
      return () => {}
    }

    module.exports = {
      apply(ctx) {
        const slots = ctx.get('slots')
        if (slots === undefined) return

        // 样式注入（幂等：先清理旧实例）
        try {
          if (typeof document !== 'undefined' && document.head) {
            const old = document.querySelectorAll('style[data-mem3d-bundle]')
            for (let i = 0; i < old.length; i++) old[i].remove()
            const style = document.createElement('style')
            style.setAttribute('data-mem3d-bundle', '')
            style.textContent = [
              '.mem3d-win { position: fixed; z-index: 4000; pointer-events: auto; width: 640px; max-width: calc(100vw - 32px); background: rgba(22,24,30,0.94); color: #e8eaf0; border: 1px solid rgba(124,156,255,0.35); border-radius: 14px; box-shadow: 0 12px 40px rgba(0,0,0,0.45); font: 13px/1.5 system-ui, "Segoe UI", sans-serif; overflow: hidden; }',
              '.mem3d-head { display: flex; align-items: center; gap: 8px; padding: 8px 12px; cursor: move; user-select: none; border-bottom: 1px solid rgba(255,255,255,0.08); }',
              '.mem3d-title { font-weight: 600; font-size: 13px; }',
              '.mem3d-badge { color: #9aa4b8; font-size: 11px; }',
              '.mem3d-select { margin-left: auto; background: rgba(255,255,255,0.08); color: #e8eaf0; border: 1px solid rgba(255,255,255,0.15); border-radius: 6px; padding: 2px 6px; font-size: 11px; max-width: 130px; }',
              '.mem3d-close { background: none; border: none; color: #9aa4b8; cursor: pointer; font-size: 15px; line-height: 1; padding: 2px 4px; border-radius: 6px; }',
              '.mem3d-close:hover { color: #fff; background: rgba(255,255,255,0.12); }',
              '.mem3d-canvas { display: block; width: 100%; height: 340px; cursor: grab; background: radial-gradient(ellipse at 50% 45%, #1d2230 0%, #12141a 70%, #0d0f14 100%); }',
              '.mem3d-canvas.dragging { cursor: grabbing; }',
              '.mem3d-foot { display: flex; align-items: center; gap: 10px; padding: 6px 12px; border-top: 1px solid rgba(255,255,255,0.08); color: #9aa4b8; font-size: 11px; flex-wrap: wrap; }',
              '.mem3d-notice { margin: 0 12px 8px; padding: 6px 10px; border-radius: 8px; background: rgba(255,193,7,0.16); color: #ffd54f; font-size: 12px; }',
              '.mem3d-ball { position: fixed; right: 20px; bottom: 20px; z-index: 4001; width: 54px; height: 54px; border-radius: 50%; background: linear-gradient(135deg, #7c9cff, #4f6ef7); color: #fff; font-size: 24px; line-height: 1; cursor: pointer; border: 2px solid rgba(255,255,255,0.55); box-shadow: 0 6px 22px rgba(0,0,0,0.45); display: flex; align-items: center; justify-content: center; pointer-events: auto; padding: 0; }',
              '.mem3d-ball:hover { transform: scale(1.08); }',
              '.mem3d-set { padding: 14px; display: flex; flex-direction: column; gap: 12px; color: var(--dsw-alias-label-primary); font: 13px/1.5 system-ui, "Segoe UI", sans-serif; }',
              '.mem3d-set h3 { margin: 0; font-size: 15px; }',
              '.mem3d-mode-card { display: flex; align-items: flex-start; gap: 8px; border: 1px solid var(--dsw-alias-border-l1); border-radius: 10px; padding: 10px 12px; cursor: pointer; background: var(--dsw-alias-bg-layer-1); }',
              '.mem3d-mode-card:hover { border-color: var(--dsw-alias-brand-primary); }',
              '.mem3d-mode-card.current { border-color: var(--dsw-alias-brand-primary); border-width: 2px; background: color-mix(in srgb, var(--dsw-alias-brand-primary) 10%, transparent); }',
              '.mem3d-mode-name { font-weight: 600; color: var(--dsw-alias-label-primary); }',
              '.mem3d-mode-desc { color: var(--dsw-alias-label-secondary); font-size: 12px; margin-top: 2px; }',
              '.mem3d-mode-count { margin-left: auto; font-size: 12px; color: var(--dsw-alias-label-secondary); white-space: nowrap; }',
              '.mem3d-set-notice { padding: 8px 12px; border-radius: 8px; color: var(--dsw-alias-state-warn-primary); background: color-mix(in srgb, var(--dsw-alias-state-warn-primary) 14%, transparent); font-size: 12px; }',
              '.mem3d-row { display: flex; align-items: center; gap: 10px; }',
              '.mem3d-danger { border: 1px solid var(--dsw-alias-state-error-primary); background: color-mix(in srgb, var(--dsw-alias-state-error-primary) 10%, transparent); color: var(--dsw-alias-state-error-primary); border-radius: 8px; padding: 6px 12px; cursor: pointer; font: inherit; }',
              '.mem3d-danger:hover { background: color-mix(in srgb, var(--dsw-alias-state-error-primary) 20%, transparent); }',
            ].join('\n')
            document.head.appendChild(style)
          }
        } catch (e) { /* 样式注入失败不影响功能 */ }

        const store = { open: false, listeners: new Set() }
        const subscribe = (fn) => { store.listeners.add(fn); return () => store.listeners.delete(fn) }
        const emit = () => store.listeners.forEach((fn) => { try { fn() } catch (e) {} })
        const setOpen = (v) => { store.open = v; emit() }

        let sceneRef = null
        let canvasEl = null
        let winEl = null
        const view = { rx: -0.55, ry: 0.45, zoom: 1, auto: true, drag: false, px: 0, py: 0, hover: null }

        function project(p, w, h) {
          let x = p.x, y = p.y, z = p.z
          const cx = Math.cos(view.ry), sx = Math.sin(view.ry)
          const x1 = x * cx - z * sx
          const z1 = x * sx + z * cx
          const cy = Math.cos(view.rx), sy = Math.sin(view.rx)
          const y1 = y * cy - z1 * sy
          const z2 = y * sy + z1 * cy
          const f = 320
          const s = (f / (f + z2 * 2)) * view.zoom * (Math.min(w, h) / 3.2)
          return { X: w / 2 + x1 * s, Y: h / 2 - y1 * s, Z: z2, S: s }
        }

        function draw3d(canvas, scene) {
          if (!canvas) return
          const g = canvas.getContext('2d')
          if (!g) return
          const w = canvas.width, h = canvas.height
          if (view.auto) view.ry += 0.0045
          g.clearRect(0, 0, w, h)
          if (!scene || !scene.nodes || scene.nodes.length === 0) {
            g.fillStyle = '#8b93a7'
            g.font = '13px system-ui'
            g.textAlign = 'center'
            g.fillText(scene && scene.error ? ('服务不可用: ' + scene.error) : '空白记忆：开始对话或写入记忆，空间里的点会实时长出来', w / 2, h / 2)
            return
          }
          const pts = {}
          for (const n of scene.nodes) { const p = project(n, w, h); pts[n.id] = { X: p.X, Y: p.Y, Z: p.Z, n } }
          const cpts = {}
          for (const c of scene.centroids || []) { cpts[c.id] = project(c, w, h) }
          g.lineWidth = 1
          for (const e of scene.edges || []) {
            const a = cpts[e.a], b = cpts[e.b]
            if (!a || !b) continue
            g.strokeStyle = 'rgba(160,170,195,0.28)'
            g.beginPath(); g.moveTo(a.X, a.Y); g.lineTo(b.X, b.Y); g.stroke()
          }
          const keys = Object.keys(pts)
          keys.sort((a, b) => pts[a].Z - pts[b].Z)
          const freshIds = new Set(scene.recent || [])
          for (const id of keys) {
            const p = pts[id], n = p.n
            const r = freshIds.has(id) ? 7.5 : 4.5
            g.globalAlpha = 0.55 + 0.45 * Math.max(0, 1 - p.Z / 6)
            if (freshIds.has(id)) {
              g.fillStyle = 'rgba(255,255,255,0.9)'
              g.beginPath(); g.arc(p.X, p.Y, r + 3.5, 0, Math.PI * 2); g.fill()
            }
            g.fillStyle = n.c || '#7c9cff'
            g.beginPath(); g.arc(p.X, p.Y, r, 0, Math.PI * 2); g.fill()
            g.globalAlpha = 1
          }
          for (const id of Object.keys(cpts)) {
            const p = cpts[id]
            g.fillStyle = '#ffffff'
            g.font = '13px system-ui'
            g.textAlign = 'center'
            g.fillText('✦', p.X, p.Y + 4)
          }
          if (view.hover) {
            const hp = view.hover
            g.fillStyle = 'rgba(10,12,16,0.92)'
            const label = hp.label.length > 42 ? hp.label.slice(0, 42) + '…' : hp.label
            g.font = '12px system-ui'
            const tw = g.measureText(label).width + 16
            g.fillRect(hp.X + 12, hp.Y - 24, tw, 20)
            g.fillStyle = '#e8eaf0'
            g.textAlign = 'left'
            g.fillText(label, hp.X + 20, hp.Y - 10)
          }
        }

        function FloatingBall() {
          const [, force] = React.useState(0)
          React.useEffect(() => subscribe(() => force((x) => x + 1)), [])
          if (store.open) return null
          return React.createElement('button', {
            type: 'button',
            className: 'mem3d-ball',
            title: '打开 3D 记忆空间',
            onClick: () => setOpen(true),
          }, '🧠')
        }

        function ToggleButton(props) {
          const [, force] = React.useState(0)
          React.useEffect(() => subscribe(() => force((x) => x + 1)), [])
          const open = store.open
          const icon = React.createElement('span', { style: { fontSize: 14, lineHeight: 1 } }, '🧠')
          const label = props.wide ? React.createElement('span', { style: { marginLeft: 7, fontSize: 12.5 } }, '3D 记忆') : null
          return React.createElement('button', {
            type: 'button',
            title: '3D 记忆空间：实时查看记忆轨迹（点击打开/关闭浮窗）',
            onClick: () => setOpen(!open),
            style: {
              display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
              cursor: 'pointer', borderRadius: 8, padding: props.wide ? '6px 10px' : '5px 7px',
              border: open ? '1px solid rgba(124,156,255,0.6)' : '1px solid transparent',
              background: open ? 'rgba(124,156,255,0.16)' : 'transparent',
              color: 'inherit', font: 'inherit',
            },
          }, icon, label)
        }

        function SettingsPanel() {
          const [modes, setModes] = React.useState(null)
          const [status, setStatus] = React.useState(null)
          const [auto, setAuto] = React.useState(null)
          const [notice, setNotice] = React.useState(null)
          const [confirmClear, setConfirmClear] = React.useState(false)
          React.useEffect(() => {
            let alive = true
            const tick = async () => {
              try {
                const ms = await apiModes()
                if (alive && ms && ms.modes) setModes(ms)
                const st = await apiStatus()
                if (alive && st) setStatus(st)
                const pg = await apiPages()
                if (alive && pg && typeof pg.auto === 'boolean') setAuto(pg.auto)
              } catch (e) {}
            }
            tick()
            const stop = interval(tick, 5000)
            return () => { alive = false; stop() }
          }, [])
          React.useEffect(() => {
            if (!notice) return
            const stop = timeout(() => setNotice(null), 10000)
            return () => stop()
          }, [notice])
          const switchMode = async (id) => {
            try {
              const r = await apiSwitchMode(id)
              if (r && r.notice) setNotice({ kind: 'ok', text: r.notice })
              else if (r && r.error) setNotice({ kind: 'err', text: String(r.error) })
              const ms = await apiModes()
              if (ms && ms.modes) setModes(ms)
              const st = await apiStatus()
              if (st) setStatus(st)
            } catch (e) { setNotice({ kind: 'err', text: String((e && e.message) || e) }) }
          }
          const toggleAuto = async () => {
            const r = await apiSetAuto(!auto)
            if (r && typeof r.auto === 'boolean') setAuto(r.auto)
          }
          const doClear = async () => {
            if (!confirmClear) { setConfirmClear(true); return }
            setConfirmClear(false)
            const r = await apiClearMode()
            if (r && r.ok) {
              setNotice({ kind: 'ok', text: '已清空当前模式（' + r.cleared + ' 条记忆）。' })
              const ms = await apiModes()
              if (ms && ms.modes) setModes(ms)
              const st = await apiStatus()
              if (st) setStatus(st)
            } else if (r && r.error) {
              setNotice({ kind: 'err', text: String(r.error) })
            }
          }
          const list = modes && modes.modes ? modes.modes : []
          const cards = list.map((m) =>
            React.createElement('div', {
              key: m.id,
              className: 'mem3d-mode-card' + (m.current ? ' current' : ''),
              onClick: () => switchMode(m.id),
            },
              React.createElement('div', null,
                React.createElement('div', { className: 'mem3d-mode-name' },
                  (m.current ? '● ' : '○ ') + m.name,
                ),
                React.createElement('div', { className: 'mem3d-mode-desc' }, m.desc),
              ),
              React.createElement('div', { className: 'mem3d-mode-count' },
                (m.memories === null ? '未加载' : m.memories + ' 条'),
              ),
            ),
          )
          const statText = status
            ? (status.mode + ' · ' + status.memories + ' 条记忆 · ' + status.regions + ' 个区域 · ' + (status.provider || ''))
            : '连接中…'
          return React.createElement('div', { className: 'mem3d-set' },
            React.createElement('h3', null, '🧠 3D 记忆设置'),
            React.createElement('div', { style: { fontSize: 12, color: 'var(--dsw-alias-label-secondary)' } }, statText),
            notice ? React.createElement('div', { className: 'mem3d-set-notice' }, String(notice.text)) : null,
            React.createElement('div', { style: { fontWeight: 600, fontSize: 13, color: 'var(--dsw-alias-label-primary)' } }, '记忆模式（共 ' + list.length + ' 种，切换后不同模式不共享记忆，新模式若从未写过即为空白）'),
            React.createElement('div', { style: { display: 'flex', flexDirection: 'column', gap: 8 } }, cards),
            React.createElement('div', { className: 'mem3d-row' },
              React.createElement('input', {
                type: 'checkbox',
                checked: auto !== false,
                onChange: toggleAuto,
                style: { cursor: 'pointer' },
              }),
              React.createElement('span', null, '自动记录对话（关闭后聊天不再自动写入记忆）'),
            ),
            React.createElement('div', { className: 'mem3d-row' },
              React.createElement('button', {
                type: 'button',
                className: 'mem3d-danger',
                onClick: doClear,
              }, confirmClear ? '⚠ 再次点击确认清空当前模式全部记忆' : '🗑 清空当前模式'),
            ),
          )
        }

        function Window3D() {
          const [open, setOpenState] = React.useState(store.open)
          const [scene, setScene] = React.useState(null)
          const [status, setStatus] = React.useState(null)
          const [modes, setModes] = React.useState(null)
          const [notice, setNotice] = React.useState(null)
          const [pos, setPos] = React.useState(null)
          const dragState = { active: false, ox: 0, oy: 0, base: null }
          React.useEffect(() => subscribe(() => setOpenState(store.open)), [])
          React.useEffect(() => {
            if (!open) return
            let alive = true
            const tick = async () => {
              try {
                const s = await apiScene()
                if (alive && s && s.nodes) { sceneRef = s; setScene(s) }
                const st = await apiStatus()
                if (alive && st) setStatus(st)
                const ms = await apiModes()
                if (alive && ms && ms.modes) setModes(ms)
              } catch (e) {}
            }
            tick()
            const stop = interval(tick, 2500)
            return () => { alive = false; stop() }
          }, [open])
          React.useEffect(() => {
            const stop = interval(() => draw3d(canvasEl, sceneRef), 40)
            return () => stop()
          }, [])
          React.useEffect(() => {
            if (!notice) return
            const stop = timeout(() => setNotice(null), 9000)
            return () => stop()
          }, [notice])
          if (!open) return null
          const switchMode = async (id) => {
            try {
              const r = await apiSwitchMode(id)
              if (r && r.notice) setNotice({ kind: 'ok', text: r.notice })
              else if (r && r.error) setNotice({ kind: 'err', text: String(r.error) })
              sceneRef = null
              setScene(null)
              const s = await apiScene()
              if (s && s.nodes) { sceneRef = s; setScene(s) }
            } catch (e) { setNotice({ kind: 'err', text: String((e && e.message) || e) }) }
          }
          const modeOptions = (modes && modes.modes ? modes.modes : []).map((m) =>
            React.createElement('option', { key: m.id, value: m.id },
              m.name + ' (' + (m.memories === null ? '未加载' : m.memories) + ' 条)' + (m.current ? ' ●' : ''),
            ),
          )
          const winStyle = pos ? { position: 'fixed', left: pos.x, top: pos.y } : { position: 'fixed', right: 18, bottom: 18 }
          const headDown = (e) => {
            dragState.active = true
            dragState.ox = e.clientX
            dragState.oy = e.clientY
            if (winEl) {
              const r = winEl.getBoundingClientRect()
              dragState.base = pos || { x: r.left, y: r.top }
            } else { dragState.base = pos || { x: 0, y: 0 } }
          }
          const headMove = (e) => {
            if (!dragState.active) return
            const nx = Math.max(8, Math.min(dragState.base.x + (e.clientX - dragState.ox), (typeof window !== 'undefined' ? window.innerWidth : 1400) - 120))
            const ny = Math.max(8, dragState.base.y + (e.clientY - dragState.oy))
            setPos({ x: nx, y: ny })
          }
          const headUp = () => { dragState.active = false }
          const canvasDown = (e) => { view.drag = true; view.auto = false; view.px = e.clientX; view.py = e.clientY }
          const canvasMove = (e) => {
            if (view.drag) {
              view.ry += (e.clientX - view.px) * 0.008
              view.rx += (e.clientY - view.py) * 0.008
              view.px = e.clientX; view.py = e.clientY
              view.hover = null
              return
            }
            const c = canvasEl
            if (!c || !sceneRef || !sceneRef.nodes) return
            const r = c.getBoundingClientRect()
            const mx = e.clientX - r.left, my = e.clientY - r.top
            let best = null, bd = 14
            for (const n of sceneRef.nodes) {
              const p = project(n, c.width, c.height)
              const d = Math.hypot(p.X - mx, p.Y - my)
              if (d < bd) { bd = d; best = { X: p.X, Y: p.Y, label: n.label } }
            }
            view.hover = best
          }
          const canvasUp = () => { view.drag = false }
          const canvasWheel = (e) => { view.zoom = Math.max(0.35, Math.min(4, view.zoom * (e.deltaY > 0 ? 0.92 : 1.08))) }
          const canvasDbl = () => { view.auto = true; view.rx = -0.55; view.ry = 0.45; view.zoom = 1 }
          const memText = status ? status.memories : (scene ? scene.memories : '…')
          const regionText = status ? status.regions : (scene ? scene.regions : '…')
          return React.createElement('div', {
            ref: (el) => { winEl = el },
            className: 'mem3d-win',
            style: winStyle,
            onMouseMove: headMove,
            onMouseUp: headUp,
          },
            React.createElement('div', { className: 'mem3d-head', onMouseDown: headDown },
              React.createElement('span', { className: 'mem3d-title' }, '🧠 3D 记忆空间'),
              React.createElement('span', { className: 'mem3d-badge' },
                (status && status.mode ? status.mode : '…') + ' · ' + memText + ' 条 · ' + regionText + ' 区' + (status && status.provider ? ' · ' + status.provider : ''),
              ),
              React.createElement('select', {
                className: 'mem3d-select',
                value: (status && status.mode) || '',
                onChange: (e) => switchMode(e.target.value),
                onMouseDown: (e) => e.stopPropagation(),
                title: '切换记忆模式（不同模式存储完全独立，切换到的模式若从未写过即为空白）',
              }, modeOptions),
              React.createElement('button', { type: 'button', className: 'mem3d-close', title: '关闭', onClick: () => setOpen(false) }, '✕'),
            ),
            notice ? React.createElement('div', { className: 'mem3d-notice' }, String(notice.text)) : null,
            React.createElement('canvas', {
              ref: (el) => { canvasEl = el },
              className: 'mem3d-canvas' + (view.drag ? ' dragging' : ''),
              width: 640,
              height: 340,
              onMouseDown: canvasDown,
              onMouseMove: canvasMove,
              onMouseUp: canvasUp,
              onMouseLeave: () => { view.hover = null },
              onWheel: canvasWheel,
              onDoubleClick: canvasDbl,
            }),
            React.createElement('div', { className: 'mem3d-foot' },
              React.createElement('span', null, '拖拽旋转 · 滚轮缩放 · 双击复位' + (view.auto ? ' · 自动旋转中' : '')),
              React.createElement('span', { style: { marginLeft: 'auto' } },
                status ? (status.mode + ' · ' + status.memories + ' 条') : '连接中…',
              ),
            ),
          )
        }

        slots.inject('sidebar.footer.action', () => slots.register(
          { name: 'sidebar.footer.action', id: 'mem3d', order: 10, label: '3D 记忆' },
          (props) => React.createElement(ToggleButton, { wide: props.wide }),
        ))
        slots.inject('shell.overlay', () => slots.register(
          { name: 'shell.overlay', id: 'mem3d-ball', order: 0, label: '3D 记忆入口' },
          () => React.createElement(FloatingBall),
        ))
        slots.inject('shell.overlay', () => slots.register(
          { name: 'shell.overlay', id: 'mem3d-window', order: 5, label: '3D 记忆空间' },
          () => React.createElement(Window3D),
        ))
        slots.inject('settings.section', () => slots.register(
          { name: 'settings.section', id: 'mem3d-memory', order: 25, label: '3D 记忆' },
          () => React.createElement(SettingsPanel),
        ))
      },
    }

    return module.exports
  },
})
