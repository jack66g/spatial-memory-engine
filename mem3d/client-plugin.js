// ============================================================================
// mem3d — DeepSeek Harness 持久化插件：Client 半区
// ============================================================================
// 职责：
//   1. 侧栏底部 "3D 记忆" 按钮（sidebar.footer.action）
//   2. 全局浮动 3D 小窗（shell.overlay）：手写 canvas 3D 渲染（零依赖），
//      轮询 Host 的 mem3d.scene 实时显示记忆轨迹；支持拖拽旋转、滚轮缩放、
//      双击复位、悬停查看记忆文本、模式切换（切换提示"新模式为空白记忆"）。
//
// 与动态插件 memviz-1/pkg-6 的 client 半区逻辑一致；不修改 sme 任何源码。
// ============================================================================
module.exports = {
  inject: ['timer'],
  apply(ctx) {
    const slots = ctx.get('slots')
    if (slots === undefined) return

    const store = { open: false, listeners: new Set() }
    const subscribe = (fn) => { store.listeners.add(fn); return () => store.listeners.delete(fn) }
    const emit = () => store.listeners.forEach((fn) => { try { fn() } catch (e) {} })
    const setOpen = (v) => { store.open = v; emit() }

    let sceneRef = null
    let canvasEl = null
    let winEl = null
    const view = { rx: -0.55, ry: 0.45, zoom: 1, auto: true, drag: false, px: 0, py: 0, hover: null }

    styles.insert(`
      .mem3d-win {
        position: fixed; z-index: 4000; pointer-events: auto;
        width: 640px; max-width: calc(100vw - 32px);
        background: rgba(22, 24, 30, 0.94); color: #e8eaf0;
        border: 1px solid rgba(124, 156, 255, 0.35);
        border-radius: 14px; box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
        font: 13px/1.5 system-ui, 'Segoe UI', sans-serif;
        overflow: hidden;
      }
      .mem3d-head {
        display: flex; align-items: center; gap: 8px;
        padding: 8px 12px; cursor: move; user-select: none;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      }
      .mem3d-title { font-weight: 600; font-size: 13px; }
      .mem3d-badge { color: #9aa4b8; font-size: 11px; }
      .mem3d-select {
        margin-left: auto; background: rgba(255, 255, 255, 0.08); color: #e8eaf0;
        border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 6px; padding: 2px 6px;
        font-size: 11px; max-width: 130px;
      }
      .mem3d-close {
        background: none; border: none; color: #9aa4b8; cursor: pointer;
        font-size: 15px; line-height: 1; padding: 2px 4px; border-radius: 6px;
      }
      .mem3d-close:hover { color: #fff; background: rgba(255, 255, 255, 0.12); }
      .mem3d-canvas {
        display: block; width: 100%; height: 340px; cursor: grab;
        background: radial-gradient(ellipse at 50% 45%, #1d2230 0%, #12141a 70%, #0d0f14 100%);
      }
      .mem3d-canvas.dragging { cursor: grabbing; }
      .mem3d-foot {
        display: flex; align-items: center; gap: 10px; padding: 6px 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.08); color: #9aa4b8; font-size: 11px;
        flex-wrap: wrap;
      }
      .mem3d-notice {
        margin: 0 12px 8px; padding: 6px 10px; border-radius: 8px;
        background: rgba(255, 193, 7, 0.16); color: #ffd54f; font-size: 12px;
      }
    `)

    // ---------------- 3D 渲染（手写透视投影，零依赖） ----------------
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
      for (const n of scene.nodes) {
        const p = project(n, w, h)
        pts[n.id] = { X: p.X, Y: p.Y, Z: p.Z, n }
      }
      const cpts = {}
      for (const c of scene.centroids || []) {
        cpts[c.id] = project(c, w, h)
      }
      // Region 邻居边（质心连线）
      g.lineWidth = 1
      for (const e of scene.edges || []) {
        const a = cpts[e.a], b = cpts[e.b]
        if (!a || !b) continue
        g.strokeStyle = 'rgba(160, 170, 195, 0.28)'
        g.beginPath()
        g.moveTo(a.X, a.Y)
        g.lineTo(b.X, b.Y)
        g.stroke()
      }
      // 记忆节点（按深度排序，新写入高亮）
      const keys = Object.keys(pts)
      keys.sort((a, b) => pts[a].Z - pts[b].Z)
      const freshIds = new Set(scene.recent || [])
      for (const id of keys) {
        const p = pts[id], n = p.n
        const r = freshIds.has(id) ? 7.5 : 4.5
        g.globalAlpha = 0.55 + 0.45 * Math.max(0, 1 - p.Z / 6)
        if (freshIds.has(id)) {
          g.fillStyle = 'rgba(255,255,255,0.9)'
          g.beginPath()
          g.arc(p.X, p.Y, r + 3.5, 0, Math.PI * 2)
          g.fill()
        }
        g.fillStyle = n.c || '#7c9cff'
        g.beginPath()
        g.arc(p.X, p.Y, r, 0, Math.PI * 2)
        g.fill()
        g.globalAlpha = 1
      }
      // Region 质心
      for (const id of Object.keys(cpts)) {
        const p = cpts[id]
        g.fillStyle = '#ffffff'
        g.font = '13px system-ui'
        g.textAlign = 'center'
        g.fillText('✦', p.X, p.Y + 4)
      }
      // 悬停提示
      if (view.hover) {
        const hp = view.hover
        g.fillStyle = 'rgba(10, 12, 16, 0.92)'
        const label = hp.label.length > 42 ? hp.label.slice(0, 42) + '…' : hp.label
        g.font = '12px system-ui'
        const tw = g.measureText(label).width + 16
        g.fillRect(hp.X + 12, hp.Y - 24, tw, 20)
        g.fillStyle = '#e8eaf0'
        g.textAlign = 'left'
        g.fillText(label, hp.X + 20, hp.Y - 10)
      }
    }

    // ---------------- 组件 ----------------
    function ToggleButton(props) {
      const [, force] = React.useState(0)
      React.useEffect(() => subscribe(() => force((x) => x + 1)), [])
      const open = store.open
      const icon = React.createElement('span', { style: { fontSize: 14, lineHeight: 1 } }, '🧠')
      const label = props.wide
        ? React.createElement('span', { style: { marginLeft: 7, fontSize: 12.5 } }, '3D 记忆')
        : null
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
            const s = await host.call('mem3d.scene', {})
            if (alive && s && s.nodes) { sceneRef = s; setScene(s) }
            const st = await host.call('mem3d.status', {})
            if (alive && st) setStatus(st)
            const ms = await host.call('mem3d.modes', {})
            if (alive && ms && ms.modes) setModes(ms)
          } catch (e) {}
        }
        tick()
        const stop = ctx.interval(tick, 2500)
        return () => { alive = false; stop() }
      }, [open])

      React.useEffect(() => {
        const stop = ctx.interval(() => draw3d(canvasEl, sceneRef), 40)
        return () => stop()
      }, [])

      React.useEffect(() => {
        if (!notice) return
        const stop = ctx.timeout(() => setNotice(null), 9000)
        return () => stop()
      }, [notice])

      if (!open) return null

      const switchMode = async (id) => {
        try {
          const r = await host.call('mem3d.switchMode', { id })
          if (r && r.notice) setNotice({ kind: 'ok', text: r.notice })
          else if (r && r.error) setNotice({ kind: 'err', text: String(r.error) })
          sceneRef = null
          setScene(null)
          const s = await host.call('mem3d.scene', {})
          if (s && s.nodes) { sceneRef = s; setScene(s) }
        } catch (e) {
          setNotice({ kind: 'err', text: String((e && e.message) || e) })
        }
      }

      const modeOptions = (modes && modes.modes ? modes.modes : []).map((m) =>
        React.createElement('option', { key: m.id, value: m.id },
          m.name + ' (' + (m.memories === null ? '未加载' : m.memories) + ' 条)' + (m.current ? ' ●' : ''),
        ),
      )

      const winStyle = pos
        ? { position: 'fixed', left: pos.x, top: pos.y }
        : { position: 'fixed', right: 18, bottom: 18 }

      const headDown = (e) => {
        dragState.active = true
        dragState.ox = e.clientX
        dragState.oy = e.clientY
        if (winEl) {
          const r = winEl.getBoundingClientRect()
          dragState.base = pos || { x: r.left, y: r.top }
        } else {
          dragState.base = pos || { x: 0, y: 0 }
        }
      }
      const headMove = (e) => {
        if (!dragState.active) return
        const nx = Math.max(8, Math.min(dragState.base.x + (e.clientX - dragState.ox), (typeof window !== 'undefined' ? window.innerWidth : 1400) - 120))
        const ny = Math.max(8, dragState.base.y + (e.clientY - dragState.oy))
        setPos({ x: nx, y: ny })
      }
      const headUp = () => { dragState.active = false }

      const canvasDown = (e) => {
        view.drag = true
        view.auto = false
        view.px = e.clientX
        view.py = e.clientY
      }
      const canvasMove = (e) => {
        if (view.drag) {
          view.ry += (e.clientX - view.px) * 0.008
          view.rx += (e.clientY - view.py) * 0.008
          view.px = e.clientX
          view.py = e.clientY
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
      const canvasWheel = (e) => {
        view.zoom = Math.max(0.35, Math.min(4, view.zoom * (e.deltaY > 0 ? 0.92 : 1.08)))
      }
      const canvasDbl = () => {
        view.auto = true
        view.rx = -0.55
        view.ry = 0.45
        view.zoom = 1
      }

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
            status ? ('自动写入 ' + status.writes + ' 条' + (status.errors ? ' · 失败 ' + status.errors : '')) : '连接中…',
          ),
        ),
      )
    }

    slots.inject('sidebar.footer.action', () => slots.register(
      { name: 'sidebar.footer.action', id: 'mem3d', order: 10, label: '3D 记忆' },
      (props) => React.createElement(ToggleButton, { wide: props.wide }),
    ))

    slots.inject('shell.overlay', () => slots.register(
      { name: 'shell.overlay', id: 'mem3d-window', order: 5, label: '3D 记忆空间' },
      () => React.createElement(Window3D),
    ))
  },
}
