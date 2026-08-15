// mem3d 包的 Host 面（web profile 进程内加载）：无操作占位。
// 真实 Host 能力（记忆工具/自动写入/边车管理）由 agent preset 加载的
// host-preset-live.js 提供；本文件只让 dsh-client-modules 能扫描到
// 本包的 dsh.client 声明并把 client-bundle.js 注入浏览器 boot 图。
// 注意：loader 会把本文件当作 cordis 插件挂载，必须导出合法插件
// （函数或带 apply() 的对象），否则挂载报 "invalid plugin"。
module.exports = { apply() {} }
