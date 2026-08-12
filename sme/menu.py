"""SME 配置菜单（工程层工具，不含任何引擎逻辑）。

用法::

    python -m sme.menu                       # 交互菜单
    python -m sme.menu --list                # 列出全部可配项（含说明）
    python -m sme.menu --show                # 查看当前配置文件内容
    python -m sme.menu --check [--ping]      # 验证配置与 LLM/embedding 连通性
    python -m sme.menu --set llm.model=deepseek-v4-flash   # 非交互设置（可多次）
    python -m sme.menu --preset kb_dynamic   # 非交互套用预设
    python -m sme.menu --config path.json    # 指定配置文件（默认 sme/config.json）

规则：
- 菜单只管理"可调项"（注册表 sme/config_items.py），引擎默认值保持代码内置；
- 每一次修改（手动/预设/--set）都会立即原子写回配置文件；
- 输入会做类型/枚举/范围/非空校验，非法输入当场报错并提示正确格式；
- 配置文件缺失/为空/损坏时自动用默认值重建（损坏文件先备份为 .bak）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows GBK 控制台
except Exception:
    pass

from sme.config_items import (
    DEFAULT_CONFIG_PATH,
    GROUPS,
    ITEM_BY_PATH,
    PRESET_BY_KEY,
    PRESETS,
    apply_preset,
    current_preset_key,
    defaults_config,
    get_value,
    parse_value,
    preset_name,
    render_value,
    save_config,
    set_value,
    type_hint,
    validate_config,
)


# --------------------------------------------------------------------------- #
# 启动加载（缺失/损坏重建）
# --------------------------------------------------------------------------- #
def _ensure_config(path: str) -> tuple[dict, list[str]]:
    """Load the config file; create/rebuild it when missing or corrupt.

    Returns (cfg, notes) where notes lists what was rebuilt.
    """
    notes: list[str] = []
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            raw = fh.read()
    except OSError:
        raw = ""

    if not raw.strip():
        cfg = defaults_config()
        save_config(path, cfg)
        notes.append(f"配置文件不存在或为空，已创建默认配置：{path}")
        return cfg, notes

    try:
        cfg = json.loads(raw)
        if not isinstance(cfg, dict):
            raise ValueError("配置文件根节点不是 JSON 对象")
    except (json.JSONDecodeError, ValueError) as exc:
        backup = path + ".bak"
        try:
            os.replace(path, backup)
        except OSError:
            pass
        cfg = defaults_config()
        save_config(path, cfg)
        notes.append(f"配置文件损坏（{exc}），已备份为 {backup} 并重建默认配置")
    return cfg, notes


# --------------------------------------------------------------------------- #
# 打印
# --------------------------------------------------------------------------- #
def _current_line(item) -> str:
    return f"当前: {render_value(item, get_value(item))}"


def _hint(item) -> str:
    required = "（必填）" if item.required else ""
    return f"{item.path} {required} | {type_hint(item)}"


def _print_help_header(path: str, cfg: dict) -> None:
    pkey = current_preset_key(cfg)
    print("=" * 64)
    print("  SME 配置菜单（配置与菜单一一对应，改动即写回）")
    print(f"  配置文件: {path}")
    print(f"  当前模式: {preset_name(pkey)}")
    print("=" * 64)


def _print_presets() -> None:
    print("\n=== 预设场景 ===")
    for p in PRESETS:
        print(f"[{p['name'].split(' ')[0]}] {p['name'][3:]}  —— {p['desc']}")
    print("[0] 返回")


def _print_groups() -> None:
    print("\n=== 手动配置（选择配置组）===")
    for i, (group, items) in enumerate(GROUPS, 1):
        print(f"[{i:>2}] {group}（{len(items)} 项）")
    print("[0] 返回")


def _print_items(cfg: dict, items: list) -> None:
    for i, item in enumerate(items, 1):
        print(f"[{i:>2}] {item.name}（{item.path}）  当前: {render_value(item, get_value(cfg, item))}")
        print(f"     说明: {item.desc}")
        if item.kind != "str" or item.required:
            print(f"     格式: {type_hint(item)}")
    print("[0] 返回")


def _print_all() -> None:
    for group, items in GROUPS:
        print(f"\n【{group}】")
        for item in items:
            print(f"  {item.name}（{item.path}）: {item.desc}")


# --------------------------------------------------------------------------- #
# 交互操作
# --------------------------------------------------------------------------- #
def _edit_item(cfg: dict, path: str, get_input=input) -> bool:
    item = ITEM_BY_PATH[path]
    old = get_value(cfg, item)
    print(f"\n修改：{item.name}（{item.path}）")
    print(f"说明: {item.desc}")
    print(f"当前: {render_value(item, old)} | 格式: {type_hint(item)}")
    while True:
        raw = get_input("输入新值（回车取消）> ").strip()
        if raw == "":
            print("  已取消")
            return False
        try:
            value = parse_value(item, raw)
        except ValueError as exc:
            print(f"  ✗ 输入无效：{exc}")
            continue
        set_value(cfg, item, value)
        return True


def _manual_loop(cfg: dict, path: str, get_input=input) -> None:
    while True:
        _print_groups()
        choice = get_input("选择配置组编号（q 返回）> ").strip().lower()
        if choice in ("q", "", "0"):
            return
        if not choice.isdigit() or not 1 <= int(choice) <= len(GROUPS):
            print("  无效编号")
            continue
        group, items = GROUPS[int(choice) - 1]
        while True:
            print(f"\n=== {group} ===")
            _print_items(cfg, items)
            pick = get_input("选择要修改的项（q 返回）> ").strip().lower()
            if pick in ("q", "", "0"):
                break
            if not pick.isdigit() or not 1 <= int(pick) <= len(items):
                print("  无效编号")
                continue
            item = items[int(pick) - 1]
            if _edit_item(cfg, item.path, get_input):
                save_config(path, cfg)
                print(f"  ✓ 已写回 {path}（重启后保持）")


def _preset_loop(cfg: dict, path: str, get_input=input) -> None:
    _print_presets()
    choice = get_input("选择预设编号> ").strip()
    if choice not in ("1", "2", "3", "4", "5"):
        print("  已取消")
        return
    preset = PRESETS[int(choice) - 1]
    changes = apply_preset(cfg, preset)
    if not changes:
        print(f"  = 已是【{preset['name'][3:]}】，无改动")
        return
    print(f"\n→ 已应用【{preset['name'][3:]}】")
    for item, old, new in changes:
        print(f"  ✓ {item.name}: {render_value(item, old)} → {render_value(item, new)}")
    save_config(path, cfg)
    print(f"  （已写回 {path}，重启后保持）")


def _view_config(cfg: dict) -> None:
    cfg = dict(cfg)
    cfg.pop("_help", None)
    print(json.dumps(cfg, ensure_ascii=False, indent=2))


def _main_loop(cfg: dict, path: str, get_input=input) -> None:
    while True:
        _print_help_header(path, cfg)
        print("[1] 预设场景    [2] 手动配置    [3] 全部配置说明    [4] 查看当前配置")
        print("[5] 重新加载    [q] 退出")
        choice = get_input("请输入编号> ").strip().lower()
        if choice in ("q", ""):
            print("  已退出")
            return
        elif choice == "1":
            _preset_loop(cfg, path, get_input)
        elif choice == "2":
            _manual_loop(cfg, path, get_input)
        elif choice == "3":
            _print_all()
        elif choice == "4":
            _view_config(cfg)
        elif choice == "5":
            cfg.clear()
            cfg.update(load_config(path))
            print("  已从文件重新加载")
        else:
            print("  无效输入")


# --------------------------------------------------------------------------- #
# 非交互模式
# --------------------------------------------------------------------------- #
def _cmd_list(cfg: dict) -> None:
    for group, items in GROUPS:
        print(f"\n【{group}】")
        for item in items:
            print(f"  {item.name}（{item.path}）  当前: {render_value(item, get_value(cfg, item))}")
            print(f"    说明: {item.desc}")


def _cmd_set(cfg: dict, path: str, pairs: list[str]) -> int:
    ok = True
    for pair in pairs:
        if "=" not in pair:
            print(f"  ✗ 无效设置（应为 key=value）：{pair}")
            ok = False
            continue
        key, _, raw = pair.partition("=")
        key = key.strip()
        item = ITEM_BY_PATH.get(key)
        if item is None:
            print(f"  ✗ 未知配置项：{key}（--list 查看全部可配项）")
            ok = False
            continue
        try:
            value = parse_value(item, raw)
        except ValueError as exc:
            print(f"  ✗ [{key}] {exc}")
            ok = False
            continue
        old = get_value(cfg, item)
        set_value(cfg, item, value)
        print(f"  ✓ {item.name}（{key}）: {render_value(item, old)} → {render_value(item, value)}")
    if ok:
        save_config(path, cfg)
        print(f"  已写回 {path}")
    return 0 if ok else 1


def _cmd_preset(cfg: dict, path: str, key: str) -> int:
    preset = PRESET_BY_KEY.get(key)
    if preset is None:
        print(f"  ✗ 未知预设：{key}（可选：{' '.join(PRESET_BY_KEY)}）")
        return 1
    changes = apply_preset(cfg, preset)
    for item, old, new in changes:
        print(f"  ✓ {item.name}: {render_value(item, old)} → {render_value(item, new)}")
    if not changes:
        print(f"  = 已是【{preset['name'][3:]}】，无改动")
    save_config(path, cfg)
    print(f"  已写回 {path}")
    return 0


def _cmd_check(cfg: dict, path: str, ping: bool = False) -> int:
    """验证配置：值合法性 + 引擎构建 + LLM/embedding 连通性。"""
    ok = True
    print("=== 配置验证 ===")
    warnings = validate_config(cfg)
    if warnings:
        ok = False
        for w in warnings:
            print(f"  ✗ {w}")
    else:
        print("  ✓ 全部配置值合法（类型/枚举/范围）")

    from sme.config import SMEConfig
    from sme.engine import SpatialMemoryEngine

    clean = {k: v for k, v in cfg.items() if k != "_help"}
    try:
        engine = SpatialMemoryEngine(SMEConfig.from_dict(clean))
        print(f"  ✓ 引擎构建成功：embedding={engine.embeddings.name} "
              f"({engine.embeddings.model_name}, dim={engine.embeddings.dim})")
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ 引擎构建失败：{exc}")
        return 1

    emb = engine.embeddings
    cfg_dim = engine.config.embedding.dim
    if emb.name != "hashing" and emb.dim and emb.dim != cfg_dim:
        print(f"  ✗ embedding 维度不匹配：模型实际 {emb.dim} 维，"
              f"配置 embedding.dim={cfg_dim}（应改为 {emb.dim}，"
              "否则空间/ANN 索引维度错位）")
        ok = False

    llm = engine.llm
    if llm.configured:
        key_state = "已填" if llm.config.api_key else "空（无鉴权头，多数服务会 401）"
        print(f"  ✓ LLM 已配置：{llm.base_url} model={llm.config.model} key={key_state}")
        if ping:
            try:
                out = llm.chat([{"role": "user", "content": "ping"}], max_tokens=8)
                print(f"  ✓ LLM 连通：{out[:40]}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ✗ LLM 请求失败：{exc}")
                ok = False
    else:
        print("  - LLM 未配置（llm.base_url 为空 → 纯离线可用）")

    emb = engine.embeddings
    if emb.name == "openai":
        if not emb.base_url or not emb.api_key:
            print(f"  ✗ embedding=openai 但 base_url/api_key 未填（将请求 {emb.base_url}）")
            ok = False
        elif ping:
            try:
                v = emb.embed(["连通性测试"])
                print(f"  ✓ embedding 连通：dim={len(v[0])}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ✗ embedding 请求失败：{exc}")
                ok = False
        else:
            print(f"  - embedding=openai（{emb.base_url}，--ping 可测连通）")
    else:
        print(f"  - embedding={emb.name}（{emb.model_name}，无需网络）")

    print(f"  {'✓ 配置可用' if ok else '✗ 存在需修复项'}（--ping 发真实请求测连通）")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SME 配置菜单（python -m sme.menu）")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help=f"配置文件路径（默认 {DEFAULT_CONFIG_PATH}）")
    parser.add_argument("--list", action="store_true", help="列出全部可配项")
    parser.add_argument("--show", action="store_true", help="查看当前配置内容")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="设置一项配置（可多次使用）")
    parser.add_argument("--preset", metavar="KEY",
                        help=f"套用预设（{' '.join(PRESET_BY_KEY)}）")
    parser.add_argument("--check", action="store_true",
                        help="验证配置与 LLM/embedding 连通性")
    parser.add_argument("--ping", action="store_true",
                        help="--check 时发真实 LLM/embedding 请求")
    args = parser.parse_args(argv)

    cfg, notes = _ensure_config(args.config)
    for note in notes:
        print(f"  ! {note}")
    warnings = validate_config(cfg)
    if warnings:
        print("  ! 配置文件中有以下无效值（将被忽略，使用默认值）：")
        for w in warnings:
            print(f"    {w}")

    if args.list:
        _cmd_list(cfg)
        return 0
    if args.show:
        _view_config(cfg)
        return 0
    if args.set:
        return _cmd_set(cfg, args.config, args.set)
    if args.preset:
        return _cmd_preset(cfg, args.config, args.preset)
    if args.check:
        return _cmd_check(cfg, args.config, ping=args.ping)
    _main_loop(cfg, args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
