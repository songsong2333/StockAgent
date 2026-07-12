"""激活码 + 功能门控 (本地软锁)。

license 格式: base32url(json_payload) + "." + base32url(hmac_sha256(payload, secret)[:10])
payload: {"tier": "pro", "exp": "2026-09-25", "mac": "a1b2c3d4"(可选绑定)}
校验: 签名匹配 + 未过期 + (可选)本机mac匹配。

secret 从环境变量读 (config.license.secret_env, 默认 STOCK_AGENT_SECRET)。
激活信息存 config/.license (gitignore)。这是软锁, 早期付费够用, 真防破解留 SaaS 阶段。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import uuid
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from common import PROJECT_ROOT

LICENSE_FILE = PROJECT_ROOT / "config" / ".license"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _secret() -> bytes:
    """从 config.license.secret_env 指定的环境变量读 secret。"""
    env_name = "STOCK_AGENT_SECRET"
    try:
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            env_name = (cfg.get("license") or {}).get("secret_env", env_name)
    except Exception:
        pass
    val = os.environ.get(env_name, "")
    return val.encode("utf-8") or b"stock-agent-default-secret"


def _b32(b: bytes) -> str:
    return base64.b32encode(b).decode("ascii").rstrip("=").lower()


def _unb32(s: str) -> bytes:
    pad = "=" * (-len(s) % 8)
    return base64.b32decode((s + pad).upper())


def machine_id() -> str:
    """本机标识(mac 的 sha256 前8位)。"""
    return hashlib.sha256(str(uuid.getnode()).encode()).hexdigest()[:8]


def encode_license(tier: str, exp: str, secret: bytes = None, mac: Optional[str] = "_self") -> str:
    """生成激活码。mac="_self" 绑定当前机器, None=不绑定。"""
    secret = secret or _secret()
    payload = {"tier": tier, "exp": exp}
    if mac == "_self":
        payload["mac"] = machine_id()
    elif mac:
        payload["mac"] = mac
    body = json.dumps(payload, separators=(",", ":")).encode()
    sig = hmac.new(secret, body, hashlib.sha256).digest()[:10]
    return _b32(body) + "." + _b32(sig)


def decode_license(code: str, secret: bytes = None) -> Optional[dict]:
    """校验激活码, 合法返回 payload, 否则 None。"""
    secret = secret or _secret()
    if not code or "." not in code:
        return None
    body_b, sig_b = code.split(".", 1)
    try:
        body = _unb32(body_b)
        sig = _unb32(sig_b)
    except Exception:
        return None
    expect = hmac.new(secret, body, hashlib.sha256).digest()[:10]
    if not hmac.compare_digest(sig, expect):
        return None
    try:
        payload = json.loads(body)
    except Exception:
        return None
    # 过期检查
    try:
        if date.fromisoformat(payload["exp"]) < date.today():
            return None
    except Exception:
        return None
    # mac 绑定检查
    if payload.get("mac") and payload["mac"] != machine_id():
        return None
    return payload


def activate(code: str) -> tuple[bool, str]:
    """校验并写入本地 license。返回 (是否成功, 消息)。"""
    payload = decode_license(code)
    if not payload:
        return False, "激活码无效(签名错/过期/机器不匹配)"
    LICENSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    LICENSE_FILE.write_text(code.strip(), encoding="utf-8")
    return True, f"已激活 {payload['tier'].upper()} (有效期至 {payload['exp']})"


def current_tier() -> str:
    """读本地 license, 返回 'pro' 或 'free'。"""
    if not LICENSE_FILE.exists():
        return "free"
    payload = decode_license(LICENSE_FILE.read_text(encoding="utf-8").strip())
    return payload["tier"] if payload else "free"


def is_pro() -> bool:
    return current_tier() == "pro"


def license_info() -> Optional[dict]:
    if not LICENSE_FILE.exists():
        return None
    return decode_license(LICENSE_FILE.read_text(encoding="utf-8").strip())


# ============== Streamlit 门控辅助 ==============
def locked_page(title: str, feature_desc: str = ""):
    """渲染锁定页: 显示 Pro 介绍 + 激活码输入。返回是否已解锁(供页面继续渲染)。"""
    import streamlit as st
    if is_pro():
        return True
    st.title(f"🔒 {title}")
    st.caption("Pro 付费功能")
    st.write(feature_desc or "此功能需激活 Pro 许可证后解锁。")
    st.divider()
    info = license_info()
    if LICENSE_FILE.exists() and not info:
        st.error("本地许可证无效或已过期, 请重新激活。")
    code = st.text_input("输入激活码", key=f"lic_{title}")
    if st.button("🔑 激活", type="primary", key=f"act_{title}"):
        ok, msg = activate(code)
        if ok:
            # 不调 st.rerun(): 按钮点击本身会从顶重跑, 下一轮 is_pro() 即为 True 自动解锁。
            # (在此调 st.rerun() 会触发二次重跑, 与已注册的导航 widget 撞 key 报错)
            st.success(msg + " 已激活，点左侧任意菜单或刷新即可解锁 Pro ✅")
        else:
            st.error(msg)
    st.caption("激活码由卖家通过发卡平台发放。这是本地软锁, 仅供学习研究。")
    return False
