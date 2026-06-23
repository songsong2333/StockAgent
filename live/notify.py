"""通知: 企业微信/钉钉 webhook + 邮件(SMTP)。"""
from __future__ import annotations

import json
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from typing import Optional

import requests

from common import setup_logger

log = setup_logger("live.notify")


def send_webhook(url: str, content: str, webhook_type: str = "wechat") -> bool:
    """发送 webhook 通知。content 为 markdown/text 文本。"""
    if not url:
        log.warning("webhook_url 为空, 跳过推送")
        return False
    if webhook_type == "dingtalk":
        payload = {"msgtype": "markdown", "markdown": {"title": "调仓信号", "text": content}}
    else:  # 企业微信
        payload = {"msgtype": "markdown", "markdown": {"content": content}}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200 and resp.json().get("errcode", 0) == 0:
            log.info("webhook 推送成功")
            return True
        log.warning(f"webhook 推送异常: {resp.status_code} {resp.text[:200]}")
        return False
    except Exception as e:
        log.error(f"webhook 推送失败: {e}")
        return False


def send_email(smtp_cfg: dict, subject: str, content: str) -> bool:
    """发送邮件。"""
    if not smtp_cfg.get("email_enable") or not smtp_cfg.get("smtp_user"):
        log.info("邮件未启用, 跳过")
        return False
    msg = MIMEText(content, "markdown", "utf-8")
    msg["Subject"] = subject
    msg["From"] = formataddr(("StockAgent", smtp_cfg["smtp_user"]))
    msg["To"] = smtp_cfg["email_to"]
    try:
        with smtplib.SMTP_SSL(smtp_cfg["smtp_host"], smtp_cfg["smtp_port"], timeout=15) as s:
            s.login(smtp_cfg["smtp_user"], smtp_cfg["smtp_pass"])
            s.sendmail(smtp_cfg["smtp_user"], smtp_cfg["email_to"].split(","), msg.as_string())
        log.info("邮件发送成功")
        return True
    except Exception as e:
        log.error(f"邮件发送失败: {e}")
        return False


def notify(cfg: dict, content: str, subject: str = "A股量化调仓信号") -> bool:
    """统一通知入口。根据 config 推送 webhook + 邮件。"""
    nc = cfg.get("notify", {})
    if not nc.get("enable", False):
        log.info("通知未启用 (notify.enable=false)")
        return False
    ok = False
    if nc.get("webhook_url"):
        ok = send_webhook(nc["webhook_url"], content, nc.get("webhook_type", "wechat")) or ok
    if nc.get("email_enable"):
        ok = send_email(nc, subject, content) or ok
    return ok
