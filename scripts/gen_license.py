"""生成 Pro 激活码 (卖家发卡用)。

需先设环境变量 STOCK_AGENT_SECRET (与 App 校验用同一个 secret):
  export STOCK_AGENT_SECRET="你的密钥"

用法:
  python scripts/gen_license.py --tier pro --days 90            # 绑定当前机器
  python scripts/gen_license.py --tier pro --days 365 --no-mac  # 不绑机器(通用码)
  python scripts/gen_license.py --tier pro --expire 2026-12-31
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.paywall import encode_license, _secret


def main():
    p = argparse.ArgumentParser(description="生成 Pro 激活码")
    p.add_argument("--tier", default="pro", choices=["pro"])
    p.add_argument("--days", type=int, default=None, help="有效天数(与--expire二选一)")
    p.add_argument("--expire", default=None, help="到期日期 YYYY-MM-DD")
    p.add_argument("--no-mac", action="store_true", help="不绑定机器(通用码)")
    args = p.parse_args()

    if not os.environ.get("STOCK_AGENT_SECRET"):
        print("⚠️  未设置 STOCK_AGENT_SECRET 环境变量, 用默认密钥(仅测试用!)")
        print("    生产请: export STOCK_AGENT_SECRET='你的密钥'\n")

    if args.expire:
        exp = args.expire
    elif args.days:
        exp = (date.today() + timedelta(days=args.days)).isoformat()
    else:
        exp = (date.today() + timedelta(days=90)).isoformat()

    mac = None if args.no_mac else "_self"
    code = encode_license(args.tier, exp, secret=_secret(), mac=mac)
    print(f"激活码 ({args.tier}, 到期 {exp}, {'不绑机器' if args.no_mac else '绑定当前机器'}):")
    print(code)
    print("\n将此码通过发卡平台售出。用户在 App 内输入即可解锁 Pro。")


if __name__ == "__main__":
    main()
