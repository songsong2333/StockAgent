# A股量化交易系统

一套个人 A 股量化交易系统，覆盖完整闭环：**每日自动采集数据 → qlib 回测验证 → 多因子选股生成调仓信号 → 推送提醒**。并提供 **Mac 桌面应用**，可点击配置、可视化回测与调仓调试。

技术栈：akshare（数据）+ qlib（回测/因子）+ LightGBM（模型）+ Streamlit（GUI）。

## 功能

- 📈 **数据采集**：akshare 拉全 A 股日线（前复权），增量更新，自动剔除 ST/新股/北交所
- 📊 **回测**：qlib Alpha158 因子 + LightGBM，内置 A 股规则（T+1、佣金、印花税、涨跌停）
- 📡 **信号调仓**：模型打分生成目标持仓，对比当前持仓输出买卖清单
- 📨 **通知**：企业微信/钉钉 webhook + 邮件（SMTP）
- ⏰ **定时**：macOS launchd 工作日盘后自动跑全流程
- 🖥 **Mac 应用**：可视化配置 + 回测图表 + 调仓调试

## 安装

```bash
cd /Users/jojo/coding/stock_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> qlib 依赖较重，如安装失败可参考 [qlib 官方文档](https://github.com/microsoft/qlib)。

## 快速开始

### 1. 初始化历史数据
```bash
# 先用少量股票测试链路
python scripts/init_history.py --symbols 000001,600000 --years 1

# 全市场近5年（耗时数小时，建议后台运行）
nohup python scripts/init_history.py --years 5 > init.log 2>&1 &
```

### 2. 运行回测
```bash
python scripts/run_backtest.py
```
回测报告输出到 `data/cache/backtest_report.md`。

### 3. 生成每日信号
```bash
python scripts/run_daily.py --dry-run      # 只生成不推送
python scripts/run_daily.py                # 生成并推送(需配置 notify)
```

## Mac 桌面应用

应用基于 Streamlit，本地运行自动打开浏览器，提供可视化配置与调试。

### 启动方式

**方式一：双击启动（最简单）**
Finder 中双击 `启动应用.command` 文件即可。

**方式二：命令行**
```bash
streamlit run app/app.py
# 或
python app/launcher.py
```

### 应用页面
- **📈 概览**：系统状态、数据覆盖、最新调仓信号
- **⚙️ 配置**：表单化编辑采集/模型/回测/通知参数，一键保存
- **🗃️ 数据采集**：查看股票池、触发历史数据采集与 qlib 转换
- **📊 回测**：运行回测，展示年化/夏普/回撤指标与净值曲线（plotly 交互图）
- **📡 信号调仓**：编辑当前持仓、生成目标持仓、查看调仓清单、推送通知

### 打包成原生 .app（可选）
```bash
pip install py2app
# 在项目根创建 setup.py 后:
py2applet --make-setup app/launcher.py
python setup.py py2app
```
生成的 `.app` 可拖入「应用程序」文件夹，像普通 Mac 应用一样使用。

## 定时任务（macOS launchd）

```bash
# 加载定时任务（工作日 16:30 自动运行每日流程）
launchctl load scheduler/com.stockagent.daily.plist

# 手动触发一次测试
launchctl start com.stockagent.daily

# 查看日志
tail -f data/cache/logs/daily.log

# 卸载
launchctl unload scheduler/com.stockagent.daily.plist
```

脚本内部会判断交易日，非交易日（周末/节假日）自动跳过。

## 配置

核心配置在 `config/config.yaml`，也可在应用的「配置」页可视化编辑：

| 模块 | 关键项 | 说明 |
|---|---|---|
| collector | adjust / exclude_st / exclude_new_days | 复权、ST/新股过滤 |
| model | model_class / train/test 区间 | 模型与训练分段 |
| backtest | topk / n_drop / commission_rate / stamp_duty | 持仓、成本、A 股规则 |
| live | topk / weight_scheme / dry_run | 目标持仓与推送开关 |
| notify | webhook_url / smtp_* | 通知渠道 |

## 项目结构

```
stock_agent/
├── app/                  # Mac 桌面应用 (Streamlit)
├── collector/            # 数据采集 (akshare → qlib bin)
├── factor/               # 因子封装 (Alpha158)
├── strategy/             # 策略 + IC 分析
├── backtest/             # qlib 回测 + 报告
├── live/                 # 信号生成 + 调仓 + 通知
├── scripts/              # 命令行入口
├── scheduler/            # launchd 定时
├── config/config.yaml    # 全局配置
└── data/                 # 原始数据 / qlib 数据 / 缓存
```

## 备注

- 信号提醒版不涉及真实资金；未来接实盘（QMT/xtquant）在 `live/broker.py` 新增适配层即可。
- akshare 接口偶有变动，采集模块已做字段容错与版本注释。
- 本系统仅供学习研究，不构成投资建议。实盘交易有风险。
