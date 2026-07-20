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
- 📊 **ETF 策略**：场内宽基/行业 ETF 的趋势择时 + 动量轮动，精确成本建模 + 信号防抖（独立于个股 qlib 管线）

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

## ETF 策略（场内基金）

针对个人小资金，用宽基/行业 ETF 做**趋势择时 + 动量轮动**，比个股选股更稳妥（分散、免印花税、门槛低、不易过拟合）。**与个股 qlib 管线完全解耦**——独立采集 + 自写精确佣金回测，不走 qlib。

**分层标的池**（`config/etf_pool.yaml`，可自行增删）：
- **宽基 + 避险**：沪深300 / 中证500 / 中证1000 / 创业板 / 科创50 + 黄金ETF
- **行业/主题**：半导体 / 电池 / 消费 / 银行 / 医药 / 军工 等

**两个策略**（应用「📊 ETF策略」页，免费）：
- **📈 趋势择时**：双均线（MA20/MA60）+ 信号防抖（连续确认 + 均线带宽过滤 + 状态保持），均线上持有、下穿空仓。适合宽基+黄金池。
- **🔄 动量轮动**：在池中选动量最强 topN。宽基池纯轮动；**行业池叠大盘择时开关**（沪深300 ETF 均线下连续确认 → 空仓避险）+ 短线动量横截面加权。

**精确成本建模**（场内 ETF 规则）：佣金 `max(金额×万2.5, 5元/笔)`，最低 5 元门槛惩罚小额频繁调仓，**免印花税**。默认周度调仓（可调月度），实测换手 ~3.6x/年，低频不磨损。

**首次采集 ETF 数据**：
```bash
# 全池近 6 年（注意：东财源连续批量请求易触发限流，失败的稍后重试即可）
python scripts/init_etf.py --years 6
# 指定几只快速测试
python scripts/init_etf.py --codes 510300,518880 --years 3
```

**命令行回测/信号**：
```bash
python -c "from common import load_config; from strategy.etf import backtest_trend; \
cfg=load_config(); bt=backtest_trend(cfg); \
print(f'累计{bt[\"cum_ret\"]:+.1%} 回撤{bt[\"max_dd\"]:.1%} 换手{bt[\"turnover_annual\"]:.1f}x/年 {bt[\"n_trades\"]}笔')"
```

数据存 `data/etf/<代码>.parquet`（独立于个股 `data/raw/`），前复权 OHLCV，全程用原始 6 位代码。

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
| etf | trend(ma/防抖) / rotation(topn/动量/大盘择时) / benchmark_etf | ETF 策略参数（成本复用 backtest 段） |
| live | topk / weight_scheme / dry_run | 目标持仓与推送开关 |
| notify | webhook_url / smtp_* | 通知渠道 |

## 项目结构

```
stock_agent/
├── app/                  # Mac 桌面应用 (Streamlit, 含 ETF策略页)
├── collector/            # 数据采集 (akshare → qlib bin; etf_collector 独立存 data/etf)
├── factor/               # 因子封装 (Alpha158)
├── strategy/             # 策略 + IC 分析 + 轮动/短线/ETF(etf.py)
├── backtest/             # qlib 回测 + 报告
├── live/                 # 信号生成 + 调仓 + 通知
├── scripts/              # 命令行入口 (init_history / init_etf / run_daily ...)
├── scheduler/            # launchd 定时
├── config/               # 全局配置 + 标的池 (watchlist/sector_leaders/etf_pool)
└── data/                 # raw(个股) / qlib_data / etf / cache
```

## 备注

- 信号提醒版不涉及真实资金；未来接实盘（QMT/xtquant）在 `live/broker.py` 新增适配层即可。
- akshare 接口偶有变动，采集模块已做字段容错与版本注释。
- 本系统仅供学习研究，不构成投资建议。实盘交易有风险。
