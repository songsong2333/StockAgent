"""规律知识库 —— 持续累积的"经济学规律"(已验证/证伪/待定 + 证据)。

存储: data/knowledge/laws.yaml(列表式, git 友好)。每条 law:
  id            稳定主键(seed 手写; 自动提议用 statement 的短 hash)
  statement     规律陈述(自然语言)
  status        tentative / confirmed / refuted
  evidence      统计证据 dict(N/CI/FDR/源工具 等)
  related_series  相关序列 id 列表
  source        agent / user(人工确认绕过自动门槛)
  created/last_tested  日期
  history       变更日志 [{date, action, note}]

agent 提议(propose_law)永远写 tentative; 升 confirmed 在工具层校验统计门槛
(N≥min_n + FDR q<fdr_alpha + 滚动稳定>0.7), 人工(source=user)可绕过。
"""
from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from common import setup_logger, load_config, PROJECT_ROOT

log = setup_logger("agent.knowledge")

_STATUSES = ("tentative", "confirmed", "refuted")


def _kb_dir(cfg: dict) -> Path:
    d = cfg.get("agent", {}).get("knowledge_dir", "data/knowledge")
    p = Path(d)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _laws_path(cfg: dict) -> Path:
    return _kb_dir(cfg) / "laws.yaml"


def _today() -> str:
    return date.today().isoformat()


def load_laws(cfg: dict) -> list:
    """读全部规律。文件不存在返回 [](不报错, 便于首跑)。"""
    p = _laws_path(cfg)
    if not p.exists():
        return []
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:
        log.warning(f"读取规律库失败: {e}")
        return []
    return data.get("laws", []) or []


def save_laws(cfg: dict, laws: list) -> None:
    """原子写规律库(先写 tmp 再 replace)。"""
    p = _laws_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump({"laws": laws}, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    tmp.replace(p)


def _slug(statement: str) -> str:
    """由陈述生成稳定短 id(前8位 sha1)。"""
    return "law-" + hashlib.sha1(statement.encode("utf-8")).hexdigest()[:8]


def query_laws(cfg: dict, query: str, limit: int = 10) -> list:
    """关键词检索规律(命中 statement/related_series/id 任一)。供 agent 提议前查重。

    query 为空时返回全部(按 last_tested 倒序, 截 limit)。匹配按命中字段加权排序。
    """
    laws = load_laws(cfg)
    if not query or not query.strip():
        return sorted(laws, key=lambda x: x.get("last_tested", ""), reverse=True)[:limit]
    q = query.strip().lower()
    kws = [w for w in q.replace(",", " ").split() if w]

    def _score(law):
        st = str(law.get("statement", "")).lower()
        rs = " ".join(str(x) for x in law.get("related_series", []) or []).lower()
        lid = str(law.get("id", "")).lower()
        s = 0
        for w in kws:
            if w in st:
                s += 3
            if w in rs:
                s += 2
            if w in lid:
                s += 1
        return s

    scored = [(s, law) for s, law in ((_score(l), l) for l in laws) if s > 0]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [law for _, law in scored[:limit]]


def propose_law(cfg: dict, statement: str, evidence: Optional[dict] = None,
                related_series: Optional[list] = None, source: str = "agent") -> dict:
    """提议一条规律(永远 tentative)。若 statement 已存在(id 命中)则更新其 evidence 不新建。

    返回新建/更新后的 law dict。
    """
    laws = load_laws(cfg)
    lid = _slug(statement)
    today = _today()
    evidence = evidence or {}
    related_series = [str(x) for x in (related_series or [])]
    existing = next((l for l in laws if l.get("id") == lid), None)
    if existing:
        existing["evidence"] = {**existing.get("evidence", {}), **evidence}
        existing["last_tested"] = today
        existing.setdefault("history", []).append({"date": today, "action": "re-tested", "note": "agent 复检更新证据"})
        log.info(f"更新规律 {lid}: {statement[:30]}...")
    else:
        existing = {"id": lid, "statement": statement, "status": "tentative",
                    "evidence": evidence, "related_series": related_series,
                    "source": source, "created": today, "last_tested": today,
                    "history": [{"date": today, "action": "proposed", "note": f"{source} 提议"}]}
        laws.append(existing)
        log.info(f"提议新规律 {lid}: {statement[:30]}...")
    save_laws(cfg, laws)
    return existing


def _meets_confirm_gate(evidence: dict, cfg: dict) -> tuple:
    """检查证据是否满足"升 confirmed"的自动门槛。返回 (ok, 缺失原因列表)。

    门槛: N≥min_n, fdr_q<fdr_alpha, stability>0.7(若提供)。缺字段算不达标。
    """
    cc = cfg.get("cycle", {})
    min_n = cc.get("min_n", 5)
    alpha = cc.get("fdr_alpha", 0.05)
    why = []
    n = evidence.get("n") or evidence.get("n_years") or evidence.get("n_events")
    if n is None or n < min_n:
        why.append(f"样本量 {n} < min_n={min_n}")
    q = evidence.get("fdr_q") or evidence.get("q")
    if q is not None and q >= alpha:
        why.append(f"FDR q={q:.3f} ≥ {alpha}(多重检验不显著)")
    stab = evidence.get("stability")
    if stab is not None and stab < 0.7:
        why.append(f"滚动稳定性 {stab:.2f} < 0.7(关系不稳定)")
    if "ci_lo" in evidence and "ci_hi" in evidence:
        if evidence["ci_lo"] <= 0 <= evidence["ci_hi"]:
            why.append("置信区间跨 0(效应不明确)")
    return (len(why) == 0, why)


def update_law_status(cfg: dict, law_id: str, status: str, reason: str = "",
                      source: str = "agent") -> dict:
    """迁移规律状态。升 confirmed: agent 需过自动门槛, user 可绕过(人工确认)。

    返回 {ok, law, reason}。失败(找不到/门槛不过)返回 ok=False + 原因。
    """
    if status not in _STATUSES:
        return {"ok": False, "reason": f"非法 status={status}(允许 {list(_STATUSES)})"}
    laws = load_laws(cfg)
    law = next((l for l in laws if l.get("id") == law_id), None)
    if law is None:
        return {"ok": False, "reason": f"找不到规律 id={law_id}"}
    if status == "confirmed" and source != "user":
        ok, why = _meets_confirm_gate(law.get("evidence", {}) or {}, cfg)
        if not ok:
            return {"ok": False, "reason": "未达 confirmed 自动门槛: " + "; ".join(why)
                    + "。需更多样本/样本外验证, 或人工确认(source=user)。"}
    old = law.get("status")
    law["status"] = status
    law["last_tested"] = _today()
    law.setdefault("history", []).append(
        {"date": _today(), "action": f"{old}→{status}", "note": reason or f"{source} 操作"})
    save_laws(cfg, laws)
    log.info(f"规律 {law_id} 状态 {old}→{status}({source})")
    return {"ok": True, "law": law}


def law_to_summary(law: dict) -> str:
    """规律 → 紧凑单行摘要(给 LLM 看)。"""
    ev = law.get("evidence", {}) or {}
    ev_str = ", ".join(f"{k}={v}" for k, v in list(ev.items())[:4])
    return (f"[{law.get('status','?')}] {law.get('id')}: {law.get('statement')} "
            f"({ev_str})")


if __name__ == "__main__":
    cfg = load_config()
    print(f"规律库 {len(load_laws(cfg))} 条:")
    for l in load_laws(cfg):
        print("  " + law_to_summary(l))
