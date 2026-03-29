# -*- coding: utf-8 -*-
"""
FGO 이벤트 주회 시뮬레이터 v3.0
================================
기능은 v2.5와 동일, 구조 재설계:
  MaterialStore  — 재료 DB, 부족/획득 추적
  EventData      — JSON 로드, 이벤트/스테이지 조회
  EfficiencyEngine — 스테이지별 AP 효율 계산
  Simulator      — 메인 루프, 로그 출력
"""

from __future__ import annotations

import argparse
import difflib
import json
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

# ===========================================================================
# ★ 사용자 설정 — 실행 전 여기만 수정하면 됨
# ===========================================================================

# 사과 보유 수량
APPLE_COUNTS: Dict[str, int] = {
    "gold":   1321,   # 금사과
    "silver":  118,   # 은사과
    "blue":    323,   # 청사과
    "copper":   11,   # 동사과
}

# 사과 1개당 회복 AP
APPLE_AP: Dict[str, float] = {
    "gold":   146.0,  # 금사과
    "silver":  73.0,  # 은사과
    "blue":    40.0,  # 청사과
    "copper":  10.0,  # 동사과
}

# 사과 외 추가 AP (이미 보유한 AP 잔량 등)
NATURAL_AP = 0.0

# ===========================================================================

APPLE_NAME_TO_POOL: Dict[str, str] = {"금사과": "gold", "은사과": "silver", "청사과": "blue", "동사과": "copper"}
APPLE_ITEM_NAMES = set(APPLE_NAME_TO_POOL)

# tier → box rarity 매핑 (materials.json의 tier 값 기준)
TIER_TO_RARITY: Dict[str, str] = {"금테": "gold", "은테": "silver", "동테": "bronze"}

CE_BASE_BONUS   = 7
CE_MAX_BONUS    = 12
CE_COPIES_PER_PLUS = 4

# ---------------------------------------------------------------------------
# 숫자/포맷 유틸
# ---------------------------------------------------------------------------

def r2(x: float) -> float:
    return round(float(x), 2)

def f2(x: float) -> str:
    return f"{float(x):,.2f}"

def f2s(x) -> str:
    try:
        if x == float("inf") or str(x).lower() == "inf":
            return "∞"
        return f2(float(x))
    except Exception:
        return str(x)

def f0(x) -> str:
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return str(x)

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# CJK 표시폭 계산
# ---------------------------------------------------------------------------

try:
    from wcwidth import wcswidth as _wcswidth  # type: ignore
    def display_width(s: str) -> int:
        return _wcswidth(s)
except ImportError:
    def display_width(s: str) -> int:  # type: ignore[misc]
        w = 0
        for ch in s:
            if unicodedata.combining(ch):
                continue
            ea = unicodedata.east_asian_width(ch)
            w += 2 if ea in ("F", "W") else 1
        return w

# ---------------------------------------------------------------------------
# 표 출력
# ---------------------------------------------------------------------------

@dataclass
class TableStyle:
    fmt:   str = "text"   # "text" | "md" | "csv"
    style: str = "box"    # "box" | "ascii"
    arrow: str = "→"
    bullet: str = "·"

_BOX   = {"top": ("┌","┬","┐"), "mid": ("├","┼","┤"), "bot": ("└","┴","┘"), "h": "─", "v": "│"}
_ASCII = {"top": ("+","+","+"), "mid": ("+","+","+"), "bot": ("+","+","+"), "h": "-", "v": "|"}

def _draw_line(widths, left, mid, right, fill) -> str:
    parts = [left]
    for i, w in enumerate(widths):
        parts.append(fill * (w + 2))
        parts.append(mid if i < len(widths) - 1 else right)
    return "".join(parts)

def _pad_cell(text: str, width: int, align: str) -> str:
    t = str(text)
    pad = max(0, width - display_width(t))
    if align == "right":
        return " " + " " * pad + t + " "
    if align == "center":
        lp = pad // 2
        return " " + " " * lp + t + " " * (pad - lp) + " "
    return " " + t + " " * pad + " "

def print_table(write, headers, rows, aligns=None, ts: TableStyle = None):
    ts = ts or TableStyle()
    if aligns is None:
        aligns = ["left"] * len(headers)

    if ts.fmt == "md":
        write("| " + " | ".join(str(h) for h in headers) + " |")
        amap = {"left": ":---", "right": "---:", "center": ":---:"}
        write("| " + " | ".join(amap.get(a, ":---") for a in aligns) + " |")
        for r in rows:
            write("| " + " | ".join(str(c) for c in r) + " |")
        return

    if ts.fmt == "csv":
        def _esc(s):
            s = str(s)
            return '"' + s.replace('"', '""') + '"' if any(c in s for c in ',"\n\r') else s
        write(",".join(_esc(h) for h in headers))
        for r in rows:
            write(",".join(_esc(c) for c in r))
        return

    # text
    chars = _BOX if ts.style == "box" else _ASCII
    v = chars["v"]
    widths = [max(display_width(str(h)), *(display_width(str(r[i])) for r in rows))
              for i, h in enumerate(headers)]
    write(_draw_line(widths, *chars["top"], fill=chars["h"]))
    write(v + v.join(_pad_cell(str(h), widths[i], "center") for i, h in enumerate(headers)) + v)
    write(_draw_line(widths, *chars["mid"], fill=chars["h"]))
    for r in rows:
        write(v + v.join(_pad_cell(str(r[i]), widths[i], aligns[i]) for i in range(len(headers))) + v)
    write(_draw_line(widths, *chars["bot"], fill=chars["h"]))

# ---------------------------------------------------------------------------
# 로거
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, filepath: str, tee: bool = False):
        self.filepath = filepath
        self.tee = tee
        self._f = open(filepath, "w", encoding="utf-8")

    def write(self, line: str = ""):
        self._f.write(line + "\n")
        if self.tee:
            print(line)

    def close(self):
        self._f.flush()
        self._f.close()

    def prepend(self, lines: List[str]):
        """파일 상단에 요약 헤더 삽입"""
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                original = f.read()
            with open(self.filepath, "w", encoding="utf-8") as f:
                for ln in lines:
                    f.write(ln + "\n")
                f.write("\n")
                f.write(original)
        except Exception:
            pass

# ---------------------------------------------------------------------------
# MaterialStore — 재료 DB, 부족/획득 추적
# ---------------------------------------------------------------------------

@dataclass
class MatRecord:
    ap:   float
    need: float
    tier: str = ""

class MaterialStore:
    """
    materials.json 기반 재료 DB.
    부족량(need) 차감과 누적 획득(cum_gained) 추적을 담당한다.
    """
    def __init__(self, materials_json: dict):
        self._records: Dict[str, MatRecord] = {}
        self.sort_index: Dict[str, int] = {}
        self.tier_order: Dict[str, int] = {}
        self.cum_gained: Dict[str, float] = {}
        self.current_run_gains: Dict[str, float] = {}
        self._load(materials_json)

    def _load(self, mj: dict):
        for i, m in enumerate(mj.get("materials", [])):
            name = str(m.get("item") or "").strip()
            if not name:
                continue

            ap_val = float(m.get("ap_per_item") or 0.0)

            need = float(m.get("need") or 0.0)

            tier = str(m.get("tier") or "").strip()
            self._records[name] = MatRecord(ap=ap_val, need=need, tier=tier)
            self.sort_index.setdefault(name, i)
            if tier and tier not in self.tier_order:
                self.tier_order[tier] = len(self.tier_order)

    # --- 조회 ---

    def record(self, name: str) -> Optional[MatRecord]:
        return self._records.get(name)

    def ap_value(self, name: str) -> float:
        """아이템 1개당 AP 가치 (need > 0일 때만)"""
        rec = self._records.get(name)
        if not rec:
            return 0.0
        if rec.need <= 0:
            return 0.0
        return rec.ap

    def need(self, name: str) -> float:
        rec = self._records.get(name)
        return rec.need if rec else 0.0

    def rarity_of(self, name: str) -> Optional[str]:
        rec = self._records.get(name)
        if not rec:
            return None
        return TIER_TO_RARITY.get(rec.tier)

    def sort_key(self, name: str):
        return (self.sort_index.get(name, 10**9),
                self.tier_order.get(self._records.get(name, MatRecord(0,0)).tier, 10**9),
                name)

    def snapshot_need(self) -> Dict[str, float]:
        return {
            n: rec.need
            for n, rec in self._records.items()
        }

    def all_names(self) -> List[str]:
        return list(self._records)

    # --- 획득 반영 ---

    def apply_gain(self, name: str, qty: float):
        """획득량을 누적/부족에 반영하고 이번 판 변동에도 기록"""
        if qty <= 0:
            return
        self.cum_gained[name] = self.cum_gained.get(name, 0.0) + qty
        self.current_run_gains[name] = self.current_run_gains.get(name, 0.0) + qty
        rec = self._records.get(name)
        if rec:
            rec.need = max(rec.need - qty, 0.0)

    def reset_run_gains(self):
        self.current_run_gains = {}

# ---------------------------------------------------------------------------
# EventData — JSON 데이터 조회 전담
# ---------------------------------------------------------------------------

class EventData:
    """
    event_quests.json 단일 파일 조회 (event_items.json 통합 후).

    통합 포맷:
      - roulette 이벤트: need_tickets, box_pool(상자 드랍풀), exchanges 키 포함
      - box 이벤트:      box_pool 키 포함
    """
    def __init__(self, quests_def: dict):
        self.quests = quests_def

    # --- 이벤트 블록 조회 ---

    def event_block(self, event: str, case: str) -> Optional[dict]:
        for e in self.quests.get("event_quests", []):
            if str(e.get("event") or "") == event and str(e.get("case") or "").lower() == case:
                return e
        return None

    def stage_def(self, event: str, case: str, stage: str, diff: str) -> Optional[dict]:
        blk = self.event_block(event, case)
        if not blk:
            return None
        for st in blk.get("stages", []):
            if str(st.get("stage") or "") == stage and str(st.get("diff") or "") == str(diff):
                return st
        return None

    def list_targets(self, event_filter: Optional[str]) -> List[Tuple[str, str]]:
        seen: set = set()
        out: List[Tuple[str, str]] = []
        for e in self.quests.get("event_quests", []):
            ev = str(e.get("event") or "")
            cs = str(e.get("case") or "").lower()
            if not ev or cs not in ("roulette", "box", "raid"):
                continue
            if event_filter and event_filter not in ev:
                continue
            key = (ev, cs)
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def need_tickets(self, event: str, default: float = 600.0) -> float:
        blk = self.event_block(event, "roulette")
        if not blk:
            return default
        try:
            return float(blk.get("need_tickets") or default)
        except Exception:
            return default

    def ce_drop_rate(self, stage: dict, event_blk: Optional[dict]) -> float:
        v = stage.get("ce_drop_rate")
        if v is None and event_blk:
            v = event_blk.get("ce_drop_rate")
        try:
            return float(v or 0.0)
        except Exception:
            return 0.0

    def box_contents(self, stage: dict, event_blk: dict) -> Dict[str, List[dict]]:
        """스테이지 내 box contents 추출 (스테이지 → 이벤트 순)"""
        box = stage.get("box") or {}
        for key in ("items", "contents"):
            if isinstance(box.get(key), dict):
                return box[key]
        if isinstance(event_blk.get("box_contents"), dict):
            return event_blk["box_contents"]
        return {}

    def box_pool_grouped(self, event: str, case: str, store: MaterialStore) -> Dict[str, List[dict]]:
        """
        이벤트 블록의 box_pool을 rarity 기준으로 gold/silver/bronze 그룹으로 분류.
        stage별 box contents가 없을 때 fallback으로 사용.
        """
        blk = self.event_block(event, case)
        if not blk:
            return {}
        pool = blk.get("box_pool", [])
        if not pool:
            return {}
        groups: Dict[str, List[dict]] = {"gold": [], "silver": [], "bronze": []}
        for d in pool:
            item = str(d.get("item") or "").strip()
            if not item:
                continue
            rar = store.rarity_of(item)
            if rar not in groups:
                continue
            entry: dict = {"item": item}
            if "qty_per_box" in d:
                entry["qty_per_box"] = float(d.get("qty_per_box") or 0.0)
            elif "rate" in d:
                entry["rate"] = float(d.get("rate") or 0.0)
            groups[rar].append(entry)
        # rate/qty 없는 그룹은 균등 배분
        for lst in groups.values():
            if lst and not any("qty_per_box" in x or "rate" in x for x in lst):
                eq = 1.0 / len(lst)
                for x in lst:
                    x["rate"] = eq
        return groups

    # --- 아이템명 검증 ---

    def validate_names(self, store: MaterialStore) -> Dict[str, List[str]]:
        known = set(store.all_names()) | APPLE_ITEM_NAMES | {"교환티켓"}
        unknown: Dict[str, List[str]] = {}

        def report(name: str, where: str):
            n = name.strip()
            if not n or n in known:
                return
            lst = unknown.setdefault(n, [])
            if len(lst) < 5 and where not in lst:
                lst.append(where)

        for e in self.quests.get("event_quests", []):
            ev = str(e.get("event") or "")
            cs = str(e.get("case") or "").lower()
            for st in e.get("stages", []):
                prefix = f"{ev}/{cs}/{st.get('stage')}[{st.get('diff')}]"
                for d in st.get("drops", []):
                    report(str(d.get("item") or ""), f"{prefix}/drops")
            for d in e.get("box_pool", []):
                report(str(d.get("item") or ""), f"{ev}/{cs}/box_pool")
            for ex in e.get("exchanges", []) or []:
                for opt in (ex.get("options") or []):
                    report(str(opt or ""), f"{ev}/{cs}/exchanges")

        return unknown

# ---------------------------------------------------------------------------
# CeTracker — 예장(CE) 상태 관리
# ---------------------------------------------------------------------------

@dataclass
class CeState:
    drops_acc: float = 0.0
    bonus: int = CE_BASE_BONUS

@dataclass
class CeUpdateResult:
    drop_rate:     float
    new_drops:     float
    new_copies:    int
    gain_int:      int
    bonus_inc:     int
    new_bonus:     int
    rem_to_next:   float

class CeTracker:
    def __init__(self):
        self._states: Dict[str, CeState] = {}

    def init(self, event: str):
        if event not in self._states:
            self._states[event] = CeState()

    def bonus(self, event: str) -> int:
        st = self._states.get(event)
        return st.bonus if st else CE_BASE_BONUS

    def state(self, event: str) -> CeState:
        return self._states.get(event, CeState())

    def update(self, event: str, p: float) -> CeUpdateResult:
        st = self._states.get(event)
        if not st:
            return CeUpdateResult(p, 0.0, 0, 0, 0, CE_BASE_BONUS, 0.0)
        prev = st.drops_acc
        new  = prev + p
        gain_int  = max(0, int(new // 1) - int(prev // 1))
        prev_steps = int(prev // CE_COPIES_PER_PLUS)
        new_steps  = int(new  // CE_COPIES_PER_PLUS)
        bonus_inc  = max(0, new_steps - prev_steps)
        new_bonus  = min(CE_MAX_BONUS, CE_BASE_BONUS + new_steps)
        st.drops_acc = new
        st.bonus     = new_bonus
        rem = max(0.0, (new_steps + 1) * CE_COPIES_PER_PLUS - new)
        return CeUpdateResult(p, new, int(new // 1), gain_int, bonus_inc, new_bonus, rem)

# ---------------------------------------------------------------------------
# EfficiencyEngine — AP 효율 계산
# ---------------------------------------------------------------------------

class EfficiencyEngine:
    """
    스테이지/이벤트 타입별 AP 효율 계산.
    MaterialStore와 EventData를 읽기만 하고 상태를 변경하지 않는다.
    """
    def __init__(self, store: MaterialStore, edata: EventData, ce: CeTracker,
                 ap_cost: float, prefer_diff: Optional[str] = None):
        self.store = store
        self.edata = edata
        self.ce    = ce
        self.ap_cost = ap_cost
        self.prefer_diff = prefer_diff

    # --- 재료 AP 합산 ---

    def _stage_ap(self, drops: List[dict]) -> float:
        return sum(self.store.ap_value(str(d.get("item") or "")) * float(d.get("rate") or 0.0)
                   for d in drops)

    def _stage_apple_refund(self, drops: List[dict]) -> float:
        """스테이지 드랍에서 사과 환급 AP를 계산한다. (드랍률 × 사과 AP값)"""
        apple_ap = {k: APPLE_AP[v] for k, v in APPLE_NAME_TO_POOL.items()}
        return sum(apple_ap[d.get("item")] * float(d.get("rate") or 0.0)
                   for d in drops if d.get("item") in apple_ap)

    def _rarity_list_ap(self, lst: List[dict]) -> float:
        if not lst:
            return 0.0
        has_qty  = any("qty_per_box" in (x or {}) for x in lst)
        has_rate = any("rate" in (x or {}) for x in lst)
        n = len(lst)
        total = 0.0
        for x in lst:
            item = str((x or {}).get("item") or "")
            if not item:
                continue
            if has_qty:
                qty = float((x or {}).get("qty_per_box") or 0.0)
            elif has_rate:
                qty = float((x or {}).get("rate") or 0.0)
            else:
                qty = 1.0 / n if n > 0 else 0.0
            total += self.store.ap_value(item) * qty
        return total

    # --- 교환티켓 배분 ---

    def _allocate_tokens(self, token_qty: float, per_token: float,
                         options: List[str]) -> List[Tuple[str, float]]:
        candidates = [
            (it, self.store.ap_value(it), self.store.need(it))
            for it in options
            if self.store.need(it) > 0
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        remain = token_qty * per_token
        result = []
        for it, _, need in candidates:
            if remain <= 0:
                break
            take = min(remain, need)
            if take > 0:
                result.append((it, take))
                remain -= take
        return result

    # --- 룰렛 상자 AP 가치 ---

    def roulette_box_value(self, event: str) -> Tuple[float, float, List[dict]]:
        """(상자AP가치, 사과환급AP, 상세내역)"""
        blk = self.edata.event_block(event, "roulette")
        if not blk:
            return 0.0, 0.0, []
        apple_ap = {k: APPLE_AP[v] for k, v in APPLE_NAME_TO_POOL.items()}
        per_box = 0.0
        refund  = 0.0
        details = []
        token_name = None
        token_rate = 0.0
        for d in blk.get("box_pool", []):
            item = str(d.get("item") or "")
            rate = float(d.get("rate") or 0.0)
            if not item:
                continue
            if item in apple_ap:
                refund += rate * apple_ap[item]
            elif item == "교환티켓":
                token_name, token_rate = item, rate
            else:
                apv = self.store.ap_value(item)
                val = apv * rate
                per_box += val
                details.append({"kind": "drop", "item": item, "qty_per_box": rate,
                                 "ap_per_item": apv, "ap_value": r2(val)})
        if token_name and token_rate > 0:
            for ex in blk.get("exchanges", []):
                if str(ex.get("token")) != token_name:
                    continue
                per_token = float(ex.get("per_token") or 1.0)
                options   = [str(x) for x in ex.get("options", []) if x]
                for it, qty in self._allocate_tokens(token_rate, per_token, options):
                    apv = self.store.ap_value(it)
                    val = apv * qty
                    per_box += val
                    details.append({"kind": "exchange", "item": it, "qty": r2(qty),
                                    "ap_per_item": apv, "ap_value": r2(val)})
        return r2(per_box), r2(refund), details

    # --- 최적 스테이지 선택 ---

    def best_roulette(self, event: str) -> Optional[dict]:
        blk = self.edata.event_block(event, "roulette")
        if not blk:
            return None
        need_tk      = self.edata.need_tickets(event)
        box_val, apple_refund, box_details = self.roulette_box_value(event)
        ce_bonus     = self.ce.bonus(event)
        candidates   = []
        for st in blk.get("stages", []):
            tkt  = float((st.get("tickets") or {}).get("base", 0.0)) + \
                   float((st.get("tickets") or {}).get("per_ce", 0.0)) * ce_bonus
            sv   = self._stage_ap(st.get("drops", []))
            s_eff = sv / self.ap_cost if self.ap_cost > 0 else 0.0
            if need_tk > 0 and tkt > 0:
                net_ap = (need_tk / tkt) * self.ap_cost - apple_refund
            else:
                net_ap = 0.0
            if net_ap <= 0:
                r_eff = total_eff = float("inf")
            else:
                r_eff = box_val / net_ap
                total_eff = s_eff + r_eff
            candidates.append({
                "event": event, "case": "roulette",
                "stage": st.get("stage"), "diff": st.get("diff"),
                "tickets_per_run": r2(tkt), "stage_val_per_run": r2(sv),
                "stage_eff_per_ap": r2(s_eff),
                "roulette_eff": r_eff if r_eff == float("inf") else r2(r_eff),
                "total_eff": total_eff if total_eff == float("inf") else r2(total_eff),
                "need_tickets": need_tk, "per_box_value": r2(box_val),
                "apple_refund_ap": r2(apple_refund), "ce_bonus": ce_bonus,
                "stage_drops": st.get("drops", []), "box_details": box_details,
            })
        if not candidates:
            return None
        return max(candidates, key=lambda x: x["total_eff"] if x["total_eff"] == float("inf") else float(x["total_eff"]))

    def best_box(self, event: str) -> Optional[dict]:
        blk = self.edata.event_block(event, "box")
        if not blk:
            return None
        candidates = []
        for st in blk.get("stages", []):
            sv       = self._stage_ap(st.get("drops", []))
            refund   = self._stage_apple_refund(st.get("drops", []))
            net_cost = self.ap_cost - refund
            box = st.get("box") or {}
            bg = float((box.get("gold")   or {}).get("base", 0.0))
            bs = float((box.get("silver") or {}).get("base", 0.0))
            bb = float((box.get("bronze") or {}).get("base", 0.0))
            contents = self.edata.box_contents(st, blk)
            if not contents:
                contents = self.edata.box_pool_grouped(event, "box", self.store)
            ag = self._rarity_list_ap(contents.get("gold", []))
            as_ = self._rarity_list_ap(contents.get("silver", []))
            ab = self._rarity_list_ap(contents.get("bronze", []))
            box_val = ag * bg + as_ * bs + ab * bb
            total_eff = (sv + box_val) / net_cost if net_cost > 0 else 0.0
            candidates.append({
                "event": event, "case": "box",
                "stage": st.get("stage"), "diff": st.get("diff"),
                "stage_val_per_run": r2(sv), "box_val_per_run": r2(box_val),
                "total_eff": r2(total_eff),
                "details": {
                    "base": {"gold": bg, "silver": bs, "bronze": bb},
                    "sum_ap": {"gold": r2(ag), "silver": r2(as_), "bronze": r2(ab)},
                    "drops": st.get("drops", []),
                },
            })
        if not candidates:
            return None
        pool = ([c for c in candidates if str(c.get("diff") or "") == str(self.prefer_diff)]
                if self.prefer_diff else candidates)
        return max(pool or candidates, key=lambda x: float(x["total_eff"]))

    def best_raid(self, event: str, runs_done: Optional[Dict[str, int]] = None) -> Optional[dict]:
        blk = self.edata.event_block(event, "raid")
        if not blk:
            return None
        candidates = []
        for st in blk.get("stages", []):
            stage = st.get("stage")
            diff  = st.get("diff")
            max_runs = st.get("max_runs")  # None이면 무제한
            # 횟수 제한 소진 시 후보에서 제외
            if max_runs is not None and runs_done is not None:
                done = runs_done.get(f"{event}|raid|{stage}|{diff}", 0)
                if done >= max_runs:
                    continue
            sv       = self._stage_ap(st.get("drops", []))
            refund   = self._stage_apple_refund(st.get("drops", []))
            net_cost = self.ap_cost - refund
            eff = sv / net_cost if net_cost > 0 else 0.0
            candidates.append({
                "event": event, "case": "raid",
                "stage": stage, "diff": diff,
                "stage_val_per_run": r2(sv), "total_eff": r2(eff),
                "apple_refund_ap": r2(refund),
                "max_runs": max_runs,
            })
        if not candidates:
            return None
        pool = ([c for c in candidates if str(c.get("diff") or "") == str(self.prefer_diff)]
                if self.prefer_diff else candidates)
        return max(pool or candidates, key=lambda x: float(x["total_eff"]))

    def compute_all_bests(self, targets: List[Tuple[str, str]],
                          runs_done: Optional[Dict[str, int]] = None) -> List[dict]:
        results = []
        for ev, cs in targets:
            if cs == "raid":
                best = self.best_raid(ev, runs_done)
            else:
                best = {"roulette": self.best_roulette,
                        "box":      self.best_box}.get(cs, lambda _: None)(ev)
            if best:
                results.append(best)
        return results

    @staticmethod
    def pick_global_best(bests: List[dict]) -> Optional[dict]:
        if not bests:
            return None
        return max(bests, key=lambda x: (
            float("inf") if x.get("total_eff") == float("inf") else float(x.get("total_eff", 0))
        ))

# ---------------------------------------------------------------------------
# Simulator — 메인 루프 + 로그 출력
# ---------------------------------------------------------------------------

class Simulator:
    def __init__(self, store: MaterialStore, edata: EventData, ce: CeTracker,
                 engine: EfficiencyEngine, logger: Logger, ts: TableStyle,
                 ap_cost: float):
        self.store  = store
        self.edata  = edata
        self.ce     = ce
        self.engine = engine
        self.log    = logger
        self.ts     = ts
        self.ap_cost = ap_cost
        self.w      = logger.write

    # --- 표 출력 헬퍼 ---

    @staticmethod
    def _progress_bar(got: float, target: float, width: int = 10) -> str:
        """ASCII 진행률 바: [████░░░░░░] 75%"""
        if target <= 0:
            return f"[{'█'*width}] 100%"
        ratio = min(got / target, 1.0)
        filled = round(ratio * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {ratio*100:.0f}%"

    def _print_eff_table(self, bests: List[dict], current_key: str = ""):
        """효율 상위 N개만 출력, 현재 스테이지는 ▶ 표시"""
        sorted_bests = sorted(bests,
                              key=lambda b: float("inf") if b.get("total_eff") == float("inf")
                              else float(b.get("total_eff", 0)),
                              reverse=True)
        TOP_N = 6
        show = sorted_bests[:TOP_N]
        current_in_top = any(
            f"{b['event']}|{b['case']}|{b['stage']}|{b['diff']}" == current_key
            for b in show
        )
        if not current_in_top and current_key:
            for b in sorted_bests[TOP_N:]:
                if f"{b['event']}|{b['case']}|{b['stage']}|{b['diff']}" == current_key:
                    show.append(b)
                    break
        rows = []
        for b in show:
            bk = f"{b['event']}|{b['case']}|{b['stage']}|{b['diff']}"
            marker = "▶" if bk == current_key else " "
            rows.append([
                f"{marker} {b.get('event')}",
                f"{b.get('stage')} [{b.get('diff')}]",
                f2s(b.get("total_eff", 0))
            ])
        if len(sorted_bests) > len(show):
            rows.append([f"  … 외 {len(sorted_bests)-len(show)}개", "", ""])
        print_table(self.w, ["이벤트", "스테이지[난이도]", "AP효율"], rows,
                    aligns=["left", "left", "right"], ts=self.ts)

    def _print_final_summary(self, initial_need):
        final_need = self.store.snapshot_need()
        rows_remain = []
        rows_done   = []
        for name in set(initial_need) | set(self.store.cum_gained) | set(final_need):
            rec    = self.store.record(name)
            target = float(initial_need.get(name, 0.0))
            got    = float(self.store.cum_gained.get(name, 0.0))
            remain = float(final_need.get(name, max(0.0, target - got)))
            # 목표가 없고(0) 획득만 있는 순수 부산물 제외
            if target == 0.0:
                continue
            # 시작 시 이미 -인 재료(초과 보유) → 달성으로 분류
            if target <= 0:
                rows_done.append([name, target, got, remain])
            elif remain > 0:
                rows_remain.append([name, target, got, remain])
            else:
                rows_done.append([name, target, got, remain])
        rows_remain.sort(key=lambda r: self.store.sort_key(r[0]))
        rows_done.sort(key=lambda r: self.store.sort_key(r[0]))

        if not rows_remain and not rows_done:
            self.w("(최종 재료 현황: 출력할 항목 없음)")
            return

        # 미달 항목
        if rows_remain:
            self.w("### 미달 재료")
            fmt = [[r[0], f0(r[1]), f0(r[2]), f0(r[3]),
                    self._progress_bar(r[2], r[1])] for r in rows_remain]
            print_table(self.w, ["재료", "목표", "획득", "부족", "진행률"], fmt,
                        aligns=["left", "right", "right", "right", "left"], ts=self.ts)

        # 달성 항목 (-인 재료 포함)
        if rows_done:
            self.w("")
            self.w("### 달성 재료")
            fmt = [[r[0], f0(r[1]), f0(r[2]),
                    self._progress_bar(r[2], r[1])] for r in rows_done]
            print_table(self.w, ["재료", "목표", "획득", "진행률"], fmt,
                        aligns=["left", "right", "right", "left"], ts=self.ts)

    # --- 획득 반영 ---

    def _apply_drops(self, drops: List[dict]):
        for d in drops:
            self.store.apply_gain(str(d.get("item") or ""), float(d.get("rate") or 0.0))

    def _open_roulette_boxes(self, event: str, n: int):
        blk = self.edata.event_block(event, "roulette")
        if not blk or n <= 0:
            return
        token_name = None
        token_rate = 0.0
        for d in blk.get("box_pool", []):
            item = str(d.get("item") or "")
            rate = float(d.get("rate") or 0.0)
            if not item or item in APPLE_ITEM_NAMES:
                continue
            if item == "교환티켓":
                token_name, token_rate = item, rate
            else:
                self.store.apply_gain(item, rate * n)
        if token_name and token_rate > 0:
            total_tokens = token_rate * n
            for ex in blk.get("exchanges", []):
                if str(ex.get("token")) != token_name:
                    continue
                per_token = float(ex.get("per_token") or 1.0)
                options   = [str(x) for x in ex.get("options", []) if x]
                for it, qty in self.engine._allocate_tokens(total_tokens, per_token, options):
                    self.store.apply_gain(it, qty)

    def _apply_box_contents(self, event: str, stage: dict, event_blk: dict):
        contents = self.edata.box_contents(stage, event_blk)
        if not contents:
            contents = self.edata.box_pool_grouped(event, "box", self.store)
        if not contents:
            return
        box = stage.get("box") or {}
        bases = {
            "gold":   float((box.get("gold")   or {}).get("base", 0.0)),
            "silver": float((box.get("silver") or {}).get("base", 0.0)),
            "bronze": float((box.get("bronze") or {}).get("base", 0.0)),
        }
        for rar, lst in contents.items():
            base = bases.get(rar, 0.0)
            if not lst or base <= 0:
                continue
            has_qty  = any("qty_per_box" in (x or {}) for x in lst)
            has_rate = any("rate" in (x or {}) for x in lst)
            n = len(lst)
            for x in lst:
                item = str((x or {}).get("item") or "")
                if not item:
                    continue
                if has_qty:
                    per = float((x or {}).get("qty_per_box") or 0.0)
                elif has_rate:
                    per = float((x or {}).get("rate") or 0.0)
                else:
                    per = 1.0 / n if n > 0 else 0.0
                self.store.apply_gain(item, per * base)

    # --- 로그 출력 ---

    def _log_stage_header(self, run_idx, ev, cs, stage, diff, best, is_status_log, has_ce_bonus):
        eff_str = f2s(best.get('total_eff', 0))
        case_label = {"roulette": "룰렛", "box": "박스", "raid": "레이드"}.get(cs, cs)
        if is_status_log:
            st = self.ce.state(ev)
            bonus_str = f"  (보너스={st.bonus})" if has_ce_bonus else ""
            self.w(f"[Run {run_idx:,}] 상태 기록 {self.ts.arrow} "
                   f"{ev} · {stage}/{diff} [{case_label}]{bonus_str}")
        else:
            self.w(f"")
            self.w(f"{'─'*60}")
            self.w(f"[Run {run_idx:,}] 스테이지 전환 {self.ts.arrow}  {ev}  [{stage}/{diff}]  "
                   f"| {case_label} | AP효율 {eff_str}")
            self.w(f"{'─'*60}")

    def _log_need_zeros(self, run_idx, need_zero_items):
        for name, b, total in need_zero_items:
            pct = 100.0
            self.w(f"  {self.ts.bullet} ✓ 재료 충족: {name}"
                   f"  누적 {f0(total)}개  (목표 달성 {pct:.0f}%)")

    def _log_ce(self, ci: CeUpdateResult):
        if ci.gain_int > 0:
            self.w(f"  {self.ts.bullet} 예장 드랍 +{ci.gain_int}장"
                   f" → 현재 {ci.new_copies}장 (기대 누계 {f2(ci.new_drops)}장)")
        if ci.bonus_inc > 0:
            self.w(f"    보너스 +{ci.bonus_inc} {self.ts.arrow} {ci.new_bonus}"
                   f"  (다음 +1까지 {f2(ci.rem_to_next)}장)")

    def _log_roulette_status(self, ev, tickets_acc, last_tpr, refund, ap_pool):
        need_tk = self.edata.need_tickets(ev)
        st  = self.ce.state(ev)
        tk_pct = min(tickets_acc / need_tk * 100, 100) if need_tk > 0 else 0
        suffix = ""
        if st.bonus < CE_MAX_BONUS:
            rem = max(0.0, CE_COPIES_PER_PLUS - (st.drops_acc % CE_COPIES_PER_PLUS))
            suffix = f", 다음+1까지 {f2(rem)}장"
        self.w(f"  {self.ts.bullet} 룰렛 티켓 {f0(tickets_acc)}/{f0(need_tk)}"
               f" ({tk_pct:.0f}%) | +{f2(last_tpr)}/판"
               f" | 환급 +{f2(refund)} AP"
               f" | 예장 보너스 {st.bonus}{suffix}")

    # --- 메인 루프 ---

    def run(self, targets: List[Tuple[str, str]], total_ap: float) -> List[Tuple]:
        self.w(f"AP 풀: {f2(total_ap)}  | AP/판={f2(self.ap_cost)}"
               f"  | 표 포맷={self.ts.fmt}, 스타일={self.ts.style}")
        self.w(f"# 룰렛 보너스: 시작 {CE_BASE_BONUS}, {CE_COPIES_PER_PLUS}장당 +1, 최대 {CE_MAX_BONUS}")

        if not targets:
            self.w("대상 이벤트 없음.")
            return []

        initial_need = self.store.snapshot_need()
        # ce_drop_rate가 있는 이벤트는 케이스 무관하게 CE 추적 등록
        for ev, cs in targets:
            blk = self.edata.event_block(ev, cs)
            if not blk:
                continue
            has_ce = any(
                self.edata.ce_drop_rate(st, blk) > 0
                for st in blk.get("stages", [])
            )
            if has_ce:
                self.ce.init(ev)

        ap_pool = total_ap
        tickets_acc:    Dict[str, float] = {}
        need_tk_cache:  Dict[str, float] = {}
        refund_cache:   Dict[str, float] = {}
        runs_done:      Dict[str, int]   = {}   # key별 누적 실행 횟수 (max_runs 제한용)
        run_idx = 0
        last_key: Optional[str] = None
        sessions: List[Tuple[str, str, str, int]] = []
        prev_sess: Optional[Tuple] = None

        while ap_pool >= self.ap_cost:
            run_idx += 1
            self.store.reset_run_gains()

            bests  = self.engine.compute_all_bests(targets, runs_done)
            best   = self.engine.pick_global_best(bests)
            if not best:
                self.w("※ 선택할 스테이지 없음 → 중단")
                break

            ev    = best["event"]
            cs    = best["case"]
            stage = best["stage"]
            diff  = best["diff"]
            key   = f"{ev}|{cs}|{stage}|{diff}"
            stage_changed = (last_key is None or key != last_key)

            # 세션 추적
            sess_key = (ev, stage, diff)
            if prev_sess != sess_key:
                sessions.append((ev, stage, diff, 1))
                prev_sess = sess_key
            else:
                n, s, d, c = sessions[-1]
                sessions[-1] = (n, s, d, c + 1)

            ap_pool -= self.ap_cost
            runs_done[key] = runs_done.get(key, 0) + 1   # 횟수 누적
            event_blk = self.edata.event_block(ev, cs)
            stage_def = self.edata.stage_def(ev, cs, stage, str(diff))
            if not stage_def:
                last_key = key
                continue

            need_before = self.store.snapshot_need()
            self._apply_drops(stage_def.get("drops", []))

            # 스테이지 드랍 사과 → ap_pool 실제 환급 (레이드/박스)
            if cs in ("raid", "box"):
                stage_apple_ap = self.engine._stage_apple_refund(stage_def.get("drops", []))
                if stage_apple_ap > 0:
                    ap_pool += stage_apple_ap

            opened = 0
            refund_gain = 0.0
            last_tpr    = 0.0

            if cs == "roulette":
                ce_bonus = self.ce.bonus(ev)
                t_def    = stage_def.get("tickets") or {}
                last_tpr = float(t_def.get("base", 0.0)) + float(t_def.get("per_ce", 0.0)) * ce_bonus
                tickets_acc[ev] = tickets_acc.get(ev, 0.0) + last_tpr
                need_tk_cache.setdefault(ev, self.edata.need_tickets(ev))
                if ev not in refund_cache:
                    _, ref, _ = self.engine.roulette_box_value(ev)
                    refund_cache[ev] = float(ref or 0.0)
                need_tk = need_tk_cache[ev]
                if need_tk > 0 and tickets_acc[ev] >= need_tk:
                    opened = int(tickets_acc[ev] // need_tk)
                    tickets_acc[ev] -= opened * need_tk
                    self._open_roulette_boxes(ev, opened)
                    refund_gain = opened * refund_cache[ev]
                    if refund_gain > 0:
                        ap_pool += refund_gain

            elif cs == "box" and event_blk:
                self._apply_box_contents(ev, stage_def, event_blk)

            need_after = self.store.snapshot_need()
            need_zeros = [
                (name, b, self.store.cum_gained.get(name, 0.0))
                for name, b in need_before.items()
                if b > 0 and need_after.get(name, 0.0) == 0
            ]

            # CE 갱신
            ce_trigger: Optional[CeUpdateResult] = None
            p = self.edata.ce_drop_rate(stage_def, event_blk)
            if p > 0:
                ci = self.ce.update(ev, p)
                if ci.new_bonus < CE_MAX_BONUS and (ci.gain_int > 0 or ci.bonus_inc > 0):
                    ce_trigger = ci

            should_log  = stage_changed or need_zeros or ce_trigger
            ce_only     = bool(ce_trigger) and not stage_changed and not need_zeros

            if should_log:
                self._log_stage_header(run_idx, ev, cs, stage, diff, best,
                                       not stage_changed, ce_trigger is not None)
                if ce_only:
                    self.w(""); self.w("")
                    last_key = key
                    continue
                self._log_need_zeros(run_idx, need_zeros)
                if ce_trigger and not ce_only:
                    self._log_ce(ce_trigger)
                if cs == "roulette":
                    self._log_roulette_status(ev, tickets_acc.get(ev, 0.0),
                                              last_tpr, refund_gain, ap_pool)
                # 효율 표는 스테이지 전환 시에만 출력
                if stage_changed:
                    bests_after = self.engine.compute_all_bests(targets, runs_done)
                    self.w("")
                    self._print_eff_table(bests_after, current_key=key)
                self.w("")

            last_key = key

        total_runs = run_idx
        ap_used = total_ap - ap_pool
        self.w("")
        self.w(f"{'═'*60}")
        self.w(f"# 시뮬레이션 종료")
        self.w(f"  총 {total_runs:,}판 | AP 소모 {f0(ap_used)} / {f0(total_ap)}"
               f" | 잔여 {f2(ap_pool)} AP")
        self.w(f"{'═'*60}")
        self.w("")
        self.w("## 최종 재료 현황")
        self._print_final_summary(initial_need)
        return sessions

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FGO 이벤트 주회 시뮬레이터")
    parser.add_argument("--materials", default="materials.json")
    parser.add_argument("--quests",    default="event_quests.json")
    parser.add_argument("--event",     default=None,   help="특정 이벤트만 필터")
    parser.add_argument("--diff",      default=None,   help="선호 난이도")
    parser.add_argument("--ap-cost",   type=float, default=40.0)
    parser.add_argument("--log",       default="run_log.txt")
    parser.add_argument("--table-format",  choices=["text","md","csv"], default="text")
    parser.add_argument("--table-style",   choices=["box","ascii"],     default="box")
    parser.add_argument("--ascii-arrow",   action="store_true")
    parser.add_argument("--ascii-bullet",  action="store_true")
    args = parser.parse_args()

    ts = TableStyle(
        fmt   = args.table_format,
        style = args.table_style,
        arrow = "->" if args.ascii_arrow  else "→",
        bullet= "-"  if args.ascii_bullet else "·",
    )

    # 데이터 로드
    materials_json = load_json(args.materials)
    quests_def     = load_json(args.quests)

    store  = MaterialStore(materials_json)
    edata  = EventData(quests_def)
    ce     = CeTracker()
    engine = EfficiencyEngine(store, edata, ce, ap_cost=args.ap_cost, prefer_diff=args.diff)
    logger = Logger(args.log, tee=False)

    # 아이템명 검증
    unknown = edata.validate_names(store)
    if unknown:
        logger.write("## 경고: materials.json에 없는 아이템명이 참조되었습니다")
        known_sorted = sorted(store.all_names())
        for nm in sorted(unknown):
            locs = ", ".join(unknown[nm])
            sug  = difflib.get_close_matches(nm, known_sorted, n=3, cutoff=0.6)
            sug_str = (" | 유사: " + ", ".join(sug)) if sug else ""
            logger.write(f"- {nm}  | 위치 예시: {locs}{sug_str}")
        logger.write("")

    # AP 풀
    total_ap = sum(APPLE_COUNTS.get(k, 0) * APPLE_AP[k] for k in APPLE_AP) + float(NATURAL_AP)

    sim      = Simulator(store, edata, ce, engine, logger, ts, ap_cost=args.ap_cost)
    targets  = edata.list_targets(args.event)
    sessions = sim.run(targets, total_ap)

    logger.close()

    # 상단 요약 삽입
    total_runs = sum(cnt for _, _, _, cnt in sessions)
    header = [
        f"## 주회 플랜  ({total_runs:,}판 / AP {f0(total_ap)} 소모 예상)",
        f"{'─'*50}",
    ]
    body = []
    for i, (name, stg, d, cnt) in enumerate(sessions):
        ap_est = int(cnt * args.ap_cost)
        body.append(f"  {i+1:>2}. {name} [{stg}/{d}]  {cnt:,}판  (~AP {ap_est:,})")
    footer = [f"{'─'*50}", ""]
    for ln in header + body + footer:
        print(ln)
    logger.prepend(header + body + footer + ["## 상세 로그 ↓", ""])
    print(f"[로그 저장 완료] {logger.filepath}")


if __name__ == "__main__":
    main()
