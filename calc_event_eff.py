# -*- coding: utf-8 -*-
"""
FGO 이벤트 주회 시뮬레이터 (Pro v2.5 — 전체 주석/정렬/정밀 출력/표 렌더 개선 + 아이템명 검증)
=============================================================================================

변경 핵심 (v2.5)
----------------
- **아이템명 검증 추가**: event_quests.json / event_items.json / rarity_map 등에서
  참조하는 아이템명이 materials.json의 재료명과 **매칭되지 않으면** 로그 상단에
  경고 목록으로 알려줍니다(위치 예시 + 유사 이름 제안).
- 표 렌더 안정화(모호폭 처리/ASCII 스타일/마크다운·CSV 지원)는 v2.4와 동일.
- 표기 규칙: 일반 로그는 **소수점 2자리**, 최종 재료 현황은 **정수**.

입력 파일 스키마(요지)
----------------------
- materials.json
  - materials: [{item, tier, ap_per_item, 보유량?, 목표량?, lack?, use?}, ...]  ← **정렬 기준**
    · 기본은 보유량/목표량으로 부족(lack)을 계산하고, 키가 없을 땐 기존 have/need 또는 lack를 호환 지원
  - rarity_map: {gold:[..], silver:[..], bronze:[..]}
- event_quests.json
  - event_quests: [
      {event, case ∈ {"roulette","box","raid"},
       stages: [
         {stage, diff, drops: [{item, rate}, ...],
          tickets?: {base, per_ce}, ce_drop_rate?,
          box?: {gold/silver/bronze:{base}, items/contents?}
         }, ...
       ],
       ce_drop_rate?, box_contents?
      }, ...
    ]
- event_items.json
  - event_items: [{event, case, need_tickets?, drops, exchanges?}, ...]
    (또는 동일 키를 event_quests로 갖는 변형을 헨들링)
"""

import json
import argparse
import unicodedata
import difflib
from typing import Dict, List, Tuple, Optional

# =============================================================================
# 상수/전역 설정
# =============================================================================

APPLE_AP_BY_POOL = {"gold": 145.0, "silver": 73.0, "blue": 40.0, "copper": 10.0}
APPLE_NAME_TO_POOL = {"금사과": "gold", "은사과": "silver", "청사과": "blue", "동사과": "copper"}
APPLE_ITEM_NAMES = set(APPLE_NAME_TO_POOL.keys())
APPLE_COUNTS = {"gold": 1000, "silver": 295, "blue": 2000, "copper": 502}
NATURAL_AP = 0.0

RARITY_MAP: Optional[Dict[str, set]] = None

# 정렬 인덱스 — materials.json의 materials 배열 "등장 순서" 기반
ITEM_SORT_INDEX: Dict[str, int] = {}   # 재료명 → 등장순 인덱스
TIER_ORDER_INDEX: Dict[str, int] = {}  # tier명 → 등장순 인덱스
ITEM_TO_TIER: Dict[str, str] = {}      # 재료명 → tier

# 룰렛 예장(CE) 관련
CE_BASE_BONUS = 7
CE_MAX_BONUS = 12
CE_COPIES_PER_PLUS = 4
CE_STATE: Dict[str, dict] = {}

# 선택 기능: 이벤트명 부분 포함 시 1판당 고정 수익
SPECIAL_PER_RUN_YIELDS = {}

# 현재 판(run) 변동 기록
CURRENT_RUN_GAINS: Optional[Dict[str, float]] = None
_AGT_CALL_DEPTH = 0

# 출력/표 스타일
TABLE_FORMAT = "text"      # "text" | "md" | "csv"
TABLE_STYLE  = "box"       # 텍스트 표일 때: 'box' | 'ascii'
AMBIGUOUS_WIDE = False     # 모호폭(A) 폭=2로 처리할지 여부
ARROW  = "→"               # 호환 모드에서 '->'로 변경 가능
BULLET = "·"               # 호환 모드에서 '-'로 변경 가능

BOX_CHARS   = {"top":("┌","┬","┐"), "mid":("├","┼","┤"), "bot":("└","┴","┘"), "h":"─", "v":"│"}
ASCII_CHARS = {"top":("+","+","+"), "mid":("+","+","+"), "bot":("+","+","+"), "h":"-", "v":"|"}

# =============================================================================
# 숫자 포맷 & 공통 유틸
# =============================================================================

def r2(x: float) -> float:
    """float → 소수점 2자리 반올림 숫자"""
    return round(float(x), 2)

def f2(x: float) -> str:
    """float → '1,234.56' 포맷 문자열"""
    return f"{float(x):,.2f}"

def f2s(x) -> str:
    """float → '∞' 처리 포함 2자리 포맷 문자열"""
    try:
        if x == float("inf") or str(x).lower() == "inf":
            return "∞"
        return f2(float(x))
    except Exception:
        return str(x)

def f0(x) -> str:
    """float → 정수 반올림 '1,234' 포맷 문자열"""
    try:
        return f"{int(round(float(x))):,}"
    except Exception:
        return str(x)

def load_json(path: str) -> dict:
    """경로에서 UTF-8 JSON 로드"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =============================================================================
# CJK 폭 인지 길이 계산
# =============================================================================

try:
    from wcwidth import wcswidth as _lib_wcswidth  # type: ignore
    def _display_width(s: str) -> int:
        """wcwidth 라이브러리가 있으면 사용"""
        return _lib_wcswidth(s)
except Exception:
    def _display_width(s: str) -> int:
        """동아시아 폭(F/W/A) 고려한 폭 계산(대체 구현)"""
        width = 0
        for ch in s:
            if unicodedata.combining(ch):
                continue
            ea = unicodedata.east_asian_width(ch)
            if ea in ("F", "W") or (ea == "A" and AMBIGUOUS_WIDE):
                width += 2
            else:
                width += 1
        return width

# =============================================================================
# materials.json → 인덱스/희귀도/정렬 초기화
# =============================================================================

def to_material_index(materials_json: dict) -> Dict[str, dict]:
    """materials 배열 → {이름:{ap,lack,use}} 인덱스 구성
    변경 사항:
    - 파일에 lack/use 키가 없더라도 동작하도록 호환 처리
    - 부족(lack)은 기본적으로 max(목표-보유, 0)으로 계산
      · 한국어 키: 보유량/목표량
      · 영문 키: have/need (이전 버전 호환)
      · 위 키가 모두 없을 때만 파일의 lack 값을 그대로 사용(레거시)
    - use 키가 없으면 True로 간주
    """
    index: Dict[str, dict] = {}
    for m in materials_json.get("materials", []):
        name = str(m.get("item") or "").strip()
        if not name:
            continue

        # AP/개
        ap_raw = m.get("ap_per_item")
        ap_val = float(ap_raw) if ap_raw is not None else 0.0

        # 새 스키마(한국어) 또는 구 스키마(영문) 지원
        have = m.get("보유량")
        need = m.get("목표량")
        if have is None and need is None:
            have = m.get("have")
            need = m.get("need")

        # 부족 계산(기본)
        lack_calc = None
        try:
            if have is not None or need is not None:
                hv = float(have or 0.0)
                nd = float(need or 0.0)
                lack_calc = max(nd - hv, 0.0)
        except Exception:
            lack_calc = None

        # 레거시 lack 값(최후의 수단)
        if lack_calc is None:
            lack_raw = m.get("lack")
            lack_calc = float(lack_raw) if lack_raw is not None else 0.0

        # use 기본 True (문자/불리언 모두 허용)
        use_val = m.get("use")
        if isinstance(use_val, str):
            use_flag = use_val.strip().upper() != "N"
        elif isinstance(use_val, (int, float)):
            use_flag = bool(use_val)
        elif isinstance(use_val, bool):
            use_flag = use_val
        else:
            use_flag = True  # 키가 없는 경우

        index[name] = {"ap": ap_val, "lack": lack_calc, "use": use_flag}
    return index

def init_rarity_map_from_materials(materials_json: dict) -> None:
    """rarity_map(gold/silver/bronze) 초기화"""
    global RARITY_MAP
    rm = materials_json.get("rarity_map")
    if isinstance(rm, dict):
        RARITY_MAP = {
            "gold":   set(str(x) for x in rm.get("gold", [])   if x),
            "silver": set(str(x) for x in rm.get("silver", []) if x),
            "bronze": set(str(x) for x in rm.get("bronze", []) if x),
        }
    else:
        RARITY_MAP = {"gold": set(), "silver": set(), "bronze": set()}

def init_sort_index_from_materials(materials_json: dict) -> None:
    """정렬용: materials 등장 순서/티어 순서 테이블 구성"""
    global ITEM_SORT_INDEX, TIER_ORDER_INDEX, ITEM_TO_TIER
    ITEM_SORT_INDEX = {}
    TIER_ORDER_INDEX = {}
    ITEM_TO_TIER = {}
    for i, m in enumerate(materials_json.get("materials", [])):
        name = str(m.get("item") or "").strip()
        tier = str(m.get("tier") or "").strip()
        if not name:
            continue
        if name not in ITEM_SORT_INDEX:
            ITEM_SORT_INDEX[name] = i
        ITEM_TO_TIER[name] = tier
        if tier and tier not in TIER_ORDER_INDEX:
            TIER_ORDER_INDEX[tier] = len(TIER_ORDER_INDEX)

def default_rarity_map() -> Dict[str, set]:
    return RARITY_MAP if RARITY_MAP is not None else {"gold": set(), "silver": set(), "bronze": set()}

def classify_rarity(item: str, rarity_map: Dict[str, set]) -> Optional[str]:
    """아이템을 rarity_map 기준으로 gold/silver/bronze 중 하나로 분류"""
    for rarity in ("gold", "silver", "bronze"):
        if item in rarity_map[rarity]:
            return rarity
    return None

def _material_sort_key(name: str):
    """정렬키: materials 등장순 → 티어순 → 이름"""
    idx = ITEM_SORT_INDEX.get(name, 10**9)
    tier = ITEM_TO_TIER.get(name, "")
    to = TIER_ORDER_INDEX.get(tier, 10**9)
    return (idx, to, name)

# =============================================================================
# 재료 상태 조회/갱신
# =============================================================================

def mat_rec(item: str, materials_index: Dict[str, dict]) -> Optional[dict]:
    return materials_index.get(item)

def mat_ap(item: str, materials_index: Dict[str, dict], respect_use_flag: bool=True) -> float:
    """아이템 1개당 AP가치(부족>0 & use=Y일 때만 유효)"""
    rec = mat_rec(item, materials_index)
    if not rec:
        return 0.0
    if respect_use_flag and not rec.get("use", True):
        return 0.0
    if float(rec.get("lack") or 0.0) <= 0.0:
        return 0.0
    return float(rec.get("ap") or 0.0)

def mat_lack(item: str, materials_index: Dict[str, dict]) -> float:
    rec = mat_rec(item, materials_index)
    return float(rec["lack"]) if rec else 0.0

def mat_usable(item: str, materials_index: Dict[str, dict]) -> bool:
    rec = mat_rec(item, materials_index)
    return (rec["use"] if rec is not None else True)

def snapshot_lack(materials_index: Dict[str, dict], respect_use_flag: bool=True) -> Dict[str, float]:
    """현재 부족량 스냅샷(정렬/표시용)"""
    return {
        name: float(rec.get("lack") or 0.0)
        for name, rec in materials_index.items()
        if not (respect_use_flag and not rec.get("use", True))
    }

# =============================================================================
# (신규) 아이템명 검증
# =============================================================================

def _iter_item_blocks(items_def: dict) -> List[dict]:
    if isinstance(items_def.get("event_items"), list):
        return items_def["event_items"]
    if isinstance(items_def.get("event_quests"), list):
        return items_def["event_quests"]
    return []

def _collect_from_box_contents(container: dict, where: str, report):
    """container: {"gold":[{item,..}], "silver":[..], "bronze":[..]}"""
    if not isinstance(container, dict):
        return
    for rar in ("gold", "silver", "bronze"):
        for ent in (container.get(rar) or []):
            item = str((ent or {}).get("item") or "").strip()
            if item:
                report(item, f"{where}/box:{rar}")

def validate_item_names(materials_index: Dict[str, dict], quests_def: dict, items_def: dict) -> Dict[str, List[str]]:
    """
    외부 JSON들이 참조하는 아이템명이 materials.json에 존재하는지 검증.

    Returns:
        {unknown_item: [위치 예시,...]}  (최대 5곳까지 샘플링)
    """
    known = set(materials_index.keys()) | APPLE_ITEM_NAMES | {"교환티켓"}
    unknown: Dict[str, List[str]] = {}

    def report(name: str, where: str):
        if not name:
            return
        n = name.strip()
        if n in known:
            return
        # 아직 모르는 이름이면 위치 누적(최대 5개)
        lst = unknown.setdefault(n, [])
        if len(lst) < 5 and where not in lst:
            lst.append(where)

    # 1) rarity_map 내 이름도 검사(오타 방지)
    rmap = default_rarity_map()
    for rar in ("gold", "silver", "bronze"):
        for nm in rmap.get(rar, []):
            if nm not in known:
                report(nm, f"materials.rarity_map.{rar}")

    # 2) quests_def 검사
    for e in quests_def.get("event_quests", []):
        ev = str(e.get("event") or "").strip()
        cs = str(e.get("case") or "").lower()
        # drops
        for st in e.get("stages", []):
            stg = str(st.get("stage") or "").strip()
            dff = str(st.get("diff") or "").strip()
            for d in st.get("drops", []):
                item = str(d.get("item") or "").strip()
                if item:
                    report(item, f"event_quests/{ev}/{cs}/{stg}[{dff}]/drops")
            # stage-level box contents(items/contents)
            box = st.get("box") or {}
            for key in ("items", "contents"):
                if key in box and isinstance(box[key], dict):
                    _collect_from_box_contents(box[key], f"event_quests/{ev}/{cs}/{stg}[{dff}]", report)
        # event-level box_contents
        if isinstance(e.get("box_contents"), dict):
            _collect_from_box_contents(e["box_contents"], f"event_quests/{ev}/{cs}", report)

    # quests_def top-level box_contents (있다면)
    if isinstance(quests_def.get("box_contents"), dict):
        _collect_from_box_contents(quests_def["box_contents"], "event_quests/top", report)

    # 3) items_def 검사
    for blk in _iter_item_blocks(items_def):
        ev = str(blk.get("event") or "").strip()
        cs = str(blk.get("case") or "").lower()
        for d in blk.get("drops", []):
            item = str(d.get("item") or "").strip()
            if item:
                report(item, f"event_items/{ev}/{cs}/drops")
        for ex in blk.get("exchanges", []) or []:
            for opt in (ex.get("options") or []):
                item = str(opt or "").strip()
                if item:
                    report(item, f"event_items/{ev}/{cs}/exchanges")
        # box case의 drops도 검사
        if cs == "box":
            for d in blk.get("drops", []):
                item = str(d.get("item") or "").strip()
                if item:
                    report(item, f"event_items/{ev}/{cs}/box_drops")

    return unknown

# =============================================================================
# 표 출력 (text/md/csv)
# =============================================================================

def _draw_line(widths, left="┌", mid="┬", right="┐", fill="─"):
    parts = [left]
    for i, w in enumerate(widths):
        parts.append(fill * (w + 2))
        parts.append(mid if i < len(widths) - 1 else right)
    return "".join(parts)

def _pad_cell(text: str, width: int, align: str) -> str:
    t = str(text)
    display = _display_width(t)
    pad = max(0, width - display)
    if align == "right":
        return " " + (" " * pad + t) + " "
    if align == "center":
        left = pad // 2; right = pad - left
        return " " + (" " * left + t + " " * right) + " "
    return " " + (t + " " * pad) + " "

def _print_table_text(write_line, headers, rows, aligns=None):
    if aligns is None:
        aligns = ["left"] * len(headers)
    chars = BOX_CHARS if TABLE_STYLE == "box" else ASCII_CHARS
    v = chars["v"]
    widths = []
    for col in range(len(headers)):
        maxw = _display_width(str(headers[col]))
        for r in rows:
            maxw = max(maxw, _display_width(str(r[col])))
        widths.append(maxw)
    write_line(_draw_line(widths, *chars["top"], fill=chars["h"]))
    write_line(v + v.join(_pad_cell(str(h), widths[i], "center") for i, h in enumerate(headers)) + v)
    write_line(_draw_line(widths, *chars["mid"], fill=chars["h"]))
    for r in rows:
        write_line(v + v.join(_pad_cell(str(r[i]), widths[i], aligns[i]) for i in range(len(headers))) + v)
    write_line(_draw_line(widths, *chars["bot"], fill=chars["h"]))

def _print_table_md(write_line, headers, rows, aligns=None):
    if aligns is None:
        aligns = ["left"] * len(headers)
    write_line("| " + " | ".join(str(h) for h in headers) + " |")
    align_map = {"left": ":---", "right": "---:", "center": ":---:"}
    write_line("| " + " | ".join(align_map.get(a, ":---") for a in aligns) + " |")
    for r in rows:
        write_line("| " + " | ".join(str(c) for c in r) + " |")

def _csv_escape(s: str) -> str:
    s = str(s)
    if any(ch in s for ch in [",", "\"", "\n", "\r"]):
        return "\"" + s.replace("\"", "\"\"") + "\""
    return s

def _print_table_csv(write_line, headers, rows, aligns=None):
    write_line(",".join(_csv_escape(h) for h in headers))
    for r in rows:
        write_line(",".join(_csv_escape(c) for c in r))

def _print_table(write_line, headers, rows, aligns=None):
    if TABLE_FORMAT == "md":
        _print_table_md(write_line, headers, rows, aligns)
    elif TABLE_FORMAT == "csv":
        _print_table_csv(write_line, headers, rows, aligns)
    else:
        _print_table_text(write_line, headers, rows, aligns)

def _print_gain_table(write_line, current_gains: Dict[str, float],
                      lack_before: Dict[str, float],
                      lack_after: Dict[str, float],
                      cum_gained: Dict[str, float]):
    if not current_gains:
        write_line("  (변동 없음)")
        return
    headers = ["항목", "+획득", "누적", "부족(전→후)"]
    names_sorted = sorted(current_gains.keys(), key=_material_sort_key)
    rows = []
    for name in names_sorted:
        gained = current_gains.get(name, 0.0)
        b = lack_before.get(name, 0.0)
        a = lack_after.get(name, 0.0)
        total = cum_gained.get(name, 0.0)
        rows.append([name, f2(gained), f2(total), f"{f2(b)} {ARROW} {f2(a)}"])
    _print_table(write_line, headers, rows, aligns=["left", "right", "right", "right"])

def _print_eff_table(write_line, best_list: List[dict], ap_cost_per_run: float):
    headers = ["이벤트", "스테이지[난이도]", "total/AP"]
    rows = []
    for b in best_list:
        ev = b.get("event")
        st = b.get("stage")
        df = b.get("diff")
        total_eff = b.get("total_eff", 0.0)
        rows.append([ev, f"{st} [{df}]", f2s(total_eff)])
    _print_table(write_line, headers, rows, aligns=["left", "left", "right"])

# =============================================================================
# 로깅
# =============================================================================

class Logger:
    """파일에 기록(선택적으로 콘솔 동시 출력)"""
    def __init__(self, filepath: str, tee: bool=False):
        self.filepath = filepath
        self.tee = tee
        self.f = open(filepath, "w", encoding="utf-8")
    def write(self, line: str=""):
        self.f.write(line + "\n")
        if self.tee:
            print(line)
    def close(self):
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass

# =============================================================================
# 획득 반영 (누적/부족 감소/변동 기록)
# =============================================================================

def apply_gain_and_track(item: str, qty: float, materials_index: Dict[str, dict],
                         cum_gained: Dict[str, float], respect_use_flag: bool=True) -> None:
    """
    획득량을 누적/부족에 반영하고, 이번 판 변동 테이블에도 기록한다.
    (재귀/중첩 호출 방지용 얕은 락 포함)
    """
    global CURRENT_RUN_GAINS, _AGT_CALL_DEPTH
    _AGT_CALL_DEPTH += 1
    try:
        if _AGT_CALL_DEPTH > 1:
            return
        if qty > 0:
            cum_gained[item] = cum_gained.get(item, 0.0) + float(qty)
            if CURRENT_RUN_GAINS is not None:
                CURRENT_RUN_GAINS[item] = CURRENT_RUN_GAINS.get(item, 0.0) + float(qty)
        rec = mat_rec(item, materials_index)
        if rec is None or (respect_use_flag and not rec.get("use", True)):
            return
        before = float(rec.get("lack") or 0.0)
        after = before - float(qty)
        rec["lack"] = after if after > 0 else 0.0
    finally:
        _AGT_CALL_DEPTH -= 1

def _apply_special_per_run_yields(event_name: str, materials_index: Dict[str, dict],
                                  cum_gained: Dict[str, float], respect_use_flag: bool=True) -> None:
    """특정 이벤트명 포함 시, 고정 per-run 보상 적용(옵션)"""
    for key, yields_ in SPECIAL_PER_RUN_YIELDS.items():
        if key in str(event_name):
            for row in yields_:
                apply_gain_and_track(str(row["item"]), float(row["qty"]),
                                     materials_index, cum_gained, respect_use_flag)

# =============================================================================
# 교환권 배분/가치 계산
# =============================================================================

def allocate_exchange_tokens_ap_first(token_qty: float, per_token: float, options: List[str],
                                      materials_index: Dict[str, dict], respect_use_flag: bool=True) -> List[Tuple[str, float]]:
    """
    교환티켓을 AP가치가 높은 아이템부터 부족량 한도 내에서 배분한다.
    Returns: [(아이템, 배분수량), ...]
    """
    candidates = []
    for it in options:
        if not mat_usable(it, materials_index):
            continue
        if mat_lack(it, materials_index) <= 0:
            continue
        apv = mat_ap(it, materials_index, respect_use_flag)
        candidates.append((it, apv, mat_lack(it, materials_index)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    remain = token_qty * per_token
    allocation = []
    for it, _, lack in candidates:
        if remain <= 0:
            break
        room = max(lack, 0.0)
        if room <= 0:
            continue
        take = min(remain, room)
        if take > 0:
            allocation.append((it, take))
            remain -= take
    return allocation

# =============================================================================
# 박스/룰렛/스테이지 가치 계산 유틸
# =============================================================================

def get_items_block(event_name: str, items_def: dict, case: str) -> Optional[dict]:
    """event_items.json에서 (event, case) 블록 추출"""
    target = str(event_name).strip()
    for b in _iter_item_blocks(items_def):
        if str(b.get("event") or "").strip() == target and str(b.get("case") or "").lower() == case:
            return b
    return None

def extract_need_tickets_from_items(event_name: str, items_def: dict, default: float=600.0) -> float:
    """룰렛 상자 1회 오픈에 필요한 티켓 수(없으면 기본값)"""
    blk = get_items_block(event_name, items_def, case="roulette")
    if not blk:
        return default
    try:
        return float(blk.get("need_tickets") or default)
    except Exception:
        return default

def compute_per_box_value_and_refund(event_name: str, items_def: dict, materials_index: Dict[str, dict],
                                     respect_use_flag: bool=True, apple_ap_map: Dict[str, float]=None) -> Tuple[float, float, List[dict]]:
    """
    룰렛 1상자 오픈 시 평균 AP가치와 사과 환급 AP를 계산
    Returns: (per_box_value, apple_refund_ap, details)
    """
    if apple_ap_map is None:
        apple_ap_map = {k: APPLE_AP_BY_POOL[v] for k, v in APPLE_NAME_TO_POOL.items()}
    blk = get_items_block(event_name, items_def, case="roulette")
    per_box = 0.0
    refund = 0.0
    details = []
    if not blk:
        return 0.0, 0.0, details
    token_name = None
    token_rate = 0.0
    for d in blk.get("drops", []):
        item = str(d.get("item") or "")
        rate = float(d.get("rate") or 0.0)
        if not item:
            continue
        if item in apple_ap_map:
            refund += rate * apple_ap_map[item]
            continue
        if item == "교환티켓":
            token_name = item
            token_rate = rate
            continue
        apv = mat_ap(item, materials_index, respect_use_flag)
        val = apv * rate
        per_box += val
        details.append({"kind": "drop", "item": item, "qty_per_box": rate, "ap_per_item": apv, "ap_value": r2(val)})
    if token_name and token_rate > 0:
        for ex in blk.get("exchanges", []):
            if str(ex.get("token")) != token_name:
                continue
            per_token = float(ex.get("per_token") or 1.0)
            options = [str(x) for x in ex.get("options", []) if x]
            alloc = allocate_exchange_tokens_ap_first(token_rate, per_token, options, materials_index, respect_use_flag)
            for it, qty in alloc:
                apv = mat_ap(it, materials_index, respect_use_flag)
                val = apv * qty
                per_box += val
                details.append({"kind": "exchange", "item": it, "qty": r2(qty), "ap_per_item": apv, "ap_value": r2(val)})
    return r2(per_box), r2(refund), details

def tickets_per_run(stage_def: dict, ce_count: int) -> float:
    """해당 스테이지 1회 클리어 시 얻는 룰렛 티켓 수(CE 보너스 반영)"""
    t = stage_def.get("tickets") or {}
    return float(t.get("base") or 0.0) + float(t.get("per_ce") or 0.0) * ce_count

def stage_value_per_run(stage_drops: List[dict], materials_index: Dict[str, dict], respect_use_flag: bool=True) -> float:
    """스테이지 드랍 기대 AP가치의 합"""
    val = 0.0
    for d in stage_drops:
        item = str(d.get("item") or ""); rate = float(d.get("rate") or 0.0)
        if not item:
            continue
        val += mat_ap(item, materials_index, respect_use_flag) * rate
    return val

def default_rarity_sum(event_name: str, items_def: dict, materials_index: Dict[str, dict],
                       respect_use_flag: bool=True) -> Tuple[float, float, float, str, List[dict]]:
    """
    event_items.json 의 box case 드랍을 rarity_map 기준으로
    gold/silver/bronze 그룹 AP가치 합으로 환산
    """
    blk = get_items_block(event_name, items_def, case="box")
    if not blk:
        return 0.0, 0.0, 0.0, "none", []
    rarity_map = default_rarity_map()
    sum_gold = sum_silver = sum_bronze = 0.0
    dbg = []
    for d in blk.get("drops", []):
        item = str(d.get("item") or ""); rate = float(d.get("rate") or 0.0)
        if not item:
            continue
        apv = mat_ap(item, materials_index, respect_use_flag)
        rar = classify_rarity(item, rarity_map)
        val = apv * rate
        if rar == "gold":   sum_gold += val
        elif rar == "silver": sum_silver += val
        elif rar == "bronze": sum_bronze += val
        dbg.append({"item": item, "rarity": rar or "?", "rate": rate, "ap_per_item": apv, "ap_value": r2(val)})
    return r2(sum_gold), r2(sum_silver), r2(sum_bronze), "event_items", dbg

def _fetch_box_contents(stage_def: dict, event_block: dict, quests_def: dict) -> Dict[str, List[dict]]:
    """event_quests에서 box contents 우선 추출(스테이지→이벤트→최상위)"""
    box = stage_def.get("box") or {}
    if isinstance(box.get("items"), dict):
        return box["items"]
    if isinstance(box.get("contents"), dict):
        return box["contents"]
    if isinstance(event_block.get("box_contents"), dict):
        return event_block["box_contents"]
    if isinstance(quests_def.get("box_contents"), dict):
        return quests_def["box_contents"]
    return {}

def _rarity_sum_ap(lst: List[dict], materials_index: Dict[str, dict], respect_use_flag: bool=True) -> float:
    """한 rarity 그룹의 항목 리스트 → AP가치 합"""
    if not lst:
        return 0.0
    total = 0.0
    has_qty = any('qty_per_box' in (x or {}) for x in lst)
    has_rate = any('rate' in (x or {}) for x in lst)
    n = len(lst)
    for x in lst or []:
        item = str((x or {}).get("item") or "")
        if not item:
            continue
        if has_qty:
            qty = float((x or {}).get("qty_per_box") or 0.0)
        elif has_rate:
            qty = float((x or {}).get("rate") or 0.0)
        else:
            qty = 1.0 / n if n > 0 else 0.0
        total += mat_ap(item, materials_index, respect_use_flag) * qty
    return total

def _build_box_contents_from_items_json(event_name: str, items_def: dict) -> Dict[str, List[dict]]:
    """
    event_items.json 만으로 gold/silver/bronze 그룹을 구성(스테이지 정의에 없을 때 fallback)
    """
    blk = get_items_block(event_name, items_def, case="box")
    if not blk:
        return {}
    rarity_map = default_rarity_map()
    groups = {"gold": [], "silver": [], "bronze": []}
    for d in blk.get("drops", []):
        item = str(d.get("item") or "").strip()
        if not item:
            continue
        rar = classify_rarity(item, rarity_map)
        if rar not in ("gold", "silver", "bronze"):
            continue
        entry = {"item": item}
        if "qty_per_box" in d:
            try:
                entry["qty_per_box"] = float(d.get("qty_per_box") or 0.0)
            except Exception:
                entry["qty_per_box"] = 0.0
        elif "rate" in d:
            try:
                entry["rate"] = float(d.get("rate") or 0.0)
            except Exception:
                entry["rate"] = 0.0
        groups[rar].append(entry)
    for rar in ("gold", "silver", "bronze"):
        lst = groups[rar]
        if not lst:
            continue
        has_qty = any("qty_per_box" in x for x in lst)
        has_rate = any("rate" in x for x in lst)
        if not has_qty and not has_rate:
            n = len(lst)
            for x in lst:
                x["rate"] = 1.0 / n if n > 0 else 0.0
    return groups

def compute_box_event_best_stage(event_name: str, quests_def: dict, items_def: dict,
                                 materials_index: Dict[str, dict], ap_cost_per_run: float,
                                 prefer_diff: Optional[str], respect_use_flag: bool=True) -> Optional[dict]:
    """박스 이벤트에서 스테이지별 (스테이지드랍+박스가치)/AP 최댓값 선택"""
    events = quests_def.get("event_quests", [])
    candidates = []
    for e in events:
        if str(e.get("event") or "") != event_name or str(e.get("case") or "").lower() != "box":
            continue
        for st in e.get("stages", []):
            stage_val = stage_value_per_run(st.get("drops", []), materials_index, respect_use_flag)
            box = st.get("box") or {}
            base_gold = float((box.get("gold") or {}).get("base", 0.0))
            base_silver = float((box.get("silver") or {}).get("base", 0.0))
            base_bronze = float((box.get("bronze") or {}).get("base", 0.0))
            contents = _fetch_box_contents(st, e, quests_def)
            dbg_rows = []; source = ""
            if contents:
                sum_gold = _rarity_sum_ap(contents.get("gold", []), materials_index, respect_use_flag)
                sum_silver = _rarity_sum_ap(contents.get("silver", []), materials_index, respect_use_flag)
                sum_bronze = _rarity_sum_ap(contents.get("bronze", []), materials_index, respect_use_flag)
                source = "stage.contents"
            else:
                sg, ss, sb, src, dbg = default_rarity_sum(event_name, items_def, materials_index, respect_use_flag)
                sum_gold, sum_silver, sum_bronze = sg, ss, sb
                source = src
                dbg_rows = dbg
            box_val = sum_gold * base_gold + sum_silver * base_silver + sum_bronze * base_bronze
            total_eff = (stage_val + box_val) / ap_cost_per_run if ap_cost_per_run > 0 else 0.0
            candidates.append({
                "stage": st.get("stage"), "diff": st.get("diff"),
                "stage_val_per_run": r2(stage_val), "box_val_per_run": r2(box_val),
                "total_eff": r2(total_eff),
                "details": {"base": {"gold": base_gold, "silver": base_silver, "bronze": base_bronze},
                            "sum_ap": {"gold": r2(sum_gold), "silver": r2(sum_silver), "bronze": r2(sum_bronze)},
                            "source": source, "drops": st.get("drops", []),
                            "event_items_dbg": dbg_rows}
            })
        break
    if not candidates:
        return None
    if prefer_diff:
        pref = [c for c in candidates if str(c.get("diff") or "") == str(prefer_diff)]
        if pref:
            return max(pref, key=lambda x: float(x["total_eff"]))
    return max(candidates, key=lambda x: float(x["total_eff"]))

def _get_event_block(quests_def: dict, event_name: str, case: str) -> Optional[dict]:
    for e in quests_def.get("event_quests", []):
        if str(e.get("event") or "") == event_name and str(e.get("case") or "").lower() == case:
            return e
    return None

def choose_best_stage_by_total_eff_roulette(event_name: str, quests_def: dict, items_def: dict,
                                            ce_count: int, need_tickets_default: float,
                                            ap_cost_per_run: float, materials_index: Dict[str, dict],
                                            respect_use_flag: bool=True) -> Optional[dict]:
    """
    룰렛 이벤트: (스테이지드랍/AP + (상자AP가치 / 유효AP상자)) 최댓값 스테이지 선택
    """
    need_tk = extract_need_tickets_from_items(event_name, items_def, default=need_tickets_default)
    per_box_value, apple_refund_ap, box_details = compute_per_box_value_and_refund(event_name, items_def, materials_index, respect_use_flag)
    events = quests_def.get("event_quests", []); candidates = []
    for e in events:
        if str(e.get("event") or "") != event_name or str(e.get("case") or "").lower() != "roulette":
            continue
        for st in e.get("stages", []):
            tpr = tickets_per_run(st, ce_count)
            stage_run_val = stage_value_per_run(st.get("drops", []), materials_index, respect_use_flag)
            stage_eff_per_ap = (stage_run_val / ap_cost_per_run) if ap_cost_per_run > 0 else 0.0
            if need_tk > 0 and tpr > 0:
                runs_per_box = need_tk / tpr
                gross_ap_per_box = runs_per_box * ap_cost_per_run
                net_ap_per_box = gross_ap_per_box - apple_refund_ap
            else:
                net_ap_per_box = 0.0
            if net_ap_per_box <= 0:
                roulette_eff = total_eff = float("inf")
            else:
                roulette_eff = per_box_value / net_ap_per_box
                total_eff = stage_eff_per_ap + roulette_eff
            candidates.append({
                "stage": st.get("stage"), "diff": st.get("diff"),
                "tickets_per_run": r2(tpr), "stage_val_per_run": r2(stage_run_val),
                "stage_eff_per_ap": r2(stage_eff_per_ap),
                "roulette_eff": (roulette_eff if roulette_eff == float("inf") else r2(roulette_eff)),
                "total_eff": (total_eff if total_eff == float("inf") else r2(total_eff)),
                "need_tickets": need_tk, "per_box_value": r2(per_box_value),
                "apple_refund_ap": r2(apple_refund_ap), "stage_drops": st.get("drops", []),
                "box_details": box_details
            })
        break
    if not candidates:
        return None
    def key_fn(x):
        return float("inf") if x["total_eff"] == float("inf") else float(x["total_eff"])
    return max(candidates, key=key_fn)

def compute_raid_best_stage(event_name: str, quests_def: dict, materials_index: Dict[str, dict],
                            ap_cost_per_run: float, respect_use_flag: bool=True, prefer_diff: Optional[str]=None) -> Optional[dict]:
    """레이드: 스테이지드랍/AP 최댓값"""
    events = quests_def.get("event_quests", []); candidates = []
    for e in events:
        if str(e.get("event") or "") != event_name or str(e.get("case") or "").lower() != "raid":
            continue
        for st in e.get("stages", []):
            stage_val = stage_value_per_run(st.get("drops", []), materials_index, respect_use_flag)
            stage_eff = (stage_val / ap_cost_per_run) if ap_cost_per_run > 0 else 0.0
            candidates.append({"stage": st.get("stage"), "diff": st.get("diff"),
                               "stage_val_per_run": r2(stage_val), "stage_eff_per_ap": r2(stage_eff),
                               "drops": st.get("drops", [])})
        break
    if not candidates:
        return None
    if prefer_diff:
        pref = [c for c in candidates if str(c.get("diff") or "") == str(prefer_diff)]
        if pref:
            return max(pref, key=lambda x: float(x["stage_eff_per_ap"]))
    return max(candidates, key=lambda x: float(x["stage_eff_per_ap"]))

# =============================================================================
# 1판 적용(드랍/룰렛/박스)
# =============================================================================

def _open_roulette_boxes_and_apply(event_name: str, num_boxes: int, items_def: dict,
                                   materials_index: Dict[str, dict], cum_gained: Dict[str, float],
                                   respect_use_flag: bool=True):
    """룰렛 상자 num_boxes개 오픈 후 드랍/교환 반영"""
    if num_boxes <= 0:
        return
    blk = get_items_block(event_name, items_def, case="roulette")
    if not blk:
        return
    token_name = None
    token_rate = 0.0
    for d in blk.get("drops", []):
        item = str(d.get("item") or "")
        rate = float(d.get("rate") or 0.0)
        if not item:
            continue
        if item in APPLE_ITEM_NAMES:
            continue
        if item == "교환티켓":
            token_name = item
            token_rate = rate
            continue
        apply_gain_and_track(item, rate * num_boxes, materials_index, cum_gained, respect_use_flag)
    if token_name and token_rate > 0:
        total_tokens = token_rate * num_boxes
        for ex in blk.get("exchanges", []):
            if str(ex.get("token")) != token_name:
                continue
            per_token = float(ex.get("per_token") or 1.0)
            options = [str(x) for x in ex.get("options", []) if x]
            alloc = allocate_exchange_tokens_ap_first(total_tokens, per_token, options, materials_index, respect_use_flag)
            for it, qty in alloc:
                apply_gain_and_track(it, qty, materials_index, cum_gained, respect_use_flag)

def _apply_stage_drops(stage_def: dict, materials_index: Dict[str, dict],
                       cum_gained: Dict[str, float], respect_use_flag: bool=True):
    """스테이지 기본 드랍 1회분 적용"""
    for d in stage_def.get("drops", []):
        item = str(d.get("item") or ""); rate = float(d.get("rate") or 0.0)
        if not item:
            continue
        apply_gain_and_track(item, rate, materials_index, cum_gained, respect_use_flag)

def _apply_box_event_contents_per_run(event_name: str, stage_def: dict, event_block: dict,
                                      quests_def: dict, items_def: dict, materials_index: Dict[str, dict],
                                      cum_gained: Dict[str, float], respect_use_flag: bool=True) -> bool:
    """
    박스 이벤트: 스테이지 클리어 1회당 상자 내용물 기대값 반영
    (스테이지 정의 우선, 없으면 event_items.json 추론)
    """
    contents = _fetch_box_contents(stage_def, event_block, quests_def)
    if not contents:
        contents = _build_box_contents_from_items_json(event_name, items_def)
        if not contents:
            return False
    box = stage_def.get("box") or {}
    base_gold   = float((box.get("gold")   or {}).get("base", 0.0))
    base_silver = float((box.get("silver") or {}).get("base", 0.0))
    base_bronze = float((box.get("bronze") or {}).get("base", 0.0))
    def _apply_list(lst, base):
        if not lst or base <= 0:
            return
        has_qty = any('qty_per_box' in (x or {}) for x in lst)
        has_rate = any('rate' in (x or {}) for x in lst)
        n = len(lst)
        for x in lst:
            item = str((x or {}).get("item") or "")
            if not item:
                continue
            if has_qty:
                per_box = float((x or {}).get("qty_per_box") or 0.0)
            elif has_rate:
                per_box = float((x or {}).get("rate") or 0.0)
            else:
                per_box = 1.0 / n if n > 0 else 0.0
            apply_gain_and_track(item, per_box * base, materials_index, cum_gained, respect_use_flag)
    _apply_list(contents.get("gold", []), base_gold)
    _apply_list(contents.get("silver", []), base_silver)
    _apply_list(contents.get("bronze", []), base_bronze)
    return True

# =============================================================================
# 대상/베스트/CE 상태
# =============================================================================

def _list_targets(quests_def: dict, event_filter: Optional[str]) -> List[Tuple[str, str]]:
    """이벤트 목록(roulette/box/raid) → [(event, case)]"""
    seen = set(); out = []
    for e in quests_def.get("event_quests", []):
        ev = str(e.get("event") or ""); cs = str(e.get("case") or "").lower()
        if not ev or cs not in ("roulette", "box", "raid"):
            continue
        if event_filter and (event_filter not in ev):
            continue
        key = (ev, cs)
        if key not in seen:
            seen.add(key); out.append(key)
    return out

def _compute_all_bests(targets, quests_def, items_def, materials_index, ap_cost_per_run, prefer_diff, respect_use_flag: bool=True):
    """각 대상 이벤트별 최적 스테이지 계산"""
    results = []
    for ev, cs in targets:
        if cs == "roulette":
            ce_bonus = _get_ce_bonus(ev)
            best = choose_best_stage_by_total_eff_roulette(ev, quests_def, items_def, ce_bonus, 600.0, ap_cost_per_run, materials_index, respect_use_flag)
            if best:
                best.update({"event": ev, "case": cs, "ce_bonus": ce_bonus})
                results.append(best)
        elif cs == "box":
            best = compute_box_event_best_stage(ev, quests_def, items_def, materials_index, ap_cost_per_run, prefer_diff, respect_use_flag)
            if best:
                best.update({"event": ev, "case": cs})
                results.append(best)
        else:
            best = compute_raid_best_stage(ev, quests_def, materials_index, ap_cost_per_run, respect_use_flag, prefer_diff)
            if best:
                results.append({"event": ev, "case": cs, "stage": best["stage"], "diff": best["diff"],
                                "stage_val_per_run": best["stage_val_per_run"], "stage_eff_per_ap": best["stage_eff_per_ap"],
                                "roulette_eff": 0.0, "total_eff": best["stage_eff_per_ap"]})
    return results

def _pick_global_best(best_list: List[dict]) -> Optional[dict]:
    """모든 후보 중 total_eff 최대 선택"""
    if not best_list:
        return None
    def key_fn(x):
        v = x.get("total_eff")
        if v == float("inf"): return float("inf")
        try: return float(v)
        except Exception: return -1e18
    return max(best_list, key=key_fn)

def _get_ce_bonus(event_name: str) -> int:
    st = CE_STATE.get(event_name)
    return int(st["bonus"]) if st else CE_BASE_BONUS

def _get_ce_drop_rate(stage_def: dict, event_block: Optional[dict]) -> float:
    """CE 드랍 기대치(스테이지 우선, 없으면 이벤트 기본치)"""
    v = stage_def.get("ce_drop_rate")
    if v is None and event_block is not None:
        v = event_block.get("ce_drop_rate")
    try:
        return float(v or 0.0)
    except Exception:
        return 0.0

def _update_ce_after_run(event_name: str, stage_def: dict, event_block: Optional[dict]) -> dict:
    """
    룰렛 CE 상태 업데이트: 기대 드랍 누적, +1 보너스 단계 계산
    Returns: 상태 변화/요약 정보
    """
    if event_name not in CE_STATE:
        return {"p":0.0,"new_drops":0.0,"new_copies_int":0,"gain_int":0,"bonus_inc":0,"new_bonus":_get_ce_bonus(event_name),"rem_to_next":0.0}
    p = _get_ce_drop_rate(stage_def, event_block)
    prev = CE_STATE[event_name]["drops_acc"]; new = prev + p
    prev_floor = int(prev // 1); new_floor = int(new // 1)
    gain_int = max(0, new_floor - prev_floor)
    prev_steps = int(prev // CE_COPIES_PER_PLUS); new_steps = int(new // CE_COPIES_PER_PLUS)
    bonus_inc = max(0, new_steps - prev_steps)
    new_bonus = min(CE_MAX_BONUS, CE_BASE_BONUS + new_steps)
    CE_STATE[event_name]["drops_acc"] = new; CE_STATE[event_name]["bonus"] = new_bonus
    next_step_target = (new_steps + 1) * CE_COPIES_PER_PLUS
    rem_to_next = max(0.0, next_step_target - new)
    return {"p":p,"new_drops":new,"new_copies_int":int(new // 1),"gain_int":gain_int,"bonus_inc":bonus_inc,"new_bonus":new_bonus,"rem_to_next":rem_to_next}

def _init_ce_state_for_targets(targets):
    for ev, cs in targets:
        if cs == "roulette" and ev not in CE_STATE:
            CE_STATE[ev] = {"drops_acc": 0.0, "bonus": CE_BASE_BONUS}

# =============================================================================
# 최종 표 & 로그 헤더
# =============================================================================

def _print_final_materials_summary(write_line, initial_lack: Dict[str, float], materials_index: Dict[str, dict],
                                   cum_gained: Dict[str, float], respect_use_flag: bool=True) -> None:
    """최종 재료 요약(정렬 규칙 유지, 정수 표기)"""
    final_lack = snapshot_lack(materials_index, respect_use_flag)
    names = set(initial_lack.keys()) | set(cum_gained.keys()) | set(final_lack.keys())
    rows = []
    for name in names:
        rec = materials_index.get(name)
        if respect_use_flag and rec is not None and not rec.get("use", True):
            continue
        target = float(initial_lack.get(name, 0.0))
        got = float(cum_gained.get(name, 0.0))
        remain = float(final_lack.get(name, max(0.0, target - got)))
        if target == 0.0 and got == 0.0 and remain == 0.0:
            continue
        rows.append([name, target, got, remain])

    rows.sort(key=lambda r: _material_sort_key(r[0]))
    if not rows:
        write_line("(최종 재료 현황: 출력할 항목 없음)")
        return
    headers = ["재료", "목표", "획득", "부족"]
    fmt_rows = [[r[0], f0(r[1]), f0(r[2]), f0(r[3])] for r in rows]
    _print_table(write_line, headers, fmt_rows, aligns=["left", "right", "right", "right"])

def _prepend_lines_to_file(filepath: str, lines: List[str]) -> None:
    """파일 상단에 요약 헤더 삽입(본문은 유지)"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            original = f.read()
        with open(filepath, "w", encoding="utf-8") as f:
            for ln in lines:
                f.write(ln + "\n")
            f.write("\n")
            f.write(original)
    except Exception:
        pass

# =============================================================================
# 메인
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--materials", default="materials.json")
    parser.add_argument("--quests", default="event_quests.json")
    parser.add_argument("--items",  default="event_items.json")
    parser.add_argument("--event",  default=None)
    parser.add_argument("--diff",   default=None)
    parser.add_argument("--ap-cost", type=float, default=40.0)
    parser.add_argument("--ignore-use-flag", action="store_true")
    parser.add_argument("--log", default="run_log.txt")
    parser.add_argument("--table-format", choices=["text", "md", "csv"], default="text",
                        help="표 출력 포맷(text, md, csv)")
    parser.add_argument("--table-style", choices=["box","ascii"], default="box",
                        help="텍스트 표 스타일: box(상자선) | ascii(+---+)")
    parser.add_argument("--ambiguous-wide", action="store_true",
                        help="모호폭(A) 문자를 폭=2로 취급(기본 1칸)")
    parser.add_argument("--ascii-arrow", action="store_true",
                        help="화살표를 '->'로 출력(기본 '→')")
    parser.add_argument("--ascii-bullet", action="store_true",
                        help="불릿을 '-'로 출력(기본 '·')")
    args = parser.parse_args()

    global TABLE_FORMAT, TABLE_STYLE, AMBIGUOUS_WIDE, ARROW, BULLET
    TABLE_FORMAT = args.table_format
    TABLE_STYLE  = args.table_style
    AMBIGUOUS_WIDE = bool(args.ambiguous_wide)
    if args.ascii_arrow:
        ARROW = "->"
    if args.ascii_bullet:
        BULLET = "-"

    logger = Logger(args.log, tee=False); w = logger.write

    # 1) 데이터 로드/인덱싱/정렬맵
    materials_json = load_json(args.materials)
    init_rarity_map_from_materials(materials_json)
    materials_index = to_material_index(materials_json)
    init_sort_index_from_materials(materials_json)

    quests_def = load_json(args.quests)
    try:
        items_def = load_json(args.items)
    except Exception:
        items_def = {}

    # 1-1) (신규) 아이템명 검증
    unknown_map = validate_item_names(materials_index, quests_def, items_def)
    if unknown_map:
        w("## 경고: materials.json에 없는 아이템명이 참조되었습니다")
        known_names_sorted = sorted(materials_index.keys())
        for nm in sorted(unknown_map.keys(), key=lambda x: x):
            locs = ", ".join(unknown_map[nm])
            suggestions = difflib.get_close_matches(nm, known_names_sorted, n=3, cutoff=0.6)
            sug = (" | 유사: " + ", ".join(suggestions)) if suggestions else ""
            w(f"- {nm}  | 위치 예시: {locs}{sug}")
        w("")

    respect_use_flag = not args.ignore_use_flag
    initial_lack = snapshot_lack(materials_index, respect_use_flag)
    targets = _list_targets(quests_def, args.event)
    _init_ce_state_for_targets(targets)

    total_ap = (
        APPLE_COUNTS.get("gold", 0)   * APPLE_AP_BY_POOL["gold"]   +
        APPLE_COUNTS.get("silver", 0) * APPLE_AP_BY_POOL["silver"] +
        APPLE_COUNTS.get("blue", 0)   * APPLE_AP_BY_POOL["blue"]   +
        APPLE_COUNTS.get("copper", 0) * APPLE_AP_BY_POOL["copper"] +
        float(NATURAL_AP or 0.0)
    )
    w(f"AP 풀: {f2(total_ap)}  | AP/판={f2(args.ap_cost)}  | 표 포맷={args.table_format}, 스타일={args.table_style}")
    w(f"# 룰렛 보너스: 시작 {CE_BASE_BONUS}, {CE_COPIES_PER_PLUS}장당 +1, 최대 {CE_MAX_BONUS}")

    if not targets:
        w("대상 이벤트 없음.")
        logger.close()
        print(f"[로그 저장 완료] {logger.filepath}")
        return

    # 2) 루프 상태
    ap_pool = total_ap
    tickets_acc_by_event: Dict[str, float] = {}
    need_tickets_cache: Dict[str, float] = {}
    roulette_refund_per_box: Dict[str, float] = {}
    cum_gained: Dict[str, float] = {}

    last_choice_key = None
    run_idx = 0
    session_segments: List[Tuple[str, str, int]] = []
    prev_session_key: Optional[Tuple[str, str]] = None

    # 3) AP가 남을 동안 반복
    while ap_pool >= args.ap_cost:
        run_idx += 1
        global CURRENT_RUN_GAINS; CURRENT_RUN_GAINS = {}

        # 이번 판 들어가기 전 기준의 각 이벤트 최적 스테이지
        bests_before = _compute_all_bests(targets, quests_def, items_def, materials_index, args.ap_cost, args.diff, respect_use_flag)
        global_best = _pick_global_best(bests_before)
        if not global_best:
            w("※ 선택할 스테이지 없음 → 중단")
            break

        ev = global_best["event"]; cs = global_best["case"]
        stage_name = global_best["stage"]; diff = global_best["diff"]
        ce_bonus_for_line = global_best.get("ce_bonus", _get_ce_bonus(ev))

        choice_key = f"{ev}|{cs}|{stage_name}|{diff}"
        stage_changed_trigger = (last_choice_key is None or choice_key != last_choice_key)
        sess_key = (ev, diff)
        if prev_session_key != sess_key:
            session_segments.append((ev, diff, 1))
            prev_session_key = sess_key
        else:
            name, d, cnt = session_segments[-1]
            session_segments[-1] = (name, d, cnt + 1)

        ap_pool -= args.ap_cost
        block = _get_event_block(quests_def, ev, cs); stage_def = None
        if block:
            for st in block.get("stages", []):
                if str(st.get("stage") or "") == stage_name and str(st.get("diff") or "") == str(diff):
                    stage_def = st; break
        if not stage_def:
            last_choice_key = choice_key; continue

        lack_before = snapshot_lack(materials_index, respect_use_flag)
        _apply_stage_drops(stage_def, materials_index, cum_gained, respect_use_flag)

        opened = 0; refund_gain = 0.0; last_tpr_for_run = 0.0
        if cs == "roulette":
            ce_bonus = _get_ce_bonus(ev)
            last_tpr_for_run = tickets_per_run(stage_def, ce_bonus)
            tickets_acc_by_event[ev] = tickets_acc_by_event.get(ev, 0.0) + last_tpr_for_run
            if ev not in need_tickets_cache:
                need_tickets_cache[ev] = extract_need_tickets_from_items(ev, items_def, default=600.0)
            if ev not in roulette_refund_per_box:
                _, ap_refund, _ = compute_per_box_value_and_refund(ev, items_def, materials_index, respect_use_flag=False)
                roulette_refund_per_box[ev] = float(ap_refund or 0.0)
            need_tk = need_tickets_cache[ev]
            if need_tk > 0 and tickets_acc_by_event[ev] >= need_tk:
                opened = int(tickets_acc_by_event[ev] // need_tk)
                tickets_acc_by_event[ev] -= opened * need_tk
                _open_roulette_boxes_and_apply(ev, opened, items_def, materials_index, cum_gained, respect_use_flag)
                refund_gain = opened * roulette_refund_per_box.get(ev, 0.0)
                if refund_gain > 0:
                    ap_pool += refund_gain
        elif cs == "box":
            if block:
                _apply_box_event_contents_per_run(ev, stage_def, block, quests_def, items_def, materials_index, cum_gained, respect_use_flag)

        _apply_special_per_run_yields(ev, materials_index, cum_gained, respect_use_flag)

        lack_after = snapshot_lack(materials_index, respect_use_flag)
        lack_zero_items = []
        for name, b in lack_before.items():
            a = lack_after.get(name, 0.0)
            if b > 0 and a == 0:
                lack_zero_items.append((name, b, cum_gained.get(name, 0.0)))

        ce_trigger_info = None
        if cs == "roulette":
            ce_info = _update_ce_after_run(ev, stage_def, block)
            if ce_info["new_bonus"] < CE_MAX_BONUS and (ce_info["gain_int"] > 0 or ce_info["bonus_inc"] > 0):
                ce_trigger_info = ce_info

        should_log = stage_changed_trigger or (len(lack_zero_items) > 0) or (ce_trigger_info is not None)
        ce_only_trigger = (ce_trigger_info is not None) and (not stage_changed_trigger) and (len(lack_zero_items) == 0)

        if should_log:
            if stage_changed_trigger:
                if cs == "roulette":
                    w(f"[Run {run_idx}] 스테이지 변경 → {ev} [{cs}] {stage_name} ({diff})  | 보너스={ce_bonus_for_line}  | total_eff={f2s(global_best['total_eff'])}")
                else:
                    w(f"[Run {run_idx}] 스테이지 변경 → {ev} [{cs}] {stage_name} ({diff})  | total_eff={f2s(global_best['total_eff'])}")
            else:
                if cs == "roulette":
                    ce_state = CE_STATE.get(ev, {"drops_acc": 0.0, "bonus": CE_BASE_BONUS})
                    current_int = int(ce_state["drops_acc"] // 1)
                    w(f"[Run {run_idx}] 상태 기록 → {ev} [{cs}] {stage_name} ({diff})  | 현재 보너스={ce_state['bonus']} | 예장={current_int}장")
                else:
                    w(f"[Run {run_idx}] 상태 기록 → {ev} [{cs}] {stage_name} ({diff})")
            if ce_only_trigger:
                w(""); w("")
                last_choice_key = choice_key
                continue

            for name, b, total in lack_zero_items:
                w(f"  {BULLET} [Run {run_idx}] 재료 충족: {name}  (부족 {f2(b)} {ARROW} 0)  | 누적 획득 {f2(total)}")

            if ce_trigger_info is not None and not ce_only_trigger:
                gi = ce_trigger_info["gain_int"]; bi = ce_trigger_info["bonus_inc"]; cur_int = ce_trigger_info["new_copies_int"]
                if gi > 0: w(f"  {BULLET} 예장 드랍: +{gi}장 (현재 {cur_int}장, 누적 기대 {f2(ce_trigger_info['new_drops'])}장)")
                if bi > 0: w(f"    → 보너스 +{bi} ⇒ 현재 {ce_trigger_info['new_bonus']}  (다음 +1까지 {f2(ce_trigger_info['rem_to_next'])}장 기대)")

            w("변동된 결과 (이 판):")
            _print_gain_table(w, CURRENT_RUN_GAINS, lack_before, lack_after, cum_gained)

            if cs == "roulette":
                need_tk = need_tickets_cache.get(ev, 0.0)
                ce_state = CE_STATE.get(ev, {"drops_acc": 0.0, "bonus": CE_BASE_BONUS})
                cur_int = int(ce_state["drops_acc"] // 1)
                suffix = ""
                if ce_state["bonus"] < CE_MAX_BONUS:
                    rem = max(0.0, CE_COPIES_PER_PLUS - (ce_state["drops_acc"] % CE_COPIES_PER_PLUS))
                    suffix = f", 다음 +1까지 {f2(rem)}장"
                w(f"룰렛: 티켓 {f2(tickets_acc_by_event.get(ev, 0.0))}/{f2(need_tk)} (+{f2(last_tpr_for_run)}/판) | 환급AP +{f2(refund_gain)} | AP 풀 {f2(ap_pool)} | 예장 {cur_int}장(보너스 {ce_state['bonus']}{suffix})")

            # 이번 판 반영 후 최적 효율 요약(정보용)
            bests_after = _compute_all_bests(targets, quests_def, items_def, materials_index, args.ap_cost, args.diff, respect_use_flag)
            w("\n현재 이벤트 효율 요약:")
            _print_eff_table(w, bests_after, args.ap_cost)
            w(""); w("")

        last_choice_key = choice_key

    w(f"# 종료: 총 실행 {run_idx}판, 잔여 AP 풀={f2(ap_pool)}")
    w("")
    w("## 최종 재료 현황 (목표/획득/부족)")
    _print_final_materials_summary(w, initial_lack, materials_index, cum_gained, respect_use_flag)
    logger.close()

    # 로그 상단 요약 블록 삽입
    header = ["## 주회 세션(순서대로) — 이벤트별 연속 주행 구간 요약"]
    body = [f"  {i+1:>2}. {name} [{d}] — {cnt}판" for i, (name, d, cnt) in enumerate(session_segments)]
    lines = header + body + ["", "## 원본 로그 ↓", ""]
    for ln in header + body:
        print(ln)
    print()
    _prepend_lines_to_file(args.log, lines)
    print(f"[로그 저장 완료] {logger.filepath}")

if __name__ == "__main__":
    main()
