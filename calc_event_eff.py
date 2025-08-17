# filename: calc_event_eff.py
# -*- coding: utf-8 -*-
"""
룰렛/박스/레이드 이벤트 효율 계산기 + 시뮬레이터 (사과 하드코딩 + 트리거형 파일 로그)

핵심 포인트
- "트리거 시점"에만 로그를 남겨 로그 폭증 방지:
  1) 스테이지(이벤트) 변경 시
  2) 어떤 재료의 부족(lack)이 0 달성되는 순간
  3) 룰렛 예장(CE) 기대 드랍 집계로 보너스 경계 통과/보너스 변화가 있을 때
- 표 기반 출력(가독성↑) + '룰렛 상태 한 줄 요약' (티켓 누적/필요, 판당 티켓, 환급AP, AP 풀, 예장/보너스)
- 보너스 12 달성 이후에는 CE 관련 로그는 출력 생략(계산은 계속) — 노이즈 축소
- 룰렛 요약 줄에 '판당 티켓' 표시
- 트리거 블록 하나 출력이 끝날 때마다 "빈 줄 2개" 삽입(가독성↑)

추가 기능
- 런 전체 종료 후 "실제 주행 순서"대로 이벤트별 연속 주행 구간(세션) 요약:
  (1) 콘솔 출력
  (2) 로그 파일 제일 앞에 프리펜드(삽입)
  예) 산타네모 → 관위대관전 → 산타네모  → 요약 3줄
- 로그 마지막에 "최종 재료 현황" 표(목표/획득/부족) 출력
  * 목표: 시작 시점의 부족치(lack 초기값)
  * 획득: 시뮬 동안 기대 획득 누계
  * 부족: 종료 시점의 남은 부족치(=실시간 lack)
"""

import json
import argparse
from typing import Dict, List, Tuple, Optional

# -------------------- 상수 --------------------
# 사과 한 알당 AP 환산값 (게임 내 일반적인 체력환산)
APPLE_AP_POOL = {"gold": 145.0, "silver": 73.0, "blue": 40.0, "copper": 10.0}

# ===== 사용자 편집(하드코딩) =====
# 대량 시뮬 목적으로 기본값은 크게 둠. 필요 시 수정.
APPLE_COUNTS = {"gold": 237, "silver": 295, "blue": 1426, "copper": 502}
NATURAL_AP = 0.0
# ================================

# -------------------- IO --------------------
def load_json(path: str) -> dict:
    """UTF-8 JSON 로드(단순 래퍼). 파일 구조 검증은 상위 로직에서 수행."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def to_material_index(materials_json: dict) -> Dict[str, dict]:
    """
    materials.json -> {item: {ap, lack, use}} 인덱스 빌드
    Parameters
    ----------
    materials_json: dict
        {"materials":[{"item":..,"ap_per_item":..,"lack":..,"use":"Y|N"}, ...]}
    Returns
    -------
    Dict[str, dict]
        {아이템명: {"ap":float, "lack":float, "use":bool}}
    Notes
    -----
    - ap: 아이템 1개당 AP 가치(사용자 제공 수치)
    - lack: 현재 부족 수량(목표치)
    - use: 계산 포함 여부. 'N'이면 효율/획득/부족에서 제외
    """
    idx: Dict[str, dict] = {}
    for m in materials_json.get("materials", []):
        name = str(m.get("item") or "").strip()
        if not name:
            continue
        ap_raw = m.get("ap_per_item")
        lack_raw = m.get("lack")
        idx[name] = {
            "ap": float(ap_raw) if ap_raw is not None else 0.0,
            "lack": float(lack_raw) if lack_raw is not None else 0.0,
            "use": str(m.get("use") or "Y").upper() == "Y",
        }
    return idx

# ----------------- 숫자/재료 helpers -----------------
def r2(x: float) -> float:
    """float → 소수 2자리 반올림 숫자(내부 계산 표시용)."""
    return round(float(x), 2)

def f2(x: float) -> str:
    """float → '1,234.56' 형식 문자열."""
    return f"{float(x):,.2f}"

def f2s(x) -> str:
    """∞ 처리 포함 포맷."""
    try:
        if x == float("inf") or str(x).lower() == "inf":
            return "∞"
        return f2(float(x))
    except:
        return str(x)

def mat_rec(item, mats):
    """재료 dict 안전 접근."""
    return mats.get(item)

def mat_ap(item: str, mats: Dict[str, dict], respect_use_flag: bool=True) -> float:
    """
    '효율 계산용' AP 가치 반환.
    Rules
    -----
    - use=False 이면 0
    - lack<=0(포화) 이면 0
    - 그 외 ap 반환
    """
    rec = mat_rec(item, mats)
    if not rec:
        return 0.0
    if respect_use_flag and not rec.get("use", True):
        return 0.0
    if float(rec.get("lack") or 0.0) <= 0.0:
        return 0.0
    return float(rec.get("ap") or 0.0)

def mat_lack(item: str, mats: Dict[str, dict]) -> float:
    """현재 부족치(lack) 조회(없으면 0)."""
    rec = mat_rec(item, mats)
    return float(rec["lack"]) if rec else 0.0

def mat_usable(item: str, mats: Dict[str, dict]) -> bool:
    """use 플래그 조회(없으면 True 가정)."""
    rec = mat_rec(item, mats)
    return (rec["use"] if rec is not None else True)

def snapshot_lack(mats: Dict[str, dict], respect_use_flag: bool=True) -> Dict[str, float]:
    """
    현재 lack 스냅샷(표/비교용).
    - use=False는 제외(옵션)
    """
    return {
        n: float(r.get("lack") or 0.0)
        for n, r in mats.items()
        if not (respect_use_flag and not r.get("use", True))
    }

# ----------------- 표 유틸(출력 가독성) -----------------
def _draw_line(widths, left="┌", mid="┬", right="┐", fill="─"):
    s = [left]
    for i, w in enumerate(widths):
        s.append(fill * (w + 2))
        s.append(mid if i < len(widths) - 1 else right)
    return "".join(s)

def _fmt_cell(text, width, align="left"):
    t = str(text)
    if align == "right":
        return " " + t.rjust(width) + " "
    if align == "center":
        return " " + t.center(width) + " "
    return " " + t.ljust(width) + " "

def _print_table(w, headers, rows, aligns=None):
    """
    간단한 테이블 렌더러(로그/콘솔 공용).
    Parameters
    ----------
    w : Callable[[str], None]
        출력 함수(Logger.write 등)
    headers : List[str]
    rows : List[List[Any]]
    aligns : Optional[List['left'|'right'|'center']]
    """
    widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    if aligns is None:
        aligns = ["left"] * len(headers)
    w(_draw_line(widths, "┌", "┬", "┐"))
    w("│" + "│".join(_fmt_cell(h, widths[i], "center") for i, h in enumerate(headers)) + "│")
    w(_draw_line(widths, "├", "┼", "┤"))
    for r in rows:
        w("│" + "│".join(_fmt_cell(r[i], widths[i], aligns[i]) for i in range(len(headers))) + "│")
    w(_draw_line(widths, "└", "┴", "┘"))

def _print_gain_table(w, current_gains: Dict[str, float],
                      lack_before: Dict[str, float],
                      lack_after: Dict[str, float],
                      cum_gained: Dict[str, float]):
    """
    이번 판 획득 내역 표.
    Columns
    -------
    항목 | +획득(이번 판 기대) | 누적(전체 기대) | 부족(전→후)
    """
    if not current_gains:
        w("  (변동 없음)")
        return
    headers = ["항목", "+획득", "누적", "부족(전→후)"]
    rows = []
    for name, gained in sorted(current_gains.items(), key=lambda x: x[1], reverse=True):
        b = lack_before.get(name, 0.0)
        a = lack_after.get(name, 0.0)
        total = cum_gained.get(name, 0.0)
        rows.append([name, f2(gained), f2(total), f"{f2(b)} → {f2(a)}"])
    _print_table(w, headers, rows, aligns=["left", "right", "right", "right"])

def _print_eff_table(w, bests: List[dict], ap_cost_per_run: float):
    """
    현재 이벤트별 최적 효율 테이블.
    Columns
    -------
    이벤트 | 케이스 | 스테이지[난이도] | stage/AP | roulette/AP | total/AP
    Notes
    -----
    - raid는 roulette/AP가 없으므로 '-' 처리
    """
    headers = ["이벤트", "케이스", "스테이지[난이도]", "stage/AP", "roulette/AP", "total/AP"]
    rows = []
    for b in bests:
        ev = b.get("event"); cs = b.get("case"); st = b.get("stage"); df = b.get("diff")
        if cs == "roulette":
            rows.append([ev, cs, f"{st} [{df}]",
                         f2s(b.get("stage_eff_per_ap", 0.0)),
                         f2s(b.get("roulette_eff", 0.0)),
                         f2s(b.get("total_eff", 0.0))])
        elif cs == "box":
            stage_val = float(b.get("stage_val_per_run") or 0.0)
            box_val = float(b.get("box_val_per_run") or 0.0)
            stage_eff = (stage_val / ap_cost_per_run) if ap_cost_per_run > 0 else 0.0
            box_eff = (box_val / ap_cost_per_run) if ap_cost_per_run > 0 else 0.0
            rows.append([ev, cs, f"{st} [{df}]", f2s(stage_eff), f2s(box_eff), f2s(stage_eff + box_eff)])
        else:  # raid
            stage_eff = float(b.get("stage_eff_per_ap") or 0.0)
            rows.append([ev, cs, f"{st} [{df}]", f2s(stage_eff), "-", f2s(stage_eff)])
    _print_table(w, headers, rows, aligns=["left", "center", "left", "right", "right", "right"])

# ----------------- 로깅 & 특수 처리 -----------------
CURRENT_RUN_GAINS: Optional[Dict[str, float]] = None  # 이번 판 획득 집계(로그용)
_AGT_CALL_DEPTH = 0  # apply_gain_and_track 재진입 가드(안전)

class Logger:
    """파일 + (옵션)콘솔 동시 출력. tee=False면 파일만."""
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

def apply_gain_and_track(item: str, qty: float, mats: Dict[str, dict],
                         cum: Dict[str, float], respect_use_flag: bool=True) -> None:
    """
    획득 적용 + 누적 갱신 + 이번 판 집계.
    Parameters
    ----------
    item : str
    qty : float
        기대 획득량(확률 기대 포함)
    cum : Dict[str, float]
        아이템별 누적 획득 기대량 집계 사전
    Notes
    -----
    - 부족(lack) 즉시 차감(0 미만이면 0으로 클램프)
    - CURRENT_RUN_GAINS: 이번 판 표시에만 사용
    - 재귀적 호출 방지 위해 전역 가드 사용
    """
    global CURRENT_RUN_GAINS, _AGT_CALL_DEPTH
    _AGT_CALL_DEPTH += 1
    try:
        if _AGT_CALL_DEPTH > 1:
            return  # 재진입 방지
        if qty > 0:
            cum[item] = cum.get(item, 0.0) + float(qty)
            if CURRENT_RUN_GAINS is not None:
                CURRENT_RUN_GAINS[item] = CURRENT_RUN_GAINS.get(item, 0.0) + float(qty)
        rec = mat_rec(item, mats)
        if rec is None or (respect_use_flag and not rec.get("use", True)):
            return
        before = float(rec.get("lack") or 0.0)
        after = before - float(qty)
        rec["lack"] = after if after > 0 else 0.0
    finally:
        _AGT_CALL_DEPTH -= 1

# 이벤트명 부분일치 → 특수 수급(서머어드벤쳐: 15종 각 0.46)
SPECIAL_PER_RUN_YIELDS = {
    "서머어드벤쳐": [
        {"item": it, "qty": 0.46}
        for it in [
            "흉골","용의송곳니","여진화약","만사의독침","마술수액","뱀의보옥","소라껍데기","고스트랜턴",
            "거인의반지","무지갯빛실타래","용의역린","유구의열매","원초의산모","기기신주","황성의조각",
        ]
    ]
}

def _apply_special_per_run_yields(event_name: str, mats: Dict[str, dict],
                                  cum: Dict[str, float], respect_use_flag: bool=True) -> None:
    """
    특정 이벤트(부분일치)에 대해 고정 기대 수급을 매 판 적용.
    - 예: '서머어드벤쳐' 포함 시 15종 각 0.46 기대치 적용
    """
    for key, yields_ in SPECIAL_PER_RUN_YIELDS.items():
        if key in str(event_name):
            for row in yields_:
                apply_gain_and_track(str(row["item"]), float(row["qty"]), mats, cum, respect_use_flag)

# ----------------- 박스 분류/가치 -----------------
def default_rarity_map() -> Dict[str, set]:
    """박스 이벤트 계산 시 레어도 그룹(합산용)."""
    gold = {
        "혼돈의발톱","만신의심장","용의역린","정령근","전마의유각","혈루석","흑수지","봉마의램프",
        "스카라베","원초의산모","주수담석","기기신주","효광노심","구십구경","진리의알","황성의조각",
        "유구의열매","도깨비불꽈리","황금가마","월광핵","운명의신성수","유령상",
    }
    silver = {
        "세계수의씨앗","고스트랜턴","팔연쌍정","뱀의보옥","봉황의깃털","무간의톱니바퀴","금단의페이지",
        "호문클루스","운제철","대기사훈장","소라껍데기","고담곡옥","영원결빙","거인의반지","오로라강",
        "한고령","재난의화살촉","광은의관","신맥영자","무지갯빛실타래","몽환의비늘가루","태양피",
        "에테르수광체","최후의꽃","유니버셜 큐브","신체의 렌즈",
    }
    bronze = {
        "영웅의증표","흉골","용의송곳니","허영의먼지","우자의사슬","만사의독침","마술수액",
        "소곡의철향","여진화약","사면의작은종","황혼의의식검","잊을수없는재","흑요예인","광기의잔재",
    }
    return {"gold": gold, "silver": silver, "bronze": bronze}

def classify_rarity(item: str, rarity_map: Dict[str, set]) -> Optional[str]:
    """아이템명 → 'gold'|'silver'|'bronze'|None"""
    for r in ("gold", "silver", "bronze"):
        if item in rarity_map[r]:
            return r
    return None

def allocate_exchange_tokens_ap_first(token_qty: float, per_token: float, options: List[str],
                                      mats: Dict[str, dict], respect_use_flag: bool=True) -> List[Tuple[str, float]]:
    """
    교환티켓 배분(최대 AP 가치 우선).
    Parameters
    ----------
    token_qty : float
        티켓 기대 수량(박스 1개 기준 rate × 박스 수)
    per_token : float
        티켓 1장당 교환 수량
    options : List[str]
        교환 가능 아이템 목록
    Returns
    -------
    List[Tuple[item, qty]]
        AP 가치가 높은 재료부터 부족치 한계까지 채우도록 배분
    Notes
    -----
    - 포화(lack≤0) 재료는 자동 제외
    """
    cand = []
    for it in options:
        if not mat_usable(it, mats):
            continue
        if mat_lack(it, mats) <= 0:
            continue
        apv = mat_ap(it, mats, respect_use_flag)
        cand.append((it, apv, mat_lack(it, mats)))
    cand.sort(key=lambda x: x[1], reverse=True)  # 가치 높은 순

    remain = token_qty * per_token
    alloc = []
    for it, _, lack in cand:
        if remain <= 0:
            break
        room = max(lack, 0.0)
        if room <= 0:
            continue
        take = min(remain, room)
        if take > 0:
            alloc.append((it, take))
            remain -= take
    return alloc

# ----------------- 데이터 접근 -----------------
def _iter_item_blocks(items_json: dict) -> List[dict]:
    """event_items/event_quests/events 어디에 있든 공통 탐색."""
    if isinstance(items_json.get("event_items"), list):
        return items_json["event_items"]
    if isinstance(items_json.get("event_quests"), list):
        return items_json["event_quests"]
    if isinstance(items_json.get("events"), list):
        return items_json["events"]
    return []

def get_items_block(event_name: str, items_json: dict, case: str) -> Optional[dict]:
    """이벤트명+케이스로 items 블록 검색."""
    target = str(event_name).strip()
    for b in _iter_item_blocks(items_json):
        if str(b.get("event") or "").strip() == target and str(b.get("case") or "").lower() == case:
            return b
    return None

def extract_need_tickets_from_items(event_name: str, items_json: dict, default: float=600.0) -> float:
    """룰렛 1박스에 필요한 티켓 수(없으면 기본 600)."""
    blk = get_items_block(event_name, items_json, case="roulette")
    if not blk:
        return default
    try:
        return float(blk.get("need_tickets") or default)
    except:
        return default

def compute_per_box_value_and_refund(event_name: str, items_json: dict, mats: Dict[str, dict],
                                     respect_use_flag: bool=True,
                                     apple_ap_map: Dict[str, float]=None) -> Tuple[float, float, List[dict]]:
    """
    룰렛 1박스 'AP 가치'와 '사과 환급 AP' 산출.
    Returns
    -------
    (per_box_value_ap, apple_refund_ap, details)
    Notes
    -----
    - drops의 일반 재료는 mat_ap로 가치 평가(포화=0)
    - 교환티켓은 'AP 가치 우선 배분' 시뮬 후 합산
    - details: 디버그/검증용(현재 화면엔 미표시)  # (미사용 반환값)
    """
    apple_ap_map = apple_ap_map or {"금사과":145.0, "은사과":73.0, "청사과":40.0, "동사과":10.0}
    blk = get_items_block(event_name, items_json, case="roulette")
    per_box = 0.0
    refund = 0.0
    details = []  # (현재 경로에서 실사용 아님)

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
        if item == "교환티켓":
            token_name = item
            token_rate = rate
            continue
        apv = mat_ap(item, mats, respect_use_flag)
        val = apv * rate
        per_box += val
        details.append({"kind":"drop","item":item,"qty_per_box":rate,"ap_per_item":apv,"ap_value":r2(val)})

    if token_name and token_rate > 0:
        for ex in blk.get("exchanges", []):
            if str(ex.get("token")) != token_name:
                continue
            per_token = float(ex.get("per_token") or 1.0)
            options = [str(x) for x in ex.get("options", []) if x]
            alloc = allocate_exchange_tokens_ap_first(token_rate, per_token, options, mats, respect_use_flag)
            for it, qty in alloc:
                apv = mat_ap(it, mats, respect_use_flag)
                val = apv * qty
                per_box += val
                details.append({"kind":"exchange","item":it,"qty":r2(qty),"ap_per_item":apv,"ap_value":r2(val)})

    return r2(per_box), r2(refund), details

# ----------------- 스테이지 공통 -----------------
def tickets_per_run(stage: dict, ce_count: int) -> float:
    """
    판당 티켓 기대량 = base + per_ce * (현재 CE 보너스 수치)
    - ce_count는 7~12 범위의 보너스 정수 그대로 사용
    """
    t = stage.get("tickets") or {}
    return float(t.get("base") or 0.0) + float(t.get("per_ce") or 0.0) * ce_count

def stage_value_per_run(stage_drops: List[dict], mats: Dict[str, dict],
                        respect_use_flag: bool=True) -> float:
    """
    스테이지 일반 드랍의 AP 기대가치 합.
    - 포화 재료는 0 가치로 평가하여 자동 제외 효과
    """
    val = 0.0
    for d in stage_drops:
        item = str(d.get("item") or ""); rate = float(d.get("rate") or 0.0)
        if not item:
            continue
        val += mat_ap(item, mats, respect_use_flag) * rate
    return val

# ----------------- 박스 케이스 -----------------
def default_rarity_sum(event_name: str, items_json: dict, mats: Dict[str, dict],
                       respect_use_flag: bool=True) -> Tuple[float, float, float, str, List[dict]]:
    """
    event_items의 박스 구성으로 레어도 합(가치) 계산.
    Returns
    -------
    (gold_sum_ap, silver_sum_ap, bronze_sum_ap, source, dbg_rows)
    Notes
    -----
    - dbg_rows: 디버그 출력용(현재 미표시)  # (미사용 반환값)
    """
    blk = get_items_block(event_name, items_json, case="box")
    if not blk:
        return 0.0, 0.0, 0.0, "none", []
    rarity_map = default_rarity_map()
    sg = ss = sb = 0.0
    dbg = []
    for d in blk.get("drops", []):
        item = str(d.get("item") or ""); rate = float(d.get("rate") or 0.0)
        if not item:
            continue
        apv = mat_ap(item, mats, respect_use_flag)
        rar = classify_rarity(item, rarity_map)
        val = apv * rate
        if rar == "gold":   sg += val
        elif rar == "silver": ss += val
        elif rar == "bronze": sb += val
        dbg.append({"item": item, "rarity": rar or "?", "rate": rate, "ap_per_item": apv, "ap_value": r2(val)})
    return r2(sg), r2(ss), r2(sb), "event_items", dbg

def _fetch_box_contents(stage: dict, event_block: dict, quests_json: dict) -> Dict[str, List[dict]]:
    """
    스테이지 정의/이벤트 정의/최상위 정의 순서로 box contents 조회.
    우선순위: stage.box.items → stage.box.contents → event.box_contents → quests.box_contents
    """
    box = stage.get("box") or {}
    if isinstance(box.get("items"), dict):
        return box["items"]
    if isinstance(box.get("contents"), dict):
        return box["contents"]
    if isinstance(event_block.get("box_contents"), dict):
        return event_block["box_contents"]
    if isinstance(quests_json.get("box_contents"), dict):
        return quests_json["box_contents"]
    return {}

def _rarity_sum_ap(lst: List[dict], mats: Dict[str, dict], respect_use_flag: bool=True) -> float:
    """contents 기반 레어도 묶음의 AP 합(포화=0)."""
    total = 0.0
    for x in lst or []:
        item = str(x.get("item") or ""); qty = float(x.get("qty_per_box") or 0.0)
        total += mat_ap(item, mats, respect_use_flag) * qty
    return total

def compute_box_event_best_stage(event_name: str, quests_json: dict, items_json: dict,
                                 mats: Dict[str, dict], ap_cost_per_run: float, prefer_diff: Optional[str],
                                 respect_use_flag: bool=True) -> Optional[dict]:
    """
    BOX 케이스: 스테이지별 (스테이지가치+박스가치)/AP 최대 후보 선택.
    - stage.contents가 있으면 그것을 우선 사용, 없으면 event_items로 추정
    """
    events = quests_json.get("event_quests", [])
    candidates = []
    for e in events:
        if str(e.get("event") or "") != event_name or str(e.get("case") or "").lower() != "box":
            continue
        for st in e.get("stages", []):
            stage_val = stage_value_per_run(st.get("drops", []), mats, respect_use_flag)
            box = st.get("box") or {}
            base_gold = float((box.get("gold") or {}).get("base", 0.0))
            base_silver = float((box.get("silver") or {}).get("base", 0.0))
            base_bronze = float((box.get("bronze") or {}).get("base", 0.0))

            contents = _fetch_box_contents(st, e, quests_json)
            dbg_rows = []; source = ""
            if contents:
                sum_gold = _rarity_sum_ap(contents.get("gold", []), mats, respect_use_flag)
                sum_silver = _rarity_sum_ap(contents.get("silver", []), mats, respect_use_flag)
                sum_bronze = _rarity_sum_ap(contents.get("bronze", []), mats, respect_use_flag)
                source = "stage.contents"
            else:
                sg, ss, sb, src, dbg = default_rarity_sum(event_name, items_json, mats, respect_use_flag)
                sum_gold, sum_silver, sum_bronze = sg, ss, sb
                source = src
                dbg_rows = dbg  # (미사용 반환값)

            box_val = sum_gold * base_gold + sum_silver * base_silver + sum_bronze * base_bronze
            total_eff = (stage_val + box_val) / ap_cost_per_run if ap_cost_per_run > 0 else 0.0

            candidates.append({
                "stage": st.get("stage"), "diff": st.get("diff"),
                "stage_val_per_run": r2(stage_val), "box_val_per_run": r2(box_val),
                "total_eff": r2(total_eff),
                "details": {
                    "base": {"gold": base_gold, "silver": base_silver, "bronze": base_bronze},
                    "sum_ap": {"gold": r2(sum_gold), "silver": r2(sum_silver), "bronze": r2(sum_bronze)},
                    "source": source, "drops": st.get("drops", []),
                    "event_items_dbg": dbg_rows,  # (미사용 반환값)
                }
            })
        break

    if not candidates:
        return None
    if prefer_diff:
        pref = [c for c in candidates if str(c.get("diff") or "") == str(prefer_diff)]
        if pref:
            return max(pref, key=lambda x: float(x["total_eff"]))
    return max(candidates, key=lambda x: float(x["total_eff"]))

# ----------------- CE(예장) 상태/업데이트 -----------------
CE_BASE_BONUS = 7
CE_MAX_BONUS = 12
CE_COPIES_PER_PLUS = 4  # 4장마다 보너스 +1
CE_STATE: Dict[str, dict] = {}  # 이벤트별 {"drops_acc": 기대누적(연속), "bonus": 현재보너스}

def _init_ce_state_for_targets(targets):
    """룰렛 케이스 대상 이벤트에 CE 상태 초기화."""
    for ev, cs in targets:
        if cs == "roulette" and ev not in CE_STATE:
            CE_STATE[ev] = {"drops_acc": 0.0, "bonus": CE_BASE_BONUS}

def _get_ce_bonus(ev: str) -> int:
    """현재 이벤트의 CE 보너스(7~12)."""
    st = CE_STATE.get(ev)
    return int(st["bonus"]) if st else CE_BASE_BONUS

def _get_ce_drop_rate(stage_def: dict, event_block: Optional[dict]) -> float:
    """예장 기대 드랍확률(소수). 스테이지 우선 → 이벤트 블록 보조."""
    v = stage_def.get("ce_drop_rate")
    if v is None and event_block is not None:
        v = event_block.get("ce_drop_rate")
    try:
        return float(v or 0.0)
    except:
        return 0.0

def _update_ce_after_run(ev: str, stage_def: dict, event_block: Optional[dict]) -> dict:
    """
    1판 주행 후 CE 기대 누적치 업데이트 + 경계/보너스 변화 감지.
    Returns
    -------
    dict: 로깅용 상태(드랍 기대 누적/정수 증가/보너스 증가/다음 경계까지 남은 기대치 등)
    """
    if ev not in CE_STATE:
        return {"p":0.0,"new_drops":0.0,"new_copies_int":0,"gain_int":0,"bonus_inc":0,
                "new_bonus":_get_ce_bonus(ev),"rem_to_next":0.0}
    p = _get_ce_drop_rate(stage_def, event_block)
    prev = CE_STATE[ev]["drops_acc"]; new = prev + p

    prev_floor = int(prev // 1); new_floor = int(new // 1)
    gain_int = max(0, new_floor - prev_floor)

    prev_steps = int(prev // CE_COPIES_PER_PLUS); new_steps = int(new // CE_COPIES_PER_PLUS)
    bonus_inc = max(0, new_steps - prev_steps)
    new_bonus = min(CE_MAX_BONUS, CE_BASE_BONUS + new_steps)

    CE_STATE[ev]["drops_acc"] = new; CE_STATE[ev]["bonus"] = new_bonus
    next_step_target = (new_steps + 1) * CE_COPIES_PER_PLUS
    rem_to_next = max(0.0, next_step_target - new)

    return {"p":p,"new_drops":new,"new_copies_int":int(new // 1),"gain_int":gain_int,
            "bonus_inc":bonus_inc,"new_bonus":new_bonus,"rem_to_next":rem_to_next}

# ----------------- 베스트 계산/선택 -----------------
def _list_targets(quests_json: dict, event_filter: Optional[str]) -> List[Tuple[str, str]]:
    """event_quests에서 (이벤트명, 케이스) 타깃 목록 생성(부분일치 필터 적용)."""
    seen = set(); out = []
    for e in quests_json.get("event_quests", []):
        ev = str(e.get("event") or ""); cs = str(e.get("case") or "").lower()
        if not ev or cs not in ("roulette", "box", "raid"):
            continue
        if event_filter and (event_filter not in ev):
            continue
        key = (ev, cs)
        if key not in seen: seen.add(key); out.append(key)
    return out

def choose_best_stage_by_total_eff_roulette(event_name: str, quests_json: dict, items_json: dict,
                                            ce_count: int, need_tickets_default: float,
                                            ap_cost_per_run: float, mats: Dict[str, dict],
                                            respect_use_flag: bool=True) -> Optional[dict]:
    """
    룰렛 케이스: (스테이지/AP + 룰렛/AP) 최대 후보 선택.
    - ce_count: 현 보너스(7~12)를 그대로 인자로 받음
    """
    need_tk = extract_need_tickets_from_items(event_name, items_json, default=need_tickets_default)
    per_box_value, apple_refund_ap, box_details = compute_per_box_value_and_refund(
        event_name, items_json, mats, respect_use_flag
    )
    # box_details: 디버그용(현재 실사용 아님)

    events = quests_json.get("event_quests", []); candidates = []
    for e in events:
        if str(e.get("event") or "") != event_name or str(e.get("case") or "").lower() != "roulette":
            continue
        for st in e.get("stages", []):
            tpr = tickets_per_run(st, ce_count)  # 판당 티켓
            stage_run_val = stage_value_per_run(st.get("drops", []), mats, respect_use_flag)
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
                "box_details": box_details,  # (미사용 반환값)
            })
        break

    if not candidates:
        return None
    def key_fn(x):  # ∞ 우선
        return float("inf") if x["total_eff"] == float("inf") else float(x["total_eff"])
    return max(candidates, key=key_fn)

def _compute_all_bests(targets, quests_json, items_json, mats, ap_cost_per_run, prefer_diff,
                       respect_use_flag: bool=True):
    """모든 타깃(이벤트×케이스)에 대해 '현재 최적 스테이지' 계산."""
    results = []
    for ev, cs in targets:
        if cs == "roulette":
            ce_bonus = _get_ce_bonus(ev)
            best = choose_best_stage_by_total_eff_roulette(
                ev, quests_json, items_json, ce_bonus, 600.0, ap_cost_per_run, mats, respect_use_flag
            )
            if best:
                best.update({"event": ev, "case": cs, "ce_bonus": ce_bonus})
                results.append(best)
        elif cs == "box":
            best = compute_box_event_best_stage(
                ev, quests_json, items_json, mats, ap_cost_per_run, prefer_diff or "90++", respect_use_flag
            )
            if best:
                best.update({"event": ev, "case": cs})
                results.append(best)
        else:  # raid
            best = compute_raid_best_stage(ev, quests_json, mats, ap_cost_per_run, respect_use_flag, prefer_diff)
            if best:
                results.append({
                    "event": ev, "case": cs, "stage": best["stage"], "diff": best["diff"],
                    "stage_val_per_run": best["stage_val_per_run"],
                    "stage_eff_per_ap": best["stage_eff_per_ap"],
                    "roulette_eff": 0.0, "total_eff": best["stage_eff_per_ap"],
                })
    return results

def _pick_global_best(bests: List[dict]) -> Optional[dict]:
    """total_eff 기준 최대 후보(∞ 지원)."""
    if not bests:
        return None
    def key_fn(x):
        v = x.get("total_eff")
        if v == float("inf"): return float("inf")
        try: return float(v)
        except: return -1e18
    return max(bests, key=key_fn)

# ----------------- 드랍 적용 -----------------
def _open_roulette_boxes_and_apply(event_name: str, num_boxes: int, items_json: dict,
                                   mats: Dict[str, dict], cum: Dict[str, float], respect_use_flag: bool=True):
    """
    룰렛 박스 개봉 시 보상 적용(교환티켓은 AP 가치 우선 배분).
    Notes
    -----
    - 사과는 AP 환급(풀 증가)만 반영하고, 재료 누적/부족에는 반영하지 않음
    """
    if num_boxes <= 0:
        return
    blk = get_items_block(event_name, items_json, case="roulette")
    if not blk:
        return
    token_name = None; token_rate = 0.0
    for d in blk.get("drops", []):
        item = str(d.get("item") or ""); rate = float(d.get("rate") or 0.0)
        if not item: continue
        if item in ("금사과","은사과","청사과","동사과"):  # 환급은 따로 처리
            continue
        if item == "교환티켓":
            token_name = item; token_rate = rate; continue
        apply_gain_and_track(item, rate * num_boxes, mats, cum, respect_use_flag)

    if token_name and token_rate > 0:
        total_tokens = token_rate * num_boxes
        for ex in blk.get("exchanges", []):
            if str(ex.get("token")) != token_name: continue
            per_token = float(ex.get("per_token") or 1.0)
            options = [str(x) for x in ex.get("options", []) if x]
            alloc = allocate_exchange_tokens_ap_first(total_tokens, per_token, options, mats, respect_use_flag)
            for it, qty in alloc:
                apply_gain_and_track(it, qty, mats, cum, respect_use_flag)

def _apply_stage_drops(stage: dict, mats: Dict[str, dict], cum: Dict[str, float], respect_use_flag: bool=True):
    """스테이지 기본 드랍 기대치 적용(각 드랍: item×rate)."""
    for d in stage.get("drops", []):
        item = str(d.get("item") or ""); rate = float(d.get("rate") or 0.0)
        if not item: continue
        apply_gain_and_track(item, rate, mats, cum, respect_use_flag)

def _apply_box_event_contents_per_run(event_name: str, stage: dict, event_block: dict, quests_json: dict,
                                      mats: Dict[str, dict], cum: Dict[str, float], respect_use_flag: bool=True) -> bool:
    """
    BOX: base(g/s/b) × contents를 1판당 기대치로 환산하여 적용.
    - stage.box.base_* × contents.qty_per_box 합산
    """
    contents = _fetch_box_contents(stage, event_block, quests_json)
    if not contents:
        return False
    box = stage.get("box") or {}
    base_gold   = float((box.get("gold")   or {}).get("base", 0.0))
    base_silver = float((box.get("silver") or {}).get("base", 0.0))
    base_bronze = float((box.get("bronze") or {}).get("base", 0.0))

    for x in contents.get("gold", []):
        apply_gain_and_track(str(x.get("item") or ""), float(x.get("qty_per_box") or 0.0) * base_gold, mats, cum, respect_use_flag)
    for x in contents.get("silver", []):
        apply_gain_and_track(str(x.get("item") or ""), float(x.get("qty_per_box") or 0.0) * base_silver, mats, cum, respect_use_flag)
    for x in contents.get("bronze", []):
        apply_gain_and_track(str(x.get("item") or ""), float(x.get("qty_per_box") or 0.0) * base_bronze, mats, cum, respect_use_flag)
    return True

def _get_event_block(quests_json: dict, event_name: str, case: str) -> Optional[dict]:
    """특정 이벤트/케이스의 정의 블록 찾기(스테이지 정의 조회 시 사용)."""
    for e in quests_json.get("event_quests", []):
        if str(e.get("event") or "") == event_name and str(e.get("case") or "").lower() == case:
            return e
    return None

# ----------------- 최종 재료 현황 -----------------
def _print_final_materials_summary(w, initial_lack: Dict[str, float],
                                   mats: Dict[str, dict], cum_gained: Dict[str, float],
                                   respect_use_flag: bool=True) -> None:
    """
    로그 마지막에 '최종 재료 현황' 표 출력.
    Columns
    -------
    재료 | 목표(초기 lack) | 획득(누계 기대) | 부족(최종 lack)
    Rules
    -----
    - use=False 재료는 제외
    - 목표/획득/부족 모두 0이면 생략
    - 정렬: 부족 desc → 목표 desc → 재료명 asc
    """
    final_lack = snapshot_lack(mats, respect_use_flag)
    names = set(initial_lack.keys()) | set(cum_gained.keys()) | set(final_lack.keys())
    rows = []
    for name in names:
        # use 플래그 체크
        rec = mats.get(name)
        if respect_use_flag and rec is not None and not rec.get("use", True):
            continue
        tgt = float(initial_lack.get(name, 0.0))
        got = float(cum_gained.get(name, 0.0))
        rem = float(final_lack.get(name, max(0.0, tgt - got)))
        if tgt == 0.0 and got == 0.0 and rem == 0.0:
            continue
        rows.append([name, tgt, got, rem])

    # 정렬
    rows.sort(key=lambda r: (-r[3], -r[1], r[0]))

    if not rows:
        w("(최종 재료 현황: 출력할 항목 없음)")
        return

    headers = ["재료", "목표", "획득", "부족"]
    fmt_rows = [[r[0], f2(r[1]), f2(r[2]), f2(r[3])] for r in rows]
    _print_table(w, headers, fmt_rows, aligns=["left", "right", "right", "right"])

# ----------------- 파일 조작: 프리펜드 -----------------
def _prepend_lines_to_file(filepath: str, lines: List[str]) -> None:
    """
    파일 맨 앞에 lines를 삽입. 실패해도 시뮬은 계속.
    (윈도우에서도 문제없도록 전부 텍스트 재기록 방식)
    """
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

# ----------------- 메인 -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", default="materials.json")
    ap.add_argument("--quests", default="event_quests.json")
    ap.add_argument("--items",  default="event_items.json")
    ap.add_argument("--event",  default=None, help="이벤트명 부분일치 필터(없으면 전부)")
    ap.add_argument("--diff",   default=None, help="박스/레이드 선호 난이도(예: 90++)")
    ap.add_argument("--ap-cost", type=float, default=40.0)
    ap.add_argument("--ignore-use-flag", action="store_true")
    ap.add_argument("--log", default="run_log.txt")
    args = ap.parse_args()

    logger = Logger(args.log, tee=False); w = logger.write

    mats = to_material_index(load_json(args.materials))
    quests_json = load_json(args.quests)
    try:
        items_json = load_json(args.items)
    except Exception:
        items_json = {}

    respect_use_flag = not args.ignore_use_flag
    # ★ 초기 목표(초기 lack) 스냅샷 저장: 최종 현황 표 작성 시 사용
    initial_lack = snapshot_lack(mats, respect_use_flag)

    targets = _list_targets(quests_json, args.event)

    # CE 상태 준비(룰렛 대상만 초기화)
    _init_ce_state_for_targets(targets)

    # AP 풀 계산(자연 AP + 사과)
    total_ap = (
        APPLE_COUNTS.get("gold", 0)   * APPLE_AP_POOL["gold"]   +
        APPLE_COUNTS.get("silver", 0) * APPLE_AP_POOL["silver"] +
        APPLE_COUNTS.get("blue", 0)   * APPLE_AP_POOL["blue"]   +
        APPLE_COUNTS.get("copper", 0) * APPLE_AP_POOL["copper"] +
        float(NATURAL_AP or 0.0)
    )
    w(f"AP 풀: {f2(total_ap)}  | AP/판={f2(args.ap_cost)}  | 로그: 트리거 시점만 기록")
    w(f"# 룰렛 보너스: 시작 {CE_BASE_BONUS}, 4장당 +1, 최대 {CE_MAX_BONUS}")

    if not targets:
        w("대상 이벤트 없음.")
        logger.close()
        print(f"[로그 저장 완료] {logger.filepath}")
        return

    # 누적 상태
    ap_pool = total_ap
    tickets_acc: Dict[str, float] = {}             # 이벤트별 룰렛 티켓 누적
    need_tickets_cache: Dict[str, float] = {}      # 이벤트별 룰렛 티켓 필요량(캐시)
    roulette_refund_per_box: Dict[str, float] = {} # 이벤트별 박스당 환급AP(캐시)
    cum_gained: Dict[str, float] = {}              # 아이템별 누적 획득 기대량

    last_choice_key = None  # 스테이지 변화 감지
    run_idx = 0

    # 이벤트별 '연속 주행 세션' 기록 (예: 산타네모 500판 → 관위대관전 30판 → 산타네모 120판)
    session_segments: List[Tuple[str, int]] = []
    prev_event_name: Optional[str] = None

    while ap_pool >= args.ap_cost:
        run_idx += 1
        # 이번 판 집계 리셋
        global CURRENT_RUN_GAINS; CURRENT_RUN_GAINS = {}

        # 현재 상태에서의 '최적' 후보 계산
        bests_before = _compute_all_bests(
            targets, quests_json, items_json, mats, args.ap_cost, args.diff, respect_use_flag
        )
        global_best = _pick_global_best(bests_before)
        if not global_best:
            w("※ 선택할 스테이지 없음 → 중단")
            break

        ev = global_best["event"]; cs = global_best["case"]
        stage_name = global_best["stage"]; diff = global_best["diff"]
        ce_bonus_for_line = global_best.get("ce_bonus", _get_ce_bonus(ev))
        choice_key = f"{ev}|{cs}|{stage_name}|{diff}"
        stage_changed_trigger = (last_choice_key is None or choice_key != last_choice_key)

        # '연속 세션' 집계 (이벤트가 바뀌면 새 세그먼트)
        if prev_event_name != ev:
            session_segments.append((ev, 1))
            prev_event_name = ev
        else:
            name, cnt = session_segments[-1]
            session_segments[-1] = (name, cnt + 1)

        # AP 차감
        ap_pool -= args.ap_cost

        # 스테이지 정의 조회
        block = _get_event_block(quests_json, ev, cs); stage_def = None
        if block:
            for st in block.get("stages", []):
                if str(st.get("stage") or "") == stage_name and str(st.get("diff") or "") == str(diff):
                    stage_def = st; break
        if not stage_def:
            last_choice_key = choice_key; continue

        # 전/후 스냅샷
        lack_before = snapshot_lack(mats, respect_use_flag)

        # 1) 스테이지 기본 드랍
        _apply_stage_drops(stage_def, mats, cum_gained, respect_use_flag)

        # 2) 케이스별 추가 처리(+룰렛 환급AP)
        opened = 0; refund_gain = 0.0; last_tpr_for_run = 0.0
        if cs == "roulette":
            ce_bonus = _get_ce_bonus(ev)
            last_tpr_for_run = tickets_per_run(stage_def, ce_bonus)  # '판당 티켓'
            tickets_acc[ev] = tickets_acc.get(ev, 0.0) + last_tpr_for_run

            if ev not in need_tickets_cache:
                need_tickets_cache[ev] = extract_need_tickets_from_items(ev, items_json, default=600.0)
            if ev not in roulette_refund_per_box:
                _, ap_refund, _ = compute_per_box_value_and_refund(ev, items_json, mats, respect_use_flag=False)
                roulette_refund_per_box[ev] = float(ap_refund or 0.0)

            need_tk = need_tickets_cache[ev]
            if need_tk > 0 and tickets_acc[ev] >= need_tk:
                opened = int(tickets_acc[ev] // need_tk)
                tickets_acc[ev] -= opened * need_tk
                _open_roulette_boxes_and_apply(ev, opened, items_json, mats, cum_gained, respect_use_flag)
                refund_gain = opened * roulette_refund_per_box.get(ev, 0.0)
                if refund_gain > 0:
                    ap_pool += refund_gain

        elif cs == "box":
            if block:
                _apply_box_event_contents_per_run(ev, stage_def, block, quests_json, mats, cum_gained, respect_use_flag)

        # 2.5) 특수 수급(이벤트명 부분일치)
        _apply_special_per_run_yields(ev, mats, cum_gained, respect_use_flag)

        # 3) 트리거 판정
        lack_after = snapshot_lack(mats, respect_use_flag)

        # (i) 이번 판으로 lack이 0이 된 재료 수집
        lack_zero_items = []
        for name, b in lack_before.items():
            a = lack_after.get(name, 0.0)
            if b > 0 and a == 0:
                lack_zero_items.append((name, b, cum_gained.get(name, 0.0)))

        # (ii) CE 경계/보너스 변화 감지
        ce_trigger_info = None
        if cs == "roulette":
            ce_info = _update_ce_after_run(ev, stage_def, block)
            # 보너스 12 이후엔 로깅 생략(계산은 계속)
            if ce_info["new_bonus"] < CE_MAX_BONUS and (ce_info["gain_int"] > 0 or ce_info["bonus_inc"] > 0):
                ce_trigger_info = ce_info

        # (iii) 실제 로깅 여부
        should_log = stage_changed_trigger or (len(lack_zero_items) > 0) or (ce_trigger_info is not None)
        ce_only_trigger = (ce_trigger_info is not None) and (not stage_changed_trigger) and (len(lack_zero_items) == 0)

        # 4) 트리거 로깅
        if should_log:
            # 헤더(스테이지 변경 or 상태 기록)
            if stage_changed_trigger:
                if cs == "roulette":
                    w(f"[Run {run_idx}] 스테이지 변경 → {ev} [{cs}] {stage_name} ({diff})  | 보너스={ce_bonus_for_line}  | total_eff={global_best['total_eff']}")
                else:
                    w(f"[Run {run_idx}] 스테이지 변경 → {ev} [{cs}] {stage_name} ({diff})  | total_eff={global_best['total_eff']}")
            else:
                if cs == "roulette":
                    ce_state = CE_STATE.get(ev, {"drops_acc": 0.0, "bonus": CE_BASE_BONUS})
                    current_int = int(ce_state["drops_acc"] // 1)
                    w(f"[Run {run_idx}] 상태 기록 → {ev} [{cs}] {stage_name} ({diff})  | 현재 보너스={ce_state['bonus']} | 예장={current_int}장")
                else:
                    w(f"[Run {run_idx}] 상태 기록 → {ev} [{cs}] {stage_name} ({diff})")

            # (A) CE만 변한 경우: 한 줄만 내고 바로 블록 종료(빈 줄 2개)
            if ce_only_trigger:
                w(""); w("")
                last_choice_key = choice_key
                continue

            # (B) lack→0 재료 상세
            for name, b, total in lack_zero_items:
                w(f"  · [Run {run_idx}] 재료 충족: {name}  (부족 {f2(b)} → 0)  | 누적 획득 {f2(total)}")

            # (C) CE 상세(동시 트리거일 때만 출력)
            if ce_trigger_info is not None and not ce_only_trigger:
                gi = ce_trigger_info["gain_int"]; bi = ce_trigger_info["bonus_inc"]; cur_int = ce_trigger_info["new_copies_int"]
                if gi > 0: w(f"  · 예장 드랍: +{gi}장 (현재 {cur_int}장, 누적 기대 {f2(ce_trigger_info['new_drops'])}장)")
                if bi > 0: w(f"    → 보너스 +{bi} ⇒ 현재 {ce_trigger_info['new_bonus']}  (다음 +1까지 {f2(ce_trigger_info['rem_to_next'])}장 기대)")

            # (D) 이번 판 변화 표
            w("변동된 결과 (이 판):")
            _print_gain_table(w, CURRENT_RUN_GAINS, lack_before, lack_after, cum_gained)

            # (E) 룰렛/예장 한 줄 요약 (+판당 티켓)
            if cs == "roulette":
                need_tk = need_tickets_cache.get(ev, 0.0)
                ce_state = CE_STATE.get(ev, {"drops_acc": 0.0, "bonus": CE_BASE_BONUS})
                cur_int = int(ce_state["drops_acc"] // 1)
                suffix = ""
                if ce_state["bonus"] < CE_MAX_BONUS:
                    rem = max(0.0, CE_COPIES_PER_PLUS - (ce_state["drops_acc"] % CE_COPIES_PER_PLUS))
                    suffix = f", 다음 +1까지 {f2(rem)}장"
                w(f"룰렛: 티켓 {f2(tickets_acc.get(ev, 0.0))}/{f2(need_tk)} (+{f2(last_tpr_for_run)}/판) | "
                  f"환급AP +{f2(refund_gain)} | AP 풀 {f2(ap_pool)} | 예장 {cur_int}장(보너스 {ce_state['bonus']}{suffix})")

            # (F) 현재 효율 요약(재계산)
            bests_after = _compute_all_bests(
                targets, quests_json, items_json, mats, args.ap_cost, args.diff, respect_use_flag
            )
            w("\n현재 이벤트 효율 요약:")
            _print_eff_table(w, bests_after, args.ap_cost)

            # 블록 종료 후 빈 줄 2개(가독성)
            w(""); w("")

        # 다음 루프 비교용
        last_choice_key = choice_key

    # 종료 정보(요약) + 최종 재료 현황 표
    w(f"# 종료: 총 실행 {run_idx}판, 잔여 AP 풀={f2(ap_pool)}")
    w("")  # 구분
    w("## 최종 재료 현황 (목표/획득/부족)")
    _print_final_materials_summary(w, initial_lack, mats, cum_gained, respect_use_flag)
    logger.close()

    # ==== 세션 요약: 콘솔 출력 + 로그 파일 맨 앞에 프리펜드 ====
    header = ["## 주회 세션(순서대로) — 이벤트별 연속 주행 구간 요약"]
    body = [f"  {i+1:>2}. {name} — {cnt}판" for i, (name, cnt) in enumerate(session_segments)]
    lines = header + body + ["", "## 원본 로그 ↓", ""]
    # 콘솔
    for ln in header + body:
        print(ln)
    print()  # 빈 줄
    # 파일 선두 삽입
    _prepend_lines_to_file(args.log, lines)

    print(f"[로그 저장 완료] {logger.filepath}")

# ----------------- 레이드 베스트(하단 배치: 선언 순서 무관) -----------------
def compute_raid_best_stage(event_name: str, quests_json: dict, mats: Dict[str, dict],
                            ap_cost_per_run: float, respect_use_flag: bool=True,
                            prefer_diff: Optional[str]=None) -> Optional[dict]:
    """
    레이드 케이스: 스테이지/AP 최대 후보.
    - 레이드는 별도 룰렛/박스 없음
    """
    events = quests_json.get("event_quests", []); candidates = []
    for e in events:
        if str(e.get("event") or "") != event_name or str(e.get("case") or "").lower() != "raid":
            continue
        for st in e.get("stages", []):
            stage_val = stage_value_per_run(st.get("drops", []), mats, respect_use_flag)
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

if __name__ == "__main__":
    main()
