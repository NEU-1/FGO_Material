# -*- coding: utf-8 -*-
"""
FGO 프리퀘 효율 계산기
======================
free_quests.json + materials.json 기반으로
프리퀘별 가중 AP효율을 계산, 효율순으로 출력.

공식:
  아이템 AP효율 = AP / 드랍율(%) * 100   (1개 획득 기대 AP, 낮을수록 좋음)
  프리퀘 효율   = Σ (materials.ap_per_item / 아이템AP효율)  (부족 아이템만)

사용법:
  python calc_freequest.py
  python calc_freequest.py --materials materials.json --drops free_quests.json
  python calc_freequest.py --top 50 --detail
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ===========================================================================
# 유틸
# ===========================================================================

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def f2(x: float) -> str:  return f"{float(x):,.2f}"
def f0(x) -> str:
    try: return f"{int(round(float(x))):,}"
    except: return str(x)

try:
    from wcwidth import wcswidth as _wcswidth
    def dw(s: str) -> int:
        w = _wcswidth(s); return w if w >= 0 else len(s)
except ImportError:
    def dw(s: str) -> int:
        w = 0
        for ch in s:
            if unicodedata.combining(ch): continue
            w += 2 if unicodedata.east_asian_width(ch) in ("F","W") else 1
        return w

# ---------------------------------------------------------------------------
# 표 출력
# ---------------------------------------------------------------------------

@dataclass
class TS:
    fmt: str = "text"; style: str = "box"

_B = {"top":("┌","┬","┐"),"mid":("├","┼","┤"),"bot":("└","┴","┘"),"h":"─","v":"│"}
_A = {"top":("+","+","+"),"mid":("+","+","+"),"bot":("+","+","+"),"h":"-","v":"|"}

def _line(ws,l,m,r,f): return l+"".join(f*(w+2)+(m if i<len(ws)-1 else r) for i,w in enumerate(ws))
def _pad(t,w,a):
    p=max(0,w-dw(t))
    if a=="right": return " "+" "*p+t+" "
    if a=="center": lp=p//2; return " "+" "*lp+t+" "*(p-lp)+" "
    return " "+t+" "*p+" "

def ptable(hdr,rows,al=None,ts=None):
    ts=ts or TS()
    if not al: al=["left"]*len(hdr)
    if not rows: print("(없음)"); return
    if ts.fmt=="md":
        am={"left":":---","right":"---:","center":":---:"}
        print("| "+" | ".join(str(h) for h in hdr)+" |")
        print("| "+" | ".join(am.get(a,":---") for a in al)+" |")
        for r in rows: print("| "+" | ".join(str(c) for c in r)+" |")
        return
    c=_B if ts.style=="box" else _A; v=c["v"]
    ws=[max(dw(str(h)),*(dw(str(r[i])) for r in rows)) for i,h in enumerate(hdr)]
    print(_line(ws,*c["top"],c["h"]))
    print(v+v.join(_pad(str(h),ws[i],"center") for i,h in enumerate(hdr))+v)
    print(_line(ws,*c["mid"],c["h"]))
    for r in rows: print(v+v.join(_pad(str(r[i]),ws[i],al[i]) for i in range(len(hdr)))+v)
    print(_line(ws,*c["bot"],c["h"]))

# ===========================================================================
# 데이터 모델
# ===========================================================================

@dataclass
class QuestEff:
    area: str; quest: str; ap: float; eff: float
    items: List[Tuple[str, float, float, float]]
    # (아이템, 드랍율%, 아이템AP효율, 기여도)

# ===========================================================================
# 효율 계산
# ===========================================================================

def compute(quests_data: dict, materials: dict) -> Tuple[
    Dict[str, Tuple[float, str, str, float]],  # 아이템별 최고효율
    List[QuestEff],                              # 프리퀘별 효율
]:
    """
    1단계: 아이템별 최고 드랍율 프리퀘 → AP효율 = AP / 드랍율(%) * 100
    2단계: 프리퀘 효율 = Σ(아이템AP효율 × 해당프리퀘드랍율/100) / 프리퀘AP
           부족 아이템(need>0)만 합산
    """
    mat_idx: Dict[str, dict] = {}
    for m in materials.get("materials", []):
        name = str(m.get("item") or "").strip()
        if name:
            mat_idx[name] = {
                "ap": float(m.get("ap_per_item") or 0),
                "need": float(m.get("need") or 0),
            }

    fqs = quests_data.get("free_quests", [])
    # 관위 연루전은 최고 난이도(Ⅶ)만 포함
    fqs = [q for q in fqs
           if q.get("area") != "관위 연루전" or "Ⅶ" in q.get("quest", "")]

    # 1단계: 아이템별 최고 AP효율 프리퀘 (AP/개가 가장 낮은 것)
    # {아이템: (best_rate, area, quest, ap)}
    item_best: Dict[str, Tuple[float, str, str, float]] = {}
    for q in fqs:
        area  = q.get("area", "")
        quest = q.get("quest", "")
        ap    = float(q.get("ap", 0))
        if ap <= 0:
            continue
        for item, rate in q.get("drops", {}).items():
            if item not in mat_idx:
                continue
            rate = float(rate)
            if rate <= 0:
                continue
            ap_per_item = ap / rate * 100
            prev = item_best.get(item)
            if prev is None:
                item_best[item] = (rate, area, quest, ap)
            else:
                prev_ap_per = prev[3] / prev[0] * 100
                if ap_per_item < prev_ap_per:
                    item_best[item] = (rate, area, quest, ap)

    # 아이템별 AP효율 캐시: AP / 드랍율(%) * 100
    item_ap_eff: Dict[str, float] = {}
    for item, (rate, _, _, ap) in item_best.items():
        item_ap_eff[item] = ap / rate * 100

    # 2단계: 프리퀘별 효율 = Σ(아이템AP효율 × 드랍율/100) / AP
    needed = {n for n, v in mat_idx.items() if v["need"] > 0}

    results: List[QuestEff] = []
    for q in fqs:
        area  = q.get("area", "")
        quest = q.get("quest", "")
        ap    = float(q.get("ap", 0))
        if ap <= 0:
            continue

        details = []
        for item, rate in q.get("drops", {}).items():
            if item not in needed or item not in item_ap_eff:
                continue
            rate = float(rate)
            if rate <= 0:
                continue
            best_ape = item_ap_eff[item]
            value = best_ape * rate / 100       # 이 퀘스트에서 이 아이템의 AP가치
            details.append((item, rate, best_ape, value))

        if not details:
            continue
        details.sort(key=lambda x: x[3], reverse=True)
        total_value = sum(v for _, _, _, v in details)
        eff = total_value / ap
        results.append(QuestEff(
            area=area, quest=quest, ap=ap,
            eff=eff, items=details,
        ))

    results.sort(key=lambda x: x.eff, reverse=True)
    return item_best, results

# ===========================================================================
# 출력
# ===========================================================================

def print_item_table(item_best, mat_idx, ts):
    all_items = [(it, r, a, q, ap) for it, (r, a, q, ap) in item_best.items()]
    # AP효율순
    all_items.sort(key=lambda x: x[4]/x[1]*100)

    print(f"\n## 아이템별 프리퀘 최고효율 (전체 {len(all_items)}개)\n")
    rows = []
    for i, (it, rate, area, quest, ap) in enumerate(all_items):
        ape = ap / rate * 100
        mat_ap = mat_idx.get(it, {}).get("ap", 0)
        need = mat_idx.get(it, {}).get("need", 0)
        diff = ape - mat_ap if mat_ap > 0 else 0
        need_mark = f"{f0(need)}" if need > 0 else "-"
        rows.append([f"{i+1:>2}", it, f"{rate:.1f}%", f"{area}/{quest}",
                      f0(ap), f2(ape), f2(mat_ap), f"{diff:+.2f}", need_mark])
    ptable(["#","아이템","드랍율","최고효율 퀘스트","AP","프리퀘AP/개","기존AP/개","차이","need"],
           rows, ["right","left","right","left","right","right","right","right","right"], ts)


def print_quest_table(results, ts, top_n=30, detail=False):
    print(f"\n## 프리퀘 효율 순위 (부족재료 기준, Top {min(top_n, len(results))})\n")

    rows = []
    for i, r in enumerate(results[:top_n]):
        items_str = ", ".join(f"{it}({f2(c)})" for it, _, _, c in r.items[:3])
        if len(r.items) > 3:
            items_str += f" 외{len(r.items)-3}"
        rows.append([f"{i+1:>3}", r.area, r.quest, f0(r.ap),
                      f2(r.eff), items_str])
    ptable(["#","지역","퀘스트","AP","효율","부족 아이템 (AP가치)"],
           rows, ["right","left","left","right","right","left"], ts)

    if detail:
        print(f"\n## 상위 5개 상세\n")
        print(f"  효율 = Σ(아이템AP효율 × 드랍율/100) / AP\n")
        for r in results[:5]:
            total_v = sum(v for _, _, _, v in r.items)
            print(f"  ▶ {r.area} / {r.quest}  (AP {f0(r.ap)}, 효율 {f2(r.eff)}, 총가치 {f2(total_v)})")
            for it, rate, ape, v in r.items:
                print(f"    · {it}  드랍={rate:.1f}%  최고AP/개={f2(ape)}  가치={f2(v)}")
            print()

# ===========================================================================
# main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="FGO 프리퀘 효율 계산기")
    parser.add_argument("--materials", default="materials.json")
    parser.add_argument("--drops", default="free_quests.json",
                        help="프리퀘 드랍률 json")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--detail", action="store_true")
    parser.add_argument("--log", default="freequest_log.txt",
                        help="결과 저장 텍스트 파일")
    parser.add_argument("--table-format", choices=["text","md","csv"], default="text")
    parser.add_argument("--table-style", choices=["box","ascii"], default="box")
    args = parser.parse_args()

    ts = TS(fmt=args.table_format, style=args.table_style)

    # print를 콘솔 + 파일 동시 출력으로 교체
    import builtins, io
    _log_f = open(args.log, "w", encoding="utf-8")
    _orig_print = builtins.print
    def _tee_print(*a, **kw):
        _orig_print(*a, **kw)
        kw.pop("flush", None)
        _orig_print(*a, **kw, file=_log_f)
    builtins.print = _tee_print

    # 로드
    print("## 데이터 로드")
    materials = load_json(args.materials)
    mat_idx = {}
    for m in materials.get("materials", []):
        name = str(m.get("item","")).strip()
        if name:
            mat_idx[name] = {"ap": float(m.get("ap_per_item") or 0),
                             "need": float(m.get("need") or 0)}
    print(f"  materials.json: {len(mat_idx)}개")

    quests_data = load_json(args.drops)
    fqs_all = quests_data.get("free_quests", [])
    fqs = [q for q in fqs_all
           if q.get("area") != "관위 연루전" or "Ⅶ" in q.get("quest", "")]
    print(f"  free_quests.json: {len(fqs_all)}개 → 필터 후 {len(fqs)}개 프리퀘")

    # 매칭 확인
    sheet_items = set()
    for q in fqs:
        sheet_items.update(q.get("drops", {}).keys())
    needed_kr = {n for n, v in mat_idx.items() if v["need"] > 0}
    matched = sheet_items & needed_kr
    unmapped = needed_kr - sheet_items
    print(f"  매칭: {len(matched)}개")
    if unmapped:
        print(f"  미매칭: {', '.join(sorted(unmapped))}")

    # 계산
    item_best, results = compute(quests_data, materials)
    print(f"  효율 계산: 프리퀘 {len(results)}개")

    # 출력
    print_item_table(item_best, mat_idx, ts)
    print_quest_table(results, ts, top_n=args.top, detail=args.detail)

    # 로그 마무리
    builtins.print = _orig_print
    _log_f.close()
    print(f"\n[저장 완료] {args.log}")


if __name__ == "__main__":
    main()
