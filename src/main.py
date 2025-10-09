#!/usr/bin/env python3
"""
Exchange rate updater (buffered-rate via hybrid EMA/volatility/percentile algorithm)

Core responsibilities:
- Fetch the current USD rate from BCA.
- Manage twice-daily execution using candidates (first run provisional, second run finalizes).
- Compute buffered_rate using a hybrid algorithm based on history:
  base = EMA(history, ema_window)
  buffer = max(sigma(history, vol_window) * sigma_factor,
               max(0, percentile(history, pct_window, pct) - base))
  hybrid = base + buffer + margin
  buffered_rate = EMA(hybrid_series, smooth_window) [use last value]

Files:
- data/history.json     (one record per date)
- data/candidates.json  (temporary candidates for today)
- data/today.json       (today's actual + buffered + final)

"""
from __future__ import annotations
import json
import re
import sys
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Dict

import requests
from bs4 import BeautifulSoup

# ----------------------------
# CONFIG / CONSTANTS
# ----------------------------
DATA_DIR = Path("data")
HISTORY_FILE = DATA_DIR / "history.json"
CANDIDATES_FILE = DATA_DIR / "candidates.json"
TODAY_FILE = DATA_DIR / "today.json"

# Hybrid buffered-rate parameters (see module docstring)
EMA_WINDOW = 30
VOL_WINDOW = 30
PCT_WINDOW = 90
PCT = 95.0
SIGMA_FACTOR = 2.0
MARGIN = 300
SMOOTH_WINDOW = 7

# HTTP / BCA
BCA_RATE_URL = "https://www.bca.co.id/id/informasi/kurs"
HTTP_TIMEOUT = 15

# Jakarta timezone offset
JAKARTA_UTC_OFFSET_HOURS = 7

# Ordered rate keys mapping to table columns 2..7
RATE_KEYS = [
    "e_rate_buying_rate",       # column 2 (td index 1)
    "e_rate_selling_rate",      # column 3 (td index 2)
    "tt_counter_buying_rate",   # column 4 (td index 3)
    "tt_counter_selling_rate",  # column 5 (td index 4)
    "bank_notes_buying_rate",   # column 6 (td index 5)
    "bank_notes_selling_rate",  # column 7 (td index 6)
]

# ----------------------------
# UTILITIES (I/O & datetime)
# ----------------------------
def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None

def write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def jakarta_today_date_str() -> str:
    # Current date in Asia/Jakarta (UTC+7)
    now_utc = datetime.now(timezone.utc)
    now_jakarta = now_utc + timedelta(hours=JAKARTA_UTC_OFFSET_HOURS)
    return now_jakarta.date().isoformat()

def now_iso_ts() -> str:
    return datetime.now(timezone.utc).isoformat()

# ----------------------------
# FETCHING
# ----------------------------
def _clean_cell_to_int(raw: str) -> Optional[int]:
    if raw is None:
        return None
    # Remove trailing decimal portion like ',00' or '.00'
    raw = re.sub(r'([.,]\d{1,2})\s*$', '', raw)
    # Keep digits only
    digits = re.sub(r'[^\d]', '', raw)
    if digits:
        try:
            return int(digits)
        except Exception:
            return None
    return None

def fetch_rate_from_bca() -> tuple[Dict[str, int], str]:
    """
    Fetch the BCA page and extract columns 2..7 for the USD row.
    Returns (rates_dict, iso_timestamp_utc).
    rates_dict contains keys (RATE_KEYS) with integer values.
    """
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.bca.co.id/",
        "Connection": "keep-alive",
    }

    html = None
    for attempt in range(3):
        try:
            resp = session.get(BCA_RATE_URL, headers=headers, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            html = resp.text
            break
        except Exception:
            if attempt == 2:
                raise
            continue

    soup = BeautifulSoup(html, "lxml")
    found: Optional[Dict[str, int]] = None

    # Attempt: find a row containing USD and extract columns 2..7 (td indices 1..6)
    for tr in soup.find_all("tr"):
        txt = (tr.get_text() or "").upper()
        if "USD" in txt:
            tds = tr.find_all("td")
            if len(tds) >= 7:
                values: Dict[str, Optional[int]] = {}
                for i, key in enumerate(RATE_KEYS, start=1):
                    raw = tds[i].get_text(strip=True)
                    values[key] = _clean_cell_to_int(raw)
                # If any value is missing, attempt fallback using numbers found in row
                if not all(values[k] is not None for k in RATE_KEYS):
                    nums = re.findall(r"\d{4,}", tr.get_text())
                    if len(nums) >= len(RATE_KEYS):
                        # Map first N numbers to keys in order
                        for idx, key in enumerate(RATE_KEYS):
                            try:
                                values[key] = int(nums[idx])
                            except Exception:
                                values[key] = None
                    elif len(nums) >= 1:
                        # If only a few numbers found, use the first found number as fallback for missing ones
                        fallback_val = int(nums[0])
                        for key in RATE_KEYS:
                            if values[key] is None:
                                values[key] = fallback_val
                # If still missing any, skip this row and continue searching
                if all(values[k] is not None for k in RATE_KEYS):
                    # convert Optional[int] -> int
                    found = {k: int(values[k]) for k in RATE_KEYS}
                    break
            else:
                # Fallback: parse large numbers in the row and try to map
                nums = re.findall(r"\d{4,}", tr.get_text())
                if len(nums) >= len(RATE_KEYS):
                    found = {}
                    for idx, key in enumerate(RATE_KEYS):
                        found[key] = int(nums[idx])
                    break
                elif len(nums) >= 1:
                    # If only one number found, use it for all keys as a last resort
                    val = int(nums[0])
                    found = {k: val for k in RATE_KEYS}
                    break

    if found is None:
        raise RuntimeError("USD rate not found when parsing BCA page.")
    return found, now_iso_ts()

def fetch_current_rate() -> tuple[Dict[str, int], str]:
    return fetch_rate_from_bca()

# ----------------------------
# HISTORY / CANDIDATES management
# ----------------------------
def load_history_rates() -> Dict[str, List[int]]:
    """Return a dict mapping each rate key to a list of historical integer rates from history.json."""
    hist = read_json(HISTORY_FILE)
    if not hist or not isinstance(hist, list):
        return {k: [] for k in RATE_KEYS}
    rates: Dict[str, List[int]] = {k: [] for k in RATE_KEYS}
    for e in hist:
        if isinstance(e, dict):
            for k in RATE_KEYS:
                v = e.get(k)
                if v is None:
                    # legacy support: check "rate" or "actual" if present (map to bank_notes_selling_rate fallback)
                    if "rate" in e and k == "bank_notes_selling_rate":
                        try:
                            rates[k].append(int(e["rate"]))
                        except Exception:
                            pass
                    elif "actual" in e and k == "bank_notes_selling_rate":
                        try:
                            rates[k].append(int(e["actual"]))
                        except Exception:
                            pass
                else:
                    try:
                        rates[k].append(int(v))
                    except Exception:
                        pass
    return rates

def load_history_entries():
    raw = read_json(HISTORY_FILE)
    if not raw:
        return []
    return raw

def save_history_entry(date_s: str, rates: Dict[str, int], buffered: Dict[str, int], source_ts: Dict[str, str], note: Optional[str] = None):
    entries = load_history_entries()
    # remove any existing entry for today
    entries = [e for e in entries if e.get("date") != date_s]
    new = {"date": date_s, "source_ts": source_ts}
    # include all rates and buffered rates
    for k in RATE_KEYS:
        new[k] = int(rates.get(k, 0))
        new[f"{k}_buffered"] = int(buffered.get(k, 0))
    if note:
        new["note"] = note
    entries.append(new)
    try:
        entries.sort(key=lambda e: e.get("date"))
    except Exception:
        pass
    write_json(HISTORY_FILE, entries)

def load_candidates() -> List[dict]:
    raw = read_json(CANDIDATES_FILE)
    if not raw or not isinstance(raw, list):
        return []
    today = jakarta_today_date_str()
    today_only = [c for c in raw if c.get("date") == today]
    return today_only

def save_candidates(cands: List[dict]):
    write_json(CANDIDATES_FILE, cands)

def write_today_json(actual: Dict[str, int], buffered: Dict[str, int], final: bool, candidates_count: int, selected_source_ts: Dict[str, str]):
    obj = {
        "date": jakarta_today_date_str(),
        "final": bool(final),
        "candidates_count": int(candidates_count),
        "selected_source_ts": selected_source_ts,
        "generated_ts_utc": now_iso_ts(),
    }
    # include actual rates and buffered rates
    for k in RATE_KEYS:
        obj[k] = int(actual.get(k, 0))
        obj[f"{k}_buffered"] = int(buffered.get(k, 0))
    write_json(TODAY_FILE, obj)

# ----------------------------
# ALGORITHM (EMA + volatility + percentile + margin, then EMA smoothing)
# ----------------------------
def ema(values: List[float] | List[int], window: int) -> float:
    if not values:
        return 0.0
    if window <= 1 or len(values) == 1:
        return float(values[-1])
    alpha = 2.0 / (window + 1.0)
    ema_val = float(values[0])
    for v in values[1:]:
        ema_val = alpha * float(v) + (1.0 - alpha) * ema_val
    return ema_val

def percentile(values: List[float] | List[int], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    k = (len(vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return vals[int(k)]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def round_up_to_hundred(x: int) -> int:
    x_int = int(x)
    return ((x_int + 99) // 100) * 100

def compute_buffered_rate_from_history(hist_rates: List[int], current_rate: int) -> int:
    """
    Compute buffered_rate using hybrid algorithm applied over the series
    (history + current_rate). Returns the last smoothed value as int.
    """
    series = list(hist_rates) + [int(current_rate)]
    if not series:
        return int(current_rate)

    hybrid_series: List[float] = []
    for i in range(len(series)):
        # Windows
        ema_slice = series[max(0, i - EMA_WINDOW + 1): i + 1]
        vol_slice = series[max(0, i - VOL_WINDOW + 1): i + 1]
        pct_slice = series[max(0, i - PCT_WINDOW + 1): i + 1]

        base = ema(ema_slice, EMA_WINDOW)

        if len(vol_slice) > 1:
            sigma = statistics.pstdev([float(x) for x in vol_slice])
        else:
            sigma = 0.0

        buffer_sigma = sigma * float(SIGMA_FACTOR)
        pct_val = percentile(pct_slice, PCT) if pct_slice else 0.0
        buffer_pct = max(0.0, pct_val - base)

        buffer_val = max(buffer_sigma, buffer_pct)
        hybrid = base + buffer_val + float(MARGIN)
        hybrid_series.append(hybrid)

    # Smoothed buffered rate is EMA over hybrid_series
    smooth = ema(hybrid_series, SMOOTH_WINDOW)
    return round_up_to_hundred(int(round(smooth)))

# ----------------------------
# SELECTION RULES (first/second execution) - extended for groups
# ----------------------------
def _choose_group_value(a_val: int, b_val: int, yesterday_val: Optional[int]) -> int:
    """
    Given two candidate values (for deciding a single scalar group),
    apply the original single-rate logic:
      - if first == yesterday and second != yesterday -> pick second
      - if second == yesterday and first != yesterday -> pick first
      - else -> pick the larger (tie -> first)
    """
    if yesterday_val is not None:
        if a_val == yesterday_val and b_val != yesterday_val:
            return b_val
        if b_val == yesterday_val and a_val != yesterday_val:
            return a_val
    # tie or both different from yesterday -> pick larger (a preferred on tie)
    return a_val if a_val >= b_val else b_val

def choose_final_candidate_grouped(cands: List[dict], yesterday_entry: Optional[dict]) -> Dict:
    """
    cands: list of dicts each with keys: date, source_ts, and RATE_KEYS
    yesterday_entry: dict entry from history (may contain RATE_KEYS)
    Returns a dict with:
      - selected_rates: mapping RATE_KEYS -> selected int
      - selected_source_ts: mapping group keys -> source_ts of chosen candidate for that group
    Grouping rules:
      - E-rate group: keys 0 and 1 (buy, sell)
      - TT-counter group: keys 2 and 3
      - Bank-notes group: keys 4 and 5
    Selection per group:
      - If there is only one candidate, use that candidate's values
      - If two candidates:
         * if first.rate == yesterday and second.rate != yesterday -> pick second
         * if second.rate == yesterday and first.rate != yesterday -> pick first
         * else when all different -> compare selling rates between the two candidates and pick the candidate with higher selling rate; then take that candidate's buy & sell for history.
    """
    if not cands:
        raise RuntimeError("No candidates to choose from.")

    # If only one candidate, straightforward
    if len(cands) == 1:
        cand = cands[0]
        selected_rates = {k: int(cand.get(k, 0)) for k in RATE_KEYS}
        selected_source_ts = { "e_rate": cand.get("source_ts", ""), "tt_counter": cand.get("source_ts", ""), "bank_notes": cand.get("source_ts", "")}
        return {"selected_rates": selected_rates, "selected_source_ts": selected_source_ts}

    # two candidates - sort chronologically
    c_sorted = sorted(cands, key=lambda x: x.get("source_ts", ""))
    a, b = c_sorted[0], c_sorted[1]

    # yesterday values per key if available
    yr = yesterday_entry or {}
    # Build result
    selected_rates: Dict[str, int] = {}
    selected_source_ts: Dict[str, str] = {}

    # Define groups: tuples of (buy_key, sell_key, group_name)
    groups = [
        (RATE_KEYS[0], RATE_KEYS[1], "e_rate"),
        (RATE_KEYS[2], RATE_KEYS[3], "tt_counter"),
        (RATE_KEYS[4], RATE_KEYS[5], "bank_notes"),
    ]

    for buy_k, sell_k, gname in groups:
        a_buy = int(a.get(buy_k, 0))
        a_sell = int(a.get(sell_k, 0))
        b_buy = int(b.get(buy_k, 0))
        b_sell = int(b.get(sell_k, 0))
        y_buy = yr.get(buy_k) if yr else None
        y_sell = yr.get(sell_k) if yr else None

        # If either candidate equals yesterday (compare pair-wise by both buy+sell equality),
        # follow original per-group "if first==yesterday and second!=yesterday -> pick second" semantics.
        # We'll consider equality by both buy and sell matching yesterday's buy & sell if yesterday provided both.
        picked_candidate = None

        if y_buy is not None and y_sell is not None:
            # check full equality with yesterday (both buy & sell)
            a_equals_yesterday = (a_buy == int(y_buy) and a_sell == int(y_sell))
            b_equals_yesterday = (b_buy == int(y_buy) and b_sell == int(y_sell))
            if a_equals_yesterday and not b_equals_yesterday:
                picked_candidate = b
            elif b_equals_yesterday and not a_equals_yesterday:
                picked_candidate = a

        if picked_candidate is None:
            # all different or unable to compare with yesterday -> compare selling rates between a and b
            # pick the candidate with higher selling rate (tie -> a)
            if a_sell >= b_sell:
                picked_candidate = a
            else:
                picked_candidate = b

        # assign selected buy & sell from picked_candidate
        selected_rates[buy_k] = int(picked_candidate.get(buy_k, 0))
        selected_rates[sell_k] = int(picked_candidate.get(sell_k, 0))
        selected_source_ts[gname] = picked_candidate.get("source_ts", "")

    return {"selected_rates": selected_rates, "selected_source_ts": selected_source_ts}

# ----------------------------
# MAIN FLOW
# ----------------------------
def main() -> int:
    ensure_data_dir()

    # load history rates for model building (dict of lists per key)
    hist_rates_map = load_history_rates()

    # fetch current observation (dict of rates)
    try:
        actual_rates, src_ts = fetch_current_rate()
    except Exception as e:
        print("ERROR: cannot fetch current rate:", e, file=sys.stderr)
        return 2

    today_s = jakarta_today_date_str()

    # Append candidate for today (preserve only today's candidates)
    candidates = load_candidates()
    # build candidate dict with all rate keys
    new_cand = {"date": today_s, "source_ts": src_ts}
    for k in RATE_KEYS:
        new_cand[k] = int(actual_rates.get(k, 0))
    candidates.append(new_cand)
    # keep only the last two candidates in chronological order
    candidates = sorted(candidates, key=lambda x: x.get("source_ts", ""))[-2:]
    save_candidates(candidates)

    # Get yesterday entry and simplify access
    yesterday_entry = None
    history_entries = load_history_entries()
    if history_entries:
        try:
            yesterday_s = (datetime.fromisoformat(today_s) - timedelta(days=1)).date().isoformat()
        except Exception:
            yesterday_s = None
        if yesterday_s:
            last_entry_for_yesterday = next((e for e in history_entries[::-1] if e.get("date") == yesterday_s), None)
            if last_entry_for_yesterday:
                yesterday_entry = last_entry_for_yesterday

    # If only one candidate -> provisional: compute buffered per rate and write today.json
    if len(candidates) == 1:
        print("First execution today: candidate recorded (provisional).")
        provisional_buffered: Dict[str, int] = {}
        for k in RATE_KEYS:
            try:
                provisional_buffered[k] = compute_buffered_rate_from_history(hist_rates_map.get(k, []), int(actual_rates.get(k, 0)))
            except Exception as e:
                print(f"Warning: provisional buffer computation failed for {k}: {e}", file=sys.stderr)
                provisional_buffered[k] = round_up_to_hundred(int(actual_rates.get(k, 0)) + int(MARGIN))
        write_today_json(actual=actual_rates, buffered=provisional_buffered, final=False, candidates_count=1, selected_source_ts={"e_rate": src_ts, "tt_counter": src_ts, "bank_notes": src_ts})
        print(f"Provisional actual rates: {actual_rates}")
        return 0

    # Two or more candidates -> finalize per grouped rules
    if len(candidates) >= 2:
        chosen_group = choose_final_candidate_grouped(candidates, yesterday_entry)
        selected_rates = chosen_group.get("selected_rates", {})
        selected_source_ts = chosen_group.get("selected_source_ts", {})

        buffered_map: Dict[str, int] = {}
        for k in RATE_KEYS:
            try:
                buffered_map[k] = compute_buffered_rate_from_history(hist_rates_map.get(k, []), int(selected_rates.get(k, 0)))
            except Exception as e:
                print(f"Buffer computation error for {k}; using margin fallback: {e}", file=sys.stderr)
                buffered_map[k] = round_up_to_hundred(int(selected_rates.get(k, 0)) + int(MARGIN))

        # Save finalized history entry (one-per-day)
        # source_ts will be a mapping per group (e_rate, tt_counter, bank_notes)
        save_history_entry(date_s=today_s, rates=selected_rates, buffered=buffered_map, source_ts=selected_source_ts, note=f"finalized from {len(candidates)} candidates")

        # Clear today's candidates
        save_candidates([])

        # Write today.json final
        write_today_json(actual=selected_rates, buffered=buffered_map, final=True, candidates_count=len(candidates), selected_source_ts=selected_source_ts)
        print(f"Finalized today's rates (groups): actual={selected_rates}, buffered={buffered_map} (candidates={len(candidates)})")
        return 0

    # fallback (shouldn't hit)
    fallback_buffered: Dict[str, int] = {}
    for k in RATE_KEYS:
        fallback_buffered[k] = round_up_to_hundred(int(actual_rates.get(k, 0)) + int(MARGIN))
    write_today_json(actual=actual_rates, buffered=fallback_buffered, final=False, candidates_count=len(candidates), selected_source_ts={"e_rate": src_ts, "tt_counter": src_ts, "bank_notes": src_ts})
    return 0


if __name__ == "__main__":
    sys.exit(main())