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
from typing import List, Optional

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
def fetch_rate_from_bca() -> tuple[int, str]:
    """
    Fetch the BCA page directly (server-side; no CORS proxy).
    Sends realistic headers to avoid 403, retries briefly,
    then parses the row containing 'USD'.
    Returns (rate_int, iso_timestamp_utc).
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
    found: Optional[int] = None

    # Attempt: find a row containing USD and extract the 7th <td>
    for tr in soup.find_all("tr"):
        txt = (tr.get_text() or "").upper()
        if "USD" in txt:
            tds = tr.find_all("td")
            if len(tds) >= 7:
                raw = tds[6].get_text(strip=True)
                # Remove trailing decimal portion like ',00' or '.00'
                raw = re.sub(r'([.,]\d{1,2})\s*$', '', raw)
                # Keep digits only
                digits = re.sub(r'[^\d]', '', raw)
                if digits:
                    found = int(digits)
                    break
            # Fallback: parse all large numbers in the row and pick the largest
            nums = re.findall(r"\d{4,}", tr.get_text())
            if nums:
                found = max(int(n) for n in nums)
                break

    if found is None:
        raise RuntimeError("USD rate not found when parsing BCA page.")
    return found, now_iso_ts()

def fetch_current_rate() -> tuple[int, str]:
    return fetch_rate_from_bca()

# ----------------------------
# HISTORY / CANDIDATES management
# ----------------------------
def load_history_rates() -> List[int]:
    """Return a list of historical integer rates from history.json."""
    hist = read_json(HISTORY_FILE)
    if not hist or not isinstance(hist, list):
        return []
    rates = []
    for e in hist:
        if isinstance(e, dict):
            if "rate" in e:
                rates.append(int(e["rate"]))
            elif "actual" in e:
                rates.append(int(e["actual"]))
    return rates

def load_history_entries():
    raw = read_json(HISTORY_FILE)
    if not raw:
        return []
    return raw

def save_history_entry(date_s: str, rate: int, buffered: int, source_ts: str, note: Optional[str] = None):
    entries = load_history_entries()
    # remove any existing entry for today
    entries = [e for e in entries if e.get("date") != date_s]
    new = {"date": date_s, "rate": int(rate), "buffered_rate": int(buffered), "source_ts": source_ts}
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

def write_today_json(actual: int, buffered: int, final: bool, candidates_count: int, selected_source_ts: str):
    obj = {
        "date": jakarta_today_date_str(),
        "actual_rate": int(actual),
        "buffered_rate": int(buffered),
        "final": bool(final),
        "candidates_count": int(candidates_count),
        "selected_source_ts": selected_source_ts,
        "generated_ts_utc": now_iso_ts(),
    }
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
# SELECTION RULES (first/second execution)
# ----------------------------
def choose_final_candidate(cands: List[dict], yesterday_rate: Optional[int]) -> dict:
    """
    cands: list of dicts each with keys: date, rate, source_ts
    Yesterday logic:
      - if first.rate == yesterday and second.rate != yesterday -> pick second
      - if second.rate == yesterday and first.rate != yesterday -> pick first
      - else -> pick candidate with larger rate (tie -> earlier)
    """
    if not cands:
        raise RuntimeError("No candidates to choose from.")
    if len(cands) == 1:
        return cands[0]
    c_sorted = sorted(cands, key=lambda x: x.get("source_ts", ""))
    a, b = c_sorted[0], c_sorted[1]
    yr = yesterday_rate
    if yr is not None:
        if a["rate"] == yr and b["rate"] != yr:
            return b
        if b["rate"] == yr and a["rate"] != yr:
            return a
    if a["rate"] >= b["rate"]:
        return a
    else:
        return b

# ----------------------------
# MAIN FLOW
# ----------------------------
def main() -> int:
    ensure_data_dir()

    # load history rates for model building
    hist_rates = load_history_rates()

    # fetch current observation
    try:
        actual_rate, src_ts = fetch_current_rate()
    except Exception as e:
        print("ERROR: cannot fetch current rate:", e, file=sys.stderr)
        return 2

    today_s = jakarta_today_date_str()

    # Append candidate for today (preserve only today's candidates)
    candidates = load_candidates()
    new_cand = {"date": today_s, "rate": int(actual_rate), "source_ts": src_ts}
    candidates.append(new_cand)
    # keep only the last two candidates in chronological order
    candidates = sorted(candidates, key=lambda x: x.get("source_ts", ""))[-2:]
    save_candidates(candidates)

    # Get yesterday rate from history, if available
    yesterday_rate: Optional[int] = None
    history_entries = load_history_entries()
    if history_entries:
        try:
            yesterday_s = (datetime.fromisoformat(today_s) - timedelta(days=1)).date().isoformat()
        except Exception:
            yesterday_s = None
        if yesterday_s:
            last_entry_for_yesterday = next((e for e in history_entries[::-1] if e.get("date") == yesterday_s), None)
            if last_entry_for_yesterday:
                yesterday_rate = int(last_entry_for_yesterday.get("rate") or last_entry_for_yesterday.get("actual") or 0)

    # If only one candidate -> provisional
    if len(candidates) == 1:
        print("First execution today: candidate recorded (provisional).")
        try:
            provisional_buffer = compute_buffered_rate_from_history(hist_rates, actual_rate)
        except Exception as e:
            print("Warning: provisional buffer computation failed:", e, file=sys.stderr)
            provisional_buffer = round_up_to_hundred(int(actual_rate) + int(MARGIN))
        write_today_json(actual=actual_rate, buffered=provisional_buffer, final=False, candidates_count=1, selected_source_ts=src_ts)
        print(f"Provisional: actual={actual_rate}, provisional_buffer={provisional_buffer}")
        return 0

    # Two or more candidates -> finalize per rules
    if len(candidates) >= 2:
        chosen = choose_final_candidate(candidates, yesterday_rate)
        chosen_rate = int(chosen["rate"])
        chosen_src_ts = chosen.get("source_ts", "")

        try:
            buffered = compute_buffered_rate_from_history(hist_rates, chosen_rate)
        except Exception as e:
            print("Buffer computation error; using margin fallback:", e, file=sys.stderr)
            buffered = round_up_to_hundred(chosen_rate + int(MARGIN))

        # Save finalized history entry (one-per-day)
        save_history_entry(date_s=today_s, rate=chosen_rate, buffered=buffered, source_ts=chosen_src_ts, note=f"finalized from {len(candidates)} candidates")

        # Clear today's candidates
        save_candidates([])

        # Write today.json final
        write_today_json(actual=chosen_rate, buffered=buffered, final=True, candidates_count=len(candidates), selected_source_ts=chosen_src_ts)
        print(f"Finalized today's rate: actual={chosen_rate}, buffered={buffered} (candidates={len(candidates)})")
        return 0

    # fallback (shouldn't hit)
    write_today_json(actual=actual_rate, buffered=round_up_to_hundred(int(actual_rate) + int(MARGIN)), final=False, candidates_count=len(candidates), selected_source_ts=src_ts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
