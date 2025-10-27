import os
import time
import math
import logging
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import requests
from rapidfuzz import fuzz
from itertools import combinations

# ============ Config from environment ============
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")  # https://the-odds-api.com/
# The Odds API sport keys (e.g., soccer_epl, soccer_uefa_champs_league, basketball_nba, tennis_atp_singles)
SPORTS = os.getenv("SPORTS", "soccer_epl,basketball_nba,tennis_atp_singles").split(",")
# Markets: h2h (moneyline/1x2), spreads, totals. For surebets we start with h2h.
MARKETS = os.getenv("MARKETS", "h2h").split(",")
REGIONS = os.getenv("REGIONS", "eu,us,uk").split(",")  # provider regions
# Minimum edge to alert, e.g. 0.006 = 0.6%
ARB_MIN_EDGE = float(os.getenv("ARB_MIN_EDGE", "0.006"))
# Poll interval in seconds
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
# Deduplication window in minutes
DEDUP_MINUTES = int(os.getenv("DEDUP_MINUTES", "20"))
# Similarity threshold for team names (0..100)
TEAM_MATCH_THRESHOLD = int(os.getenv("TEAM_MATCH_THRESHOLD", "85"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ============ Telegram utils ============
def tg_send(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logging.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set. Message not sent.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=15)
        if resp.status_code != 200:
            logging.warning("Telegram sendMessage status=%s body=%s", resp.status_code, resp.text[:300])
    except Exception as e:
        logging.exception("Telegram send error: %s", e)

# ============ Odds provider (The Odds API) ============
class TheOddsAPIProvider:
    BASE = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_odds(self, sport_key: str, markets: List[str], regions: List[str]) -> Optional[List[dict]]:
        params = {
            "apiKey": self.api_key,
            "markets": ",".join(markets),
            "regions": ",".join(regions),
            "oddsFormat": "decimal",
            "dateFormat": "iso"
        }
        url = f"{self.BASE}/sports/{sport_key}/odds"
        try:
            r = requests.get(url, params=params, timeout=25)
            if r.status_code != 200:
                logging.warning("TheOddsAPI %s %s -> %s %s", sport_key, markets, r.status_code, r.text[:300])
                return None
            return r.json()
        except Exception as e:
            logging.exception("fetch_odds error: %s", e)
            return None

# ============ Surebet model ============
@dataclass
class Surebet:
    sport: str
    event_id: str
    commence_time: str
    home: str
    away: str
    market_key: str  # h2h
    selection_books: Dict[str, Tuple[str, float]]  # outcome_name -> (bookmaker_title, odds)
    inv_sum: float
    edge: float  # 1 - inv_sum
    hash_key: str

    def stake_split(self, bank: float) -> Dict[str, float]:
        invs = {out: 1.0/odds for out, (_, odds) in self.selection_books.items()}
        denom = sum(invs.values())
        return {out: bank * inv/denom for out, inv in invs.items()}

# ============ Matching and arbitrage logic ============
def norm_team(name: str) -> str:
    return name.replace(".", " ").replace("-", " ").lower().strip()

def fuzzy_same(a: str, b: str, threshold: int) -> bool:
    score = fuzz.token_set_ratio(a, b)
    return score >= threshold

def find_arbs_in_event(event: dict, sport_key: str) -> List[Surebet]:
    surebets: List[Surebet] = []
    event_id = event.get("id", "")
    start = event.get("commence_time", "")
    home = event.get("home_team", "")
    away = event.get("away_team", "")

    market_key = "h2h"
    outcomes: Dict[str, List[Tuple[str, float]]] = {}

    for bm in event.get("bookmakers", []):
        btitle = bm.get("title") or bm.get("key")
        for m in bm.get("markets", []):
            if m.get("key") != market_key:
                continue
            for o in m.get("outcomes", []):
                name = o.get("name")
                price = o.get("price")
                if not (name and isinstance(price, (int, float)) and price > 1.01):
                    continue
                outcomes.setdefault(name, []).append((btitle, float(price)))

    if not outcomes:
        return surebets

    # Canonicalize outcome names
    canon: Dict[str, List[Tuple[str, float]]] = {}
    for out_name, lst in outcomes.items():
        if fuzzy_same(out_name, home, TEAM_MATCH_THRESHOLD):
            canon.setdefault("HOME", []).extend(lst)
        elif fuzzy_same(out_name, away, TEAM_MATCH_THRESHOLD):
            canon.setdefault("AWAY", []).extend(lst)
        elif out_name.lower() in ("draw", "x"):
            canon.setdefault("DRAW", []).extend(lst)
        else:
            canon.setdefault(out_name, []).extend(lst)

    def best_by_outcome(lst: List[Tuple[str, float]]) -> Tuple[str, float]:
        return max(lst, key=lambda t: t[1])

    # Two-way
    if len(canon.keys()) == 2 and "DRAW" not in canon:
        o1, o2 = list(canon.keys())
        b1_title, b1_odds = best_by_outcome(canon[o1])
        alt2 = sorted(canon[o2], key=lambda t: t[1], reverse=True)
        b2_title, b2_odds = None, None
        for t in alt2:
            if t[0] != b1_title:
                b2_title, b2_odds = t
                break
        if b2_title is None:
            return surebets

        inv_sum = 1.0/b1_odds + 1.0/b2_odds
        if inv_sum < 1.0 - 1e-12:
            edge = 1.0 - inv_sum
            if edge >= ARB_MIN_EDGE:
                sel = {o1: (b1_title, b1_odds), o2: (b2_title, b2_odds)}
                key = f"{sport_key}|{event_id}|{market_key}|{b1_title}|{b2_title}"
                surebets.append(Surebet(
                    sport=sport_key,
                    event_id=event_id,
                    commence_time=start,
                    home=home, away=away,
                    market_key=market_key,
                    selection_books=sel,
                    inv_sum=inv_sum,
                    edge=edge,
                    hash_key=hashlib.md5(key.encode()).hexdigest(),
                ))
        return surebets

    # Three-way (1X2)
    if all(k in canon for k in ("HOME", "DRAW", "AWAY")):
        tops = {
            "HOME": sorted(canon["HOME"], key=lambda t: t[1], reverse=True),
            "DRAW": sorted(canon["DRAW"], key=lambda t: t[1], reverse=True),
            "AWAY": sorted(canon["AWAY"], key=lambda t: t[1], reverse=True),
        }
        limit = 3
        best_triplet = None
        best_edge = -1.0
        for h in tops["HOME"][:limit]:
            for d in tops["DRAW"][:limit]:
                for a in tops["AWAY"][:limit]:
                    books = {h[0], d[0], a[0]}
                    if len(books) < 2:
                        continue
                    inv_sum = 1.0/h[1] + 1.0/d[1] + 1.0/a[1]
                    if inv_sum < 1.0 - 1e-12:
                        edge = 1.0 - inv_sum
                        if edge > best_edge:
                            best_edge = edge
                            best_triplet = (h, d, a, inv_sum)
        if best_triplet and best_edge >= ARB_MIN_EDGE:
            (h, d, a, inv_sum) = best_triplet
            sel = {"HOME": (h[0], h[1]), "DRAW": (d[0], d[1]), "AWAY": (a[0], a[1])}
            key = f"{sport_key}|{event_id}|{market_key}|{sel['HOME'][0]}|{sel['DRAW'][0]}|{sel['AWAY'][0]}"
            surebets.append(Surebet(
                sport=sport_key,
                event_id=event_id,
                commence_time=start,
                home=home, away=away,
                market_key=market_key,
                selection_books=sel,
                inv_sum=inv_sum,
                edge=best_edge,
                hash_key=hashlib.md5(key.encode()).hexdigest(),
            ))
    return surebets

# ============ Deduper ============
class Deduper:
    def __init__(self, ttl_minutes: int):
        self.ttl = ttl_minutes * 60
        self.store: Dict[str, float] = {}

    def seen(self, key: str) -> bool:
        now = time.time()
        for k, t in list(self.store.items()):
            if now - t > self.ttl:
                self.store.pop(k, None)
        if key in self.store:
            return True
        self.store[key] = now
        return False

# ============ Message formatting ============
def format_msg(s):
    bank = float(os.getenv("BANK_SAMPLE", "1000"))
    splits = s.stake_split(bank)
    lines = []
    lines.append(f"<b>ARBITRAGE FOUND ({s.sport})</b>")
    lines.append(f"⏱ <b>Start:</b> {s.commence_time}")
    lines.append(f"🏟 <b>Match:</b> {s.home} vs {s.away}")
    lines.append(f"🧮 <b>Market:</b> {s.market_key}")
    lines.append("")
    for outcome, (book, odds) in s.selection_books.items():
        lines.append(f"• {outcome}: {odds:.3f} @ <i>{book}</i>")
    lines.append("")
    edge_pct = (1.0 - s.inv_sum) * 100
    lines.append(f"✅ <b>Edge:</b> {edge_pct:.2f}% (Σ 1/odds = {s.inv_sum:.4f})")
    lines.append(f"<b>Stake split (bank {bank:.0f}):</b>")
    for outcome, stake in splits.items():
        lines.append(f"• {outcome}: {stake:.2f}")
    lines.append(f"💰 <b>Locked profit:</b> {bank * (1.0 - s.inv_sum):.2f}")
    return "\n".join(lines)

def main():
    if not ODDS_API_KEY:
        logging.error("ODDS_API_KEY not set. Get a key at https://the-odds-api.com/ and set it in .env")
        return
    provider = TheOddsAPIProvider(ODDS_API_KEY)
    dedup = Deduper(DEDUP_MINUTES)

    logging.info("Start. SPORTS=%s MARKETS=%s REGIONS=%s EDGE>=%.3f", SPORTS, MARKETS, REGIONS, ARB_MIN_EDGE)

    while True:
        try:
            total_found = 0
            for sport in SPORTS:
                data = provider.fetch_odds(sport, MARKETS, REGIONS)
                if not data:
                    continue
                for ev in data:
                    arbs = find_arbs_in_event(ev, sport)
                    for s in arbs:
                        if dedup.seen(s.hash_key):
                            continue
                        msg = format_msg(s)
                        tg_send(msg)
                        total_found += 1
            logging.info("Cycle done. Found arbs: %d", total_found)
        except Exception as e:
            logging.exception("Main loop error: %s", e)

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
