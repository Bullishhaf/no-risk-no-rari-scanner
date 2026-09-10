"""
No Risk No Rari Scanner — Discord bot version.

Runs on GitHub Actions (real internet access, unlike a sandboxed agent
session). Pulls real S&P 500 quotes via yfinance, ranks movers, renders
the dashboard to a PNG with a headless browser, and posts that image
straight into a Discord channel via a webhook.

State (which tickers were already seen today, and at what price) is
carried between runs using ./state.json, restored/saved by the GitHub
Actions cache step in the workflow — see .github/workflows/scan.yml.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

STATE_PATH = "state.json"
OUT_HTML = "scanner.html"
OUT_PNG = "scanner.png"
PRICE_MIN = 5.0
VOL_MIN = 300_000
TOP_N = 10
BATCH_SIZE = 100


def et_now():
    # Uses the system's tz database so DST is handled correctly.
    import zoneinfo
    return datetime.now(zoneinfo.ZoneInfo("America/New_York"))


def get_sp500_tickers():
    """Scrape the current S&P 500 constituent list from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False).tolist()
    names = dict(zip(tickers, df["Security"].astype(str).tolist()))
    return tickers, names


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def fetch_history(tickers, period="40d"):
    """Batched daily OHLCV for the whole universe. Returns a dict
    ticker -> DataFrame (daily bars, most recent last)."""
    out = {}
    for batch in chunked(tickers, BATCH_SIZE):
        for attempt in range(3):
            try:
                data = yf.download(
                    tickers=batch, period=period, interval="1d",
                    group_by="ticker", threads=True, progress=False,
                    auto_adjust=False,
                )
                break
            except Exception:
                time.sleep(3)
        else:
            continue
        for t in batch:
            try:
                df = data[t] if len(batch) > 1 else data
                df = df.dropna(how="all")
                if not df.empty:
                    out[t] = df
            except Exception:
                pass
        time.sleep(1)
    return out


def fetch_intraday(tickers):
    """Batched today's intraday bars (for live price + volume so far)."""
    out = {}
    for batch in chunked(tickers, BATCH_SIZE):
        for attempt in range(3):
            try:
                data = yf.download(
                    tickers=batch, period="1d", interval="15m",
                    group_by="ticker", threads=True, progress=False,
                    auto_adjust=False,
                )
                break
            except Exception:
                time.sleep(3)
        else:
            continue
        for t in batch:
            try:
                df = data[t] if len(batch) > 1 else data
                df = df.dropna(how="all")
                if not df.empty:
                    out[t] = df
            except Exception:
                pass
        time.sleep(1)
    return out


def build_universe(tickers, names, hist, intraday, today_date):
    rows = []
    for t in tickers:
        h = hist.get(t)
        i = intraday.get(t)
        if h is None or i is None or i.empty:
            continue
        try:
            # Keep only fully-closed daily bars strictly before today —
            # yfinance sometimes includes a partial "today" row in the
            # daily history depending on when it's called, and we don't
            # want that masquerading as a previous close.
            h_closed = h[h.index.date < today_date]
            if len(h_closed) < 6:
                continue
            price = float(i["Close"].dropna().iloc[-1])
            day_vol = float(i["Volume"].fillna(0).sum())
            prev_close = float(h_closed["Close"].iloc[-1])
            close_5d = float(h_closed["Close"].iloc[-6])
            close_20d = float(h_closed["Close"].iloc[-21]) if len(h_closed) >= 21 else None
            avg_vol_20d = float(h_closed["Volume"].tail(20).mean())
            if price <= 0 or prev_close <= 0:
                continue
            chg = (price - prev_close) / prev_close * 100
            wrs = ((price - close_5d) / close_5d * 100) if close_5d else None
            m20 = ((price - close_20d) / close_20d * 100) if close_20d else None
            rows.append({
                "ticker": t, "name": names.get(t, t),
                "price": price, "chg": chg, "vol": day_vol,
                "avg_vol_20d": avg_vol_20d, "wrs": wrs, "m20": m20,
            })
        except Exception:
            continue
    return rows


def percentile_rank(rows, key):
    vals = sorted((r[key] for r in rows if r.get(key) is not None))
    n = len(vals)
    ranks = {}
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        idx = vals.index(v)  # fine at this scale (~500 rows)
        ranks[r["ticker"]] = 0 if n <= 1 else round(idx / (n - 1) * 99)
    return ranks


def liq_dots(avg_dollar_vol):
    if avg_dollar_vol >= 200_000_000:
        return 3
    if avg_dollar_vol >= 20_000_000:
        return 2
    return 1


def dots(n):
    return "●" * n + "○" * (3 - n)


def load_state(today):
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH) as f:
            data = json.load(f)
        if data.get("date") != today:
            return {}
        return data.get("tickers", {})
    except Exception:
        return {}


def save_state(today, state):
    with open(STATE_PATH, "w") as f:
        json.dump({"date": today, "tickers": state}, f)


def compute(rows, state):
    filt = [r for r in rows if r["price"] >= PRICE_MIN and r["vol"] >= VOL_MIN]
    irs_rank = percentile_rank(filt, "chg")
    wrs_rank = percentile_rank([r for r in filt if r.get("wrs") is not None], "wrs")
    m20_rank = percentile_rank([r for r in filt if r.get("m20") is not None], "m20")

    filt.sort(key=lambda r: -r["chg"])
    calls = filt[:TOP_N]
    filt.sort(key=lambda r: r["chg"])
    puts = filt[:TOP_N]

    now_iso = datetime.now(timezone.utc).isoformat()

    def enrich(row_list):
        out = []
        for r in row_list:
            t = r["ticker"]
            if t in state:
                fsp, fst = state[t]["firstSeenPrice"], state[t]["firstSeenTime"]
            else:
                fsp, fst = r["price"], now_iso
                state[t] = {"firstSeenPrice": fsp, "firstSeenTime": fst}
            since = 0.0 if fsp == 0 else (r["price"] - fsp) / fsp * 100
            out.append({
                **r,
                "irs": irs_rank.get(t, 0),
                "wrsRank": wrs_rank.get(t, 0),
                "m20Rank": m20_rank.get(t, 0),
                "liq": liq_dots(r["price"] * r["avg_vol_20d"]),
                "firstSeenPrice": fsp, "firstSeenTime": fst,
                "sincePct": round(since, 2),
            })
        return out

    return {
        "calls": enrich(calls), "puts": enrich(puts),
        "universe_size": len(filt), "state": state,
    }


def fmt_vol(v):
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v/1_000:.0f}K"
    return str(int(v))


def row_html(r, kind):
    sign = "+" if r["chg"] >= 0 else ""
    ssign = "+" if r["sincePct"] >= 0 else ""
    cls = "pos" if r["chg"] >= 0 else "neg"
    scls = "pos" if r["sincePct"] > 0 else ("neg" if r["sincePct"] < 0 else "flat")
    fst = datetime.fromisoformat(r["firstSeenTime"]).astimezone(
        __import__("zoneinfo").ZoneInfo("America/New_York")
    ).strftime("%-I:%M %p ET")
    return f"""<tr>
      <td class="tkr {kind}">{r['ticker']}</td>
      <td class="company">{r['name']}</td>
      <td class="num">{r['price']:.2f}</td>
      <td class="num {cls}">{sign}{r['chg']:.2f}%</td>
      <td class="num dim">{fmt_vol(r['vol'])}</td>
      <td class="num">{r['irs']}</td>
      <td class="num dim">{r['wrsRank']}</td>
      <td class="num dim">{r['m20Rank']}</td>
      <td class="dots">{dots(r['liq'])}</td>
      <td class="num {scls}">{ssign}{r['sincePct']:.2f}%</td>
      <td class="num dim time">{fst}</td>
    </tr>"""


TEMPLATE = open(os.path.join(os.path.dirname(__file__), "template.html")).read()


def render_html(computed, as_of, market_open):
    pill_class = "pill-open" if market_open else "pill-closed"
    pill_text = "OPEN" if market_open else "CLOSED"
    html = TEMPLATE.format(
        pill_class=pill_class, pill_text=pill_text, as_of=as_of,
        n_calls=len(computed["calls"]), n_puts=len(computed["puts"]),
        universe_size=computed["universe_size"],
        calls_rows="\n".join(row_html(r, "call") for r in computed["calls"]),
        puts_rows="\n".join(row_html(r, "put") for r in computed["puts"]),
    )
    with open(OUT_HTML, "w") as f:
        f.write(html)
    return html


def screenshot():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1180, "height": 900})
        page.goto("file://" + os.path.abspath(OUT_HTML))
        page.wait_for_timeout(400)
        page.screenshot(path=OUT_PNG, full_page=True)
        b.close()


def post_to_discord(webhook_url, computed, as_of):
    top_call = computed["calls"][0] if computed["calls"] else None
    top_put = computed["puts"][0] if computed["puts"] else None
    content_bits = [f"**Scan — {as_of}**"]
    if top_call:
        content_bits.append(f"Top call: **{top_call['ticker']}** {top_call['chg']:+.2f}%")
    if top_put:
        content_bits.append(f"Top put: **{top_put['ticker']}** {top_put['chg']:+.2f}%")
    content = "  |  ".join(content_bits)
    with open(OUT_PNG, "rb") as f:
        resp = requests.post(
            webhook_url,
            data={"content": content},
            files={"file": ("scanner.png", f, "image/png")},
            timeout=30,
        )
    resp.raise_for_status()


def main():
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("No DISCORD_WEBHOOK_URL set — aborting.")
        sys.exit(1)

    now = et_now()
    force = os.environ.get("FORCE_RUN", "").lower() in ("1", "true", "yes")
    if not force and now.weekday() >= 5:
        print("Weekend — skipping.")
        return
    if not force and (now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 16):
        print(f"Outside market hours ({now.strftime('%H:%M')} ET) — skipping.")
        return
    if force:
        print(f"FORCE_RUN set — bypassing weekday/market-hours check ({now.strftime('%a %H:%M')} ET).")

    today = now.strftime("%Y-%m-%d")
    state = load_state(today)

    print("Fetching S&P 500 ticker list...")
    tickers, names = get_sp500_tickers()
    print(f"{len(tickers)} tickers")

    print("Fetching daily history...")
    hist = fetch_history(tickers)
    print(f"history for {len(hist)} tickers")

    print("Fetching intraday quotes...")
    intraday = fetch_intraday(tickers)
    print(f"intraday for {len(intraday)} tickers")

    rows = build_universe(tickers, names, hist, intraday, now.date())
    print(f"{len(rows)} tickers with complete data")
    if not rows:
        print("No usable data this run — skipping post.")
        return

    computed = compute(rows, state)
    save_state(today, computed["state"])

    as_of = now.strftime("%b %-d, %-I:%M %p ET (live)")
    render_html(computed, as_of, market_open=True)
    screenshot()
    post_to_discord(webhook_url, computed, as_of)
    print("Posted to Discord.")


if __name__ == "__main__":
    main()
