# src/stock_data_service.py
import asyncio
import logging
import os
import random
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ib_insync import IB, Stock, util

try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

logging.basicConfig(level=logging.INFO)


# ----------------------------- Pacing Guard -----------------------------------
class RateLimiter:
    """
    Simple token-bucket-ish limiter: allow at most `max_calls` within `period` seconds.
    Uses monotonic loop time and sleeps when the bucket is full.
    """
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = float(period)
        self._times = deque()

    async def acquire(self):
        loop = asyncio.get_event_loop()
        now = loop.time()
        # purge expired timestamps
        while self._times and (now - self._times[0]) > self.period:
            self._times.popleft()

        if len(self._times) < self.max_calls:
            self._times.append(now)
            return

        # need to wait until the earliest timestamp falls out of the window
        wait = self.period - (now - self._times[0])
        if wait > 0:
            await asyncio.sleep(wait)
        # after sleeping, record this call
        now = loop.time()
        while self._times and (now - self._times[0]) > self.period:
            self._times.popleft()
        self._times.append(now)


class StockDataService:
    """
    Fetches historical price data via Interactive Brokers (ib_insync) and
    persists/retrieves it through the provided `db_client`.

    Public API (unchanged):
      - get_tickers(directory, filename) -> List[str]
      - get_prices_for_tickers(tickers, start_date, end_date) -> List[str]
      - fetch_prices_from_db(tickers, start_date, end_date) -> Dict[str, dict]
      - get_rates_spreads_data() -> Dict[str, Dict[str, list]]
      - close()
    """

    # --- IB request guardrails -------------------------------------------------
    # Concurrency: number of simultaneous workers pulling from the queue.
    _CONCURRENCY = int(os.getenv("IB_HIST_CONCURRENCY", "4"))

    # Retries per ticker on transient errors/timeouts.
    _RETRIES = int(os.getenv("IB_HIST_RETRIES", "3"))

    # Per-request timeout (seconds) for reqHistoricalDataAsync
    _TIMEOUT_SEC = int(os.getenv("IB_HIST_TIMEOUT_SEC", "60"))

    # Max daily span IB will allow for 1D bars (~30y)
    _MAX_DAYS_DAILY = 365 * 30

    # Historical pacing (defaults: ~60 requests per 10 minutes).
    _MAX_REQS = int(os.getenv("IB_HIST_MAX_REQS", "60"))
    _WINDOW_SEC = float(os.getenv("IB_HIST_WINDOW_SEC", "600"))

    # Optional tiny jitter between requests to avoid sync bursts.
    _JITTER_SEC = float(os.getenv("IB_HIST_JITTER_SEC", "0.15"))

    def __init__(
        self,
        db_client,
        ib_host: str = "127.0.0.1",
        ib_port: int = 7496,
        client_id: Optional[int] = None,
        market_data_type: int = 3,  # 1=real-time, 3=delayed
        max_connect_retries: int = 5,
    ) -> None:
        self.db_client = db_client
        self.ib = IB()

        # choose a random clientId if none provided
        cid = client_id or random.randint(2, 50_000)
        last_err = None
        for _ in range(max_connect_retries):
            try:
                self.ib.connect(ib_host, ib_port, clientId=cid, timeout=5)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                cid = random.randint(2, 50_000)
        else:
            raise ConnectionError(f"Unable to connect to IB: {last_err}")

        try:
            self.ib.reqMarketDataType(market_data_type)
        except Exception:
            pass

        logging.info(
            "Connected to IB Gateway at %s:%s with clientId %s",
            ib_host, ib_port, cid,
        )

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------
    def get_tickers(self, directory: str, filename: str) -> List[str]:
        """
        Return tickers from `directory/filename` CSV.
        Accepts a 'ticker' (any case) column or single-column CSVs.
        For 'fx.csv', returns Yahoo-style '=X' (these are skipped for IB).
        """
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            logging.warning("Ticker file not found: %s", path)
            return []

        try:
            df = pd.read_csv(path)
            col = [c for c in df.columns if c.lower() == "ticker"]
            if col:
                tickers = df[col[0]].dropna().astype(str).tolist()
            else:
                tickers = df.iloc[:, 0].dropna().astype(str).tolist()
        except Exception:
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.read().strip().splitlines()[1:]
            tickers = [ln.split(",")[0].strip() for ln in lines]

        if filename.lower() == "fx.csv":
            tickers = [f"{t}=X" for t in tickers]
        return tickers

    def get_prices_for_tickers(self, tickers: List[str], start_date: str, end_date: str) -> List[str]:
        """
        Download & store data for any tickers that aren't up to date in the DB.
        This is now a STREAMING pipeline: it processes tickers via a queue and
        writes each symbol to the DB as soon as it's fetched. This avoids
        memory blow-ups and handles arbitrarily long ticker lists.
        """
        # Filter to symbols we can fetch from IB
        candidates = [t for t in tickers if self._is_fetchable_by_ib(t)]
        # Respect existing DB presence (latest business day heuristic)
        need = [t for t in candidates if not self.db_client.data_exists(t)]
        if not need:
            logging.info("All tickers already present.")
            return []

        days = self._clamp_days(start_date, end_date)
        downloaded = self.ib.run(self._process_stream(need, days))
        return downloaded

    def fetch_prices_from_db(self, tickers, start_date, end_date):
        """
        Pure DB read. Do NOT download here; the data_fetcher process handles that.
        Returns dict-of-lists for the tickers that exist; missing tickers return empty dicts.
        """
        out = {}
        for t in tickers:
            data = self.db_client.fetch_prices(t, start_date, end_date)
            if data and data.get("date"):
                out[t] = data
        return out


    # --------------------------------------------------------------------------
    # Internals
    # --------------------------------------------------------------------------
    @staticmethod
    def _is_fetchable_by_ib(symbol: str) -> bool:
        # Skip Yahoo-style FX symbols like 'EURUSD=X'; IB uses Forex(...) contracts for FX.
        return not symbol.endswith("=X")

    def _clamp_days(self, start_date: str, end_date: str) -> int:
        d = max((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days, 1)
        if d > self._MAX_DAYS_DAILY:
            logging.warning("Requested %d days; clamping to %d (~30y).", d, self._MAX_DAYS_DAILY)
        return min(d, self._MAX_DAYS_DAILY)

    def _duration_str(self, days: int) -> str:
        """IB rule: durations > 365 days must be expressed in years ('Y')."""
        if days <= 365:
            return f"{days} D"
        years = max(1, math.ceil(days / 365))
        return f"{years} Y"

    # ---------- Streaming pipeline (queue + workers + rate limiting) ----------
    async def _process_stream(self, tickers: List[str], days: int) -> List[str]:
        """
        Process an arbitrary number of tickers with bounded concurrency and
        pacing control. Each successful fetch is written to the DB immediately.
        Returns the list of tickers that were successfully stored.
        """
        q: asyncio.Queue[str] = asyncio.Queue()
        for t in tickers:
            q.put_nowait(t)

        limiter = RateLimiter(self._MAX_REQS, self._WINDOW_SEC)
        successes: List[str] = []
        failures: List[Tuple[str, str]] = []  # (symbol, reason)

        async def worker(worker_id: int):
            while True:
                try:
                    symbol = await q.get()
                except asyncio.CancelledError:
                    break
                try:
                    df = await self._fetch_one(symbol, days, limiter)
                    if df is not None and not df.empty:
                        try:
                            self.db_client.insert_stock_data(df, symbol)
                            logging.info("Stored %s rows for %s", len(df), symbol)
                            successes.append(symbol)
                        except Exception as e:  # noqa: BLE001
                            failures.append((symbol, f"DB insert failed: {e}"))
                    else:
                        # empty/no data is not fatal; just record
                        failures.append((symbol, "no data"))
                except Exception as e:  # noqa: BLE001
                    failures.append((symbol, str(e)))
                finally:
                    q.task_done()

        workers = [asyncio.create_task(worker(i)) for i in range(max(1, self._CONCURRENCY))]
        await q.join()
        for w in workers:
            w.cancel()
        # Optionally, log failures summary
        if failures:
            logging.warning("Completed with %d failures (showing first 5): %s",
                            len(failures), failures[:5])
        return successes

    async def _fetch_one(self, symbol: str, days: int, limiter: RateLimiter) -> Optional[pd.DataFrame]:
        """
        Single-symbol fetch with retries, using the shared rate limiter.
        Also qualifies the contract before requesting history.
        """
        contract = Stock(symbol, "SMART", "USD")
        try:
            qualified = await self.ib.qualifyContractsAsync(contract)
            contract = qualified[0] if qualified else contract
        except Exception:
            # not fatal; IB can still often resolve
            pass

        for attempt in range(1, self._RETRIES + 1):
            try:
                if self._JITTER_SEC:
                    await asyncio.sleep(self._JITTER_SEC)  # tiny stagger to avoid bursts
                await limiter.acquire()  # enforce HMDS pacing
                bars = await asyncio.wait_for(
                    self.ib.reqHistoricalDataAsync(
                        contract,
                        endDateTime="",
                        durationStr=self._duration_str(days),
                        barSizeSetting="1 day",
                        whatToShow="TRADES",
                        useRTH=False,   # full session for daily bars
                        formatDate=1,
                        keepUpToDate=False,
                    ),
                    timeout=self._TIMEOUT_SEC,
                )
                if not bars:
                    return None
                return self._normalize_ib_df(util.df(bars))
            except Exception as e:
                err_txt = str(e).lower()
                # If it smells like pacing, pause longer before retrying
                if "pacing" in err_txt or "max rate" in err_txt or "query" in err_txt and "limit" in err_txt:
                    backoff = min(60, 10 * attempt)
                else:
                    backoff = 3 * attempt
                logging.warning("Attempt %d/%d failed for %s (%s). Retrying in %ss…",
                                attempt, self._RETRIES, symbol, e, backoff)
                await asyncio.sleep(backoff)

        logging.error("Giving up on %s after %d attempts.", symbol, self._RETRIES)
        return None

    @staticmethod
    def _normalize_ib_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize ib_insync bars to (date, open, high, low, close, volume) with
        date as YYYY-MM-DD strings.
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        if "date" not in df.columns:
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        cols = ["date", "open", "high", "low", "close", "volume"]
        df = df[cols].copy()
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    # --------------------------------------------------------------------------
    # FRED helpers
    # --------------------------------------------------------------------------
    def get_rates_spreads_data(self) -> Dict[str, Dict[str, list]]:
        """
        Download a set of FRED series and return dict-of-lists that the Dash app expects.
        """
        if pdr is None:
            logging.error("pandas_datareader is not installed; cannot fetch FRED data.")
            return {}

        names = [
            "2Y-10Y Spread", "5Y Breakeven", "HY-OAS", "IG Spread", "High Yield",
            "3M t-bill", "2Y t-note", "5Y t-note", "10Y t-note", "30Y t-note",
        ]
        tickers = [
            "T10Y2Y", "T5YIE", "BAMLH0A0HYM2", "BAMLC0A4CBBB", "BAMLH0A0HYM2EY",
            "DTB3", "DGS2", "DGS5", "DGS10", "DGS30",
        ]
        out: Dict[str, Dict[str, list]] = {}
        for fred, name in zip(tickers, names, strict=False):
            try:
                df = pdr.DataReader(fred, "fred").dropna()
                out[name] = {"date": df.index.tolist(), "close": df[fred].tolist()}
            except Exception as e:  # noqa: BLE001
                logging.warning("FRED fetch failed for %s: %s", fred, e)
        return out

    # --------------------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------------------
    def close(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
            logging.info("Disconnected from IB")
