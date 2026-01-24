"""Taiwan stock lookup tool (TWSE daily data)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, Dict, Iterable, List, Tuple

import requests


@dataclass(slots=True)
class StockQuote:
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class TaiwanStockTool:
    """
    Tool: tw_stock

    Purpose:
      Fetch the most recent daily quote for a Taiwan stock ticker.

    Input:
      symbol: string (e.g., 2330)
      date: optional string (YYYYMMDD or YYYYMM); defaults to current month
    """

    base_url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY"

    def run(self, symbol: str, date: str | None = None) -> Dict[str, Any]:
        """Fetch a daily quote; defaults to the most recent trading day in the month."""
        return self.fetch_daily(symbol, date)

    def fetch_daily(self, symbol: str, date: str | None = None) -> Dict[str, Any]:
        symbol = self._normalize_symbol(symbol)
        if date:
            yyyymmdd = self._normalize_date(date)
            month = yyyymmdd[:6]
            for offset in range(13):
                records = self.fetch_month(symbol, month)
                if records:
                    if offset == 0:
                        for record in records:
                            if record["date"] == yyyymmdd:
                                return record
                        prior = [r for r in records if r["date"] <= yyyymmdd]
                        if prior:
                            return prior[-1]
                    else:
                        return records[-1]
                month = self._shift_month(month, -1)
            raise ValueError(f"No data for {symbol} on or before {yyyymmdd}.")

        month = datetime.now().strftime("%Y%m")
        records = self.fetch_month(symbol, month)
        if not records:
            raise ValueError(f"No data returned for {symbol} in {month}.")
        return records[-1]

    def fetch_month(self, symbol: str, month: str) -> List[Dict[str, Any]]:
        symbol = self._normalize_symbol(symbol)
        month = self._normalize_month(month)
        params = {"response": "json", "date": f"{month}01", "stockNo": symbol}
        response = requests.get(self.base_url, params=params, timeout=5)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or []
        records: List[Dict[str, Any]] = []
        for row in data:
            if len(row) < 7:
                continue
            date_ymd = self._parse_roc_date(str(row[0]))
            if not date_ymd:
                continue
            records.append(
                {
                    "symbol": symbol,
                    "date": date_ymd,
                    "open": self._parse_float(row[3]),
                    "high": self._parse_float(row[4]),
                    "low": self._parse_float(row[5]),
                    "close": self._parse_float(row[6]),
                    "volume": self._parse_int(row[1]),
                }
            )
        records.sort(key=lambda item: item["date"])
        return records

    def fetch_range(self, symbol: str, start: str, end: str) -> List[Dict[str, Any]]:
        symbol = self._normalize_symbol(symbol)
        start_ymd = self._normalize_date(start)
        end_ymd = self._normalize_date(end)
        months = self._month_span(start_ymd[:6], end_ymd[:6])
        records: List[Dict[str, Any]] = []
        for month in months:
            records.extend(self.fetch_month(symbol, month))
        return [r for r in records if start_ymd <= r["date"] <= end_ymd]

    def fetch_recent(self, symbol: str, count: int) -> List[Dict[str, Any]]:
        symbol = self._normalize_symbol(symbol)
        if count <= 0:
            raise ValueError("count must be positive")
        current_month = datetime.now().strftime("%Y%m")
        records: List[Dict[str, Any]] = []
        for month in self._month_span(self._shift_month(current_month, -24), current_month):
            records.extend(self.fetch_month(symbol, month))
            if len(records) >= count:
                break
        records.sort(key=lambda item: item["date"])
        if len(records) < count:
            raise ValueError(f"Not enough data to return {count} days for {symbol}.")
        return records[-count:]

    def _normalize_symbol(self, symbol: str) -> str:
        symbol = symbol.strip().upper()
        match = re.fullmatch(r"(?:TPE:|TW:|TWSE:)?(\d{4,6})(?:\.TW)?", symbol)
        if not match:
            raise ValueError(f"Invalid Taiwan ticker symbol: {symbol}")
        return match.group(1)

    def _normalize_month(self, date: str) -> str:
        match = re.search(r"(\d{4})\D*(\d{1,2})", date)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            return f"{year:04d}{month:02d}"
        digits = re.sub(r"\D", "", date)
        if len(digits) >= 6:
            return digits[:6]
        raise ValueError(f"Invalid date format: {date}")

    def _normalize_date(self, date: str) -> str:
        match = re.search(r"(\d{4})\D*(\d{1,2})\D*(\d{1,2})", date)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            return f"{year:04d}{month:02d}{day:02d}"
        digits = re.sub(r"\D", "", date)
        if len(digits) >= 8:
            return digits[:8]
        if len(digits) == 6:
            return digits + "01"
        raise ValueError(f"Invalid date format: {date}")

    def _parse_roc_date(self, value: str) -> str | None:
        match = re.match(r"^\s*(\d{2,3})/(\d{1,2})/(\d{1,2})\s*$", value)
        if not match:
            return None
        roc_year = int(match.group(1))
        year = roc_year + 1911
        month = int(match.group(2))
        day = int(match.group(3))
        return f"{year:04d}{month:02d}{day:02d}"

    def _month_span(self, start: str, end: str) -> List[str]:
        months: List[str] = []
        current = start
        while current <= end:
            months.append(current)
            current = self._shift_month(current, 1)
        return months

    def _shift_month(self, yyyymm: str, delta: int) -> str:
        year = int(yyyymm[:4])
        month = int(yyyymm[4:6])
        month += delta
        while month > 12:
            month -= 12
            year += 1
        while month < 1:
            month += 12
            year -= 1
        return f"{year:04d}{month:02d}"

    def _parse_float(self, value: Any) -> float:
        text = str(value).replace(",", "").strip()
        if text in {"--", ""}:
            return 0.0
        return float(text)

    def _parse_int(self, value: Any) -> int:
        text = str(value).replace(",", "").strip()
        if text in {"--", ""}:
            return 0
        return int(float(text))
