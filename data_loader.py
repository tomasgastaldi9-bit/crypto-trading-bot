from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import BinanceConfig


@dataclass
class BinanceDataLoader:
    """Load historical Binance kline data into a normalized DataFrame."""

    config: BinanceConfig

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_klines(self) -> pd.DataFrame:
        """Fetch klines from Binance or load them from cache."""

        cache_path = self._cache_path()
        if self.config.use_cache and cache_path.exists():
            return self._load_from_cache(cache_path)

        data = self._download_klines()
        if self.config.use_cache:
            data.to_csv(cache_path, index=False)
        return data

    def _load_from_cache(self, cache_path: Path) -> pd.DataFrame:
        data = pd.read_csv(cache_path, parse_dates=["timestamp"])
        return self._normalize_dataframe(data)

    def _download_klines(self) -> pd.DataFrame:
        start_ms = self._to_milliseconds(self.config.start_date)
        end_ms = self._to_milliseconds(self.config.end_date) if self.config.end_date else None
        rows: list[list[Any]] = []
        cursor = start_ms

        while True:
            params = {
                "symbol": self.config.symbol,
                "interval": self.config.interval,
                "limit": self.config.request_limit,
                "startTime": cursor,
            }
            if end_ms is not None:
                params["endTime"] = end_ms

            response = self.session.get(
                f"{self.config.base_url}{self.config.klines_endpoint}",
                params=params,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            payload: list[list[Any]] = response.json()

            if not payload:
                break

            rows.extend(payload)
            last_open_time = int(payload[-1][0])
            cursor = last_open_time + 1

            if len(payload) < self.config.request_limit:
                break
            if end_ms is not None and last_open_time >= end_ms:
                break

        if not rows:
            raise ValueError("No kline data was returned by Binance for the requested range.")

        frame = pd.DataFrame(
            rows,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )
        frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        normalized = frame[["timestamp", "open", "high", "low", "close", "volume"]]
        return self._normalize_dataframe(normalized)

    def _normalize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        normalized = data.copy()
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
        numeric_columns = ["open", "high", "low", "close", "volume"]
        normalized[numeric_columns] = normalized[numeric_columns].astype(float)
        normalized = normalized.sort_values("timestamp").drop_duplicates(subset="timestamp")
        normalized = normalized.reset_index(drop=True)
        return normalized

    def _cache_path(self) -> Path:
        end_date = self.config.end_date or "latest"
        safe_name = f"{self.config.symbol}_{self.config.interval}_{self.config.start_date}_{end_date}.csv"
        return self.config.cache_dir / safe_name.replace(":", "-")

    @staticmethod
    def _to_milliseconds(value: str) -> int:
        timestamp = pd.Timestamp(value, tz="UTC")
        return int(timestamp.timestamp() * 1000)
