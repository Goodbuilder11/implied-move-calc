from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, StockSnapshotRequest
from dotenv import load_dotenv
import os
import sys

DEFAULT_EXPIRATION_DATE = datetime.now().strftime("%Y-%m-%d")
DEFAULT_TOLERANCE_RATIO = 0.10  # ±10% fallback window for nearest strike


# --- Minimal ANSI styling helpers for nicer CLI output ---
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
FG_CYAN = "\033[36m"
FG_MAGENTA = "\033[35m"
FG_YELLOW = "\033[33m"
FG_GREEN = "\033[32m"
FG_WHITE = "\033[97m"


def _supports_color() -> bool:
    """Return True if the output supports ANSI colors."""
    try:
        return sys.stdout.isatty() and (os.getenv("TERM") not in (None, "dumb"))
    except Exception:
        return False


def _style(text: str, *effects: str) -> str:
    """Apply ANSI styles if supported; otherwise return plain text."""
    if not _supports_color():
        return str(text)
    return "".join(effects) + str(text) + RESET


@dataclass
class OptionMidQuote:
    symbol: str
    bid_price: float
    ask_price: float
    mid_price: float


class MarketDataService:
    """Light wrapper around Alpaca market data clients with helpful utilities."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        self.stock_client = StockHistoricalDataClient(api_key, api_secret)
        self.option_client = OptionHistoricalDataClient(api_key, api_secret)

    def get_current_stock_price(self, symbol: str) -> int:
        """Return the latest trade price for a stock as an int (truncated),
        matching the original behavior that cast the price to int.
        """
        request = StockSnapshotRequest(symbol_or_symbols=symbol)
        snapshot = self.stock_client.get_stock_snapshot(request)
        # Handle both single-object and mapping responses from SDKs
        try:
            stock_data = snapshot[symbol]  # type: ignore[index]
        except Exception:
            stock_data = snapshot
        price = getattr(getattr(stock_data, "latest_trade", None), "price", None)
        if price is None:
            raise ValueError(f"Unable to retrieve latest trade price for {symbol}")
        return int(price)

    def is_valid_stock_symbol(self, symbol: str) -> bool:
        """Return True if the symbol appears valid (i.e., snapshot returns a price)."""
        try:
            _ = self.get_current_stock_price(symbol)
            return True
        except Exception:
            return False

    def _extract_strike_from_symbol(self, option_symbol: str) -> Optional[float]:
        """Extract strike price from an OCC option symbol, e.g. 'O:SPY241018C00370500'.

        OCC format encodes strike in the last 8 digits with 1/1000 precision.
        This method is resilient to an optional 'O:' prefix.
        """
        try:
            tail = option_symbol.split(":")[-1]
            strike_millis_str = tail[-8:]
            if not strike_millis_str.isdigit():
                return None
            return int(strike_millis_str) / 1000.0
        except Exception:
            return None

    def _extract_expiration_from_symbol(
        self, option_symbol: str
    ) -> Optional[datetime.date]:
        """Extract expiration date from an OCC option symbol.

        Example: 'O:TSLA250815C00370000' -> 2025-08-15
        The 6 chars before the 'C'/'P' flag are YYMMDD.
        """
        try:
            tail = option_symbol.split(":")[-1]
            cp_index = len(tail) - 8 - 1  # position of 'C'/'P'
            if cp_index < 6:
                return None
            cp_flag = tail[cp_index]
            if cp_flag not in ("C", "P"):
                return None
            expiry_str = tail[cp_index - 6 : cp_index]
            if not expiry_str.isdigit():
                return None
            yy = int(expiry_str[0:2])
            mm = int(expiry_str[2:4])
            dd = int(expiry_str[4:6])
            year = 2000 + yy
            return datetime(year, mm, dd).date()
        except Exception:
            return None

    def get_option_mid_quote(
        self,
        underlying_symbol: str,
        option_type: str,  # "call" or "put"
        strike_price: float,
        expiration_date: str,
        tolerance_ratio: float = 0.10,
    ) -> OptionMidQuote:
        """Fetch the nearest option around the given strike/expiry and compute mid price.

        If an exact strike match is unavailable, searches within ±tolerance_ratio
        and selects the contract with the strike closest to the requested strike.
        If the requested expiration has no options, search any expiration and
        pick the nearest expiration on or after the requested date.
        """
        # Treat values > 1 as percentages for convenience (e.g., 10 -> 0.10)
        if tolerance_ratio > 1.0:
            tolerance_ratio = tolerance_ratio / 100.0

        lower_bound = strike_price * (1.0 - max(tolerance_ratio, 0.0))
        upper_bound = strike_price * (1.0 + max(tolerance_ratio, 0.0))

        requested_expiry_date = datetime.strptime(expiration_date, "%Y-%m-%d").date()

        def request_chain(
            lb: Optional[float], ub: Optional[float], include_expiration: bool
        ):
            kwargs = dict(
                underlying_symbol=underlying_symbol,
                type=option_type,
            )
            if include_expiration:
                kwargs["expiration_date"] = expiration_date
            if lb is not None:
                kwargs["strike_price_gte"] = lb
            if ub is not None:
                kwargs["strike_price_lte"] = ub
            return self.option_client.get_option_chain(OptionChainRequest(**kwargs))

        # First attempt: within tolerance window
        chain = request_chain(lower_bound, upper_bound, include_expiration=True)

        # Fallback: if nothing found, try without bounds (same expiry)
        if not chain:
            chain = request_chain(None, None, include_expiration=True)

        # If still nothing, broaden to any expiration and filter later
        if not chain:
            chain = request_chain(lower_bound, upper_bound, include_expiration=False)
        if not chain:
            chain = request_chain(None, None, include_expiration=False)

        if not chain:
            raise ValueError(
                f"No {option_type} options found for {underlying_symbol} expiring {expiration_date}"
            )

        # Normalize chain to mapping keyed by symbol if SDK returns a list-like
        if not isinstance(chain, dict):
            try:
                chain = {getattr(item, "symbol"): item for item in chain}
            except Exception:
                # Last-resort: create indices as keys to allow iteration
                chain = {str(idx): item for idx, item in enumerate(list(chain))}

        # Determine the nearest expiration on or after the requested date
        target_expiry = None
        expiry_by_symbol: dict[str, datetime.date] = {}
        min_future_delta = float("inf")

        for symbol, option_data in chain.items():
            expiry = getattr(option_data, "expiration_date", None)
            if expiry is None:
                expiry = self._extract_expiration_from_symbol(symbol)
            else:
                try:
                    expiry = getattr(expiry, "date", lambda: expiry)()
                except Exception:
                    pass
            if expiry is None:
                continue
            expiry_by_symbol[symbol] = expiry
            delta_days = (expiry - requested_expiry_date).days
            if delta_days >= 0 and delta_days < min_future_delta:
                min_future_delta = delta_days
                target_expiry = expiry

        # If no future expiry found, fall back to absolutely nearest expiry
        if target_expiry is None and expiry_by_symbol:
            target_expiry = min(
                expiry_by_symbol.values(),
                key=lambda d: abs((d - requested_expiry_date).days),
            )

        # Pick the symbol whose strike is nearest to the requested strike, optionally constrained by target_expiry
        selected_symbol = None
        selected_option_data = None
        smallest_abs_diff = float("inf")

        for symbol, option_data in chain.items():
            if target_expiry is not None:
                expiry = expiry_by_symbol.get(symbol)
                if expiry is None or expiry != target_expiry:
                    continue
            strike_from_attr = getattr(option_data, "strike_price", None)
            if strike_from_attr is None:
                # Some SDKs nest details or require parsing from symbol
                strike = self._extract_strike_from_symbol(symbol)
            else:
                try:
                    strike = float(strike_from_attr)
                except Exception:
                    strike = None

            if strike is None:
                continue

            diff = abs(strike - strike_price)
            if diff < smallest_abs_diff:
                smallest_abs_diff = diff
                selected_symbol = symbol
                selected_option_data = option_data

        # As a final fallback, if expiry filtering removed all, pick nearest strike from all symbols
        if selected_symbol is None:
            for symbol, option_data in chain.items():
                strike_from_attr = getattr(option_data, "strike_price", None)
                if strike_from_attr is None:
                    strike = self._extract_strike_from_symbol(symbol)
                else:
                    try:
                        strike = float(strike_from_attr)
                    except Exception:
                        strike = None
                if strike is None:
                    continue
                diff = abs(strike - strike_price)
                if diff < smallest_abs_diff:
                    smallest_abs_diff = diff
                    selected_symbol = symbol
                    selected_option_data = option_data

        if selected_symbol is None or selected_option_data is None:
            raise ValueError(
                f"Unable to determine nearest strike to {strike_price} for {underlying_symbol} {option_type} {expiration_date}"
            )

        # Compute mid price defensively in case attributes are missing
        quote = getattr(selected_option_data, "latest_quote", None)
        bid_price = (
            getattr(quote, "bid_price", None)
            if quote is not None
            else getattr(selected_option_data, "bid_price", None)
        )
        ask_price = (
            getattr(quote, "ask_price", None)
            if quote is not None
            else getattr(selected_option_data, "ask_price", None)
        )
        if bid_price is None or ask_price is None:
            raise ValueError(
                f"Quote data unavailable for {selected_symbol}. Try adjusting tolerance or choosing a different strike/expiry."
            )
        mid_price = (bid_price + ask_price) / 2
        return OptionMidQuote(
            symbol=selected_symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            mid_price=mid_price,
        )


def compute_implied_move_percent(straddle_price: float, current_price: int) -> float:
    """Return the implied move percentage rounded to two decimals."""
    return round((straddle_price / current_price) * 100, 2)


def prompt_for_stock_symbol(data_service: MarketDataService) -> Optional[str]:
    """Prompt the user until a valid stock ticker is entered or 'q' to quit.

    Returns None if the user requests to quit.
    """
    while True:
        raw = input("Enter the stock symbol (or 'q' to quit): ")
        text = (raw or "").strip()
        if text.lower() in {"q", "quit"}:
            return None
        if not text:
            print("Symbol cannot be empty. Please try again.")
            continue
        symbol = text.upper()
        if data_service.is_valid_stock_symbol(symbol):
            return symbol
        print(f"'{symbol}' is not a valid ticker. Please try again.")


def _load_credentials_from_env() -> tuple[str, str]:
    """Load Alpaca credentials from environment (supports .env in development)."""
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing API credentials. Set ALPACA_API_KEY and ALPACA_API_SECRET in your environment or .env file."
        )
    return api_key, api_secret


def main() -> None:
    api_key, api_secret = _load_credentials_from_env()
    data_service = MarketDataService(api_key, api_secret)

    while True:
        try:
            stock_symbol = prompt_for_stock_symbol(data_service)
            if stock_symbol is None:
                print("Exiting.")
                return

            expiration_date = DEFAULT_EXPIRATION_DATE

            current_price = data_service.get_current_stock_price(stock_symbol)
            strike_price = float(current_price)

            # Attempt to fetch option quotes; if unavailable, re-prompt
            call_quote = data_service.get_option_mid_quote(
                underlying_symbol=stock_symbol,
                option_type="call",
                strike_price=strike_price,
                expiration_date=expiration_date,
                tolerance_ratio=DEFAULT_TOLERANCE_RATIO,
            )

            put_quote = data_service.get_option_mid_quote(
                underlying_symbol=stock_symbol,
                option_type="put",
                strike_price=strike_price,
                expiration_date=expiration_date,
                tolerance_ratio=DEFAULT_TOLERANCE_RATIO,
            )

            call_mid_price = call_quote.mid_price
            put_mid_price = put_quote.mid_price
            straddle_price = put_mid_price + call_mid_price

            implied_move = compute_implied_move_percent(straddle_price, current_price)
            upper_bound = current_price + straddle_price
            lower_bound = current_price - straddle_price

            # concise summary box
            line1_plain = (
                f"{stock_symbol}  ${current_price:.2f} | Exp: {expiration_date}"
            )
            line2_plain = f"ATM: ${strike_price:.2f}"
            line3_plain = f"Implied Move: ±{implied_move:.2f}%"
            line4_plain = f"Range: ${lower_bound:.2f} – ${upper_bound:.2f}"

            line1_styled = (
                f"{_style(stock_symbol, BOLD, FG_WHITE)}  "
                f"{_style(f'${current_price:.2f}', FG_YELLOW, BOLD)} | Exp: "
                f"{_style(expiration_date, FG_CYAN)}"
            )
            line2_styled = f"ATM: {_style(f'${strike_price:.2f}', FG_WHITE)}"
            line3_styled = (
                f"Implied Move: {_style(f'±{implied_move:.2f}%', FG_YELLOW, BOLD)}"
            )
            line4_styled = _style(line4_plain, DIM)

            lines_plain = [line1_plain, line2_plain, line3_plain, line4_plain]
            lines_styled = [line1_styled, line2_styled, line3_styled, line4_styled]

            content_width = max(len(s) for s in lines_plain)
            horizontal = "─" * (content_width + 2)
            top = _style("┌" + horizontal + "┐", FG_CYAN, DIM)
            bottom = _style("└" + horizontal + "┘", FG_CYAN, DIM)

            print(top)
            for plain, styled in zip(lines_plain, lines_styled):
                padding = " " * (content_width - len(plain))
                print("│ " + styled + padding + " │")
            print(bottom)

            # Success; exit after one successful run
            return

        except ValueError as e:
            print(f"{e}")
            print("Please try a different ticker or type 'q' to quit.")
            continue
        except KeyboardInterrupt:
            print("\nExiting.")
            return


if __name__ == "__main__":
    main()
