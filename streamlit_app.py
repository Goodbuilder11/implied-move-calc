from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import os

import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import (
    OptionChainRequest,
    StockBarsRequest,
)
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

from implied_move import (
    DEFAULT_TOLERANCE_RATIO,
    ImpliedMoveResult,
    MarketDataService,
    compute_implied_move,
)


# --------------
# Page settings
# --------------
st.set_page_config(
    page_title="Implied Move Dashboard",
    page_icon=None,
    layout="wide",
)


# --------------
# Helpers / Caching
# --------------
@st.cache_resource(show_spinner=False)
def get_clients() -> Tuple[StockHistoricalDataClient, OptionHistoricalDataClient]:
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing API credentials. Set ALPACA_API_KEY and ALPACA_API_SECRET in environment or .env."
        )
    stock_client = StockHistoricalDataClient(api_key, api_secret)
    option_client = OptionHistoricalDataClient(api_key, api_secret)
    return stock_client, option_client


@st.cache_resource(show_spinner=False)
def get_data_service() -> MarketDataService:
    stock_client, option_client = get_clients()
    # Reuse the wrapper class for consistency
    # Note: The class internally constructs clients, so we re-init with creds
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    return MarketDataService(api_key, api_secret)  # type: ignore[arg-type]


@st.cache_data(show_spinner=False, ttl=60)
def fetch_current_price(symbol: str) -> float:
    data_service = get_data_service()
    return float(data_service.get_current_stock_price(symbol))


@st.cache_data(show_spinner=False, ttl=60)
def fetch_option_chain(
    underlying_symbol: str,
    expiration_date: Optional[str],
    strike_min: Optional[float],
    strike_max: Optional[float],
    option_type: Optional[str] = None,  # "call" | "put" | None (both)
) -> Dict[str, Any]:
    """Fetch option chain from Alpaca; returns mapping keyed by symbol.

    If option_type is None, fetch both and merge.
    """
    _, option_client = get_clients()

    def _req(t: Optional[str]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "underlying_symbol": underlying_symbol,
        }
        if t is not None:
            kwargs["type"] = t
        if expiration_date:
            kwargs["expiration_date"] = expiration_date
        if strike_min is not None:
            kwargs["strike_price_gte"] = strike_min
        if strike_max is not None:
            kwargs["strike_price_lte"] = strike_max
        chain = option_client.get_option_chain(OptionChainRequest(**kwargs))
        if not chain:
            return {}
        if isinstance(chain, dict):
            return chain
        try:
            return {getattr(item, "symbol"): item for item in chain}
        except Exception:
            return {str(i): item for i, item in enumerate(list(chain))}

    if option_type is None:
        a = _req("call")
        b = _req("put")
        a.update(b)
        return a
    return _req(option_type)


def _safe_date(val: Any) -> Optional[pd.Timestamp]:
    try:
        if val is None:
            return None
        if hasattr(val, "date"):
            return pd.Timestamp(val.date())
        return pd.Timestamp(val)
    except Exception:
        return None


def _safe_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def option_chain_to_df(
    chain_map: Dict[str, Any],
    default_type: Optional[str] = None,
) -> pd.DataFrame:
    """Convert Alpaca chain mapping to a tidy DataFrame."""
    records: List[Dict[str, Any]] = []
    for symbol, opt in chain_map.items():
        quote = getattr(opt, "latest_quote", None)
        bid = _safe_float(getattr(quote, "bid_price", getattr(opt, "bid_price", None)))
        ask = _safe_float(getattr(quote, "ask_price", getattr(opt, "ask_price", None)))
        mid = (bid + ask) / 2.0 if (bid is not None and ask is not None) else None
        strike = _safe_float(getattr(opt, "strike_price", None))
        expiry = getattr(opt, "expiration_date", None)
        expiry_ts = _safe_date(expiry)
        # type can be missing; try to infer from symbol
        t = getattr(opt, "type", None) or getattr(opt, "option_type", None)
        if not t:
            try:
                tail = symbol.split(":")[-1]
                t = "call" if "C" in tail else ("put" if "P" in tail else default_type)
            except Exception:
                t = default_type
        iv = _safe_float(getattr(opt, "implied_volatility", None))
        oi = _safe_float(getattr(opt, "open_interest", None))
        volume = _safe_float(getattr(opt, "volume", None))

        records.append(
            {
                "symbol": symbol,
                "type": t,
                "strike": strike,
                "expiration": expiry_ts,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread": (
                    (ask - bid) if (bid is not None and ask is not None) else None
                ),
                "iv": iv,
                "open_interest": oi,
                "volume": volume,
            }
        )

    # Ensure we return a DataFrame with expected columns even if empty
    expected_cols = [
        "symbol",
        "type",
        "strike",
        "expiration",
        "bid",
        "ask",
        "mid",
        "spread",
        "iv",
        "open_interest",
        "volume",
    ]
    if not records:
        return pd.DataFrame(columns=expected_cols)

    df = pd.DataFrame.from_records(records)
    # Sort only if columns exist
    sort_cols = [c for c in ["strike", "type"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=True, ignore_index=True)
    return df


@st.cache_data(show_spinner=False, ttl=300)
def fetch_stock_bars(symbol: str, days: int = 60) -> pd.DataFrame:
    stock_client, _ = get_clients()
    end = datetime.utcnow()
    start = end - timedelta(days=days + 5)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        adjustment="split",
        feed="iex",
        limit=1000,
    )
    bars = stock_client.get_stock_bars(req)
    try:
        data = bars[symbol]  # type: ignore[index]
    except Exception:
        data = getattr(bars, symbol, None) or bars

    # Normalize to a DataFrame
    rows: List[Dict[str, Any]] = []
    for bar in list(data):
        rows.append(
            {
                "timestamp": pd.to_datetime(getattr(bar, "timestamp", None)),
                "open": _safe_float(getattr(bar, "open", None)),
                "high": _safe_float(getattr(bar, "high", None)),
                "low": _safe_float(getattr(bar, "low", None)),
                "close": _safe_float(getattr(bar, "close", None)),
                "volume": _safe_float(getattr(bar, "volume", None)),
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["timestamp"]).sort_values("timestamp")
    return df


def make_dark_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117"
    )
    return fig


# --------------
# Sidebar Controls
# --------------
with st.sidebar:
    st.markdown("### Controls")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    expiration = st.text_input("Expiration (YYYY-MM-DD)", value=today_str)
    tolerance_pct = st.slider(
        "Strike tolerance for ATM selection (±%)",
        min_value=1,
        max_value=50,
        value=int(DEFAULT_TOLERANCE_RATIO * 100),
        step=1,
    )
    window_pct = st.slider(
        "Strike window for chain charts (±% of spot)",
        min_value=5,
        max_value=50,
        value=20,
        step=1,
        help="Defines how far above/below spot to fetch strikes for visualization.",
    )
    days_hist = st.slider(
        "Underlying history (days)", min_value=20, max_value=365, value=90, step=10
    )
    run_btn = st.button("Run / Refresh", use_container_width=True)


st.title("Implied Move Dashboard")
st.caption("Dashboard for at-the-money straddle analysis and option chain insights.")


def run_analysis() -> (
    Tuple[Optional[ImpliedMoveResult], Optional[pd.DataFrame], Optional[float]]
):
    if not ticker:
        st.warning("Enter a ticker to begin.")
        return None, None, None

    try:
        spot = fetch_current_price(ticker)
    except Exception as e:
        st.error(f"Could not fetch current price for {ticker}: {e}")
        return None, None, None

    # Compute implied move using core logic
    try:
        result = compute_implied_move(
            symbol=ticker,
            expiration_date=expiration,
            tolerance_ratio=float(tolerance_pct) / 100.0,
            data_service=get_data_service(),
        )
    except Exception as e:
        st.error(f"Implied move calculation failed: {e}")
        result = None

    # Fetch option chain around spot for charts
    strike_min = spot * (1 - window_pct / 100.0)
    strike_max = spot * (1 + window_pct / 100.0)
    try:
        chain_map = fetch_option_chain(
            underlying_symbol=ticker,
            expiration_date=expiration,
            strike_min=strike_min,
            strike_max=strike_max,
            option_type=None,
        )
        chain_df = option_chain_to_df(chain_map)
        if chain_df is not None and not chain_df.empty:
            # Derivations only when data exists
            chain_df["moneyness"] = chain_df["strike"] / spot
            chain_df["spread_pct"] = chain_df["spread"] / chain_df["mid"]
        else:
            chain_df = None
    except Exception as e:
        st.warning(f"Could not load option chain for charts: {e}")
        chain_df = None

    return result, chain_df, spot


# Auto-run initially and on button
if "__ran__" not in st.session_state or run_btn:
    st.session_state["__ran__"] = True
    result, chain_df, spot = run_analysis()
else:
    result, chain_df, spot = run_analysis()


# --------------
# Top Metrics
# --------------
if result:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Price", f"${result.current_price:,.2f}")
    c2.metric("Implied Move", f"±{result.implied_move_percent:.2f}%")
    c3.metric("Straddle Mid", f"${result.straddle_price:,.2f}")
    c4.metric("Lower Bound", f"${result.lower_bound:,.2f}")
    c5.metric("Upper Bound", f"${result.upper_bound:,.2f}")
    c6.metric("Expiry", result.expiration_date)

    # Indicator gauge
    try:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=result.implied_move_percent,
                number={"suffix": "%"},
                delta={"reference": 0},
                gauge={
                    "axis": {"range": [0, max(10.0, result.implied_move_percent * 2)]},
                    "bar": {"color": "#6C63FF"},
                    "steps": [
                        {"range": [0, result.implied_move_percent], "color": "#2a2f3a"},
                        {
                            "range": [
                                result.implied_move_percent,
                                max(10.0, result.implied_move_percent * 2),
                            ],
                            "color": "#1a1e27",
                        },
                    ],
                },
                title={"text": "Implied Move Gauge"},
            )
        )
        st.plotly_chart(make_dark_figure(fig_gauge), use_container_width=True)
    except Exception:
        pass


# --------------
# Underlying Price Chart
# --------------
if ticker:
    try:
        bars_df = fetch_stock_bars(ticker, days_hist)
        if not bars_df.empty:
            candle = go.Figure(
                data=[
                    go.Candlestick(
                        x=bars_df["timestamp"],
                        open=bars_df["open"],
                        high=bars_df["high"],
                        low=bars_df["low"],
                        close=bars_df["close"],
                        name="OHLC",
                    )
                ]
            )
            candle.update_layout(title=f"{ticker} Daily Candles", height=400)
            st.plotly_chart(make_dark_figure(candle), use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load underlying bars: {e}")


# --------------
# Option Chain Visuals
# --------------
if chain_df is not None and not chain_df.empty and result is not None:
    st.markdown("### Option Chain (focused window)")

    # Pivot for straddle by strike
    call_df = chain_df[chain_df["type"] == "call"][
        ["strike", "mid", "iv", "open_interest", "volume", "spread", "spread_pct"]
    ]
    put_df = chain_df[chain_df["type"] == "put"][
        ["strike", "mid", "iv", "open_interest", "volume", "spread", "spread_pct"]
    ]
    merged = pd.merge(
        call_df, put_df, on="strike", how="outer", suffixes=("_call", "_put")
    )
    merged["straddle_mid"] = merged[["mid_call", "mid_put"]].sum(axis=1, skipna=True)

    # Charts row 1: Straddle vs Strike + Mid by Type
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=merged["strike"],
                y=merged["straddle_mid"],
                mode="lines+markers",
                name="Straddle Mid",
                line=dict(color="#6C63FF"),
            )
        )
        fig.add_vline(
            x=result.strike_price,
            line_dash="dash",
            line_color="#888",
            annotation_text="ATM",
            annotation_position="top",
        )
        fig.update_layout(
            title="Straddle Mid vs Strike", xaxis_title="Strike", yaxis_title="Price"
        )
        st.plotly_chart(make_dark_figure(fig), use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=call_df["strike"],
                y=call_df["mid"],
                mode="lines+markers",
                name="Call Mid",
                line=dict(color="#3DDC97"),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=put_df["strike"],
                y=put_df["mid"],
                mode="lines+markers",
                name="Put Mid",
                line=dict(color="#FF6B6B"),
            )
        )
        fig2.add_vline(x=result.strike_price, line_dash="dash", line_color="#888")
        fig2.update_layout(
            title="Call/Put Mid vs Strike", xaxis_title="Strike", yaxis_title="Price"
        )
        st.plotly_chart(make_dark_figure(fig2), use_container_width=True)

    # Charts row 2: Bid-Ask spread and (if available) IV Smile
    col3, col4 = st.columns(2)
    with col3:
        fig3 = go.Figure()
        fig3.add_trace(
            go.Scatter(
                x=call_df["strike"],
                y=call_df["spread_pct"],
                mode="lines+markers",
                name="Call Spread %",
                line=dict(color="#F7B32B"),
            )
        )
        fig3.add_trace(
            go.Scatter(
                x=put_df["strike"],
                y=put_df["spread_pct"],
                mode="lines+markers",
                name="Put Spread %",
                line=dict(color="#E15554"),
            )
        )
        fig3.add_vline(x=result.strike_price, line_dash="dash", line_color="#888")
        fig3.update_layout(
            title="Bid-Ask Spread % vs Strike",
            xaxis_title="Strike",
            yaxis_title="Spread / Mid",
        )
        st.plotly_chart(make_dark_figure(fig3), use_container_width=True)

    with col4:
        if call_df["iv"].notna().any() or put_df["iv"].notna().any():
            fig4 = go.Figure()
            if call_df["iv"].notna().any():
                fig4.add_trace(
                    go.Scatter(
                        x=call_df["strike"],
                        y=call_df["iv"],
                        mode="lines+markers",
                        name="Call IV",
                        line=dict(color="#00C7FF"),
                    )
                )
            if put_df["iv"].notna().any():
                fig4.add_trace(
                    go.Scatter(
                        x=put_df["strike"],
                        y=put_df["iv"],
                        mode="lines+markers",
                        name="Put IV",
                        line=dict(color="#B084CC"),
                    )
                )
            fig4.add_vline(x=result.strike_price, line_dash="dash", line_color="#888")
            fig4.update_layout(
                title="IV Smile (if available)",
                xaxis_title="Strike",
                yaxis_title="Implied Volatility",
            )
            st.plotly_chart(make_dark_figure(fig4), use_container_width=True)
        else:
            st.info("Implied volatility not available in this dataset.")

    # Charts row 3: Open interest and volume if available
    col5, col6 = st.columns(2)
    with col5:
        if merged[["open_interest_call", "open_interest_put"]].notna().any().any():
            fig5 = go.Figure()
            if merged["open_interest_call"].notna().any():
                fig5.add_trace(
                    go.Bar(
                        x=merged["strike"],
                        y=merged["open_interest_call"],
                        name="Call OI",
                    )
                )
            if merged["open_interest_put"].notna().any():
                fig5.add_trace(
                    go.Bar(
                        x=merged["strike"], y=merged["open_interest_put"], name="Put OI"
                    )
                )
            fig5.update_layout(barmode="group", title="Open Interest by Strike")
            st.plotly_chart(make_dark_figure(fig5), use_container_width=True)
        else:
            st.info("Open interest not available.")

    with col6:
        if merged[["volume_call", "volume_put"]].notna().any().any():
            fig6 = go.Figure()
            if merged["volume_call"].notna().any():
                fig6.add_trace(
                    go.Bar(
                        x=merged["strike"], y=merged["volume_call"], name="Call Volume"
                    )
                )
            if merged["volume_put"].notna().any():
                fig6.add_trace(
                    go.Bar(
                        x=merged["strike"], y=merged["volume_put"], name="Put Volume"
                    )
                )
            fig6.update_layout(barmode="group", title="Volume by Strike")
            st.plotly_chart(make_dark_figure(fig6), use_container_width=True)
        else:
            st.info("Volume not available.")

    # Data table + download
    with st.expander("View / Download Chain Data"):
        st.dataframe(chain_df, use_container_width=True)
        st.download_button(
            label="Download CSV",
            data=chain_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_{expiration}_chain_window.csv",
            mime="text/csv",
        )


# --------------
# Footer / Debug
# --------------
st.caption(
    "Data from Alpaca. This app estimates an at-the-money straddle and visualizes option chain metrics."
)
