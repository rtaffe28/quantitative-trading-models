"""
Multi-Signal Ensemble Strategy

Combines the best-performing signals from prior strategy iterations into
a single strategy that stays more fully invested by accepting entries from
multiple uncorrelated signal types:

Signal sources (any one can trigger an entry):
  A. Trend Pullback: ADX > 20 + RSI < 45 + price > 150 SMA
  B. Squeeze Breakout: BB inside KC + momentum positive & accelerating
  C. OBV Divergence: price lower low + OBV higher low + uptrend
  D. VWAP Discount: price > 200 SMA + price < VWAP * 0.97 (new signal)

Each signal fires independently. On any given day, the strategy picks
the best candidates across ALL signal types, scored by a composite metric.

Exit rules (unified):
  - Trailing stop: 8% from highest price since entry
  - Momentum exit: RSI > 75 (overextended)
  - Trend break: price < 200 SMA

Stock filter:
  - Price > 200 SMA (long-term uptrend)
  - Avg daily dollar volume > $15M (liquidity)
  - Universe: broad large-cap + growth mix

The thesis: strategies #1-#4 each had real edges (positive expectancy,
high win rates, low drawdowns) but low capital utilization. By combining
4 uncorrelated signals, we should fire more often and deploy more capital
while maintaining the disciplined entry/exit framework.
"""

import pandas as pd
import numpy as np


def _squeeze(close, high, low):
    """Compute squeeze, returns (prices, close, obv, high, low)."""
    pass


def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _compute_adx(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    return dx.ewm(alpha=alpha, adjust=False).mean()


def _check_signal_a(df, close, current_price, sma_150):
    """Trend Pullback: ADX > 20, RSI < 45, price > 150 SMA."""
    if current_price <= sma_150:
        return False, 0

    high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
    low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']

    adx = _compute_adx(high, low, close, 14)
    current_adx = float(adx.iloc[-1])
    if pd.isna(current_adx) or current_adx <= 20:
        return False, 0

    rsi = _compute_rsi(close, 14)
    current_rsi = float(rsi.iloc[-1])
    if pd.isna(current_rsi) or current_rsi >= 45:
        return False, 0

    # Score: lower RSI = better entry (more oversold in strong trend)
    score = (45 - current_rsi) / 45
    return True, score


def _check_signal_b(df, close, current_price, sma_50):
    """Squeeze Breakout: BB inside KC + positive accelerating momentum."""
    if current_price <= sma_50:
        return False, 0

    high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
    low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']

    # Bollinger Bands
    bb_sma = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    bb_upper = bb_sma + 2.0 * bb_std
    bb_lower = bb_sma - 2.0 * bb_std

    # Keltner Channels
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    kc_ema = close.ewm(span=20, adjust=False).mean()
    kc_upper = kc_ema + 1.5 * atr
    kc_lower = kc_ema - 1.5 * atr

    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    if len(squeeze_on) < 2:
        return False, 0

    is_sq = bool(squeeze_on.iloc[-1])
    was_sq = bool(squeeze_on.iloc[-2])

    if not (is_sq or was_sq):
        return False, 0

    # Momentum
    mom = close - close.rolling(window=20).mean()
    mom_cur = float(mom.iloc[-1])
    mom_prev = float(mom.iloc[-2])
    if pd.isna(mom_cur) or pd.isna(mom_prev):
        return False, 0
    if mom_cur <= 0 or mom_cur <= mom_prev:
        return False, 0

    score = mom_cur / current_price  # normalized momentum
    return True, score


def _check_signal_c(df, close, volume, current_price, sma_200):
    """OBV Divergence: price lower low, OBV higher low, uptrend."""
    if current_price <= sma_200:
        return False, 0

    lookback = 30
    if len(close) < lookback + 5:
        return False, 0

    # OBV
    direction = np.sign(close.diff())
    obv = (direction * volume).fillna(0).cumsum()

    half = lookback // 2
    recent = slice(-half, None)
    earlier = slice(-lookback, -half)

    price_recent_low = float(close.iloc[recent].min())
    price_earlier_low = float(close.iloc[earlier].min())
    obv_recent_low = float(obv.iloc[recent].min())
    obv_earlier_low = float(obv.iloc[earlier].min())

    if any(pd.isna(v) for v in [price_recent_low, price_earlier_low,
                                 obv_recent_low, obv_earlier_low]):
        return False, 0

    if not (price_recent_low < price_earlier_low and obv_recent_low > obv_earlier_low):
        return False, 0

    # Score by how far below the 20 SMA (more discount = more reversion potential)
    sma_20 = float(close.rolling(window=20).mean().iloc[-1])
    if pd.isna(sma_20) or sma_20 == 0:
        return False, 0
    score = max(0, (sma_20 - current_price) / sma_20)
    return True, score


def _check_signal_d(df, close, volume, current_price, sma_200):
    """VWAP Discount: price > 200 SMA but > 3% below 20-day VWAP."""
    if current_price <= sma_200:
        return False, 0

    high = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
    low = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']

    typical_price = (high + low + close) / 3
    tp_volume = typical_price * volume
    rolling_tp_vol = tp_volume.rolling(window=20).sum()
    rolling_vol = volume.rolling(window=20).sum()
    vwap = rolling_tp_vol / rolling_vol

    current_vwap = float(vwap.iloc[-1])
    if pd.isna(current_vwap) or current_vwap == 0:
        return False, 0

    discount = (current_vwap - current_price) / current_vwap
    if discount < 0.03:
        return False, 0

    score = discount  # bigger discount = higher score
    return True, score


def create_ensemble_strategy(
    tickers: list,
    min_dollar_volume: float = 15_000_000,
    max_positions: int = 10,
    trailing_stop_pct: float = 0.08,
    rsi_exit: float = 75.0,
):
    """
    Ensemble strategy combining 4 signal types.

    Args:
        tickers: Universe to scan
        min_dollar_volume: Minimum avg daily dollar volume for liquidity
        max_positions: Max simultaneous positions
        trailing_stop_pct: Trailing stop from peak price (0.08 = 8%)
        rsi_exit: RSI above this triggers exit
    """
    state = {'held': {}}  # ticker -> {entry_price, peak_price, signal_type}

    def strategy(date, portfolio, market_data, actions):
        # --- EXIT pass ---
        for ticker in list(state['held'].keys()):
            if ticker not in market_data['data']:
                continue
            df = market_data['data'][ticker]
            current_price = market_data['prices'].get(ticker, 0)
            if current_price <= 0:
                continue

            info = state['held'][ticker]
            info['peak_price'] = max(info['peak_price'], current_price)

            close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']

            should_exit = False

            # Trailing stop
            if current_price < info['peak_price'] * (1 - trailing_stop_pct):
                should_exit = True

            # RSI overextension
            if len(close) >= 14:
                rsi = _compute_rsi(close, 14)
                current_rsi = float(rsi.iloc[-1])
                if not pd.isna(current_rsi) and current_rsi > rsi_exit:
                    should_exit = True

            # Trend break: below 200 SMA
            if len(close) >= 200:
                sma_200 = float(close.rolling(window=200).mean().iloc[-1])
                if not pd.isna(sma_200) and current_price < sma_200:
                    should_exit = True

            if should_exit and ticker in portfolio.positions:
                shares = int(portfolio.positions[ticker].shares)
                if shares > 0:
                    actions.sell_stock(portfolio, ticker, shares, current_price)
                    del state['held'][ticker]

        # --- ENTRY pass ---
        if len(state['held']) >= max_positions:
            return

        candidates = []
        for ticker in tickers:
            if ticker in state['held']:
                continue
            if ticker not in market_data['data']:
                continue

            df = market_data['data'][ticker]
            current_price = market_data['prices'].get(ticker, 0)
            if current_price <= 0 or len(df) < 200:
                continue

            close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
            volume = df['Volume'].squeeze() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']

            # Liquidity filter
            avg_vol = float(volume.tail(20).mean())
            if pd.isna(avg_vol) or avg_vol * current_price < min_dollar_volume:
                continue

            # Precompute shared values
            sma_200 = float(close.rolling(window=200).mean().iloc[-1])
            sma_150 = float(close.rolling(window=150).mean().iloc[-1])
            sma_50 = float(close.rolling(window=50).mean().iloc[-1])
            if any(pd.isna(v) for v in [sma_200, sma_150, sma_50]):
                continue

            # Check all 4 signals
            signals_fired = []

            fired_a, score_a = _check_signal_a(df, close, current_price, sma_150)
            if fired_a:
                signals_fired.append(('trend_pullback', score_a))

            fired_b, score_b = _check_signal_b(df, close, current_price, sma_50)
            if fired_b:
                signals_fired.append(('squeeze_breakout', score_b))

            fired_c, score_c = _check_signal_c(df, close, volume, current_price, sma_200)
            if fired_c:
                signals_fired.append(('obv_divergence', score_c))

            fired_d, score_d = _check_signal_d(df, close, volume, current_price, sma_200)
            if fired_d:
                signals_fired.append(('vwap_discount', score_d))

            if not signals_fired:
                continue

            # Composite score: best individual signal + bonus for multiple signals
            best_score = max(s for _, s in signals_fired)
            signal_count_bonus = len(signals_fired) * 0.1
            composite = best_score + signal_count_bonus
            best_signal = max(signals_fired, key=lambda x: x[1])[0]

            candidates.append((ticker, composite, current_price, best_signal, len(signals_fired)))

        # Rank by composite score
        candidates.sort(key=lambda x: x[1], reverse=True)

        slots = max_positions - len(state['held'])
        for ticker, score, price, signal_type, n_signals in candidates[:slots]:
            allocation = 1.0 / max_positions
            available_cash = portfolio.cash * allocation
            shares = int(available_cash / price)
            if shares > 0:
                actions.buy_stock(portfolio, ticker, shares, price)
                state['held'][ticker] = {
                    'entry_price': price,
                    'peak_price': price,
                    'signal_type': signal_type,
                    'n_signals': n_signals,
                }

    return strategy
