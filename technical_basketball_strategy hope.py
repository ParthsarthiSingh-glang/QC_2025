"""
Basketball Trading Strategy Using Technical Analysis Indicators

This strategy adapts traditional financial technical indicators to basketball data:
- Score differential as "price"
- Win probability as "price" 
- Momentum indicators based on recent performance
- Bollinger Bands around expected performance ranges
"""

from enum import Enum
from typing import Optional, List
import math
from collections import deque

class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    return

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    return 0

class TechnicalStrategy:
    """Basketball strategy using technical analysis indicators adapted for sports."""

    def reset_state(self) -> None:
        """Reset all tracking variables."""
        # Game state
        self.home_score = 0
        self.away_score = 0
        self.time_remaining = 2880.0
        self.game_total_time = 2880.0
        
        # Market tracking
        self.position_size = 0
        self.capital_remaining = None
        self.best_bid = None
        self.best_ask = None
        
        # Time series data (treating time as x-axis, various metrics as y-axis)
        self.score_differential_history = deque(maxlen=50)  # Last 50 data points
        self.win_probability_history = deque(maxlen=50)
        self.home_momentum_history = deque(maxlen=30)
        self.away_momentum_history = deque(maxlen=30)
        self.efficiency_differential_history = deque(maxlen=40)
        
        # Technical indicator values
        self.ma_short = None  # 5-period moving average
        self.ma_long = None   # 15-period moving average
        self.macd_line = None
        self.macd_signal = None
        self.macd_histogram = None
        self.rsi = None
        self.bb_upper = None  # Bollinger Band upper
        self.bb_middle = None # Bollinger Band middle (SMA)
        self.bb_lower = None  # Bollinger Band lower
        
        # Tracking for calculations
        self.home_efficiency = {"made": 0, "attempted": 0}
        self.away_efficiency = {"made": 0, "attempted": 0}
        self.recent_events = deque(maxlen=20)
        self.momentum_scores = deque(maxlen=30)
        
        # Trading parameters
        self.min_edge = 0.025  # 2.5% minimum edge
        self.max_position = 8.0

    def __init__(self) -> None:
        self.reset_state()

    def calculate_base_win_probability(self) -> float:
        """Calculate base win probability using score and time."""
        if self.time_remaining <= 0:
            return 1.0 if self.home_score > self.away_score else 0.0
            
        score_diff = self.home_score - self.away_score
        time_factor = 1 - (self.time_remaining / self.game_total_time)
        
        # Adjust importance of score differential by time remaining
        time_weight = 1 + (2.5 * time_factor)
        adjusted_diff = score_diff * time_weight
        
        # Sigmoid conversion to probability
        probability = 1 / (1 + math.exp(-adjusted_diff / 7.0))
        return max(0.01, min(0.99, probability))

    def calculate_momentum_score(self) -> float:
        """Calculate current momentum (-100 to +100, positive favors home)."""
        if len(self.recent_events) < 5:
            return 0.0
            
        momentum = 0.0
        
        # Weight recent events more heavily
        for i, event in enumerate(self.recent_events):
            weight = (i + 1) / len(self.recent_events)  # More recent = higher weight
            
            if event['type'] == 'SCORE':
                if event['team'] == 'home':
                    momentum += 10 * weight
                elif event['team'] == 'away':
                    momentum -= 10 * weight
            elif event['type'] == 'STEAL':
                if event['team'] == 'home':
                    momentum += 5 * weight
                elif event['team'] == 'away':
                    momentum -= 5 * weight
            elif event['type'] == 'TURNOVER':
                if event['team'] == 'home':
                    momentum -= 3 * weight
                elif event['team'] == 'away':
                    momentum += 3 * weight
        
        return max(-100, min(100, momentum))

    def update_time_series(self) -> None:
        """Update all time series data with current values."""
        # Score differential (home - away)
        score_diff = self.home_score - self.away_score
        self.score_differential_history.append(score_diff)
        
        # Win probability
        win_prob = self.calculate_base_win_probability()
        self.win_probability_history.append(win_prob * 100)  # Convert to 0-100 scale
        
        # Momentum score
        momentum = self.calculate_momentum_score()
        self.momentum_scores.append(momentum)
        
        # Efficiency differential
        if self.home_efficiency["attempted"] > 0 and self.away_efficiency["attempted"] > 0:
            home_eff = self.home_efficiency["made"] / self.home_efficiency["attempted"]
            away_eff = self.away_efficiency["made"] / self.away_efficiency["attempted"]
            eff_diff = (home_eff - away_eff) * 100  # Percentage difference
            self.efficiency_differential_history.append(eff_diff)

    def calculate_moving_averages(self) -> None:
        """Calculate short and long term moving averages on win probability."""
        if len(self.win_probability_history) >= 5:
            self.ma_short = sum(list(self.win_probability_history)[-5:]) / 5
        
        if len(self.win_probability_history) >= 15:
            self.ma_long = sum(list(self.win_probability_history)[-15:]) / 15

    def calculate_macd(self) -> None:
        """Calculate MACD indicator on win probability data."""
        if len(self.win_probability_history) < 15:
            return
            
        # Simple approximation of MACD using moving averages
        if self.ma_short is not None and self.ma_long is not None:
            self.macd_line = self.ma_short - self.ma_long
            
            # Calculate signal line (EMA of MACD line approximated as SMA)
            if len(self.win_probability_history) >= 20:
                # Store MACD values for signal calculation
                if not hasattr(self, 'macd_history'):
                    self.macd_history = deque(maxlen=10)
                self.macd_history.append(self.macd_line)
                
                if len(self.macd_history) >= 5:
                    self.macd_signal = sum(list(self.macd_history)[-5:]) / 5
                    self.macd_histogram = self.macd_line - self.macd_signal

    def calculate_rsi(self) -> None:
        """Calculate RSI on momentum scores."""
        if len(self.momentum_scores) < 15:
            return
            
        # Calculate price changes (momentum changes)
        changes = []
        momentum_list = list(self.momentum_scores)
        for i in range(1, len(momentum_list)):
            changes.append(momentum_list[i] - momentum_list[i-1])
        
        if len(changes) < 14:
            return
            
        # Calculate gains and losses
        gains = [change if change > 0 else 0 for change in changes[-14:]]
        losses = [-change if change < 0 else 0 for change in changes[-14:]]
        
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        
        if avg_loss == 0:
            self.rsi = 100
        else:
            rs = avg_gain / avg_loss
            self.rsi = 100 - (100 / (1 + rs))

    def calculate_bollinger_bands(self) -> None:
        """Calculate Bollinger Bands on score differential."""
        if len(self.score_differential_history) < 20:
            return
            
        data = list(self.score_differential_history)[-20:]  # Last 20 periods
        
        # Calculate middle line (SMA)
        self.bb_middle = sum(data) / len(data)
        
        # Calculate standard deviation
        variance = sum((x - self.bb_middle) ** 2 for x in data) / len(data)
        std_dev = math.sqrt(variance)
        
        # Calculate bands (2 standard deviations)
        self.bb_upper = self.bb_middle + (2 * std_dev)
        self.bb_lower = self.bb_middle - (2 * std_dev)

    def update_all_indicators(self) -> None:
        """Update all technical indicators."""
        self.update_time_series()
        self.calculate_moving_averages()
        self.calculate_macd()
        self.calculate_rsi()
        self.calculate_bollinger_bands()

    def generate_trading_signals(self) -> tuple[bool, Side, float, str]:
        """Generate trading signals based on technical indicators."""
        signals = []
        signal_strength = 0
        reasons = []
        
        current_win_prob = self.calculate_base_win_probability()
        
        # Moving Average Crossover Signal
        if self.ma_short is not None and self.ma_long is not None:
            if self.ma_short > self.ma_long:
                signals.append("MA_BULLISH")
                signal_strength += 2
                reasons.append(f"MA crossover bullish (short: {self.ma_short:.1f}, long: {self.ma_long:.1f})")
            elif self.ma_short < self.ma_long:
                signals.append("MA_BEARISH")
                signal_strength -= 2
                reasons.append(f"MA crossover bearish (short: {self.ma_short:.1f}, long: {self.ma_long:.1f})")
        
        # MACD Signal
        if self.macd_line is not None and self.macd_signal is not None:
            if self.macd_line > self.macd_signal and self.macd_histogram > 0:
                signals.append("MACD_BULLISH")
                signal_strength += 3
                reasons.append(f"MACD bullish (line: {self.macd_line:.2f}, signal: {self.macd_signal:.2f})")
            elif self.macd_line < self.macd_signal and self.macd_histogram < 0:
                signals.append("MACD_BEARISH")
                signal_strength -= 3
                reasons.append(f"MACD bearish (line: {self.macd_line:.2f}, signal: {self.macd_signal:.2f})")
        
        # RSI Oversold/Overbought
        if self.rsi is not None:
            if self.rsi < 30:  # Oversold - bullish signal
                signals.append("RSI_OVERSOLD")
                signal_strength += 2
                reasons.append(f"RSI oversold ({self.rsi:.1f})")
            elif self.rsi > 70:  # Overbought - bearish signal
                signals.append("RSI_OVERBOUGHT")
                signal_strength -= 2
                reasons.append(f"RSI overbought ({self.rsi:.1f})")
        
        # Bollinger Bands Signal
        if all(x is not None for x in [self.bb_upper, self.bb_middle, self.bb_lower]):
            current_score_diff = self.home_score - self.away_score
            
            if current_score_diff <= self.bb_lower:  # Touch lower band - bullish
                signals.append("BB_OVERSOLD")
                signal_strength += 2
                reasons.append(f"Score diff at lower BB ({current_score_diff} <= {self.bb_lower:.1f})")
            elif current_score_diff >= self.bb_upper:  # Touch upper band - bearish
                signals.append("BB_OVERBOUGHT")
                signal_strength -= 2
                reasons.append(f"Score diff at upper BB ({current_score_diff} >= {self.bb_upper:.1f})")
        
        # Determine overall signal
        should_trade = abs(signal_strength) >= 4  # Need multiple confirmations
        side = Side.BUY if signal_strength > 0 else Side.SELL
        
        # Position sizing based on signal strength and confidence
        confidence = min(abs(signal_strength) / 8.0, 1.0)
        position_size = confidence * 2.5  # Max 2.5 contracts per trade
        
        reason_text = "; ".join(reasons) if reasons else "No clear signals"
        
        return should_trade, side, position_size, reason_text

    def execute_trade(self, side: Side, size: float, reason: str) -> None:
        """Execute trade with logging."""
        if size < 0.5:  # Minimum trade size
            return
            
        # Limit total position
        proposed_position = self.position_size + (size if side == Side.BUY else -size)
        if abs(proposed_position) > self.max_position:
            return
            
        print(f"TECHNICAL TRADE: {side} {size:.1f} contracts | Reason: {reason}")
        place_market_order(side, Ticker.TEAM_A, size)

    def should_trade_based_on_market(self, calculated_prob: float) -> tuple[bool, Side, float]:
        """Additional market-based trading logic."""
        if self.best_bid is None or self.best_ask is None:
            return False, None, 0
            
        mid_price = (self.best_bid + self.best_ask) / 2
        market_prob = mid_price / 100.0
        edge = calculated_prob - market_prob
        
        if abs(edge) >= self.min_edge:
            side = Side.BUY if edge > 0 else Side.SELL
            size = min(abs(edge) * 10, 2.0)  # Scale with edge size
            return True, side, size
            
        return False, None, 0

    def update_tracking_data(self, event_type: str, home_away: str, shot_type: str = None) -> None:
        """Update tracking data for technical calculations."""
        # Add to recent events
        if event_type in ["SCORE", "MISSED", "TURNOVER", "STEAL", "BLOCK", "FOUL"]:
            self.recent_events.append({
                "type": event_type,
                "team": home_away,
                "time": self.time_remaining
            })
        
        # Update shooting efficiency
        if event_type in ["SCORE", "MISSED"] and shot_type:
            if home_away == "home":
                self.home_efficiency["attempted"] += 1
                if event_type == "SCORE":
                    self.home_efficiency["made"] += 1
            elif home_away == "away":
                self.away_efficiency["attempted"] += 1
                if event_type == "SCORE":
                    self.away_efficiency["made"] += 1

    # Trading interface methods
    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        print(f"Market trade: {side} {quantity} @ {price}")

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        if side == Side.BUY:
            self.best_bid = price
        else:
            self.best_ask = price

    def on_account_update(self, ticker: Ticker, side: Side, price: float, 
                         quantity: float, capital_remaining: float) -> None:
        if side == Side.BUY:
            self.position_size += quantity
        else:
            self.position_size -= quantity
        self.capital_remaining = capital_remaining
        print(f"Position: {self.position_size:.1f} contracts, Capital: ${capital_remaining:.2f}")

    def on_game_event_update(self, event_type: str, home_away: str, home_score: int,
                           away_score: int, player_name: Optional[str],
                           substituted_player_name: Optional[str], shot_type: Optional[str],
                           assist_player: Optional[str], rebound_type: Optional[str],
                           coordinate_x: Optional[float], coordinate_y: Optional[float],
                           time_seconds: Optional[float]) -> None:
        """Main event processing with technical analysis."""
        
        # Update basic game state
        if time_seconds is not None:
            self.time_remaining = time_seconds
        self.home_score = home_score
        self.away_score = away_score
        
        # Skip non-actionable events
        if event_type in ["NOTHING", "UNKNOWN"]:
            return
            
        # Update tracking data
        self.update_tracking_data(event_type, home_away, shot_type)
        
        # Update all technical indicators
        self.update_all_indicators()
        
        # Generate and execute technical trading signals
        should_trade, side, size, reason = self.generate_trading_signals()
        if should_trade:
            self.execute_trade(side, size, reason)
        
        # Additional market-based trading
        current_prob = self.calculate_base_win_probability()
        market_trade, market_side, market_size = self.should_trade_based_on_market(current_prob)
        if market_trade and not should_trade:  # Don't double-trade
            self.execute_trade(market_side, market_size, f"Market edge: prob={current_prob:.3f}")
        
        # Logging for significant events
        if event_type in ["SCORE", "TURNOVER", "STEAL", "TIMEOUT"]:
            indicators = f"MA: {self.ma_short:.1f}/{self.ma_long:.1f}" if self.ma_short and self.ma_long else "MA: N/A"
            indicators += f" | MACD: {self.macd_line:.2f}" if self.macd_line else " | MACD: N/A"
            indicators += f" | RSI: {self.rsi:.1f}" if self.rsi else " | RSI: N/A"
            print(f"{event_type} {home_score}-{away_score} | {indicators}")
        
        if event_type == "END_GAME":
            print(f"Game ended: {home_score}-{away_score} | Final position: {self.position_size}")
            self.reset_state()

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        if bids:
            self.best_bid = bids[0][0]
        if asks:
            self.best_ask = asks[0][0]