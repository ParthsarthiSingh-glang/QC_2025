"""
Advanced Basketball Trading Strategy

This strategy combines statistical modeling with momentum analysis
to predict win probabilities and execute profitable trades.
"""

from enum import Enum
from typing import Optional
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

class Strategy:
    """Advanced basketball trading strategy with multiple analytical components."""

    def reset_state(self) -> None:
        """Reset all game state tracking variables."""
        # Game state
        self.home_score = 0
        self.away_score = 0
        self.time_remaining = 2880.0  # Default game length
        self.game_total_time = 2880.0
        
        # Market tracking
        self.current_market_price = 50.0  # Start at 50/50
        self.best_bid = None
        self.best_ask = None
        self.position_size = 0
        self.capital_remaining = None
        
        # Event tracking for momentum analysis
        self.recent_events = deque(maxlen=20)  # Last 20 meaningful events
        self.scoring_runs = {"home": 0, "away": 0}  # Current scoring runs
        self.last_score_team = None
        
        # Advanced metrics
        self.home_efficiency = {"made": 0, "attempted": 0}
        self.away_efficiency = {"made": 0, "attempted": 0}
        self.home_turnovers = 0
        self.away_turnovers = 0
        self.home_fouls = 0
        self.away_fouls = 0
        
        # Momentum indicators
        self.momentum_score = 0  # -100 to 100, positive favors home
        self.recent_home_points = deque(maxlen=10)
        self.recent_away_points = deque(maxlen=10)
        
        # Trading parameters
        self.min_edge = 0.03  # Minimum 3% edge to trade
        self.max_position = 10.0  # Maximum position size
        self.confidence_threshold = 0.65  # Minimum confidence for large trades

    def __init__(self) -> None:
        self.reset_state()

    def calculate_base_win_probability(self) -> float:
        """Calculate base win probability using score differential and time."""
        if self.time_remaining <= 0:
            return 1.0 if self.home_score > self.away_score else 0.0
            
        score_diff = self.home_score - self.away_score
        time_factor = 1 - (self.time_remaining / self.game_total_time)
        
        # Base model: score differential becomes more important as time decreases
        time_weight = 1 + (2 * time_factor)  # Weight increases from 1 to 3
        adjusted_diff = score_diff * time_weight
        
        # Sigmoid function to convert to probability
        probability = 1 / (1 + math.exp(-adjusted_diff / 8.0))
        
        return max(0.01, min(0.99, probability))

    def calculate_momentum_adjustment(self) -> float:
        """Calculate momentum-based probability adjustment."""
        if len(self.recent_events) < 5:
            return 0.0
            
        momentum_factors = []
        
        # Recent scoring efficiency
        if self.home_efficiency["attempted"] > 0 and self.away_efficiency["attempted"] > 0:
            home_eff = self.home_efficiency["made"] / self.home_efficiency["attempted"]
            away_eff = self.away_efficiency["made"] / self.away_efficiency["attempted"]
            efficiency_diff = home_eff - away_eff
            momentum_factors.append(efficiency_diff * 0.15)
        
        # Turnover differential
        turnover_diff = self.away_turnovers - self.home_turnovers
        momentum_factors.append(turnover_diff * 0.02)
        
        # Current scoring run
        run_diff = self.scoring_runs["home"] - self.scoring_runs["away"]
        run_factor = math.tanh(run_diff / 6.0) * 0.08  # Cap at ~8%
        momentum_factors.append(run_factor)
        
        # Recent point differential (last 10 scoring events)
        if self.recent_home_points and self.recent_away_points:
            recent_home = sum(self.recent_home_points)
            recent_away = sum(self.recent_away_points)
            recent_diff = recent_home - recent_away
            recent_factor = math.tanh(recent_diff / 10.0) * 0.06
            momentum_factors.append(recent_factor)
        
        return sum(momentum_factors)

    def calculate_situational_adjustments(self) -> float:
        """Calculate adjustments based on specific game situations."""
        adjustments = 0.0
        time_factor = 1 - (self.time_remaining / self.game_total_time)
        
        # Late game scenarios
        if time_factor > 0.8:  # Last 20% of game
            score_diff = abs(self.home_score - self.away_score)
            
            # Close games become more volatile
            if score_diff <= 3:
                adjustments += 0.02 if self.momentum_score > 0 else -0.02
            
            # Foul situation advantages
            if self.away_fouls > self.home_fouls + 2:
                adjustments += 0.03  # Home team advantage in bonus
            elif self.home_fouls > self.away_fouls + 2:
                adjustments -= 0.03
        
        return adjustments

    def get_win_probability(self) -> float:
        """Calculate comprehensive win probability for home team."""
        base_prob = self.calculate_base_win_probability()
        momentum_adj = self.calculate_momentum_adjustment()
        situational_adj = self.calculate_situational_adjustments()
        
        final_prob = base_prob + momentum_adj + situational_adj
        return max(0.01, min(0.99, final_prob))

    def should_trade(self, calculated_prob: float, market_price: float) -> tuple[bool, Side, float]:
        """Determine if we should trade and with what size."""
        if market_price is None:
            return False, None, 0
            
        market_prob = market_price / 100.0
        edge = calculated_prob - market_prob
        
        # Require minimum edge
        if abs(edge) < self.min_edge:
            return False, None, 0
        
        # Calculate position size based on Kelly criterion (simplified)
        confidence = min(abs(edge) / 0.2, 1.0)  # Scale confidence
        base_size = confidence * 3.0  # Base size up to 3 contracts
        
        # Adjust size based on game situation
        time_factor = 1 - (self.time_remaining / self.game_total_time)
        if time_factor > 0.75:  # Increase size in final quarter
            base_size *= 1.5
        
        # Limit position size
        proposed_size = min(base_size, self.max_position - abs(self.position_size))
        
        if proposed_size < 0.5:  # Minimum trade size
            return False, None, 0
        
        side = Side.BUY if edge > 0 else Side.SELL
        return True, side, proposed_size

    def update_event_tracking(self, event_type: str, home_away: str, 
                            home_score: int, away_score: int, shot_type: str = None):
        """Update all tracking variables based on new event."""
        # Track meaningful events
        if event_type in ["SCORE", "MISSED", "TURNOVER", "STEAL", "BLOCK", "FOUL"]:
            self.recent_events.append({
                "type": event_type,
                "team": home_away,
                "time": self.time_remaining
            })
        
        # Update scoring runs
        if event_type == "SCORE":
            points_scored = home_score - self.home_score + away_score - self.away_score
            
            if home_away == "home":
                if self.last_score_team == "home":
                    self.scoring_runs["home"] += points_scored
                else:
                    self.scoring_runs["home"] = points_scored
                    self.scoring_runs["away"] = 0
                self.recent_home_points.append(points_scored)
                self.last_score_team = "home"
            elif home_away == "away":
                if self.last_score_team == "away":
                    self.scoring_runs["away"] += points_scored
                else:
                    self.scoring_runs["away"] = points_scored
                    self.scoring_runs["home"] = 0
                self.recent_away_points.append(points_scored)
                self.last_score_team = "away"
        
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
        
        # Track turnovers and fouls
        if event_type == "TURNOVER":
            if home_away == "home":
                self.home_turnovers += 1
            elif home_away == "away":
                self.away_turnovers += 1
        
        if event_type == "FOUL":
            if home_away == "home":
                self.home_fouls += 1
            elif home_away == "away":
                self.away_fouls += 1

    def execute_trading_decision(self, win_prob: float):
        """Execute trading logic based on calculated probability."""
        if self.best_bid is None or self.best_ask is None:
            return
        
        # Use mid-price for decision making
        mid_price = (self.best_bid + self.best_ask) / 2
        
        should_trade_flag, side, size = self.should_trade(win_prob, mid_price)
        
        if should_trade_flag:
            print(f"Trading signal: {side} {size:.1f} contracts at prob {win_prob:.3f} vs market {mid_price:.1f}")
            
            if side == Side.BUY and win_prob > 0.6:
                # Buy when we think home team is undervalued
                place_market_order(Side.BUY, Ticker.TEAM_A, size)
            elif side == Side.SELL and win_prob < 0.4:
                # Sell when we think home team is overvalued  
                place_market_order(Side.SELL, Ticker.TEAM_A, size)

    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        """Track market trades to understand sentiment."""
        self.current_market_price = price
        print(f"Market trade: {side} {quantity} @ {price}")

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        """Track orderbook updates to maintain market prices."""
        if side == Side.BUY:
            self.best_bid = price
        else:
            self.best_ask = price

    def on_account_update(self, ticker: Ticker, side: Side, price: float, 
                         quantity: float, capital_remaining: float) -> None:
        """Track our position and capital."""
        if side == Side.BUY:
            self.position_size += quantity
        else:
            self.position_size -= quantity
        
        self.capital_remaining = capital_remaining
        print(f"Position update: {self.position_size:.1f} contracts, ${capital_remaining:.2f} remaining")

    def on_game_event_update(self, event_type: str, home_away: str, home_score: int, 
                           away_score: int, player_name: Optional[str], 
                           substituted_player_name: Optional[str], shot_type: Optional[str],
                           assist_player: Optional[str], rebound_type: Optional[str],
                           coordinate_x: Optional[float], coordinate_y: Optional[float],
                           time_seconds: Optional[float]) -> None:
        """Main event processing function."""
        
        if time_seconds is not None:
            self.time_remaining = time_seconds
        
        # Update scores
        self.home_score = home_score
        self.away_score = away_score
        
        # Update tracking
        self.update_event_tracking(event_type, home_away, home_score, away_score, shot_type)
        
        # Skip non-actionable events
        if event_type in ["NOTHING", "UNKNOWN"]:
            return
        
        # Calculate new win probability
        win_prob = self.get_win_probability()
        
        # Log significant events with our assessment
        if event_type != "NOTHING":
            print(f"{event_type} {home_score}-{away_score} | Time: {time_seconds:.0f}s | Win Prob: {win_prob:.3f}")
        
        # Execute trading logic
        if event_type in ["SCORE", "TURNOVER", "STEAL", "BLOCK", "MISSED"]:
            self.execute_trading_decision(win_prob)
        
        # Reset state on game end
        if event_type == "END_GAME":
            print(f"Game ended: {home_score}-{away_score} | Final position: {self.position_size}")
            self.reset_state()

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        """Process orderbook snapshot."""
        if bids:
            self.best_bid = bids[0][0]  # Best bid price
        if asks:
            self.best_ask = asks[0][0]  # Best ask price