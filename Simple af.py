"""
Simple Basketball Trading Strategy - Focus on What Actually Matters

Core factors that determine basketball games:
1. Score differential vs time remaining
2. Recent momentum (who's scoring)
3. Turnovers (possession changes)

That's it. Keep it simple and focus on execution.
"""

from enum import Enum
from typing import Optional
import math

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
    """Simple basketball trading strategy focused on core factors."""

    def reset_state(self) -> None:
        """Reset game state."""
        # Basic game info
        self.home_score = 0
        self.away_score = 0
        self.time_remaining = 2880.0
        
        # Recent momentum tracking (last 10 events)
        self.recent_scores = []  # List of (team, points, time)
        self.recent_turnovers = []  # List of (team, time)
        
        # Market info
        self.position_size = 0
        self.best_bid = None
        self.best_ask = None
        
        # Simple parameters
        self.min_edge = 0.03  # Need 3% edge to trade
        self.max_position = 8.0

    def __init__(self) -> None:
        self.reset_state()

    def calculate_win_probability(self) -> float:
        """Calculate home team win probability using core factors."""
        
        # Factor 1: Score + Time (most important)
        score_diff = self.home_score - self.away_score
        time_factor = 1 - (self.time_remaining / 2880.0)  # 0 to 1
        
        # Score matters more as time runs out
        time_weight = 1 + (2 * time_factor)  # 1 to 3
        weighted_diff = score_diff * time_weight
        
        # Convert to probability using sigmoid
        base_prob = 1 / (1 + math.exp(-weighted_diff / 8.0))
        
        # Factor 2: Recent momentum (last 5 minutes of scoring)
        current_time = self.time_remaining
        recent_home_points = 0
        recent_away_points = 0
        
        for team, points, time in self.recent_scores:
            if abs(time - current_time) <= 300:  # Last 5 minutes
                if team == "home":
                    recent_home_points += points
                elif team == "away":
                    recent_away_points += points
        
        momentum_diff = recent_home_points - recent_away_points
        momentum_adj = math.tanh(momentum_diff / 6.0) * 0.08  # Max 8% adjustment
        
        # Factor 3: Recent turnovers (last 3 minutes)
        recent_home_tos = 0
        recent_away_tos = 0
        
        for team, time in self.recent_turnovers:
            if abs(time - current_time) <= 180:  # Last 3 minutes
                if team == "home":
                    recent_home_tos += 1
                elif team == "away":
                    recent_away_tos += 1
        
        turnover_diff = recent_away_tos - recent_home_tos  # Good for home if away has more
        turnover_adj = turnover_diff * 0.03  # 3% per turnover
        
        # Combine factors
        final_prob = base_prob + momentum_adj + turnover_adj
        
        # Keep in bounds
        return max(0.01, min(0.99, final_prob))

    def should_trade(self) -> tuple[bool, Side, float]:
        """Simple trading decision."""
        if self.best_bid is None or self.best_ask is None:
            return False, None, 0
        
        # Our probability vs market probability
        our_prob = self.calculate_win_probability()
        market_price = (self.best_bid + self.best_ask) / 2
        market_prob = market_price / 100.0
        
        edge = our_prob - market_prob
        
        # Need minimum edge
        if abs(edge) < self.min_edge:
            return False, None, 0
        
        # Position size based on edge size
        size = min(abs(edge) * 15, 3.0)  # Max 3 contracts per trade
        
        # Check position limits
        proposed_position = self.position_size + (size if edge > 0 else -size)
        if abs(proposed_position) > self.max_position:
            return False, None, 0
        
        side = Side.BUY if edge > 0 else Side.SELL
        return True, side, size

    def update_recent_data(self, event_type: str, home_away: str, 
                          home_score: int, away_score: int, shot_type: str = None):
        """Update recent momentum data."""
        
        # Track scoring events
        if event_type == "SCORE":
            points = 0
            if shot_type == "THREE_POINT":
                points = 3
            elif shot_type == "FREE_THROW":
                points = 1
            else:
                points = 2
            
            self.recent_scores.append((home_away, points, self.time_remaining))
            
            # Keep only recent events (last 15 scores)
            if len(self.recent_scores) > 15:
                self.recent_scores.pop(0)
        
        # Track turnovers
        elif event_type == "TURNOVER":
            self.recent_turnovers.append((home_away, self.time_remaining))
            
            # Keep only recent turnovers (last 10)
            if len(self.recent_turnovers) > 10:
                self.recent_turnovers.pop(0)

    def execute_trade(self, side: Side, size: float, edge: float):
        """Execute trade with logging."""
        print(f"TRADE: {side} {size:.1f} contracts | Edge: {edge:.3f} | Score: {self.home_score}-{self.away_score} | Time: {self.time_remaining:.0f}")
        place_market_order(side, Ticker.TEAM_A, size)

    # Required interface methods
    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        pass

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

    def on_game_event_update(self, event_type: str, home_away: str, home_score: int,
                           away_score: int, player_name: Optional[str],
                           substituted_player_name: Optional[str], shot_type: Optional[str],
                           assist_player: Optional[str], rebound_type: Optional[str],
                           coordinate_x: Optional[float], coordinate_y: Optional[float],
                           time_seconds: Optional[float]) -> None:
        """Main event processing - keep it simple."""
        
        # Update basic state
        if time_seconds is not None:
            self.time_remaining = time_seconds
        self.home_score = home_score
        self.away_score = away_score
        
        # Skip meaningless events
        if event_type in ["NOTHING", "UNKNOWN"]:
            return
        
        # Update our tracking data
        self.update_recent_data(event_type, home_away, home_score, away_score, shot_type)
        
        # Make trading decision on important events only
        if event_type in ["SCORE", "TURNOVER", "STEAL"]:
            should_trade, side, size = self.should_trade()
            
            if should_trade:
                our_prob = self.calculate_win_probability()
                market_prob = (self.best_bid + self.best_ask) / 200 if self.best_bid and self.best_ask else 0.5
                edge = our_prob - market_prob
                self.execute_trade(side, size, edge)
        
        # Basic event logging
        if event_type not in ["NOTHING"]:
            win_prob = self.calculate_win_probability()
            print(f"{event_type} | {home_score}-{away_score} | P: {win_prob:.3f} | Pos: {self.position_size}")
        
        # Reset on game end
        if event_type == "END_GAME":
            print(f"Game ended: {home_score}-{away_score} | Final position: {self.position_size}")
            self.reset_state()

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        if bids:
            self.best_bid = bids[0][0]
        if asks:
            self.best_ask = asks[0][0]