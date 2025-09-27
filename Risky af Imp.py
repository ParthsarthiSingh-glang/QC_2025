"""
Advanced Basketball Trading Strategy

This strategy combines statistical modeling with momentum analysis
to predict win probabilities and execute profitable trades.
"""

from enum import Enum
from typing import Optional
import math
from collections import deque
import time

class Side(Enum):
    BUY = 0
    SELL = 1

class Ticker(Enum):
    TEAM_A = 0

def place_market_order(side: Side, ticker: Ticker, quantity: float) -> None:
    # placeholder - replace with real exchange call
    print(f"[EXCHANGE] MARKET {side.name} {quantity:.2f} on {ticker.name}")

def place_limit_order(side: Side, ticker: Ticker, quantity: float, price: float, ioc: bool = False) -> int:
    # placeholder - replace with real exchange call
    print(f"[EXCHANGE] LIMIT {side.name} {quantity:.2f} @ {price:.2f} on {ticker.name} (IOC={ioc})")
    return 0

def cancel_order(ticker: Ticker, order_id: int) -> bool:
    # placeholder - replace with real exchange call
    print(f"[EXCHANGE] CANCEL {order_id} on {ticker.name}")
    return True

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
        self.current_market_price = 50.0  # Start at 50/50 (scale 0-100)
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
        
        # Trading parameters (aggressive / big-investment mode)
        # NOTE: we intentionally allow trading on small edges and permit very large positions.
        self.min_edge = 0.005       # Minimum 0.5% edge to trade (was 0.03)
        self.max_position = 10000.0 # Allow very large exposure (user requested "invest lot")
        self.confidence_threshold = 0.51  # relaxed threshold (may be unused)

        # For debug/troubleshooting
        self.debug = True

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
            momentum_factors.append(efficiency_diff * 0.20)
        
        # Turnover differential
        turnover_diff = self.away_turnovers - self.home_turnovers
        momentum_factors.append(turnover_diff * 0.04)
        
        # Current scoring run
        run_diff = self.scoring_runs["home"] - self.scoring_runs["away"]
        run_factor = math.tanh(run_diff / 6.0) * 0.08  # Cap at ~8%
        momentum_factors.append(run_factor)
        
        # Recent point differential (last 10 scoring events)
        if self.recent_home_points and self.recent_away_points:
            recent_home = sum(self.recent_home_points)
            recent_away = sum(self.recent_away_points)
            recent_diff = recent_home - recent_away
            recent_factor = math.tanh(recent_diff / 10.0) * 0.08
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
            if self.debug:
                print("[DEBUG] should_trade: no market_price (None)")
            return False, None, 0
        
        # Robustly handle market_price expressed as 0..1 or 0..100
        if market_price <= 1.0:
            market_prob = market_price
        else:
            market_prob = market_price / 100.0

        edge = calculated_prob - market_prob

        # Debug print of raw probs
        if self.debug:
            print(f"[DEBUG] should_trade: calc_prob={calculated_prob:.4f}, market_price={market_price:.4f}, market_prob={market_prob:.4f}, edge={edge:.4f}")

        # Require minimum edge (much lower now so we take small positive signals)
        if abs(edge) < self.min_edge:
            if self.debug:
                print(f"[DEBUG] should_trade: edge {edge:.4f} below min_edge {self.min_edge}")
            return False, None, 0
        
        # Aggressive scaling: reach full confidence faster (e.g., 2% edge -> full)
        scaling_divisor = 0.02  # full confidence at 2% edge
        confidence = min(abs(edge) / scaling_divisor, 1.0)
        # Use big allocation: scale up to max_position
        base_size = confidence * self.max_position

        # Adjust size based on game situation: bigger late in game (more conviction)
        time_factor = 1 - (self.time_remaining / self.game_total_time)
        if time_factor > 0.75:  # final quarter
            base_size *= 1.5  # amplify

        # Cap by remaining capacity
        proposed_size = min(base_size, self.max_position - abs(self.position_size))

        # NOTE: user requested to "invest lot even on a bit positive" — so we DO NOT block small sizes.
        if self.debug:
            print(f"[DEBUG] should_trade: confidence={confidence:.3f}, base_size={base_size:.1f}, proposed_size={proposed_size:.1f}")

        # If after capping there's nothing available, skip
        if proposed_size <= 0:
            if self.debug:
                print("[DEBUG] should_trade: proposed_size <= 0 after cap -> skip")
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
        # Determine usable mid price: prefer orderbook mid, otherwise fall back to last trade price
        if self.best_bid is not None and self.best_ask is not None:
            mid_price = (self.best_bid + self.best_ask) / 2
        else:
            # fallback to last traded price or initial current_market_price
            mid_price = self.current_market_price if self.current_market_price is not None else self.current_market_price

        # Defensive: if still None, skip
        if mid_price is None:
            if self.debug:
                print("[DEBUG] execute_trading_decision: no mid_price available -> skip")
            return

        # Debug: show the market & model numbers
        market_prob = mid_price if mid_price <= 1.0 else mid_price / 100.0
        if self.debug:
            print(f"[DEBUG] execute_trading_decision: mid_price={mid_price:.4f}, market_prob={market_prob:.4f}, win_prob={win_prob:.4f}")

        should_trade_flag, side, size = self.should_trade(win_prob, mid_price)
        
        if should_trade_flag:
            # aggressive execution: place market order immediately (user requested big investment behavior)
            print(f"[TRADE] Trading signal -> side={side.name}, size={size:.1f}, win_prob={win_prob:.3f}, market={mid_price:.2f}")
            # Place market order directly (no extra win_prob gate)
            place_market_order(side, Ticker.TEAM_A, size)
        else:
            if self.debug:
                print("[DEBUG] execute_trading_decision: should_trade_flag==False (no trade)")

    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        """Track market trades to understand sentiment."""
        self.current_market_price = price
        if self.debug:
            print(f"[MARKET] Trade update: {side.name} {quantity} @ {price}")
        # no immediate trading here — game events trigger decisions

    def on_orderbook_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        """Track orderbook updates to maintain market prices."""
        if side == Side.BUY:
            self.best_bid = price
        else:
            self.best_ask = price
        if self.debug:
            print(f"[MARKET] Orderbook update: {side.name} {quantity} @ {price}")

    def on_account_update(self, ticker: Ticker, side: Side, price: float, 
                         quantity: float, capital_remaining: float) -> None:
        """Track our position and capital."""
        if side == Side.BUY:
            self.position_size += quantity
        else:
            self.position_size -= quantity
        
        self.capital_remaining = capital_remaining
        if self.debug:
            print(f"[ACCOUNT] Position update: {self.position_size:.1f} contracts, ${capital_remaining:.2f} remaining")

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
            print(f"[EVENT] {event_type} {home_score}-{away_score} | Time: {time_seconds:.0f}s | Win Prob: {win_prob:.3f}")
        
        # Execute trading logic for actionable events
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
        if self.debug:
            print(f"[MARKET] Snapshot: best_bid={self.best_bid}, best_ask={self.best_ask}")
