"""
Advanced Basketball Trading Strategy for Competition

Key improvements over the previous version:
1. Dynamic time-based probability weighting
2. Shot location analysis with coordinate data
3. Advanced momentum decay modeling
4. Player substitution pattern recognition
5. Adaptive confidence scoring
6. Multi-timeframe analysis
7. Market timing optimization
"""

from enum import Enum
from typing import Optional, Dict, List
import math
from collections import deque, defaultdict

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
    """Competition-optimized basketball trading strategy."""

    def reset_state(self) -> None:
        """Reset all game state variables."""
        # Basic game state
        self.home_score = 0
        self.away_score = 0
        self.time_remaining = 2880.0
        self.game_total_time = 2880.0
        
        # Market state
        self.position_size = 0
        self.capital_remaining = None
        self.best_bid = None
        self.best_ask = None
        self.last_trade_time = None
        
        # Enhanced shot tracking with location analysis
        self.shot_zones = {
            "home": {"paint": {"made": 0, "attempted": 0}, 
                    "mid_range": {"made": 0, "attempted": 0},
                    "three_point": {"made": 0, "attempted": 0},
                    "free_throw": {"made": 0, "attempted": 0}},
            "away": {"paint": {"made": 0, "attempted": 0},
                    "mid_range": {"made": 0, "attempted": 0}, 
                    "three_point": {"made": 0, "attempted": 0},
                    "free_throw": {"made": 0, "attempted": 0}}
        }
        
        # Advanced momentum tracking with decay
        self.momentum_events = deque(maxlen=30)
        self.scoring_sequences = {"home": [], "away": []}
        self.possession_outcomes = deque(maxlen=20)
        
        # Player impact tracking
        self.player_performance = defaultdict(lambda: {"positive": 0, "negative": 0})
        self.active_players = {"home": set(), "away": set()}
        
        # Multi-timeframe probability tracking
        self.probability_snapshots = deque(maxlen=100)
        self.recent_prob_changes = deque(maxlen=10)
        
        # Situational awareness
        self.timeout_usage = {"home": 0, "away": 0}
        self.foul_situation = {"home": 0, "away": 0}
        self.clutch_performance = {"home": {"success": 0, "attempts": 0}, 
                                 "away": {"success": 0, "attempts": 0}}
        
        # Trading optimization
        self.trade_history = []
        self.market_price_history = deque(maxlen=50)
        self.volatility_estimate = 0.05
        
        # Competition parameters - more aggressive than production
        self.min_edge = 0.02  # 2% minimum edge
        self.max_position = 12.0
        self.confidence_multiplier = 1.3

    def __init__(self) -> None:
        self.reset_state()

    def classify_shot_zone(self, x: float, y: float, shot_type: str) -> str:
        """Classify shot location into zones for analysis."""
        if shot_type == "FREE_THROW":
            return "free_throw"
        elif shot_type in ["DUNK", "LAYUP"]:
            return "paint"
        elif shot_type == "THREE_POINT":
            return "three_point"
        else:
            # Use coordinates for mid-range vs paint classification
            if x is not None and y is not None:
                distance_from_basket = math.sqrt((x - 25)**2 + y**2)
                return "paint" if distance_from_basket <= 8 else "mid_range"
            return "mid_range"

    def calculate_advanced_efficiency(self, team: str) -> Dict[str, float]:
        """Calculate zone-specific shooting efficiency."""
        zones = self.shot_zones[team]
        efficiency = {}
        
        for zone, stats in zones.items():
            if stats["attempted"] > 0:
                pct = stats["made"] / stats["attempted"]
                # Weight by expected value
                if zone == "three_point":
                    efficiency[zone] = pct * 3
                elif zone == "free_throw":
                    efficiency[zone] = pct * 1
                else:
                    efficiency[zone] = pct * 2
            else:
                efficiency[zone] = 0
        
        return efficiency

    def calculate_momentum_with_decay(self) -> float:
        """Calculate momentum with time-based decay."""
        if not self.momentum_events:
            return 0
            
        current_time = self.time_remaining
        momentum = 0
        total_weight = 0
        
        for event in self.momentum_events:
            # Calculate time decay (events lose impact over time)
            time_diff = abs(current_time - event["time"])
            decay_factor = math.exp(-time_diff / 300)  # 5-minute half-life
            
            # Base event values
            event_values = {
                "SCORE": 10, "STEAL": 8, "BLOCK": 6, "ASSIST": 4,
                "REBOUND_OFFENSIVE": 5, "REBOUND_DEFENSIVE": 2,
                "TURNOVER": -6, "MISSED": -2, "FOUL": -3
            }
            
            base_value = event_values.get(event["type"], 0)
            
            # Context multipliers
            if event.get("clutch", False):  # Late game situations
                base_value *= 1.5
            if event.get("and_one", False):  # Score + foul
                base_value *= 1.3
            
            weight = decay_factor
            if event["team"] == "home":
                momentum += base_value * weight
            else:
                momentum -= base_value * weight
                
            total_weight += weight
        
        return momentum / max(total_weight, 1) if total_weight > 0 else 0

    def detect_scoring_runs(self) -> Dict[str, int]:
        """Detect current scoring runs for each team."""
        runs = {"home": 0, "away": 0}
        
        if not self.possession_outcomes:
            return runs
            
        current_team = None
        current_run = 0
        
        # Look at recent possessions in reverse order
        for outcome in reversed(self.possession_outcomes):
            if outcome["result"] == "SCORE":
                if current_team == outcome["team"]:
                    current_run += outcome["points"]
                else:
                    if current_team is not None:
                        break  # End of run
                    current_team = outcome["team"]
                    current_run = outcome["points"]
            else:
                break  # Run ended with non-score
        
        if current_team:
            runs[current_team] = current_run
            
        return runs

    def calculate_clutch_factor(self) -> float:
        """Calculate clutch performance factor for late game."""
        time_factor = 1 - (self.time_remaining / self.game_total_time)
        
        if time_factor < 0.75:  # Not clutch time yet
            return 0
            
        clutch_weight = (time_factor - 0.75) * 4  # Scale from 0 to 1
        
        home_clutch = 0
        away_clutch = 0
        
        if self.clutch_performance["home"]["attempts"] > 0:
            home_clutch = self.clutch_performance["home"]["success"] / self.clutch_performance["home"]["attempts"]
        if self.clutch_performance["away"]["attempts"] > 0:
            away_clutch = self.clutch_performance["away"]["success"] / self.clutch_performance["away"]["attempts"]
            
        clutch_diff = (home_clutch - away_clutch) * clutch_weight
        return clutch_diff * 0.15  # Max 15% adjustment

    def calculate_comprehensive_win_probability(self) -> float:
        """Multi-factor win probability calculation."""
        # Base probability from score and time
        score_diff = self.home_score - self.away_score
        time_factor = 1 - (self.time_remaining / self.game_total_time)
        
        # Dynamic time weighting based on game phase
        if time_factor < 0.5:  # First half - score matters less
            time_weight = 1 + time_factor
        elif time_factor < 0.8:  # Third quarter - moderate weighting  
            time_weight = 1.5 + time_factor
        else:  # Final period - score matters most
            time_weight = 2.5 + (2 * time_factor)
            
        adjusted_diff = score_diff * time_weight
        base_prob = 1 / (1 + math.exp(-adjusted_diff / 8.0))
        
        # Efficiency adjustments
        home_eff = self.calculate_advanced_efficiency("home")
        away_eff = self.calculate_advanced_efficiency("away")
        
        efficiency_adj = 0
        for zone in home_eff:
            zone_diff = home_eff[zone] - away_eff[zone]
            efficiency_adj += zone_diff * 0.02  # 2% per efficiency point
            
        # Momentum adjustment with decay
        momentum = self.calculate_momentum_with_decay()
        momentum_adj = math.tanh(momentum / 20) * 0.12  # Cap at 12%
        
        # Scoring run adjustment
        runs = self.detect_scoring_runs()
        run_diff = runs["home"] - runs["away"]
        run_adj = math.tanh(run_diff / 8) * 0.08  # Cap at 8%
        
        # Clutch factor
        clutch_adj = self.calculate_clutch_factor()
        
        # Foul situation (late game)
        foul_adj = 0
        if time_factor > 0.8:
            foul_diff = self.foul_situation["away"] - self.foul_situation["home"]
            foul_adj = math.tanh(foul_diff / 3) * 0.06  # Cap at 6%
        
        # Combine all factors
        final_prob = base_prob + efficiency_adj + momentum_adj + run_adj + clutch_adj + foul_adj
        
        # Ensure bounds
        return max(0.01, min(0.99, final_prob))

    def calculate_confidence_score(self) -> float:
        """Calculate confidence in our probability estimate."""
        confidence_factors = []
        
        # Sample size confidence
        total_shots = sum(sum(zone["attempted"] for zone in team.values()) 
                         for team in self.shot_zones.values())
        sample_confidence = min(total_shots / 50, 1.0)  # Full confidence at 50+ shots
        confidence_factors.append(sample_confidence)
        
        # Momentum consistency
        if len(self.momentum_events) >= 10:
            recent_momentum = [e for e in self.momentum_events if abs(self.time_remaining - e["time"]) < 300]
            if recent_momentum:
                home_events = sum(1 for e in recent_momentum if e["team"] == "home" and e["type"] in ["SCORE", "STEAL"])
                consistency = abs(home_events - len(recent_momentum)/2) / (len(recent_momentum)/2 + 1)
                confidence_factors.append(consistency)
        
        # Time-based confidence (more confident as game progresses)
        time_confidence = 1 - (self.time_remaining / self.game_total_time)
        confidence_factors.append(time_confidence)
        
        # Volatility-based confidence (lower confidence in high volatility)
        if self.probability_snapshots:
            recent_probs = list(self.probability_snapshots)[-10:]
            if len(recent_probs) > 1:
                prob_variance = sum((p - sum(recent_probs)/len(recent_probs))**2 for p in recent_probs) / len(recent_probs)
                volatility_confidence = 1 / (1 + prob_variance * 100)
                confidence_factors.append(volatility_confidence)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5

    def should_trade_advanced(self, calculated_prob: float, confidence: float) -> tuple[bool, Side, float]:
        """Advanced trading decision with market timing."""
        if self.best_bid is None or self.best_ask is None:
            return False, None, 0
            
        # Use bid-ask midpoint
        mid_price = (self.best_bid + self.best_ask) / 2
        market_prob = mid_price / 100.0
        edge = calculated_prob - market_prob
        
        # Dynamic edge threshold based on confidence and game phase
        time_factor = 1 - (self.time_remaining / self.game_total_time)
        base_threshold = self.min_edge * (2 - confidence)  # Lower threshold for high confidence
        
        # Reduce threshold in final quarter (more opportunities)
        if time_factor > 0.75:
            base_threshold *= 0.7
            
        if abs(edge) < base_threshold:
            return False, None, 0
            
        # Position sizing with Kelly-inspired approach
        kelly_fraction = abs(edge) / (1 - calculated_prob if edge > 0 else calculated_prob)
        base_size = kelly_fraction * 4 * confidence * self.confidence_multiplier
        
        # Size multipliers
        size_multipliers = 1.0
        
        # Increase size for high-confidence late game situations
        if time_factor > 0.8 and confidence > 0.7:
            size_multipliers *= 1.4
            
        # Increase size during clear momentum shifts
        momentum = self.calculate_momentum_with_decay()
        if abs(momentum) > 15 and ((edge > 0 and momentum > 0) or (edge < 0 and momentum < 0)):
            size_multipliers *= 1.2
            
        final_size = min(base_size * size_multipliers, 4.0)  # Cap single trade
        
        # Check position limits
        proposed_position = self.position_size + (final_size if edge > 0 else -final_size)
        if abs(proposed_position) > self.max_position:
            final_size = self.max_position - abs(self.position_size)
            
        if final_size < 0.3:  # Minimum viable trade
            return False, None, 0
            
        side = Side.BUY if edge > 0 else Side.SELL
        return True, side, final_size

    def update_player_tracking(self, event_type: str, player_name: str, home_away: str) -> None:
        """Track individual player performance impact."""
        if not player_name or event_type == "SUBSTITUTION":
            return
            
        impact_values = {
            "SCORE": 3, "ASSIST": 2, "STEAL": 2, "BLOCK": 2, 
            "REBOUND": 1, "TURNOVER": -2, "FOUL": -1, "MISSED": -0.5
        }
        
        impact = impact_values.get(event_type, 0)
        if impact > 0:
            self.player_performance[player_name]["positive"] += impact
        else:
            self.player_performance[player_name]["negative"] += abs(impact)

    def process_game_event(self, event_type: str, home_away: str, home_score: int, 
                          away_score: int, shot_type: str = None, 
                          coordinate_x: float = None, coordinate_y: float = None,
                          player_name: str = None) -> None:
        """Process individual game events with enhanced tracking."""
        
        # Track clutch situations
        time_factor = 1 - (self.time_remaining / self.game_total_time)
        is_clutch = time_factor > 0.8 and abs(home_score - away_score) <= 6
        
        # Enhanced momentum event tracking
        if event_type in ["SCORE", "STEAL", "BLOCK", "TURNOVER", "MISSED", "FOUL"]:
            momentum_event = {
                "type": event_type,
                "team": home_away,
                "time": self.time_remaining,
                "clutch": is_clutch,
                "score_before": (self.home_score, self.away_score)
            }
            self.momentum_events.append(momentum_event)
        
        # Shot zone tracking with coordinates
        if event_type in ["SCORE", "MISSED"] and shot_type and home_away in ["home", "away"]:
            zone = self.classify_shot_zone(coordinate_x, coordinate_y, shot_type)
            self.shot_zones[home_away][zone]["attempted"] += 1
            if event_type == "SCORE":
                self.shot_zones[home_away][zone]["made"] += 1
                
                # Track clutch performance
                if is_clutch:
                    self.clutch_performance[home_away]["attempts"] += 1
                    self.clutch_performance[home_away]["success"] += 1
            elif is_clutch:
                self.clutch_performance[home_away]["attempts"] += 1
        
        # Possession outcome tracking
        if event_type in ["SCORE", "TURNOVER", "MISSED"]:
            points_scored = 0
            if event_type == "SCORE":
                if shot_type == "THREE_POINT":
                    points_scored = 3
                elif shot_type == "FREE_THROW":
                    points_scored = 1
                else:
                    points_scored = 2
            
            self.possession_outcomes.append({
                "team": home_away,
                "result": event_type,
                "points": points_scored,
                "time": self.time_remaining
            })
        
        # Player impact tracking
        if player_name:
            self.update_player_tracking(event_type, player_name, home_away)
        
        # Foul tracking
        if event_type == "FOUL" and home_away in ["home", "away"]:
            self.foul_situation[home_away] += 1
            
        # Timeout tracking
        if event_type == "TIMEOUT" and home_away in ["home", "away"]:
            self.timeout_usage[home_away] += 1

    def execute_trade_with_logging(self, side: Side, size: float, edge: float, confidence: float, reason: str) -> None:
        """Execute trade with comprehensive logging."""
        current_prob = self.calculate_comprehensive_win_probability()
        
        trade_info = {
            "time": self.time_remaining,
            "side": side,
            "size": size,
            "edge": edge,
            "confidence": confidence,
            "probability": current_prob,
            "reason": reason,
            "score": (self.home_score, self.away_score)
        }
        self.trade_history.append(trade_info)
        
        print(f"ADVANCED TRADE: {side} {size:.1f} | Edge: {edge:.3f} | Conf: {confidence:.2f} | {reason}")
        place_market_order(side, Ticker.TEAM_A, size)

    # Interface methods
    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        self.market_price_history.append(price)
        
        # Update volatility estimate
        if len(self.market_price_history) >= 10:
            recent_prices = list(self.market_price_history)[-10:]
            price_changes = [abs(recent_prices[i] - recent_prices[i-1]) for i in range(1, len(recent_prices))]
            self.volatility_estimate = sum(price_changes) / len(price_changes) if price_changes else 0.05

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

    def on_game_event_update(self, event_type: str, home_away: str, home_score: int,
                           away_score: int, player_name: Optional[str],
                           substituted_player_name: Optional[str], shot_type: Optional[str],
                           assist_player: Optional[str], rebound_type: Optional[str],
                           coordinate_x: Optional[float], coordinate_y: Optional[float],
                           time_seconds: Optional[float]) -> None:
        """Main event processing with advanced analytics."""
        
        # Update time and scores
        if time_seconds is not None:
            self.time_remaining = time_seconds
        self.home_score = home_score
        self.away_score = away_score
        
        # Skip non-actionable events
        if event_type in ["NOTHING", "UNKNOWN"]:
            return
        
        # Process the event
        self.process_game_event(event_type, home_away, home_score, away_score, 
                               shot_type, coordinate_x, coordinate_y, player_name)
        
        # Calculate probability and confidence
        win_prob = self.calculate_comprehensive_win_probability()
        confidence = self.calculate_confidence_score()
        
        # Store probability snapshot
        self.probability_snapshots.append(win_prob)
        if len(self.probability_snapshots) > 1:
            prob_change = win_prob - self.probability_snapshots[-2]
            self.recent_prob_changes.append(prob_change)
        
        # Trading decision
        should_trade, side, size = self.should_trade_advanced(win_prob, confidence)
        
        if should_trade:
            edge = win_prob - (self.best_bid + self.best_ask) / 200 if self.best_bid and self.best_ask else 0
            reason = f"{event_type} | Mom: {self.calculate_momentum_with_decay():.1f}"
            self.execute_trade_with_logging(side, size, edge, confidence, reason)
        
        # Detailed logging for significant events
        if event_type in ["SCORE", "STEAL", "TURNOVER", "TIMEOUT", "FOUL"]:
            momentum = self.calculate_momentum_with_decay()
            runs = self.detect_scoring_runs()
            print(f"{event_type} {home_score}-{away_score} | P: {win_prob:.3f} | C: {confidence:.2f} | Mom: {momentum:.1f} | Runs: {runs['home']}-{runs['away']}")
        
        if event_type == "END_GAME":
            print(f"Game ended {home_score}-{away_score} | Final position: {self.position_size} | Trades: {len(self.trade_history)}")
            self.reset_state()

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        if bids:
            self.best_bid = bids[0][0]
        if asks:
            self.best_ask = asks[0][0]