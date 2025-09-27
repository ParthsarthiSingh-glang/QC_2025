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
    """Advanced basketball trading strategy with risk & profit discipline."""

    def reset_state(self) -> None:
        # Game state
        self.home_score = 0
        self.away_score = 0
        self.time_remaining = 2880.0
        self.game_total_time = 2880.0

        # Market tracking
        self.current_market_price = 50.0
        self.best_bid = None
        self.best_ask = None
        self.position_size = 0
        self.capital_remaining = 10000.0  # starting capital (example)

        # Event tracking
        self.recent_events = deque(maxlen=20)
        self.scoring_runs = {"home": 0, "away": 0}
        self.last_score_team = None

        # Metrics
        self.home_efficiency = {"made": 0, "attempted": 0}
        self.away_efficiency = {"made": 0, "attempted": 0}
        self.home_turnovers = 0
        self.away_turnovers = 0
        self.home_fouls = 0
        self.away_fouls = 0

        # Momentum
        self.momentum_score = 0
        self.recent_home_points = deque(maxlen=10)
        self.recent_away_points = deque(maxlen=10)

        # Trading params
        self.min_edge = 0.005
        self.max_position = 10000.0
        self.confidence_threshold = 0.51

        # Risk/profit parameters
        self.profit_target = 30000.0  # cut here
        self.debug = True

    def __init__(self) -> None:
        self.reset_state()

    # ---------- Core win prob models ----------

    def calculate_base_win_probability(self) -> float:
        if self.time_remaining <= 0:
            return 1.0 if self.home_score > self.away_score else 0.0

        score_diff = self.home_score - self.away_score
        time_factor = 1 - (self.time_remaining / self.game_total_time)
        time_weight = 1 + (2 * time_factor)
        adjusted_diff = score_diff * time_weight
        probability = 1 / (1 + math.exp(-adjusted_diff / 8.0))
        return max(0.01, min(0.99, probability))

    def calculate_momentum_adjustment(self) -> float:
        if len(self.recent_events) < 5:
            return 0.0
        momentum_factors = []
        if self.home_efficiency["attempted"] > 0 and self.away_efficiency["attempted"] > 0:
            home_eff = self.home_efficiency["made"] / self.home_efficiency["attempted"]
            away_eff = self.away_efficiency["made"] / self.away_efficiency["attempted"]
            efficiency_diff = home_eff - away_eff
            momentum_factors.append(efficiency_diff * 0.20)
        turnover_diff = self.away_turnovers - self.home_turnovers
        momentum_factors.append(turnover_diff * 0.04)
        run_diff = self.scoring_runs["home"] - self.scoring_runs["away"]
        run_factor = math.tanh(run_diff / 6.0) * 0.08
        momentum_factors.append(run_factor)
        if self.recent_home_points and self.recent_away_points:
            recent_home = sum(self.recent_home_points)
            recent_away = sum(self.recent_away_points)
            recent_diff = recent_home - recent_away
            recent_factor = math.tanh(recent_diff / 10.0) * 0.08
            momentum_factors.append(recent_factor)
        return sum(momentum_factors)

    def calculate_situational_adjustments(self) -> float:
        adjustments = 0.0
        time_factor = 1 - (self.time_remaining / self.game_total_time)
        if time_factor > 0.8:
            score_diff = abs(self.home_score - self.away_score)
            if score_diff <= 3:
                adjustments += 0.02 if self.momentum_score > 0 else -0.02
            if self.away_fouls > self.home_fouls + 2:
                adjustments += 0.03
            elif self.home_fouls > self.away_fouls + 2:
                adjustments -= 0.03
        return adjustments

    def get_win_probability(self) -> float:
        base_prob = self.calculate_base_win_probability()
        momentum_adj = self.calculate_momentum_adjustment()
        situational_adj = self.calculate_situational_adjustments()
        final_prob = base_prob + momentum_adj + situational_adj
        return max(0.01, min(0.99, final_prob))

    # ---------- Trading logic ----------

    def should_trade(self, calculated_prob: float, market_price: float):
        if market_price is None:
            return False, None, 0
        market_prob = market_price if market_price <= 1.0 else market_price / 100.0
        edge = calculated_prob - market_prob
        if abs(edge) < self.min_edge:
            return False, None, 0
        scaling_divisor = 0.02
        confidence = min(abs(edge) / scaling_divisor, 1.0)
        base_size = confidence * self.max_position
        time_factor = 1 - (self.time_remaining / self.game_total_time)
        if time_factor > 0.75:
            base_size *= 1.5
        proposed_size = min(base_size, self.max_position - abs(self.position_size))
        if proposed_size <= 0:
            return False, None, 0
        side = Side.BUY if edge > 0 else Side.SELL
        return True, side, proposed_size

    def execute_trading_decision(self, win_prob: float):
        if self.best_bid is not None and self.best_ask is not None:
            mid_price = (self.best_bid + self.best_ask) / 2
        else:
            mid_price = self.current_market_price
        if mid_price is None:
            return
        should_trade_flag, side, size = self.should_trade(win_prob, mid_price)
        if should_trade_flag:
            print(f"[TRADE] Signal -> {side.name}, size={size:.1f}, win_prob={win_prob:.3f}, market={mid_price:.2f}")
            place_market_order(side, Ticker.TEAM_A, size)

    # ---------- Event & market updates ----------

    def on_trade_update(self, ticker: Ticker, side: Side, quantity: float, price: float) -> None:
        self.current_market_price = price
        print(f"[MARKET] Trade update: {side.name} {quantity} @ {price}")

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
        print(f"[ACCOUNT] Position={self.position_size:.1f}, Capital=${capital_remaining:.2f}")
        # risk/profit discipline check
        self.check_risk_and_take_profit()

    def on_game_event_update(self, event_type: str, home_away: str, home_score: int,
                             away_score: int, player_name: Optional[str],
                             substituted_player_name: Optional[str], shot_type: Optional[str],
                             assist_player: Optional[str], rebound_type: Optional[str],
                             coordinate_x: Optional[float], coordinate_y: Optional[float],
                             time_seconds: Optional[float]) -> None:
        if time_seconds is not None:
            self.time_remaining = time_seconds
        self.home_score = home_score
        self.away_score = away_score
        if event_type in ["NOTHING", "UNKNOWN"]:
            return
        win_prob = self.get_win_probability()
        print(f"[EVENT] {event_type} {home_score}-{away_score} | Time {time_seconds:.0f}s | WinProb {win_prob:.3f}")
        if event_type in ["SCORE", "TURNOVER", "STEAL", "BLOCK", "MISSED"]:
            self.execute_trading_decision(win_prob)
        if event_type == "END_GAME":
            print(f"Game ended {home_score}-{away_score} | Final position {self.position_size}")
            self.reset_state()

    def on_orderbook_snapshot(self, ticker: Ticker, bids: list, asks: list) -> None:
        if bids:
            self.best_bid = bids[0][0]
        if asks:
            self.best_ask = asks[0][0]

    # ---------- Risk & profit discipline ----------

    def check_risk_and_take_profit(self):
        """Cut all positions if profit target is reached or account is drained."""
        if self.capital_remaining is None or self.current_market_price is None:
            return
        total_equity = self.capital_remaining + (self.position_size * self.current_market_price)
        if total_equity <= 0:
            print("[RISK] Account wiped. Flattening.")
            self.reset_state()
            return
        if total_equity >= self.profit_target:
            print(f"[RISK] Profit target hit (${total_equity:.2f}). Cutting positions.")
            if self.position_size > 0:
                place_market_order(Side.SELL, Ticker.TEAM_A, self.position_size)
            elif self.position_size < 0:
                place_market_order(Side.BUY, Ticker.TEAM_A, abs(self.position_size))
            self.reset_state()
