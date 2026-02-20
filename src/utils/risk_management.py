import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger
from config.settings import INITIAL_CAPITAL

class RiskManager:
    """Risk management for options strategies"""
    
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = 0.02  # 2% max risk per trade
        self.max_portfolio_risk = 0.20  # 20% max portfolio risk
        self.max_correlation_exposure = 0.50  # 50% max in correlated positions
        
    def calculate_position_size(self, strategy_max_loss: float, 
                              confidence_score: float = 0.5) -> int:
        """
        Calculate position size based on risk parameters
        
        Args:
            strategy_max_loss: Maximum loss per contract
            confidence_score: Strategy confidence (0-1)
            
        Returns:
            Number of contracts to trade
        """
        if strategy_max_loss <= 0:
            return 0
        
        # Base risk amount
        risk_amount = self.current_capital * self.max_risk_per_trade
        
        # Adjust for confidence
        adjusted_risk = risk_amount * confidence_score
        
        # Calculate position size
        position_size = int(adjusted_risk / strategy_max_loss)
        
        return max(1, position_size)  # Minimum 1 contract
    
    def check_trade_approval(self, strategy_summary: Dict[str, Any], 
                           existing_positions: List[Dict] = None) -> Dict[str, Any]:
        """
        Check if a trade meets risk criteria
        
        Args:
            strategy_summary: Strategy details
            existing_positions: Current positions
            
        Returns:
            Approval decision and reasoning
        """
        approval = {
            'approved': True,
            'reasons': [],
            'warnings': [],
            'suggested_size': 1
        }
        
        existing_positions = existing_positions or []
        
        # Check individual trade risk
        max_loss = strategy_summary.get('max_loss', 0)
        if max_loss > self.current_capital * self.max_risk_per_trade:
            approval['approved'] = False
            approval['reasons'].append(f"Trade risk (${max_loss}) exceeds max per trade (${self.current_capital * self.max_risk_per_trade:.2f})")
        
        # Check portfolio risk
        total_current_risk = sum(pos.get('max_loss', 0) for pos in existing_positions)
        if total_current_risk + max_loss > self.current_capital * self.max_portfolio_risk:
            approval['approved'] = False
            approval['reasons'].append(f"Portfolio risk would exceed maximum ({self.max_portfolio_risk*100}%)")
        
        # Check for over-concentration
        same_strategy_count = sum(1 for pos in existing_positions 
                                if pos.get('strategy_name') == strategy_summary.get('strategy_name'))
        
        if same_strategy_count >= 5:  # Max 5 of same strategy
            approval['warnings'].append(f"High concentration in {strategy_summary.get('strategy_name')} strategy")
        
        # Calculate suggested position size
        if approval['approved']:
            confidence = self._calculate_confidence_score(strategy_summary)
            approval['suggested_size'] = self.calculate_position_size(max_loss, confidence)
        
        return approval
    
    def _calculate_confidence_score(self, strategy_summary: Dict[str, Any]) -> float:
        """Calculate confidence score based on strategy metrics"""
        # This is a simplified version - you can enhance based on technical indicators
        base_confidence = 0.5
        
        # Adjust based on profit/loss ratio
        max_profit = strategy_summary.get('max_profit', 0)
        max_loss = strategy_summary.get('max_loss', 1)
        
        if max_loss > 0:
            profit_ratio = max_profit / max_loss
            if profit_ratio > 0.5:
                base_confidence += 0.2
            elif profit_ratio < 0.3:
                base_confidence -= 0.2
        
        return max(0.1, min(1.0, base_confidence))
    
    def update_capital(self, pnl: float):
        """Update current capital based on trade P&L"""
        self.current_capital += pnl
        logger.info(f"Capital updated: {self.current_capital:.2f} (P&L: {pnl:.2f})")
    
    def get_risk_metrics(self, positions: List[Dict] = None) -> Dict[str, Any]:
        """Get current risk metrics"""
        positions = positions or []
        
        total_risk = sum(pos.get('max_loss', 0) for pos in positions)
        total_exposure = sum(pos.get('entry_debit', pos.get('entry_credit', 0)) for pos in positions)
        
        return {
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'total_return': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100,
            'total_risk': total_risk,
            'risk_percentage': (total_risk / self.current_capital) * 100,
            'total_exposure': total_exposure,
            'exposure_percentage': (total_exposure / self.current_capital) * 100,
            'available_risk': (self.current_capital * self.max_portfolio_risk) - total_risk,
            'num_positions': len(positions)
        }

class PerformanceTracker:
    """Track strategy performance metrics"""
    
    def __init__(self):
        self.trades = []
        self.daily_pnl = []
        
    def add_trade(self, trade_result: Dict[str, Any]):
        """Add a completed trade"""
        self.trades.append(trade_result)
        
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {}
        
        df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = df['pnl'].sum()
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        max_win = df['pnl'].max()
        max_loss = df['pnl'].min()
        
        # Advanced metrics
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else float('inf')
        
        # Sharpe ratio (simplified)
        returns = df['pnl']
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Consecutive metrics
        df['win'] = df['pnl'] > 0
        consecutive_wins = self._max_consecutive(df['win'], True)
        consecutive_losses = self._max_consecutive(df['win'], False)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate * 100,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': max_win,
            'max_loss': max_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'expectancy': (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        }
    
    def _max_consecutive(self, series: pd.Series, value: bool) -> int:
        """Calculate maximum consecutive occurrences of a value"""
        if series.empty:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for val in series:
            if val == value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive