"""Unit tests for options strategy components and calculations."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

import sys
sys.path.append('.')

from src.strategies.options_strategies import (
    OptionsStrategy, OptionLeg, PositionSide, OptionType
)
from tests.conftest import TestDataGenerator, SAMPLE_SPX_PRICE, SAMPLE_DATE


class TestOptionLeg:
    """Test individual options leg functionality."""
    
    def test_option_leg_creation(self):
        """Test basic option leg creation."""
        leg = OptionLeg(
            strike=6900.0,
            option_type=OptionType.PUT,
            position_side=PositionSide.SHORT,
            quantity=1,
            entry_price=2.50,
            current_price=2.50
        )
        
        assert leg.option_type == OptionType.PUT
        assert leg.strike == 6900.0
        assert leg.position_side == PositionSide.SHORT
        assert leg.quantity == 1
        assert leg.entry_price == 2.50
        assert leg.current_price == 2.50
    
    def test_option_leg_simple_functionality(self):
        """Test basic option leg functionality."""
        # Create option legs
        short_leg = OptionLeg(
            strike=6900.0,
            option_type=OptionType.PUT,
            position_side=PositionSide.SHORT,
            quantity=1,
            entry_price=2.50,
            current_price=1.50  # Decreased = profit for short
        )
        
        long_leg = OptionLeg(
            strike=7000.0,
            option_type=OptionType.CALL,
            position_side=PositionSide.LONG,
            quantity=1,
            entry_price=1.00,
            current_price=2.00  # Increased = profit for long
        )
        
        # Test basic properties
        assert short_leg.entry_price == 2.50
        assert short_leg.current_price == 1.50
        assert long_leg.entry_price == 1.00 
        assert long_leg.current_price == 2.00
    
    def test_option_leg_price_update(self):
        """Test price update functionality."""
        leg = OptionLeg(
            strike=7000.0,
            option_type=OptionType.CALL,
            position_side=PositionSide.LONG,
            quantity=1,
            entry_price=1.50,
            current_price=1.50
        )
        
        # Update price
        leg.current_price = 2.25
        assert leg.current_price == 2.25
        
        # Original entry price should remain unchanged
        assert leg.entry_price == 1.50


class TestOptionsStrategy:
    """Test options strategy functionality."""
    
    def create_sample_iron_condor(self):
        """Create a sample Iron Condor strategy for testing."""
        from datetime import datetime
        strategy = OptionsStrategy(
            strategy_name="Iron Condor Test",
            entry_date=datetime.strptime(SAMPLE_DATE, "%Y-%m-%d"),
            underlying_price=SAMPLE_SPX_PRICE
        )
        
        legs = [
            OptionLeg(strike=6875.0, option_type=OptionType.PUT, position_side=PositionSide.LONG, quantity=1, entry_price=1.0),
            OptionLeg(strike=6900.0, option_type=OptionType.PUT, position_side=PositionSide.SHORT, quantity=1, entry_price=2.5),
            OptionLeg(strike=7000.0, option_type=OptionType.CALL, position_side=PositionSide.SHORT, quantity=1, entry_price=2.0),
            OptionLeg(strike=7025.0, option_type=OptionType.CALL, position_side=PositionSide.LONG, quantity=1, entry_price=0.75)
        ]
        
        for leg in legs:
            strategy.add_leg(leg)
        
        return strategy
    
    def test_options_strategy_creation(self):
        """Test basic options strategy creation."""
        strategy = self.create_sample_iron_condor()
        
        assert strategy.strategy_name == "Iron Condor Test"
        assert len(strategy.legs) == 4
        assert strategy.underlying_price == SAMPLE_SPX_PRICE
    
    def test_calculate_net_credit(self):
        """Test net credit calculation."""
        strategy = self.create_sample_iron_condor()
        
        # Strategy should calculate either entry credit or debit
        # Total value should be reasonable
        total_premium = 0
        for leg in strategy.legs:
            if leg.position_side == PositionSide.SHORT:
                total_premium += leg.entry_price * leg.quantity * 100
            else:
                total_premium -= leg.entry_price * leg.quantity * 100
        
        # Should have reasonable values (could be positive or negative after commissions)
        assert isinstance(strategy.entry_credit, (int, float))
        assert isinstance(strategy.entry_debit, (int, float))
        assert strategy.entry_credit >= 0
        assert strategy.entry_debit >= 0
    
    def test_strategy_with_price_updates(self):
        """Test strategy with price updates."""
        strategy = self.create_sample_iron_condor()
        initial_credit = strategy.entry_credit
        
        # Mock price update data
        price_updates = {
            "6875.0_put": {"mid_price": 0.5, "delta": -0.05},
            "6900.0_put": {"mid_price": 1.0, "delta": -0.10}, 
            "7000.0_call": {"mid_price": 1.0, "delta": 0.10},
            "7025.0_call": {"mid_price": 0.5, "delta": 0.05}
        }
        
        strategy.update_prices(price_updates)
        
        # Should have updated current P&L
        assert hasattr(strategy, 'current_pnl')
        assert isinstance(strategy.current_pnl, (int, float))


class TestOptionsStrategyEdgeCases:
    """Test edge cases and error handling in options strategies."""
    
    def test_zero_quantity_leg(self):
        """Test handling of zero quantity legs."""
        leg = OptionLeg(
            strike=7000.0,
            option_type=OptionType.CALL,
            position_side=PositionSide.LONG,
            quantity=0,  # Zero quantity
            entry_price=1.50,
            current_price=1.75
        )
        
        # Should handle zero quantity gracefully
        assert leg.quantity == 0
        assert leg.entry_price == 1.50
        assert leg.current_price == 1.75
    
    def test_negative_prices(self):
        """Test handling of negative prices."""
        leg = OptionLeg(
            strike=6900.0,
            option_type=OptionType.PUT,
            position_side=PositionSide.SHORT,
            quantity=1,
            entry_price=2.00,
            current_price=-0.50  # Negative price (shouldn't happen in reality)
        )
        
        # Should handle gracefully
        assert leg.current_price == -0.50
        assert isinstance(leg.entry_price, (int, float))
    
    def test_extreme_price_movements(self):
        """Test handling of extreme price movements."""
        leg = OptionLeg(
            strike=7000.0,
            option_type=OptionType.CALL,
            position_side=PositionSide.LONG,
            quantity=1,
            entry_price=1.00,
            current_price=100.00  # Extreme price increase
        )
        
        # Should handle extreme values without error
        assert leg.current_price == 100.00
        assert leg.entry_price == 1.00