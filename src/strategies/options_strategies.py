import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"

@dataclass
class OptionLeg:
    """Single option leg in a strategy"""
    strike: float
    option_type: OptionType
    position_side: PositionSide
    quantity: int
    entry_price: float
    current_price: float = 0.0
    expiration: datetime = None
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    iv: float = 0.0

class OptionsStrategy:
    """Base class for options strategies"""
    
    def __init__(self, strategy_name: str, entry_date: datetime, 
                 underlying_price: float, commission: float = 0.65):
        self.strategy_name = strategy_name
        self.entry_date = entry_date
        self.underlying_price = underlying_price
        self.commission = commission
        self.legs: List[OptionLeg] = []
        self.entry_credit = 0.0
        self.entry_debit = 0.0
        self.max_profit = 0.0
        self.max_loss = 0.0
        self.breakeven_points = []
        self.current_pnl = 0.0
        self.is_closed = False
        self.exit_date = None
        self.exit_reason = None
    
    def add_leg(self, leg: OptionLeg):
        """Add an option leg to the strategy"""
        self.legs.append(leg)
        self._calculate_strategy_metrics()
    
    def _calculate_strategy_metrics(self):
        """Calculate strategy-level metrics"""
        if not self.legs:
            return
        
        # Calculate net credit/debit
        net_premium = 0.0
        total_commission = 0.0
        
        for leg in self.legs:
            entry_price = leg.entry_price or 0.0  # Handle None
            if leg.position_side == PositionSide.SHORT:
                net_premium += entry_price * leg.quantity * 100  # Options are x100
            else:
                net_premium -= entry_price * leg.quantity * 100
            
            total_commission += self.commission * abs(leg.quantity)
        
        net_premium -= total_commission
        
        if net_premium > 0:
            self.entry_credit = net_premium
        else:
            self.entry_debit = abs(net_premium)
    
    def update_prices(self, options_data: Dict[str, Dict]):
        """Update current prices for all legs"""
        updated_count = 0
        for leg in self.legs:
            key = f"{leg.strike}_{leg.option_type.value}"
            if key in options_data:
                option_data = options_data[key]
                old_price = leg.current_price
                leg.current_price = option_data.get('mid_price', leg.current_price)
                leg.delta = option_data.get('delta', leg.delta)
                leg.gamma = option_data.get('gamma', leg.gamma)
                leg.theta = option_data.get('theta', leg.theta)
                leg.vega = option_data.get('vega', leg.vega)
                leg.iv = option_data.get('iv', leg.iv)
                
                if leg.current_price > 0:
                    updated_count += 1
                    logger.debug(f"Updated {key}: {old_price} -> {leg.current_price}")
                else:
                    logger.debug(f"No valid price for {key} in options_data")
            else:
                logger.debug(f"Key {key} not found in options_data (available: {list(options_data.keys())[:5]}...)")
        
        logger.debug(f"Strategy price update: {updated_count}/{len(self.legs)} legs updated")
        
        self._calculate_current_pnl()
    
    def _calculate_current_pnl(self):
        """Calculate current P&L"""
        current_value = 0.0
        
        for leg in self.legs:
            leg_value = leg.current_price * leg.quantity * 100
            if leg.position_side == PositionSide.SHORT:
                leg_value = -leg_value
            current_value += leg_value
        
        # P&L = current value - initial value
        initial_value = -self.entry_credit if self.entry_credit > 0 else -self.entry_debit
        self.current_pnl = current_value - initial_value
    
    def close_position(self, exit_date: datetime, exit_reason: str = "Manual"):
        """Close the strategy position"""
        self.is_closed = True
        self.exit_date = exit_date
        self.exit_reason = exit_reason
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get strategy summary"""
        return {
            'strategy_name': self.strategy_name,
            'entry_date': self.entry_date,
            'exit_date': self.exit_date,
            'underlying_price': self.underlying_price,
            'entry_credit': self.entry_credit,
            'entry_debit': self.entry_debit,
            'current_pnl': self.current_pnl,
            'max_profit': self.max_profit,
            'max_loss': self.max_loss,
            'is_closed': self.is_closed,
            'exit_reason': self.exit_reason,
            'num_legs': len(self.legs)
        }

class IronCondor(OptionsStrategy):
    """Iron Condor options strategy"""
    
    def __init__(self, entry_date: datetime, underlying_price: float,
                 put_short_strike: float, put_long_strike: float,
                 call_short_strike: float, call_long_strike: float,
                 quantity: int = 1, expiration: datetime = None,
                 options_data: Dict = None, commission: float = 0.65):
        
        super().__init__("Iron Condor", entry_date, underlying_price, commission)
        
        self.put_short_strike = put_short_strike
        self.put_long_strike = put_long_strike
        self.call_short_strike = call_short_strike
        self.call_long_strike = call_long_strike
        self.quantity = quantity
        self.expiration = expiration or entry_date  # 0DTE
        
        if options_data:
            self._create_legs_from_data(options_data)
        
        self._calculate_iron_condor_metrics()
    
    def _create_legs_from_data(self, options_data: Dict):
        """Create option legs from market data ensuring 4 different strikes"""
        strikes_and_types = [
            (self.put_long_strike, OptionType.PUT, PositionSide.LONG),
            (self.put_short_strike, OptionType.PUT, PositionSide.SHORT),
            (self.call_short_strike, OptionType.CALL, PositionSide.SHORT),
            (self.call_long_strike, OptionType.CALL, PositionSide.LONG)
        ]
        
        # Get all available strikes for both puts and calls
        put_strikes = [float(key.split('_')[0]) for key in options_data.keys() if key.endswith('_put')]
        call_strikes = [float(key.split('_')[0]) for key in options_data.keys() if key.endswith('_call')]
        
        if not put_strikes or not call_strikes:
            logger.warning(f"Insufficient option data - Puts: {len(put_strikes)}, Calls: {len(call_strikes)}")
            return
        
        # Sort strikes to help with selection
        put_strikes = sorted(set(put_strikes))
        call_strikes = sorted(set(call_strikes))
        
        # Find 4 different strikes for Iron Condor, ensuring proper ordering
        used_strikes = set()
        actual_legs = []
        
        for target_strike, option_type, position_side in strikes_and_types:
            available_strikes = put_strikes if option_type == OptionType.PUT else call_strikes
            
            # Find the best available strike that hasn't been used and maintains Iron Condor structure
            best_strike = None
            best_distance = float('inf')
            
            for candidate_strike in available_strikes:
                # Skip if already used (prevents duplicate strikes)
                if candidate_strike in used_strikes:
                    continue
                
                # Check if this strike maintains proper Iron Condor ordering
                if self._is_valid_iron_condor_strike(candidate_strike, option_type, position_side, actual_legs):
                    distance = abs(candidate_strike - target_strike)
                    if distance < best_distance:
                        best_distance = distance
                        best_strike = candidate_strike
            
            if best_strike is None:
                logger.warning(f"Cannot find suitable {option_type.value} strike for {position_side.value} leg")
                continue
                
            # Mark strike as used and create the leg
            used_strikes.add(best_strike)
            key = f"{best_strike}_{option_type.value}"
            
            if key in options_data:
                option_info = options_data[key]
                leg = OptionLeg(
                    strike=best_strike,
                    option_type=option_type,
                    position_side=position_side,
                    quantity=self.quantity,
                    entry_price=option_info.get('mid_price', 0.0),
                    current_price=option_info.get('mid_price', 0.0),
                    expiration=self.expiration,
                    delta=option_info.get('delta', 0.0),
                    gamma=option_info.get('gamma', 0.0),
                    theta=option_info.get('theta', 0.0),
                    vega=option_info.get('vega', 0.0),
                    iv=option_info.get('iv', 0.0)
                )
                self.add_leg(leg)
                actual_legs.append((best_strike, option_type, position_side))
                logger.debug(f"Added {option_type.value} {position_side.value} leg at strike {best_strike}")
            else:
                logger.warning(f"Option data not found for {key}")
        
        # Validate that we have a proper Iron Condor
        if len(self.legs) != 4:
            logger.warning(f"Incomplete Iron Condor: only {len(self.legs)} legs created")
        else:
            # Log the final strike structure
            put_legs = [leg for leg in self.legs if leg.option_type == OptionType.PUT]
            call_legs = [leg for leg in self.legs if leg.option_type == OptionType.CALL]
            
            if len(put_legs) == 2 and len(call_legs) == 2:
                put_strikes_final = sorted([leg.strike for leg in put_legs])
                call_strikes_final = sorted([leg.strike for leg in call_legs])
                logger.info(f"Iron Condor created: Put spread {put_strikes_final[0]}/{put_strikes_final[1]}, Call spread {call_strikes_final[0]}/{call_strikes_final[1]}")
            else:
                logger.warning("Invalid Iron Condor leg distribution")
    
    def _is_valid_iron_condor_strike(self, candidate_strike: float, option_type: OptionType, 
                                   position_side: PositionSide, existing_legs: list) -> bool:
        """
        Validate if a candidate strike maintains proper Iron Condor structure
        
        Iron Condor structure:
        - Put Long (lowest strike) < Put Short < Call Short < Call Long (highest strike)
        """
        if not existing_legs:
            return True  # First leg, any strike is valid
        
        # Get existing strikes by type and position
        existing_puts = [(strike, pos) for strike, opt_type, pos in existing_legs if opt_type == OptionType.PUT]
        existing_calls = [(strike, pos) for strike, opt_type, pos in existing_legs if opt_type == OptionType.CALL]
        
        if option_type == OptionType.PUT:
            if position_side == PositionSide.LONG:
                # Put long should be the lowest strike overall and lower than any existing put short
                for existing_strike, existing_pos in existing_puts:
                    if existing_pos == PositionSide.SHORT and candidate_strike >= existing_strike:
                        return False
                # Should be lower than any existing call strike
                for existing_strike, _ in existing_calls:
                    if candidate_strike >= existing_strike:
                        return False
            else:  # PositionSide.SHORT
                # Put short should be higher than any existing put long but lower than any call strike
                for existing_strike, existing_pos in existing_puts:
                    if existing_pos == PositionSide.LONG and candidate_strike <= existing_strike:
                        return False
                for existing_strike, _ in existing_calls:
                    if candidate_strike >= existing_strike:
                        return False
        else:  # OptionType.CALL
            if position_side == PositionSide.SHORT:
                # Call short should be higher than any put strike but lower than any existing call long
                for existing_strike, _ in existing_puts:
                    if candidate_strike <= existing_strike:
                        return False
                for existing_strike, existing_pos in existing_calls:
                    if existing_pos == PositionSide.LONG and candidate_strike >= existing_strike:
                        return False
            else:  # PositionSide.LONG
                # Call long should be the highest strike overall
                for existing_strike, _ in existing_puts:
                    if candidate_strike <= existing_strike:
                        return False
                for existing_strike, existing_pos in existing_calls:
                    if existing_pos == PositionSide.SHORT and candidate_strike <= existing_strike:
                        return False
        
        return True
    
    def _calculate_iron_condor_metrics(self):
        """Calculate Iron Condor specific metrics"""
        if len(self.legs) < 4:
            logger.warning("Incomplete Iron Condor - need 4 legs")
            return
            
        # Get actual strikes from legs
        put_strikes = [leg.strike for leg in self.legs if leg.option_type == OptionType.PUT]
        call_strikes = [leg.strike for leg in self.legs if leg.option_type == OptionType.CALL]
        
        if len(put_strikes) != 2 or len(call_strikes) != 2:
            logger.warning("Invalid Iron Condor structure")
            return
            
        put_long_strike = min(put_strikes)
        put_short_strike = max(put_strikes)
        call_short_strike = min(call_strikes)
        call_long_strike = max(call_strikes)
        
        put_spread_width = put_short_strike - put_long_strike
        call_spread_width = call_long_strike - call_short_strike
        
        # Max profit = net credit received
        if self.entry_credit > 0:
            self.max_profit = self.entry_credit
        
        # Max loss = spread width - net credit
        max_spread_width = max(put_spread_width, call_spread_width)
        if max_spread_width > 0 and self.quantity is not None:
            self.max_loss = (max_spread_width * 100 * self.quantity)
            if self.entry_credit > 0:
                self.max_loss -= self.entry_credit
        
        # Breakeven points
        if self.entry_credit > 0 and self.quantity is not None and self.quantity > 0:
            credit_per_contract = self.entry_credit / (100 * self.quantity)
            self.breakeven_points = [
                put_short_strike - credit_per_contract,
                call_short_strike + credit_per_contract
            ]
    
    def get_profit_at_expiration(self, underlying_price: float) -> float:
        """Calculate profit/loss at expiration for given underlying price"""
        if len(self.legs) < 4 or self.quantity is None or self.quantity <= 0:
            return 0.0
            
        # Get actual strikes from legs
        put_strikes = [leg.strike for leg in self.legs if leg.option_type == OptionType.PUT]
        call_strikes = [leg.strike for leg in self.legs if leg.option_type == OptionType.CALL]
        
        if len(put_strikes) != 2 or len(call_strikes) != 2:
            return 0.0
            
        put_long_strike = min(put_strikes)
        put_short_strike = max(put_strikes)
        call_short_strike = min(call_strikes)
        call_long_strike = max(call_strikes)
        
        put_spread_pnl = 0.0
        call_spread_pnl = 0.0
        
        # Put spread P&L
        if underlying_price <= put_long_strike:
            put_spread_pnl = -(put_short_strike - put_long_strike) * 100 * self.quantity
        elif underlying_price > put_short_strike:
            put_spread_pnl = 0.0
        else:
            put_spread_pnl = -(underlying_price - put_short_strike) * 100 * self.quantity
        
        # Call spread P&L
        if underlying_price >= call_long_strike:
            call_spread_pnl = -(call_long_strike - call_short_strike) * 100 * self.quantity
        elif underlying_price < call_short_strike:
            call_spread_pnl = 0.0
        else:
            call_spread_pnl = -(underlying_price - call_short_strike) * 100 * self.quantity
        
        total_pnl = put_spread_pnl + call_spread_pnl + self.entry_credit
        return total_pnl

class VerticalSpread(OptionsStrategy):
    """Vertical spread (Call or Put) strategy"""
    
    def __init__(self, entry_date: datetime, underlying_price: float,
                 short_strike: float, long_strike: float,
                 option_type: OptionType, quantity: int = 1,
                 expiration: datetime = None, options_data: Dict = None,
                 commission: float = 0.65):
        
        strategy_name = f"{option_type.value.title()} Spread"
        super().__init__(strategy_name, entry_date, underlying_price, commission)
        
        self.short_strike = short_strike
        self.long_strike = long_strike
        self.option_type = option_type
        self.quantity = quantity
        self.expiration = expiration or entry_date  # 0DTE
        
        if options_data:
            self._create_legs_from_data(options_data)
        
        self._calculate_spread_metrics()
    
    def _create_legs_from_data(self, options_data: Dict):
        """Create option legs from market data"""
        strikes_and_sides = [
            (self.short_strike, PositionSide.SHORT),
            (self.long_strike, PositionSide.LONG)
        ]
        
        for strike, position_side in strikes_and_sides:
            # Find closest available strike
            available_strikes = [float(key.split('_')[0]) for key in options_data.keys() 
                               if key.endswith(f'_{self.option_type.value}')]
            
            if not available_strikes:
                logger.warning(f"No {self.option_type.value} options available")
                continue
                
            closest_strike = min(available_strikes, key=lambda x: abs(x - strike))
            key = f"{closest_strike}_{self.option_type.value}"
            
            if key in options_data:
                option_info = options_data[key]
                leg = OptionLeg(
                    strike=closest_strike,  # Use actual strike, not target
                    option_type=self.option_type,
                    position_side=position_side,
                    quantity=self.quantity,
                    entry_price=option_info.get('mid_price', 0.0),
                    current_price=option_info.get('mid_price', 0.0),
                    expiration=self.expiration,
                    delta=option_info.get('delta', 0.0),
                    gamma=option_info.get('gamma', 0.0),
                    theta=option_info.get('theta', 0.0),
                    vega=option_info.get('vega', 0.0),
                    iv=option_info.get('iv', 0.0)
                )
                self.add_leg(leg)
            else:
                logger.warning(f"Option data not found for {key}")
    
    def _calculate_spread_metrics(self):
        """Calculate spread specific metrics"""
        spread_width = abs(self.long_strike - self.short_strike)
        
        if self.entry_credit > 0:
            # Credit spread
            self.max_profit = self.entry_credit
            self.max_loss = (spread_width * 100 * self.quantity) - self.entry_credit
            
            if self.option_type == OptionType.PUT:
                # Put credit spread (bearish)
                self.breakeven_points = [self.short_strike - (self.entry_credit / (100 * self.quantity))]
            else:
                # Call credit spread (bearish)
                self.breakeven_points = [self.short_strike + (self.entry_credit / (100 * self.quantity))]
        else:
            # Debit spread
            self.max_loss = self.entry_debit
            self.max_profit = (spread_width * 100 * self.quantity) - self.entry_debit
            
            if self.option_type == OptionType.CALL:
                # Call debit spread (bullish)
                self.breakeven_points = [self.long_strike + (self.entry_debit / (100 * self.quantity))]
            else:
                # Put debit spread (bearish)
                self.breakeven_points = [self.long_strike - (self.entry_debit / (100 * self.quantity))]
    
    def get_profit_at_expiration(self, underlying_price: float) -> float:
        """Calculate profit/loss at expiration for given underlying price"""
        if self.option_type == OptionType.CALL:
            return self._call_spread_pnl(underlying_price)
        else:
            return self._put_spread_pnl(underlying_price)
    
    def _call_spread_pnl(self, underlying_price: float) -> float:
        """Calculate call spread P&L at expiration"""
        if self.long_strike < self.short_strike:
            # Call debit spread (bullish)
            if underlying_price <= self.long_strike:
                return -self.entry_debit
            elif underlying_price >= self.short_strike:
                return (self.short_strike - self.long_strike) * 100 * self.quantity - self.entry_debit
            else:
                return (underlying_price - self.long_strike) * 100 * self.quantity - self.entry_debit
        else:
            # Call credit spread (bearish)
            if underlying_price <= self.short_strike:
                return self.entry_credit
            elif underlying_price >= self.long_strike:
                return self.entry_credit - (self.long_strike - self.short_strike) * 100 * self.quantity
            else:
                return self.entry_credit - (underlying_price - self.short_strike) * 100 * self.quantity
    
    def _put_spread_pnl(self, underlying_price: float) -> float:
        """Calculate put spread P&L at expiration"""
        if self.long_strike > self.short_strike:
            # Put debit spread (bearish)
            if underlying_price >= self.long_strike:
                return -self.entry_debit
            elif underlying_price <= self.short_strike:
                return (self.long_strike - self.short_strike) * 100 * self.quantity - self.entry_debit
            else:
                return (self.long_strike - underlying_price) * 100 * self.quantity - self.entry_debit
        else:
            # Put credit spread (bullish)
            if underlying_price >= self.short_strike:
                return self.entry_credit
            elif underlying_price <= self.long_strike:
                return self.entry_credit - (self.short_strike - self.long_strike) * 100 * self.quantity
            else:
                return self.entry_credit - (self.short_strike - underlying_price) * 100 * self.quantity

class StrategyBuilder:
    """Factory class for building options strategies"""
    
    @staticmethod
    def build_iron_condor(entry_date: datetime, underlying_price: float,
                         options_data: Dict, put_distance: int = 50,
                         call_distance: int = 50, spread_width: int = 25,
                         quantity: int = 1, expiration: datetime = None) -> IronCondor:
        """
        Build Iron Condor strategy with specified parameters
        
        Args:
            entry_date: Entry date
            underlying_price: Current underlying price
            options_data: Available options data
            put_distance: Distance for put short strike from underlying
            call_distance: Distance for call short strike from underlying
            spread_width: Width of each spread
            quantity: Number of contracts
            expiration: Expiration date
            
        Returns:
            IronCondor strategy object
        """
        put_short_strike = underlying_price - put_distance
        put_long_strike = put_short_strike - spread_width
        call_short_strike = underlying_price + call_distance
        call_long_strike = call_short_strike + spread_width
        
        return IronCondor(
            entry_date=entry_date,
            underlying_price=underlying_price,
            put_short_strike=put_short_strike,
            put_long_strike=put_long_strike,
            call_short_strike=call_short_strike,
            call_long_strike=call_long_strike,
            quantity=quantity,
            expiration=expiration,
            options_data=options_data
        )
    
    @staticmethod
    def build_call_spread(entry_date: datetime, underlying_price: float,
                         options_data: Dict, strike_distance: int = 25,
                         spread_width: int = 25, is_debit: bool = True,
                         quantity: int = 1, expiration: datetime = None) -> VerticalSpread:
        """Build Call Spread strategy"""
        if is_debit:
            # Call debit spread (bullish)
            long_strike = underlying_price + strike_distance
            short_strike = long_strike + spread_width
        else:
            # Call credit spread (bearish)
            short_strike = underlying_price + strike_distance
            long_strike = short_strike + spread_width
        
        return VerticalSpread(
            entry_date=entry_date,
            underlying_price=underlying_price,
            short_strike=short_strike,
            long_strike=long_strike,
            option_type=OptionType.CALL,
            quantity=quantity,
            expiration=expiration,
            options_data=options_data
        )
    
    @staticmethod
    def build_put_spread(entry_date: datetime, underlying_price: float,
                        options_data: Dict, strike_distance: int = 25,
                        spread_width: int = 25, is_debit: bool = True,
                        quantity: int = 1, expiration: datetime = None) -> VerticalSpread:
        """Build Put Spread strategy"""
        if is_debit:
            # Put debit spread (bearish)
            long_strike = underlying_price - strike_distance
            short_strike = long_strike - spread_width
        else:
            # Put credit spread (bullish)
            short_strike = underlying_price - strike_distance
            long_strike = short_strike - spread_width
        
        return VerticalSpread(
            entry_date=entry_date,
            underlying_price=underlying_price,
            short_strike=short_strike,
            long_strike=long_strike,
            option_type=OptionType.PUT,
            quantity=quantity,
            expiration=expiration,
            options_data=options_data
        )