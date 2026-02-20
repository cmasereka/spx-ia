import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from loguru import logger
from config.settings import MACD_PARAMS, RSI_PARAMS, BOLLINGER_PARAMS

class TechnicalIndicators:
    """Calculate technical indicators for SPX trading strategies"""
    
    def __init__(self):
        self.macd_params = MACD_PARAMS
        self.rsi_params = RSI_PARAMS
        self.bb_params = BOLLINGER_PARAMS
    
    def calculate_macd(self, prices: pd.Series, 
                      fast_period: int = None, 
                      slow_period: int = None, 
                      signal_period: int = None) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Price series (typically close prices)
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            Dictionary with MACD, signal line, and histogram
        """
        fast_period = fast_period or self.macd_params['fast_period']
        slow_period = slow_period or self.macd_params['slow_period']
        signal_period = signal_period or self.macd_params['signal_period']
        
        try:
            # Calculate EMAs
            ema_fast = prices.ewm(span=fast_period).mean()
            ema_slow = prices.ewm(span=slow_period).mean()
            
            # MACD line
            macd_line = ema_fast - ema_slow
            
            # Signal line
            signal_line = macd_line.ewm(span=signal_period).mean()
            
            # Histogram
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'macd_signal': signal_line,
                'macd_histogram': histogram
            }
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {}
    
    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index)
        
        Args:
            prices: Price series
            period: RSI calculation period
            
        Returns:
            RSI series
        """
        period = period or self.rsi_params['period']
        
        try:
            delta = prices.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
            
            # Calculate RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series()
    
    def calculate_bollinger_bands(self, prices: pd.Series, 
                                 period: int = None, 
                                 std_dev: float = None) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            Dictionary with upper, middle, lower bands and additional metrics
        """
        period = period or self.bb_params['period']
        std_dev = std_dev or self.bb_params['std_dev']
        
        try:
            # Calculate middle band (SMA)
            middle_band = prices.rolling(window=period).mean()
            
            # Calculate standard deviation
            rolling_std = prices.rolling(window=period).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (rolling_std * std_dev)
            lower_band = middle_band - (rolling_std * std_dev)
            
            # Additional metrics
            bb_width = (upper_band - lower_band) / middle_band * 100
            bb_position = (prices - lower_band) / (upper_band - lower_band)
            
            return {
                'bb_upper': upper_band,
                'bb_middle': middle_band,
                'bb_lower': lower_band,
                'bb_width': bb_width,
                'bb_position': bb_position
            }
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {}
    
    def calculate_all_indicators(self, df: pd.DataFrame, 
                               price_column: str = 'close') -> pd.DataFrame:
        """
        Calculate all technical indicators for a DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            price_column: Column name for price data
            
        Returns:
            DataFrame with all indicators added
        """
        result_df = df.copy()
        
        if price_column not in df.columns:
            logger.error(f"Price column '{price_column}' not found in DataFrame")
            return result_df
        
        prices = df[price_column]
        
        try:
            # Calculate MACD
            macd_data = self.calculate_macd(prices)
            for key, series in macd_data.items():
                result_df[key] = series
            
            # Calculate RSI
            result_df['rsi'] = self.calculate_rsi(prices)
            
            # Calculate Bollinger Bands
            bb_data = self.calculate_bollinger_bands(prices)
            for key, series in bb_data.items():
                result_df[key] = series
            
            logger.info("All technical indicators calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return result_df
    
    def get_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            DataFrame with signal columns added
        """
        signals_df = df.copy()
        
        try:
            # MACD signals
            signals_df['macd_bullish'] = (
                (df['macd'] > df['macd_signal']) & 
                (df['macd'].shift(1) <= df['macd_signal'].shift(1))
            )
            signals_df['macd_bearish'] = (
                (df['macd'] < df['macd_signal']) & 
                (df['macd'].shift(1) >= df['macd_signal'].shift(1))
            )
            
            # RSI signals
            signals_df['rsi_oversold'] = df['rsi'] < self.rsi_params['oversold']
            signals_df['rsi_overbought'] = df['rsi'] > self.rsi_params['overbought']
            
            # Bollinger Bands signals
            signals_df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.8
            signals_df['bb_upper_touch'] = df['close'] >= df['bb_upper']
            signals_df['bb_lower_touch'] = df['close'] <= df['bb_lower']
            
            # Composite signals for options strategies
            signals_df['bullish_signal'] = (
                signals_df['macd_bullish'] | 
                (signals_df['rsi_oversold'] & (df['rsi'] > df['rsi'].shift(1))) |
                signals_df['bb_lower_touch']
            )
            
            signals_df['bearish_signal'] = (
                signals_df['macd_bearish'] | 
                (signals_df['rsi_overbought'] & (df['rsi'] < df['rsi'].shift(1))) |
                signals_df['bb_upper_touch']
            )
            
            # Neutral/range-bound signals (good for Iron Condor)
            signals_df['neutral_signal'] = (
                (~signals_df['bullish_signal']) & 
                (~signals_df['bearish_signal']) &
                (df['rsi'] > 40) & (df['rsi'] < 60) &
                (df['bb_position'] > 0.2) & (df['bb_position'] < 0.8)
            )
            
            logger.info("Trading signals generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
        
        return signals_df
    
    def calculate_momentum_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum score for position sizing
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Momentum score series (0-100)
        """
        try:
            # Normalize indicators to 0-100 scale
            macd_norm = self._normalize_series(df['macd_histogram'], 0, 100)
            rsi_norm = df['rsi']
            bb_norm = df['bb_position'] * 100
            
            # Weighted momentum score
            momentum_score = (
                macd_norm * 0.4 +  # MACD weight: 40%
                rsi_norm * 0.4 +   # RSI weight: 40%
                bb_norm * 0.2      # BB position weight: 20%
            )
            
            return momentum_score.clip(0, 100)
            
        except Exception as e:
            logger.error(f"Error calculating momentum score: {e}")
            return pd.Series()
    
    def _normalize_series(self, series: pd.Series, 
                         target_min: float = 0, 
                         target_max: float = 100) -> pd.Series:
        """
        Normalize series to target range
        
        Args:
            series: Input series
            target_min: Target minimum value
            target_max: Target maximum value
            
        Returns:
            Normalized series
        """
        series_min = series.min()
        series_max = series.max()
        
        if series_max == series_min:
            return pd.Series([target_min] * len(series), index=series.index)
        
        normalized = (series - series_min) / (series_max - series_min)
        scaled = normalized * (target_max - target_min) + target_min
        
        return scaled
    
    def get_indicator_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of technical indicators
        
        Args:
            df: DataFrame with indicators
            
        Returns:
            Summary dictionary
        """
        summary = {}
        
        try:
            indicator_columns = [
                'macd', 'macd_signal', 'macd_histogram', 'rsi',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position'
            ]
            
            for col in indicator_columns:
                if col in df.columns:
                    summary[col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'current': df[col].iloc[-1] if len(df) > 0 else None
                    }
            
            # Signal counts
            signal_columns = [col for col in df.columns if '_signal' in col or 'bullish' in col or 'bearish' in col]
            for col in signal_columns:
                if col in df.columns:
                    summary[f"{col}_count"] = df[col].sum()
                    summary[f"{col}_pct"] = (df[col].sum() / len(df)) * 100
            
        except Exception as e:
            logger.error(f"Error generating indicator summary: {e}")
        
        return summary