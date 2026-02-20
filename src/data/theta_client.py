import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import requests
import time
from loguru import logger
import base64
import json

class ThetaDataClient:
    """Client for downloading SPX options data from ThetaData REST API"""
    
    def __init__(self, username: str, password: str, port: int = 25503, symbol: str = "SPXW"):
        self.username = username
        self.password = password
        self.symbol = symbol  # SPXW for 0DTE, SPX for monthly
        self.base_url = f"http://127.0.0.1:{port}"  # ThetaData Terminal port
        self.session = requests.Session()
        self.connected = False
        self._setup_auth()
    
    def _setup_auth(self):
        """Setup authentication for API requests"""
        try:
            # Create basic auth header
            credentials = f"{self.username}:{self.password}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            self.session.headers.update({
                'Authorization': f'Basic {encoded_credentials}',
                'Content-Type': 'application/json'
            })
            logger.info("ThetaData API authentication configured")
        except Exception as e:
            logger.error(f"Failed to setup authentication: {e}")
    
    def _connect(self):
        """Test connection to ThetaData Terminal"""
        try:
            logger.info(f"Attempting to connect to ThetaData Terminal at {self.base_url}")
            
            # Test with a simple request using v2 API (since v3 endpoints are 404)
            # But use the updated parameter names
            response = self.session.get(f"{self.base_url}/v2/list/expirations", 
                                      params={'symbol': self.symbol}, timeout=5)
            
            if response.status_code == 200:
                self.connected = True
                logger.info("Connected to ThetaData Terminal successfully")
                return True
            elif response.status_code == 404:
                # Try alternative endpoint
                response = self.session.get(f"{self.base_url}/v2/hist/stock/ohlc", 
                                          params={'symbol': self.symbol, 'start_date': '20240101', 'end_date': '20240102'}, timeout=5)
                if response.status_code == 200:
                    self.connected = True
                    logger.info(f"Connected to ThetaData Terminal successfully using {self.symbol}")
                    return True
                else:
                    logger.error(f"ThetaData API endpoints not responding correctly. Status: {response.status_code}")
                    logger.info(f"Tried symbol: {self.symbol}. For 0DTE use SPXW, for monthly use SPX")
                    return False
            else:
                logger.error(f"Failed to connect: HTTP {response.status_code}")
                return False
        except requests.exceptions.ConnectionError as e:
            if "Connection refused" in str(e) or "No connection could be made" in str(e):
                logger.error("ThetaData Terminal not running. Please start ThetaData Terminal first.")
                logger.info("Download from: https://thetadata.com")
            else:
                logger.error(f"Connection error: {e}")
            return False
        except requests.exceptions.Timeout:
            logger.error("Connection timeout. ThetaData Terminal may not be responding.")
            return False
        except Exception as e:
            logger.error(f"Unexpected connection error: {e}")
            return False
    
    def connect(self):
        """Public method to connect"""
        return self._connect()
    
    def disconnect(self):
        """Disconnect from ThetaData API"""
        if self.session:
            self.session.close()
            self.connected = False
            logger.info("Disconnected from ThetaData API")
    
    def get_spx_underlying_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get SPX underlying price data using REST API v3
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with SPX price data
        """
        if not self.connected:
            if not self._connect():
                logger.error("Cannot fetch data - not connected to ThetaData Terminal")
                return pd.DataFrame()
        
        try:
            # Format dates for API (YYYYMMDD)
            start_str = start_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')
            
            # API v2 endpoint with updated parameter names
            url = f"{self.base_url}/v2/hist/stock/ohlc"
            params = {
                'symbol': self.symbol,  # SPXW or SPX
                'start_date': start_str,
                'end_date': end_str,
                'ivl': '86400000'  # Daily interval in milliseconds
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return pd.DataFrame()
            
            data = response.json()
            
            if not data.get('response'):
                logger.warning(f"No underlying data found for SPX from {start_date} to {end_date}")
                return pd.DataFrame()
            
            # Convert to DataFrame - v2 format
            records = []
            response_data = data['response']
            
            # v2 API typically returns array format: [timestamp, open, high, low, close, volume]
            if isinstance(response_data, list):
                for item in response_data:
                    if isinstance(item, list) and len(item) >= 5:
                        # Array format: [timestamp, open, high, low, close, volume]
                        timestamp = datetime.fromtimestamp(item[0] / 1000)  # Convert from milliseconds
                        records.append({
                            'date': timestamp,
                            'open': item[1] / 1000,    # ThetaData prices are in millidollars
                            'high': item[2] / 1000,
                            'low': item[3] / 1000,
                            'close': item[4] / 1000,
                            'volume': item[5] if len(item) > 5 else 0
                        })
                    elif isinstance(item, dict):
                        # Dictionary format fallback
                        records.append({
                            'date': pd.to_datetime(item.get('date', item.get('timestamp'))),
                            'open': float(item.get('open', 0)) / 1000,
                            'high': float(item.get('high', 0)) / 1000,
                            'low': float(item.get('low', 0)) / 1000,
                            'close': float(item.get('close', 0)) / 1000,
                            'volume': int(item.get('volume', 0))
                        })
            
            df = pd.DataFrame(records)
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                # Remove any duplicate dates and sort
                df = df.drop_duplicates(subset=['date']).sort_values('date')
                logger.info(f"Retrieved {len(df)} SPX underlying records from {start_date} to {end_date}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving SPX underlying data: {e}")
            return pd.DataFrame()
    
    def get_spx_options_chain(self, date: datetime, dte: int = 0) -> pd.DataFrame:
        """
        Get SPX options chain for a specific date and DTE
        
        Args:
            date: Trading date
            dte: Days to expiration (0 for 0DTE)
            
        Returns:
            DataFrame with options chain data
        """
        if not self.connected:
            if not self._connect():
                logger.error("Cannot fetch data - not connected to ThetaData Terminal")
                return pd.DataFrame()
        
        try:
            # Calculate expiration date
            exp_date = date + timedelta(days=dte)
            
            # Format dates for API
            date_str = date.strftime('%Y%m%d')
            exp_str = exp_date.strftime('%Y%m%d')
            
            # First, get available expirations
            exp_url = f"{self.base_url}/v2/list/expirations"
            exp_params = {'root': 'SPX'}
            exp_response = self.session.get(exp_url, params=exp_params, timeout=10)
            
            if exp_response.status_code != 200:
                logger.error(f"Failed to get expirations: {exp_response.status_code}")
                return pd.DataFrame()
            
            expirations = exp_response.json().get('response', [])
            
            # Find closest expiration to our target
            target_exp_timestamp = int(exp_date.timestamp() * 1000)
            closest_exp = min(expirations, key=lambda x: abs(x - target_exp_timestamp))
            closest_exp_date = datetime.fromtimestamp(closest_exp / 1000)
            
            # Get options chain for this expiration
            chain_url = f"{self.base_url}/v2/hist/option/ohlc"
            chain_params = {
                'root': 'SPX',
                'exp': closest_exp_date.strftime('%Y%m%d'),
                'start_date': date_str,
                'end_date': date_str,
                'ivl': '86400000'
            }
            
            response = self.session.get(chain_url, params=chain_params, timeout=60)
            
            if response.status_code != 200:
                logger.error(f"Options chain request failed: {response.status_code} - {response.text}")
                return pd.DataFrame()
            
            data = response.json()
            
            if not data.get('response'):
                logger.warning(f"No options data found for SPX on {date} with {dte} DTE")
                return pd.DataFrame()
            
            # Process options data
            records = []
            for item in data['response']:
                # Parse the option identifier
                # Format: .SPX240119C04700000 or similar
                contract = item.get('contract', '')
                if not contract:
                    continue
                
                # Extract strike and option type from contract
                try:
                    # Find 'C' or 'P' in the contract string
                    if 'C' in contract and 'P' in contract:
                        # Find which comes first
                        c_pos = contract.find('C')
                        p_pos = contract.find('P')
                        
                        if c_pos != -1 and (p_pos == -1 or c_pos < p_pos):
                            option_type = 'call'
                            strike_str = contract[c_pos+1:]
                        else:
                            option_type = 'put'
                            strike_str = contract[p_pos+1:]
                    elif 'C' in contract:
                        option_type = 'call'
                        strike_str = contract[contract.find('C')+1:]
                    elif 'P' in contract:
                        option_type = 'put'
                        strike_str = contract[contract.find('P')+1:]
                    else:
                        continue
                    
                    # Extract strike price
                    strike = float(strike_str) / 1000 if strike_str.isdigit() else 0
                    
                    if strike == 0:
                        continue
                    
                    # Extract OHLC data
                    # Format: [timestamp, open, high, low, close, volume]
                    ohlc = item.get('data', [[]])[0] if item.get('data') else []
                    
                    if len(ohlc) < 5:
                        continue
                    
                    timestamp = datetime.fromtimestamp(ohlc[0] / 1000)
                    
                    records.append({
                        'date': timestamp,
                        'expiration': closest_exp_date,
                        'strike': strike,
                        'option_type': option_type,
                        'open': ohlc[1] / 100,    # Convert from cents
                        'high': ohlc[2] / 100,
                        'low': ohlc[3] / 100,
                        'close': ohlc[4] / 100,
                        'volume': ohlc[5] if len(ohlc) > 5 else 0,
                        'bid': ohlc[4] / 100 * 0.95,  # Approximate bid/ask
                        'ask': ohlc[4] / 100 * 1.05,
                        'mid_price': ohlc[4] / 100,
                        'delta': 0.0,
                        'gamma': 0.0,
                        'theta': 0.0,
                        'vega': 0.0,
                        'iv': 0.0
                    })
                    
                except (ValueError, IndexError) as e:
                    logger.debug(f"Error parsing contract {contract}: {e}")
                    continue
            
            df = pd.DataFrame(records)
            
            if not df.empty:
                # Add calculated fields
                df['bid_ask_spread'] = df['ask'] - df['bid']
                df['spread_pct'] = (df['bid_ask_spread'] / df['mid_price']) * 100
                
                logger.info(f"Retrieved {len(df)} SPX options contracts for {date}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving SPX options chain: {e}")
            return pd.DataFrame()
    
    def get_options_data_range(self, start_date: datetime, end_date: datetime, dte: int = 0) -> List[pd.DataFrame]:
        """
        Get options chain data for a date range
        
        Args:
            start_date: Start date
            end_date: End date
            dte: Days to expiration
            
        Returns:
            List of DataFrames, one for each trading day
        """
        data_list = []
        current_date = start_date
        
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                df = self.get_spx_options_chain(current_date, dte)
                if not df.empty:
                    data_list.append(df)
                
                # Rate limiting to avoid API limits
                time.sleep(1)
            
            current_date += timedelta(days=1)
        
        logger.info(f"Retrieved options data for {len(data_list)} trading days")
        return data_list
    
    def get_iron_condor_data(self, date: datetime, underlying_price: float, 
                           put_distance: int = 50, call_distance: int = 50) -> Dict[str, Any]:
        """
        Get specific options data for Iron Condor strategy
        
        Args:
            date: Trading date
            underlying_price: Current SPX price
            put_distance: Distance for put spread strikes
            call_distance: Distance for call spread strikes
            
        Returns:
            Dictionary with Iron Condor leg data
        """
        options_chain = self.get_spx_options_chain(date)
        
        if options_chain.empty:
            return {}
        
        # Define strikes for Iron Condor
        put_short_strike = underlying_price - put_distance
        put_long_strike = underlying_price - put_distance - 25
        call_short_strike = underlying_price + call_distance
        call_long_strike = underlying_price + call_distance + 25
        
        # Find closest strikes in the options chain
        ic_data = {}
        strikes = [put_long_strike, put_short_strike, call_short_strike, call_long_strike]
        names = ['put_long', 'put_short', 'call_short', 'call_long']
        option_types = ['put', 'put', 'call', 'call']
        
        for strike, name, opt_type in zip(strikes, names, option_types):
            closest_option = self._find_closest_option(options_chain, strike, opt_type)
            if closest_option is not None:
                ic_data[name] = closest_option
        
        return ic_data
    
    def _find_closest_option(self, options_df: pd.DataFrame, target_strike: float, 
                           option_type: str) -> Optional[Dict]:
        """
        Find the closest option to target strike
        
        Args:
            options_df: Options chain DataFrame
            target_strike: Target strike price
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with option data or None
        """
        filtered_options = options_df[options_df['option_type'] == option_type].copy()
        
        if filtered_options.empty:
            return None
        
        filtered_options['strike_diff'] = abs(filtered_options['strike'] - target_strike)
        closest_idx = filtered_options['strike_diff'].idxmin()
        
        return filtered_options.loc[closest_idx].to_dict()