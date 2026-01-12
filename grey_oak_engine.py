"""
GreyOak SlopeTrigger v4.1 Engine
For Streamlit Cloud Deployment
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

class Config:
    """Configuration class"""
    BASE_DIR = Path("./greyoak_v4_production")
    OUTPUT_DIR = BASE_DIR.parent / "compressed_outputs"
    
    # Create directories
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Subdirectories matching original structure
    SUBDIRS = [
        "daily_signals",
        "summary", 
        "debug",
        "strong_signals",
        "top_signals"
    ]
    
    # Create all subdirectories
    for subdir in SUBDIRS:
        (BASE_DIR / subdir).mkdir(parents=True, exist_ok=True)

class GreyOakProcessor:
    """Main processor for GreyOak signals"""
    
    def __init__(self):
        self.cache = {}
        self.failed_symbols = []
        self.stats = {
            'start_time': None,
            'end_time': None,
            'processed': 0,
            'total': 0
        }
    
    def analyze_universe(self, symbols, max_workers=4):
        """Analyze list of symbols"""
        self.stats['start_time'] = datetime.now()
        self.stats['total'] = len(symbols)
        
        all_signals = []
        
        # Process in batches for better performance
        batch_size = min(20, max_workers * 2)
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_signals = self._process_batch(batch, max_workers)
            all_signals.extend(batch_signals)
        
        self.stats['end_time'] = datetime.now()
        self.stats['processed'] = len(all_signals)
        
        # Create DataFrame
        signals_df = pd.DataFrame(all_signals)
        
        if signals_df.empty:
            return None
        
        # Calculate summary
        summary = self._calculate_summary(signals_df)
        
        return {
            'signals_df': signals_df,
            'summary': summary,
            'failed_symbols': self.failed_symbols
        }
    
    def _process_batch(self, symbols, max_workers):
        """Process a batch of symbols"""
        signals = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._process_symbol, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        signals.append(result)
                except Exception as e:
                    self.failed_symbols.append(symbol)
        
        return signals
    
    def _process_symbol(self, symbol):
        """Process a single symbol"""
        try:
            # Fetch data
            df = self._fetch_data(symbol)
            if df is None or df.empty:
                return None
            
            # Calculate indicators
            df = self._calculate_indicators(df)
            
            # Calculate score
            score_result = self._calculate_score(df)
            
            # Prepare signal
            signal_data = {
                'symbol': symbol,
                'score': score_result['score'],
                'signal': score_result['signal'],
                'price': df['Close'].iloc[-1],
                'rsi': score_result.get('rsi', 0),
                'macd': score_result.get('macd', 0),
                'volume': df['Volume'].iloc[-1],
                'timestamp': datetime.now().isoformat()
            }
            
            return signal_data
            
        except Exception as e:
            raise Exception(f"Error processing {symbol}: {str(e)}")
    
    def _fetch_data(self, symbol, days=30):
        """Fetch market data"""
        cache_key = f"{symbol}_{days}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d")
            
            if df.empty or len(df) < 10:
                return None
            
            # Cache data
            self.cache[cache_key] = df
            return df
            
        except:
            return None
    
    def _calculate_indicators(self, df):
        """Calculate technical indicators"""
        if len(df) < 10:
            return df
        
        # Simple RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Simple MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        
        return df
    
    def _calculate_score(self, df):
        """Calculate GreyOak score"""
        if df.empty or len(df) < 2:
            return {"score": 0, "signal": "NEUTRAL"}
        
        latest = df.iloc[-1]
        
        # Simple scoring based on RSI
        rsi = latest['RSI'] if pd.notna(latest['RSI']) else 50
        macd = latest['MACD'] if pd.notna(latest['MACD']) else 0
        
        # Base score from RSI
        if rsi < 30:
            base_score = 20
        elif rsi > 70:
            base_score = -20
        else:
            base_score = (50 - rsi) * 0.4  # Convert to -8 to +8 range
        
        # Adjust with MACD
        if macd > 0:
            macd_factor = 1 + (macd / latest['Close']) * 100
        else:
            macd_factor = 1 - abs(macd / latest['Close']) * 100
        
        final_score = base_score * macd_factor
        
        # Determine signal
        if final_score > 15:
            signal = "STRONG_GREEN"
        elif final_score < -15:
            signal = "STRONG_RED"
        else:
            signal = "NEUTRAL"
        
        return {
            "score": round(final_score, 2),
            "signal": signal,
            "rsi": round(rsi, 2),
            "macd": round(macd, 4)
        }
    
    def _calculate_summary(self, signals_df):
        """Calculate summary statistics"""
        total = len(signals_df)
        strong_green = len(signals_df[signals_df['signal'] == 'STRONG_GREEN'])
        strong_red = len(signals_df[signals_df['signal'] == 'STRONG_RED'])
        
        processing_time = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        success_rate = (total / self.stats['total']) * 100 if self.stats['total'] > 0 else 0
        
        return {
            'total_signals': total,
            'strong_green': strong_green,
            'strong_red': strong_red,
            'neutral': total - strong_green - strong_red,
            'success_rate': round(success_rate, 1),
            'processing_time': round(processing_time, 2),
            'failed_count': len(self.failed_symbols),
            'date': datetime.now().strftime("%Y-%m-%d")
        }
    
    def get_sample_data(self, count=50):
        """Get sample data for demo"""
        symbols = [f"STOCK{i:03d}.NS" for i in range(1, count + 1)]
        
        signals = []
        for symbol in symbols:
            score = np.random.uniform(-25, 25)
            
            if score > 15:
                signal = "STRONG_GREEN"
            elif score < -15:
                signal = "STRONG_RED"
            else:
                signal = "NEUTRAL"
            
            signals.append({
                "symbol": symbol,
                "score": round(score, 2),
                "signal": signal,
                "price": round(np.random.uniform(50, 5000), 2),
                "rsi": round(np.random.uniform(20, 80), 2),
                "macd": round(np.random.uniform(-5, 5), 3),
                "timestamp": datetime.now().isoformat()
            })
        
        return pd.DataFrame(signals)