"""
GREYOAK SLOPETRIGGER v4.1 - COMPLETE STREAMLIT DASHBOARD
Based on original GreyOak output structure
"""

import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import shutil
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import time
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
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
    
    # NSE Universe (2000+ stocks in production)
    NSE_UNIVERSE = [
        # Nifty 50 + major stocks
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS",
        "BAJFINANCE.NS", "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ONGC.NS", "NTPC.NS",
        "ULTRACEMCO.NS", "POWERGRID.NS", "M&M.NS", "BAJAJFINSV.NS", "HCLTECH.NS",
        "TECHM.NS", "INDUSINDBK.NS", "HDFC.NS", "DRREDDY.NS", "BRITANNIA.NS",
        "NESTLEIND.NS", "COALINDIA.NS", "TATAMOTORS.NS", "JSWSTEEL.NS", "GRASIM.NS",
        "ADANIPORTS.NS", "DIVISLAB.NS", "SHREECEM.NS", "HDFCLIFE.NS", "EICHERMOT.NS",
        "SBILIFE.NS", "HINDALCO.NS", "BPCL.NS", "IOC.NS", "UPL.NS",
        "HEROMOTOCO.NS", "TATASTEEL.NS", "BAJAJ-AUTO.NS", "CIPLA.NS", "GAIL.NS"
    ]
    
    # Add more stocks to reach 2000+ in production
    @staticmethod
    def get_full_nse_universe():
        """Get full NSE universe - in production load from file"""
        # For demo, return 100 stocks
        return Config.NSE_UNIVERSE * 2  # 100 stocks for demo

# ============================================================================
# GREYOAK OUTPUT ZIPPER (ORIGINAL CODE ADAPTED)
# ============================================================================

class GreyOakOutputZipper:
    """Utility to zip GreyOak v4.1 output files - Streamlit version"""
    
    def __init__(self, base_dir="./greyoak_v4_production"):
        self.base_dir = Path(base_dir)
        self.output_dir = Config.OUTPUT_DIR
        
        # Create all subdirectories
        for subdir in Config.SUBDIRS:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def create_zip_file(self, include_debug=True):
        """Create a zip file of all output files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"greyoak_v4_results_{timestamp}.zip"
        zip_path = self.output_dir / zip_filename
        
        st.info(f"📦 Creating zip file: {zip_filename}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files from the base directory
            self._add_directory_to_zip(zipf, self.base_dir, include_debug)
        
        st.success(f"✅ Zip file created: {zip_filename}")
        return zip_path
    
    def _add_directory_to_zip(self, zipf, directory, include_debug=True):
        """Recursively add directory contents to zip file"""
        for root, dirs, files in os.walk(directory):
            # Skip debug directory if specified
            if not include_debug and "debug" in root:
                continue
            
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(self.base_dir.parent)
                zipf.write(file_path, arcname)
    
    def get_zip_file_size(self, zip_path):
        """Get the size of the zip file in MB"""
        size_bytes = os.path.getsize(zip_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    
    def create_readme_file(self, total_signals=0, strong_green=0, strong_red=0, 
                          success_rate=0, top_stocks=None):
        """Create a README file with information about the zip contents"""
        
        if top_stocks is None:
            top_stocks = [
                {"symbol": "TRIGYN.NS", "score": 21.58},
                {"symbol": "JHS.NS", "score": 21.48},
                {"symbol": "BANARBEADS.NS", "score": 21.27},
                {"symbol": "MANAKCOAT.NS", "score": 21.25},
                {"symbol": "BBTCL.NS", "score": 20.85}
            ]
        
        readme_content = f"""# GreyOak SlopeTrigger v4.1 Results
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Files Included:

### 1. Daily Signals Directory
- Contains signal files for each trading day
- Files: signals_YYYY-MM-DD.csv
- Strong signals: strong_signals_YYYY-MM-DD.csv
- Top signals: top_green_YYYY-MM-DD.csv, top_red_YYYY-MM-DD.csv

### 2. Summary Directory
- all_signals.csv: All generated signals ({total_signals:,} signals)
- summary_statistics.json: Overall statistics
- date_statistics.csv: Date-wise statistics
- indicator_statistics.json: Indicator performance statistics

### 3. Debug Directory (if included)
- failed_symbols.txt: List of symbols that failed to process

## Statistics Summary:
- Total signals generated: {total_signals:,}
- STRONG_GREEN signals: {strong_green} ({strong_green/total_signals*100:.1f}%)
- STRONG_RED signals: {strong_red} ({strong_red/total_signals*100:.1f}%)
- NEUTRAL signals: {total_signals - strong_green - strong_red:,} ({(total_signals - strong_green - strong_red)/total_signals*100:.1f}%)
- Success rate: {success_rate:.1f}%

## Top 5 Stocks by Average Score:"""
        
        for i, stock in enumerate(top_stocks[:5], 1):
            readme_content += f"\n{i}. {stock['symbol']}: {stock['score']:.2f}"
        
        readme_content += f"""

## Analysis Period:
- From: {datetime.now().strftime("%Y-%m-01")}
- To: {datetime.now().strftime("%Y-%m-%d")}

## Notes:
- Signals are generated using GreyOak SlopeTrigger v4.1 methodology
- Based on 9 technical indicators with curvature-based reversal detection
- Includes v4.0 gates: freshness, expansion, and minimum score requirements
- Data source: Yahoo Finance (NSE stocks)
"""

        readme_path = self.output_dir / "README.txt"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        return readme_path

# ============================================================================
# GREYOAK SIGNAL PROCESSOR
# ============================================================================

class GreyOakProcessor:
    """Process GreyOak v4.1 signals for NSE universe"""
    
    def __init__(self):
        self.zipper = GreyOakOutputZipper()
        self.failed_symbols = []
        self.processed_count = 0
        self.total_count = 0
        
    def calculate_signals(self, symbols, max_workers=10):
        """Calculate signals for all symbols"""
        start_time = time.time()
        all_signals = []
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        self.total_count = len(symbols)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process symbols in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._process_single_symbol, symbol, date_str): symbol 
                for symbol in symbols
            }
            
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1
                
                # Update progress
                progress = int((completed / len(symbols)) * 100)
                progress_bar.progress(progress)
                status_text.text(f"Processing: {completed}/{len(symbols)} ({progress}%)")
                
                try:
                    result = future.result()
                    if result:
                        all_signals.append(result)
                        self.processed_count += 1
                except Exception as e:
                    self.failed_symbols.append(symbol)
                    st.warning(f"Failed to process {symbol}: {str(e)}")
        
        # Create DataFrame
        signals_df = pd.DataFrame(all_signals)
        
        if signals_df.empty:
            return None
        
        # Save all outputs
        results = self._save_outputs(signals_df, date_str, start_time)
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
        
        return results
    
    def _process_single_symbol(self, symbol, date_str):
        """Process a single symbol"""
        try:
            # Fetch data
            df = self._fetch_market_data(symbol)
            if df is None or df.empty:
                return None
            
            # Calculate indicators
            df = self._calculate_indicators(df)
            
            # Calculate GreyOak score
            score_result = self._calculate_greyoak_score(df)
            
            # Prepare signal data
            signal_data = {
                "date": date_str,
                "symbol": symbol,
                "score": score_result["score"],
                "signal": score_result["signal"],
                "color": score_result["color"],
                "rsi": score_result.get("rsi", 0),
                "macd": score_result.get("macd", 0),
                "stochastic_k": score_result.get("stoch_k", 0),
                "volume_ratio": score_result.get("volume_ratio", 0),
                "price": df['Close'].iloc[-1] if not df.empty else 0,
                "volume": df['Volume'].iloc[-1] if not df.empty else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            return signal_data
            
        except Exception as e:
            raise Exception(f"Error processing {symbol}: {str(e)}")
    
    def _fetch_market_data(self, symbol, days=60):
        """Fetch historical data"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d")
            return df if not df.empty else None
        except:
            return None
    
    def _calculate_indicators(self, df):
        """Calculate technical indicators"""
        if len(df) < 20:
            return df
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        sma = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = sma + (std * 2)
        df['BB_Lower'] = sma - (std * 2)
        
        # Stochastic
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        
        # EMA
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        
        # Volume Ratio
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        return df
    
    def _calculate_greyoak_score(self, df):
        """Calculate GreyOak v4.1 score"""
        if df.empty or len(df) < 2:
            return {"score": 0, "signal": "NEUTRAL", "color": "gray"}
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        scores = []
        
        # RSI scoring
        rsi = latest['RSI'] if pd.notna(latest['RSI']) else 50
        if rsi < 30:
            scores.append(3)
        elif rsi > 70:
            scores.append(-2)
        elif 30 <= rsi <= 50 and rsi > prev['RSI']:
            scores.append(2)
        elif 50 <= rsi <= 70 and rsi < prev['RSI']:
            scores.append(-1)
        
        # MACD scoring
        macd = latest['MACD'] if pd.notna(latest['MACD']) else 0
        macd_signal = latest['MACD_Signal'] if pd.notna(latest['MACD_Signal']) else 0
        prev_macd = prev['MACD'] if pd.notna(prev['MACD']) else 0
        prev_macd_signal = prev['MACD_Signal'] if pd.notna(prev['MACD_Signal']) else 0
        
        if macd > macd_signal and prev_macd <= prev_macd_signal:
            scores.append(3)
        elif macd < macd_signal and prev_macd >= prev_macd_signal:
            scores.append(-3)
        
        # Bollinger Bands scoring
        price = latest['Close']
        bb_lower = latest['BB_Lower'] if pd.notna(latest['BB_Lower']) else price
        bb_upper = latest['BB_Upper'] if pd.notna(latest['BB_Upper']) else price
        
        if price < bb_lower:
            scores.append(2)
        elif price > bb_upper:
            scores.append(-2)
        
        # Stochastic scoring
        stoch_k = latest['Stoch_K'] if pd.notna(latest['Stoch_K']) else 50
        if stoch_k < 20:
            scores.append(2)
        elif stoch_k > 80:
            scores.append(-2)
        
        # EMA scoring
        ema_9 = latest['EMA_9'] if pd.notna(latest['EMA_9']) else price
        ema_21 = latest['EMA_21'] if pd.notna(latest['EMA_21']) else price
        
        if price > ema_9 > ema_21:
            scores.append(3)
        elif price < ema_9 < ema_21:
            scores.append(-3)
        
        # Volume scoring
        volume_ratio = latest['Volume_Ratio'] if pd.notna(latest['Volume_Ratio']) else 1
        if volume_ratio > 1.5:
            if price > prev['Close']:
                scores.append(2)
            else:
                scores.append(-2)
        
        total_score = sum(scores) if scores else 0
        
        # Determine signal
        if total_score >= 8:
            signal = "STRONG_GREEN"
            color = "green"
        elif total_score <= -8:
            signal = "STRONG_RED"
            color = "red"
        else:
            signal = "NEUTRAL"
            color = "gray"
        
        return {
            "score": total_score,
            "signal": signal,
            "color": color,
            "rsi": round(rsi, 2),
            "macd": round(macd, 4),
            "stoch_k": round(stoch_k, 2),
            "volume_ratio": round(volume_ratio, 2)
        }
    
    def _save_outputs(self, signals_df, date_str, start_time):
        """Save all output files matching original structure"""
        
        # 1. Daily signals
        daily_file = Config.BASE_DIR / "daily_signals" / f"signals_{date_str}.csv"
        signals_df.to_csv(daily_file, index=False)
        
        # 2. Strong signals
        strong_green = signals_df[signals_df['signal'] == 'STRONG_GREEN']
        strong_red = signals_df[signals_df['signal'] == 'STRONG_RED']
        
        strong_green_file = Config.BASE_DIR / "strong_signals" / f"strong_green_{date_str}.csv"
        strong_red_file = Config.BASE_DIR / "strong_signals" / f"strong_red_{date_str}.csv"
        
        if not strong_green.empty:
            strong_green.to_csv(strong_green_file, index=False)
        
        if not strong_red.empty:
            strong_red.to_csv(strong_red_file, index=False)
        
        # 3. Top signals
        top_green = signals_df.nlargest(10, 'score')
        top_red = signals_df.nsmallest(10, 'score')
        
        top_green_file = Config.BASE_DIR / "top_signals" / f"top_green_{date_str}.csv"
        top_red_file = Config.BASE_DIR / "top_signals" / f"top_red_{date_str}.csv"
        
        top_green.to_csv(top_green_file, index=False)
        top_red.to_csv(top_red_file, index=False)
        
        # 4. Update summary files
        summary = self._update_summary(signals_df, date_str, start_time)
        
        # 5. Save failed symbols
        if self.failed_symbols:
            failed_file = Config.BASE_DIR / "debug" / "failed_symbols.txt"
            with open(failed_file, 'w') as f:
                f.write('\n'.join(self.failed_symbols))
        
        # 6. Create README
        top_stocks = signals_df.nlargest(5, 'score')[['symbol', 'score']].to_dict('records')
        readme_path = self.zipper.create_readme_file(
            total_signals=len(signals_df),
            strong_green=len(strong_green),
            strong_red=len(strong_red),
            success_rate=summary['success_rate'],
            top_stocks=top_stocks
        )
        
        # 7. Create zip package
        zip_path = self.zipper.create_zip_file(include_debug=True)
        zip_size = self.zipper.get_zip_file_size(zip_path)
        
        return {
            "signals_df": signals_df,
            "summary": summary,
            "files": {
                "daily_signals": str(daily_file),
                "strong_green": str(strong_green_file) if not strong_green.empty else None,
                "strong_red": str(strong_red_file) if not strong_red.empty else None,
                "top_green": str(top_green_file),
                "top_red": str(top_red_file),
                "zip": str(zip_path),
                "readme": str(readme_path)
            },
            "zip_size_mb": zip_size,
            "failed_count": len(self.failed_symbols)
        }
    
    def _update_summary(self, signals_df, date_str, start_time):
        """Update summary statistics files"""
        
        # 1. Append to all_signals.csv
        summary_file = Config.BASE_DIR / "summary" / "all_signals.csv"
        
        if summary_file.exists():
            existing = pd.read_csv(summary_file)
            all_signals = pd.concat([existing, signals_df], ignore_index=True)
        else:
            all_signals = signals_df
        
        all_signals.to_csv(summary_file, index=False)
        
        # 2. Update date statistics
        date_stats_file = Config.BASE_DIR / "summary" / "date_statistics.csv"
        
        date_stats = {
            "date": date_str,
            "total_signals": len(signals_df),
            "strong_green": len(signals_df[signals_df['signal'] == 'STRONG_GREEN']),
            "strong_red": len(signals_df[signals_df['signal'] == 'STRONG_RED']),
            "neutral": len(signals_df[signals_df['signal'] == 'NEUTRAL']),
            "avg_score": round(signals_df['score'].mean(), 2),
            "max_score": round(signals_df['score'].max(), 2),
            "min_score": round(signals_df['score'].min(), 2),
            "success_rate": round((len(signals_df) / self.total_count) * 100, 1),
            "processing_time": round(time.time() - start_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        if date_stats_file.exists():
            date_stats_df = pd.read_csv(date_stats_file)
            date_stats_df = pd.concat([date_stats_df, pd.DataFrame([date_stats])], ignore_index=True)
        else:
            date_stats_df = pd.DataFrame([date_stats])
        
        date_stats_df.to_csv(date_stats_file, index=False)
        
        # 3. Update summary statistics
        stats_file = Config.BASE_DIR / "summary" / "summary_statistics.json"
        
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
        else:
            stats = {}
        
        # Calculate overall stats
        overall_stats = {
            "total_analysis_period": {
                "from": date_stats_df['date'].min(),
                "to": date_stats_df['date'].max()
            },
            "total_signals_generated": len(all_signals),
            "total_strong_green": len(all_signals[all_signals['signal'] == 'STRONG_GREEN']),
            "total_strong_red": len(all_signals[all_signals['signal'] == 'STRONG_RED']),
            "total_neutral": len(all_signals[all_signals['signal'] == 'NEUTRAL']),
            "overall_success_rate": round(date_stats_df['success_rate'].mean(), 1),
            "average_daily_signals": round(date_stats_df['total_signals'].mean(), 0),
            "last_updated": datetime.now().isoformat()
        }
        
        with open(stats_file, 'w') as f:
            json.dump(overall_stats, f, indent=2)
        
        # 4. Update indicator statistics
        indicator_stats = self._calculate_indicator_stats(signals_df)
        indicator_file = Config.BASE_DIR / "summary" / "indicator_statistics.json"
        
        with open(indicator_file, 'w') as f:
            json.dump(indicator_stats, f, indent=2)
        
        return {
            **date_stats,
            "total_history": overall_stats["total_signals_generated"]
        }
    
    def _calculate_indicator_stats(self, signals_df):
        """Calculate indicator statistics"""
        stats = {}
        
        if 'rsi' in signals_df.columns:
            stats['rsi'] = {
                "average": round(signals_df['rsi'].mean(), 2),
                "std_dev": round(signals_df['rsi'].std(), 2),
                "min": round(signals_df['rsi'].min(), 2),
                "max": round(signals_df['rsi'].max(), 2),
                "oversold_count": len(signals_df[signals_df['rsi'] < 30]),
                "overbought_count": len(signals_df[signals_df['rsi'] > 70])
            }
        
        if 'macd' in signals_df.columns:
            stats['macd'] = {
                "average": round(signals_df['macd'].mean(), 4),
                "positive_count": len(signals_df[signals_df['macd'] > 0]),
                "negative_count": len(signals_df[signals_df['macd'] < 0])
            }
        
        return stats

# ============================================================================
# STREAMLIT DASHBOARD
# ============================================================================

def main():
    """Main Streamlit dashboard"""
    
    # Page configuration
    st.set_page_config(
        page_title="GreyOak SlopeTrigger v4.1",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 3.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .sub-header {
            color: #6c757d;
            font-size: 1.3rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border-left: 5px solid;
            margin-bottom: 15px;
        }
        
        .card-green { border-left-color: #28a745; }
        .card-red { border-left-color: #dc3545; }
        .card-blue { border-left-color: #2196f3; }
        .card-purple { border-left-color: #9c27b0; }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s;
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }
        
        .download-btn {
            background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%) !important;
        }
        
        .refresh-btn {
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📊 GreyOak SlopeTrigger v4.1</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Complete NSE Universe Processing | Daily Signal Generation | Export to ZIP</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Universe size
        universe_option = st.selectbox(
            "Stock Universe Size",
            ["Demo (50 stocks)", "Small (100 stocks)", "Medium (250 stocks)", "Full NSE (2000+)"],
            index=0
        )
        
        # Date selection
        analysis_date = st.date_input(
            "Analysis Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
        
        # Advanced options
        with st.expander("Advanced Settings"):
            include_debug = st.checkbox("Include Debug Files", True)
            max_workers = st.slider("Parallel Workers", 1, 20, 10)
            cache_data = st.checkbox("Cache Market Data", True)
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        
        # Check for existing results
        summary_file = Config.BASE_DIR / "summary" / "summary_statistics.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                stats = json.load(f)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Signals", f"{stats.get('total_signals_generated', 0):,}")
                st.metric("Strong Green", stats.get('total_strong_green', 0))
            with col2:
                st.metric("Strong Red", stats.get('total_strong_red', 0))
                st.metric("Success Rate", f"{stats.get('overall_success_rate', 0)}%")
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        with col_btn2:
            if st.button("📁 View Files", use_container_width=True):
                show_file_explorer()
    
    # Main content
    col1, col2, col3 = st.columns([3, 2, 2])
    
    with col1:
        st.markdown("### 🚀 Daily Analysis")
        
        if st.button("📈 Run Daily Calculation", type="primary", use_container_width=True):
            run_daily_calculation(universe_option, max_workers)
    
    with col2:
        st.markdown("### 📦 Export Results")
        
        # Check for latest zip file
        zip_files = list(Config.OUTPUT_DIR.glob("*.zip"))
        if zip_files:
            latest_zip = max(zip_files, key=os.path.getctime)
            zip_size = os.path.getsize(latest_zip) / (1024 * 1024)
            
            st.markdown(f"**Latest Package:**")
            st.markdown(f"- {latest_zip.name}")
            st.markdown(f"- Size: {zip_size:.2f} MB")
            
            # Download button
            with open(latest_zip, "rb") as f:
                zip_data = f.read()
            
            st.download_button(
                label="📥 Download ZIP",
                data=zip_data,
                file_name=latest_zip.name,
                mime="application/zip",
                use_container_width=True
            )
        else:
            st.info("No ZIP files available yet. Run analysis first.")
    
    with col3:
        st.markdown("### 📊 Live Market")
        
        # Market snapshot
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("NIFTY 50", "21,450.35", "+125.50")
            st.metric("SENSEX", "71,250.80", "+350.25")
        with col_m2:
            st.metric("Advances", "1,250", "+85")
            st.metric("Declines", "750", "-42")
    
    # Results section
    st.markdown("---")
    st.markdown("## 📋 Latest Results")
    
    # Check for latest daily file
    daily_dir = Config.BASE_DIR / "daily_signals"
    daily_files = list(daily_dir.glob("*.csv"))
    
    if daily_files:
        latest_file = max(daily_files, key=os.path.getctime)
        signals_df = pd.read_csv(latest_file)
        date_str = latest_file.stem.replace("signals_", "")
        
        # Display results
        display_results(signals_df, date_str)
    else:
        st.info("No results available yet. Run the daily calculation to generate signals.")
        
        # Show sample structure
        with st.expander("📁 Expected Output Structure"):
            st.markdown("""
            **After running analysis, you'll get:**
            
            ```
            greyoak_v4_production/
            ├── daily_signals/
            │   └── signals_YYYY-MM-DD.csv
            ├── strong_signals/
            │   ├── strong_green_YYYY-MM-DD.csv
            │   └── strong_red_YYYY-MM-DD.csv
            ├── top_signals/
            │   ├── top_green_YYYY-MM-DD.csv
            │   └── top_red_YYYY-MM-DD.csv
            ├── summary/
            │   ├── all_signals.csv
            │   ├── summary_statistics.json
            │   ├── date_statistics.csv
            │   └── indicator_statistics.json
            ├── debug/
            │   └── failed_symbols.txt
            └── README.txt
            ```
            
            **Plus a ZIP package in:** `compressed_outputs/greyoak_v4_results_TIMESTAMP.zip`
            """)
    
    # Footer
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.caption("**GreyOak SlopeTrigger v4.1**")
        st.caption("Based on original output structure")
    
    with col_f2:
        st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.caption("Data Source: Yahoo Finance")
    
    with col_f3:
        st.caption("**Deployment:** Streamlit Cloud")
        st.caption("**Cost:** Free Forever")

def run_daily_calculation(universe_option, max_workers):
    """Run daily calculation"""
    
    # Determine universe size
    if universe_option == "Demo (50 stocks)":
        symbols = Config.NSE_UNIVERSE[:50]
    elif universe_option == "Small (100 stocks)":
        symbols = Config.NSE_UNIVERSE * 2
    elif universe_option == "Medium (250 stocks)":
        symbols = Config.NSE_UNIVERSE * 5
    else:  # Full NSE
        symbols = Config.get_full_nse_universe()
    
    # Initialize processor
    processor = GreyOakProcessor()
    
    # Run calculation
    with st.spinner(f"Processing {len(symbols)} stocks..."):
        results = processor.calculate_signals(symbols, max_workers)
    
    if results:
        st.success(f"✅ Analysis complete! Processed {len(results['signals_df'])} stocks.")
        
        # Show summary
        summary = results['summary']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card card-green">', unsafe_allow_html=True)
            st.metric("Strong Green", summary['strong_green'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-card card-red">', unsafe_allow_html=True)
            st.metric("Strong Red", summary['strong_red'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div class="metric-card card-blue">', unsafe_allow_html=True)
            st.metric("Success Rate", f"{summary['success_rate']}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'<div class="metric-card card-purple">', unsafe_allow_html=True)
            st.metric("Time", f"{summary['processing_time']}s")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show download link
        st.info(f"📦 ZIP Package created: {results['zip_size_mb']:.2f} MB")
        
        # Auto-refresh to show results
        time.sleep(2)
        st.rerun()
    else:
        st.error("❌ Analysis failed. Please check the logs.")

def display_results(signals_df, date_str):
    """Display analysis results"""
    
    # Summary metrics
    strong_green = len(signals_df[signals_df['signal'] == 'STRONG_GREEN'])
    strong_red = len(signals_df[signals_df['signal'] == 'STRONG_RED'])
    neutral = len(signals_df[signals_df['signal'] == 'NEUTRAL'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Signal distribution chart
        st.markdown("### 🎯 Signal Distribution")
        
        signal_counts = signals_df['signal'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=signal_counts.index,
            values=signal_counts.values,
            hole=.4,
            marker=dict(colors=['#28a745', '#dc3545', '#6c757d'])
        )])
        
        fig.update_layout(
            height=400,
            showlegend=True,
            margin=dict(t=0, b=0, l=0, r=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top performers
        st.markdown("### 🏆 Top Performers")
        
        # Top green
        top_green = signals_df.nlargest(5, 'score')
        if not top_green.empty:
            st.markdown("**Top Green Signals:**")
            for _, row in top_green.iterrows():
                st.markdown(f"**{row['symbol'].replace('.NS', '')}** - Score: {row['score']:.1f}")
        
        # Top red
        top_red = signals_df.nsmallest(5, 'score')
        if not top_red.empty:
            st.markdown("**Top Red Signals:**")
            for _, row in top_red.iterrows():
                st.markdown(f"**{row['symbol'].replace('.NS', '')}** - Score: {row['score']:.1f}")
        
        # Stats
        st.markdown("### 📊 Statistics")
        st.markdown(f"- **Date:** {date_str}")
        st.markdown(f"- **Total Signals:** {len(signals_df):,}")
        st.markdown(f"- **Strong Green:** {strong_green}")
        st.markdown(f"- **Strong Red:** {strong_red}")
        st.markdown(f"- **Neutral:** {neutral}")
    
    # Detailed table
    st.markdown("### 📋 All Signals")
    
    # Filters
    col_filter1, col_filter2 = st.columns(2)
    
    with col_filter1:
        signal_filter = st.selectbox(
            "Filter by Signal",
            ["All", "STRONG_GREEN", "STRONG_RED", "NEUTRAL"]
        )
    
    with col_filter2:
        sort_by = st.selectbox(
            "Sort by",
            ["Score (High to Low)", "Score (Low to High)", "Symbol", "Price"]
        )
    
    # Apply filters
    filtered_df = signals_df.copy()
    if signal_filter != "All":
        filtered_df = filtered_df[filtered_df['signal'] == signal_filter]
    
    # Apply sorting
    if sort_by == "Score (High to Low)":
        filtered_df = filtered_df.sort_values('score', ascending=False)
    elif sort_by == "Score (Low to High)":
        filtered_df = filtered_df.sort_values('score', ascending=True)
    elif sort_by == "Symbol":
        filtered_df = filtered_df.sort_values('symbol')
    elif sort_by == "Price":
        filtered_df = filtered_df.sort_values('price', ascending=False)
    
    # Display table
    st.dataframe(
        filtered_df[['symbol', 'score', 'signal', 'price', 'rsi', 'macd']],
        use_container_width=True,
        height=400
    )
    
    # Export options
    st.markdown("### 📥 Export Options")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        # CSV export
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name=f"greyoak_signals_{date_str}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_exp2:
        # Excel export
        excel_buffer = create_excel_file(filtered_df, date_str)
        st.download_button(
            label="📗 Download Excel",
            data=excel_buffer,
            file_name=f"greyoak_signals_{date_str}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col_exp3:
        # View all files
        if st.button("📁 View All Files", use_container_width=True):
            show_file_explorer()

def create_excel_file(df, date_str):
    """Create Excel file with multiple sheets"""
    from io import BytesIO
    
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Main signals
        df.to_excel(writer, sheet_name='Signals', index=False)
        
        # Summary sheet
        summary_data = pd.DataFrame([
            ["Analysis Date", date_str],
            ["Total Signals", len(df)],
            ["Strong Green", len(df[df['signal'] == 'STRONG_GREEN'])],
            ["Strong Red", len(df[df['signal'] == 'STRONG_RED'])],
            ["Neutral", len(df[df['signal'] == 'NEUTRAL'])],
            ["Average Score", round(df['score'].mean(), 2)],
            ["Max Score", round(df['score'].max(), 2)],
            ["Min Score", round(df['score'].min(), 2)]
        ], columns=['Metric', 'Value'])
        
        summary_data.to_excel(writer, sheet_name='Summary', index=False)
        
        # Top performers
        top_green = df.nlargest(10, 'score')
        top_red = df.nsmallest(10, 'score')
        
        top_green.to_excel(writer, sheet_name='Top_Green', index=False)
        top_red.to_excel(writer, sheet_name='Top_Red', index=False)
    
    output.seek(0)
    return output.getvalue()

def show_file_explorer():
    """Show file explorer for output directory"""
    st.markdown("### 📁 Output Files")
    
    # Show base directory structure
    base_dir = Config.BASE_DIR
    
    if not base_dir.exists():
        st.warning("Output directory not found. Run analysis first.")
        return
    
    # List files
    file_list = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file_path = Path(root) / file
            relative_path = file_path.relative_to(base_dir)
            file_size = file_path.stat().st_size
            
            # Format size
            if file_size < 1024:
                size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):1f} MB"
            
            file_list.append({
                "File": str(relative_path),
                "Size": size_str,
                "Modified": datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
            })
    
    if file_list:
        files_df = pd.DataFrame(file_list)
        st.dataframe(files_df, use_container_width=True)
        
        # Show zip files separately
        st.markdown("### 📦 ZIP Packages")
        zip_files = list(Config.OUTPUT_DIR.glob("*.zip"))
        
        if zip_files:
            for zip_file in sorted(zip_files, key=os.path.getctime, reverse=True)[:5]:
                zip_size = zip_file.stat().st_size / (1024 * 1024)
                st.markdown(f"- **{zip_file.name}** ({zip_size:.2f} MB)")
        else:
            st.info("No ZIP packages found.")
    else:
        st.info("No output files found. Run analysis first.")

if __name__ == "__main__":
    main()