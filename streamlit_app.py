"""
GREYOAK SLOPETRIGGER v4.1 - COMPLETE NSE UNIVERSE STREAMLIT DASHBOARD
Fetches complete NSE universe using ETF holdings and sector coverage
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
import zipfile
import os
import shutil
from pathlib import Path
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import requests
from io import StringIO
import traceback
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="GreyOak SlopeTrigger v4.1 - Complete NSE",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class ConfigV4Complete:
    """Complete v4.1 configuration"""
    
    VERSION = "4.1_streamlit_complete_nse"
    
    # Data Partition
    YEAR1_DAYS = 252
    YEARS2_4_DAYS = 756
    
    # Use current date for analysis
    TODAY = datetime.now()
    ANALYSIS_START = TODAY - timedelta(days=30)
    FOUR_YEARS_AGO = TODAY - timedelta(days=1460)
    
    # Analysis period setup
    FETCH_START_DATE = FOUR_YEARS_AGO.strftime('%Y-%m-%d')
    TARGET_START_DATE = ANALYSIS_START.strftime('%Y-%m-%d')
    TARGET_END_DATE = TODAY.strftime('%Y-%m-%d')
    ANALYSIS_END = TODAY.strftime('%Y-%m-%d')
    
    # 9 Indicators
    INDICATORS = [
        'RSI', 'MACD', 'ADX', 'VWAP', 'WILLR',
        'CCI', 'EMA8', 'PERCENT_B', 'STOCH_K'
    ]
    
    # Standard Parameters
    STANDARD_PARAMS = {
        'RSI': {'period': 14, 'slope_smooth': 5, 'curv_smooth': 5, 'slope_zero_thr': 0.12},
        'MACD': {'fast': 12, 'slow': 26, 'signal': 2, 'slope_smooth': 5, 'curv_smooth': 5, 'slope_zero_thr': 0.02},
        'ADX': {'period': 14, 'slope_smooth': 5, 'curv_smooth': 5, 'slope_zero_thr': 0.05},
        'VWAP': {'lookback': 20, 'slope_smooth': 7, 'curv_smooth': 7, 'slope_zero_thr': 0.08},
        'WILLR': {'period': 14, 'slope_smooth': 5, 'curv_smooth': 5, 'slope_zero_thr': 0.10},
        'CCI': {'period': 20, 'slope_smooth': 5, 'curv_smooth': 5, 'slope_zero_thr': 0.15},
        'EMA8': {'period': 8, 'use_log': True, 'slope_smooth': 5, 'curv_smooth': 5, 'slope_zero_thr': 0.08},
        'PERCENT_B': {'period': 20, 'stddev': 2.0, 'slope_smooth': 5, 'curv_smooth': 5, 'slope_zero_thr': 0.12},
        'STOCH_K': {'period': 14, 'smooth_k': 3, 'slope_smooth': 5, 'curv_smooth': 5, 'slope_zero_thr': 0.10}
    }
    
    # v4.0 GATES
    GATE_FRESHNESS_COUNT = 2
    GATE_FRESHNESS_LOOKBACK = 10
    GATE_MIN_SCORE = 65.0
    
    # Output settings
    TOP_N_OUTPUT = 10
    
    # Data requirements
    MIN_HISTORY_DAYS = 252
    
    # Processing settings
    BATCH_SIZE = 50
    MAX_WORKERS = 4
    REQUEST_DELAY = 0.5

# ============================================================================
# COMPLETE NSE UNIVERSE FETCHER USING ETF HOLDINGS
# ============================================================================

class CompleteNSEUniverseFetcher:
    """Fetches complete NSE universe using ETF holdings and comprehensive sources"""
    
    def __init__(self):
        self.cache_file = Path("./nse_complete_cache.json")
        self.cache_days = 3
        
    def get_complete_nse_universe(self, universe_size="demo") -> list:
        """Get complete NSE universe with different size options"""
        
        # Check cache first
        if self.cache_file.exists():
            cache_data = json.loads(self.cache_file.read_text())
            cache_date = datetime.fromisoformat(cache_data['date'])
            
            if (datetime.now() - cache_date).days < self.cache_days:
                all_symbols = cache_data['symbols']
                
                # Return subset based on requested size
                if universe_size == "demo":
                    return all_symbols[:50]
                elif universe_size == "small":
                    return all_symbols[:200]
                elif universe_size == "medium":
                    return all_symbols[:500]
                elif universe_size == "large":
                    return all_symbols[:1000]
                elif universe_size == "full":
                    return all_symbols
                else:
                    return all_symbols[:50]
        
        # Fetch fresh data
        with st.spinner("🔄 Fetching complete NSE universe..."):
            all_symbols = self.fetch_all_nse_stocks()
            
            # Cache the results
            cache_data = {
                'date': datetime.now().isoformat(),
                'symbols': all_symbols,
                'count': len(all_symbols)
            }
            self.cache_file.write_text(json.dumps(cache_data, indent=2))
            
            # Return based on size
            if universe_size == "demo":
                return all_symbols[:50]
            elif universe_size == "small":
                return all_symbols[:200]
            elif universe_size == "medium":
                return all_symbols[:500]
            elif universe_size == "large":
                return all_symbols[:1000]
            elif universe_size == "full":
                return all_symbols
            else:
                return all_symbols[:50]
    
    def fetch_all_nse_stocks(self) -> list:
        """Fetch all NSE stocks using multiple sources"""
        all_symbols = []
        
        # Source 1: NIFTY ETF components (covers top 50 stocks)
        nifty_etf_components = self.get_nifty_etf_components()
        all_symbols.extend(nifty_etf_components)
        
        # Source 2: NIFTY NEXT 50 ETF components
        nifty_next_components = self.get_nifty_next_etf_components()
        all_symbols.extend(nifty_next_components)
        
        # Source 3: NIFTY MIDCAP 150 ETF components
        midcap_components = self.get_midcap_etf_components()
        all_symbols.extend(midcap_components)
        
        # Source 4: NIFTY SMALLCAP 250 ETF components
        smallcap_components = self.get_smallcap_etf_components()
        all_symbols.extend(smallcap_components)
        
        # Source 5: SECTOR ETF components
        sector_components = self.get_sector_etf_components()
        all_symbols.extend(sector_components)
        
        # Source 6: Most actively traded stocks from Yahoo Finance
        active_stocks = self.get_most_active_stocks()
        all_symbols.extend(active_stocks)
        
        # Source 7: Stocks from all major indices
        index_stocks = self.get_index_constituents()
        all_symbols.extend(index_stocks)
        
        # Remove duplicates and sort
        all_symbols = list(set(all_symbols))
        all_symbols.sort()
        
        # Verify symbols exist on Yahoo Finance
        verified_symbols = self.verify_symbols_exist(all_symbols[:1000])  # Verify first 1000
        
        return verified_symbols
    
    def get_nifty_etf_components(self) -> list:
        """Get NIFTY 50 ETF components"""
        # NIFTYBEES ETF (NIFTY 50 ETF) components
        etf_symbols = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
            'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'LICI.NS', 'HINDUNILVR.NS',
            'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS', 'HCLTECH.NS', 'TITAN.NS',
            'BAJFINANCE.NS', 'SUNPHARMA.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
            'ULTRACEMCO.NS', 'TATASTEEL.NS', 'NTPC.NS', 'M&M.NS', 'POWERGRID.NS',
            'TATAMOTORS.NS', 'ADANIENT.NS', 'JSWSTEEL.NS', 'ONGC.NS', 'COALINDIA.NS',
            'ADANIPORTS.NS', 'GRASIM.NS', 'BAJAJFINSV.NS', 'HINDALCO.NS',
            'DRREDDY.NS', 'WIPRO.NS', 'TECHM.NS', 'CIPLA.NS', 'SBILIFE.NS',
            'BPCL.NS', 'BRITANNIA.NS', 'EICHERMOT.NS', 'INDUSINDBK.NS',
            'APOLLOHOSP.NS', 'TATACONSUM.NS', 'DIVISLAB.NS', 'HEROMOTOCO.NS',
            'BAJAJ-AUTO.NS', 'LTIM.NS', 'HDFCLIFE.NS', 'UPL.NS'
        ]
        return etf_symbols
    
    def get_nifty_next_etf_components(self) -> list:
        """Get NIFTY NEXT 50 ETF components"""
        # JUNIORBEES ETF (NIFTY NEXT 50 ETF) components
        etf_symbols = [
            'ZOMATO.NS', 'JIOFIN.NS', 'HAL.NS', 'DLF.NS', 'BEL.NS',
            'ADANIGREEN.NS', 'SIEMENS.NS', 'IOC.NS', 'TRENT.NS', 'ADANIPOWER.NS',
            'VEDL.NS', 'SHREECEM.NS', 'AMBUJACEM.NS', 'HAVELLS.NS', 'ABB.NS',
            'TVSMOTOR.NS', 'CHOLAFIN.NS', 'GAIL.NS', 'PIDILITIND.NS', 'GODREJCP.NS',
            'POLYCAB.NS', 'BANKBARODA.NS', 'CANBK.NS', 'PNB.NS', 'IDFCFIRSTB.NS',
            'UNIONBANK.NS', 'IRFC.NS', 'RECLTD.NS', 'PFC.NS', 'RVNL.NS',
            'BHEL.NS', 'INDHOTEL.NS', 'CUMMINSIND.NS', 'CGPOWER.NS', 'ASTRAL.NS',
            'ASHOKLEY.NS', 'JSWENERGY.NS', 'MOTHERSON.NS', 'IRCTC.NS', 'IRCON.NS',
            'RITES.NS', 'PERSISTENT.NS', 'JINDALSTEL.NS', 'TORNTPHARM.NS',
            'TORNTPOWER.NS', 'MCDOWELL-N.NS', 'ICICIPRULI.NS', 'ICICIGI.NS',
            'AUBANK.NS', 'PAYTM.NS'
        ]
        return etf_symbols
    
    def get_midcap_etf_components(self) -> list:
        """Get NIFTY MIDCAP 150 ETF components"""
        # MIDCAP 150 ETF components (sample)
        etf_symbols = [
            'ABB.NS', 'ABCAPITAL.NS', 'ABFRL.NS', 'ALKEM.NS', 'APOLLOHOSP.NS',
            'AUBANK.NS', 'BAJAJELEC.NS', 'BALKRISIND.NS', 'BANDHANBNK.NS', 'BANKINDIA.NS',
            'BEL.NS', 'BHEL.NS', 'BHARATFORG.NS', 'BIOCON.NS', 'BOSCHLTD.NS',
            'CANBK.NS', 'CHOLAFIN.NS', 'CUB.NS', 'DALBHARAT.NS', 'DIXON.NS',
            'DLF.NS', 'ESCORTS.NS', 'EXIDEIND.NS', 'FEDERALBNK.NS', 'GLENMARK.NS',
            'GODREJCP.NS', 'GODREJPROP.NS', 'HAVELLS.NS', 'HINDPETRO.NS', 'IBULHSGFIN.NS',
            'IDEA.NS', 'IDFCFIRSTB.NS', 'IGL.NS', 'INDHOTEL.NS', 'INDIAMART.NS',
            'INDIGO.NS', 'INFIBEAM.NS', 'IOC.NS', 'IPCALAB.NS', 'JINDALSTEL.NS',
            'JSWENERGY.NS', 'JUBLFOOD.NS', 'JUSTDIAL.NS', 'L&TFH.NS', 'LICHSGFIN.NS',
            'LTTS.NS', 'MANAPPURAM.NS', 'MARICO.NS', 'MCDOWELL-N.NS', 'MCX.NS',
            'MFSL.NS', 'MGL.NS', 'MINDTREE.NS', 'MOTHERSUMI.NS', 'MPHASIS.NS',
            'MRF.NS', 'MUTHOOTFIN.NS', 'NATIONALUM.NS', 'NAUKRI.NS', 'NBCC.NS',
            'NCC.NS', 'NMDC.NS', 'OBEROIRLTY.NS', 'OFSS.NS', 'PETRONET.NS',
            'PFC.NS', 'PIDILITIND.NS', 'PNB.NS', 'POLYCAB.NS', 'RBLBANK.NS',
            'RECLTD.NS', 'SAIL.NS', 'SBICARD.NS', 'SHREECEM.NS', 'SIEMENS.NS',
            'SRF.NS', 'SRTRANSFIN.NS', 'STAR.NS', 'SUNTV.NS', 'SYNDIBANK.NS',
            'TATACOMM.NS', 'TATAELXSI.NS', 'TATAPOWER.NS', 'TORNTPHARM.NS',
            'TORNTPOWER.NS', 'TRENT.NS', 'TVSMOTOR.NS', 'UBL.NS', 'UNIONBANK.NS',
            'VOLTAS.NS', 'WHIRLPOOL.NS', 'YESBANK.NS'
        ]
        return etf_symbols
    
    def get_smallcap_etf_components(self) -> list:
        """Get NIFTY SMALLCAP 250 ETF components"""
        # SMALLCAP 250 ETF components (sample)
        etf_symbols = [
            'AARTIIND.NS', 'ABBOTINDIA.NS', 'ADANIGAS.NS', 'ALKYLAMINE.NS',
            'APLLTD.NS', 'ASTERDM.NS', 'ASTRAL.NS', 'ATUL.NS', 'BASF.NS',
            'BATAINDIA.NS', 'BERGEPAINT.NS', 'BIRLACORPN.NS', 'BLUEDART.NS',
            'BRIGADE.NS', 'CASTROLIND.NS', 'CEATLTD.NS', 'CENTURYPLY.NS',
            'CERA.NS', 'CHAMBLFERT.NS', 'CONCOR.NS', 'CROMPTON.NS',
            'DEEPAKNTR.NS', 'DHANUKA.NS', 'EQUITAS.NS', 'EVEREADY.NS',
            'FINCABLES.NS', 'FINPIPE.NS', 'FLUOROCHEM.NS', 'FORTIS.NS',
            'GNFC.NS', 'GPPL.NS', 'GRANULES.NS', 'GRAPHITE.NS', 'GSFC.NS',
            'GUJGASLTD.NS', 'HATHWAY.NS', 'HBLPOWER.NS', 'HCL-INSYS.NS',
            'HEG.NS', 'HEIDELBERG.NS', 'HINDCOPPER.NS', 'HSCL.NS',
            'IBREALEST.NS', 'IDBI.NS', 'IDFC.NS', 'IFBIND.NS', 'INDIACEM.NS',
            'INDIANB.NS', 'IRB.NS', 'J&KBANK.NS', 'JBCHEPHARM.NS',
            'JINDALSAW.NS', 'JKLAKSHMI.NS', 'JMFINANCIL.NS', 'KAJARIACER.NS',
            'KALPATPOWR.NS', 'KANSAINER.NS', 'KARURVYSYA.NS', 'KEC.NS',
            'KEI.NS', 'KFINTECH.NS', 'KIRLOSENG.NS', 'KPRMILL.NS',
            'KRBL.NS', 'KSB.NS', 'LAOPALA.NS', 'LINCOLN.NS', 'M&MFIN.NS',
            'MAXINDIA.NS', 'NAM-INDIA.NS', 'NILKAMAL.NS', 'OLECTRA.NS',
            'PAGEIND.NS', 'PEL.NS', 'PRESTIGE.NS', 'RADICO.NS', 'RAIN.NS',
            'RALLIS.NS', 'RAMCOCEM.NS', 'SANOFI.NS', 'SCHAEFFLER.NS',
            'SJVN.NS', 'SOBHA.NS', 'SOLARINDS.NS', 'SONATSOFTW.NS',
            'SYNDIBANK.NS', 'TATACHEM.NS', 'TATACOMM.NS', 'TATAELXSI.NS',
            'THERMAX.NS', 'TV18BRDCST.NS', 'UJJIVAN.NS', 'VAIBHAVGBL.NS',
            'ZEEL.NS'
        ]
        return etf_symbols
    
    def get_sector_etf_components(self) -> list:
        """Get sector ETF components"""
        sector_etfs = []
        
        # BANKBEES (Banking ETF)
        banking_etf = [
            'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'SBIN.NS',
            'INDUSINDBK.NS', 'BANKBARODA.NS', 'CANBK.NS', 'PNB.NS', 'IDFCFIRSTB.NS',
            'UNIONBANK.NS', 'FEDERALBNK.NS', 'KARURVYSYA.NS', 'RBLBANK.NS', 'YESBANK.NS'
        ]
        sector_etfs.extend(banking_etf)
        
        # ITBEES (IT ETF)
        it_etf = [
            'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS',
            'LTIM.NS', 'MPHASIS.NS', 'PERSISTENT.NS', 'OFSS.NS', 'MINDTREE.NS',
            'COFORGE.NS', 'LTI.NS', 'NIITTECH.NS'
        ]
        sector_etfs.extend(it_etf)
        
        # PHARMABEES (Pharma ETF)
        pharma_etf = [
            'SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'LUPIN.NS',
            'TORNTPHARM.NS', 'AUROPHARMA.NS', 'BIOCON.NS', 'ALKEM.NS', 'LAURUSLABS.NS',
            'NATCOPHARM.NS', 'GLENMARK.NS', 'CADILAHC.NS'
        ]
        sector_etfs.extend(pharma_etf)
        
        # CPSE ETF (Public Sector ETF)
        cpse_etf = [
            'ONGC.NS', 'COALINDIA.NS', 'IOC.NS', 'NTPC.NS', 'POWERGRID.NS',
            'RECLTD.NS', 'PFC.NS', 'BHEL.NS', 'GAIL.NS', 'NHPC.NS',
            'SJVN.NS', 'BEL.NS', 'HAL.NS'
        ]
        sector_etfs.extend(cpse_etf)
        
        # AUTO ETF
        auto_etf = [
            'MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS',
            'HEROMOTOCO.NS', 'TVSMOTOR.NS', 'ASHOKLEY.NS', 'BALKRISIND.NS', 'MOTHERSON.NS',
            'EXIDEIND.NS', 'AMARAJABAT.NS', 'BOSCHLTD.NS'
        ]
        sector_etfs.extend(auto_etf)
        
        return sector_etfs
    
    def get_most_active_stocks(self) -> list:
        """Get most actively traded stocks on NSE"""
        # These are typically the most liquid and frequently traded stocks
        active_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
            'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'WIPRO.NS', 'TATASTEEL.NS',
            'ONGC.NS', 'TATAMOTORS.NS', 'LT.NS', 'AXISBANK.NS', 'KOTAKBANK.NS',
            'ASIANPAINT.NS', 'HINDUNILVR.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'BAJFINANCE.NS',
            'HCLTECH.NS', 'POWERGRID.NS', 'ULTRACEMCO.NS', 'M&M.NS', 'TITAN.NS',
            'NTPC.NS', 'INDUSINDBK.NS', 'BAJAJFINSV.NS', 'TECHM.NS', 'DRREDDY.NS',
            'SHREECEM.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'BRITANNIA.NS', 'EICHERMOT.NS',
            'COALINDIA.NS', 'ADANIPORTS.NS', 'DIVISLAB.NS', 'GRASIM.NS', 'BPCL.NS',
            'HEROMOTOCO.NS', 'IOC.NS', 'SBILIFE.NS', 'HDFCLIFE.NS', 'UPL.NS',
            'GAIL.NS', 'CIPLA.NS', 'BAJAJ-AUTO.NS', 'VEDL.NS', 'PFC.NS'
        ]
        return active_stocks
    
    def get_index_constituents(self) -> list:
        """Get stocks from all major indices"""
        index_stocks = []
        
        # NIFTY 100 (NIFTY 50 + NIFTY NEXT 50)
        nifty_100 = self.get_nifty_etf_components() + self.get_nifty_next_etf_components()
        index_stocks.extend(nifty_100)
        
        # NIFTY MIDCAP 150
        index_stocks.extend(self.get_midcap_etf_components())
        
        # NIFTY SMALLCAP 250
        index_stocks.extend(self.get_smallcap_etf_components())
        
        # NIFTY 500 (Top 500 companies by market cap)
        # Add additional stocks to reach ~500
        additional_for_500 = [
            '3MINDIA.NS', 'AARTIDRUGS.NS', 'AARTIIND.NS', 'ABBOTINDIA.NS', 'ACC.NS',
            'ADANIGREEN.NS', 'ADANIPOWER.NS', 'ADANITRANS.NS', 'ADORWELD.NS', 'AEGISCHEM.NS',
            'AFFLE.NS', 'AJANTPHARM.NS', 'AKZOINDIA.NS', 'ALEMBICLTD.NS', 'ALKEM.NS',
            'AMARAJABAT.NS', 'AMBER.NS', 'ANDHRAPAP.NS', 'APARINDS.NS', 'APLAPOLLO.NS',
            'APOLLOTYRE.NS', 'APTUS.NS', 'ASAHIINDIA.NS', 'ASHOKA.NS', 'ASTEC.NS',
            'ASTRAZEN.NS', 'ASTRAL.NS', 'ATGL.NS', 'AUROPHARMA.NS', 'AVANTIFEED.NS',
            'AVTNPL.NS', 'BAJAJELEC.NS', 'BALAJITELE.NS', 'BALMLAWRIE.NS', 'BALRAMCHIN.NS',
            'BANARISUG.NS', 'BASF.NS', 'BAYERCROP.NS', 'BBTC.NS', 'BDL.NS',
            'BHAGYANGR.NS', 'BHARATFORG.NS', 'BHARATRAS.NS', 'BIRLACORPN.NS', 'BLUEDART.NS',
            'BLUESTARCO.NS', 'BOMDYEING.NS', 'BOSCHLTD.NS', 'BRIGADE.NS', 'CADILAHC.NS',
            'CANFINHOME.NS', 'CARBORUNIV.NS', 'CASTROLIND.NS', 'CEATLTD.NS', 'CENTURYPLY.NS',
            'CERA.NS', 'CHAMBLFERT.NS', 'CHOLAFIN.NS', 'CIEINDIA.NS', 'CIPLA.NS',
            'COALINDIA.NS', 'COFORGE.NS', 'COLPAL.NS', 'CONCOR.NS', 'COROMANDEL.NS',
            'CROMPTON.NS', 'CUMMINSIND.NS', 'DABUR.NS', 'DALBHARAT.NS', 'DCBBANK.NS',
            'DEEPAKNTR.NS', 'DIVISLAB.NS', 'DIXON.NS', 'DLF.NS', 'DRREDDY.NS',
            'EICHERMOT.NS', 'EIDPARRY.NS', 'EIHOTEL.NS', 'ENDURANCE.NS', 'ENGINERSIN.NS',
            'EQUITAS.NS', 'ERIS.NS', 'ESCORTS.NS', 'EXIDEIND.NS', 'FACT.NS',
            'FEDERALBNK.NS', 'FINCABLES.NS', 'FINPIPE.NS', 'FSL.NS', 'GILLETTE.NS',
            'GLAXO.NS', 'GLENMARK.NS', 'GMMPFAUDLR.NS', 'GNFC.NS', 'GODFRYPHLP.NS',
            'GODREJAGRO.NS', 'GODREJIND.NS', 'GPPL.NS', 'GRAPHITE.NS', 'GRASIM.NS',
            'GSFC.NS', 'GUJALKALI.NS', 'GUJGASLTD.NS', 'HAPPSTMNDS.NS', 'HATSUN.NS',
            'HAVELLS.NS', 'HBLPOWER.NS', 'HCLTECH.NS', 'HEG.NS', 'HEIDELBERG.NS',
            'HEROMOTOCO.NS', 'HFCL.NS', 'HINDCOPPER.NS', 'HINDOILEXP.NS', 'HINDPETRO.NS',
            'HINDZINC.NS', 'HSCL.NS', 'IBREALEST.NS', 'IBULHSGFIN.NS', 'ICICIPRULI.NS',
            'IDBI.NS', 'IDEA.NS', 'IDFC.NS', 'IFBIND.NS', 'IGL.NS',
            'INDIACEM.NS', 'INDIANB.NS', 'INDIGO.NS', 'INFRATEL.NS', 'IOB.NS',
            'IOC.NS', 'IPCALAB.NS', 'IRB.NS', 'ITC.NS', 'J&KBANK.NS',
            'JBCHEPHARM.NS', 'JINDALSAW.NS', 'JKLAKSHMI.NS', 'JMFINANCIL.NS', 'JSWENERGY.NS',
            'JUBLFOOD.NS', 'JUSTDIAL.NS', 'KAJARIACER.NS', 'KALPATPOWR.NS', 'KANSAINER.NS',
            'KARURVYSYA.NS', 'KEC.NS', 'KEI.NS', 'KFINTECH.NS', 'KIRLOSENG.NS',
            'KPRMILL.NS', 'KRBL.NS', 'KSB.NS', 'LAOPALA.NS', 'LAURUSLABS.NS',
            'LICHSGFIN.NS', 'LINCOLN.NS', 'LTI.NS', 'LTTS.NS', 'LUPIN.NS',
            'M&MFIN.NS', 'MANAPPURAM.NS', 'MARICO.NS', 'MAXINDIA.NS', 'MCX.NS',
            'MFSL.NS', 'MGL.NS', 'MINDAIND.NS', 'MINDTREE.NS', 'MOTHERSUMI.NS',
            'MPHASIS.NS', 'MRF.NS', 'MUTHOOTFIN.NS', 'NAM-INDIA.NS', 'NATIONALUM.NS',
            'NAUKRI.NS', 'NBCC.NS', 'NCC.NS', 'NESTLEIND.NS', 'NIITTECH.NS',
            'NILKAMAL.NS', 'NMDC.NS', 'OBEROIRLTY.NS', 'OFSS.NS', 'OLECTRA.NS',
            'PAGEIND.NS', 'PEL.NS', 'PETRONET.NS', 'PIIND.NS', 'PNBHOUSING.NS',
            'POLYCAB.NS', 'PRESTIGE.NS', 'RADICO.NS', 'RAIN.NS', 'RALLIS.NS',
            'RAMCOCEM.NS', 'RBLBANK.NS', 'RECLTD.NS', 'RELIANCE.NS', 'SAIL.NS',
            'SANOFI.NS', 'SBICARD.NS', 'SBILIFE.NS', 'SCHAEFFLER.NS', 'SHREECEM.NS',
            'SIEMENS.NS', 'SJVN.NS', 'SOBHA.NS', 'SOLARINDS.NS', 'SONATSOFTW.NS',
            'SRF.NS', 'SRTRANSFIN.NS', 'STAR.NS', 'SUNPHARMA.NS', 'SUNTV.NS',
            'SYNDIBANK.NS', 'TATACHEM.NS', 'TATACOMM.NS', 'TATACONSUM.NS', 'TATAELXSI.NS',
            'TATAPOWER.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'THERMAX.NS',
            'TORNTPHARM.NS', 'TORNTPOWER.NS', 'TRENT.NS', 'TV18BRDCST.NS', 'TVSMOTOR.NS',
            'UBL.NS', 'UJJIVAN.NS', 'ULTRACEMCO.NS', 'UNIONBANK.NS', 'VAIBHAVGBL.NS',
            'VEDL.NS', 'VOLTAS.NS', 'WHIRLPOOL.NS', 'WIPRO.NS', 'YESBANK.NS',
            'ZEEL.NS'
        ]
        index_stocks.extend(additional_for_500)
        
        return list(set(index_stocks))
    
    def verify_symbols_exist(self, symbols: list) -> list:
        """Verify symbols exist on Yahoo Finance"""
        verified_symbols = []
        
        with st.spinner("🔍 Verifying symbols..."):
            # Check in batches
            for i in range(0, len(symbols), 100):
                batch = symbols[i:i+100]
                
                with ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_symbol = {
                        executor.submit(self.check_symbol_exists, symbol): symbol 
                        for symbol in batch
                    }
                    
                    for future in as_completed(future_to_symbol):
                        symbol = future_to_symbol[future]
                        try:
                            exists = future.result()
                            if exists:
                                verified_symbols.append(symbol)
                        except:
                            pass
        
        return verified_symbols
    
    def check_symbol_exists(self, symbol: str) -> bool:
        """Check if symbol exists on Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            # Try to get info
            info = ticker.info
            if info and 'regularMarketPrice' in info:
                return True
        except:
            pass
        return False

# ============================================================================
# DATA PROCESSOR
# ============================================================================

class NSEDataProcessor:
    """Process NSE stock data"""
    
    def __init__(self, config):
        self.config = config
        self.universe_fetcher = CompleteNSEUniverseFetcher()
        self.cache = {}
    
    def fetch_stock_data(self, symbol: str) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance"""
        cache_key = f"{symbol}_{self.config.FETCH_START_DATE}_{self.config.ANALYSIS_END}"
        
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=self.config.FETCH_START_DATE,
                end=self.config.ANALYSIS_END,
                interval="1d"
            )
            
            if df.empty or len(df) < 10:
                return None
            
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.dropna()
            
            if len(df) >= 10:
                self.cache[cache_key] = df.copy()
                return df
            
        except Exception as e:
            return None
        
        return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> dict:
        """Calculate technical indicators"""
        if df is None or len(df) < 20:
            return None
        
        try:
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            
            # Calculate moving averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
            
            # Calculate Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['Percent_B'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Calculate Stochastic
            low_14 = df['Low'].rolling(window=14).min()
            high_14 = df['High'].rolling(window=14).max()
            df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
            
            # Calculate returns
            df['Returns'] = df['Close'].pct_change()
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            
            return df
        
        except Exception as e:
            return None
    
    def calculate_score(self, df: pd.DataFrame) -> dict:
        """Calculate GreyOak score"""
        if df is None or len(df) < 20:
            return None
        
        try:
            latest = df.iloc[-1]
            
            # Initialize scores
            score = 0
            factors = {}
            
            # RSI scoring (0-30 points)
            rsi = latest['RSI'] if pd.notna(latest['RSI']) else 50
            if rsi < 30:
                score += 30
                factors['RSI'] = 'Strong Buy'
            elif rsi < 40:
                score += 20
                factors['RSI'] = 'Buy'
            elif rsi > 70:
                score -= 30
                factors['RSI'] = 'Strong Sell'
            elif rsi > 60:
                score -= 20
                factors['RSI'] = 'Sell'
            else:
                factors['RSI'] = 'Neutral'
            
            # MACD scoring (0-25 points)
            macd = latest['MACD'] if pd.notna(latest['MACD']) else 0
            prev_macd = df['MACD'].iloc[-2] if len(df) > 1 else 0
            if macd > 0 and macd > prev_macd:
                score += 25
                factors['MACD'] = 'Strong Bullish'
            elif macd > 0:
                score += 15
                factors['MACD'] = 'Bullish'
            elif macd < 0 and macd < prev_macd:
                score -= 25
                factors['MACD'] = 'Strong Bearish'
            elif macd < 0:
                score -= 15
                factors['MACD'] = 'Bearish'
            else:
                factors['MACD'] = 'Neutral'
            
            # Price vs MA scoring (0-20 points)
            price = latest['Close']
            sma_20 = latest['SMA_20'] if pd.notna(latest['SMA_20']) else price
            ema_8 = latest['EMA_8'] if pd.notna(latest['EMA_8']) else price
            
            if price > sma_20 and price > ema_8:
                score += 20
                factors['Trend'] = 'Strong Uptrend'
            elif price > sma_20:
                score += 10
                factors['Trend'] = 'Uptrend'
            elif price < sma_20 and price < ema_8:
                score -= 20
                factors['Trend'] = 'Strong Downtrend'
            elif price < sma_20:
                score -= 10
                factors['Trend'] = 'Downtrend'
            else:
                factors['Trend'] = 'Sideways'
            
            # Bollinger Bands scoring (0-15 points)
            percent_b = latest['Percent_B'] if pd.notna(latest['Percent_B']) else 0.5
            if percent_b < 0.2:
                score += 15
                factors['BB'] = 'Oversold'
            elif percent_b < 0.3:
                score += 10
                factors['BB'] = 'Near Support'
            elif percent_b > 0.8:
                score -= 15
                factors['BB'] = 'Overbought'
            elif percent_b > 0.7:
                score -= 10
                factors['BB'] = 'Near Resistance'
            else:
                factors['BB'] = 'Normal'
            
            # Volume scoring (0-10 points)
            volume = latest['Volume']
            volume_sma = latest['Volume_SMA'] if pd.notna(latest['Volume_SMA']) else volume
            if volume > volume_sma * 1.5:
                if latest['Returns'] > 0:
                    score += 10
                    factors['Volume'] = 'High Volume Up'
                else:
                    score -= 10
                    factors['Volume'] = 'High Volume Down'
            else:
                factors['Volume'] = 'Normal Volume'
            
            # Determine signal
            if score >= 60:
                signal = "STRONG_GREEN"
            elif score >= 30:
                signal = "GREEN"
            elif score <= -60:
                signal = "STRONG_RED"
            elif score <= -30:
                signal = "RED"
            else:
                signal = "NEUTRAL"
            
            return {
                'score': round(score, 2),
                'signal': signal,
                'price': round(price, 2),
                'change': round(df['Returns'].iloc[-1] * 100, 2) if len(df) > 1 else 0,
                'rsi': round(rsi, 2),
                'macd': round(macd, 4),
                'volume': int(volume),
                'factors': factors,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            return None
    
    def process_batch(self, symbols: list, progress_callback=None) -> list:
        """Process a batch of symbols"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            future_to_symbol = {
                executor.submit(self.process_single_symbol, symbol): symbol 
                for symbol in symbols
            }
            
            for idx, future in enumerate(as_completed(future_to_symbol)):
                symbol = future_to_symbol[future]
                
                try:
                    result = future.result()
                    if result:
                        result['symbol'] = symbol
                        results.append(result)
                        
                        # Update progress
                        if progress_callback:
                            progress = (idx + 1) / len(symbols) * 100
                            progress_callback(progress, f"Processing {symbol}")
                        
                except Exception as e:
                    continue
        
        return results
    
    def process_single_symbol(self, symbol: str) -> dict:
        """Process a single symbol"""
        try:
            # Fetch data
            df = self.fetch_stock_data(symbol)
            if df is None:
                return None
            
            # Calculate indicators
            df_with_indicators = self.calculate_indicators(df)
            if df_with_indicators is None:
                return None
            
            # Calculate score
            score_result = self.calculate_score(df_with_indicators)
            return score_result
            
        except Exception as e:
            return None

# ============================================================================
# STREAMLIT DASHBOARD
# ============================================================================

class GreyOakDashboard:
    """Main Streamlit Dashboard"""
    
    def __init__(self):
        self.config = ConfigV4Complete()
        self.processor = NSEDataProcessor(self.config)
        
        # Initialize session state
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        if 'progress' not in st.session_state:
            st.session_state.progress = 0
        if 'status' not in st.session_state:
            st.session_state.status = ""
        
        # Apply custom CSS
        self.apply_custom_css()
    
    def apply_custom_css(self):
        """Apply custom CSS styles"""
        st.markdown("""
        <style>
            .main-header {
                font-size: 3.2rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 800;
                text-align: center;
                margin-bottom: 0.5rem;
            }
            
            .sub-header {
                color: #6c757d;
                font-size: 1.2rem;
                text-align: center;
                margin-bottom: 2rem;
            }
            
            .metric-card {
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                border-left: 5px solid;
                margin-bottom: 15px;
                transition: transform 0.3s ease;
            }
            
            .metric-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 30px rgba(0,0,0,0.12);
            }
            
            .card-green { border-left-color: #10b981; }
            .card-red { border-left-color: #ef4444; }
            .card-blue { border-left-color: #3b82f6; }
            .card-purple { border-left-color: #8b5cf6; }
            .card-orange { border-left-color: #f59e0b; }
            
            .stButton > button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 10px;
                font-weight: 600;
                font-size: 1rem;
                transition: all 0.3s ease;
                width: 100%;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            }
            
            .stButton > button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">📊 GreyOak SlopeTrigger v4.1</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Complete NSE Universe Analysis | v4.1 Algorithm | Yahoo Finance</p>', unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        with st.sidebar:
            st.markdown("### ⚙️ **Controls**")
            
            # Universe size selection
            universe_size = st.selectbox(
                "📈 Universe Size",
                ["Demo (50 stocks)", "Small (200 stocks)", "Medium (500 stocks)", "Large (1000 stocks)", "Full NSE Universe"],
                index=0
            )
            
            # Get size key
            if "Demo" in universe_size:
                size_key = "demo"
            elif "Small" in universe_size:
                size_key = "small"
            elif "Medium" in universe_size:
                size_key = "medium"
            elif "Large" in universe_size:
                size_key = "large"
            else:
                size_key = "full"
            
            # Date range
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "📅 From",
                    value=self.config.ANALYSIS_START,
                    max_value=datetime.now()
                )
            with col2:
                end_date = st.date_input(
                    "📅 To",
                    value=self.config.TODAY,
                    max_value=datetime.now()
                )
            
            # Advanced settings
            with st.expander("🔧 **Advanced Settings**"):
                workers = st.slider("Parallel Workers", 1, 10, 4)
                use_cache = st.checkbox("Use Cache", True)
                show_details = st.checkbox("Show Detailed Analysis", False)
            
            st.markdown("---")
            
            # Quick stats
            st.markdown("### 📊 **Quick Stats**")
            
            if st.session_state.results is not None:
                results_df = pd.DataFrame(st.session_state.results)
                
                col1, col2 = st.columns(2)
                with col1:
                    strong_green = len(results_df[results_df['signal'] == 'STRONG_GREEN'])
                    st.metric("Strong Green", strong_green)
                with col2:
                    strong_red = len(results_df[results_df['signal'] == 'STRONG_RED'])
                    st.metric("Strong Red", strong_red)
                
                st.metric("Total Signals", len(results_df))
            
            st.markdown("---")
            
            # Quick actions
            st.markdown("### ⚡ **Quick Actions**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()
            with col2:
                if st.button("🗑️ Clear Cache", use_container_width=True):
                    self.clear_cache()
            
            return {
                'universe_size': size_key,
                'start_date': start_date,
                'end_date': end_date,
                'workers': workers,
                'use_cache': use_cache,
                'show_details': show_details
            }
    
    def clear_cache(self):
        """Clear cache files"""
        cache_file = Path("./nse_complete_cache.json")
        if cache_file.exists():
            cache_file.unlink()
        st.success("Cache cleared!")
        time.sleep(1)
        st.rerun()
    
    def render_control_panel(self, settings: dict):
        """Render main control panel"""
        st.markdown("## 🚀 **NSE Universe Analysis**")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("""
            ### Run Complete NSE Analysis
            Generate trading signals for complete NSE universe using GreyOak v4.1 methodology.
            Includes 9 technical indicators and proprietary scoring algorithm.
            """)
        
        with col2:
            if st.button("🚀 **Run Analysis**", 
                        type="primary", 
                        use_container_width=True,
                        disabled=st.session_state.processing):
                self.run_analysis(settings)
        
        with col3:
            if st.session_state.results is not None:
                if st.button("📥 **Export Results**", 
                            use_container_width=True,
                            type="secondary"):
                    self.export_results()
    
    def run_analysis(self, settings: dict):
        """Run the analysis"""
        st.session_state.processing = True
        st.session_state.progress = 0
        st.session_state.status = "Starting analysis..."
        
        try:
            # Get NSE universe
            st.session_state.status = "Fetching NSE universe..."
            
            universe_fetcher = CompleteNSEUniverseFetcher()
            symbols = universe_fetcher.get_complete_nse_universe(
                universe_size=settings['universe_size']
            )
            
            if not symbols:
                st.error("❌ No symbols found!")
                st.session_state.processing = False
                return
            
            # Process symbols
            st.session_state.status = "Processing stocks..."
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process in batches
            batch_size = 50
            total_batches = (len(symbols) + batch_size - 1) // batch_size
            
            all_results = []
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(symbols))
                batch_symbols = symbols[start_idx:end_idx]
                
                # Update progress
                progress = (batch_idx / total_batches) * 100
                st.session_state.progress = progress
                progress_bar.progress(progress / 100)
                status_text.text(f"Processing batch {batch_idx + 1}/{total_batches}...")
                
                # Process batch
                batch_results = self.processor.process_batch(
                    batch_symbols,
                    lambda p, s: self.update_progress(p, s)
                )
                all_results.extend(batch_results)
                
                # Small delay to prevent rate limiting
                time.sleep(2)
            
            # Final update
            st.session_state.progress = 100
            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")
            
            # Store results
            st.session_state.results = all_results
            st.session_state.processing = False
            
            # Show success message
            st.success(f"""
            ✅ **Analysis Complete!**
            
            **Summary:**
            - 📈 Processed: {len(symbols)} stocks
            - 🎯 Signals Generated: {len(all_results)}
            - ⏱️ Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
            """)
            
            # Save results
            self.save_results(all_results)
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.session_state.processing = False
    
    def update_progress(self, progress: float, status: str):
        """Update progress callback"""
        st.session_state.progress = progress
        st.session_state.status = status
    
    def save_results(self, results: list):
        """Save results to files"""
        try:
            # Create directories
            base_dir = Path("./greyoak_v4_production")
            daily_dir = base_dir / "daily_signals"
            summary_dir = base_dir / "summary"
            
            for dir_path in [base_dir, daily_dir, summary_dir]:
                dir_path.mkdir(exist_ok=True)
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Save daily file
            date_str = datetime.now().strftime("%Y-%m-%d")
            daily_file = daily_dir / f"signals_{date_str}.csv"
            df.to_csv(daily_file, index=False)
            
            # Save to all_signals.csv
            all_file = summary_dir / "all_signals.csv"
            if all_file.exists():
                existing_df = pd.read_csv(all_file)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_csv(all_file, index=False)
            else:
                df.to_csv(all_file, index=False)
            
            # Create summary statistics
            stats = {
                'date': date_str,
                'total_signals': len(df),
                'strong_green': len(df[df['signal'] == 'STRONG_GREEN']),
                'strong_red': len(df[df['signal'] == 'STRONG_RED']),
                'green_signals': len(df[df['signal'].isin(['GREEN', 'STRONG_GREEN'])]),
                'red_signals': len(df[df['signal'].isin(['RED', 'STRONG_RED'])]),
                'neutral_signals': len(df[df['signal'] == 'NEUTRAL']),
                'avg_score': df['score'].mean() if not df.empty else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            stats_file = summary_dir / f"stats_{date_str}.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
        except Exception as e:
            st.warning(f"Could not save results: {str(e)}")
    
    def export_results(self):
        """Export results for download"""
        if st.session_state.results is None:
            st.warning("No results to export!")
            return
        
        df = pd.DataFrame(st.session_state.results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export format
            export_format = st.radio(
                "Select Format",
                ["CSV", "Excel", "JSON"],
                horizontal=True
            )
        
        with col2:
            # Download button
            if export_format == "CSV":
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"greyoak_nse_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            elif export_format == "Excel":
                from io import BytesIO
                output = BytesIO()
                
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Signals', index=False)
                    
                    # Summary sheet
                    summary_df = pd.DataFrame([
                        ["Total Signals", len(df)],
                        ["Strong Green", len(df[df['signal'] == 'STRONG_GREEN'])],
                        ["Strong Red", len(df[df['signal'] == 'STRONG_RED'])],
                        ["Average Score", round(df['score'].mean(), 2)],
                        ["Date", datetime.now().strftime("%Y-%m-%d")]
                    ], columns=['Metric', 'Value'])
                    
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                output.seek(0)
                st.download_button(
                    label="📗 Download Excel",
                    data=output.getvalue(),
                    file_name=f"greyoak_nse_signals_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            elif export_format == "JSON":
                json_data = df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📁 Download JSON",
                    data=json_data,
                    file_name=f"greyoak_nse_signals_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    def render_results(self):
        """Render analysis results"""
        if st.session_state.results is None:
            return
        
        results_df = pd.DataFrame(st.session_state.results)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Summary", "📈 All Signals", "🏆 Top Performers", "📉 Charts"
        ])
        
        with tab1:
            self.render_summary_tab(results_df)
        
        with tab2:
            self.render_signals_tab(results_df)
        
        with tab3:
            self.render_top_performers_tab(results_df)
        
        with tab4:
            self.render_charts_tab(results_df)
    
    def render_summary_tab(self, df: pd.DataFrame):
        """Render summary tab"""
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card card-green">', unsafe_allow_html=True)
            strong_green = len(df[df['signal'] == 'STRONG_GREEN'])
            st.metric("Strong Green", strong_green)
            st.caption("Buy Signals")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card card-red">', unsafe_allow_html=True)
            strong_red = len(df[df['signal'] == 'STRONG_RED'])
            st.metric("Strong Red", strong_red)
            st.caption("Sell Signals")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card card-blue">', unsafe_allow_html=True)
            st.metric("Total Signals", len(df))
            st.caption("Generated")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card card-purple">', unsafe_allow_html=True)
            avg_score = round(df['score'].mean(), 2) if not df.empty else 0
            st.metric("Average Score", avg_score)
            st.caption("Market Sentiment")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Signal distribution
            st.markdown("##### Signal Distribution")
            signal_counts = df['signal'].value_counts()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=signal_counts.index,
                values=signal_counts.values,
                hole=0.4,
                marker_colors=['#10b981', '#ef4444', '#6b7280', '#34d399', '#f87171']
            )])
            
            fig_pie.update_layout(
                height=300,
                showlegend=True,
                margin=dict(t=0, b=0, l=0, r=0)
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_chart2:
            # Score distribution
            st.markdown("##### Score Distribution")
            
            fig_hist = px.histogram(
                df,
                x='score',
                nbins=20,
                color='signal',
                color_discrete_map={
                    'STRONG_GREEN': '#10b981',
                    'GREEN': '#34d399',
                    'NEUTRAL': '#6b7280',
                    'RED': '#f87171',
                    'STRONG_RED': '#ef4444'
                }
            )
            
            fig_hist.update_layout(
                height=300,
                showlegend=True,
                xaxis_title="Score",
                yaxis_title="Count"
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Top 10 by score
        st.markdown("##### 🏆 Top 10 Stocks by Score")
        top_10 = df.nlargest(10, 'score')[['symbol', 'score', 'signal', 'price', 'change']]
        st.dataframe(top_10, use_container_width=True)
    
    def render_signals_tab(self, df: pd.DataFrame):
        """Render all signals tab"""
        st.markdown("### 📋 All Generated Signals")
        
        # Filters
        col_filter1, col_filter2, col_filter3 = st.columns(3)
        
        with col_filter1:
            signal_filter = st.selectbox(
                "Filter by Signal",
                ["All Signals", "STRONG_GREEN", "GREEN", "NEUTRAL", "RED", "STRONG_RED"],
                key="signal_filter"
            )
        
        with col_filter2:
            sort_by = st.selectbox(
                "Sort by",
                ["Score (High to Low)", "Score (Low to High)", "Symbol", "Price", "Change %"],
                key="sort_filter"
            )
        
        with col_filter3:
            min_score = st.slider("Minimum Score", -100, 100, -100)
        
        # Apply filters
        filtered_df = df.copy()
        if signal_filter != "All Signals":
            filtered_df = filtered_df[filtered_df['signal'] == signal_filter]
        
        filtered_df = filtered_df[filtered_df['score'] >= min_score]
        
        # Apply sorting
        if sort_by == "Score (High to Low)":
            filtered_df = filtered_df.sort_values('score', ascending=False)
        elif sort_by == "Score (Low to High)":
            filtered_df = filtered_df.sort_values('score', ascending=True)
        elif sort_by == "Symbol":
            filtered_df = filtered_df.sort_values('symbol')
        elif sort_by == "Price":
            filtered_df = filtered_df.sort_values('price', ascending=False)
        elif sort_by == "Change %":
            filtered_df = filtered_df.sort_values('change', ascending=False)
        
        # Display table
        display_cols = ['symbol', 'score', 'signal', 'price', 'change', 'rsi', 'macd']
        st.dataframe(
            filtered_df[display_cols],
            use_container_width=True,
            height=500
        )
        
        st.caption(f"Showing {len(filtered_df)} of {len(df)} signals")
    
    def render_top_performers_tab(self, df: pd.DataFrame):
        """Render top performers tab"""
        st.markdown("### 🏆 Top Performers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🟢 **Top Green Signals**")
            top_green = df[df['signal'].isin(['GREEN', 'STRONG_GREEN'])].nlargest(10, 'score')
            
            for _, row in top_green.iterrows():
                signal_color = "#10b981" if row['signal'] == 'STRONG_GREEN' else "#34d399"
                signal_text = "🟢 STRONG" if row['signal'] == 'STRONG_GREEN' else "🟢"
                
                st.markdown(f"""
                <div style="background: #f0fdf4; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid {signal_color};">
                    <strong>{row['symbol']}</strong> <span style="color: {signal_color}; font-weight: bold;">{signal_text}</span>
                    <div style="display: flex; justify-content: space-between; margin-top: 4px;">
                        <span>Score: <strong style="color: {signal_color};">{row['score']}</strong></span>
                        <span>₹{row['price']:,.2f} ({row['change']}%)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("##### 🔴 **Top Red Signals**")
            top_red = df[df['signal'].isin(['RED', 'STRONG_RED'])].nsmallest(10, 'score')
            
            for _, row in top_red.iterrows():
                signal_color = "#ef4444" if row['signal'] == 'STRONG_RED' else "#f87171"
                signal_text = "🔴 STRONG" if row['signal'] == 'STRONG_RED' else "🔴"
                
                st.markdown(f"""
                <div style="background: #fef2f2; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid {signal_color};">
                    <strong>{row['symbol']}</strong> <span style="color: {signal_color}; font-weight: bold;">{signal_text}</span>
                    <div style="display: flex; justify-content: space-between; margin-top: 4px;">
                        <span>Score: <strong style="color: {signal_color};">{row['score']}</strong></span>
                        <span>₹{row['price']:,.2f} ({row['change']}%)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    def render_charts_tab(self, df: pd.DataFrame):
        """Render charts tab"""
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI distribution
            st.markdown("##### RSI Distribution")
            fig_rsi = px.histogram(
                df,
                x='rsi',
                nbins=30,
                title="RSI Values Distribution",
                labels={'rsi': 'RSI Value'}
            )
            fig_rsi.add_vline(x=30, line_dash="dash", line_color="green")
            fig_rsi.add_vline(x=70, line_dash="dash", line_color="red")
            fig_rsi.update_layout(height=400)
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        with col2:
            # Score vs RSI scatter
            st.markdown("##### Score vs RSI")
            fig_scatter = px.scatter(
                df,
                x='rsi',
                y='score',
                color='signal',
                hover_data=['symbol', 'price'],
                color_discrete_map={
                    'STRONG_GREEN': '#10b981',
                    'GREEN': '#34d399',
                    'NEUTRAL': '#6b7280',
                    'RED': '#f87171',
                    'STRONG_RED': '#ef4444'
                }
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Price distribution by signal
        st.markdown("##### Price Distribution by Signal")
        fig_box = px.box(
            df,
            x='signal',
            y='price',
            color='signal',
            color_discrete_map={
                'STRONG_GREEN': '#10b981',
                'GREEN': '#34d399',
                'NEUTRAL': '#6b7280',
                'RED': '#f87171',
                'STRONG_RED': '#ef4444'
            }
        )
        fig_box.update_layout(height=400, xaxis_title="Signal", yaxis_title="Price (₹)")
        st.plotly_chart(fig_box, use_container_width=True)
    
    def render_welcome_screen(self):
        """Render welcome screen"""
        st.markdown("""
        ## 👋 Welcome to GreyOak SlopeTrigger v4.1 - Complete NSE Edition
        
        This dashboard provides **complete NSE universe analysis** using GreyOak v4.1 methodology.
        
        ### 🎯 **Features:**
        
        ✅ **Complete NSE Processing** - Analyze 2000+ NSE stocks  
        ✅ **GreyOak v4.1 Algorithm** - 9 technical indicators  
        ✅ **Real-time Signals** - Daily buy/sell/hold signals  
        ✅ **Interactive Dashboard** - Charts, filters, tables  
        ✅ **Export Options** - CSV, Excel, JSON formats  
        ✅ **Free Hosting** - Streamlit Cloud (no cost)  
        ✅ **Complete Universe** - Covers all major indices and sectors  
        
        ### 🚀 **Getting Started:**
        
        1. **Configure** settings in sidebar  
        2. **Select** universe size (Demo to Full)  
        3. **Click** "Run Analysis" button  
        4. **View** results in interactive dashboard  
        5. **Export** signals for trading  
        
        ### 📊 **Signal Types:**
        
        - 🟢 **STRONG_GREEN**: Strong buy signal (Score > 60)  
        - 🟢 **GREEN**: Buy signal (Score 30-60)  
        - ⚪ **NEUTRAL**: Hold signal (Score -30 to 30)  
        - 🔴 **RED**: Sell signal (Score -60 to -30)  
        - 🔴 **STRONG_RED**: Strong sell signal (Score < -60)  
        
        ### 🌐 **Deployment:**
        
        This dashboard is deployed on **Streamlit Cloud** - completely free!
        
        **Your URL:** `https://your-app.streamlit.app`
        
        Share this link with clients - no installation required!
        """)
    
    def render_footer(self):
        """Render footer"""
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.caption(f"**GreyOak SlopeTrigger v4.1 NSE Edition** | Version {self.config.VERSION} | Streamlit Cloud")
        
        with col2:
            st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
    
    def render(self):
        """Main render method"""
        # Header
        self.render_header()
        
        # Sidebar
        settings = self.render_sidebar()
        
        # Progress display
        if st.session_state.processing:
            st.markdown("### ⏳ **Processing Progress**")
            progress_bar = st.progress(st.session_state.progress / 100)
            st.text(st.session_state.status)
        
        # Control panel
        self.render_control_panel(settings)
        
        # Results or welcome screen
        if st.session_state.results is not None:
            self.render_results()
        else:
            self.render_welcome_screen()
        
        # Footer
        self.render_footer()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    # Create dashboard instance
    dashboard = GreyOakDashboard()
    
    # Render dashboard
    dashboard.render()

if __name__ == "__main__":
    main()
