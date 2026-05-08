###############################################################################
# Global ETF Price Discovery Scanner v5.0 (Naive / Anti-Overfitting Redesign)
# ============================================================================
# DESIGN PHILOSOPHY:
#   - "Naive" = simple, transparent, no optimization, no magic numbers
#   - 3 ORTHOGONAL SIGNAL AXES directly mapping to 3 investment objectives:
#     1. TCS (Trend Continuation Score): 모멘텀 지속 중 + 향후 지속 가능성
#     2. TFS (Trend Formation Score):  모멘텀 형성 초기 단계 포착
#     3. OER (Overextension Risk):     역추세 리스크에 노출된 종목 식별
#   - RSS (Relative Strength Score): 벤치마크 대비 상대강도 (보조 축)
#   - CLASSIFICATION directly reflects which objective each ETF matches
#   - Cross-sectional PERCENTILE RANK for fair comparison across asset classes
#   - NO smooth_score, NO adaptive boundaries, NO complex transformations
#
# PRESERVED from v4.x:
#   - 180+ ETF universe, Multi-benchmark, Real-time price
#   - PDF export, 7-day history, Master Summary, 1W/1M/3M/Custom comparison
#   - Signal Validity Verification (adapted to new axes)
#
# ADDED in current version:
#   - 1-Week Class Trend Tracking for CONTINUATION candidates.
#   - Momentum Exhaustion Filter (🟤 EXHAUSTING)
###############################################################################

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import json
import pickle
import os

try:
    from graph_engine import PriceDiscoveryGraph
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False

try:
    from hedge_strategies import (
        score_all_strategies, compute_combined_signal, compute_regime,
        compute_category_stats,
    )
    HAS_HEDGE = True
except ImportError:
    HAS_HEDGE = False

HAS_FACTOR_EFFICACY = False  # deferred import to avoid circular dependency

print("=" * 80)
print("  Global ETF Price Discovery Scanner v5.0 (Naive Redesign)")
print("  3-Axis Orthogonal Signal Architecture:")
print("   Axis 1 — TCS: Trend Continuation Score (established momentum)")
print("   Axis 2 — TFS: Trend Formation Score   (early/new momentum)")
print("   Axis 3 — OER: Overextension Risk       (mean-reversion exposure)")
print("   Aux   — RSS: Multi-Horizon Momentum Percentile (no benchmark)")
print("  Classification: CONTINUATION / FORMATION / OVEREXTENDED / EXHAUSTING / NEUTRAL / DOWNTREND")
print("=" * 80)


###############################################################################
# SECTION 0: UTILITIES
###############################################################################

def sf(value, default=0.0):
    """Safe float conversion."""
    try:
        if isinstance(value, (pd.Series, pd.DataFrame)):
            value = value.iloc[0] if hasattr(value, 'iloc') and len(value) == 1 else default
        r = float(value)
        return r if np.isfinite(r) else default
    except: return default

def ss(data):
    """Safe series extraction."""
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0] if data.shape[1] >= 1 else pd.Series(dtype=float)
    return data if isinstance(data, pd.Series) else pd.Series(data)

def fmt_data_as_of(df):
    if df is None or df.empty: return "N/A"
    ts = pd.Timestamp(df.index[-1])
    return ts.strftime('%Y-%m-%d %H:%M') if (ts.hour or ts.minute) else ts.strftime('%Y-%m-%d')

def pct_rank(value, arr):
    """Simple percentile rank of value within array. Returns 0-100."""
    arr = np.array(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2: return 50.0
    return float(np.sum(arr < value)) / (len(arr) - 1) * 100

def compute_rsi(close, period=14):
    """Standard RSI-14."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period, min_periods=period).mean()
    last_gain, last_loss = sf(gain.iloc[-1]), sf(loss.iloc[-1], 1e-10)
    if last_loss < 1e-10: return 100.0
    return 100.0 - 100.0 / (1.0 + last_gain / last_loss)


###############################################################################
# SECTION 1: ETF UNIVERSE + BENCHMARK CONFIG
###############################################################################

CATEGORY_BENCHMARK = {
    "EQ_Broad": "SPY", "EQ_Technology": "XLK", "EQ_Healthcare": "XLV",
    "EQ_Financials": "XLF", "EQ_ConsDisc": "XLY", "EQ_ConsStaples": "XLP",
    "EQ_Industrials": "XLI", "EQ_Energy": "XLE", "EQ_Materials": "XLB",
    "EQ_Utilities": "XLU", "EQ_RealEstate": "XLRE", "EQ_CommServices": "XLC",
    "EQ_Factor": "SPY", "EQ_Thematic": "QQQ",
    "Intl_Developed": "VEA", "Emerging_Markets": "EEM",
    "FI_Short": "SHY", "FI_Intermediate": "AGG", "FI_Long": "TLT",
    "FI_Credit": "HYG", "FI_Inflation": "TIP", "FI_International": "BNDX",
    "Commodities": "DBC", "Real_Assets": "VNQ",
    "Korea_Equity": "069500.KS", "Currency_Vol": "UUP", "Multi_Asset": "AOR",
}

# Multiple benchmarks per category — excess return = median across alternatives
# Reduces hindsight bias from single benchmark selection
CATEGORY_BENCHMARK_ALT = {
    "EQ_Broad": ["SPY", "RSP", "VTI"],
    "EQ_Technology": ["XLK", "QQQ"],
    "EQ_Healthcare": ["XLV", "IBB"],
    "EQ_Financials": ["XLF", "KRE"],
    "EQ_ConsDisc": ["XLY", "SPY"],
    "EQ_ConsStaples": ["XLP", "SPY"],
    "EQ_Industrials": ["XLI", "SPY"],
    "EQ_Energy": ["XLE", "SPY"],
    "EQ_Materials": ["XLB", "SPY"],
    "EQ_Utilities": ["XLU", "SPY"],
    "EQ_RealEstate": ["XLRE", "VNQ"],
    "EQ_CommServices": ["XLC", "SPY"],
    "EQ_Factor": ["SPY", "MTUM"],
    "EQ_Thematic": ["QQQ", "SPY"],
    "Intl_Developed": ["VEA", "EFA", "SPDW"],
    "Emerging_Markets": ["EEM", "VWO", "IEMG"],
    "FI_Short": ["SHY", "VCSH"],
    "FI_Intermediate": ["AGG", "BND"],
    "FI_Long": ["TLT", "IEF"],
    "FI_Credit": ["HYG", "USHY"],
    "FI_Inflation": ["TIP", "VTIP"],
    "FI_International": ["BNDX", "EMB"],
    "Commodities": ["DBC", "GSG", "PDBC"],
    "Real_Assets": ["VNQ", "VNQI"],
    "Korea_Equity": ["069500.KS", "102110.KS"],
    "Currency_Vol": ["UUP"],
    "Multi_Asset": ["AOR", "AOA"],
    "STK_Technology": ["XLK", "QQQ"],
    "STK_CommServices": ["XLC", "QQQ"],
    "STK_Healthcare": ["XLV", "IBB"],
    "STK_Financials": ["XLF", "KRE"],
    "STK_ConsDisc": ["XLY", "SPY"],
    "STK_ConsStaples": ["XLP", "SPY"],
    "STK_Industrials": ["XLI", "SPY"],
    "STK_Energy": ["XLE", "SPY"],
    "STK_Materials": ["XLB", "SPY"],
    "STK_Utilities": ["XLU", "SPY"],
    "STK_RealEstate": ["XLRE", "VNQ"],
    "STK_Korea": ["069500.KS", "102110.KS"],
    "STK_Japan": ["EWJ", "BBJP"],
    "STK_China_ADR": ["KWEB", "FXI"],
    "STK_Europe": ["VGK", "EZU"],
    "STK_India": ["INDA"],
}

GLOBAL_ETF_UNIVERSE = {
    "EQ_Broad": {"tickers": {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow Jones 30", "RSP": "S&P 500 Equal Weight", "IWF": "Russell 1000 Growth", "IWD": "Russell 1000 Value", "VUG": "Large Cap Growth", "VTV": "Large Cap Value", "IJH": "S&P MidCap 400", "IWM": "Russell 2000", "IJR": "S&P SmallCap 600", "VBR": "Small Cap Value", "VBK": "Small Cap Growth", "MGK": "Mega Cap Growth", "MGV": "Mega Cap Value", "OEF": "S&P 100 Mega Cap"}},
    "EQ_Technology": {"tickers": {"XLK": "Technology", "SMH": "Semiconductors", "SOXX": "Semiconductor Index", "CIBR": "Cybersecurity", "HACK": "Cybersecurity", "BOTZ": "Robotics & AI"}},
    "EQ_Healthcare": {"tickers": {"XLV": "Health Care", "IBB": "Biotech", "XBI": "Biotech Equal Weight"}},
    "EQ_Financials": {"tickers": {"XLF": "Financials", "KRE": "Regional Banks"}},
    "EQ_ConsDisc": {"tickers": {"XLY": "Consumer Discretionary", "ITB": "Home Construction", "XHB": "Homebuilders"}},
    "EQ_ConsStaples": {"tickers": {"XLP": "Consumer Staples"}},
    "EQ_Industrials": {"tickers": {"XLI": "Industrials", "ITA": "Aerospace & Defense", "PAVE": "Infrastructure"}},
    "EQ_Energy": {"tickers": {"XLE": "Energy"}},
    "EQ_Materials": {"tickers": {"XLB": "Materials"}},
    "EQ_Utilities": {"tickers": {"XLU": "Utilities"}},
    "EQ_RealEstate": {"tickers": {"XLRE": "Real Estate"}},
    "EQ_CommServices": {"tickers": {"XLC": "Communication Services"}},
    "EQ_Factor": {"tickers": {"MTUM": "Momentum", "QUAL": "Quality", "USMV": "Min Volatility", "VLUE": "Value Factor", "SIZE": "Size Factor", "SCHD": "Dividend Growth", "VIG": "Dividend Appreciation", "DVY": "High Dividend Yield", "NOBL": "Dividend Aristocrats", "COWZ": "Free Cash Flow Yield", "MOAT": "Wide Moat", "SPHQ": "S&P 500 Quality", "SPMO": "S&P 500 Momentum", "DYNF": "Dynamic Multi-Factor"}},
    "EQ_Thematic": {"tickers": {"AIQ": "AI & Big Data", "ROBO": "Robotics & Automation", "ARKG": "Genomic Revolution", "ARKW": "Next Gen Internet", "DRIV": "Autonomous & EV", "UFO": "Space", "SKYY": "Cloud Computing", "FINX": "Fintech", "EDOC": "Telemedicine", "QCLN": "Clean Edge Green Energy", "BATT": "Battery Technology", "REMX": "Rare Earth Metals", "XSD": "Semiconductor SPDR", "IGV": "Software", "CLOU": "Cloud Computing", "SHLD": "Global Defense", "463250.KS": "TIGER K방산", "ARKK": "Disruptive Innovation", "TAN": "Solar Energy", "ICLN": "Clean Energy", "LIT": "Lithium & Battery"}},
    "Intl_Developed": {"tickers": {"VEA": "FTSE Developed ex-US", "EFA": "MSCI EAFE", "IEFA": "MSCI EAFE Core", "SPDW": "S&P Developed ex-US", "VGK": "FTSE Europe", "EZU": "Eurozone", "HEDJ": "Europe Hedged", "FEZ": "Euro Stoxx 50", "EWJ": "MSCI Japan", "BBJP": "Japan BetaBuilders", "DXJ": "Japan Hedged Equity", "EWG": "Germany", "EWU": "United Kingdom", "EWQ": "France", "EWL": "Switzerland", "EWA": "Australia", "EWC": "Canada", "EIS": "Israel"}},
    "Emerging_Markets": {"tickers": {"VWO": "FTSE Emerging Markets", "EEM": "MSCI Emerging Markets", "IEMG": "MSCI EM Core", "EMXC": "EM ex-China", "EWZ": "Brazil", "EWT": "Taiwan", "EWY": "South Korea", "KORU": "South Korea Bull 3X", "INDA": "India", "FXI": "China Large-Cap", "KWEB": "China Internet", "MCHI": "MSCI China", "EWW": "Mexico", "THD": "Thailand", "VNM": "Vietnam", "EIDO": "Indonesia", "TUR": "Turkey", "EZA": "South Africa", "GXG": "Colombia", "ECH": "Chile"}},
    "FI_Short": {"tickers": {"SHY": "1-3Y Treasury", "VCSH": "Short-Term Corp", "JAAA": "AAA CLO", "SGOV": "0-3M Treasury", "BIL": "1-3M T-Bill", "SHV": "0-1Y Treasury", "FLOT": "Floating Rate Bond", "USFR": "Floating Rate Treasury", "JPST": "Ultra-Short Income"}},
    "FI_Intermediate": {"tickers": {"IEI": "3-7Y Treasury", "VCIT": "Intermediate Corp", "AGG": "US Aggregate Bond", "BND": "Total Bond Market", "MBB": "MBS", "JBBB": "BB-B CLO", "IGIB": "5-10Y IG Corp", "GOVT": "US Treasury Full Curve"}},
    "FI_Long": {"tickers": {"IEF": "7-10Y Treasury", "TLH": "10-20Y Treasury", "TLT": "20+Y Treasury", "LQD": "Investment Grade Corp", "EDV": "Extended Duration STRIPS", "ZROZ": "25+Y Zero Coupon", "VGLT": "Long-Term Treasury", "VCLT": "Long-Term Corp"}},
    "FI_Credit": {"tickers": {"HYG": "High Yield Corp", "USHY": "Broad High Yield", "PFF": "Preferred Stock", "ANGL": "Fallen Angel HY", "BKLN": "Senior Loan", "SRLN": "Blackstone Senior Loan", "FALN": "iShares Fallen Angels"}},
    "FI_Inflation": {"tickers": {"TIP": "TIPS Bond", "VTIP": "Short-Term TIPS", "SCHP": "Schwab TIPS", "STIP": "0-5Y TIPS", "LTPZ": "15+Y TIPS"}},
    "FI_International": {"tickers": {"BNDX": "Total Intl Bond", "IAGG": "Intl Aggregate", "EMB": "EM Bonds USD", "CEMB": "EM HC Bonds", "LEMB": "EM LC Bonds", "VWOB": "EM Government Bond", "IGOV": "Intl Treasury", "BWX": "SPDR Intl Treasury"}},
    "Commodities": {"tickers": {"GLD": "Gold", "SLV": "Silver", "GDX": "Gold Miners", "GDXJ": "Junior Gold Miners", "USO": "Crude Oil (WTI)", "BNO": "Brent Crude Oil", "UNG": "Natural Gas", "PPLT": "Platinum", "PALL": "Palladium", "DBA": "Agriculture", "DBC": "Commodity Index", "GSG": "S&P GSCI Commodity", "XOP": "S&P Oil and Exploration", "COPX": "Copper Miners", "WEAT": "Wheat", "CORN": "Corn", "URA": "Uranium", "CPER": "Copper", "SOYB": "Soybeans", "NIGS": "Sugar", "CANE": "Sugar", "PICK": "Metal Mining", "SIL": "Silver Miners", "SILJ": "Junior Silver Miners", "URNM": "Uranium Miners", "NUKZ": "Nuclear Energy", "XME": "S&P Metals & Mining", "GUNR": "Natural Resources", "MOO": "Agribusiness", "FTGC": "Commodity Strategy", "PDBC": "Optimum Yield Commodity", "COMT": "Commodity Strategy Broad", "FCG": "Natural Gas E&P", "OIH": "Oil Services", "GLDM": "Gold Mini", "SGOL": "Gold Physical", "IAU": "Gold Trust", "AAAU": "Goldman Gold", "BAR": "Gold Shares"}},
    "Real_Assets": {"tickers": {"VNQ": "US Real Estate", "VNQI": "Intl Real Estate", "IYR": "US Real Estate Broad", "REM": "Mortgage REITs", "AMLP": "MLP Energy", "MLPX": "MLP Energy", "IFRA": "Infrastructure", "WOOD": "Timber & Forestry", "IBIT": "Bitcoin", "ETHA": "Ethereum"}},
    "Korea_Equity": {"tickers": {"069500.KS": "KODEX 200", "229200.KS": "KODEX 코스닥150", "091160.KS": "KODEX 반도체", "487240.KS": "AI핵심전력설비","305720.KS": "KODEX 2차전지", "102110.KS": "TIGER 200", "396500.KS": "TIGER 반도체TOP10", "292150.KS": "TIGER 코리아TOP10", "381170.KS": "TIGER 미국테크TOP10", "381180.KS": "TIGER 미국필라델피아반도체나스닥", "466920.KS": "SOL 조선TOP3플러스", "395160.KS": "KODEX AI반도체", "161510.KS": "PLUS 고배당주"}},
    "Currency_Vol": {"tickers": {"UUP": "US Dollar Bullish", "FXE": "Euro", "FXY": "Japanese Yen", "FXB": "British Pound", "FXA": "Australian Dollar", "CYB": "Chinese Yuan", "VIXY": "VIX Short-Term"}},
    "Multi_Asset": {"tickers": {"AOR": "Growth Allocation", "AOA": "Aggressive Allocation", "AOM": "Moderate Allocation", "AOK": "Conservative Allocation", "RPAR": "Risk Parity", "GAA": "Global Asset Allocation"}}
}

###############################################################################
# SECTION 1-B: INDIVIDUAL STOCK UNIVERSE (extensible structure)
###############################################################################

STOCK_UNIVERSE = {
    # ══════════════════════════════════════════════════════════════════════
    # 1. Technology — 반도체 / 소프트웨어 / 하드웨어 / 네트워킹 / 스토리지
    #    (Mag7 tech + Semicon + Software tech + AI Infra tech)
    # ══════════════════════════════════════════════════════════════════════
    "STK_Technology": {"tickers": {
        # ── ex-Mag7 Technology ──
        "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA",
        # ── Semiconductors (전체) ──
        "TSM": "TSMC", "AVGO": "Broadcom", "ASML": "ASML",
        "AMD": "AMD", "QCOM": "Qualcomm", "TXN": "Texas Instruments",
        "ARM": "Arm Holdings", "AMAT": "Applied Materials",
        "LRCX": "Lam Research", "MU": "Micron",
        "KLAC": "KLA Corp", "ADI": "Analog Devices",
        "INTC": "Intel", "SNPS": "Synopsys",
        "CDNS": "Cadence Design", "MRVL": "Marvell Technology",
        "NXPI": "NXP Semiconductors", "MPWR": "Monolithic Power",
        "ON": "ON Semiconductor", "MCHP": "Microchip Technology",
        "GFS": "GlobalFoundries", "STM": "STMicroelectronics",
        "ENTG": "Entegris", "SWKS": "Skyworks Solutions",
        "UMC": "United Microelectronics", "CRUS": "Cirrus Logic",
        "ONTO": "Onto Innovation", "RMBS": "Rambus",
        "LSCC": "Lattice Semiconductor", "MBLY": "Mobileye",
        "WOLF": "Wolfspeed", "ACLS": "Axcelis Technologies",
        "MKSI": "MKS Instruments", "SLAB": "Silicon Labs",
        "ALGM": "Allegro MicroSystems",
        # ── Software / Enterprise / Cybersecurity (GICS: Technology) ──
        "ORCL": "Oracle", "SAP": "SAP", "CRM": "Salesforce",
        "PLTR": "Palantir", "INTU": "Intuit", "NOW": "ServiceNow",
        "ADBE": "Adobe", "SHOP": "Shopify",
        "APP": "AppLovin", "PANW": "Palo Alto Networks",
        "CRWD": "CrowdStrike", "FTNT": "Fortinet",
        "WDAY": "Workday", "SNOW": "Snowflake",
        "TEAM": "Atlassian", "DDOG": "Datadog",
        "FICO": "Fair Isaac", "ZS": "Zscaler",
        "NET": "Cloudflare", "HUBS": "HubSpot",
        "VEEV": "Veeva Systems", "ANSS": "Ansys",
        "COIN": "Coinbase", "MDB": "MongoDB",
        "BILL": "BILL Holdings", "TWLO": "Twilio",
        "OKTA": "Okta", "PATH": "UiPath",
        "DOCU": "DocuSign", "MNDY": "Monday.com",
        "S": "SentinelOne", "SE": "Sea Limited",
        "GRAB": "Grab Holdings", "ESTC": "Elastic",
        "IOT": "Samsara", "GEN": "Gen Digital",
        # ── AI Infra — Technology portion (네트워킹 / 스토리지 / 커넥터 / 하드웨어) ──
        "ANET": "Arista Networks", "APH": "Amphenol",
        "CSCO": "Cisco", "DELL": "Dell Technologies",
        "HPE": "HP Enterprise", "KEYS": "Keysight Technologies",
        "NTAP": "NetApp", "NTNX": "Nutanix",
        "PSTG": "Pure Storage", "SMCI": "Super Micro Computer",
        "STX": "Seagate", "WDC": "Western Digital",
        "GLW": "Corning", "LITE": "Lumentum", "COHR": "Coherent",
        "CLS": "Celestica", "FLEX": "Flex", "TEL": "TE Connectivity",
        # ── Samsung Q.Pack additions ──
        "ACN": "Accenture", "IBM": "IBM",
        "MSI": "Motorola Solutions", "SNDK": "SanDisk",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 2. Communication Services — 디지털 광고 / 미디어 / 엔터테인먼트
    #    (Mag7 comm + Software social/media tickers)
    # ══════════════════════════════════════════════════════════════════════
    "STK_CommServices": {"tickers": {
        "GOOGL": "Alphabet", "META": "Meta Platforms",
        "PINS": "Pinterest", "SNAP": "Snap", "SPOT": "Spotify",
        "RBLX": "Roblox", "ROKU": "Roku", "TTD": "Trade Desk",
        # ── Samsung Q.Pack additions ──
        "CMCSA": "Comcast", "DIS": "Walt Disney",
        "GOOG": "Alphabet Class C", "NFLX": "Netflix",
        "T": "AT&T", "TMUS": "T-Mobile US", "VZ": "Verizon",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 3. Healthcare — 제약 / 바이오 / 메드테크 / 보험 / 서비스
    # ══════════════════════════════════════════════════════════════════════
    "STK_Healthcare": {"tickers": {
        "LLY": "Eli Lilly", "UNH": "UnitedHealth",
        "NVO": "Novo Nordisk", "JNJ": "Johnson & Johnson",
        "ABBV": "AbbVie", "MRK": "Merck",
        "AZN": "AstraZeneca", "TMO": "Thermo Fisher",
        "ABT": "Abbott Labs", "ISRG": "Intuitive Surgical",
        "DHR": "Danaher", "AMGN": "Amgen",
        "SYK": "Stryker", "PFE": "Pfizer",
        "BSX": "Boston Scientific", "VRTX": "Vertex Pharma",
        "GILD": "Gilead Sciences", "MDT": "Medtronic",
        "BMY": "Bristol-Myers Squibb", "REGN": "Regeneron",
        "CI": "Cigna Group", "HCA": "HCA Healthcare",
        "ZTS": "Zoetis", "MCK": "McKesson",
        "BDX": "Becton Dickinson", "EW": "Edwards Lifesciences",
        "GEHC": "GE HealthCare", "A": "Agilent Technologies",
        "IQV": "IQVIA", "IDXX": "IDEXX Labs",
        # ── 추가 ──
        "MRNA": "Moderna", "BIIB": "Biogen",
        "ALNY": "Alnylam Pharma", "ALGN": "Align Technology",
        "HOLX": "Hologic", "RMD": "ResMed",
        "PODD": "Insulet Corp", "INCY": "Incyte",
        "ILMN": "Illumina", "DXCM": "DexCom",
        "WAT": "Waters Corp", "CNC": "Centene",
        "MOH": "Molina Healthcare", "TECH": "Bio-Techne",
        # ── Samsung Q.Pack additions ──
        "CVS": "CVS Health", "NVS": "Novartis",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 6. Financials — 은행 / 결제 / 보험 / 거래소 / 자산운용 / PE
    # ══════════════════════════════════════════════════════════════════════
    "STK_Financials": {"tickers": {
        "BRK-B": "Berkshire Hathaway", "JPM": "JPMorgan Chase",
        "V": "Visa", "MA": "Mastercard",
        "BAC": "Bank of America", "WFC": "Wells Fargo",
        "AXP": "American Express", "GS": "Goldman Sachs",
        "MS": "Morgan Stanley", "SPGI": "S&P Global",
        "BLK": "BlackRock", "PGR": "Progressive",
        "C": "Citigroup", "SCHW": "Charles Schwab",
        "FI": "Fiserv", "CB": "Chubb",
        "MRSH": "Marsh McLennan", "ICE": "Intercontinental Exchange",
        "MCO": "Moody's", "CME": "CME Group",
        "AON": "Aon", "PYPL": "PayPal",
        "PNC": "PNC Financial", "USB": "US Bancorp",
        "COF": "Capital One", "TRV": "Travelers",
        "MET": "MetLife", "AFL": "Aflac",
        "ALL": "Allstate", "MSCI": "MSCI Inc",
        # ── 추가 ──
        "KKR": "KKR & Co", "APO": "Apollo Global",
        "ARES": "Ares Management", "TROW": "T. Rowe Price",
        "RJF": "Raymond James", "NDAQ": "Nasdaq Inc",
        "MKTX": "MarketAxess", "WTW": "Willis Towers Watson",
        "FITB": "Fifth Third Bancorp", "MTB": "M&T Bank",
        "HBAN": "Huntington Bancshares", "SYF": "Synchrony Financial",
        "HOOD": "Robinhood",
        # ── Samsung Q.Pack additions ──
        "BAM": "Brookfield Asset Mgmt", "BCS": "Barclays",
        "BK": "Bank of New York Mellon", "BMO": "Bank of Montreal",
        "BN": "Brookfield Corp", "BNS": "Bank of Nova Scotia",
        "BX": "Blackstone", "CM": "Canadian Imperial Bank",
        "HSBC": "HSBC Holdings", "IBKR": "Interactive Brokers",
        "ITUB": "Itau Unibanco", "LYG": "Lloyds Banking",
        "RY": "Royal Bank of Canada", "TD": "Toronto-Dominion Bank",
        "UBS": "UBS Group",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 5. Consumer Discretionary — 리테일 / 외식 / 레저 / 럭셔리 / 자동차
    #    (Mag7 consumer disc + STK_Consumer discretionary portion)
    # ══════════════════════════════════════════════════════════════════════
    "STK_ConsDisc": {"tickers": {
        # ── ex-Mag7 Consumer Discretionary ──
        "AMZN": "Amazon", "TSLA": "Tesla",
        # ── Retail / Travel / Leisure / Luxury ──
        "HD": "Home Depot",
        "LVMUY": "LVMH", "BKNG": "Booking Holdings",
        "LOW": "Lowe's", "TJX": "TJX Companies",
        "NKE": "Nike", "SBUX": "Starbucks",
        "CMG": "Chipotle", "ABNB": "Airbnb",
        "ORLY": "O'Reilly Auto", "AZO": "AutoZone",
        "RCL": "Royal Caribbean", "TGT": "Target",
        "ROST": "Ross Stores", "YUM": "Yum! Brands",
        "HLT": "Hilton", "LULU": "Lululemon",
        "MCD": "McDonald's", "EL": "Estee Lauder",
        "MELI": "MercadoLibre", "CPNG": "Coupang",
        "DPZ": "Domino's Pizza", "DECK": "Deckers Outdoor",
        "ULTA": "Ulta Beauty", "MAR": "Marriott Intl",
        "DKNG": "DraftKings", "CCL": "Carnival Corp",
        "LVS": "Las Vegas Sands", "MGM": "MGM Resorts",
        "WYNN": "Wynn Resorts", "ETSY": "Etsy",
        "TPR": "Tapestry", "GRMN": "Garmin",
        # ── GICS reclassification ──
        "DASH": "DoorDash",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 6. Consumer Staples — 음료 / 가정용품 / 식품 / 담배
    # ══════════════════════════════════════════════════════════════════════
    "STK_ConsStaples": {"tickers": {
        "WMT": "Walmart", "PG": "Procter & Gamble",
        "KO": "Coca-Cola", "PEP": "PepsiCo",
        "PM": "Philip Morris Intl", "CL": "Colgate-Palmolive",
        "MDLZ": "Mondelez", "MNST": "Monster Beverage",
        "MO": "Altria", "KHC": "Kraft Heinz",
        # ── GICS reclassification + Samsung Q.Pack additions ──
        "COST": "Costco",
        "BTI": "British American Tobacco", "BUD": "Anheuser-Busch InBev",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 7. Industrials & Defense — 방산 / 항공 / 자본재 / 운송 / 폐기물
    #    (STK_Industrials + AI Infra industrial portion)
    # ══════════════════════════════════════════════════════════════════════
    "STK_Industrials": {"tickers": {
        "GE": "GE Aerospace", "CAT": "Caterpillar",
        "RTX": "RTX Corp", "UNP": "Union Pacific",
        "HON": "Honeywell", "LMT": "Lockheed Martin",
        "BA": "Boeing", "DE": "Deere",
        "UPS": "UPS", "WM": "Waste Management",
        "GD": "General Dynamics", "PH": "Parker-Hannifin",
        "CTAS": "Cintas", "TDG": "TransDigm",
        "NOC": "Northrop Grumman", "ITW": "Illinois Tool Works",
        "MMM": "3M", "RSG": "Republic Services",
        "CSX": "CSX Corp", "EMR": "Emerson Electric",
        "CARR": "Carrier Global", "FDX": "FedEx",
        "NSC": "Norfolk Southern", "AXON": "Axon Enterprise",
        "VRSK": "Verisk Analytics", "OTIS": "Otis Worldwide",
        "IR": "Ingersoll Rand", "ROK": "Rockwell Automation",
        "DOV": "Dover Corp", "WAB": "Wabtec",
        "HWM": "Howmet Aerospace", "HEI": "HEICO Corp",
        "TT": "Trane Technologies", "FAST": "Fastenal",
        "XYL": "Xylem", "LHX": "L3Harris Technologies",
        "LDOS": "Leidos", "BWXT": "BWX Technologies",
        "TXT": "Textron", "GWW": "W.W. Grainger",
        "SNA": "Snap-on", "NDSN": "Nordson",
        "J": "Jacobs Solutions",
        # ── ex-AI Infra Industrials portion ──
        "AME": "Ametek", "EME": "EMCOR Group",
        "ETN": "Eaton Corp", "FTV": "Fortive",
        "GNRC": "Generac", "HUBB": "Hubbell",
        "POWL": "Powell Industries", "PWR": "Quanta Services",
        "TDY": "Teledyne", "VLTO": "Veralto",
        "VRT": "Vertiv", "WCC": "WESCO Intl",
        "AAON": "AAON Inc",
        # ── GICS reclassification + Samsung Q.Pack additions ──
        "UBER": "Uber",
        "ADP": "Automatic Data Processing", "CMI": "Cummins",
        "CP": "Canadian Pacific Kansas City", "JCI": "Johnson Controls",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 8. Energy — 석유 / 가스 / 정유 / 미드스트림 / 우라늄
    # ══════════════════════════════════════════════════════════════════════
    "STK_Energy": {"tickers": {
        "XOM": "Exxon Mobil", "CVX": "Chevron",
        "SHEL": "Shell", "TTE": "TotalEnergies",
        "COP": "ConocoPhillips", "ENB": "Enbridge",
        "EOG": "EOG Resources", "SLB": "Schlumberger",
        "MPC": "Marathon Petroleum", "FANG": "Diamondback Energy",
        "PSX": "Phillips 66", "VLO": "Valero Energy",
        "OXY": "Occidental Petroleum", "BKR": "Baker Hughes",
        "HAL": "Halliburton", "DVN": "Devon Energy",
        "KMI": "Kinder Morgan", "WMB": "Williams Companies",
        "OKE": "ONEOK",
        # ── 우라늄 / 원자력 ──
        "CCJ": "Cameco (Uranium)", "UEC": "Uranium Energy Corp",
        "LEU": "Centrus Energy", "DNN": "Denison Mines",
        "NXE": "NexGen Energy", "UUUU": "Energy Fuels",
        # ── Samsung Q.Pack additions ──
        "CNQ": "Canadian Natural Resources", "EPD": "Enterprise Products Partners",
        "EQNR": "Equinor", "PBR": "Petrobras",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 9. Materials — 화학 / 금속 / 광업 / 농업자재 / 리튬
    # ══════════════════════════════════════════════════════════════════════
    "STK_Materials": {"tickers": {
        "LIN": "Linde", "SHW": "Sherwin-Williams",
        "BHP": "BHP Group", "FCX": "Freeport-McMoRan",
        "APD": "Air Products", "ECL": "Ecolab",
        "NEM": "Newmont Mining", "CTVA": "Corteva",
        "NUE": "Nucor", "DD": "DuPont",
        "DOW": "Dow Inc", "VMC": "Vulcan Materials",
        "MLM": "Martin Marietta", "CE": "Celanese",
        "PPG": "PPG Industries", "IFF": "Intl Flavors & Fragrances",
        "EMN": "Eastman Chemical", "MP": "MP Materials",
        "ALB": "Albemarle",
        # ── 광업 ──
        "RIO": "Rio Tinto", "VALE": "Vale SA",
        "GOLD": "Barrick Gold", "WPM": "Wheaton Precious Metals",
        "TECK": "Teck Resources",
        # ── 구리 / 비철금속 ──
        "SCCO": "Southern Copper", "HBM": "Hudbay Minerals",
        "ERO": "Ero Copper", "IVPAF": "Ivanhoe Mines",
        # ── 귀금속 (금/은) ──
        "AEM": "Agnico Eagle Mines", "FNV": "Franco-Nevada",
        "RGLD": "Royal Gold", "KGC": "Kinross Gold",
        "AGI": "Alamos Gold", "EGO": "Eldorado Gold",
        "PAAS": "Pan American Silver", "AG": "First Majestic Silver",
        "HL": "Hecla Mining", "MAG": "MAG Silver",
        # ── 리튬 / 배터리 원자재 ──
        "SQM": "Sociedad Quimica (Lithium)", "LAC": "Lithium Americas",
        "ALTM": "Arcadium Lithium", "PLL": "Piedmont Lithium",
        # ── 농업 / 비료 / 곡물 ──
        "NTR": "Nutrien", "MOS": "Mosaic Company",
        "CF": "CF Industries", "FMC": "FMC Corp",
        "ADM": "Archer-Daniels-Midland", "BG": "Bunge Global",
        "ANDE": "Andersons",
        # ── 철강 / 특수금속 ──
        "STLD": "Steel Dynamics", "CLF": "Cleveland-Cliffs",
        "RS": "Reliance Steel", "CMC": "Commercial Metals",
        "ATI": "ATI Inc (Specialty Alloys)",
        # ── Samsung Q.Pack additions ──
        "CRH": "CRH plc",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 10. Utilities — 전력 / 에너지 인프라 / 재생에너지
    #     (ex-AI Infra utilities portion)
    # ══════════════════════════════════════════════════════════════════════
    "STK_Utilities": {"tickers": {
        "CEG": "Constellation Energy", "GEV": "GE Vernova",
        "NRG": "NRG Energy", "VST": "Vistra Energy",
        "ENPH": "Enphase Energy",
        # ── Samsung Q.Pack additions ──
        "DUK": "Duke Energy", "NEE": "NextEra Energy",
        "NGG": "National Grid", "SO": "Southern Company",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 11. Real Estate — 데이터센터 REIT
    #     (ex-AI Infra real estate portion)
    # ══════════════════════════════════════════════════════════════════════
    "STK_RealEstate": {"tickers": {
        "DLR": "Digital Realty", "EQIX": "Equinix",
        # ── Samsung Q.Pack additions ──
        "AMT": "American Tower", "PLD": "Prologis", "WELL": "Welltower",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 12. Korea — KOSPI 대형주 + 핵심 중형주
    # ══════════════════════════════════════════════════════════════════════
    "STK_Korea": {"tickers": {
        "005930.KS": "Samsung Elec", "000660.KS": "SK Hynix",
        "207940.KS": "Samsung Biologics", "005380.KS": "현대자동차",
        "068270.KS": "셀트리온", "005490.KS": "POSCO홀딩스",
        "035420.KS": "NAVER", "006400.KS": "Samsung SDI",
        "000270.KS": "기아", "051910.KS": "LG Chem",
        "012450.KS": "한화에어로스페이스", "105560.KS": "KB금융",
        "055550.KS": "신한지주", "034730.KS": "SK",
        "035720.KS": "Kakao", "028260.KS": "삼성물산",
        "066570.KS": "LG전자", "032830.KS": "삼성생명",
        "086790.KS": "하나금융지주", "009150.KS": "삼성전기",
        "017670.KS": "SK텔레콤", "003550.KS": "LG",
        "033780.KS": "KT&G", "030200.KS": "KT",
        "010130.KS": "고려아연", "047050.KS": "포스코인터내셔널",
        "010120.KS": "LS Electric", "003670.KS": "포스코퓨처엠",
        "005935.KS": "Samsung Elec Pref", "018260.KS": "삼성SDS",
        # ── 추가 ──
        "373220.KS": "LG에너지솔루션", "352820.KS": "하이브",
        "000810.KS": "삼성화재", "259960.KS": "크래프톤",
        "402340.KS": "SK스퀘어", "011070.KS": "LG이노텍",
        "096770.KS": "SK이노베이션", "034020.KS": "두산에너빌리티",
        "078930.KS": "GS", "316140.KS": "우리금융지주",
        "036570.KS": "엔씨소프트", "011200.KS": "HMM",
        "267260.KS": "HD현대일렉트릭", "042700.KS": "한미반도체",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 13. Japan — ADR + 미국 상장 (yfinance 안정성 우선)
    # ══════════════════════════════════════════════════════════════════════
    "STK_Japan": {"tickers": {
        "TM": "Toyota Motor", "SONY": "Sony Group",
        "MUFG": "Mitsubishi UFJ Financial", "SMFG": "Sumitomo Mitsui Financial",
        "MFG": "Mizuho Financial", "NMR": "Nomura Holdings",
        "HMC": "Honda Motor", "NTT": "Nippon Telegraph & Tel",
        "IX": "ORIX Corp", "TAK": "Takeda Pharmaceutical",
        "SNE": "Sony (ADR alt)", "DSCSY": "Disco Corp",
        "NTDOY": "Nintendo", "KYOCY": "Kyocera",
        "FANUY": "Fanuc", "TOELY": "Tokyo Electron",
        "HXSCL": "Hamamatsu Photonics", "KNBWY": "Kobe Steel",
        "DNZOY": "Denso", "ITOCY": "ITOCHU",
        "MSBHF": "Mitsubishi Heavy Ind", "SBHSY": "SoftBank Group",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 14. China ADR — 미국 상장 중국 주식
    # ══════════════════════════════════════════════════════════════════════
    "STK_China_ADR": {"tickers": {
        "BABA": "Alibaba", "PDD": "PDD Holdings",
        "JD": "JD.com", "BIDU": "Baidu",
        "NIO": "NIO", "LI": "Li Auto",
        "XPEV": "XPeng", "NTES": "NetEase",
        "TCOM": "Trip.com", "BILI": "Bilibili",
        "ZTO": "ZTO Express", "TME": "Tencent Music",
        "YUMC": "Yum China", "MNSO": "Miniso",
        "TAL": "TAL Education", "FUTU": "Futu Holdings",
        "WB": "Weibo", "VIPS": "Vipshop",
        "BEKE": "KE Holdings", "GDS": "GDS Holdings",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 15. Europe — ADR / 미국 상장 유럽 주식
    # ══════════════════════════════════════════════════════════════════════
    "STK_Europe": {"tickers": {
        "UL": "Unilever", "GSK": "GSK plc",
        "SNY": "Sanofi", "DEO": "Diageo",
        "BP": "BP plc", "RACE": "Ferrari",
        "RELX": "RELX Group", "ING": "ING Groep",
        "PHG": "Philips", "ARGX": "argenx (EU Biotech)",
        "STLA": "Stellantis", "TEF": "Telefonica",
        "ERIC": "Ericsson", "NOK": "Nokia",
        "DB": "Deutsche Bank", "CS": "Credit Suisse",
        "SAN": "Banco Santander", "BBVA": "BBVA",
        "WPP": "WPP plc", "LOGI": "Logitech",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 16. India — ADR / 미국 상장 인도 주식
    # ══════════════════════════════════════════════════════════════════════
    "STK_India": {"tickers": {
        "INFY": "Infosys", "WIT": "Wipro",
        "HDB": "HDFC Bank", "IBN": "ICICI Bank",
        "RDY": "Dr Reddy's Labs", "TTM": "Tata Motors",
        "MMYT": "MakeMyTrip", "WNS": "WNS Holdings",
        "SIFY": "Sify Technologies",
    }},
}

STOCK_BENCHMARK = {
    "STK_Technology": "XLK",
    "STK_CommServices": "XLC",
    "STK_Healthcare": "XLV",
    "STK_Financials": "XLF",
    "STK_ConsDisc": "XLY",
    "STK_ConsStaples": "XLP",
    "STK_Industrials": "XLI",
    "STK_Energy": "XLE",
    "STK_Materials": "XLB",
    "STK_Utilities": "XLU",
    "STK_RealEstate": "XLRE",
    "STK_Korea": "069500.KS",
    "STK_Japan": "EWJ",
    "STK_China_ADR": "KWEB",
    "STK_Europe": "VGK",
    "STK_India": "INDA",
}

###############################################################################
# SECTION 1-C: STOCK THEME MAPPING (sub-theme within each category)
###############################################################################

STOCK_THEMES = {
    # ── Mag7 ──
    "AAPL": "Consumer Hardware", "MSFT": "Cloud Platform", "GOOGL": "Cloud Platform",
    "AMZN": "Cloud Platform", "NVDA": "AI Chip", "META": "Social/Ad Platform",
    "TSLA": "EV/Energy",
    # ── Semicon ──
    "TSM": "Foundry", "AVGO": "Logic Design", "ASML": "Equipment (Litho)",
    "AMD": "Logic Design", "QCOM": "Mobile/RF", "TXN": "Analog",
    "ARM": "IP/Design", "AMAT": "Equipment (Deposition)", "LRCX": "Equipment (Etch)",
    "MU": "Memory", "KLAC": "Equipment (Inspection)", "ADI": "Analog",
    "INTC": "Logic Design", "SNPS": "EDA", "CDNS": "EDA",
    "MRVL": "Logic Design", "NXPI": "Auto/IoT", "MPWR": "Power IC",
    "ON": "Power/Auto", "MCHP": "MCU/Embedded", "GFS": "Foundry",
    "STM": "Analog/Mixed", "ENTG": "Materials", "SWKS": "Mobile/RF",
    "UMC": "Foundry", "CRUS": "Audio IC", "ONTO": "Equipment (Inspection)",
    "RMBS": "IP/Design", "LSCC": "FPGA", "MBLY": "Auto Vision",
    # ── Software ──
    "ORCL": "Enterprise DB", "SAP": "Enterprise ERP", "CRM": "Enterprise CRM",
    "PLTR": "Data/AI Analytics", "INTU": "SMB/Finance SW", "NOW": "Enterprise Workflow",
    "ADBE": "Creative/Design", "SHOP": "E-commerce Platform", "UBER": "Transport/Ride-Hail",
    "APP": "AdTech/Mobile", "PANW": "Cybersecurity", "CRWD": "Cybersecurity",
    "FTNT": "Cybersecurity", "WDAY": "Enterprise HR", "TTD": "AdTech",
    "SNOW": "Data/AI Analytics", "DASH": "Delivery Platform", "TEAM": "DevOps/Collab",
    "DDOG": "Observability", "FICO": "Credit Analytics", "ZS": "Cybersecurity",
    "NET": "Edge/CDN", "HUBS": "Marketing SaaS", "VEEV": "Vertical SaaS (Pharma)",
    "ANSS": "Simulation/CAE", "COIN": "Crypto Exchange", "MDB": "Database",
    "RBLX": "Gaming Platform", "BILL": "SMB Fintech", "TWLO": "CPaaS/API",
    # ── AI Infra ──
    "CSCO": "Networking", "ETN": "Power/Transformer", "ANET": "Networking",
    "EQIX": "DC REIT", "APH": "Connector", "CEG": "Nuclear/Power",
    "GEV": "Power Equipment", "DELL": "Server/Hardware", "DLR": "DC REIT",
    "VRT": "Cooling/Thermal", "VST": "Power Generation", "TEL": "Connector",
    "PWR": "DC Construction", "GLW": "Optical Fiber", "AME": "Instruments",
    "KEYS": "Test/Measurement", "HPE": "Server/Hardware", "SMCI": "Server/Hardware",
    "VLTO": "Water/Process", "NTAP": "Storage", "PSTG": "Storage",
    "EME": "DC Construction", "HUBB": "Power/Transformer", "NRG": "Power Generation",
    "WDC": "Storage", "STX": "Storage", "FLEX": "Contract Mfg",
    "COHR": "Optical Transceiver", "LITE": "Optical Transceiver", "NTNX": "HCI/Cloud SW",
    # ── Healthcare ──
    "LLY": "Big Pharma (GLP-1)", "UNH": "Managed Care", "NVO": "Big Pharma (GLP-1)",
    "JNJ": "Diversified Pharma", "ABBV": "Big Pharma (Immuno)", "MRK": "Big Pharma (Oncology)",
    "AZN": "EU Pharma", "TMO": "Life Science Tools", "ABT": "Diagnostics/Devices",
    "ISRG": "Surgical Robotics", "DHR": "Life Science Tools", "AMGN": "Biotech",
    "SYK": "Orthopedics", "PFE": "Big Pharma (Vaccine)", "BSX": "Cardio Devices",
    "VRTX": "Biotech (Rare)", "GILD": "Biotech (Antiviral)", "MDT": "Cardio Devices",
    "BMY": "Big Pharma (Oncology)", "REGN": "Biotech (Immuno)", "CI": "Managed Care",
    "HCA": "Hospital", "ZTS": "Animal Health", "MCK": "Distribution",
    "BDX": "Medical Supplies", "EW": "Cardio Devices", "GEHC": "Imaging/Diagnostics",
    "A": "Life Science Tools", "IQV": "Clinical Trials/CRO", "IDXX": "Veterinary Dx",
    # ── Financials ──
    "BRK-B": "Conglomerate", "JPM": "Mega Bank", "V": "Payments",
    "MA": "Payments", "BAC": "Mega Bank", "WFC": "Mega Bank",
    "AXP": "Payments", "GS": "Investment Bank", "MS": "Investment Bank",
    "SPGI": "Data/Ratings", "BLK": "Asset Management", "PGR": "Insurance (P&C)",
    "C": "Mega Bank", "SCHW": "Broker/Wealth", "FI": "Payments Tech",
    "CB": "Insurance (P&C)", "MRSH": "Insurance Broker", "ICE": "Exchange",
    "MCO": "Data/Ratings", "CME": "Exchange", "AON": "Insurance Broker",
    "PYPL": "Digital Payments", "PNC": "Regional Bank", "USB": "Regional Bank",
    "COF": "Consumer Finance", "TRV": "Insurance (P&C)", "MET": "Insurance (Life)",
    "AFL": "Insurance (Life)", "ALL": "Insurance (P&C)", "MSCI": "Data/Index",
    # ── Consumer ──
    "WMT": "Mass Retail", "HD": "Home Improvement", "PG": "Household Staples",
    "COST": "Warehouse Club (Staples)", "LVMUY": "Luxury", "KO": "Beverages",
    "PEP": "Beverages", "PM": "Tobacco", "MCD": "QSR",
    "BKNG": "Online Travel", "LOW": "Home Improvement", "TJX": "Off-Price Retail",
    "NKE": "Athletic/Apparel", "SBUX": "QSR", "CL": "Household Staples",
    "MDLZ": "Packaged Food", "CMG": "QSR", "ABNB": "Online Travel",
    "ORLY": "Auto Parts Retail", "AZO": "Auto Parts Retail", "MNST": "Beverages",
    "RCL": "Cruise/Leisure", "TGT": "Mass Retail", "ROST": "Off-Price Retail",
    "YUM": "QSR", "HLT": "Hotels/Lodging", "LULU": "Athletic/Apparel",
    "MO": "Tobacco", "KHC": "Packaged Food", "EL": "Beauty/Luxury",
    # ── Industrials ──
    "GE": "Aerospace Engine", "CAT": "Heavy Equipment", "RTX": "Defense Systems",
    "UNP": "Rail Transport", "HON": "Diversified Industrial", "LMT": "Defense Prime",
    "BA": "Commercial Aero", "DE": "Ag Equipment", "UPS": "Parcel/Logistics",
    "WM": "Waste Management", "GD": "Defense Prime", "PH": "Motion/Control",
    "CTAS": "Uniform/Facility", "TDG": "Aero Components", "NOC": "Defense (C4ISR)",
    "ITW": "Diversified Industrial", "MMM": "Diversified Industrial",
    "RSG": "Waste Management", "CSX": "Rail Transport", "EMR": "Automation/Control",
    "CARR": "HVAC/Building", "FDX": "Parcel/Logistics", "NSC": "Rail Transport",
    "AXON": "Public Safety Tech", "VRSK": "Data Analytics", "OTIS": "Elevator/Building",
    "IR": "Flow/Climate", "ROK": "Factory Automation", "DOV": "Diversified Industrial",
    "WAB": "Rail Equipment",
    # ── Energy & Materials ──
    "XOM": "Oil Major", "CVX": "Oil Major", "LIN": "Industrial Gas",
    "SHEL": "Oil Major (EU)", "BHP": "Diversified Mining", "TTE": "Oil Major (EU)",
    "COP": "E&P", "SHW": "Coatings/Paint", "ENB": "Midstream/Pipeline",
    "EOG": "E&P", "SLB": "Oilfield Services", "FCX": "Copper Mining",
    "APD": "Industrial Gas", "ECL": "Specialty Chemical", "NEM": "Gold Mining",
    "MPC": "Refining", "FANG": "E&P (Permian)", "PSX": "Refining",
    "VLO": "Refining", "OXY": "E&P", "BKR": "Oilfield Services",
    "CTVA": "Ag Chemical", "NUE": "Steel", "DD": "Specialty Chemical",
    "DOW": "Commodity Chemical", "VMC": "Aggregates", "MLM": "Aggregates",
    "HAL": "Oilfield Services", "DVN": "E&P", "CE": "Specialty Chemical",
    # ── Korea ──
    "005930.KS": "Semiconductor", "000660.KS": "Memory",
    "207940.KS": "Bio CDMO", "005380.KS": "Auto OEM",
    "068270.KS": "Biosimilar", "005490.KS": "Steel/Holdings",
    "035420.KS": "Internet Platform", "006400.KS": "Battery",
    "000270.KS": "Auto OEM", "051910.KS": "Battery/Chemical",
    "012450.KS": "Defense/Aerospace", "105560.KS": "Banking",
    "055550.KS": "Banking", "034730.KS": "Holding/Telecom",
    "035720.KS": "Internet Platform", "028260.KS": "Construction/Holding",
    "066570.KS": "Consumer Electronics", "032830.KS": "Insurance",
    "086790.KS": "Banking", "009150.KS": "Electronic Components",
    "017670.KS": "Telecom", "003550.KS": "Holding",
    "033780.KS": "Tobacco/Staples", "030200.KS": "Telecom",
    "010130.KS": "Zinc/Smelting", "047050.KS": "Trading/Steel",
    "010120.KS": "Electric Infra", "003670.KS": "Battery Materials",
    "005935.KS": "Semiconductor", "018260.KS": "IT Services",
    # ════════════════════ 추가 종목 테마 ════════════════════
    # Semicon 추가
    "WOLF": "SiC/Wide Bandgap", "ACLS": "Equipment (Ion Implant)",
    "MKSI": "Instruments/Photonics", "SLAB": "IoT/Timing", "ALGM": "Magnetic Sensors",
    # Software 추가
    "OKTA": "Identity/Zero Trust", "PATH": "RPA/Automation",
    "DOCU": "Digital Signature", "MNDY": "Project Management",
    "S": "Cybersecurity (EDR)", "SE": "E-commerce/Gaming (SEA)",
    "GRAB": "Super App (SEA)", "PINS": "Social Commerce",
    "SNAP": "Social/AR", "SPOT": "Audio Streaming",
    "ROKU": "Streaming Platform", "ESTC": "Search/Observability",
    "IOT": "IoT Platform", "GEN": "Consumer Security",
    # AI Infra 추가
    "CLS": "Contract Mfg (AI)", "POWL": "Power/Transformer",
    "AAON": "Cooling/HVAC", "FTV": "Industrial IoT",
    "TDY": "Instruments/Imaging", "GNRC": "Backup Power",
    "WCC": "Electrical Distribution", "ENPH": "Solar/Microinverter",
    # Healthcare 추가
    "MRNA": "mRNA Therapeutics", "BIIB": "Neuroscience",
    "ALNY": "RNAi Therapeutics", "ALGN": "Dental/Ortho",
    "HOLX": "Women's Health Dx", "RMD": "Sleep/Respiratory",
    "PODD": "Diabetes Devices", "INCY": "Biotech (Oncology)",
    "ILMN": "Genomics/Sequencing", "DXCM": "CGM/Diabetes",
    "WAT": "Analytical Instruments", "CNC": "Managed Care (Medicaid)",
    "MOH": "Managed Care (Medicaid)", "TECH": "Life Science Reagents",
    # Financials 추가
    "KKR": "Private Equity", "APO": "Private Equity",
    "ARES": "Alternative Asset Mgmt", "TROW": "Asset Management",
    "RJF": "Wealth Management", "NDAQ": "Exchange",
    "MKTX": "Bond Trading", "WTW": "Insurance Broker",
    "FITB": "Regional Bank", "MTB": "Regional Bank",
    "HBAN": "Regional Bank", "SYF": "Consumer Finance",
    "HOOD": "Retail Trading",
    # Consumer 추가
    "MELI": "E-commerce (LatAm)", "CPNG": "E-commerce (Korea)",
    "DPZ": "QSR (Pizza)", "DECK": "Footwear/Outdoor",
    "ULTA": "Beauty Retail", "MAR": "Hotels/Lodging",
    "DKNG": "Sports Betting", "CCL": "Cruise/Leisure",
    "LVS": "Gaming/Casino", "MGM": "Gaming/Casino",
    "WYNN": "Gaming/Casino", "ETSY": "Online Marketplace",
    "TPR": "Luxury (Accessible)", "GRMN": "Wearables/Nav",
    # Industrials 추가
    "HWM": "Aerospace Metals", "HEI": "Aero Components",
    "TT": "HVAC/Climate", "FAST": "Fasteners/MRO",
    "XYL": "Water Technology", "LHX": "Defense Electronics",
    "LDOS": "Defense IT", "BWXT": "Nuclear Components",
    "TXT": "Aviation (Multi)", "GWW": "Industrial Distribution",
    "SNA": "Professional Tools", "NDSN": "Precision Dispensing",
    "J": "Engineering Services",
    # Energy & Materials 추가
    "RIO": "Diversified Mining", "VALE": "Iron Ore Mining",
    "GOLD": "Gold Mining", "WPM": "Precious Metals Streaming",
    "TECK": "Diversified Mining", "ALB": "Lithium",
    "KMI": "Midstream/Pipeline", "WMB": "Midstream/Pipeline",
    "OKE": "Midstream/Pipeline", "PPG": "Coatings",
    "IFF": "Flavors/Fragrances", "EMN": "Specialty Chemical",
    "MP": "Rare Earth",
    # Energy & Materials 추가 (원자재 확장)
    "CCJ": "Uranium Mining", "UEC": "Uranium Mining",
    "LEU": "Uranium Enrichment", "DNN": "Uranium Mining",
    "NXE": "Uranium Mining", "UUUU": "Uranium/Rare Earth",
    "SCCO": "Copper Mining", "HBM": "Copper/Zinc Mining",
    "ERO": "Copper Mining", "IVPAF": "Copper Mining (DRC)",
    "AEM": "Gold Mining (Senior)", "FNV": "Precious Metals Streaming",
    "RGLD": "Precious Metals Streaming", "KGC": "Gold Mining",
    "AGI": "Gold Mining", "EGO": "Gold Mining",
    "PAAS": "Silver Mining", "AG": "Silver Mining",
    "HL": "Silver/Gold Mining", "MAG": "Silver Mining (Explorer)",
    "SQM": "Lithium/Specialty Chem", "LAC": "Lithium Mining",
    "ALTM": "Lithium", "PLL": "Lithium Mining",
    "NTR": "Potash/Fertilizer", "MOS": "Phosphate/Potash",
    "CF": "Nitrogen Fertilizer", "FMC": "Crop Chemicals",
    "ADM": "Grain Trading/Processing", "BG": "Grain Trading/Processing",
    "ANDE": "Grain/Ethanol",
    "STLD": "Steel (EAF)", "CLF": "Iron Ore/Steel",
    "RS": "Steel Distribution", "CMC": "Steel (Rebar/Merchant)",
    "ATI": "Specialty Alloys (Aero/Energy)",
    # Korea 추가
    "373220.KS": "Battery (Cell)", "352820.KS": "Entertainment/K-Pop",
    "000810.KS": "Insurance (P&C)", "259960.KS": "Gaming",
    "402340.KS": "Holding/Investment", "011070.KS": "Camera Module",
    "096770.KS": "Energy/Battery", "034020.KS": "Nuclear/Power Plant",
    "078930.KS": "Holding/Energy", "316140.KS": "Banking",
    "036570.KS": "Gaming (MMORPG)", "011200.KS": "Shipping/Container",
    "267260.KS": "Transformer/Switchgear", "042700.KS": "Semicon Equipment",
    # ── Japan ADR ──
    "TM": "Auto OEM", "SONY": "Entertainment/Semicon",
    "MUFG": "Mega Bank", "SMFG": "Mega Bank",
    "MFG": "Mega Bank", "NMR": "Investment Bank",
    "HMC": "Auto OEM", "NTT": "Telecom/IT Services",
    "IX": "Leasing/Financial", "TAK": "Big Pharma",
    "SNE": "Entertainment/Semicon", "DSCSY": "Semicon Equipment (Grind)",
    "NTDOY": "Gaming", "KYOCY": "Electronic Components",
    "FANUY": "Factory Automation", "TOELY": "Semicon Equipment (Etch/Dep)",
    "HXSCL": "Photonics/Sensors", "KNBWY": "Steel/Materials",
    "DNZOY": "Auto Parts (Tier1)", "ITOCY": "Trading/Conglomerate",
    "MSBHF": "Defense/Heavy Industry", "SBHSY": "Tech Investment/Telecom",
    # ── China ADR ──
    "BABA": "E-commerce/Cloud", "PDD": "E-commerce (Value)",
    "JD": "E-commerce/Logistics", "BIDU": "Search/AI",
    "NIO": "EV", "LI": "EV (EREV)",
    "XPEV": "EV", "NTES": "Gaming/Education",
    "TCOM": "Online Travel", "BILI": "Video/Gaming",
    "ZTO": "Express Delivery", "TME": "Music Streaming",
    "YUMC": "QSR (China)", "MNSO": "Value Retail",
    "TAL": "Education", "FUTU": "Online Brokerage",
    "WB": "Social Media", "VIPS": "Discount E-commerce",
    "BEKE": "Real Estate Platform", "GDS": "Data Center",
    # ── Europe ADR ──
    "UL": "Consumer Staples", "GSK": "Big Pharma",
    "SNY": "Big Pharma", "DEO": "Spirits/Beverages",
    "BP": "Oil Major", "RACE": "Luxury Auto",
    "RELX": "Data Analytics/Publishing", "ING": "Banking",
    "PHG": "Health Tech/Imaging", "ARGX": "Biotech (Autoimmune)",
    "ERIC": "Telecom Equipment", "NOK": "Telecom Equipment",
    "DB": "Investment Bank", "CS": "Investment Bank",
    "SAN": "Banking", "BBVA": "Banking",
    "WPP": "Advertising", "LOGI": "Peripherals/Hardware",
    "STLA": "Auto OEM", "TEF": "Telecom",
    # ── India ADR ──
    "INFY": "IT Services/Outsourcing", "WIT": "IT Services/Outsourcing",
    "HDB": "Private Bank", "IBN": "Private Bank",
    "RDY": "Generic Pharma", "TTM": "Auto OEM",
    "MMYT": "Online Travel", "WNS": "BPO/Analytics",
    "SIFY": "IT Infrastructure",
    # ════════════════════ Samsung Q.Pack additions ════════════════════
    # Technology additions
    "ACN": "IT Consulting", "IBM": "IT Consulting",
    "MSI": "Communications Equipment", "SNDK": "Storage",
    # CommServices additions
    "CMCSA": "Cable/Broadband", "DIS": "Media/Entertainment",
    "GOOG": "Cloud Platform", "NFLX": "Streaming Platform",
    "T": "Telecom", "TMUS": "Wireless Telecom", "VZ": "Telecom",
    # Healthcare additions
    "CVS": "Healthcare Retail/PBM", "NVS": "EU Pharma",
    # Financials additions
    "BAM": "Alternative Asset Mgmt", "BCS": "Mega Bank",
    "BK": "Custody Bank", "BMO": "Mega Bank",
    "BN": "Conglomerate", "BNS": "Mega Bank",
    "BX": "Private Equity", "CM": "Mega Bank",
    "HSBC": "Mega Bank", "IBKR": "Online Brokerage",
    "ITUB": "Mega Bank", "LYG": "Mega Bank",
    "RY": "Mega Bank", "TD": "Mega Bank", "UBS": "Investment Bank",
    # RealEstate additions
    "AMT": "Tower REIT", "PLD": "Industrial REIT", "WELL": "Healthcare REIT",
    # Industrials additions
    "ADP": "HR/Payroll Services", "CMI": "Engine/Power Systems",
    "CP": "Rail Transport", "JCI": "Building Technology",
    # Materials additions
    "CRH": "Building Materials",
    # Energy additions
    "CNQ": "E&P (Oil Sands)", "EPD": "Midstream/Pipeline",
    "EQNR": "Oil Major (EU)", "PBR": "Oil Major (EM)",
    # Utilities additions
    "DUK": "Regulated Utility", "NEE": "Renewable Utility",
    "NGG": "Regulated Utility", "SO": "Regulated Utility",
    # ConsStaples additions
    "BTI": "Tobacco", "BUD": "Beverages",
}

# ── Theme Consolidation: 336 granular → ~47 macro themes ──
_THEME_CONSOLIDATION = {
    # Semiconductor
    "AI Chip": "Semiconductor Design", "Logic Design": "Semiconductor Design",
    "IP/Design": "Semiconductor Design", "FPGA": "Semiconductor Design",
    "Auto Vision": "Semiconductor Design", "Semiconductor": "Semiconductor Design",
    "Entertainment/Semicon": "Semiconductor Design",
    "Foundry": "Semiconductor Foundry",
    "Memory": "Semiconductor Memory",
    "Analog": "Semiconductor Analog", "Analog/Mixed": "Semiconductor Analog",
    "Power IC": "Semiconductor Analog", "Power/Auto": "Semiconductor Analog",
    "MCU/Embedded": "Semiconductor Analog", "Audio IC": "Semiconductor Analog",
    "Auto/IoT": "Semiconductor Analog", "Mobile/RF": "Semiconductor Analog",
    "Magnetic Sensors": "Semiconductor Analog", "IoT/Timing": "Semiconductor Analog",
    "SiC/Wide Bandgap": "Semiconductor Analog",
    "EDA": "Semiconductor Equipment & EDA", "Equipment (Litho)": "Semiconductor Equipment & EDA",
    "Equipment (Deposition)": "Semiconductor Equipment & EDA",
    "Equipment (Etch)": "Semiconductor Equipment & EDA",
    "Equipment (Inspection)": "Semiconductor Equipment & EDA",
    "Equipment (Ion Implant)": "Semiconductor Equipment & EDA",
    "Materials": "Semiconductor Equipment & EDA", "Instruments/Photonics": "Semiconductor Equipment & EDA",
    "Semicon Equipment": "Semiconductor Equipment & EDA",
    "Semicon Equipment (Grind)": "Semiconductor Equipment & EDA",
    "Semicon Equipment (Etch/Dep)": "Semiconductor Equipment & EDA",
    # Software & Internet
    "Enterprise DB": "Enterprise Software", "Enterprise ERP": "Enterprise Software",
    "Enterprise CRM": "Enterprise Software", "Enterprise Workflow": "Enterprise Software",
    "Enterprise HR": "Enterprise Software", "SMB/Finance SW": "Enterprise Software",
    "Data/AI Analytics": "Enterprise Software", "DevOps/Collab": "Enterprise Software",
    "Observability": "Enterprise Software", "Database": "Enterprise Software",
    "Marketing SaaS": "Enterprise Software", "Vertical SaaS (Pharma)": "Enterprise Software",
    "Search/Observability": "Enterprise Software", "IoT Platform": "Enterprise Software",
    "Project Management": "Enterprise Software", "HCI/Cloud SW": "Enterprise Software",
    "RPA/Automation": "Enterprise Software", "Simulation/CAE": "Enterprise Software",
    "Credit Analytics": "Enterprise Software", "Creative/Design": "Enterprise Software",
    "Cybersecurity": "Cybersecurity", "Cybersecurity (EDR)": "Cybersecurity",
    "Identity/Zero Trust": "Cybersecurity", "Consumer Security": "Cybersecurity",
    "Cloud Platform": "Cloud & Platform", "Edge/CDN": "Cloud & Platform",
    "CPaaS/API": "Cloud & Platform", "Digital Signature": "Cloud & Platform",
    "Internet Platform": "Cloud & Platform", "Real Estate Platform": "Cloud & Platform",
    "Search/AI": "Cloud & Platform",
    "AdTech": "Digital Advertising & Media", "AdTech/Mobile": "Digital Advertising & Media",
    "Social/Ad Platform": "Digital Advertising & Media", "Social Commerce": "Digital Advertising & Media",
    "Social/AR": "Digital Advertising & Media", "Social Media": "Digital Advertising & Media",
    "Advertising": "Digital Advertising & Media",
    "Audio Streaming": "Digital Media & Entertainment", "Streaming Platform": "Digital Media & Entertainment",
    "Music Streaming": "Digital Media & Entertainment", "Video/Gaming": "Digital Media & Entertainment",
    "Gaming Platform": "Digital Media & Entertainment", "Gaming/Education": "Digital Media & Entertainment",
    "Entertainment/K-Pop": "Digital Media & Entertainment", "Gaming": "Digital Media & Entertainment",
    "Gaming (MMORPG)": "Digital Media & Entertainment", "Sports Betting": "Digital Media & Entertainment",
    "E-commerce Platform": "E-commerce & Delivery", "E-commerce (LatAm)": "E-commerce & Delivery",
    "E-commerce (Korea)": "E-commerce & Delivery", "E-commerce (Value)": "E-commerce & Delivery",
    "E-commerce/Cloud": "E-commerce & Delivery", "E-commerce/Gaming (SEA)": "E-commerce & Delivery",
    "E-commerce/Logistics": "E-commerce & Delivery", "Discount E-commerce": "E-commerce & Delivery",
    "Delivery Platform": "E-commerce & Delivery", "Express Delivery": "E-commerce & Delivery",
    "Online Marketplace": "E-commerce & Delivery", "Super App (SEA)": "E-commerce & Delivery",
    "Value Retail": "E-commerce & Delivery", "Mobility Platform": "E-commerce & Delivery",
    "Education": "E-commerce & Delivery",
    "Crypto Exchange": "Fintech & Digital Finance", "SMB Fintech": "Fintech & Digital Finance",
    "Digital Payments": "Fintech & Digital Finance", "Online Brokerage": "Fintech & Digital Finance",
    "Retail Trading": "Fintech & Digital Finance",
    # AI & Data Center Infra
    "Networking": "Data Center & Networking", "DC REIT": "Data Center & Networking",
    "DC Construction": "Data Center & Networking", "Server/Hardware": "Data Center & Networking",
    "Data Center": "Data Center & Networking", "Connector": "Data Center & Networking",
    "Optical Transceiver": "Data Center & Networking", "Optical Fiber": "Data Center & Networking",
    "Contract Mfg": "Data Center & Networking", "Contract Mfg (AI)": "Data Center & Networking",
    "Storage": "Data Center & Networking",
    "Power/Transformer": "Power & Energy Infra", "Nuclear/Power": "Power & Energy Infra",
    "Power Equipment": "Power & Energy Infra", "Power Generation": "Power & Energy Infra",
    "Backup Power": "Power & Energy Infra", "Electrical Distribution": "Power & Energy Infra",
    "Solar/Microinverter": "Power & Energy Infra", "Transformer/Switchgear": "Power & Energy Infra",
    "Nuclear/Power Plant": "Power & Energy Infra", "Electric Infra": "Power & Energy Infra",
    "Cooling/Thermal": "Industrial Technology", "Cooling/HVAC": "Industrial Technology",
    "Instruments": "Industrial Technology", "Test/Measurement": "Industrial Technology",
    "Industrial IoT": "Industrial Technology", "Instruments/Imaging": "Industrial Technology",
    "Photonics/Sensors": "Industrial Technology", "Water/Process": "Industrial Technology",
    "Public Safety Tech": "Industrial Technology",
    "Consumer Hardware": "Consumer Hardware", "Peripherals/Hardware": "Consumer Hardware",
    "Wearables/Nav": "Consumer Hardware", "Camera Module": "Consumer Hardware",
    "Consumer Electronics": "Consumer Hardware", "Electronic Components": "Consumer Hardware",
    # Healthcare
    "Big Pharma (GLP-1)": "Big Pharma", "Big Pharma (Immuno)": "Big Pharma",
    "Big Pharma (Oncology)": "Big Pharma", "Big Pharma (Vaccine)": "Big Pharma",
    "Diversified Pharma": "Big Pharma", "EU Pharma": "Big Pharma",
    "Generic Pharma": "Big Pharma", "Big Pharma": "Big Pharma",
    "Biotech": "Biotech", "Biotech (Rare)": "Biotech", "Biotech (Antiviral)": "Biotech",
    "Biotech (Immuno)": "Biotech", "Biotech (Oncology)": "Biotech",
    "Biotech (Autoimmune)": "Biotech", "mRNA Therapeutics": "Biotech",
    "Neuroscience": "Biotech", "RNAi Therapeutics": "Biotech",
    "Biosimilar": "Biotech", "Bio CDMO": "Biotech",
    "Managed Care": "Healthcare Services", "Managed Care (Medicaid)": "Healthcare Services",
    "Hospital": "Healthcare Services", "Distribution": "Healthcare Services",
    "Animal Health": "Healthcare Services", "Veterinary Dx": "Healthcare Services",
    "Clinical Trials/CRO": "Healthcare Services",
    "Medical Supplies": "Medical Devices & Diagnostics", "Cardio Devices": "Medical Devices & Diagnostics",
    "Surgical Robotics": "Medical Devices & Diagnostics", "Orthopedics": "Medical Devices & Diagnostics",
    "Imaging/Diagnostics": "Medical Devices & Diagnostics", "Diagnostics/Devices": "Medical Devices & Diagnostics",
    "Dental/Ortho": "Medical Devices & Diagnostics", "Women's Health Dx": "Medical Devices & Diagnostics",
    "Sleep/Respiratory": "Medical Devices & Diagnostics", "Diabetes Devices": "Medical Devices & Diagnostics",
    "CGM/Diabetes": "Medical Devices & Diagnostics", "Health Tech/Imaging": "Medical Devices & Diagnostics",
    "Life Science Tools": "Life Science Tools", "Genomics/Sequencing": "Life Science Tools",
    "Analytical Instruments": "Life Science Tools", "Life Science Reagents": "Life Science Tools",
    # Financials
    "Mega Bank": "Banks", "Regional Bank": "Banks", "Banking": "Banks", "Private Bank": "Banks",
    "Payments": "Payments & Exchanges", "Payments Tech": "Payments & Exchanges",
    "Exchange": "Payments & Exchanges", "Bond Trading": "Payments & Exchanges",
    "Investment Bank": "Investment Banking & Asset Mgmt", "Asset Management": "Investment Banking & Asset Mgmt",
    "Private Equity": "Investment Banking & Asset Mgmt", "Alternative Asset Mgmt": "Investment Banking & Asset Mgmt",
    "Wealth Management": "Investment Banking & Asset Mgmt", "Broker/Wealth": "Investment Banking & Asset Mgmt",
    "Insurance (P&C)": "Insurance", "Insurance (Life)": "Insurance",
    "Insurance Broker": "Insurance", "Insurance": "Insurance",
    "Data/Ratings": "Financial Data & Analytics", "Data/Index": "Financial Data & Analytics",
    "Data Analytics": "Financial Data & Analytics", "Data Analytics/Publishing": "Financial Data & Analytics",
    "Consumer Finance": "Consumer Finance", "Leasing/Financial": "Consumer Finance",
    "Conglomerate": "Conglomerate & Holding", "Holding": "Conglomerate & Holding",
    "Holding/Telecom": "Conglomerate & Holding", "Holding/Energy": "Conglomerate & Holding",
    "Holding/Investment": "Conglomerate & Holding", "Trading/Conglomerate": "Conglomerate & Holding",
    "Tech Investment/Telecom": "Conglomerate & Holding",
    # Consumer
    "Mass Retail": "Retail", "Warehouse Club": "Retail", "Home Improvement": "Retail",
    "Off-Price Retail": "Retail", "Auto Parts Retail": "Retail", "Beauty Retail": "Retail",
    "QSR": "Restaurants & Leisure", "QSR (Pizza)": "Restaurants & Leisure",
    "QSR (China)": "Restaurants & Leisure", "Online Travel": "Restaurants & Leisure",
    "Hotels/Lodging": "Restaurants & Leisure", "Cruise/Leisure": "Restaurants & Leisure",
    "Gaming/Casino": "Restaurants & Leisure",
    "Beverages": "Consumer Staples", "Household Staples": "Consumer Staples",
    "Packaged Food": "Consumer Staples", "Tobacco": "Consumer Staples",
    "Tobacco/Staples": "Consumer Staples", "Spirits/Beverages": "Consumer Staples",
    "Consumer Staples": "Consumer Staples",
    "Luxury": "Consumer Brands", "Luxury (Accessible)": "Consumer Brands",
    "Luxury Auto": "Consumer Brands", "Beauty/Luxury": "Consumer Brands",
    "Athletic/Apparel": "Consumer Brands", "Footwear/Outdoor": "Consumer Brands",
    # Industrials
    "Aerospace Engine": "Aerospace & Defense", "Commercial Aero": "Aerospace & Defense",
    "Defense Systems": "Aerospace & Defense", "Defense Prime": "Aerospace & Defense",
    "Defense (C4ISR)": "Aerospace & Defense", "Defense Electronics": "Aerospace & Defense",
    "Defense IT": "Aerospace & Defense", "Nuclear Components": "Aerospace & Defense",
    "Aero Components": "Aerospace & Defense", "Aerospace Metals": "Aerospace & Defense",
    "Aviation (Multi)": "Aerospace & Defense", "Defense/Aerospace": "Aerospace & Defense",
    "Defense/Heavy Industry": "Aerospace & Defense",
    "Heavy Equipment": "Industrial Equipment", "Ag Equipment": "Industrial Equipment",
    "Motion/Control": "Industrial Equipment", "Factory Automation": "Industrial Equipment",
    "Automation/Control": "Industrial Equipment", "Precision Dispensing": "Industrial Equipment",
    "Professional Tools": "Industrial Equipment", "Fasteners/MRO": "Industrial Equipment",
    "Rail Equipment": "Industrial Equipment", "Diversified Industrial": "Industrial Equipment",
    "Industrial Distribution": "Industrial Equipment", "Engineering Services": "Industrial Equipment",
    "Rail Transport": "Transport & Logistics", "Parcel/Logistics": "Transport & Logistics",
    "Shipping/Container": "Transport & Logistics",
    "Waste Management": "Environmental & Water", "Water Technology": "Environmental & Water",
    "Flow/Climate": "Environmental & Water",
    "HVAC/Building": "Building & Construction", "HVAC/Climate": "Building & Construction",
    "Elevator/Building": "Building & Construction", "Uniform/Facility": "Building & Construction",
    "Construction/Holding": "Building & Construction",
    "Telecom": "Telecom & IT Services", "Telecom Equipment": "Telecom & IT Services",
    "Telecom/IT Services": "Telecom & IT Services", "IT Services": "Telecom & IT Services",
    "IT Services/Outsourcing": "Telecom & IT Services", "IT Infrastructure": "Telecom & IT Services",
    "BPO/Analytics": "Telecom & IT Services",
    # Energy
    "Oil Major": "Oil & Gas", "Oil Major (EU)": "Oil & Gas", "E&P": "Oil & Gas",
    "E&P (Permian)": "Oil & Gas", "Oilfield Services": "Oil & Gas",
    "Refining": "Oil & Gas", "Midstream/Pipeline": "Oil & Gas",
    # Materials
    "Gold Mining": "Precious Metals", "Gold Mining (Senior)": "Precious Metals",
    "Silver Mining": "Precious Metals", "Silver Mining (Explorer)": "Precious Metals",
    "Silver/Gold Mining": "Precious Metals", "Precious Metals Streaming": "Precious Metals",
    "Copper Mining": "Base Metals & Mining", "Copper Mining (DRC)": "Base Metals & Mining",
    "Copper/Zinc Mining": "Base Metals & Mining", "Diversified Mining": "Base Metals & Mining",
    "Iron Ore Mining": "Base Metals & Mining", "Zinc/Smelting": "Base Metals & Mining",
    "Iron Ore/Steel": "Steel & Metals", "Steel": "Steel & Metals",
    "Steel (EAF)": "Steel & Metals", "Steel (Rebar/Merchant)": "Steel & Metals",
    "Steel Distribution": "Steel & Metals", "Steel/Holdings": "Steel & Metals",
    "Steel/Materials": "Steel & Metals", "Specialty Alloys (Aero/Energy)": "Steel & Metals",
    "Trading/Steel": "Steel & Metals",
    "Uranium Mining": "Uranium & Nuclear Fuel", "Uranium Enrichment": "Uranium & Nuclear Fuel",
    "Uranium/Rare Earth": "Uranium & Nuclear Fuel", "Rare Earth": "Uranium & Nuclear Fuel",
    "Lithium": "Battery & EV Materials", "Lithium Mining": "Battery & EV Materials",
    "Lithium/Specialty Chem": "Battery & EV Materials", "Battery": "Battery & EV Materials",
    "Battery (Cell)": "Battery & EV Materials", "Battery/Chemical": "Battery & EV Materials",
    "Battery Materials": "Battery & EV Materials", "Energy/Battery": "Battery & EV Materials",
    "EV": "Auto & EV", "EV (EREV)": "Auto & EV", "EV/Energy": "Auto & EV",
    "Auto OEM": "Auto & EV", "Auto Parts (Tier1)": "Auto & EV",
    "Industrial Gas": "Chemicals", "Specialty Chemical": "Chemicals",
    "Commodity Chemical": "Chemicals", "Ag Chemical": "Chemicals",
    "Coatings/Paint": "Chemicals", "Coatings": "Chemicals",
    "Flavors/Fragrances": "Chemicals", "Aggregates": "Chemicals",
    "Potash/Fertilizer": "Agriculture & Food", "Phosphate/Potash": "Agriculture & Food",
    "Nitrogen Fertilizer": "Agriculture & Food", "Crop Chemicals": "Agriculture & Food",
    "Grain Trading/Processing": "Agriculture & Food", "Grain/Ethanol": "Agriculture & Food",
    # Samsung Q.Pack additions
    "IT Consulting": "Enterprise Software", "Communications Equipment": "Industrial Technology",
    "Cable/Broadband": "Telecom & IT Services", "Media/Entertainment": "Digital Media & Entertainment",
    "Wireless Telecom": "Telecom & IT Services",
    "Healthcare Retail/PBM": "Healthcare Services",
    "Custody Bank": "Banks", "Online Brokerage": "Fintech & Digital Finance",
    "Tower REIT": "Data Center & Networking", "Industrial REIT": "Real Estate",
    "Healthcare REIT": "Real Estate",
    "HR/Payroll Services": "Enterprise Software", "Engine/Power Systems": "Industrial Equipment",
    "Building Technology": "Building & Construction",
    "Building Materials": "Chemicals",
    "E&P (Oil Sands)": "Oil & Gas", "Oil Major (EM)": "Oil & Gas",
    "Regulated Utility": "Utilities", "Renewable Utility": "Utilities",
    "Transport/Ride-Hail": "Transport & Logistics",
    "Warehouse Club (Staples)": "Consumer Staples",
}

# Apply consolidation — STOCK_THEMES keeps granular detail,
# consolidated version available via get_consolidated_theme()
def get_consolidated_theme(ticker):
    """Return macro theme for a ticker."""
    raw = STOCK_THEMES.get(ticker, "-")
    return _THEME_CONSOLIDATION.get(raw, raw)

STOCK_THEMES_CONSOLIDATED = {tk: get_consolidated_theme(tk) for tk in STOCK_THEMES}


###############################################################################
# SECTION 1-D: ETF SUBTHEME MAPPING (unified with stock themes via Option B)
# ETFs and stocks share the same SubTheme namespace — overlapping sub-buckets
# (e.g., SMH/SOXX/091160.KS share "Semiconductor Design" with NVDA/AMD).
###############################################################################

ETF_SUBTHEMES = {
    # ── EQ_Broad ──
    "SPY": "Broad Market", "QQQ": "Broad Market", "DIA": "Broad Market",
    "RSP": "Broad Market", "IWF": "Broad Market", "IWD": "Broad Market",
    "VUG": "Broad Market", "VTV": "Broad Market", "IJH": "Broad Market",
    "IWM": "Broad Market", "IJR": "Broad Market", "VBR": "Broad Market",
    "VBK": "Broad Market", "MGK": "Broad Market", "MGV": "Broad Market",
    "OEF": "Broad Market",
    # ── EQ_Technology ──
    "XLK": "Sector Broad - Tech",
    "SMH": "Semiconductor Design", "SOXX": "Semiconductor Design",
    "CIBR": "Cybersecurity", "HACK": "Cybersecurity",
    "BOTZ": "Robotics & AI",
    # ── EQ_Healthcare ──
    "XLV": "Sector Broad - Healthcare",
    "IBB": "Biotech", "XBI": "Biotech",
    # ── EQ_Financials ──
    "XLF": "Sector Broad - Financials", "KRE": "Banks",
    # ── EQ_ConsDisc ──
    "XLY": "Sector Broad - ConsDisc",
    "ITB": "Building & Construction", "XHB": "Building & Construction",
    # ── EQ_ConsStaples ──
    "XLP": "Sector Broad - ConsStaples",
    # ── EQ_Industrials ──
    "XLI": "Sector Broad - Industrials",
    "ITA": "Aerospace & Defense", "PAVE": "Industrial Equipment",
    # ── EQ_Energy / Materials / Utilities / RealEstate / CommServices ──
    "XLE": "Sector Broad - Energy",
    "XLB": "Sector Broad - Materials",
    "XLU": "Sector Broad - Utilities",
    "XLRE": "Sector Broad - RealEstate",
    "XLC": "Sector Broad - CommServices",
    # ── EQ_Factor ──
    "MTUM": "Factor - Momentum", "QUAL": "Factor - Quality",
    "USMV": "Factor - Min Vol", "VLUE": "Factor - Value",
    "SIZE": "Factor - Size", "SCHD": "Factor - Dividend",
    "VIG": "Factor - Dividend", "DVY": "Factor - Dividend",
    "NOBL": "Factor - Dividend", "COWZ": "Factor - Quality",
    "MOAT": "Factor - Quality", "SPHQ": "Factor - Quality",
    "SPMO": "Factor - Momentum", "DYNF": "Factor - Multi",
    # ── EQ_Thematic ──
    "AIQ": "Cloud & Platform", "ROBO": "Robotics & AI",
    "ARKG": "Biotech", "ARKW": "Cloud & Platform",
    "DRIV": "Auto & EV", "UFO": "Aerospace & Defense",
    "SKYY": "Cloud & Platform", "FINX": "Fintech & Digital Finance",
    "EDOC": "Healthcare Services", "QCLN": "Power & Energy Infra",
    "BATT": "Battery & EV Materials", "REMX": "Uranium & Nuclear Fuel",
    "XSD": "Semiconductor Design", "IGV": "Cloud & Platform",
    "CLOU": "Cloud & Platform", "SHLD": "Aerospace & Defense",
    "463250.KS": "Aerospace & Defense", "ARKK": "Disruptive Innovation",
    "TAN": "Power & Energy Infra", "ICLN": "Power & Energy Infra",
    "LIT": "Battery & EV Materials",
    # ── Intl_Developed ──
    "VEA": "Developed Markets", "EFA": "Developed Markets",
    "IEFA": "Developed Markets", "SPDW": "Developed Markets",
    "VGK": "Europe", "EZU": "Europe", "HEDJ": "Europe", "FEZ": "Europe",
    "EWG": "Europe", "EWU": "Europe", "EWQ": "Europe", "EWL": "Europe",
    "EWJ": "Japan", "BBJP": "Japan", "DXJ": "Japan",
    "EWA": "Asia Pacific", "EWC": "North America", "EIS": "Middle East",
    # ── Emerging_Markets ──
    "VWO": "EM Broad", "EEM": "EM Broad", "IEMG": "EM Broad", "EMXC": "EM Broad",
    "EWZ": "Latin America", "EWW": "Latin America",
    "GXG": "Latin America", "ECH": "Latin America",
    "EWT": "Asia Pacific", "EWY": "Korea (Index)", "KORU": "Korea (Index)",
    "INDA": "India",
    "FXI": "China", "KWEB": "China", "MCHI": "China",
    "THD": "Asia Pacific", "VNM": "Asia Pacific", "EIDO": "Asia Pacific",
    "TUR": "Other EM", "EZA": "Other EM",
    # ── FI_Short ──
    "SHY": "Treasury - Short", "SGOV": "Treasury - Short",
    "BIL": "Treasury - Short", "SHV": "Treasury - Short", "JPST": "Treasury - Short",
    "VCSH": "IG Corporate - Short",
    "JAAA": "CLO",
    "FLOT": "Floating Rate", "USFR": "Floating Rate",
    # ── FI_Intermediate ──
    "IEI": "Treasury - Intermediate", "IEF": "Treasury - Intermediate",
    "GOVT": "Treasury - Intermediate",
    "VCIT": "IG Corporate - Intermediate", "IGIB": "IG Corporate - Intermediate",
    "AGG": "Aggregate Bond", "BND": "Aggregate Bond",
    "MBB": "MBS", "JBBB": "CLO",
    # ── FI_Long ──
    "TLH": "Treasury - Long", "TLT": "Treasury - Long",
    "EDV": "Treasury - Long", "ZROZ": "Treasury - Long", "VGLT": "Treasury - Long",
    "LQD": "IG Corporate - Long", "VCLT": "IG Corporate - Long",
    # ── FI_Credit ──
    "HYG": "High Yield", "USHY": "High Yield",
    "FALN": "High Yield", "ANGL": "High Yield",
    "BKLN": "Senior Loans", "SRLN": "Senior Loans",
    "PFF": "Preferred",
    # ── FI_Inflation ──
    "TIP": "Inflation-Linked", "VTIP": "Inflation-Linked",
    "SCHP": "Inflation-Linked", "STIP": "Inflation-Linked", "LTPZ": "Inflation-Linked",
    # ── FI_International ──
    "BNDX": "International Bonds", "IAGG": "International Bonds",
    "IGOV": "International Bonds", "BWX": "International Bonds",
    "EMB": "EM Bonds", "CEMB": "EM Bonds", "LEMB": "EM Bonds", "VWOB": "EM Bonds",
    # ── Commodities — Precious Metals / Mining ──
    "GLD": "Precious Metals", "SLV": "Precious Metals",
    "GLDM": "Precious Metals", "SGOL": "Precious Metals",
    "IAU": "Precious Metals", "AAAU": "Precious Metals", "BAR": "Precious Metals",
    "PPLT": "Precious Metals", "PALL": "Precious Metals",
    "GDX": "Precious Metals", "GDXJ": "Precious Metals",
    "SIL": "Precious Metals", "SILJ": "Precious Metals",
    # ── Commodities — Energy / Ag / Industrial ──
    "USO": "Energy Commodities", "BNO": "Energy Commodities", "UNG": "Energy Commodities",
    "DBA": "Agriculture & Food", "WEAT": "Agriculture & Food",
    "CORN": "Agriculture & Food", "SOYB": "Agriculture & Food",
    "CANE": "Agriculture & Food", "NIGS": "Agriculture & Food",
    "MOO": "Agriculture & Food",
    "DBC": "Broad Commodity", "GSG": "Broad Commodity",
    "FTGC": "Broad Commodity", "PDBC": "Broad Commodity", "COMT": "Broad Commodity",
    "XOP": "Oil & Gas", "OIH": "Oil & Gas", "FCG": "Oil & Gas",
    "COPX": "Base Metals & Mining", "CPER": "Base Metals & Mining",
    "PICK": "Base Metals & Mining", "XME": "Base Metals & Mining",
    "URA": "Uranium & Nuclear Fuel", "URNM": "Uranium & Nuclear Fuel",
    "NUKZ": "Uranium & Nuclear Fuel",
    "GUNR": "Natural Resources",
    # ── Real_Assets ──
    "VNQ": "Sector Broad - RealEstate", "VNQI": "Sector Broad - RealEstate",
    "IYR": "Sector Broad - RealEstate", "REM": "Sector Broad - RealEstate",
    "AMLP": "Oil & Gas", "MLPX": "Oil & Gas",
    "IFRA": "Industrial Equipment", "WOOD": "Agriculture & Food",
    "IBIT": "Crypto", "ETHA": "Crypto",
    # ── Korea_Equity ──
    "069500.KS": "Korea (Index)", "229200.KS": "Korea (Index)",
    "091160.KS": "Semiconductor Design",
    "487240.KS": "Power & Energy Infra",
    "305720.KS": "Battery & EV Materials",
    "102110.KS": "Korea (Index)",
    "396500.KS": "Semiconductor Design",
    "292150.KS": "Korea (Index)",
    "381170.KS": "Sector Broad - Tech",
    "381180.KS": "Semiconductor Design",
    "466920.KS": "Industrial Equipment",
    "395160.KS": "Semiconductor Design",
    "161510.KS": "Factor - Dividend",
    # ── Currency_Vol ──
    "UUP": "Currency", "FXE": "Currency", "FXY": "Currency",
    "FXB": "Currency", "FXA": "Currency", "CYB": "Currency",
    "VIXY": "Volatility",
    # ── Multi_Asset ──
    "AOR": "Asset Allocation", "AOA": "Asset Allocation",
    "AOM": "Asset Allocation", "AOK": "Asset Allocation",
    "RPAR": "Asset Allocation", "GAA": "Asset Allocation",
}


###############################################################################
# SECTION 2: DATA ENGINE
###############################################################################

@dataclass
class ETFData:
    ticker: str; name: str; category: str
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    valid: bool = False; realtime_updated: bool = False
    market_cap: float = 0.0  # USD

def _standardize(df):
    if df is None or df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    df = df.loc[~df.index.duplicated(keep='last')].sort_index()
    for c in ['Open','High','Low','Close','Volume']:
        if c not in df.columns: return pd.DataFrame()
    df = df.dropna(subset=['Close'])
    df['Volume'] = df['Volume'].fillna(0)
    return df

def _get_usdkrw_rate():
    """Cache USD/KRW exchange rate (fetched once per session)."""
    if not hasattr(_get_usdkrw_rate, '_rate'):
        try:
            fi = yf.Ticker('USDKRW=X').fast_info
            _get_usdkrw_rate._rate = float(fi.get('lastPrice', fi.get('last_price', 1400)))
        except:
            _get_usdkrw_rate._rate = 1400.0
    return _get_usdkrw_rate._rate

def _market_cap_usd(ticker, fi=None):
    """Extract market cap in USD from yfinance fast_info/info. Handles KRW conversion & ETFs."""
    try:
        if fi is None:
            fi = yf.Ticker(ticker).fast_info
        mc = fi.get('marketCap', fi.get('market_cap', None))
        cur = fi.get('currency', 'USD')
        # ETFs: market_cap is None → try totalAssets from .info
        if mc is None or (isinstance(mc, float) and not np.isfinite(mc)):
            try:
                ta = yf.Ticker(ticker).info.get('totalAssets')
                if ta and ta > 0:
                    mc = float(ta)
                    # totalAssets for Korean ETFs in KRW
                    if cur == 'KRW':
                        mc = mc / _get_usdkrw_rate()
                    return mc
            except: pass
            return 0.0
        mc = float(mc)
        if not np.isfinite(mc):
            return 0.0
        # Convert KRW to USD
        if cur == 'KRW':
            mc = mc / _get_usdkrw_rate()
        return mc
    except:
        return 0.0

def _fetch_market_cap(ticker):
    """Fetch market cap in USD via yfinance."""
    return _market_cap_usd(ticker)

def _apply_realtime(df, ticker):
    if df is None or df.empty: return df, False, 0.0
    try:
        fi = yf.Ticker(ticker).fast_info
        lp = float(fi.get('lastPrice', fi.get('last_price', 0)))
        mc = _market_cap_usd(ticker, fi)
        if lp <= 0 or not np.isfinite(lp): return df, False, mc
    except: return df, False, 0.0
    today = pd.Timestamp(datetime.today().date())
    last_bar = pd.Timestamp(df.index[-1].date()) if hasattr(df.index[-1], 'date') else pd.Timestamp(df.index[-1])
    if last_bar == today:
        df.loc[df.index[-1], 'Close'] = lp
        df.loc[df.index[-1], 'High'] = max(float(df.loc[df.index[-1], 'High']), lp)
        df.loc[df.index[-1], 'Low'] = min(float(df.loc[df.index[-1], 'Low']), lp)
        return df, True, mc
    elif last_bar < today and today.weekday() < 5:
        pc = float(df['Close'].iloc[-1])
        nr = pd.DataFrame({'Open':[lp],'High':[max(pc,lp)],'Low':[min(pc,lp)],'Close':[lp],'Volume':[0]}, index=[today])
        df = pd.concat([df, nr]).loc[~pd.concat([df, nr]).index.duplicated(keep='last')].sort_index()
        return df, True, mc
    return df, False, mc

class DataEngine:
    def __init__(self, lookback_days=365, custom_date=None, use_realtime=True):
        self.use_realtime = use_realtime
        today = datetime.today()
        self.end_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
        if custom_date:
            try:
                td = datetime.strptime(custom_date, '%Y-%m-%d')
                start = min(today - timedelta(days=lookback_days), td - timedelta(days=lookback_days))
            except: start = today - timedelta(days=lookback_days)
        else: start = today - timedelta(days=lookback_days)
        self.start_date = start.strftime('%Y-%m-%d')

    def download_single(self, ticker, name, category):
        etf = ETFData(ticker=ticker, name=name, category=category)
        for _ in range(3):
            try:
                df = yf.download(ticker, start=self.start_date, progress=False, auto_adjust=False)
                if df is not None and not df.empty:
                    df = _standardize(df)
                    if not df.empty and len(df) >= 60:
                        if 'Adj Close' in df.columns:
                            af = (df['Adj Close'] / df['Close']).fillna(1.0)
                            for c in ['Open','High','Low','Close']: df[c] = df[c] * af
                        if self.use_realtime:
                            df, rt, mc = _apply_realtime(df, ticker)
                            etf.realtime_updated = rt
                            etf.market_cap = mc
                        else:
                            etf.market_cap = _fetch_market_cap(ticker)
                        etf.df, etf.valid = df, True
                        return etf
            except: pass
            time.sleep(0.15)
        return etf

    def download_universe(self, categories=None, universe=None):
        src = universe if universe is not None else GLOBAL_ETF_UNIVERSE
        tasks = [(t, n, c) for c in (categories or list(src.keys()))
                 if c in src for t, n in src[c]["tickers"].items()]
        tasks.sort(key=lambda x: x[0])
        label = "stocks" if universe is not None else "ETFs"
        all_tickers = [t for t, n, c in tasks]
        meta = {t: (n, c) for t, n, c in tasks}
        print(f"\n📡 Batch-downloading {len(all_tickers)} {label} (Start: {self.start_date})...")

        # ── batch download (single API call, much faster) ──
        try:
            bulk = yf.download(all_tickers, start=self.start_date, progress=True,
                               auto_adjust=False, group_by='ticker', threads=True)
        except Exception as e:
            print(f"   ⚠️ Batch download failed ({e}), falling back to sequential...")
            bulk = None

        results = {}; rt = 0; failed = []
        if bulk is not None and not bulk.empty:
            for ticker in all_tickers:
                name, cat = meta[ticker]
                etf = ETFData(ticker=ticker, name=name, category=cat)
                try:
                    if len(all_tickers) == 1:
                        tdf = bulk.copy()
                    else:
                        tdf = bulk[ticker].copy() if ticker in bulk.columns.get_level_values(0) else None
                    if tdf is None or tdf.empty or tdf.dropna(how='all').empty:
                        failed.append(ticker); continue
                    tdf = _standardize(tdf)
                    if tdf.empty or len(tdf) < 60:
                        failed.append(ticker); continue
                    if 'Adj Close' in tdf.columns:
                        af = (tdf['Adj Close'] / tdf['Close']).fillna(1.0)
                        for c in ['Open','High','Low','Close']: tdf[c] = tdf[c] * af
                    if self.use_realtime:
                        tdf, rtu, mc = _apply_realtime(tdf, ticker)
                        etf.realtime_updated = rtu
                        etf.market_cap = mc
                        if rtu: rt += 1
                    else:
                        etf.market_cap = _fetch_market_cap(ticker)
                    etf.df, etf.valid = tdf, True
                    results[ticker] = etf
                except Exception:
                    failed.append(ticker)
        else:
            failed = all_tickers

        # ── sequential fallback for failed tickers ──
        if failed:
            print(f"   ↻ Retrying {len(failed)} failed tickers sequentially...")
            for t in failed:
                n, c = meta[t]
                etf = self.download_single(t, n, c)
                if etf.valid:
                    results[etf.ticker] = etf
                    if etf.realtime_updated: rt += 1

        print(f"   ✅ {len(results)} valid {label}, {rt} real-time updated")
        return results


###############################################################################
# SECTION 3: NAIVE 3-AXIS DISCOVERY DETECTOR
###############################################################################

class NaiveDiscoveryDetector:
    def __init__(self):
        self.benchmark_map: Dict[str, pd.DataFrame] = {}
        self.benchmark_data = None

    def load_benchmarks(self, all_data, extra_benchmarks=None):
        combined = dict(CATEGORY_BENCHMARK)
        if extra_benchmarks:
            combined.update(extra_benchmarks)
        needed = set(combined.values()) | {"SPY"}
        loaded = {t: all_data[t].df for t in needed if t in all_data and all_data[t].valid}
        for cat, bt in combined.items():
            self.benchmark_map[cat] = loaded.get(bt, loaded.get("SPY"))
        self.benchmark_data = loaded.get("SPY", next(iter(loaded.values()), None))
        print(f"📌 Benchmarks loaded: {len(self.benchmark_map)} categories")

    def compute_raw(self, df, category=""):
        close = ss(df['Close'])
        volume = ss(df['Volume'])

        # ── Long-term MAs ──
        sma50 = close.rolling(50, min_periods=40).mean()
        sma200 = close.rolling(200, min_periods=120).mean() if len(close) >= 120 else close.rolling(50, min_periods=40).mean()

        # ── Short-term MAs ──
        ema10 = close.ewm(span=10, min_periods=8).mean()
        sma20 = close.rolling(20, min_periods=15).mean()

        last_close = sf(close.iloc[-1])
        last_sma50 = sf(sma50.iloc[-1])
        last_sma200 = sf(sma200.iloc[-1])
        last_ema10 = sf(ema10.iloc[-1])
        last_sma20 = sf(sma20.iloc[-1])

        # ── Long-term trend flags ──
        above_sma50 = 1 if (last_sma50 > 0 and last_close > last_sma50) else 0
        above_sma200 = 1 if (last_sma200 > 0 and last_close > last_sma200) else 0
        golden_cross = 1 if (last_sma200 > 0 and last_sma50 > last_sma200) else 0

        # ── Short-term trend flags ──
        above_ema10 = 1 if (last_ema10 > 0 and last_close > last_ema10) else 0
        above_sma20 = 1 if (last_sma20 > 0 and last_close > last_sma20) else 0
        ema10_above_sma20 = 1 if (last_sma20 > 0 and last_ema10 > last_sma20) else 0

        # ── Long-term slopes ──
        sma50_10d_ago = sf(sma50.iloc[-11]) if len(sma50) >= 11 else last_sma50
        sma50_slope = ((last_sma50 / sma50_10d_ago) - 1) * 100 if sma50_10d_ago > 0 else 0.0

        sma50_20d_ago = sf(sma50.iloc[-21]) if len(sma50) >= 21 else last_sma50
        sma50_30d_ago = sf(sma50.iloc[-31]) if len(sma50) >= 31 else sma50_20d_ago
        slope_20d_ago = (sma50_20d_ago / sma50_30d_ago - 1) if sma50_30d_ago > 0 else 0

        sma200_20d_ago = sf(sma200.iloc[-21]) if len(sma200) >= 21 else last_sma200
        sma200_slope = ((last_sma200 / sma200_20d_ago) - 1) * 100 if sma200_20d_ago > 0 else 0.0
        sma50_sma200_spread = ((last_sma50 / last_sma200) - 1) * 100 if last_sma200 > 0 else 0.0

        # ── Short-term slopes ──
        sma20_5d_ago = sf(sma20.iloc[-6]) if len(sma20) >= 6 else last_sma20
        sma20_slope = ((last_sma20 / sma20_5d_ago) - 1) * 100 if sma20_5d_ago > 0 else 0.0

        sma20_10d_ago = sf(sma20.iloc[-11]) if len(sma20) >= 11 else last_sma20
        sma20_15d_ago = sf(sma20.iloc[-16]) if len(sma20) >= 16 else sma20_10d_ago
        slope_sma20_10d_ago = (sma20_10d_ago / sma20_15d_ago - 1) if sma20_15d_ago > 0 else 0

        # ── Trend ages ──
        above_mask = close > sma50
        trend_age = 0
        for v in above_mask.values[::-1]:
            if v: trend_age += 1
            else: break

        above_mask_short = close > sma20
        trend_age_short = 0
        for v in above_mask_short.values[::-1]:
            if v: trend_age_short += 1
            else: break

        # ── Distances ──
        sma50_dist = ((last_close / last_sma50) - 1) * 100 if last_sma50 > 0 else 0.0
        sma20_dist = ((last_close / last_sma20) - 1) * 100 if last_sma20 > 0 else 0.0
        rsi = compute_rsi(close, 14)

        window_52w = min(252, len(close) - 1)
        if window_52w >= 60:
            high_52w = sf(close.rolling(window_52w, min_periods=60).max().iloc[-1])
            low_52w = sf(close.rolling(window_52w, min_periods=60).min().iloc[-1])
        else:
            high_52w, low_52w = last_close, last_close
        pct_from_high = ((last_close / high_52w) - 1) * 100 if high_52w > 0 else 0
        range_pct = (last_close - low_52w) / max(high_52w - low_52w, 1e-10) * 100

        # ── Volume indicators ──
        vol_valid = volume[volume > 0]
        vol_3d = sf(vol_valid.iloc[-3:].mean()) if len(vol_valid) >= 3 else sf(vol_valid.mean()) if len(vol_valid) > 0 else 0
        vol_5d = sf(vol_valid.iloc[-5:].mean()) if len(vol_valid) >= 5 else sf(vol_valid.mean()) if len(vol_valid) > 0 else 0
        vol_10d = sf(vol_valid.iloc[-10:].mean(), 1.0) if len(vol_valid) >= 10 else max(sf(vol_valid.mean(), 1.0), 1.0)
        vol_20d = sf(vol_valid.iloc[-20:].mean(), 1.0) if len(vol_valid) >= 20 else max(sf(vol_valid.mean(), 1.0), 1.0)
        vol_ratio = vol_5d / max(vol_20d, 1.0)
        vol_ratio_3d_10d = vol_3d / max(vol_10d, 1.0)

        # ── Breakouts ──
        high_20 = sf(close.rolling(20, min_periods=15).max().shift(1).iloc[-1]) if len(close) >= 20 else last_close
        breakout_20d = 1 if last_close > high_20 else 0
        high_10 = sf(close.rolling(10, min_periods=8).max().shift(1).iloc[-1]) if len(close) >= 10 else last_close
        breakout_10d = 1 if last_close > high_10 else 0

        # ── Returns ──
        close_5d_ago = sf(close.iloc[-6]) if len(close) >= 6 else last_close
        close_10d_ago = sf(close.iloc[-11]) if len(close) >= 11 else last_close
        close_21d_ago = sf(close.iloc[-22]) if len(close) >= 22 else last_close
        close_63d_ago = sf(close.iloc[-64]) if len(close) >= 64 else last_close
        close_126d_ago = sf(close.iloc[-127]) if len(close) >= 127 else last_close
        ret_5d = ((last_close / close_5d_ago) - 1) * 100 if close_5d_ago > 0 else 0.0
        ret_10d = ((last_close / close_10d_ago) - 1) * 100 if close_10d_ago > 0 else 0.0
        ret_21d = ((last_close / close_21d_ago) - 1) * 100 if close_21d_ago > 0 else 0.0
        ret_63d = ((last_close / close_63d_ago) - 1) * 100 if close_63d_ago > 0 else 0.0
        ret_126d = ((last_close / close_126d_ago) - 1) * 100 if close_126d_ago > 0 else 0.0

        # ── 1D Return ──
        close_1d_ago = sf(close.iloc[-2]) if len(close) >= 2 else last_close
        ret_1d = ((last_close / close_1d_ago) - 1) * 100 if close_1d_ago > 0 else 0.0

        # ── 12-1M Return (Jegadeesh & Titman 1993): 12개월 수익률에��� 최근 1개월 제외 ──
        close_252d_ago = sf(close.iloc[-253]) if len(close) >= 253 else close_126d_ago
        ret_12_1m = ((close_21d_ago / close_252d_ago) - 1) * 100 if close_252d_ago > 0 else ret_126d
        # ── 1Y Return (252 trading days) ──
        ret_252d = ((last_close / close_252d_ago) - 1) * 100 if close_252d_ago > 0 else 0.0

        # ── 36-12M Return (De Bondt & Thaler 1985): 36개월 수익률에서 최근 12개월 제외 ──
        # Long-term reversal signal: 756 거래일(~3년) 전 → 252 거래일(~1년) 전 구간 수익률
        close_756d_ago = sf(close.iloc[-757]) if len(close) >= 757 else None
        if close_756d_ago is not None and close_756d_ago > 0 and close_252d_ago > 0:
            ret_36_12m = ((close_252d_ago / close_756d_ago) - 1) * 100
        else:
            ret_36_12m = None  # 데이터 부족 시 None → 중립 처리

        # ── 3Y / 5Y Annualized Returns ──
        if close_756d_ago is not None and close_756d_ago > 0:
            ret_3y_ann = ((last_close / close_756d_ago) ** (252 / 756) - 1) * 100
        else:
            ret_3y_ann = None
        n_5y = min(1260, len(close) - 1)
        close_1260d_ago = sf(close.iloc[-n_5y - 1]) if n_5y >= 1200 else None
        if close_1260d_ago is not None and close_1260d_ago > 0:
            ret_5y_ann = ((last_close / close_1260d_ago) ** (252 / n_5y) - 1) * 100
        else:
            ret_5y_ann = None

        # ── Realized Volatility (60일 연환산) ──
        daily_rets = close.pct_change().dropna()
        if len(daily_rets) >= 60:
            realized_vol = float(daily_rets.iloc[-60:].std() * np.sqrt(252) * 100)
        elif len(daily_rets) > 5:
            realized_vol = float(daily_rets.std() * np.sqrt(252) * 100)
        else:
            realized_vol = 20.0
        realized_vol = max(realized_vol, 1.0)

        # ── 3Y Annualized Volatility ──
        if len(daily_rets) >= 756:
            vol_3y_ann = float(daily_rets.iloc[-756:].std() * np.sqrt(252) * 100)
        elif len(daily_rets) >= 252:
            vol_3y_ann = float(daily_rets.std() * np.sqrt(252) * 100)
        else:
            vol_3y_ann = None

        # ── Volatility Contraction Ratio (VCR, O'Neil base tightness) ──
        # 최근 5일 변동성 / 과거 40일 변동성 — 1 미만이면 가격 수렴(base 형성)
        if len(daily_rets) >= 40:
            vol_5d_std = float(daily_rets.iloc[-5:].std()) if len(daily_rets) >= 5 else 0.01
            vol_40d_std = float(daily_rets.iloc[-40:].std())
            vcr = vol_5d_std / max(vol_40d_std, 1e-10)
        else:
            vcr = 1.0

        # ── Vol-Adjusted Momentum (AQR 방식) ──
        vol_adj_mom = ret_126d / realized_vol

        # ── Post-Event Drift (PEAD proxy, AQR underreaction) ──
        # 최근 30거래일 중 |1일 수익률| ≥ 3% 인 가장 큰 이벤트일을 식별 →
        # 그 이벤트 이후 누적수익률을 이벤트 방향(부호)에 맞춰 정규화.
        # 양수면 이벤트 방향으로 가격이 underreact 후 계속 표류 = AQR drift signal.
        gap_drift_30d = 0.0
        gap_event_age = 0  # 이벤트 발생 후 경과일 (0=이벤트 없음)
        if len(close) >= 31:
            try:
                rets_30 = close.pct_change().iloc[-30:].fillna(0).values * 100
                if len(rets_30) > 0:
                    max_abs_idx = int(np.argmax(np.abs(rets_30)))
                    max_abs_ret = float(rets_30[max_abs_idx])
                    if abs(max_abs_ret) >= 3.0:
                        # 이벤트일 종가 (close 인덱스 기준): -30 + max_abs_idx
                        event_offset = -30 + max_abs_idx
                        if event_offset < 0 and abs(event_offset) <= len(close):
                            event_close = sf(close.iloc[event_offset])
                            if event_close > 0:
                                drift = ((last_close / event_close) - 1) * 100
                                gap_drift_30d = drift if max_abs_ret > 0 else -drift
                                gap_event_age = -event_offset  # 양의 일수
            except Exception:
                pass

        avg_vol_20d = sf(vol_valid.iloc[-20:].mean()) if len(vol_valid) >= 20 else sf(vol_valid.mean()) if len(vol_valid) > 0 else 0
        avg_price_5d = sf(close.iloc[-5:].mean()) if len(close) >= 5 else last_close
        adv_usd = avg_vol_20d * avg_price_5d

        # ── Strategy-specific indicators ──
        # SMA150 (Minervini SEPA)
        sma150 = close.rolling(150, min_periods=100).mean() if len(close) >= 100 else sma50
        last_sma150 = sf(sma150.iloc[-1])
        above_sma150 = 1 if (last_sma150 > 0 and last_close > last_sma150) else 0
        sma150_20d_ago_v = sf(sma150.iloc[-21]) if len(sma150) >= 21 else last_sma150
        sma150_slope = ((last_sma150 / sma150_20d_ago_v) - 1) * 100 if sma150_20d_ago_v > 0 else 0.0

        # Ichimoku Kinko Hyo
        high_s = ss(df['High']) if 'High' in df.columns else close
        low_s = ss(df['Low']) if 'Low' in df.columns else close
        tenkan = (sf(high_s.iloc[-9:].max()) + sf(low_s.iloc[-9:].min())) / 2 if len(high_s) >= 9 else last_close
        kijun = (sf(high_s.iloc[-26:].max()) + sf(low_s.iloc[-26:].min())) / 2 if len(high_s) >= 26 else last_close
        senkou_a = (tenkan + kijun) / 2
        senkou_b = (sf(high_s.iloc[-52:].max()) + sf(low_s.iloc[-52:].min())) / 2 if len(high_s) >= 52 else last_close
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        ichimoku_above_cloud = 1 if last_close > cloud_top else 0
        ichimoku_below_cloud = 1 if last_close < cloud_bottom else 0
        ichimoku_tk_bull = 1 if tenkan > kijun else 0
        ichimoku_cloud_green = 1 if senkou_a > senkou_b else 0
        close_26_ago_v = sf(close.iloc[-27]) if len(close) >= 27 else last_close
        ichimoku_chikou_bull = 1 if last_close > close_26_ago_v else 0

        # OBV slope (Wyckoff)
        obv_slope = 0.0
        if len(close) >= 20 and len(volume) >= 20:
            try:
                price_chg = close.diff()
                obv_series = (volume * price_chg.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))).cumsum()
                obv_20 = obv_series.iloc[-20:].values
                obv_x = np.arange(len(obv_20))
                obv_fit = np.polyfit(obv_x, obv_20.astype(float), 1)[0]
                obv_slope = obv_fit / max(abs(float(obv_series.iloc[-1])), 1.0) * 100
            except:
                obv_slope = 0.0

        # Distribution days count (Wyckoff / Institutional Flow)
        dist_days = 0
        if len(close) >= 26 and len(volume) >= 26:
            for i in range(-25, 0):
                try:
                    if float(close.iloc[i]) < float(close.iloc[i - 1]) and float(volume.iloc[i]) > vol_20d:
                        dist_days += 1
                except:
                    pass

        # Average close position in candle (Institutional Flow)
        avg_close_pos = 0.5
        if 'High' in df.columns and 'Low' in df.columns and len(df) >= 10:
            try:
                h10 = high_s.iloc[-10:].values.astype(float)
                l10 = low_s.iloc[-10:].values.astype(float)
                c10 = close.iloc[-10:].values.astype(float)
                ranges = h10 - l10
                mask = ranges > 0
                if mask.sum() > 0:
                    avg_close_pos = float(np.mean((c10[mask] - l10[mask]) / ranges[mask]))
            except:
                avg_close_pos = 0.5

        # Darvas Box metrics
        darvas_box_days = 0
        darvas_box_range = 100.0
        darvas_breakout = 0
        if len(close) >= 40:
            try:
                c40 = close.iloc[-40:]
                box_high_val = float(c40.max())
                high_pos = c40.values.argmax()
                since = c40.iloc[high_pos:]
                if len(since) >= 2:
                    box_low_val = float(since.min())
                    darvas_box_days = len(since)
                    darvas_box_range = (box_high_val - box_low_val) / max(box_high_val, 1e-10) * 100
                    darvas_breakout = 1 if last_close >= box_high_val else 0
            except:
                pass

        # Money Flow Index (Institutional Flow)
        mfi = 50.0
        if 'High' in df.columns and 'Low' in df.columns and len(df) >= 15:
            try:
                tp = (high_s + low_s + close) / 3
                raw_mf = tp * volume
                tp_diff = tp.diff()
                pos_mf = sf(raw_mf.where(tp_diff > 0, 0).rolling(14, min_periods=10).sum().iloc[-1])
                neg_mf = sf(raw_mf.where(tp_diff < 0, 0).rolling(14, min_periods=10).sum().iloc[-1], 1.0)
                if neg_mf > 0:
                    mfi = 100 - 100 / (1 + pos_mf / max(neg_mf, 1e-10))
            except:
                mfi = 50.0

        return {
            # ── Long-term (기존) ──
            'above_sma50': above_sma50,
            'above_sma200': above_sma200,
            'golden_cross': golden_cross,
            'sma50_slope': sma50_slope,
            'sma50_slope_was_neg': 1 if slope_20d_ago < 0 and sma50_slope > 0 else 0,
            'sma200_slope': sma200_slope,
            'sma50_sma200_spread': sma50_sma200_spread,
            'trend_age': trend_age,
            'sma50_dist': sma50_dist,
            # ── Short-term (신규) ──
            'above_ema10': above_ema10,
            'above_sma20': above_sma20,
            'ema10_above_sma20': ema10_above_sma20,
            'sma20_slope': sma20_slope,
            'sma20_slope_was_neg': 1 if slope_sma20_10d_ago < 0 and sma20_slope > 0 else 0,
            'trend_age_short': trend_age_short,
            'sma20_dist': sma20_dist,
            'breakout_10d': breakout_10d,
            'vol_ratio_3d_10d': vol_ratio_3d_10d,
            'ret_5d': ret_5d,
            'ret_10d': ret_10d,
            # ── Common ──
            'rsi': rsi,
            'pct_from_high': pct_from_high,
            'range_pct': range_pct,
            'vol_ratio': vol_ratio,
            'breakout_20d': breakout_20d,
            'ret_21d': ret_21d,
            'ret_63d': ret_63d,
            'ret_126d': ret_126d,
            'ret_12_1m': ret_12_1m,
            'ret_1d': ret_1d,
            'ret_252d': ret_252d,
            'ret_36_12m': ret_36_12m,
            'ret_3y_ann': ret_3y_ann,
            'ret_5y_ann': ret_5y_ann,
            'realized_vol': realized_vol,
            'vol_3y_ann': vol_3y_ann,
            'vcr': vcr,
            'vol_adj_mom': vol_adj_mom,
            'gap_drift_30d': gap_drift_30d,
            'gap_event_age': gap_event_age,
            'adv_usd': adv_usd,
            'last_close': last_close,
            # ── Strategy indicators ──
            'above_sma150': above_sma150,
            'sma150_slope': sma150_slope,
            'ichimoku_above_cloud': ichimoku_above_cloud,
            'ichimoku_below_cloud': ichimoku_below_cloud,
            'ichimoku_tk_bull': ichimoku_tk_bull,
            'ichimoku_cloud_green': ichimoku_cloud_green,
            'ichimoku_chikou_bull': ichimoku_chikou_bull,
            'obv_slope': obv_slope,
            'dist_days': dist_days,
            'avg_close_pos': avg_close_pos,
            'darvas_box_days': darvas_box_days,
            'darvas_box_range': darvas_box_range,
            'darvas_breakout': darvas_breakout,
            'mfi': mfi,
        }

    # ── Blending weights (class-level constants) ──
    W_SHORT_TCS = 0.40;  W_LONG_TCS = 0.60
    W_SHORT_TFS = 0.50;  W_LONG_TFS = 0.50
    W_SHORT_RSS = 0.35;  W_LONG_RSS = 0.65

    # ── Continuous scoring helper (sigmoid-like) ──
    # Binary 0/1 threshold 대신 거리 기반 연속 점수 (노이즈 감소)
    # dist_pct = (close - MA) / MA * 100
    # buffer_pct = 임계값 주변 완충 구간 (%)
    @staticmethod
    def _cont_score(dist_pct, max_pts, buffer_pct=2.0):
        """Continuous score: 0 at -buffer, max_pts at +buffer, linear between."""
        if dist_pct >= buffer_pct:
            return max_pts
        elif dist_pct <= -buffer_pct:
            return 0
        else:
            return max_pts * (dist_pct + buffer_pct) / (2.0 * buffer_pct)

    # ── TCS: Trend Continuation ──
    @classmethod
    def score_tcs_short(cls, raw):
        pts = 0
        # Continuous: EMA10/SMA20 거리 기반 (±2% 버퍼)
        pts += cls._cont_score(raw['sma20_dist'], 20, 2.0)   # above_sma20 대체
        ema10_dist = raw.get('sma20_dist', 0) * 0.8  # EMA10은 SMA20보다 가격에 가까움 (근사)
        pts += cls._cont_score(ema10_dist, 20, 1.5)           # above_ema10 대체
        # EMA10 > SMA20: sma20_dist가 양수이면 대체로 EMA10 > SMA20
        pts += cls._cont_score(raw['sma20_dist'], 20, 1.0)    # ema10_above_sma20 대체
        # Slope: 이미 연속값
        if raw['sma20_slope'] > 0:    pts += 20
        # Trend age: 이미 연속값이나 5일 기준 완화
        if raw['trend_age_short'] > 5: pts += 20
        elif raw['trend_age_short'] > 2: pts += 10
        return min(100, round(pts))

    @classmethod
    def score_tcs_long(cls, raw):
        pts = 0
        # Continuous: SMA50 거리 기반 (±3% 버퍼)
        pts += cls._cont_score(raw['sma50_dist'], 20, 3.0)    # above_sma50 대체
        # Golden cross: SMA50-SMA200 spread 기반 (±2% 버퍼)
        pts += cls._cont_score(raw['sma50_sma200_spread'], 20, 2.0)  # golden_cross 대체
        # Slope: 이미 연속값
        if raw['sma50_slope'] > 0:    pts += 20
        # Trend age: 단계적 점수
        ta = raw['trend_age']
        if ta > 20:     pts += 20
        elif ta > 10:   pts += 12
        elif ta > 5:    pts += 5
        # SMA200: 거리 기반 (±5% 버퍼)
        sma200_dist = raw.get('sma50_dist', 0) + raw.get('sma50_sma200_spread', 0)  # 근사
        pts += cls._cont_score(sma200_dist, 10, 5.0)          # above_sma200 대체
        if raw['sma200_slope'] > 0:   pts += 10
        return min(100, round(pts))

    @classmethod
    def score_tcs(cls, raw):
        s = cls.score_tcs_short(raw)
        l = cls.score_tcs_long(raw)
        return round(cls.W_SHORT_TCS * s + cls.W_LONG_TCS * l), s, l

    # ── TFS: Trend Formation ──
    @classmethod
    def score_tfs_short(cls, raw):
        pts = 0
        age = raw['trend_age_short']
        # Continuous: SMA20 돌파 강도 × 신선도
        sma20_d = raw['sma20_dist']
        if sma20_d > 0 and 1 <= age <= 5:
            pts += cls._cont_score(sma20_d, 30, 3.0)  # 돌파 거리 기반
        elif sma20_d > 0 and 6 <= age <= 10:
            pts += cls._cont_score(sma20_d, 15, 3.0)
        # Volume: 연속 점수 (binary 1.3 → continuous 1.0~2.0 구간)
        vr3 = raw['vol_ratio_3d_10d']
        if vr3 > 2.0:    pts += 25
        elif vr3 > 1.5:  pts += 20
        elif vr3 > 1.3:  pts += 15
        elif vr3 > 1.1:  pts += 8    # 기존에 0이던 구간에 부분 점수
        # Breakout: 연속 점수 (거리 기반)
        pfh_10d = raw.get('pct_from_high', -10)  # 10일 고점 근접도 근사
        pts += cls._cont_score(pfh_10d + 5, 25, 5.0)  # -5%~0% → 0~25
        # Slope reversal: 유지
        if raw['sma20_slope_was_neg'] and raw['sma20_slope'] > 0: pts += 20
        return min(100, round(pts))

    @classmethod
    def score_tfs_long(cls, raw):
        pts = 0
        age = raw['trend_age']
        sma50_d = raw['sma50_dist']
        if sma50_d > 0 and 1 <= age <= 15:
            pts += cls._cont_score(sma50_d, 30, 5.0)
        elif sma50_d > 0 and 16 <= age <= 30:
            pts += cls._cont_score(sma50_d, 15, 5.0)
        # Volume: 단계적 (binary → graduated)
        vr = raw['vol_ratio']
        if vr > 1.8:    pts += 25
        elif vr > 1.4:  pts += 18
        elif vr > 1.2:  pts += 12
        elif vr > 1.05: pts += 5
        # Breakout 20d: 유지 (20일 윈도우라 상대적 안정)
        if raw['breakout_20d']:     pts += 25
        # Slope reversal: 유지
        if raw['sma50_slope_was_neg'] and raw['sma50_slope'] > 0: pts += 20
        return min(100, round(pts))

    @classmethod
    def score_tfs(cls, raw):
        s = cls.score_tfs_short(raw)
        l = cls.score_tfs_long(raw)
        return round(cls.W_SHORT_TFS * s + cls.W_LONG_TFS * l), s, l

    # ── OER: Overextension Risk (unified, both timeframes) ──
    @staticmethod
    def score_oer(raw):
        pts = 0
        # Short-term overextension (SMA20)
        sd = raw['sma20_dist']
        if sd > 8:    pts += 15
        elif sd > 5:  pts += 8
        # Long-term overextension (SMA50)
        ld = raw['sma50_dist']
        if ld > 15:   pts += 35
        elif ld > 10: pts += 25
        elif ld > 5:  pts += 12
        # RSI
        rsi = raw['rsi']
        if rsi > 80:   pts += 25
        elif rsi > 70: pts += 12
        # 52-week high proximity
        if raw['pct_from_high'] > -2: pts += 15
        # Long-term reversal risk (36-12M, De Bondt & Thaler)
        # reversal_pctile is injected by compute_percentile_ranks; absent in single-ticker mode
        rev = raw.get('reversal_pctile')
        if rev is not None:
            if rev >= 90:   pts += 15   # 상위 10%: 강한 장기 과열
            elif rev >= 80: pts += 8    # 상위 20%: 경미한 장기 과열
        return min(100, pts)

    # ── RSS: Percentile Ranks ──
    @staticmethod
    def compute_percentile_ranks(all_raw: Dict[str, dict]) -> Dict[str, dict]:
        tickers = list(all_raw.keys())
        # All indicators to rank cross-sectionally
        indicators = ['sma50_slope', 'trend_age', 'sma50_dist', 'rsi',
                       'range_pct', 'vol_ratio', 'ret_21d', 'ret_63d', 'ret_126d',
                       'ret_12_1m', 'vol_adj_mom',
                       # Short-term additions
                       'sma20_slope', 'ret_5d', 'ret_10d', 'vol_ratio_3d_10d',
                       # URS inputs (AQR underreaction)
                       'gap_drift_30d']
        arrays = {ind: np.array([all_raw[t].get(ind, 0.0) for t in tickers], dtype=float) for ind in indicators}

        # ── ret_36_12m: None 허용 지표 (데이터 3년 미만 시 None) ──
        ret_36_12m_vals = [all_raw[t].get('ret_36_12m') for t in tickers]
        valid_36_12m = np.array([v for v in ret_36_12m_vals if v is not None], dtype=float)

        # ── Category-level stats (URS LeadLagGap & PeerDispersion 용) ──
        # raw['_category']는 run_scan에서 주입; 없으면 'Unknown'
        cat_to_tickers = defaultdict(list)
        for t in tickers:
            cat = all_raw[t].get('_category', 'Unknown')
            cat_to_tickers[cat].append(t)

        cat_stats = {}  # cat -> {'mean_ret_63d', 'rss_std'} — RSS는 1차 패스 후
        for cat, tlist in cat_to_tickers.items():
            ret63_arr = np.array([all_raw[t].get('ret_63d', 0.0) for t in tlist], dtype=float)
            cat_stats[cat] = {
                'mean_ret_63d': float(np.nanmean(ret63_arr)) if len(ret63_arr) > 0 else 0.0,
                'n': len(tlist),
            }

        ranks = {}
        # ── Pass 1: 모든 일반 percentile + RSS 계산 ──
        for i, t in enumerate(tickers):
            r = {}
            for ind in indicators:
                r[ind + '_pctile'] = pct_rank(arrays[ind][i], arrays[ind])

            v36 = ret_36_12m_vals[i]
            if v36 is not None and len(valid_36_12m) >= 5:
                r['reversal_pctile'] = pct_rank(v36, valid_36_12m)
            else:
                r['reversal_pctile'] = 50.0
            all_raw[t]['reversal_pctile'] = r['reversal_pctile']

            r['rss_long'] = (r['ret_12_1m_pctile'] + r['ret_63d_pctile']
                             + r['vol_adj_mom_pctile'] + r['sma50_slope_pctile']
                             + r['range_pct_pctile']) / 5.0
            r['rss_short'] = (r['ret_5d_pctile'] + r['ret_10d_pctile']
                              + r['ret_21d_pctile'] + r['sma20_slope_pctile']
                              + r['vol_ratio_3d_10d_pctile']) / 5.0

            ws = NaiveDiscoveryDetector.W_SHORT_RSS
            wl = NaiveDiscoveryDetector.W_LONG_RSS
            r['rss'] = ws * r['rss_short'] + wl * r['rss_long']
            ranks[t] = r

        # ── Pass 2: 카테고리 RSS 평균/분산 ──
        for cat, tlist in cat_to_tickers.items():
            rss_arr = np.array([ranks[t]['rss'] for t in tlist], dtype=float)
            cat_stats[cat]['rss_mean'] = float(np.nanmean(rss_arr)) if len(rss_arr) > 0 else 50.0
            cat_stats[cat]['rss_std'] = float(np.nanstd(rss_arr)) if len(rss_arr) > 1 else 0.0

        # ── Pass 3: URS 4 컴포넌트 (raw → cross-sectional pctile → blend) ──
        # 1) LeadLagGap_raw: 카테고리 평균 ret_63d - 본인 ret_63d, 단 (a) 카테고리가 상승추세
        #    (mean>0) AND (b) 본인 vol_ratio_3d_10d > 1.0 (어텐션 시작) 일 때만 양수 부여.
        # 2) AttentionPriceGap_raw: vol_ratio_3d_10d_pctile - ret_5d_pctile (어텐션 高 / 가격反응 低)
        # 3) FundamentalProxyDrift_raw: gap_drift_30d_pctile (이미 계산됨)
        # 4) PeerDispersion_raw: 카테고리 RSS 표준편차 (분산이 클수록 leader/laggard 분화)
        ll_raw = np.zeros(len(tickers))
        ap_raw = np.zeros(len(tickers))
        pd_raw = np.zeros(len(tickers))
        for i, t in enumerate(tickers):
            cat = all_raw[t].get('_category', 'Unknown')
            cs = cat_stats.get(cat, {'mean_ret_63d': 0.0, 'rss_std': 0.0})
            cat_mean_63d = cs['mean_ret_63d']
            ticker_ret_63d = all_raw[t].get('ret_63d', 0.0)
            attn = all_raw[t].get('vol_ratio_3d_10d', 1.0)
            # LeadLagGap: 양수=laggard (잡아야 할 underreaction)
            if cat_mean_63d > 0 and attn > 1.0:
                ll_raw[i] = cat_mean_63d - ticker_ret_63d
            else:
                ll_raw[i] = 0.0  # 조건 불충족 시 중립 = 평균
            ap_raw[i] = ranks[t]['vol_ratio_3d_10d_pctile'] - ranks[t]['ret_5d_pctile']
            pd_raw[i] = cs.get('rss_std', 0.0)

        for i, t in enumerate(tickers):
            r = ranks[t]
            ll_pctile = pct_rank(ll_raw[i], ll_raw)
            ap_pctile = pct_rank(ap_raw[i], ap_raw)
            fd_pctile = r['gap_drift_30d_pctile']  # 이미 양수=드리프트 강함 방향
            pd_pctile = pct_rank(pd_raw[i], pd_raw)

            urs = (0.40 * ll_pctile + 0.30 * ap_pctile
                   + 0.20 * fd_pctile + 0.10 * pd_pctile)
            r['urs_leadlag'] = round(ll_pctile, 1)
            r['urs_attn_gap'] = round(ap_pctile, 1)
            r['urs_drift'] = round(fd_pctile, 1)
            r['urs_dispersion'] = round(pd_pctile, 1)
            r['urs'] = round(urs, 1)

        return ranks

    # ── Helper: Parse previous classification → (short_dir, long_dir, was_overextended) ──
    @staticmethod
    def _parse_prev_class(cls_str):
        """Hysteresis 적용을 위해 이전 분류로부터 방향 정보 복원."""
        if not cls_str:
            return (None, None, False)
        # Override 분류는 하단 base 추정
        if cls_str == "🟡 OVEREXTENDED":
            return ("UP", "UP", True)  # bullish base
        if cls_str == "🔵 FORMATION":
            return ("UP", "UP", False)
        if cls_str == "🔴 CYCLE_PEAK":
            return ("DOWN", "UP", False)
        if cls_str == "🟤 EXHAUSTING":
            return ("FLAT", "UP", False)
        if cls_str == "🟦 LAGGING_CATCHUP":
            return ("FLAT", "UP", False)
        # Base 3x3 matrix reverse mapping
        REVERSE_MATRIX = {
            "🟢 CONTINUATION": ("UP", "UP"),
            "🔵 RECOVERY": ("UP", "FLAT"),
            "🟣 COUNTER_RALLY": ("UP", "DOWN"),
            "🟡 CONSOLIDATION": ("FLAT", "UP"),
            "🟠 NEUTRAL": ("FLAT", "FLAT"),
            "🟤 FADING": ("FLAT", "DOWN"),
            "🔶 PULLBACK": ("DOWN", "UP"),
            "⚠️ WEAKENING": ("DOWN", "FLAT"),
            "⬇️ DOWNTREND": ("DOWN", "DOWN"),
        }
        if cls_str in REVERSE_MATRIX:
            s, l = REVERSE_MATRIX[cls_str]
            return (s, l, False)
        return (None, None, False)

    # ── Classify: Dual-Timeframe 3×3 Matrix + Overrides ──
    @staticmethod
    def classify(raw, tcs_short, tcs_long, tfs_short, tfs_long, oer,
                 adaptive=None, urs=None, buffer_mode="adaptive",
                 prev_classification=None):
        """Classify ticker. adaptive=dict with optional threshold overrides.
        urs=Underreaction Score (0-100); LAGGING_CATCHUP 분류에 사용 (선택).

        buffer_mode:
          - "fixed":    legacy hard rule (short ±0.5%, long ±1%)
          - "adaptive": volatility-adjusted buffer (Phase 1 — default)

        prev_classification: 이전 시점 분류 문자열 (Universal Hysteresis 활성화).
          - None이면 일반 enter 임계값만 적용 (legacy)
          - 제공되면 enter(strict) / exit(loose) 분리 임계값으로 hysteresis 적용
            → 일별 임계값 진동 노이즈 방지
        """
        # Adaptive thresholds (or defaults)
        oer_thresh = (adaptive or {}).get('oer_overextended', 60)
        tfs_thresh = (adaptive or {}).get('tfs_formation', 50)

        # ── Phase 1: Volatility-Adjusted Buffer ──
        # 자산별 변동성에 비례하여 buffer 크기 조정
        # - 저변동성 (TLT 등): 좁은 buffer로 작은 움직임 포착
        # - 고변동성 (TSLA 등): 넓은 buffer로 whipsaw 방지
        #
        # realized_vol는 daily σ (decimal). 20-day → ×√20, 60-day → ×√60
        # Multiplier 0.4 (short), 0.6 (long): 데이터 분포 분석에 기반.
        # max() bound: 매우 저변동성 자산에서도 최소 buffer 보장.
        if buffer_mode == "adaptive":
            rv = sf(raw.get('realized_vol', 0.02))
            # realized_vol는 percent로 저장됨 (예: 30 = 30%/yr).
            # daily σ = annualized σ / √252
            daily_sigma_pct = rv / np.sqrt(252) if rv > 1 else rv * 100  # robust to either format
            short_buf = max(0.3, 0.4 * daily_sigma_pct * np.sqrt(20))
            long_buf  = max(0.7, 0.6 * daily_sigma_pct * np.sqrt(60))
            # Cap at sensible upper bound to avoid degenerate cases
            short_buf = min(short_buf, 3.0)
            long_buf  = min(long_buf, 6.0)
        else:
            short_buf = 0.5
            long_buf  = 1.0

        # ── Universal Hysteresis (Option B) ──
        # 이전 분류 정보 → enter/exit 임계값 분리하여 임계점 진동 방지
        prev_short, prev_long, prev_was_oe = NaiveDiscoveryDetector._parse_prev_class(prev_classification)
        # Hysteresis 비율: enter는 strict, exit는 loose (40~50%)
        short_buf_exit = short_buf * 0.4
        long_buf_exit  = long_buf * 0.5

        sma20_d = raw['sma20_dist']
        sma20_s = raw['sma20_slope']

        # Step 1: Short-term direction with hysteresis
        if prev_short == "UP":
            # Loose exit from UP: 거리만 줄어도 slope가 양수면 UP 유지
            if sma20_d > short_buf_exit and sma20_s > 0:
                short_dir = "UP"
            elif sma20_d < -short_buf and sma20_s < 0:
                short_dir = "DOWN"
            else:
                short_dir = "FLAT"
        elif prev_short == "DOWN":
            if sma20_d < -short_buf_exit and sma20_s < 0:
                short_dir = "DOWN"
            elif sma20_d > short_buf and sma20_s > 0:
                short_dir = "UP"
            else:
                short_dir = "FLAT"
        else:  # FLAT or unknown — strict enter
            if sma20_d > short_buf and sma20_s > 0:
                short_dir = "UP"
            elif sma20_d < -short_buf and sma20_s < 0:
                short_dir = "DOWN"
            else:
                short_dir = "FLAT"

        # Step 2: Long-term direction with hysteresis
        sma50_d = raw['sma50_dist']
        spread = raw['sma50_sma200_spread']
        slope50 = raw['sma50_slope']

        if prev_long == "UP":
            if sma50_d > long_buf_exit and (spread > 0 or slope50 > 0):
                long_dir = "UP"
            elif sma50_d < -long_buf and (spread < 0 or slope50 < 0):
                long_dir = "DOWN"
            else:
                long_dir = "FLAT"
        elif prev_long == "DOWN":
            if sma50_d < -long_buf_exit and (spread < 0 or slope50 < 0):
                long_dir = "DOWN"
            elif sma50_d > long_buf and (spread > 0 or slope50 > 0):
                long_dir = "UP"
            else:
                long_dir = "FLAT"
        else:  # FLAT or unknown — strict enter
            if sma50_d > long_buf and (spread > 0 or slope50 > 0):
                long_dir = "UP"
            elif sma50_d < -long_buf and (spread < 0 or slope50 < 0):
                long_dir = "DOWN"
            else:
                long_dir = "FLAT"

        # Step 3: 3×3 Matrix
        MATRIX = {
            ("UP",   "UP"):   "🟢 CONTINUATION",
            ("UP",   "FLAT"): "🔵 RECOVERY",
            ("UP",   "DOWN"): "🟣 COUNTER_RALLY",
            ("FLAT", "UP"):   "🟡 CONSOLIDATION",
            ("FLAT", "FLAT"): "🟠 NEUTRAL",
            ("FLAT", "DOWN"): "🟤 FADING",
            ("DOWN", "UP"):   "🔶 PULLBACK",
            ("DOWN", "FLAT"): "⚠️ WEAKENING",
            ("DOWN", "DOWN"): "⬇️ DOWNTREND",
        }
        base_cls = MATRIX[(short_dir, long_dir)]

        # ─────────────────────────────────────────────────────────────
        # Step 4: Overrides — 우선순위 (강한 위험 → 약한 신호 순)
        # ─────────────────────────────────────────────────────────────

        # (a) CYCLE_PEAK: 36-12M 상위 극단 + 12-1M 절대 둔화 + 단기 미상승.
        # FIX #2: 비례 비교(ret_12_1m < ret_36_12m * 0.3) 제거 — 부호 반전 트랩 회피.
        # 절대 임계로 변경: ret_36_12m > 30 (3년 구간 의미있는 상승) AND ret_12_1m < 5 (최근 1년 둔화)
        # FIX #7: short_dir != UP을 short_dir == DOWN으로 좁힘 (FLAT은 신선한 매수 기회 가능)
        rev = raw.get('reversal_pctile')
        ret_36_12m = raw.get('ret_36_12m')
        if (rev is not None and rev >= 85 and ret_36_12m is not None
                and ret_36_12m > 30 and raw['ret_12_1m'] < 5
                and short_dir == "DOWN"):
            return "🔴 CYCLE_PEAK"

        # (b) EXHAUSTING: 오랜 상승 추세 + 단기 명확한 둔화.
        # FIX #2: 비례 비교(ret_21d < ret_63d/3) 제거.
        # 절대 임계: ret_63d > 5 (의미있는 상승) AND ret_21d < 0 (최근 1개월 음전)
        if (raw['trend_age'] > 60 and raw['ret_63d'] > 5
                and raw['ret_21d'] < 0 and long_dir == "UP"):
            return "🟤 EXHAUSTING"

        # (c) OVEREXTENDED: 과열 + 매수 후보 base.
        # FIX #6: PULLBACK과 COUNTER_RALLY도 가드 대상에 포함.
        # CONTINUATION은 강한 모멘텀이므로 OER 체크 적용 (의도적).
        # Hysteresis: 이전이 OVEREXTENDED였다면 진입 60 → 이탈 50으로 완화.
        oer_threshold_for_check = (oer_thresh - 10) if prev_was_oe else oer_thresh
        if oer >= oer_threshold_for_check and base_cls in (
                "🟢 CONTINUATION", "🔵 RECOVERY", "🟡 CONSOLIDATION",
                "🔶 PULLBACK", "🟣 COUNTER_RALLY"):
            return "🟡 OVEREXTENDED"

        # (d) FORMATION: 새 추세 형성 초기 + 거래량/돌파 확인 + 장기 우호적.
        # FIX #3: CONTINUATION이면 절대 강등하지 않음 (강한 신호 보존).
        # FIX #4: trend_age_short<=5 노이즈 → trend_age_short<=10 로 완화하되,
        #         거래량(vol_ratio_3d_10d > 1.3) 또는 20일 돌파 확인 필수.
        # FIX #9: long_dir UP 또는 FLAT 모두 허용 (RECOVERY 단계의 신규 추세도 FORMATION).
        breakout_confirm = (raw.get('breakout_20d', 0) == 1
                            or raw.get('vol_ratio_3d_10d', 1.0) > 1.3)
        if (base_cls != "🟢 CONTINUATION"
                and tfs_short >= tfs_thresh
                and raw['trend_age_short'] <= 10
                and breakout_confirm
                and long_dir in ("UP", "FLAT")
                and short_dir == "UP"):
            return "🔵 FORMATION"

        # (e) LAGGING_CATCHUP (NEW, AQR underreaction): URS 상위 + 단기 횡보/약세 + 장기 미하락.
        # FIX #10: 카테고리 leader가 먼저 움직였고 본인은 아직 안 움직인 종목 포착.
        # 조건: URS≥75 (강한 underreaction signal) + short_dir != UP (이미 안 움직임)
        #       + long_dir != DOWN (장기 약세 종목은 catchup 신뢰 낮음)
        if (urs is not None and urs >= 75
                and short_dir != "UP" and long_dir != "DOWN"
                and base_cls in ("🟡 CONSOLIDATION", "🟠 NEUTRAL", "🔶 PULLBACK")):
            return "🟦 LAGGING_CATCHUP"

        return base_cls

    @staticmethod
    def score_urs(ranks=None):
        """URS (Underreaction Score, 0-100). Cross-sectional 계산이 필수이므로
        compute_percentile_ranks가 ranks dict에 'urs' 를 미리 주입.
        ranks 없는 single-ticker 모드에서는 50 (중립) 반환."""
        if ranks and 'urs' in ranks:
            return float(ranks['urs'])
        return 50.0

    @staticmethod
    def composite(tcs, tfs, rss, oer, urs=50.0):
        # AQR underreaction (URS) 15% 비중 추가, TCS/TFS/RSS 비례 축소
        return round(0.30 * tcs + 0.25 * tfs + 0.30 * rss + 0.15 * urs, 1)

    # ── O'Neil (CANSLIM) Long / Short Signals ──
    # William O'Neil 방법론 기반 매수/매도 시그널 스코어링 (0-100)
    # Long: 피벗 근접 + 거래량 확인 + RS강도 + MA구조 + 베이스 돌파
    # Short: 지지선 이탈 + 하락 거래량 + RS약세 + MA붕괴 + 추세 악화

    @staticmethod
    def score_oneil_long(raw, ranks=None):
        """O'Neil Long Score (0-100): 매수 시그널 강도"""
        pts = 0

        # (1) Pivot Proximity — 52주 고점 근접도 (25점)
        # O'Neil: 적정 매수 시점은 피벗(52주 고점) 돌파 또는 5% 이내
        pfh = raw['pct_from_high']
        if pfh > -2:    pts += 25   # 고점 2% 이내: 돌파 임박/진행
        elif pfh > -5:  pts += 18   # 5% 이내: 베이스 상단
        elif pfh > -10: pts += 8    # 10% 이내: 정상 조정 범위

        # (2) Volume Surge — 거래량 확인 (20점)
        # O'Neil: 돌파 시 거래량 50%+ 증가 필수
        vr = raw['vol_ratio']       # 5d / 20d
        vr3 = raw['vol_ratio_3d_10d']  # 3d / 10d (더 즉각적)
        best_vr = max(vr, vr3)
        if best_vr > 2.0:   pts += 20   # 100%+ 거래량 폭증
        elif best_vr > 1.5: pts += 15   # 50%+ 증가 (O'Neil 기준)
        elif best_vr > 1.2: pts += 8    # 20%+ 증가

        # (3) Relative Strength — RS Rating (20점)
        # O'Neil: RS 80 이상인 종목만 매수 대상
        if ranks:
            rs = ranks.get('rss', 50)
        else:
            rs = min(100, max(0, 50 + (raw['ret_63d'] + raw['ret_126d']) / 4))
        if rs >= 85:   pts += 20
        elif rs >= 75: pts += 15
        elif rs >= 60: pts += 8

        # (4) MA Structure — 이동평균 구조 (20점)
        # O'Neil: 50일선/200일선 위, 골든크로스 확인
        if raw['above_sma50']:   pts += 6
        if raw['above_sma200']:  pts += 5
        if raw['golden_cross']:  pts += 5
        if raw['sma50_slope'] > 0: pts += 4   # 50일선 상승 중

        # (5) Base Breakout Quality — 베이스 형성 후 돌파 (15점)
        # O'Neil: 타이트한 가격 수렴(VCR < 0.8) 후 돌파가 가장 신뢰도 높음
        vcr = raw.get('vcr', 1.0)
        has_breakout = raw['breakout_20d'] or raw['breakout_10d']
        if vcr < 0.6 and has_breakout:    pts += 15   # 극도로 타이트 + 돌파
        elif vcr < 0.8 and has_breakout:  pts += 12   # 타이트 + 돌파
        elif has_breakout:                pts += 6    # 돌파만 (수렴 없이)
        elif vcr < 0.7:                   pts += 4    # 수렴 중 (돌파 대기)

        return min(100, pts)

    @staticmethod
    def score_oneil_short(raw, ranks=None):
        """O'Neil Short Score (0-100): 매도/공매도 시그널 강도"""
        pts = 0

        # (1) Support Breakdown — 지지선 이탈 (25점)
        # O'Neil: 50일선/200일선 하향 이탈은 기관 매도 신호
        if not raw['above_sma50'] and not raw['above_sma200']:
            pts += 25   # 양대 지지선 모두 이탈
        elif not raw['above_sma50']:
            pts += 15   # 50일선 이탈
        elif not raw['above_sma20']:
            pts += 5    # 20일선 이탈 (경고)

        # (2) Volume on Decline — 하락 시 거래량 증가 (20점)
        # O'Neil: 하락 시 거래량 급증 = 기관 분배(Distribution)
        declining = raw['ret_5d'] < 0
        vr = max(raw['vol_ratio'], raw['vol_ratio_3d_10d'])
        if declining and vr > 2.0:   pts += 20   # 급락 + 대량거래
        elif declining and vr > 1.5: pts += 15   # 하락 + 거래량 증가
        elif declining and vr > 1.2: pts += 8    # 하락 + 소폭 거래량
        elif not declining and vr < 0.7: pts += 5  # 반등에 거래량 미미 (약한 매수)

        # (3) RS Weakness — 상대강도 약세 (20점)
        # O'Neil: RS 하위 종목은 시장 하락 시 가장 큰 폭으로 하락
        if ranks:
            rs = ranks.get('rss', 50)
        else:
            rs = min(100, max(0, 50 + (raw['ret_63d'] + raw['ret_126d']) / 4))
        if rs <= 15:   pts += 20
        elif rs <= 25: pts += 15
        elif rs <= 40: pts += 8

        # (4) MA Deterioration — 이동평균 붕괴 (20점)
        # O'Neil: 데스크로스 + 하향 기울기 = 기관 이탈 확인
        if not raw['golden_cross']:       pts += 6    # 데스크로스
        if raw['sma50_slope'] < 0:        pts += 6    # 50일선 하락
        if raw['sma200_slope'] < 0:       pts += 4    # 200일선 하락
        if raw['sma20_slope'] < 0:        pts += 4    # 20일선 하락

        # (5) Trend Deterioration — 추세 악화 (15점)
        # 52주 저점 근접 + 단기 모멘텀 악화
        range_pct = raw['range_pct']
        if range_pct < 15:    pts += 10   # 52주 저점 근접
        elif range_pct < 30:  pts += 5    # 하위 30% 구간
        if raw['ret_21d'] < -5 and raw['ret_63d'] < -10:
            pts += 5   # 지속적 하락 모멘텀

        return min(100, pts)

    # ── Event-Driven Risk Assessment ──

    @staticmethod
    def detect_event_flag(raw):
        """이벤트 드리븐 스파이크 감지.
        단기 수익이 중기 대비 비정상적이고 거래량 급증 + 초기 추세 → EVENT flag.
        Returns: (is_event: bool, risk_reasons: list[str])
        """
        reasons = []

        # (1) 단기 > 중기 가속: ret_5d가 ret_21d 전체보다 큼
        if raw['ret_5d'] > 0 and raw['ret_21d'] > 0:
            if raw['ret_5d'] > raw['ret_21d'] * 0.7:
                reasons.append("spike_5d>21d")

        # (2) 거래량 급증 (3일/10일 비율 1.5x+)
        if raw['vol_ratio_3d_10d'] > 1.5:
            reasons.append(f"vol_surge_{raw['vol_ratio_3d_10d']:.1f}x")

        # (3) 초기 추세 (SMA20 위 10일 이내)
        if raw['trend_age_short'] <= 10:
            reasons.append(f"early_trend_{raw['trend_age_short']}d")

        # (4) 변동성 확대 (VCR > 1.3: 최근 변동성이 기저 대비 급등)
        vcr = raw.get('vcr', 1.0)
        if vcr > 1.3:
            reasons.append(f"vol_expand_{vcr:.2f}")

        # EVENT = 위 조건 중 2개 이상 동시 충족
        is_event = len(reasons) >= 2
        return is_event, reasons

    @staticmethod
    def compute_structural_quality(raw, ranks=None):
        """구조적 모멘텀 품질 점수 (0-100).
        변동성 조정 모멘텀 + 장기 추세 안정성 기반.
        이벤트 드리븐과 구조적 모멘텀을 구분하는 보조 지표.
        """
        pts = 0

        # (1) Vol-adjusted momentum 백분위 (30점)
        if ranks:
            vam_p = ranks.get('vol_adj_mom_pctile', 50)
        else:
            vam_p = min(100, max(0, 50 + raw.get('vol_adj_mom', 0) * 10))
        if vam_p >= 80:   pts += 30
        elif vam_p >= 60: pts += 20
        elif vam_p >= 40: pts += 10

        # (2) 장기 추세 안정성: trend_age (25점)
        ta = raw['trend_age']
        if ta >= 40:   pts += 25   # 2개월+ 안정 추세
        elif ta >= 20: pts += 18   # 1개월+
        elif ta >= 10: pts += 10   # 2주+
        elif ta >= 5:  pts += 5

        # (3) VCR 안정성: 낮을수록 베이스 형성 완료 (20점)
        vcr = raw.get('vcr', 1.0)
        if vcr < 0.6:   pts += 20
        elif vcr < 0.8:  pts += 15
        elif vcr < 1.0:  pts += 8
        # VCR > 1.3 = 변동성 확대 → 0점

        # (4) 12개월 모멘텀 존재 (25점)
        ret_12_1m = raw.get('ret_12_1m', 0)
        if ret_12_1m > 30:   pts += 25
        elif ret_12_1m > 15: pts += 18
        elif ret_12_1m > 5:  pts += 10
        elif ret_12_1m > 0:  pts += 5

        return min(100, pts)

    @staticmethod
    def compute_alpha_potential(raw, ranks=None):
        """Alpha Potential Score (0-100).
        분류(classification)에 무관하게 숨겨진 상승 잠재력을 포착.
        매집 품질 + 변동성 셋업 + 모멘텀 구조 + 상대강도 + 돌파 근접도.
        """
        pts = 0

        # (1) Accumulation Quality (25점) — 기관 매집 시그널
        obv_s = raw.get('obv_slope', 0)
        close_pos = raw.get('avg_close_pos', 0.5)
        mfi = raw.get('mfi', 50)
        # OBV 상승 + 가격 횡보/하락 = 괴리 매집
        price_flat_or_down = raw.get('ret_21d', 0) <= 2
        if obv_s > 1.0 and price_flat_or_down:
            pts += 10  # strong quiet accumulation
        elif obv_s > 0.3 and price_flat_or_down:
            pts += 6
        elif obv_s > 0:
            pts += 3
        # Close position (고점 마감 지속)
        if close_pos >= 0.75:   pts += 8
        elif close_pos >= 0.6:  pts += 5
        elif close_pos >= 0.5:  pts += 2
        # MFI (자금 유입)
        if mfi >= 65:   pts += 7
        elif mfi >= 50: pts += 4
        elif mfi >= 40: pts += 2

        # (2) Volatility Setup (20점) — 에너지 축적
        vcr = raw.get('vcr', 1.0)
        dist_d = raw.get('dist_days', 0)
        if vcr < 0.5:    pts += 12
        elif vcr < 0.65:  pts += 9
        elif vcr < 0.8:   pts += 5
        elif vcr < 1.0:   pts += 2
        # Distribution days 부재 = 공급 압력 없음
        if dist_d <= 1:   pts += 8
        elif dist_d <= 3: pts += 5
        elif dist_d <= 5: pts += 2

        # (3) Momentum Structure (25점) — 장기 추세 인프라 건재
        if raw.get('golden_cross', 0):     pts += 7  # SMA50 > SMA200
        if raw.get('sma50_slope', 0) > 0:  pts += 7  # SMA50 기울기 양
        ret_12_1m = raw.get('ret_12_1m', 0)
        if ret_12_1m > 20:    pts += 6
        elif ret_12_1m > 5:   pts += 4
        elif ret_12_1m > 0:   pts += 2
        if raw.get('sma200_slope', 0) > 0: pts += 5  # SMA200 기울기 양

        # (4) Relative Strength (15점) — 유니버스 대비 위치
        if ranks:
            rss_p = ranks.get('rss', 50)
        else:
            avg_r = (raw.get('ret_21d', 0) + raw.get('ret_63d', 0) + raw.get('ret_126d', 0)) / 3
            rss_p = min(100, max(0, 50 + avg_r * 2))
        if rss_p >= 75:   pts += 10
        elif rss_p >= 60: pts += 7
        elif rss_p >= 50: pts += 4
        # 섹터 대비 (vol_adj_mom 활용)
        vam = raw.get('vol_adj_mom', 0)
        if vam > 1.5:   pts += 5
        elif vam > 0.5: pts += 3
        elif vam > 0:   pts += 1

        # (5) Breakout Proximity (15점) — 돌파 임박도
        pfh = raw.get('pct_from_high', 100)  # 52주 고점 대비 거리(%)
        if pfh <= 3:    pts += 8
        elif pfh <= 7:  pts += 5
        elif pfh <= 15: pts += 2
        # 20일 box 돌파 근접
        if raw.get('breakout_20d', 0):  pts += 7
        elif raw.get('breakout_10d', 0): pts += 4

        return min(100, pts)

    def analyze_single(self, df, category="", prev_classification=None):
        raw = self.compute_raw(df, category)
        tcs_b, tcs_s, tcs_l = self.score_tcs(raw)
        tfs_b, tfs_s, tfs_l = self.score_tfs(raw)
        oer = self.score_oer(raw)
        # Fallback RSS without cross-sectional data
        rss_s = min(100, max(0, 50 + raw['ret_5d'] * 5))
        avg_ret = (raw['ret_21d'] + raw['ret_63d'] + raw['ret_126d']) / 3.0
        rss_l = min(100, max(0, 50 + avg_ret * 2))
        rss = round(self.W_SHORT_RSS * rss_s + self.W_LONG_RSS * rss_l)
        urs = 50.0  # 단일 ticker 모드 → 중립 (cross-sectional 불가)
        comp = self.composite(tcs_b, tfs_b, rss, oer, urs)
        cls = self.classify(raw, tcs_s, tcs_l, tfs_s, tfs_l, oer,
                            prev_classification=prev_classification)
        return {'composite': comp, 'tcs': tcs_b, 'tfs': tfs_b, 'oer': oer, 'rss': rss,
                'urs': urs,
                'tcs_short': tcs_s, 'tcs_long': tcs_l,
                'tfs_short': tfs_s, 'tfs_long': tfs_l,
                'classification': cls, 'rsi': raw['rsi'], 'trend_age': raw['trend_age'],
                'sma50_dist': raw['sma50_dist'], 'adv_usd': raw['adv_usd'],
                'last_close': raw['last_close'], 'raw': raw}


###############################################################################
# SECTION 4: PORTFOLIO ELIGIBILITY
###############################################################################

def evaluate_eligible(analysis, adv_usd, min_adv=5_000_000, comp_threshold=55):
    """Portfolio eligibility 평가.
    부적격 클래스: DOWNTREND, EXHAUSTING, FADING, COUNTER_RALLY, CYCLE_PEAK, WEAKENING.
      - WEAKENING (DOWN, FLAT): 단기 약세 + 장기 횡보 → 매수 진입 위험 (#8 fix).
      - OVEREXTENDED는 위험 신호이나 차익실현/관망용 — 부적격은 아니나 CLASS_RANK=1.
    LAGGING_CATCHUP은 적격 (URS 기반 catch-up 매수 후보)."""
    cls = analysis['classification']
    comp = analysis['composite']
    reasons = []
    if cls == "⬇️ DOWNTREND":
        reasons.append("Downtrend")
    if cls == "🟤 EXHAUSTING":
        reasons.append("Exhausting")
    if cls == "🟤 FADING":
        reasons.append("Fading")
    if cls == "🟣 COUNTER_RALLY":
        reasons.append("CounterRally")
    if cls == "🔴 CYCLE_PEAK":
        reasons.append("CyclePeak")
    if cls == "⚠️ WEAKENING":
        reasons.append("Weakening")
    if comp < comp_threshold:
        reasons.append("LowScore")
    if adv_usd < min_adv:
        reasons.append(f"Liq({adv_usd/1e6:.1f}M)")
    eligible = len(reasons) == 0
    return eligible, "/".join(reasons) if reasons else "None"


###############################################################################
# SECTION 5: 7-DAY HISTORY + SIGNAL VALIDITY
###############################################################################

def compute_7day_history(all_data, detector):
    ref = list(all_data.keys())[0]
    dates = all_data[ref].df.index
    n = min(8, max(2, len(dates) - 60))
    hist_dates = dates[-n:]
    history = {}
    total = len(all_data)
    print(f"\n📈 7-Day History ({n} days, {total} ETFs)...")
    for ti, (ticker, etf) in enumerate(sorted(all_data.items())):
        th = []
        for d in hist_dates:
            df_cut = etf.df[etf.df.index <= d]
            if len(df_cut) < 60: continue
            try:
                a = detector.analyze_single(df_cut, etf.category)
                el, _ = evaluate_eligible(a, a['adv_usd'])
                th.append({'date': d, 'composite': a['composite'], 'tcs': a['tcs'],
                           'tfs': a['tfs'], 'oer': a['oer'], 'class': a['classification'],
                           'eligible': el})
            except: pass
        history[ticker] = th
        if (ti + 1) % 50 == 0: print(f"   ... {ti+1}/{total}")
    print(f"   ✅ Done")
    return history

class SignalValidityEngine:
    SCORE_BUCKETS = [(0,30,"0-30"),(30,50,"30-50"),(50,70,"50-70"),(70,100.1,"70-100")]
    FORWARD_DAYS = [5, 21, 63, 126, 252]   # 1W, 1M, 3M, 6M, 12M (trading days)
    PRIMARY_FWD = 21             # primary period for validity scoring (1M)
    CLASS_SHORT = {
        "⬇️ DOWNTREND":"DOWN", "🟤 FADING":"FADING", "🟣 COUNTER_RALLY":"CNTR",
        "⚠️ WEAKENING":"WEAK", "🟠 NEUTRAL":"NEUTRAL", "🟡 CONSOLIDATION":"CONSOL",
        "🔶 PULLBACK":"PULL", "🔵 RECOVERY":"RECV", "🔵 FORMATION":"FORM",
        "🟡 OVEREXTENDED":"OVEXT", "🟤 EXHAUSTING":"EXHAUST", "🟢 CONTINUATION":"CONT",
        "🔴 CYCLE_PEAK":"CPEAK", "🟦 LAGGING_CATCHUP":"LAGCU",
    }
    # FIX #13: OVEREXTENDED는 매수 후보 그룹에서 분리 → rank 1 (위험 신호)
    # 신규 LAGGING_CATCHUP은 매수 후보 (rank 2)
    CLASS_RANK = {
        "⬇️ DOWNTREND":0, "🟤 FADING":0, "🟣 COUNTER_RALLY":0, "🔴 CYCLE_PEAK":0,
        "⚠️ WEAKENING":1, "🟠 NEUTRAL":1, "🟤 EXHAUSTING":1, "🟡 OVEREXTENDED":1,
        "🟡 CONSOLIDATION":2, "🔶 PULLBACK":2, "🔵 RECOVERY":2,
        "🔵 FORMATION":2, "🟦 LAGGING_CATCHUP":2,
        "🟢 CONTINUATION":3,
    }

    # Walk-forward: train on older data, test on recent data
    TRAIN_RATIO = 0.75       # first 75% for calibration, last 25% for OOS
    EMBARGO_DAYS = 5         # purged CV: exclude N days around train/test boundary

    def __init__(self, n_eval=24, lookback_td=252):
        self.n_eval = n_eval; self.lookback_td = lookback_td
        self.observations = []       # ALL observations (train + test)
        self.train_obs = []          # train-only (for threshold calibration)
        self.test_obs = []           # test-only (for OOS reporting)
        # Legacy (variable-forward) stats — kept for backward compat
        self.bucket_stats = {}; self.class_stats = {}; self.etf_stats = {}
        # Fixed-forward stats: {fwd_days: {bucket/class/ticker: stats}}
        self.fwd_bucket_stats = {}; self.fwd_class_stats = {}; self.fwd_eligible_stats = {}
        # OOS fixed-forward stats (test set only)
        self.oos_bucket_stats = {}; self.oos_class_stats = {}
        # Transition hit rate
        self.transition_counts = defaultdict(int)
        self.transition_totals = defaultdict(int)
        self.transition_hit = {}   # (from_cls, to_cls) -> stats
        # Score-weighted hit rate
        self.score_weighted = {}   # bucket -> weighted_hit
        # Adaptive thresholds (calibrated from train set)
        self.adaptive_thresholds = {}
        self.computed = False

    def compute(self, all_data, detector):
        ref = list(all_data.keys())[0]
        dates = all_data[ref].df.index
        n_avail = len(dates)
        max_fwd = max(self.FORWARD_DAYS)
        if n_avail < self.lookback_td + 60: return
        start_i = max(60, n_avail - self.lookback_td - 1)
        end_i = n_avail - max_fwd - 1   # leave room for forward return
        if end_i <= start_i: end_i = n_avail - 2
        eval_indices = sorted(set(np.linspace(start_i, end_i, self.n_eval, dtype=int)))
        eval_dates = [dates[i] for i in eval_indices]

        # Walk-forward split: train | embargo | test
        n_eval_pts = len(eval_dates)
        train_end = int(n_eval_pts * self.TRAIN_RATIO)
        embargo_pts = max(1, self.EMBARGO_DAYS // max(1, (n_avail // n_eval_pts)))
        test_start = min(train_end + embargo_pts, n_eval_pts - 1)
        train_dates = set(str(d.date()) if hasattr(d, 'date') else str(d) for d in eval_dates[:train_end])
        test_dates = set(str(d.date()) if hasattr(d, 'date') else str(d) for d in eval_dates[test_start:])

        print(f"\n🔍 Validity Engine v3: {n_eval_pts} eval points × "
              f"{len(all_data)} tickers, fwd={self.FORWARD_DAYS}d | "
              f"train={train_end} embargo={embargo_pts} test={n_eval_pts - test_start}")

        prev_cls = {}
        for ei, ed in enumerate(eval_dates):
            cur_cls = {}
            for ticker, etf in all_data.items():
                if not etf.valid: continue
                df_e = etf.df[etf.df.index <= ed]
                if len(df_e) < 60: continue
                # Hysteresis: 이전 eval point의 분류를 전달 → bi-weekly 노이즈 방지
                _prev_cls_for_ticker = prev_cls.get(ticker)
                try: a = detector.analyze_single(df_e, etf.category,
                                                  prev_classification=_prev_cls_for_ticker)
                except: continue
                cur_cls[ticker] = a['classification']
                ec = sf(df_e['Close'].iloc[-1])
                if ec <= 0: continue

                # Variable forward return (to current — legacy)
                cc = sf(etf.df['Close'].iloc[-1])
                fwd_ret_var = (cc / ec - 1) * 100

                # Multi-benchmark excess return: median across alternatives
                bench_rets_var = []
                alt_benchmarks = CATEGORY_BENCHMARK_ALT.get(etf.category, [])
                primary_bench = detector.benchmark_map.get(etf.category, detector.benchmark_data)
                if alt_benchmarks:
                    for btk in alt_benchmarks:
                        bdata = all_data.get(btk)
                        if bdata and bdata.valid:
                            try:
                                bc = ss(bdata.df['Close'])
                                br = (sf(bc.iloc[-1]) / sf(bc.asof(ed)) - 1) * 100
                                bench_rets_var.append(br)
                            except: pass
                if not bench_rets_var and primary_bench is not None:
                    try:
                        bc = ss(primary_bench['Close'])
                        bench_rets_var.append((sf(bc.iloc[-1]) / sf(bc.asof(ed)) - 1) * 100)
                    except: pass
                b_ret_var = float(np.median(bench_rets_var)) if bench_rets_var else 0.0

                # Fixed forward returns: 1~252 daily for curve, detailed stats for FORWARD_DAYS only
                fwd_rets = {}; fwd_bench = {}; fwd_daily = {}
                tk_close = ss(etf.df['Close'])
                tk_dates = etf.df.index
                try:
                    tk_ed_idx = tk_dates.get_loc(ed, method='ffill')
                except (KeyError, TypeError):
                    tk_ed_idx = tk_dates.searchsorted(ed, side='right') - 1
                tk_n = len(tk_dates)
                # All daily forward returns (1~252) for hit rate curve
                # + primary benchmark forward return for each day (category-specific)
                pb_close = None
                pb_dates = None
                if primary_bench is not None:
                    pb_close = ss(primary_bench['Close'])
                    pb_dates = primary_bench.index
                for fd in range(1, 253):
                    fi = tk_ed_idx + fd
                    if 0 <= fi < tk_n:
                        fc = sf(tk_close.iloc[fi])
                        fwd_rets[fd] = (fc / ec - 1) * 100
                        # Benchmark return for this day
                        if pb_close is not None:
                            try:
                                fwd_date = tk_dates[fi]
                                b_entry = sf(pb_close.asof(ed))
                                b_exit = sf(pb_close.asof(fwd_date))
                                fwd_bench[fd] = (b_exit / b_entry - 1) * 100 if b_entry > 0 else 0.0
                            except:
                                fwd_bench.setdefault(fd, 0.0)
                    else:
                        fwd_rets[fd] = None
                # Detailed stats (daily returns, multi-benchmark) only for primary FORWARD_DAYS
                for fd in self.FORWARD_DAYS:
                    fi = tk_ed_idx + fd
                    if 0 <= fi < tk_n:
                        # Daily returns for risk-adjusted calc
                        seg = tk_close.iloc[tk_ed_idx:fi+1]
                        if len(seg) > 1:
                            dr = seg.pct_change().dropna()
                            fwd_daily[fd] = dr.values.tolist()
                        # Multi-benchmark forward return
                        bdt = tk_dates[fi]
                        fb_rets = []
                        for btk in alt_benchmarks:
                            bdata = all_data.get(btk)
                            if bdata and bdata.valid:
                                try:
                                    bc = ss(bdata.df['Close'])
                                    fb_rets.append((sf(bc.asof(bdt)) / sf(bc.asof(ed)) - 1) * 100)
                                except: pass
                        if not fb_rets and primary_bench is not None:
                            try:
                                bc = ss(primary_bench['Close'])
                                fb_rets.append((sf(bc.asof(bdt)) / sf(bc.asof(ed)) - 1) * 100)
                            except: pass
                        fwd_bench[fd] = float(np.median(fb_rets)) if fb_rets else 0.0

                bucket = "70-100"
                for lo, hi, lbl in self.SCORE_BUCKETS:
                    if lo <= a['composite'] < hi: bucket = lbl; break

                eval_str = str(ed.date()) if hasattr(ed, 'date') else str(ed)
                # Multi-strategy consensus score
                _raw = a.get('raw', {})
                try:
                    _hedge = score_all_strategies(_raw, ranks=None, regime='transition', cat_stats=None)
                    _hedge['oneil_long'] = NaiveDiscoveryDetector.score_oneil_long(_raw, None)
                    _hedge['oneil_short'] = NaiveDiscoveryDetector.score_oneil_short(_raw, None)
                    _combined = compute_combined_signal(_hedge)
                    _consensus = round(_combined['combined_long'] * 0.5 + a['composite'] * 0.5, 1)
                except:
                    _combined = {'combined_long': 0, 'combined_short': 0, 'long_count': 0, 'short_count': 0, 'net_signal': 'NEUTRAL', 'conviction': 0}
                    _consensus = a['composite']

                obs_entry = {
                    'ticker': ticker, 'score': _consensus,
                    'score_composite': a['composite'],
                    'combined_long': _combined['combined_long'],
                    'combined_short': _combined['combined_short'],
                    'long_count': _combined['long_count'],
                    'short_count': _combined['short_count'],
                    'net_signal': _combined['net_signal'],
                    'conviction': _combined['conviction'],
                    'tcs': a['tcs'], 'tfs': a['tfs'], 'oer': a['oer'],
                    'classification': a['classification'], 'bucket': bucket,
                    'eligible': a.get('classification') not in (
                        "⬇️ DOWNTREND", "🟤 EXHAUSTING", "🟤 FADING", "🟣 COUNTER_RALLY",
                        "🔴 CYCLE_PEAK", "⚠️ WEAKENING"
                    ) and a['composite'] >= 55,
                    'fwd_return': fwd_ret_var, 'bench_return': b_ret_var,
                    'excess_return': fwd_ret_var - b_ret_var,
                    'fwd_rets': fwd_rets, 'fwd_bench': fwd_bench,
                    'fwd_daily': fwd_daily,
                    'eval_date': eval_str,
                    'split': 'train' if eval_str in train_dates else
                             'test' if eval_str in test_dates else 'embargo',
                    # Phase 1: classify() 재실행을 위한 raw 필드 보존
                    'sma20_dist': sf(_raw.get('sma20_dist', 0)),
                    'sma20_slope': sf(_raw.get('sma20_slope', 0)),
                    'sma50_dist': sf(_raw.get('sma50_dist', 0)),
                    'sma50_slope': sf(_raw.get('sma50_slope', 0)),
                    'sma50_sma200_spread': sf(_raw.get('sma50_sma200_spread', 0)),
                    'realized_vol': sf(_raw.get('realized_vol', 0)),
                }
                self.observations.append(obs_entry)
                if ticker in prev_cls:
                    self.transition_counts[(prev_cls[ticker], a['classification'])] += 1
                    self.transition_totals[prev_cls[ticker]] += 1
            prev_cls = cur_cls
            if (ei+1) % 6 == 0: print(f"   ... eval {ei+1}/{len(eval_dates)}")

        # Split observations into train/test
        self.train_obs = [o for o in self.observations if o.get('split') == 'train']
        self.test_obs = [o for o in self.observations if o.get('split') == 'test']
        print(f"   📊 Observations: {len(self.observations)} total "
              f"(train={len(self.train_obs)}, test={len(self.test_obs)})")

        self._aggregate()
        self._aggregate_fixed_forward()
        self._aggregate_oos()
        self._aggregate_transitions()
        self._aggregate_score_weighted()
        self._calibrate_adaptive_thresholds()
        self.computed = True
        self._print()

    # ── OOS aggregation (test set only, fixed forward) ──
    def _aggregate_oos(self):
        """Aggregate fixed-forward hit rates using ONLY out-of-sample test observations."""
        for fd in self.FORWARD_DAYS:
            bg, cg = defaultdict(list), defaultdict(list)
            for o in self.test_obs:
                if o['fwd_rets'].get(fd) is None: continue
                bg[o['bucket']].append(o)
                cg[o['classification']].append(o)
            self.oos_bucket_stats[fd] = {k: self._agg_fixed(v, fd) for k, v in bg.items()}
            self.oos_class_stats[fd] = {k: self._agg_fixed(v, fd) for k, v in cg.items()}

    # ── Adaptive Threshold Calibration (from train set) ──
    def _calibrate_adaptive_thresholds(self):
        """Calibrate classification thresholds from train set distributions."""
        if not self.train_obs:
            self.adaptive_thresholds = {}
            return
        oer_vals = [o['oer'] for o in self.train_obs if o['oer'] > 0]
        tfs_vals = [o['tfs'] for o in self.train_obs if o['tfs'] > 0]
        comp_vals = [o['score'] for o in self.train_obs]

        # OER: top 20% → overextended threshold
        oer_thresh = float(np.percentile(oer_vals, 80)) if oer_vals else 60
        # TFS: top 30% → formation threshold
        tfs_thresh = float(np.percentile(tfs_vals, 70)) if tfs_vals else 50
        # Composite: top 40% → eligibility threshold
        comp_thresh = float(np.percentile(comp_vals, 60)) if comp_vals else 55

        self.adaptive_thresholds = {
            'oer_overextended': round(max(40, min(80, oer_thresh)), 1),
            'tfs_formation': round(max(30, min(70, tfs_thresh)), 1),
            'composite_eligible': round(max(40, min(70, comp_thresh)), 1),
        }
        print(f"   🎯 Adaptive thresholds: OER≥{self.adaptive_thresholds['oer_overextended']}, "
              f"TFS≥{self.adaptive_thresholds['tfs_formation']}, "
              f"Comp≥{self.adaptive_thresholds['composite_eligible']}")

    # ── Legacy aggregation (variable forward) ──
    def _agg_group(self, obs):
        if not obs: return {'n': 0, 'hit_rate': 0, 'exc_hit': 0, 'avg_ret': 0, 'avg_exc': 0}
        n = len(obs); fr = [o['fwd_return'] for o in obs]; er = [o['excess_return'] for o in obs]
        return {'n': n, 'hit_rate': round(sum(1 for r in fr if r > 0) / n * 100, 1),
                'exc_hit': round(sum(1 for r in er if r > 0) / n * 100, 1),
                'avg_ret': round(np.mean(fr), 2), 'avg_exc': round(np.mean(er), 2)}

    def _aggregate(self):
        bg, cg, eg = defaultdict(list), defaultdict(list), defaultdict(list)
        for o in self.observations:
            bg[o['bucket']].append(o); cg[o['classification']].append(o); eg[o['ticker']].append(o)
        self.bucket_stats = {k: self._agg_group(v) for k, v in bg.items()}
        self.class_stats = {k: self._agg_group(v) for k, v in cg.items()}
        self.etf_stats = {k: self._agg_group(v) for k, v in eg.items()}

    # ── Fixed-forward aggregation (5d/10d/21d) ──
    def _agg_fixed(self, obs, fd):
        """Aggregate fixed-forward hit rate for a group of observations."""
        valid = [o for o in obs if o['fwd_rets'].get(fd) is not None]
        if not valid: return {'n': 0, 'hit_rate': 0, 'exc_hit': 0, 'avg_ret': 0, 'avg_exc': 0,
                              'sharpe': 0, 'risk_adj_hit': 0}
        n = len(valid)
        fr = [o['fwd_rets'][fd] for o in valid]
        er = [o['fwd_rets'][fd] - o['fwd_bench'].get(fd, 0) for o in valid]
        hit = round(sum(1 for r in fr if r > 0) / n * 100, 1)
        exc = round(sum(1 for r in er if r > 0) / n * 100, 1)
        avg_r = round(np.mean(fr), 2)
        avg_e = round(np.mean(er), 2)
        # Risk-adjusted hit rate: positive Sharpe ratio
        sharpes = []
        for o in valid:
            daily = o['fwd_daily'].get(fd, [])
            if len(daily) >= 2:
                mu = np.mean(daily); sd = np.std(daily)
                sharpes.append(mu / sd * np.sqrt(252) if sd > 1e-10 else 0.0)
        sharpe = round(np.mean(sharpes), 2) if sharpes else 0.0
        risk_adj = round(sum(1 for s in sharpes if s > 0) / len(sharpes) * 100, 1) if sharpes else 0.0
        return {'n': n, 'hit_rate': hit, 'exc_hit': exc, 'avg_ret': avg_r, 'avg_exc': avg_e,
                'sharpe': sharpe, 'risk_adj_hit': risk_adj}

    def _aggregate_fixed_forward(self):
        for fd in self.FORWARD_DAYS:
            bg, cg, el_g = defaultdict(list), defaultdict(list), defaultdict(list)
            for o in self.observations:
                if o['fwd_rets'].get(fd) is None: continue
                bg[o['bucket']].append(o)
                cg[o['classification']].append(o)
                if o['eligible']:
                    el_g['eligible'].append(o)
                    el_g[o['bucket']].append(o)
                    el_g[o['classification']].append(o)
            self.fwd_bucket_stats[fd] = {k: self._agg_fixed(v, fd) for k, v in bg.items()}
            self.fwd_class_stats[fd] = {k: self._agg_fixed(v, fd) for k, v in cg.items()}
            self.fwd_eligible_stats[fd] = {k: self._agg_fixed(v, fd) for k, v in el_g.items()}

    # ── Transition hit rate ──
    def _aggregate_transitions(self):
        """Compute hit rate for classification transitions (from→to)."""
        # Group observations by ticker and sort by eval_date to find transitions
        ticker_obs = defaultdict(list)
        for o in self.observations:
            ticker_obs[o['ticker']].append(o)

        trans_obs = defaultdict(list)   # (from_cls, to_cls) -> [obs of to_cls]
        for tk, obs_list in ticker_obs.items():
            obs_list.sort(key=lambda x: x['eval_date'])
            for i in range(1, len(obs_list)):
                prev_cls = obs_list[i-1]['classification']
                cur_cls = obs_list[i]['classification']
                if prev_cls != cur_cls:
                    trans_obs[(prev_cls, cur_cls)].append(obs_list[i])

        self.transition_hit = {}
        for (fc, tc), obs in trans_obs.items():
            if len(obs) < 3: continue
            # Use 10d forward as primary
            pfd = self.PRIMARY_FWD
            stats = self._agg_fixed(obs, pfd) if any(o['fwd_rets'].get(pfd) is not None for o in obs) \
                    else self._agg_group(obs)
            self.transition_hit[(fc, tc)] = stats

    # ── Score-weighted hit rate ──
    def _aggregate_score_weighted(self):
        """Weight hit rate by composite score — higher scores count more."""
        for fd in self.FORWARD_DAYS:
            bg = defaultdict(list)
            for o in self.observations:
                if o['fwd_rets'].get(fd) is None: continue
                bg[o['bucket']].append(o)
            weighted = {}
            for bk, obs in bg.items():
                if not obs: continue
                total_w = sum(o['score'] for o in obs)
                if total_w < 1e-10: continue
                w_hit = sum(o['score'] for o in obs if o['fwd_rets'][fd] > 0) / total_w * 100
                w_exc = sum(o['score'] for o in obs
                            if o['fwd_rets'][fd] - o['fwd_bench'].get(fd, 0) > 0) / total_w * 100
                weighted[bk] = {'w_hit': round(w_hit, 1), 'w_exc': round(w_exc, 1), 'n': len(obs)}
            self.score_weighted[fd] = weighted

    # ── Console output ──
    def _print(self):
        W = 110
        print(f"\n{'='*W}")
        print("[SIGNAL VALIDITY v3: Walk-Forward | In-Sample + Out-of-Sample]")
        print(f"  Train obs: {len(self.train_obs)} | Test obs: {len(self.test_obs)} | "
              f"Embargo: {self.EMBARGO_DAYS}d | Adaptive: {self.adaptive_thresholds}")

        # ── OOS Summary ──
        if self.oos_bucket_stats:
            print(f"\n  ══ OUT-OF-SAMPLE (test set only) ══")
            for fd in self.FORWARD_DAYS:
                oos = self.oos_bucket_stats.get(fd, {})
                if not oos: continue
                print(f"  ── {fd}d OOS ──")
                print(f"  {'Bucket':<10} {'N':>5} {'AbsHit':>7} {'ExcHit':>7} {'AvgRet':>8} {'AvgExc':>8}")
                for lbl in ["0-30","30-50","50-70","70-100"]:
                    s = oos.get(lbl, {'n':0,'hit_rate':0,'exc_hit':0,'avg_ret':0,'avg_exc':0})
                    if s['n'] == 0: continue
                    print(f"  {lbl:<10} {s['n']:>5} {s['hit_rate']:>6.1f}% {s['exc_hit']:>6.1f}% "
                          f"{s['avg_ret']:>7.2f}% {s['avg_exc']:>7.2f}%")

        print(f"\n  ══ FULL SAMPLE (all observations) ══")
        for fd in self.FORWARD_DAYS:
            print(f"\n  ── {fd}-Day Forward ──")
            print(f"  {'Bucket':<10} {'N':>5} {'AbsHit':>7} {'ExcHit':>7} {'AvgRet':>8} {'AvgExc':>8} {'Sharpe':>7} {'RiskAdj':>8} | {'W_Hit':>6} {'W_Exc':>6}")
            for lbl in ["0-30","30-50","50-70","70-100"]:
                s = self.fwd_bucket_stats.get(fd, {}).get(lbl, {'n':0,'hit_rate':0,'exc_hit':0,'avg_ret':0,'avg_exc':0,'sharpe':0,'risk_adj_hit':0})
                sw = self.score_weighted.get(fd, {}).get(lbl, {'w_hit':0,'w_exc':0})
                print(f"  {lbl:<10} {s['n']:>5} {s['hit_rate']:>6.1f}% {s['exc_hit']:>6.1f}% "
                      f"{s['avg_ret']:>7.2f}% {s['avg_exc']:>7.2f}% {s['sharpe']:>7.2f} {s['risk_adj_hit']:>7.1f}% | "
                      f"{sw.get('w_hit',0):>5.1f}% {sw.get('w_exc',0):>5.1f}%")
            # Eligible-only
            el = self.fwd_eligible_stats.get(fd, {}).get('eligible', {'n':0,'hit_rate':0,'exc_hit':0,'avg_ret':0,'avg_exc':0,'sharpe':0,'risk_adj_hit':0})
            if el['n'] > 0:
                print(f"  {'ELIGIBLE':<10} {el['n']:>5} {el['hit_rate']:>6.1f}% {el['exc_hit']:>6.1f}% "
                      f"{el['avg_ret']:>7.2f}% {el['avg_exc']:>7.2f}% {el['sharpe']:>7.2f} {el['risk_adj_hit']:>7.1f}%")

        pfd = self.PRIMARY_FWD
        print(f"\n  ── Classification Hit Rate ({pfd}d Forward) ──")
        all_classes = list(self.CLASS_SHORT.keys())
        fd10 = self.fwd_class_stats.get(pfd, {})
        for cls in all_classes:
            s = fd10.get(cls, {'n':0})
            if s['n'] == 0: continue
            short = self.CLASS_SHORT.get(cls, cls[:12])
            print(f"  {short:<12} {s['n']:>5} {s['hit_rate']:>6.1f}% {s['exc_hit']:>6.1f}% "
                  f"{s['avg_ret']:>7.2f}% {s['avg_exc']:>7.2f}% {s.get('sharpe',0):>7.2f} {s.get('risk_adj_hit',0):>7.1f}%")

        # Transition hit rates (top transitions)
        if self.transition_hit:
            print(f"\n  ── Top Transition Hit Rates ({self.PRIMARY_FWD}d Forward) ──")
            print(f"  {'From→To':<25} {'N':>5} {'AbsHit':>7} {'ExcHit':>7} {'AvgRet':>8}")
            sorted_trans = sorted(self.transition_hit.items(), key=lambda x: -x[1].get('exc_hit', 0))
            for (fc, tc), s in sorted_trans[:15]:
                fc_s = self.CLASS_SHORT.get(fc, fc[:6])
                tc_s = self.CLASS_SHORT.get(tc, tc[:6])
                print(f"  {fc_s}→{tc_s:<18} {s['n']:>5} {s['hit_rate']:>6.1f}% {s['exc_hit']:>6.1f}% {s['avg_ret']:>7.2f}%")

        print(f"\n  ── Transition Matrix (% row→col) ──")
        observed = set()
        for (cf, ct) in self.transition_counts: observed.add(cf); observed.add(ct)
        classes = [c for c in all_classes if c in observed]
        shorts = [self.CLASS_SHORT[c] for c in classes]
        header_label = 'From\\To'
        print(f"  {header_label:<12}" + "".join(f"{s:>9}" for s in shorts))
        for cf in classes:
            tot = self.transition_totals.get(cf, 0)
            row = f"  {self.CLASS_SHORT[cf]:<12}"
            for ct in classes:
                cnt = self.transition_counts.get((cf, ct), 0)
                row += f"{(cnt/tot*100 if tot else 0):>8.1f}%"
            print(f"{row} (n={tot})")
        print(f"{'='*W}")

    def get_validity(self, ticker, comp, cls):
        if not self.computed: return {'val_prob': 50.0, 'val_persist': 50.0, 'val_conf': 'N/A',
                                       'val_hit_1w': 50.0, 'val_hit_1m': 50.0, 'val_hit_3m': 50.0}
        bucket = "70-100"
        for lo, hi, lbl in self.SCORE_BUCKETS:
            if lo <= comp < hi: bucket = lbl; break

        # Prefer OOS stats; fall back to full sample (using primary forward period)
        pfd = self.PRIMARY_FWD
        oos_bs = self.oos_bucket_stats.get(pfd, {}).get(bucket, {'exc_hit': 50, 'n': 0})
        oos_cs = self.oos_class_stats.get(pfd, {}).get(cls, {'exc_hit': 50, 'n': 0})

        # If OOS has sufficient data (>=5 obs), prefer it; else fall back to full
        if oos_bs.get('n', 0) >= 5:
            bs = oos_bs
        else:
            bs = self.fwd_bucket_stats.get(pfd, {}).get(bucket, {'exc_hit': 50, 'n': 0})
        if oos_cs.get('n', 0) >= 5:
            cs = oos_cs
        else:
            cs = self.fwd_class_stats.get(pfd, {}).get(cls, {'exc_hit': 50, 'n': 0})
        es = self.etf_stats.get(ticker, {'exc_hit': 50, 'n': 0})
        w, p = [], []
        if bs['n'] > 0: w.append(min(bs['n'], 50)); p.append(bs['exc_hit'])
        if cs['n'] > 0: w.append(min(cs['n'], 50)); p.append(cs['exc_hit'])
        if es['n'] > 0: w.append(min(es['n'], 30) * 1.5); p.append(es['exc_hit'])
        val = sum(a*b for a, b in zip(w, p)) / sum(w) if w else 50.0

        # Persistence
        tot = self.transition_totals.get(cls, 0)
        persist = 50.0
        if tot > 0:
            cr = self.CLASS_RANK.get(cls, 1)
            keep = sum(self.transition_counts.get((cls, ct), 0)
                       for ct, r in self.CLASS_RANK.items() if r >= cr)
            persist = keep / tot * 100

        # Per-period hit rates (from fixed-forward bucket stats)
        hit_by_fd = {}
        for fd in self.FORWARD_DAYS:
            fd_bs = self.fwd_bucket_stats.get(fd, {}).get(bucket, {'exc_hit': 50})
            hit_by_fd[fd] = fd_bs['exc_hit']

        total_n = bs.get('n', 0) + es.get('n', 0)
        conf = "H" if total_n >= 30 else "M" if total_n >= 10 else "L"
        return {
            'val_prob': round(val, 1),
            'val_persist': round(persist, 1),
            'val_conf': conf,
            'val_hit_1w': round(hit_by_fd.get(5, 50), 1),
            'val_hit_1m': round(hit_by_fd.get(21, 50), 1),
            'val_hit_3m': round(hit_by_fd.get(63, 50), 1),
        }


###############################################################################
# SECTION 6: VISUALIZATION ENGINE
###############################################################################

class VizEngine:
    C = {'bg':'#0a0e17','panel':'#111827','text':'#e5e7eb','cyan':'#06b6d4',
         'green':'#22c55e','red':'#ef4444','yellow':'#f59e0b','orange':'#f97316',
         'blue':'#3b82f6','gray':'#6b7280','purple':'#8b5cf6'}

    def __init__(self):
        plt.rcParams.update({'figure.facecolor': self.C['bg'], 'axes.facecolor': self.C['panel'], 'text.color': self.C['text']})

    def _text_page(self, lines, title="", pdf=None, fontsize=7):
        fig = plt.figure(figsize=(28, max(8, len(lines)*0.14)))
        ax = fig.add_subplot(111); ax.axis('off')
        ax.text(0.01, 0.99, "\n".join(lines), transform=ax.transAxes, fontsize=fontsize, family='monospace', va='top', color=self.C['text'])
        plt.tight_layout()
        if pdf: pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close(fig)

    def plot_master(self, results, pdf=None):
        lines = [
            "Global ETF Price Discovery Scanner v5.0 — MASTER SUMMARY",
            "  TCS=Trend Continuation | TFS=Trend Formation | OER=Overextension Risk | RSS=Multi-Horizon Momentum",
            "  Class: 🟢CONT / 🔵FORM / 🟡OVEREXT / 🟤EXHAUST / 🟠NEUTRAL / ⬇️DOWN  |  Val%=Validity  Pst%=Persistence",
            "=" * 200,
            f"{'Rk':>3} {'Ticker':>8} {'Name':<18} {'Cat':<14} {'AsOf':>10} | "
            f"{'Comp':>5} {'TCS':>3} {'TFS':>3} {'OER':>3} {'RSS':>3} {'Class':<14} {'Elg':>3} {'Rej':<15} | "
            f"{'Val%':>5} {'Pst':>4} {'C':>1} | "
            f"{'1W_Sc':>5} {'Ret':>6} | {'1M_Sc':>5} {'Ret':>6} | {'3M_Sc':>5} {'Ret':>6}",
            "-" * 200
        ]
        for i, r in enumerate(results):
            el = 'Y' if r['eligible'] else '.'
            cls_short = r['classification'][:13]
            vp = r.get('val_prob', 50.0); ps = r.get('val_persist', 50.0); vc = r.get('val_conf', 'N')
            lines.append(
                f"{i+1:>3} {r['ticker']:>8} {r['name'][:17]:<18} {r['category'][:13]:<14} "
                f"{r['data_as_of'][-10:]:>10} | "
                f"{r['composite']:>5.1f} {r['tcs']:>3} {r['tfs']:>3} {r['oer']:>3} {r['rss']:>3} "
                f"{cls_short:<14} {el:>3} {r['rejection'][:14]:<15} | "
                f"{vp:>5.1f} {ps:>4.0f} {vc:>1} | "
                f"{r['score_1w']:>5.1f} {r['ret_1w']:>5.2f}% | "
                f"{r['score_1m']:>5.1f} {r['ret_1m']:>5.2f}% | "
                f"{r['score_3m']:>5.1f} {r['ret_3m']:>5.2f}%"
            )
        self._text_page(lines, pdf=pdf, fontsize=6.5)

    def plot_validity(self, ve, results, pdf=None):
        if not ve.computed: return
        lines = ["="*100, "SIGNAL VALIDITY VERIFICATION (Past 1-Month)", "="*100, "",
                 "━━━ Score Bucket Analysis ━━━",
                 f"{'Bucket':<10} {'N':>5} {'AbsHit':>7} {'ExcHit':>7} {'AvgRet':>8} {'AvgExc':>8}", "─"*60]
        for lbl in ["0-30","30-50","50-70","70-100"]:
            s = ve.bucket_stats.get(lbl, {'n':0,'hit_rate':0,'exc_hit':0,'avg_ret':0,'avg_exc':0})
            lines.append(f"{lbl:<10} {s['n']:>5} {s['hit_rate']:>6.1f}% {s['exc_hit']:>6.1f}% {s['avg_ret']:>7.2f}% {s['avg_exc']:>7.2f}%")
        lines += ["", "━━━ Classification Analysis ━━━",
                   f"{'Class':<12} {'N':>5} {'AbsHit':>7} {'ExcHit':>7} {'AvgRet':>8} {'AvgExc':>8}", "─"*60]
        for cls in ["⬇️ DOWNTREND","🟠 NEUTRAL","🔵 FORMATION","🟢 CONTINUATION","🟡 OVEREXTENDED","🟤 EXHAUSTING","🔴 CYCLE_PEAK"]:
            s = ve.class_stats.get(cls, {'n':0,'hit_rate':0,'exc_hit':0,'avg_ret':0,'avg_exc':0})
            lines.append(f"{SignalValidityEngine.CLASS_SHORT.get(cls,'?'):<12} {s['n']:>5} {s['hit_rate']:>6.1f}% {s['exc_hit']:>6.1f}% {s['avg_ret']:>7.2f}% {s['avg_exc']:>7.2f}%")
        el = sorted([r for r in results if r['eligible']], key=lambda x: -x.get('val_prob', 0))[:20]
        if el:
            lines += ["", "━━━ Top 20 Eligible by Validity ━━━",
                       f"{'Rk':>3} {'Ticker':>8} {'Name':<18} {'Comp':>5} {'TCS':>3} {'TFS':>3} {'OER':>3} {'Class':<14} {'Val%':>5} {'Pst':>4}", "─"*85]
            for i, r in enumerate(el):
                lines.append(f"{i+1:>3} {r['ticker']:>8} {r['name'][:17]:<18} {r['composite']:>5.1f} {r['tcs']:>3} {r['tfs']:>3} {r['oer']:>3} {r['classification'][:13]:<14} {r.get('val_prob',50):>5.1f} {r.get('val_persist',50):>4.0f}")
        self._text_page(lines, pdf=pdf, fontsize=7)

    def plot_3axis_bar(self, results, pdf=None):
        eligible = sorted([r for r in results if r['eligible']], key=lambda x: -x['composite'])[:35]
        if not eligible: return
        fig, axes = plt.subplots(1, 3, figsize=(20, max(6, len(eligible)*0.35)), sharey=True)
        labels = [f"{r['ticker']} ({r['name'][:14]})" for r in eligible]
        for ax, key, title, color in zip(axes, ['tcs','tfs','oer'],
            ['TCS (Continuation)','TFS (Formation)','OER (Overextension)'],
            [self.C['green'], self.C['blue'], self.C['yellow']]):
            vals = [r[key] for r in eligible]
            ax.barh(labels, vals, color=color, alpha=0.8, height=0.6)
            ax.set_title(title, fontsize=11, fontweight='bold', color=self.C['cyan'])
            ax.set_xlim(0, 105); ax.tick_params(colors=self.C['text'], labelsize=7)
            ax.grid(True, axis='x', alpha=0.15)
        axes[0].invert_yaxis()
        fig.suptitle('📊 3-Axis Signal Decomposition (Eligible ETFs)', fontsize=14, fontweight='bold', color=self.C['cyan'], y=1.02)
        plt.tight_layout()
        if pdf: pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close(fig)

    def plot_comparison(self, results, label, key_s, key_e, key_r, pdf=None):
        lines = [
            f"Current vs {label} — Full List",
            "  TCS=Trend Continuation | TFS=Trend Formation | OER=Overextension Risk | RSS=Multi-Horizon Momentum",
            "=" * 200,
            f"{'Rk':>3} {'Ticker':>8} {'Name':<18} {'Cat':<14} {'AsOf':>10} | "
            f"{'Comp':>5} {'TCS':>3} {'TFS':>3} {'OER':>3} {'RSS':>3} {'Class':<14} {'Elg':>3} {'Rej':<15} | "
            f"{'Val%':>5} {'Pst':>4} {'C':>1} | "
            f"{'1W_Sc':>5} {'Ret':>6} | {'1M_Sc':>5} {'Ret':>6} | {'3M_Sc':>5} {'Ret':>6}",
            "-" * 200
        ]
        for i, r in enumerate(results):
            el = 'Y' if r['eligible'] else '.'
            cls_short = r['classification'][:13]
            vp = r.get('val_prob', 50.0); ps = r.get('val_persist', 50.0); vc = r.get('val_conf', 'N')
            lines.append(
                f"{i+1:>3} {r['ticker']:>8} {r['name'][:17]:<18} {r['category'][:13]:<14} "
                f"{r['data_as_of'][-10:]:>10} | "
                f"{r['composite']:>5.1f} {r['tcs']:>3} {r['tfs']:>3} {r['oer']:>3} {r['rss']:>3} "
                f"{cls_short:<14} {el:>3} {r['rejection'][:14]:<15} | "
                f"{vp:>5.1f} {ps:>4.0f} {vc:>1} | "
                f"{r['score_1w']:>5.1f} {r['ret_1w']:>5.2f}% | "
                f"{r['score_1m']:>5.1f} {r['ret_1m']:>5.2f}% | "
                f"{r['score_3m']:>5.1f} {r['ret_3m']:>5.2f}%"
            )
        up = [r['ticker'] for r in results if r['eligible'] and not r.get(key_e, False)]
        lines.append(f" Upgraded: {json.dumps(up)}")
        self._text_page(lines, pdf=pdf, fontsize=6.5)

    def plot_category_comparison(self, results, label, key_s, key_e, key_r, pdf=None):
        cats = sorted(set(r['category'] for r in results))
        lines = [
            f"CATEGORY SUMMARY: Current vs {label}",
            "  TCS=Trend Continuation | TFS=Trend Formation | OER=Overextension Risk | RSS=Multi-Horizon Momentum",
            "=" * 200,
        ]
        hdr = (f"{'Rk':>3} {'Ticker':>8} {'Name':<18} {'Cat':<14} {'AsOf':>10} | "
               f"{'Comp':>5} {'TCS':>3} {'TFS':>3} {'OER':>3} {'RSS':>3} {'Class':<14} {'Elg':>3} {'Rej':<15} | "
               f"{'Val%':>5} {'Pst':>4} {'C':>1} | "
               f"{'1W_Sc':>5} {'Ret':>6} | {'1M_Sc':>5} {'Ret':>6} | {'3M_Sc':>5} {'Ret':>6}")
        for cat in cats:
            lines.append(f"\n━━━ [{cat}] ━━━")
            lines.append(hdr)
            lines.append("-" * 200)
            cr = sorted([r for r in results if r['category']==cat],
                        key=lambda x: (-x['composite'], x['ticker']))
            for i, r in enumerate(cr):
                el = 'Y' if r['eligible'] else '.'
                cls_short = r['classification'][:13]
                vp = r.get('val_prob', 50.0); ps = r.get('val_persist', 50.0); vc = r.get('val_conf', 'N')
                lines.append(
                    f"{i+1:>3} {r['ticker']:>8} {r['name'][:17]:<18} {r['category'][:13]:<14} "
                    f"{r['data_as_of'][-10:]:>10} | "
                    f"{r['composite']:>5.1f} {r['tcs']:>3} {r['tfs']:>3} {r['oer']:>3} {r['rss']:>3} "
                    f"{cls_short:<14} {el:>3} {r['rejection'][:14]:<15} | "
                    f"{vp:>5.1f} {ps:>4.0f} {vc:>1} | "
                    f"{r['score_1w']:>5.1f} {r['ret_1w']:>5.2f}% | "
                    f"{r['score_1m']:>5.1f} {r['ret_1m']:>5.2f}% | "
                    f"{r['score_3m']:>5.1f} {r['ret_3m']:>5.2f}%"
                )
            up = [r['ticker'] for r in cr if r['eligible'] and not r.get(key_e, False)]
            if up:
                lines.append(f"  📈 Upgraded: {json.dumps(up)}")
        self._text_page(lines, pdf=pdf, fontsize=6.5)

    def plot_7day_trend(self, history, results, pdf=None):
        eligible = [r['ticker'] for r in results if r['eligible']][:30]
        if not eligible: return
        fig, ax = plt.subplots(figsize=(16, 9))
        cmap = plt.cm.get_cmap('tab20', max(len(eligible), 1))
        for i, t in enumerate(eligible):
            h = history.get(t, [])
            if len(h) < 2: continue
            ax.plot([x['date'] for x in h], [x['composite'] for x in h], marker='o', markersize=3, linewidth=1.5, color=cmap(i), label=t, alpha=0.85)
        ax.axhline(y=55, color=self.C['orange'], linestyle=':', alpha=0.5, label='Eligible(55)')
        ax.set_title('📈 7-Day Composite Trend (Eligible ETFs)', fontsize=13, fontweight='bold', color=self.C['cyan'])
        ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.01, 1.0), ncol=1, framealpha=0.3, labelcolor=self.C['text'])
        ax.grid(True, alpha=0.15); plt.tight_layout()
        if pdf: pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close(fig)

    def plot_portfolio_candidates(self, results, pdf=None):
        cands = sorted([r for r in results if r['eligible']], key=lambda x: -x['composite'])
        if not cands: return
        fig, ax = plt.subplots(figsize=(14, max(5, len(cands)*0.4)))
        bars = ax.barh([f"{c['ticker']} ({c['name'][:18]})" for c in cands],
                       [c['composite'] for c in cands], color=self.C['cyan'], alpha=0.8, height=0.6)
        for bar, c in zip(bars, cands):
            cls_s = c['classification'][:10]
            ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                    f"TCS:{c['tcs']} TFS:{c['tfs']} OER:{c['oer']} {cls_s} Val:{c.get('val_prob',50):.0f}%",
                    va='center', fontsize=7, color='white')
        ax.axvline(x=55, color=self.C['orange'], linestyle=':')
        ax.set_title('📋 Portfolio Candidates (v5.0 — 3-Axis Naive)', fontsize=12, fontweight='bold', color=self.C['cyan'])
        plt.tight_layout()
        if pdf: pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close(fig)

    def plot_continuation_trend(self, results, history, pdf=None):
        """NEW: Plots the 1-week classification trend for CONTINUATION candidates."""
        cont_results = [r for r in results if "CONTINUATION" in r['classification']]
        if not cont_results: return

        def _short(c):
            if "DOWN" in c: return "DOWN"
            if "NEUT" in c: return "NEUT"
            if "FORM" in c: return "FORM"
            if "CONT" in c: return "CONT"
            if "OVER" in c: return "OVEX"
            if "EXHA" in c: return "EXHA"
            return "UNKN"

        lines = [
            "CONTINUATION CLASS: 1-Week Class Trend Tracking",
            "=" * 120,
            f"{'Rk':>3} {'Ticker':>8} {'Name':<18} | Trend (Past 7 Days: Oldest -> Newest)",
            "-" * 120
        ]

        for i, r in enumerate(cont_results):
            t = r['ticker']
            h = history.get(t, [])
            trend_strs = [_short(x['class']) for x in h]
            trend_line = " -> ".join(trend_strs)
            lines.append(f"{i+1:>3} {t:>8} {r['name'][:17]:<18} | {trend_line}")

        self._text_page(lines, pdf=pdf, fontsize=8)


###############################################################################
# SECTION 7: MAIN PIPELINE
###############################################################################

def run_scan(categories=None, lookback_days=365, custom_date=None,
             use_realtime=True, run_validation=False,
             include_stocks=False, stock_categories=None):

    # ── Universal Hysteresis: 이전 스캔의 classification 로드 ──
    # 일별 분류 진동(false positive)을 방지하기 위해 직전 시점 분류를 classify()에 전달
    prev_cls_map = {}
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scan_cache.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                _prev_cache = pickle.load(f)
            for r in _prev_cache.get("results", []):
                prev_cls_map[r.get("ticker", "")] = r.get("classification", "")
            print(f"   📋 Loaded {len(prev_cls_map)} previous classifications for hysteresis")
        except Exception as e:
            print(f"   ⚠️  Could not load previous classifications: {e}")

    engine = DataEngine(lookback_days=lookback_days, custom_date=custom_date, use_realtime=use_realtime)
    all_data = engine.download_universe(categories)
    if not all_data: return pd.DataFrame(), [], {}

    if include_stocks:
        stock_data = engine.download_universe(categories=stock_categories, universe=STOCK_UNIVERSE)
        all_data.update(stock_data)
        print(f"   📊 Combined universe: {len(all_data)} total (ETFs + Stocks)")

    detector = NaiveDiscoveryDetector()
    detector.load_benchmarks(all_data, extra_benchmarks=STOCK_BENCHMARK if include_stocks else None)

    print(f"\n📊 Phase 1: Computing raw indicators for {len(all_data)} tickers...")
    all_raw = {}
    for ticker, etf in sorted(all_data.items()):
        if etf.valid and len(etf.df) >= 60:
            try: all_raw[ticker] = detector.compute_raw(etf.df, etf.category)
            except: pass

    # ── 카테고리 태그 주입 (URS LeadLagGap & PeerDispersion 에서 사용) ──
    for ticker, raw in all_raw.items():
        raw['_category'] = all_data[ticker].category if ticker in all_data else 'Unknown'

    print(f"📊 Phase 2: Cross-sectional percentile ranking...")
    all_ranks = NaiveDiscoveryDetector.compute_percentile_ranks(all_raw)

    # ── Hedge strategy pre-computation ──
    market_regime = 'transition'
    cat_stats = {}
    if HAS_HEDGE:
        # Compute market regime from SPY
        spy_raw = all_raw.get('SPY')
        market_regime = compute_regime(spy_raw) if spy_raw else 'transition'
        print(f"   📈 Market regime: {market_regime}")
        # Compute category stats for relative value
        cat_stats = compute_category_stats(all_raw)

    ve = SignalValidityEngine()
    ve.compute(all_data, detector)

    print(f"📊 Phase 4: Scoring and classification...")
    results = []

    for ticker in sorted(all_raw.keys()):
        etf = all_data[ticker]
        raw = all_raw[ticker]
        ranks = all_ranks[ticker]
        try:
            tcs, tcs_s, tcs_l = NaiveDiscoveryDetector.score_tcs(raw)
            tfs, tfs_s, tfs_l = NaiveDiscoveryDetector.score_tfs(raw)
            oer = NaiveDiscoveryDetector.score_oer(raw)
            rss_s = round(ranks['rss_short'], 1)
            rss_l = round(ranks['rss_long'], 1)
            rss = round(ranks['rss'], 1)
            urs = NaiveDiscoveryDetector.score_urs(ranks)
            comp = NaiveDiscoveryDetector.composite(tcs, tfs, rss, oer, urs)
            adaptive = ve.adaptive_thresholds if ve.computed else {}
            cls = NaiveDiscoveryDetector.classify(raw, tcs_s, tcs_l, tfs_s, tfs_l, oer,
                                                   adaptive=adaptive, urs=urs,
                                                   prev_classification=prev_cls_map.get(ticker))

            # O'Neil Long / Short scores
            oneil_long = NaiveDiscoveryDetector.score_oneil_long(raw, ranks)
            oneil_short = NaiveDiscoveryDetector.score_oneil_short(raw, ranks)

            # Hedge fund strategy scores
            hedge_scores = {}
            combined_sig = {}
            if HAS_HEDGE:
                hedge_scores = score_all_strategies(raw, ranks, market_regime, cat_stats)
                all_strat_scores = {'oneil_long': oneil_long, 'oneil_short': oneil_short}
                all_strat_scores.update(hedge_scores)
                combined_sig = compute_combined_signal(all_strat_scores)

            # Event-driven risk & structural quality
            event_flag, event_reasons = NaiveDiscoveryDetector.detect_event_flag(raw)
            structural_q = NaiveDiscoveryDetector.compute_structural_quality(raw, ranks)
            alpha_potential = NaiveDiscoveryDetector.compute_alpha_potential(raw, ranks)

            comp_t = adaptive.get('composite_eligible', 55)
            eligible, rejection = evaluate_eligible(
                {'classification': cls, 'composite': comp}, raw['adv_usd'],
                comp_threshold=comp_t,
            )

            validity = ve.get_validity(ticker, comp, cls)
            current_close = raw['last_close']
            data_as_of = fmt_data_as_of(etf.df)

            def _hist_analysis(df_hist, fwd_days=21):
                if df_hist is None or df_hist.empty or len(df_hist) < 60:
                    return 0.0, False, 0.0
                a = detector.analyze_single(df_hist, etf.category)
                comp_t = adaptive.get('composite_eligible', 55) if adaptive else 55
                el, _ = evaluate_eligible(a, a['adv_usd'], comp_threshold=comp_t)
                hc = sf(df_hist['Close'].iloc[-1])
                # Fixed forward return (no look-ahead): use close N days after signal
                hist_end_idx = len(df_hist) - 1
                full_close = ss(etf.df['Close'])
                # Find the position of hist end date in full data
                try:
                    pos = etf.df.index.get_loc(df_hist.index[-1])
                except (KeyError, TypeError):
                    pos = etf.df.index.searchsorted(df_hist.index[-1], side='right') - 1
                fwd_pos = pos + fwd_days
                if fwd_pos < len(full_close):
                    fwd_close = sf(full_close.iloc[fwd_pos])
                else:
                    fwd_close = sf(full_close.iloc[-1])  # fallback to latest
                ret = ((fwd_close / hc) - 1) * 100 if hc > 0 else 0.0
                return a['composite'], el, ret

            df_1w = etf.df[etf.df.index <= (etf.df.index[-1] - pd.Timedelta(days=7))]
            sc_1w, el_1w, ret_1w = _hist_analysis(df_1w)

            df_1m = etf.df[etf.df.index <= (etf.df.index[-1] - pd.Timedelta(days=30))]
            sc_1m, el_1m, ret_1m = _hist_analysis(df_1m)

            df_3m = etf.df[etf.df.index <= (etf.df.index[-1] - pd.Timedelta(days=90))]
            sc_3m, el_3m, ret_3m = _hist_analysis(df_3m)

            if custom_date:
                df_cst = etf.df[etf.df.index <= custom_date]
                sc_cst, el_cst, ret_cst = _hist_analysis(df_cst)
            else:
                sc_cst, el_cst, ret_cst = 0., False, 0.

            results.append({
                'ticker': ticker, 'name': etf.name, 'category': etf.category,
                'market_cap': etf.market_cap,
                'data_as_of': data_as_of, 'realtime_updated': etf.realtime_updated,
                'composite': comp, 'tcs': tcs, 'tfs': tfs, 'oer': oer, 'rss': rss,
                'urs': urs,
                'urs_leadlag': ranks.get('urs_leadlag', 50.0),
                'urs_attn_gap': ranks.get('urs_attn_gap', 50.0),
                'urs_drift': ranks.get('urs_drift', 50.0),
                'urs_dispersion': ranks.get('urs_dispersion', 50.0),
                'gap_drift_30d': round(raw.get('gap_drift_30d', 0.0), 2),
                'gap_event_age': raw.get('gap_event_age', 0),
                'tcs_short': tcs_s, 'tcs_long': tcs_l,
                'tfs_short': tfs_s, 'tfs_long': tfs_l,
                'rss_short': rss_s, 'rss_long': rss_l,
                'classification': cls, 'eligible': eligible, 'rejection': rejection,
                'rsi': round(raw['rsi'], 1), 'trend_age': raw['trend_age'],
                'sma50_dist': round(raw['sma50_dist'], 2), 'adv_usd': raw['adv_usd'],
                # Phase 1: classify() 재실행을 위해 raw 필드 보존
                'sma20_dist': round(raw.get('sma20_dist', 0), 2),
                'sma20_slope': round(raw.get('sma20_slope', 0), 4),
                'sma50_sma200_spread': round(raw.get('sma50_sma200_spread', 0), 2),
                'oneil_long': oneil_long, 'oneil_short': oneil_short,
                'event_flag': event_flag, 'event_reasons': "/".join(event_reasons) if event_reasons else "",
                'structural_q': structural_q,
                'alpha_potential': alpha_potential,
                'realized_vol': round(raw['realized_vol'], 2),
                'ret_1d': round(raw['ret_1d'], 2),
                'ret_5d': round(raw['ret_5d'], 2),
                'ret_21d': round(raw['ret_21d'], 2),
                'ret_63d': round(raw['ret_63d'], 2),
                'ret_126d': round(raw['ret_126d'], 2),
                'ret_252d': round(raw['ret_252d'], 2),
                'ret_3y_ann': round(raw['ret_3y_ann'], 2) if raw.get('ret_3y_ann') is not None else None,
                'ret_5y_ann': round(raw['ret_5y_ann'], 2) if raw.get('ret_5y_ann') is not None else None,
                'vol_3y_ann': round(raw['vol_3y_ann'], 2) if raw.get('vol_3y_ann') is not None else None,
                'ret_12_1m': round(raw['ret_12_1m'], 2),
                'above_sma50': raw['above_sma50'],
                'above_sma200': raw['above_sma200'],
                'golden_cross': raw['golden_cross'],
                'sma50_slope': round(raw['sma50_slope'], 4),
                'ret_36_12m': round(raw['ret_36_12m'], 2) if raw.get('ret_36_12m') is not None else None,
                'reversal_pctile': round(raw.get('reversal_pctile', 50), 1),
                **validity,
                'score_1w': sc_1w, 'eligible_1w': el_1w, 'ret_1w': ret_1w,
                'score_1m': sc_1m, 'eligible_1m': el_1m, 'ret_1m': ret_1m,
                'score_3m': sc_3m, 'eligible_3m': el_3m, 'ret_3m': ret_3m,
                'score_custom': sc_cst, 'eligible_custom': el_cst, 'ret_custom': ret_cst,
                **hedge_scores,
                **combined_sig,
            })
        except Exception as e:
            print(f"   ⚠️ {ticker}: {e}")

    results.sort(key=lambda x: (-x['composite'], x['ticker']))

    # ═══════════════════════════════════════════════════════════════════════
    # CONSOLE OUTPUT
    # ═══════════════════════════════════════════════════════════════════════
    n_el = sum(1 for r in results if r['eligible'])
    n_rt = sum(1 for r in results if r.get('realtime_updated'))

    cls_dist = defaultdict(int)
    for r in results: cls_dist[r['classification']] += 1

    print(f"\n🔴 RT: {n_rt}/{len(results)} | ELIGIBLE: {n_el}")
    print(f"   Classification: {dict(cls_dist)}")

    non_el_top = [r for r in results[:20] if not r['eligible']]
    if non_el_top:
        print(f"\n⚠️  Top-scoring non-eligible (debug):")
        for r in non_el_top[:5]:
            print(f"   {r['ticker']:>8} Comp={r['composite']:.1f} Class={r['classification'][:10]} Rej={r['rejection']} ADV=${r['adv_usd']/1e6:.1f}M")

    W = 190
    print(f"\n{'='*W}")
    print("[MASTER SUMMARY v5.0: 3-Axis Naive Architecture]")
    print(f"{'Rk':>3} {'Ticker':>8} {'Name':<18} {'Cat':<14} | {'Comp':>5} {'TCS':>3} {'TFS':>3} {'OER':>3} {'RSS':>3} {'Class':<14} {'Elg':>3} | {'Val%':>5} {'Pst':>4} | {'1W':>5} {'Ret':>6} | {'1M':>5} {'Ret':>6} | {'3M':>5} {'Ret':>6}")
    print(f"{'─'*W}")
    for i, r in enumerate(results):
        el = '✅' if r['eligible'] else '  '
        print(f"{i+1:>3} {r['ticker']:>8} {r['name'][:17]:<18} {r['category'][:13]:<14} | "
              f"{r['composite']:>5.1f} {r['tcs']:>3} {r['tfs']:>3} {r['oer']:>3} {r['rss']:>3} "
              f"{r['classification'][:13]:<14} {el:>3} | "
              f"{r.get('val_prob',50):>5.1f} {r.get('val_persist',50):>4.0f} | "
              f"{r['score_1w']:>5.1f} {r['ret_1w']:>5.2f}% | "
              f"{r['score_1m']:>5.1f} {r['ret_1m']:>5.2f}% | "
              f"{r['score_3m']:>5.1f} {r['ret_3m']:>5.2f}%")
    print(f"{'='*W}")

    cats = sorted(set(r['category'] for r in results))
    for label, ks, ke, kr in [("1-WEEK",'score_1w','eligible_1w','ret_1w'),("1-MONTH",'score_1m','eligible_1m','ret_1m'),("3-MONTH",'score_3m','eligible_3m','ret_3m')]:
        print(f"\n{'='*W}\n[CATEGORY SUMMARY: Current vs {label}]")
        print(f"{'Rk':>3} {'Ticker':>8} {'Name':<18} {'Cat':<14} | {'Comp':>5} {'TCS':>3} {'TFS':>3} {'OER':>3} {'RSS':>3} {'Class':<14} {'Elg':>3} | {'Val%':>5} {'Pst':>4} | {'1W':>5} {'Ret':>6} | {'1M':>5} {'Ret':>6} | {'3M':>5} {'Ret':>6}")
        for cat in cats:
            print(f"\n━━━ [{cat}] ━━━")
            print(f"{'─'*W}")
            cr = sorted([r for r in results if r['category']==cat], key=lambda x: (-x['composite'], x['ticker']))
            for j, r in enumerate(cr):
                el = '✅' if r['eligible'] else '  '
                print(f"{j+1:>3} {r['ticker']:>8} {r['name'][:17]:<18} {r['category'][:13]:<14} | "
                      f"{r['composite']:>5.1f} {r['tcs']:>3} {r['tfs']:>3} {r['oer']:>3} {r['rss']:>3} "
                      f"{r['classification'][:13]:<14} {el:>3} | "
                      f"{r.get('val_prob',50):>5.1f} {r.get('val_persist',50):>4.0f} | "
                      f"{r['score_1w']:>5.1f} {r['ret_1w']:>5.2f}% | "
                      f"{r['score_1m']:>5.1f} {r['ret_1m']:>5.2f}% | "
                      f"{r['score_3m']:>5.1f} {r['ret_3m']:>5.2f}%")
            up = [r['ticker'] for r in cr if r['eligible'] and not r[ke]]
            if up: print(f"  📈 Upgraded: {json.dumps(up)}")

    if custom_date:
        print(f"\n{'─'*W}\n[CUSTOM DATE: Current vs {custom_date}]")
        print(f"{'Rk':>3} {'Ticker':>8} {'Name':<18} {'Cat':<14} | {'Comp':>5} {'TCS':>3} {'TFS':>3} {'OER':>3} {'RSS':>3} {'Class':<14} {'Elg':>3} | {'Val%':>5} {'Pst':>4} | {'Cst_Sc':>6} {'Elg':>3} {'Ret':>7}")
        print(f"{'─'*W}")
        for i, r in enumerate(results):
            if r['score_custom'] == 0: continue
            el = '✅' if r['eligible'] else '  '
            ec = '✅' if r['eligible_custom'] else '  '
            print(f"{i+1:>3} {r['ticker']:>8} {r['name'][:17]:<18} {r['category'][:13]:<14} | "
                  f"{r['composite']:>5.1f} {r['tcs']:>3} {r['tfs']:>3} {r['oer']:>3} {r['rss']:>3} "
                  f"{r['classification'][:13]:<14} {el:>3} | "
                  f"{r.get('val_prob',50):>5.1f} {r.get('val_persist',50):>4.0f} | "
                  f"{r['score_custom']:>6.1f} {ec:>3} {r['ret_custom']:>6.1f}%")

    # 7-Day History
    history_7d = compute_7day_history(all_data, detector)

    # ───────────────────────────────────────────────────────────────────────
    # NEW: CONTINUATION Class 1-Week Trend Tracking (Console Output)
    # ───────────────────────────────────────────────────────────────────────
    cont_results = [r for r in results if "CONTINUATION" in r['classification']]
    if cont_results:
        def _short_cls(c):
            if "DOWNTREND" in c: return "DOWN"
            if "FADING" in c: return "FADE"
            if "COUNTER" in c: return "CNTR"
            if "WEAKEN" in c: return "WEAK"
            if "NEUTRAL" in c: return "NEUT"
            if "CONSOL" in c: return "CNSLD"
            if "PULLBACK" in c: return "PULL"
            if "RECOVERY" in c: return "RECV"
            if "FORMATION" in c: return "FORM"
            if "OVEREXT" in c: return "OVEX"
            if "EXHAUST" in c: return "EXHA"
            if "CONT" in c: return "CONT"
            return "UNKN"

        print(f"\n{'='*100}")
        print("[CONTINUATION CLASS: 1-Week Class Trend Tracking]")
        print(f"{'Ticker':>8} {'Name':<18} | Trend (Past 7 Days: Oldest -> Newest)")
        print(f"{'─'*100}")
        for r in cont_results:
            t = r['ticker']
            h = history_7d.get(t, [])
            trend_strs = [_short_cls(x['class']) for x in h]
            trend_line = " -> ".join(trend_strs)
            print(f"{t:>8} {r['name'][:17]:<18} | {trend_line}")
        print(f"{'='*100}")

    if run_validation:
        pass

    # ═══════════════════════════════════════════════════════════════════════
    # PDF
    # ═══════════════════════════════════════════════════════════════════════
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    pdf_fn = os.path.join(reports_dir, f"Omega(PD_v5{'_STK' if include_stocks else ''})_{datetime.today().strftime('%Y%m%d')}.pdf")
    print(f"\n📊 Generating PDF: {pdf_fn}")
    pp = PdfPages(pdf_fn)
    viz = VizEngine()

    viz.plot_master(results, pp)
    viz.plot_validity(ve, results, pp)
    viz.plot_3axis_bar(results, pp)
    viz.plot_comparison(results, "1-Week", 'score_1w', 'eligible_1w', 'ret_1w', pp)
    viz.plot_comparison(results, "1-Month", 'score_1m', 'eligible_1m', 'ret_1m', pp)
    viz.plot_comparison(results, "3-Month", 'score_3m', 'eligible_3m', 'ret_3m', pp)
    viz.plot_category_comparison(results, "1-Week", 'score_1w', 'eligible_1w', 'ret_1w', pp)
    viz.plot_category_comparison(results, "1-Month", 'score_1m', 'eligible_1m', 'ret_1m', pp)
    viz.plot_category_comparison(results, "3-Month", 'score_3m', 'eligible_3m', 'ret_3m', pp)
    if custom_date:
        viz.plot_comparison(results, f"Custom({custom_date})", 'score_custom', 'eligible_custom', 'ret_custom', pp)
        viz.plot_category_comparison(results, f"Custom({custom_date})", 'score_custom', 'eligible_custom', 'ret_custom', pp)

    # NEW PDF Tracking Output
    viz.plot_continuation_trend(results, history_7d, pp)

    viz.plot_portfolio_candidates(results, pp)
    viz.plot_7day_trend(history_7d, results, pp)

    pp.close()
    print(f"✅ PDF saved: '{pdf_fn}'")

    # ═══════════════════════════════════════════════════════════════════════
    # TOP-LONG BACKTEST — weekly replay over past 12 months
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n📊 Phase 7: Top-Long Backtest (weekly snapshots, ~50 weeks)...")
    top_long_bt = []
    try:
        ref_ticker = list(all_data.keys())[0]
        all_dates = all_data[ref_ticker].df.index
        n_dates = len(all_dates)
        bullish_cls = {"🟢 CONTINUATION", "🔵 FORMATION", "🟡 CONSOLIDATION",
                       "🔶 PULLBACK", "🔵 RECOVERY", "🟦 LAGGING_CATCHUP"}
        fwd_map = {"1W": 5, "1M": 21, "3M": 63}

        # Category → Sector mapping for group aggregation
        _CAT_SECTOR_MAP = {
            "EQ_Broad": "Equity", "EQ_Technology": "Equity", "EQ_Healthcare": "Equity",
            "EQ_Financials": "Equity", "EQ_ConsDisc": "Equity", "EQ_ConsStaples": "Equity",
            "EQ_Industrials": "Equity", "EQ_Energy": "Equity", "EQ_Materials": "Equity",
            "EQ_Utilities": "Equity", "EQ_RealEstate": "Equity", "EQ_CommServices": "Equity",
            "EQ_Factor": "Equity", "EQ_Thematic": "Equity",
            "Intl_Developed": "Equity", "Emerging_Markets": "Equity", "Korea_Equity": "Equity",
            "FI_Short": "Fixed Income", "FI_Intermediate": "Fixed Income",
            "FI_Long": "Fixed Income", "FI_Credit": "Fixed Income",
            "FI_Inflation": "Fixed Income", "FI_International": "Fixed Income",
            "Commodities": "Commodities", "Real_Assets": "Real Assets",
            "Currency_Vol": "Macro", "Multi_Asset": "Multi Asset",
        }

        # Weekly evaluation points (~5 trading days apart, up to 52 weeks)
        eval_offsets = [i * 5 for i in range(1, 53)]
        eval_offsets = [o for o in eval_offsets if o + 5 < n_dates - 60]

        for offset in eval_offsets:
            eval_idx = n_dates - 1 - offset
            if eval_idx < 60:
                continue
            eval_date = all_dates[eval_idx]
            eval_date_str = str(eval_date.date()) if hasattr(eval_date, 'date') else str(eval_date)[:10]

            # Compute raw + score at eval_date for all tickers
            snap_raw = {}
            snap_scores = {}
            for ticker, etf in all_data.items():
                if not etf.valid:
                    continue
                df_cut = etf.df[etf.df.index <= eval_date]
                if len(df_cut) < 60:
                    continue
                try:
                    raw = detector.compute_raw(df_cut, etf.category)
                    raw['_category'] = etf.category  # URS LeadLag/PeerDispersion 용
                    snap_raw[ticker] = raw
                except:
                    continue

            if len(snap_raw) < 10:
                continue

            # Cross-sectional percentile ranks
            snap_ranks = NaiveDiscoveryDetector.compute_percentile_ranks(snap_raw)

            # Score + classify + O'Neil
            for ticker in snap_raw:
                raw = snap_raw[ticker]
                ranks = snap_ranks[ticker]
                tcs, tcs_s, tcs_l = NaiveDiscoveryDetector.score_tcs(raw)
                tfs, tfs_s, tfs_l = NaiveDiscoveryDetector.score_tfs(raw)
                oer = NaiveDiscoveryDetector.score_oer(raw)
                rss = round(ranks['rss'], 1)
                urs = NaiveDiscoveryDetector.score_urs(ranks)
                comp = NaiveDiscoveryDetector.composite(tcs, tfs, rss, oer, urs)
                cls = NaiveDiscoveryDetector.classify(raw, tcs_s, tcs_l, tfs_s, tfs_l, oer, urs=urs)
                oneil_long = NaiveDiscoveryDetector.score_oneil_long(raw, ranks)
                el_reasons = []
                if cls in ("⬇️ DOWNTREND", "🟤 EXHAUSTING", "🟤 FADING", "🟣 COUNTER_RALLY", "🔴 CYCLE_PEAK", "⚠️ WEAKENING"):
                    el_reasons.append("cls")
                if comp < 55:
                    el_reasons.append("score")
                eligible = len(el_reasons) == 0
                oneil_short = NaiveDiscoveryDetector.score_oneil_short(raw, ranks)
                ev_flag, ev_reasons = NaiveDiscoveryDetector.detect_event_flag(raw)
                sq = NaiveDiscoveryDetector.compute_structural_quality(raw, ranks)
                snap_scores[ticker] = {
                    'comp': comp, 'cls': cls,
                    'tcs': tcs, 'tfs': tfs, 'oer': oer, 'rss': rss,
                    'oneil_long': oneil_long, 'oneil_short': oneil_short,
                    'event_flag': ev_flag, 'event_reasons': "/".join(ev_reasons) if ev_reasons else "",
                    'structural_q': sq,
                    'rsi': round(raw['rsi'], 1), 'trend_age': raw['trend_age'],
                    'eligible': eligible, 'raw': raw,
                }

            # ── Group-level aggregate stats for ENTIRE scored universe ──
            _grp_buckets = {'sector': {}, 'category': {}, 'theme': {}}
            for _tk in snap_scores:
                _etf = all_data[_tk]
                _cat = _etf.category
                _theme = STOCK_THEMES_CONSOLIDATED.get(_tk, '-')
                # Stock tickers: sector from category prefix (STK_Technology → Technology)
                if _cat.startswith('STK_'):
                    _sector = _cat.replace('STK_', '')
                else:
                    _sector = _CAT_SECTOR_MAP.get(_cat, 'Other')
                _sc = snap_scores[_tk]
                for _gtype, _gkey in [('category', _cat), ('sector', _sector), ('theme', _theme)]:
                    if _gkey == '-':
                        continue
                    if _gkey not in _grp_buckets[_gtype]:
                        _grp_buckets[_gtype][_gkey] = {'comps': [], 'tcs': [], 'tfs': [], 'rss': [], 'oer': [], 'cls': []}
                    _b = _grp_buckets[_gtype][_gkey]
                    _b['comps'].append(_sc['comp']); _b['tcs'].append(_sc['tcs']); _b['tfs'].append(_sc['tfs'])
                    _b['rss'].append(_sc['rss']); _b['oer'].append(_sc['oer']); _b['cls'].append(_sc['cls'])
            group_agg = {}
            for _gtype in ('sector', 'category', 'theme'):
                group_agg[_gtype] = {}
                for _gkey, _v in _grp_buckets[_gtype].items():
                    _n = len(_v['comps'])
                    _nb = sum(1 for c in _v['cls'] if c in bullish_cls)
                    group_agg[_gtype][_gkey] = {
                        'n': _n,
                        'avg_comp': round(sum(_v['comps']) / _n, 1),
                        'avg_tcs': round(sum(_v['tcs']) / _n, 1),
                        'avg_tfs': round(sum(_v['tfs']) / _n, 1),
                        'avg_rss': round(sum(_v['rss']) / _n, 1),
                        'avg_oer': round(sum(_v['oer']) / _n, 1),
                        'pct_bullish': round(_nb / _n * 100, 1),
                    }

            # Select top 10 strong long
            cands = [(t, s) for t, s in snap_scores.items()
                     if s['eligible'] and s['cls'] in bullish_cls]
            cands.sort(key=lambda x: -(x[1]['oneil_long'] * 0.5 + x[1]['comp'] * 0.5))
            top10 = cands[:10]

            if not top10:
                continue

            # Measure per-ticker forward returns for each period
            # 날짜 기반: 각 종목 DataFrame에서 eval_date 위치를 개별 탐색
            ticker_details = []  # per-ticker rows
            period_rets = {}     # {period: [ret, ...]} for summary
            for ticker, scores in top10:
                etf = all_data[ticker]
                tk_close = ss(etf.df['Close'])
                tk_dates = etf.df.index
                # 종목별 eval_date 위치 찾기
                tk_eval_pos = tk_dates.searchsorted(eval_date, side='right') - 1
                if tk_eval_pos < 0 or tk_eval_pos >= len(tk_close):
                    continue
                entry = sf(tk_close.iloc[tk_eval_pos])
                tk_last_date = tk_dates[-1]
                _theme = STOCK_THEMES_CONSOLIDATED.get(ticker, '-')
                _cat = etf.category
                if _cat.startswith('STK_'):
                    _sec = _cat.replace('STK_', '')
                else:
                    _sec = _CAT_SECTOR_MAP.get(_cat, 'Other')
                # Compute all 8 hedge strategy scores from raw
                _raw_bt = scores.get('raw', {})
                try:
                    _hs = score_all_strategies(_raw_bt, ranks=None, regime='transition', cat_stats=None)
                    _hs['oneil_long'] = scores['oneil_long']
                    _hs['oneil_short'] = scores['oneil_short']
                    _cs = compute_combined_signal(_hs)
                except:
                    _hs = {}
                    _cs = {'combined_long': 0, 'combined_short': 0, 'long_count': 0, 'short_count': 0, 'net_signal': 'NEUTRAL', 'conviction': 0}
                td = {'ticker': ticker, 'name': etf.name,
                      'sector': _sec,
                      'category': etf.category.replace('STK_', ''),
                      'theme': _theme,
                      'mktcap_B': round(etf.market_cap / 1e9, 2) if etf.market_cap > 0 else 0,
                      'comp': round(scores['comp'], 1),
                      'cls': scores['cls'].split(' ', 1)[-1] if ' ' in scores['cls'] else scores['cls'],
                      'net_signal': _cs.get('net_signal', 'NEUTRAL'),
                      'long_count': _cs.get('long_count', 0),
                      'short_count': _cs.get('short_count', 0),
                      'conviction': round(_cs.get('conviction', 0), 1),
                      'oneil_long': scores['oneil_long'],
                      'oneil_short': scores['oneil_short'],
                      'minervini_long': round(_hs.get('minervini_long', 0), 0),
                      'wyckoff_long': round(_hs.get('wyckoff_long', 0), 0),
                      'ichimoku_long': round(_hs.get('ichimoku_long', 0), 0),
                      'darvas_long': round(_hs.get('darvas_long', 0), 0),
                      'regime_long': round(_hs.get('regime_long', 0), 0),
                      'flow_long': round(_hs.get('flow_long', 0), 0),
                      'relval_long': round(_hs.get('relval_long', 0), 0),
                      'event_flag': scores['event_flag'],
                      'event_reasons': scores['event_reasons'],
                      'structural_q': scores['structural_q'],
                      'alpha_potential': NaiveDiscoveryDetector.compute_alpha_potential(_raw_bt, None),
                      'tcs': scores['tcs'], 'tfs': scores['tfs'],
                      'oer': scores['oer'], 'rss': scores['rss'],
                      'rsi': scores['rsi'], 'trend_age': scores['trend_age']}
                for period_name, fwd_days in fwd_map.items():
                    fwd_pos = tk_eval_pos + fwd_days
                    # forward 기간이 완전히 경과한 경우만 산출
                    # fwd_pos가 종목 데이터 범위 안에 있어야 하고,
                    # 해당 위치의 날짜가 실제 마지막 데이터 날짜 이내여야 함
                    if entry > 0 and fwd_pos < len(tk_close):
                        fwd_date = tk_dates[fwd_pos]
                        # 마지막 데이터 날짜 기준: eval_date + fwd calendar days 가 tk_last_date 이내인지 확인
                        # (trading days → calendar 변환: fwd_days * 1.5 근사)
                        eval_dt = pd.Timestamp(eval_date)
                        required_cal_days = int(fwd_days * 1.45) + 2  # 거래일→캘린더일 변환 (보수적)
                        if eval_dt + pd.Timedelta(days=required_cal_days) <= pd.Timestamp(tk_last_date):
                            exit_p = sf(tk_close.iloc[fwd_pos])
                            ret = round(((exit_p / entry) - 1) * 100, 2)
                        else:
                            ret = None  # 캘린더 기준 아직 미경과
                    else:
                        ret = None  # 데이터 부족
                    td[f'ret_{period_name}'] = ret
                    if ret is not None:
                        period_rets.setdefault(period_name, []).append(ret)
                # Cumulative return: eval_date → 최신 데이터 날짜
                if entry > 0:
                    latest_p = sf(tk_close.iloc[-1])
                    td['ret_CUM'] = round(((latest_p / entry) - 1) * 100, 2)
                else:
                    td['ret_CUM'] = None
                if td['ret_CUM'] is not None:
                    period_rets.setdefault('CUM', []).append(td['ret_CUM'])
                ticker_details.append(td)

            # Summary stats
            summary = {}
            has_any_data = False
            for period_name in list(fwd_map.keys()) + ['CUM']:
                rets = period_rets.get(period_name, [])
                if rets:
                    summary[period_name] = {
                        'avg_ret': round(sum(rets) / len(rets), 2),
                        'hit_rate': round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1),
                    }
                    has_any_data = True
                else:
                    summary[period_name] = {'avg_ret': 0, 'hit_rate': 0}

            # 유효한 forward return이 하나도 없는 시점은 제외
            if has_any_data:
                top_long_bt.append({
                    'eval_date': eval_date_str,
                    'n_picks': len(top10),
                    'tickers': ticker_details,
                    'summary': summary,
                    'group_agg': group_agg,
                })

        print(f"   ✅ Top-Long Backtest: {len(eval_offsets)} weekly snapshots, {len(top_long_bt)} records")
    except Exception as e:
        print(f"   ⚠️ Top-Long Backtest failed: {e}")
        import traceback; traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════════════
    # GRAPHRAG — Knowledge Graph Analysis
    # ═══════════════════════════════════════════════════════════════════════
    graph_data = {}
    if HAS_GRAPH:
        print(f"\n🔗 Phase 6: GraphRAG Knowledge Graph...")
        try:
            pg = PriceDiscoveryGraph()
            pg.build(results, STOCK_THEMES, GLOBAL_ETF_UNIVERSE, STOCK_UNIVERSE,
                     CATEGORY_BENCHMARK, STOCK_BENCHMARK, history_7d)
            pg.detect_communities()
            pg.analyze_all()

            # Console output
            llm_text = pg.export_for_llm()
            print(llm_text[:3000])
            if len(llm_text) > 3000:
                print(f"   ... ({len(llm_text)} chars total, truncated for console)")

            graph_data = {
                'community_stats': pg.community_stats,
                'communities': pg.communities,
                'insights': pg.insights,
                'viz_data': pg.viz_data,
                'summary': pg.get_summary_stats(),
                'formation_pipeline': pg.query_formation_pipeline(),
                'llm_export': llm_text,
            }
        except Exception as e:
            print(f"   ⚠️ GraphRAG failed: {e}")
            import traceback; traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════════════
    # FACTOR EFFICACY — Reverse Factor Model (5 Methodologies)
    # ═══════════════════════════════════════════════════════════════════════
    factor_efficacy_data = {}
    try:
        from factor_efficacy import FactorEfficacyEngine
        print(f"\n📊 Phase 8: Factor Efficacy Analysis (Reverse Factor Model)...")
        fe = FactorEfficacyEngine(all_data, detector)
        factor_efficacy_data = fe.run()
    except ImportError:
        pass
    except Exception as e:
        print(f"   ⚠️ Factor Efficacy failed: {e}")
        import traceback; traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════════════
    # CACHE — save results + history + validity + graph for dashboard
    # ═══════════════════════════════════════════════════════════════════════
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scan_cache.pkl")
    try:
        cache = {
            "cache_version": 3,
            "results": results,
            "history": history_7d,
            "ve_bucket": ve.bucket_stats,
            "ve_class": ve.class_stats,
            "ve_transitions": dict(ve.transition_counts),
            "ve_transition_totals": dict(ve.transition_totals),
            "ve_observations": ve.observations,
            "ve_fwd_bucket": ve.fwd_bucket_stats,
            "ve_fwd_class": ve.fwd_class_stats,
            "ve_fwd_eligible": ve.fwd_eligible_stats,
            "ve_transition_hit": {f"{k[0]}|||{k[1]}": v for k, v in ve.transition_hit.items()},
            "ve_score_weighted": ve.score_weighted,
            "ve_oos_bucket": ve.oos_bucket_stats,
            "ve_oos_class": ve.oos_class_stats,
            "ve_adaptive_thresholds": ve.adaptive_thresholds,
            "graph": graph_data,
            "top_long_bt": top_long_bt,
            "factor_efficacy": factor_efficacy_data,
            "scan_time": datetime.today().isoformat(),
            "include_stocks": include_stocks,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"💾 Cache saved: {cache_path}")
    except Exception as e:
        print(f"⚠️ Cache save failed: {e}")

    return pd.DataFrame(results), results, all_data


# =============================================================================
if __name__ == "__main__":
    df_results, full_results, all_data = run_scan(
        lookback_days=365*5,
        custom_date="2026-02-27",
        use_realtime=True,
        include_stocks=True,      # M7 + LITE 포함
    )
