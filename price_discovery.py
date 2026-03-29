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
    "US_Equity_Core": "SPY", "US_Sectors": "SPY", "US_Factors": "SPY",
    "Intl_Developed": "VEA", "Emerging_Markets": "EEM", "Fixed_Income": "AGG",
    "Commodities": "DBC", "Real_Assets": "VNQ", "Thematic": "QQQ",
    "Korea_Equity": "069500.KS", "Currency_Vol": "UUP", "Multi_Asset": "AOR",
}

GLOBAL_ETF_UNIVERSE = {
    "US_Equity_Core": {"tickers": {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "DIA": "Dow Jones 30", "RSP": "S&P 500 Equal Weight", "IWF": "Russell 1000 Growth", "IWD": "Russell 1000 Value", "VUG": "Large Cap Growth", "VTV": "Large Cap Value", "IJH": "S&P MidCap 400", "IWM": "Russell 2000", "IJR": "S&P SmallCap 600", "VBR": "Small Cap Value", "VBK": "Small Cap Growth", "MGK": "Mega Cap Growth", "MGV": "Mega Cap Value", "OEF": "S&P 100 Mega Cap"}},
    "US_Sectors": {"tickers": {"XLK": "Technology", "XLF": "Financials", "XLV": "Health Care", "XLE": "Energy", "XLI": "Industrials", "XLY": "Consumer Discretionary", "XLP": "Consumer Staples", "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Communication Services", "XLB": "Materials", "SMH": "Semiconductors", "SOXX": "Semiconductor Index", "IBB": "Biotech", "XBI": "Biotech Equal Weight", "ITA": "Aerospace & Defense", "CIBR": "Cybersecurity", "TAN": "Solar Energy", "ICLN": "Clean Energy", "PAVE": "Infrastructure", "ITB": "Home Construction", "KRE": "Regional Banks", "XHB": "Homebuilders", "ARKK": "Disruptive Innovation", "HACK": "Cybersecurity", "BOTZ": "Robotics & AI", "LIT": "Lithium & Battery"}},
    "US_Factors": {"tickers": {"MTUM": "Momentum", "QUAL": "Quality", "USMV": "Min Volatility", "VLUE": "Value Factor", "SIZE": "Size Factor", "SCHD": "Dividend Growth", "VIG": "Dividend Appreciation", "DVY": "High Dividend Yield", "NOBL": "Dividend Aristocrats", "COWZ": "Free Cash Flow Yield", "MOAT": "Wide Moat", "SPHQ": "S&P 500 Quality", "SPMO": "S&P 500 Momentum", "DYNF": "Dynamic Multi-Factor"}},
    "Intl_Developed": {"tickers": {"VEA": "FTSE Developed ex-US", "EFA": "MSCI EAFE", "IEFA": "MSCI EAFE Core", "SPDW": "S&P Developed ex-US", "VGK": "FTSE Europe", "EZU": "Eurozone", "HEDJ": "Europe Hedged", "FEZ": "Euro Stoxx 50", "EWJ": "MSCI Japan", "BBJP": "Japan BetaBuilders", "DXJ": "Japan Hedged Equity", "EWG": "Germany", "EWU": "United Kingdom", "EWQ": "France", "EWL": "Switzerland", "EWA": "Australia", "EWC": "Canada", "EIS": "Israel"}},
    "Emerging_Markets": {"tickers": {"VWO": "FTSE Emerging Markets", "EEM": "MSCI Emerging Markets", "IEMG": "MSCI EM Core", "EMXC": "EM ex-China", "EWZ": "Brazil", "EWT": "Taiwan", "EWY": "South Korea", "INDA": "India", "FXI": "China Large-Cap", "KWEB": "China Internet", "MCHI": "MSCI China", "EWW": "Mexico", "THD": "Thailand", "VNM": "Vietnam", "EIDO": "Indonesia", "TUR": "Turkey", "EZA": "South Africa", "GXG": "Colombia", "ECH": "Chile"}},
    "Fixed_Income": {"tickers": {"SHY": "1-3Y Treasury", "IEI": "3-7Y Treasury", "IEF": "7-10Y Treasury", "TLT": "20+Y Treasury", "TLH": "10-20Y Treasury", "LQD": "Investment Grade Corp", "HYG": "High Yield Corp", "USHY": "Broad High Yield", "VCIT": "Intermediate Corp", "VCSH": "Short-Term Corp", "AGG": "US Aggregate Bond", "BND": "Total Bond Market", "TIP": "TIPS", "VTIP": "Short-Term TIPS", "BNDX": "Total Intl Bond", "EMB": "EM Bonds USD", "CEMB": "EM HC Bonds", "LEMB": "EM LC Bonds", "IAGG": "Intl Aggregate", "PFF": "Preferred Stock", "JAAA": "AAA CLO", "JBBB": "BB-B CLO", "MBB": "MBS"}},
    "Commodities": {"tickers": {"GLD": "Gold", "SLV": "Silver", "GDX": "Gold Miners", "GDXJ": "Junior Gold Miners", "USO": "Crude Oil (WTI)", "BNO": "Brent Crude Oil", "UNG": "Natural Gas", "PPLT": "Platinum", "PALL": "Palladium", "DBA": "Agriculture", "DBC": "Commodity Index", "GSG": "S&P GSCI Commodity", "XOP": "S&P Oil and Exploration", "COPX": "Copper Miners", "WEAT": "Wheat", "CORN": "Corn", "URA": "Uranium"}},
    "Real_Assets": {"tickers": {"VNQ": "US Real Estate", "VNQI": "Intl Real Estate", "IYR": "US Real Estate Broad", "REM": "Mortgage REITs", "AMLP": "MLP Energy", "MLPX": "MLP Energy", "IFRA": "Infrastructure", "WOOD": "Timber & Forestry", "IBIT": "Bitcoin", "ETHA": "Ethereum"}},
    "Thematic": {"tickers": {"AIQ": "AI & Big Data", "ROBO": "Robotics & Automation", "ARKG": "Genomic Revolution", "ARKW": "Next Gen Internet", "DRIV": "Autonomous & EV", "UFO": "Space", "SKYY": "Cloud Computing", "FINX": "Fintech", "EDOC": "Telemedicine", "QCLN": "Clean Edge Green Energy", "BATT": "Battery Technology", "REMX": "Rare Earth Metals", "XSD": "Semiconductor SPDR", "IGV": "Software", "CLOU": "Cloud Computing", "SHLD": "Global Defense", "463250.KS":"TIGER K방산"}},
    "Korea_Equity": {"tickers": {"069500.KS": "KODEX 200", "229200.KS": "KODEX 코스닥150", "091160.KS": "KODEX 반도체", "487240.KS": "AI핵심전력설비","305720.KS": "KODEX 2차전지", "102110.KS": "TIGER 200", "396500.KS": "TIGER 반도체TOP10", "292150.KS": "TIGER 코리아TOP10", "381170.KS": "TIGER 미국테크TOP10", "381180.KS": "TIGER 미국필라델피아반도체나스닥", "466920.KS": "SOL 조선TOP3플러스", "395160.KS": "KODEX AI반도체", "161510.KS": "PLUS 고배당주"}},
    "Currency_Vol": {"tickers": {"UUP": "US Dollar Bullish", "FXE": "Euro", "FXY": "Japanese Yen", "FXB": "British Pound", "FXA": "Australian Dollar", "CYB": "Chinese Yuan", "VIXY": "VIX Short-Term"}},
    "Multi_Asset": {"tickers": {"AOR": "Growth Allocation", "AOA": "Aggressive Allocation", "AOM": "Moderate Allocation", "AOK": "Conservative Allocation", "RPAR": "Risk Parity", "GAA": "Global Asset Allocation"}}
}

###############################################################################
# SECTION 1-B: INDIVIDUAL STOCK UNIVERSE (extensible structure)
###############################################################################

STOCK_UNIVERSE = {
    # ══════════════════════════════════════════════════════════════════════
    # 1. Magnificent 7 — 글로벌 PM 필수 모니터링 (별도 트래킹 그룹)
    # ══════════════════════════════════════════════════════════════════════
    "STK_Mag7": {"tickers": {
        "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
        "AMZN": "Amazon", "NVDA": "NVIDIA", "META": "Meta Platforms",
        "TSLA": "Tesla",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 2. Semiconductors — 설계 / 파운드리 / 장비 / EDA / 아날로그
    # ══════════════════════════════════════════════════════════════════════
    "STK_Semicon": {"tickers": {
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
        # ── 추가 ──
        "WOLF": "Wolfspeed", "ACLS": "Axcelis Technologies",
        "MKSI": "MKS Instruments", "SLAB": "Silicon Labs",
        "ALGM": "Allegro MicroSystems",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 3. Software & Cybersecurity — Enterprise / SaaS / 보안 / 플랫폼
    # ══════════════════════════════════════════════════════════════════════
    "STK_Software": {"tickers": {
        "ORCL": "Oracle", "SAP": "SAP", "CRM": "Salesforce",
        "PLTR": "Palantir", "INTU": "Intuit", "NOW": "ServiceNow",
        "ADBE": "Adobe", "SHOP": "Shopify", "UBER": "Uber",
        "APP": "AppLovin", "PANW": "Palo Alto Networks",
        "CRWD": "CrowdStrike", "FTNT": "Fortinet",
        "WDAY": "Workday", "TTD": "Trade Desk",
        "SNOW": "Snowflake", "DASH": "DoorDash",
        "TEAM": "Atlassian", "DDOG": "Datadog",
        "FICO": "Fair Isaac", "ZS": "Zscaler",
        "NET": "Cloudflare", "HUBS": "HubSpot",
        "VEEV": "Veeva Systems", "ANSS": "Ansys",
        "COIN": "Coinbase", "MDB": "MongoDB",
        "RBLX": "Roblox", "BILL": "BILL Holdings",
        "TWLO": "Twilio",
        # ── 추가 ──
        "OKTA": "Okta", "PATH": "UiPath",
        "DOCU": "DocuSign", "MNDY": "Monday.com",
        "S": "SentinelOne", "SE": "Sea Limited",
        "GRAB": "Grab Holdings", "PINS": "Pinterest",
        "SNAP": "Snap", "SPOT": "Spotify",
        "ROKU": "Roku", "ESTC": "Elastic",
        "IOT": "Samsara", "GEN": "Gen Digital",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 4. AI / Data Center Infrastructure — 전력 / 네트워킹 / 스토리지 / 냉각
    # ══════════════════════════════════════════════════════════════════════
    "STK_AI_Infra": {"tickers": {
        "CSCO": "Cisco", "ETN": "Eaton Corp",
        "ANET": "Arista Networks", "EQIX": "Equinix",
        "APH": "Amphenol", "CEG": "Constellation Energy",
        "GEV": "GE Vernova", "DELL": "Dell Technologies",
        "DLR": "Digital Realty", "VRT": "Vertiv",
        "VST": "Vistra Energy", "TEL": "TE Connectivity",
        "PWR": "Quanta Services", "GLW": "Corning",
        "AME": "Ametek", "KEYS": "Keysight Technologies",
        "HPE": "HP Enterprise", "SMCI": "Super Micro Computer",
        "VLTO": "Veralto", "NTAP": "NetApp",
        "PSTG": "Pure Storage", "EME": "EMCOR Group",
        "HUBB": "Hubbell", "NRG": "NRG Energy",
        "WDC": "Western Digital", "STX": "Seagate",
        "FLEX": "Flex", "COHR": "Coherent",
        "LITE": "Lumentum", "NTNX": "Nutanix",
        # ── 추가 ──
        "CLS": "Celestica", "POWL": "Powell Industries",
        "AAON": "AAON Inc", "FTV": "Fortive",
        "TDY": "Teledyne", "GNRC": "Generac",
        "WCC": "WESCO Intl", "ENPH": "Enphase Energy",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 5. Healthcare — 제약 / 바이오 / 메드테크 / 보험 / 서비스
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
        "MMC": "Marsh & McLennan", "ICE": "Intercontinental Exchange",
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
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 7. Consumer — 소비재 / 필수소비 / 리테일 / 외식 / 럭셔리 / 레저
    # ══════════════════════════════════════════════════════════════════════
    "STK_Consumer": {"tickers": {
        "WMT": "Walmart", "HD": "Home Depot",
        "PG": "Procter & Gamble", "COST": "Costco",
        "LVMUY": "LVMH", "KO": "Coca-Cola",
        "PEP": "PepsiCo", "PM": "Philip Morris Intl",
        "MCD": "McDonald's", "BKNG": "Booking Holdings",
        "LOW": "Lowe's", "TJX": "TJX Companies",
        "NKE": "Nike", "SBUX": "Starbucks",
        "CL": "Colgate-Palmolive", "MDLZ": "Mondelez",
        "CMG": "Chipotle", "ABNB": "Airbnb",
        "ORLY": "O'Reilly Auto", "AZO": "AutoZone",
        "MNST": "Monster Beverage", "RCL": "Royal Caribbean",
        "TGT": "Target", "ROST": "Ross Stores",
        "YUM": "Yum! Brands", "HLT": "Hilton",
        "LULU": "Lululemon", "MO": "Altria",
        "KHC": "Kraft Heinz", "EL": "Estee Lauder",
        # ── 추가 ──
        "MELI": "MercadoLibre", "CPNG": "Coupang",
        "DPZ": "Domino's Pizza", "DECK": "Deckers Outdoor",
        "ULTA": "Ulta Beauty", "MAR": "Marriott Intl",
        "DKNG": "DraftKings", "CCL": "Carnival Corp",
        "LVS": "Las Vegas Sands", "MGM": "MGM Resorts",
        "WYNN": "Wynn Resorts", "ETSY": "Etsy",
        "TPR": "Tapestry", "GRMN": "Garmin",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 8. Industrials & Defense — 방산 / 항공 / 자본재 / 운송 / 폐기물
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
        # ── 추가 ──
        "HWM": "Howmet Aerospace", "HEI": "HEICO Corp",
        "TT": "Trane Technologies", "FAST": "Fastenal",
        "XYL": "Xylem", "LHX": "L3Harris Technologies",
        "LDOS": "Leidos", "BWXT": "BWX Technologies",
        "TXT": "Textron", "GWW": "W.W. Grainger",
        "SNA": "Snap-on", "NDSN": "Nordson",
        "J": "Jacobs Solutions",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 9. Energy & Materials — 석유 / 가스 / 광업 / 화학 / 소재
    # ══════════════════════════════════════════════════════════════════════
    "STK_Energy_Materials": {"tickers": {
        "XOM": "Exxon Mobil", "CVX": "Chevron",
        "LIN": "Linde", "SHEL": "Shell",
        "BHP": "BHP Group", "TTE": "TotalEnergies",
        "COP": "ConocoPhillips", "SHW": "Sherwin-Williams",
        "ENB": "Enbridge", "EOG": "EOG Resources",
        "SLB": "Schlumberger", "FCX": "Freeport-McMoRan",
        "APD": "Air Products", "ECL": "Ecolab",
        "NEM": "Newmont Mining", "MPC": "Marathon Petroleum",
        "FANG": "Diamondback Energy", "PSX": "Phillips 66",
        "VLO": "Valero Energy", "OXY": "Occidental Petroleum",
        "BKR": "Baker Hughes", "CTVA": "Corteva",
        "NUE": "Nucor", "DD": "DuPont",
        "DOW": "Dow Inc", "VMC": "Vulcan Materials",
        "MLM": "Martin Marietta", "HAL": "Halliburton",
        "DVN": "Devon Energy", "CE": "Celanese",
        # ── 추가 ──
        "RIO": "Rio Tinto", "VALE": "Vale SA",
        "GOLD": "Barrick Gold", "WPM": "Wheaton Precious Metals",
        "TECK": "Teck Resources", "ALB": "Albemarle",
        "KMI": "Kinder Morgan", "WMB": "Williams Companies",
        "OKE": "ONEOK", "PPG": "PPG Industries",
        "IFF": "Intl Flavors & Fragrances", "EMN": "Eastman Chemical",
        "MP": "MP Materials",
    }},
    # ══════════════════════════════════════════════════════════════════════
    # 10. Korea — KOSPI 대형주 + 핵심 중형주
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
}

STOCK_BENCHMARK = {
    "STK_Mag7": "QQQ",
    "STK_Semicon": "SMH",
    "STK_Software": "IGV",
    "STK_AI_Infra": "QQQ",
    "STK_Healthcare": "XLV",
    "STK_Financials": "XLF",
    "STK_Consumer": "XLY",
    "STK_Industrials": "XLI",
    "STK_Energy_Materials": "XLE",
    "STK_Korea": "069500.KS",
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
    "ADBE": "Creative/Design", "SHOP": "E-commerce Platform", "UBER": "Mobility Platform",
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
    "CB": "Insurance (P&C)", "MMC": "Insurance Broker", "ICE": "Exchange",
    "MCO": "Data/Ratings", "CME": "Exchange", "AON": "Insurance Broker",
    "PYPL": "Digital Payments", "PNC": "Regional Bank", "USB": "Regional Bank",
    "COF": "Consumer Finance", "TRV": "Insurance (P&C)", "MET": "Insurance (Life)",
    "AFL": "Insurance (Life)", "ALL": "Insurance (P&C)", "MSCI": "Data/Index",
    # ── Consumer ──
    "WMT": "Mass Retail", "HD": "Home Improvement", "PG": "Household Staples",
    "COST": "Warehouse Club", "LVMUY": "Luxury", "KO": "Beverages",
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
    # Korea 추가
    "373220.KS": "Battery (Cell)", "352820.KS": "Entertainment/K-Pop",
    "000810.KS": "Insurance (P&C)", "259960.KS": "Gaming",
    "402340.KS": "Holding/Investment", "011070.KS": "Camera Module",
    "096770.KS": "Energy/Battery", "034020.KS": "Nuclear/Power Plant",
    "078930.KS": "Holding/Energy", "316140.KS": "Banking",
    "036570.KS": "Gaming (MMORPG)", "011200.KS": "Shipping/Container",
    "267260.KS": "Transformer/Switchgear", "042700.KS": "Semicon Equipment",
}


###############################################################################
# SECTION 2: DATA ENGINE
###############################################################################

@dataclass
class ETFData:
    ticker: str; name: str; category: str
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    valid: bool = False; realtime_updated: bool = False

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

def _apply_realtime(df, ticker):
    if df is None or df.empty: return df, False
    try:
        fi = yf.Ticker(ticker).fast_info
        lp = float(fi.get('lastPrice', fi.get('last_price', 0)))
        if lp <= 0 or not np.isfinite(lp): return df, False
    except: return df, False
    today = pd.Timestamp(datetime.today().date())
    last_bar = pd.Timestamp(df.index[-1].date()) if hasattr(df.index[-1], 'date') else pd.Timestamp(df.index[-1])
    if last_bar == today:
        df.loc[df.index[-1], 'Close'] = lp
        df.loc[df.index[-1], 'High'] = max(float(df.loc[df.index[-1], 'High']), lp)
        df.loc[df.index[-1], 'Low'] = min(float(df.loc[df.index[-1], 'Low']), lp)
        return df, True
    elif last_bar < today and today.weekday() < 5:
        pc = float(df['Close'].iloc[-1])
        nr = pd.DataFrame({'Open':[lp],'High':[max(pc,lp)],'Low':[min(pc,lp)],'Close':[lp],'Volume':[0]}, index=[today])
        df = pd.concat([df, nr]).loc[~pd.concat([df, nr]).index.duplicated(keep='last')].sort_index()
        return df, True
    return df, False

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
                            df, rt = _apply_realtime(df, ticker)
                            etf.realtime_updated = rt
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
                        tdf, rtu = _apply_realtime(tdf, ticker)
                        etf.realtime_updated = rtu
                        if rtu: rt += 1
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

        sma50 = close.rolling(50, min_periods=40).mean()
        sma200 = close.rolling(200, min_periods=120).mean() if len(close) >= 120 else close.rolling(50, min_periods=40).mean()

        last_close = sf(close.iloc[-1])
        last_sma50 = sf(sma50.iloc[-1])
        last_sma200 = sf(sma200.iloc[-1])

        above_sma50 = 1 if (last_sma50 > 0 and last_close > last_sma50) else 0
        golden_cross = 1 if (last_sma200 > 0 and last_sma50 > last_sma200) else 0

        sma50_10d_ago = sf(sma50.iloc[-11]) if len(sma50) >= 11 else last_sma50
        sma50_slope = ((last_sma50 / sma50_10d_ago) - 1) * 100 if sma50_10d_ago > 0 else 0.0

        sma50_20d_ago = sf(sma50.iloc[-21]) if len(sma50) >= 21 else last_sma50
        sma50_30d_ago = sf(sma50.iloc[-31]) if len(sma50) >= 31 else sma50_20d_ago
        slope_20d_ago = (sma50_20d_ago / sma50_30d_ago - 1) if sma50_30d_ago > 0 else 0

        above_mask = close > sma50
        trend_age = 0
        for v in above_mask.values[::-1]:
            if v: trend_age += 1
            else: break

        sma50_dist = ((last_close / last_sma50) - 1) * 100 if last_sma50 > 0 else 0.0
        rsi = compute_rsi(close, 14)

        window_52w = min(252, len(close) - 1)
        if window_52w >= 60:
            high_52w = sf(close.rolling(window_52w, min_periods=60).max().iloc[-1])
            low_52w = sf(close.rolling(window_52w, min_periods=60).min().iloc[-1])
        else:
            high_52w, low_52w = last_close, last_close
        pct_from_high = ((last_close / high_52w) - 1) * 100 if high_52w > 0 else 0
        range_pct = (last_close - low_52w) / max(high_52w - low_52w, 1e-10) * 100

        vol_valid = volume[volume > 0]
        vol_5d = sf(vol_valid.iloc[-5:].mean()) if len(vol_valid) >= 5 else sf(vol_valid.mean()) if len(vol_valid) > 0 else 0
        vol_20d = sf(vol_valid.iloc[-20:].mean(), 1.0) if len(vol_valid) >= 20 else max(sf(vol_valid.mean(), 1.0), 1.0)
        vol_ratio = vol_5d / max(vol_20d, 1.0)

        high_20 = sf(close.rolling(20, min_periods=15).max().shift(1).iloc[-1]) if len(close) >= 20 else last_close
        breakout_20d = 1 if last_close > high_20 else 0

        close_21d_ago = sf(close.iloc[-22]) if len(close) >= 22 else last_close
        close_63d_ago = sf(close.iloc[-64]) if len(close) >= 64 else last_close
        close_126d_ago = sf(close.iloc[-127]) if len(close) >= 127 else last_close
        ret_21d = ((last_close / close_21d_ago) - 1) * 100 if close_21d_ago > 0 else 0.0
        ret_63d = ((last_close / close_63d_ago) - 1) * 100 if close_63d_ago > 0 else 0.0
        ret_126d = ((last_close / close_126d_ago) - 1) * 100 if close_126d_ago > 0 else 0.0

        # ── 12-1M Return (Jegadeesh & Titman 1993): 12개월 수익률에서 최근 1개월 제외 ──
        # 단기 반전(short-term reversal) 오염을 회피하는 학계 표준 모멘텀 팩터
        close_252d_ago = sf(close.iloc[-253]) if len(close) >= 253 else close_126d_ago
        ret_12_1m = ((close_21d_ago / close_252d_ago) - 1) * 100 if close_252d_ago > 0 else ret_126d

        # ── Realized Volatility (60일 연환산): return 정규화 기준 ──
        daily_rets = close.pct_change().dropna()
        if len(daily_rets) >= 60:
            realized_vol = float(daily_rets.iloc[-60:].std() * np.sqrt(252) * 100)
        elif len(daily_rets) > 5:
            realized_vol = float(daily_rets.std() * np.sqrt(252) * 100)
        else:
            realized_vol = 20.0
        realized_vol = max(realized_vol, 1.0)

        # ── Vol-Adjusted Momentum (AQR 방식): ret / vol → 고변동 종목 과대평가 방지 ──
        vol_adj_mom = ret_126d / realized_vol

        avg_vol_20d = sf(vol_valid.iloc[-20:].mean()) if len(vol_valid) >= 20 else sf(vol_valid.mean()) if len(vol_valid) > 0 else 0
        avg_price_5d = sf(close.iloc[-5:].mean()) if len(close) >= 5 else last_close
        adv_usd = avg_vol_20d * avg_price_5d

        return {
            'above_sma50': above_sma50,
            'golden_cross': golden_cross,
            'sma50_slope': sma50_slope,
            'sma50_slope_was_neg': 1 if slope_20d_ago < 0 and sma50_slope > 0 else 0,
            'trend_age': trend_age,
            'sma50_dist': sma50_dist,
            'rsi': rsi,
            'pct_from_high': pct_from_high,
            'range_pct': range_pct,
            'vol_ratio': vol_ratio,
            'breakout_20d': breakout_20d,
            'ret_21d': ret_21d,
            'ret_63d': ret_63d,
            'ret_126d': ret_126d,
            'ret_12_1m': ret_12_1m,
            'realized_vol': realized_vol,
            'vol_adj_mom': vol_adj_mom,
            'adv_usd': adv_usd,
            'last_close': last_close,
        }

    @staticmethod
    def score_tcs(raw):
        pts = 0
        if raw['above_sma50']:    pts += 25
        if raw['golden_cross']:   pts += 25
        if raw['sma50_slope'] > 0: pts += 25
        if raw['trend_age'] > 20: pts += 25
        return min(100, pts)

    @staticmethod
    def score_tfs(raw):
        pts = 0
        age = raw['trend_age']
        if raw['above_sma50'] and 1 <= age <= 15:   pts += 30
        elif raw['above_sma50'] and 16 <= age <= 30: pts += 15
        if raw['vol_ratio'] > 1.2:  pts += 25
        if raw['breakout_20d']:     pts += 25
        if raw['sma50_slope_was_neg'] and raw['sma50_slope'] > 0: pts += 20
        return min(100, pts)

    @staticmethod
    def score_oer(raw):
        pts = 0
        dist = raw['sma50_dist']
        if dist > 15:   pts += 60
        elif dist > 10: pts += 40
        elif dist > 5:  pts += 20
        rsi = raw['rsi']
        if rsi > 80:   pts += 40
        elif rsi > 70: pts += 20
        if raw['pct_from_high'] > -2: pts += 20
        return min(100, pts)

    @staticmethod
    def compute_percentile_ranks(all_raw: Dict[str, dict]) -> Dict[str, dict]:
        tickers = list(all_raw.keys())
        indicators = ['sma50_slope', 'trend_age', 'sma50_dist', 'rsi',
                       'range_pct', 'vol_ratio', 'ret_21d', 'ret_63d', 'ret_126d',
                       'ret_12_1m', 'vol_adj_mom']
        arrays = {ind: np.array([all_raw[t][ind] for t in tickers], dtype=float) for ind in indicators}

        ranks = {}
        for i, t in enumerate(tickers):
            r = {}
            for ind in indicators:
                r[ind + '_pctile'] = pct_rank(arrays[ind][i], arrays[ind])
            # RSS v2: 학술/실무 기반 5-factor momentum composite
            #   1. ret_12_1m  — Jegadeesh & Titman(1993) 12-1M 모멘텀 (단기 반전 제외)
            #   2. ret_63d    — 중기 3개월 모멘텀
            #   3. vol_adj_mom — AQR 방식 변동성 조정 6M 모멘텀
            #   4. sma50_slope — 추세 강도
            #   5. range_pct  — George & Hwang(2004) 52주 고점 근접도
            r['rss'] = (r['ret_12_1m_pctile'] + r['ret_63d_pctile']
                        + r['vol_adj_mom_pctile'] + r['sma50_slope_pctile']
                        + r['range_pct_pctile']) / 5.0
            ranks[t] = r
        return ranks

    @staticmethod
    def classify(raw, tcs, tfs, oer):
        if not raw['above_sma50']:
            return "⬇️ DOWNTREND"
        if oer >= 60:
            return "🟡 OVEREXTENDED"
        if tfs >= 50 and raw['trend_age'] <= 20:
            return "🔵 FORMATION"

        # ─── NEW: Exhaustion Filter ───
        # 60일 이상 된 추세이면서 최근 21일의 수익률 모멘텀이 과거 63일 수익률의 1/3에도 못 미치면 소진된 것으로 간주
        if raw['trend_age'] > 60 and raw['ret_21d'] < (raw['ret_63d'] / 3.0):
            return "🟤 EXHAUSTING"

        if tcs >= 75:
            return "🟢 CONTINUATION"
        return "🟠 NEUTRAL"

    @staticmethod
    def composite(tcs, tfs, rss, oer):
        return round(0.35 * tcs + 0.30 * tfs + 0.35 * rss, 1)

    def analyze_single(self, df, category=""):
        raw = self.compute_raw(df, category)
        tcs = self.score_tcs(raw)
        tfs = self.score_tfs(raw)
        oer = self.score_oer(raw)
        avg_ret = (raw['ret_21d'] + raw['ret_63d'] + raw['ret_126d']) / 3.0
        rss = min(100, max(0, 50 + avg_ret * 2))
        comp = self.composite(tcs, tfs, rss, oer)
        cls = self.classify(raw, tcs, tfs, oer)
        return {'composite': comp, 'tcs': tcs, 'tfs': tfs, 'oer': oer, 'rss': rss,
                'classification': cls, 'rsi': raw['rsi'], 'trend_age': raw['trend_age'],
                'sma50_dist': raw['sma50_dist'], 'adv_usd': raw['adv_usd'],
                'last_close': raw['last_close']}


###############################################################################
# SECTION 4: PORTFOLIO ELIGIBILITY
###############################################################################

def evaluate_eligible(analysis, adv_usd, min_adv=5_000_000):
    cls = analysis['classification']
    comp = analysis['composite']
    reasons = []
    if cls == "⬇️ DOWNTREND":
        reasons.append("Downtrend")
    if cls == "🟤 EXHAUSTING":
        reasons.append("Exhausting")
    if comp < 55:
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
    CLASS_SHORT = {"⬇️ DOWNTREND":"DOWN","🟠 NEUTRAL":"NEUTRAL",
                   "🔵 FORMATION":"FORMATION","🟢 CONTINUATION":"CONT",
                   "🟡 OVEREXTENDED":"OVEREXT", "🟤 EXHAUSTING":"EXHAUST"}

    def __init__(self, n_eval=12, lookback_td=63):
        self.n_eval = n_eval; self.lookback_td = lookback_td
        self.observations = []; self.bucket_stats = {}; self.class_stats = {}
        self.etf_stats = {}; self.transition_counts = defaultdict(int)
        self.transition_totals = defaultdict(int); self.computed = False

    def compute(self, all_data, detector):
        ref = list(all_data.keys())[0]
        dates = all_data[ref].df.index
        n_avail = len(dates)
        if n_avail < self.lookback_td + 60: return
        start_i = max(60, n_avail - self.lookback_td - 1)
        end_i = n_avail - 2
        eval_indices = sorted(set(np.linspace(start_i, end_i, self.n_eval, dtype=int)))
        eval_dates = [dates[i] for i in eval_indices]
        print(f"\n🔍 Validity Engine: {len(eval_dates)} eval points × {len(all_data)} ETFs...")
        prev_cls = {}
        for ei, ed in enumerate(eval_dates):
            cur_cls = {}
            for ticker, etf in all_data.items():
                if not etf.valid: continue
                df_e = etf.df[etf.df.index <= ed]
                if len(df_e) < 60: continue
                try: a = detector.analyze_single(df_e, etf.category)
                except: continue
                cur_cls[ticker] = a['classification']
                ec = sf(df_e['Close'].iloc[-1]); cc = sf(etf.df['Close'].iloc[-1])
                if ec <= 0: continue
                fwd_ret = (cc / ec - 1) * 100
                bench_df = detector.benchmark_map.get(etf.category, detector.benchmark_data)
                b_ret = 0.0
                if bench_df is not None:
                    try:
                        bc = ss(bench_df['Close'])
                        b_ret = (sf(bc.iloc[-1]) / sf(bc.asof(ed)) - 1) * 100
                    except: pass
                bucket = "70-100"
                for lo, hi, lbl in self.SCORE_BUCKETS:
                    if lo <= a['composite'] < hi: bucket = lbl; break
                self.observations.append({'ticker': ticker, 'score': a['composite'],
                    'tcs': a['tcs'], 'tfs': a['tfs'], 'oer': a['oer'],
                    'classification': a['classification'], 'bucket': bucket,
                    'fwd_return': fwd_ret, 'bench_return': b_ret,
                    'excess_return': fwd_ret - b_ret,
                    'eval_date': str(ed.date()) if hasattr(ed, 'date') else str(ed)})
                if ticker in prev_cls:
                    self.transition_counts[(prev_cls[ticker], a['classification'])] += 1
                    self.transition_totals[prev_cls[ticker]] += 1
            prev_cls = cur_cls
            if (ei+1) % 4 == 0: print(f"   ... eval {ei+1}/{len(eval_dates)}")

        self._aggregate(); self.computed = True; self._print()

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

    def _print(self):
        print(f"\n{'='*90}\n[SIGNAL VALIDITY: Score Bucket]")
        print(f"{'Bucket':<10} {'N':>5} {'AbsHit':>7} {'ExcHit':>7} {'AvgRet':>8} {'AvgExc':>8}")
        for lbl in ["0-30","30-50","50-70","70-100"]:
            s = self.bucket_stats.get(lbl, {'n':0,'hit_rate':0,'exc_hit':0,'avg_ret':0,'avg_exc':0})
            print(f"{lbl:<10} {s['n']:>5} {s['hit_rate']:>6.1f}% {s['exc_hit']:>6.1f}% {s['avg_ret']:>7.2f}% {s['avg_exc']:>7.2f}%")
        print(f"\n[SIGNAL VALIDITY: Classification]")
        for cls in ["⬇️ DOWNTREND","🟠 NEUTRAL","🔵 FORMATION","🟢 CONTINUATION","🟡 OVEREXTENDED", "🟤 EXHAUSTING"]:
            s = self.class_stats.get(cls, {'n':0,'hit_rate':0,'exc_hit':0,'avg_ret':0,'avg_exc':0})
            short = self.CLASS_SHORT.get(cls, cls[:12])
            print(f"{short:<12} {s['n']:>5} {s['hit_rate']:>6.1f}% {s['exc_hit']:>6.1f}% {s['avg_ret']:>7.2f}% {s['avg_exc']:>7.2f}%")
        print(f"\n[Transition Matrix (% row→col)]")
        classes = ["⬇️ DOWNTREND","🟠 NEUTRAL","🔵 FORMATION","🟢 CONTINUATION","🟡 OVEREXTENDED", "🟤 EXHAUSTING"]
        shorts = [self.CLASS_SHORT[c] for c in classes]
        header_label = 'From\\To'
        print(f"{header_label:<12}" + "".join(f"{s:>10}" for s in shorts))
        for cf in classes:
            tot = self.transition_totals.get(cf, 0)
            row = f"{self.CLASS_SHORT[cf]:<12}"
            for ct in classes:
                cnt = self.transition_counts.get((cf, ct), 0)
                row += f"{(cnt/tot*100 if tot else 0):>9.1f}%"
            print(f"{row} (n={tot})")
        print(f"{'='*90}")

    def get_validity(self, ticker, comp, cls):
        if not self.computed: return {'val_prob': 50.0, 'val_persist': 50.0, 'val_conf': 'N/A'}
        bucket = "70-100"
        for lo, hi, lbl in self.SCORE_BUCKETS:
            if lo <= comp < hi: bucket = lbl; break
        bs = self.bucket_stats.get(bucket, {'exc_hit': 50, 'n': 0})
        cs = self.class_stats.get(cls, {'exc_hit': 50, 'n': 0})
        es = self.etf_stats.get(ticker, {'exc_hit': 50, 'n': 0})
        w, p = [], []
        if bs['n'] > 0: w.append(min(bs['n'], 50)); p.append(bs['exc_hit'])
        if cs['n'] > 0: w.append(min(cs['n'], 50)); p.append(cs['exc_hit'])
        if es['n'] > 0: w.append(min(es['n'], 30) * 1.5); p.append(es['exc_hit'])
        val = sum(a*b for a, b in zip(w, p)) / sum(w) if w else 50.0
        tot = self.transition_totals.get(cls, 0)
        persist = 50.0
        if tot > 0:
            CLASS_RANK = {"⬇️ DOWNTREND":0,"🟠 NEUTRAL":1,"🟤 EXHAUSTING":1,"🔵 FORMATION":2,"🟡 OVEREXTENDED":2,"🟢 CONTINUATION":3}
            cr = CLASS_RANK.get(cls, 1)
            keep = sum(self.transition_counts.get((cls, ct), 0) for ct, r in CLASS_RANK.items() if r >= cr)
            persist = keep / tot * 100
        total_n = bs.get('n', 0) + es.get('n', 0)
        conf = "H" if total_n >= 30 else "M" if total_n >= 10 else "L"
        return {'val_prob': round(val, 1), 'val_persist': round(persist, 1), 'val_conf': conf}


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
        for cls in ["⬇️ DOWNTREND","🟠 NEUTRAL","🔵 FORMATION","🟢 CONTINUATION","🟡 OVEREXTENDED","🟤 EXHAUSTING"]:
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

    print(f"📊 Phase 2: Cross-sectional percentile ranking...")
    all_ranks = NaiveDiscoveryDetector.compute_percentile_ranks(all_raw)

    ve = SignalValidityEngine()
    ve.compute(all_data, detector)

    print(f"📊 Phase 4: Scoring and classification...")
    results = []

    for ticker in sorted(all_raw.keys()):
        etf = all_data[ticker]
        raw = all_raw[ticker]
        ranks = all_ranks[ticker]
        try:
            tcs = NaiveDiscoveryDetector.score_tcs(raw)
            tfs = NaiveDiscoveryDetector.score_tfs(raw)
            oer = NaiveDiscoveryDetector.score_oer(raw)
            rss = round(ranks['rss'], 1)
            comp = NaiveDiscoveryDetector.composite(tcs, tfs, rss, oer)
            cls = NaiveDiscoveryDetector.classify(raw, tcs, tfs, oer)

            eligible, rejection = evaluate_eligible(
                {'classification': cls, 'composite': comp}, raw['adv_usd']
            )

            validity = ve.get_validity(ticker, comp, cls)
            current_close = raw['last_close']
            data_as_of = fmt_data_as_of(etf.df)

            def _hist_analysis(df_hist):
                if df_hist is None or df_hist.empty or len(df_hist) < 60:
                    return 0.0, False, 0.0
                a = detector.analyze_single(df_hist, etf.category)
                el, _ = evaluate_eligible(a, a['adv_usd'])
                hc = sf(df_hist['Close'].iloc[-1])
                ret = ((current_close / hc) - 1) * 100 if hc > 0 else 0.0
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
                'data_as_of': data_as_of, 'realtime_updated': etf.realtime_updated,
                'composite': comp, 'tcs': tcs, 'tfs': tfs, 'oer': oer, 'rss': rss,
                'classification': cls, 'eligible': eligible, 'rejection': rejection,
                'rsi': round(raw['rsi'], 1), 'trend_age': raw['trend_age'],
                'sma50_dist': round(raw['sma50_dist'], 2), 'adv_usd': raw['adv_usd'],
                **validity,
                'score_1w': sc_1w, 'eligible_1w': el_1w, 'ret_1w': ret_1w,
                'score_1m': sc_1m, 'eligible_1m': el_1m, 'ret_1m': ret_1m,
                'score_3m': sc_3m, 'eligible_3m': el_3m, 'ret_3m': ret_3m,
                'score_custom': sc_cst, 'eligible_custom': el_cst, 'ret_custom': ret_cst,
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
            if "DOWN" in c: return "DOWN"
            if "NEUT" in c: return "NEUT"
            if "FORM" in c: return "FORM"
            if "CONT" in c: return "CONT"
            if "OVER" in c: return "OVEX"
            if "EXHA" in c: return "EXHA"
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
    pdf_fn = f"Omega(PD_v5{'_STK' if include_stocks else ''})_{datetime.today().strftime('%Y%m%d')}.pdf"
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
    # CACHE — save results + history + validity for dashboard instant-load
    # ═══════════════════════════════════════════════════════════════════════
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scan_cache.pkl")
    try:
        cache = {
            "results": results,
            "history": history_7d,
            "ve_bucket": ve.bucket_stats,
            "ve_class": ve.class_stats,
            "ve_transitions": dict(ve.transition_counts),
            "ve_transition_totals": dict(ve.transition_totals),
            "ve_observations": ve.observations,
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
