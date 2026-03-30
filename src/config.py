"""
Central configuration for Matrix Trader AI.
All environment variables, constants, and symbol lists.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ───────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
_chat_id_raw = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_CHAT_ID = int(_chat_id_raw) if _chat_id_raw else 0
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

# ─── Trading Config ─────────────────────────────────────────
CAPITAL = float(os.getenv("CAPITAL", "10000"))
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "2"))
DB_PATH = os.getenv("DB_PATH", "data/matrix_trader.db")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ─── Thresholds ─────────────────────────────────────────────
CRYPTO_CONFIDENCE_THRESHOLD = 50
BIST_CONFIDENCE_THRESHOLD = 55
MIN_CONFIDENCE = 45  # Global minimum for any signal
SIGNAL_COOLDOWN_HOURS = 1
SIGNAL_COOLDOWN_MINUTES = SIGNAL_COOLDOWN_HOURS * 60  # 60 min

# ─── Groq Model ─────────────────────────────────────────────
# llama-4-scout: 30K TPM, 500K TPD (vs old 70b: 12K TPM, 100K TPD)
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ─── Timeframes for Multi-TF Analysis ───────────────────────
CRYPTO_TIMEFRAMES = ["5m", "15m", "1h", "4h"]
BIST_TIMEFRAMES = ["1h", "1d", "1wk"]

# ─── BIST 100 Symbols ───────────────────────────────────────
BIST_100 = [
    "AEFES", "AFYON", "AGESA", "AKBNK", "AKFGY", "AKFYE", "AKSA", "AKSEN",
    "ALARK", "ALFAS", "ARCLK", "ARDYZ", "ASELS", "ASUZU", "AYDEM", "AYGAZ",
    "BERA", "BIMAS", "BIOEN", "BRISA", "BRYAT", "BUCIM", "CCOLA", "CIMSA",
    "CWENE", "DOAS", "DOHOL", "ECILC", "EGEEN", "EKGYO", "ENJSA", "ENKAI",
    "EREGL", "EUPWR", "FROTO", "GARAN", "GENIL", "GESAN", "GUBRF", "HALKB",
    "HEKTS", "ISCTR", "ISGYO", "KCHOL", "KERVT", "KMPUR", "KONTR", "KONYA",
    "KOZAA", "KOZAL", "KRDMD", "MGROS", "MPARK", "OBAMS", "ODAS", "OTKAR",
    "OYAKC", "PETKM", "PGSUS", "QUAGR", "SAHOL", "SASA", "SDTTR",
    "SISE", "SKBNK", "SOKM", "TABGD", "TAVHL", "TCELL", "THYAO", "TKFEN",
    "TKNSA", "TOASO", "TRGYO", "TRILC", "TTKOM", "TTRAK", "TUKAS",
    "TUPRS", "TURSG", "ULKER", "VAKBN", "VESBE", "VESTL", "YKBNK",
    "YATAS", "YEOTK", "ZOREN",
]

# ─── Binance Top 100 Crypto ─────────────────────────────────
CRYPTO_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT",
    "TRX/USDT", "MATIC/USDT", "SHIB/USDT", "LTC/USDT", "UNI/USDT",
    "ATOM/USDT", "XLM/USDT", "NEAR/USDT", "FIL/USDT", "APT/USDT",
    "ARB/USDT", "OP/USDT", "HBAR/USDT", "VET/USDT", "ALGO/USDT",
    "FTM/USDT", "AAVE/USDT", "GRT/USDT", "STX/USDT", "SAND/USDT",
    "MANA/USDT", "THETA/USDT", "AXS/USDT", "EOS/USDT", "EGLD/USDT",
    "FLOW/USDT", "XTZ/USDT", "CHZ/USDT", "GALA/USDT", "CAKE/USDT",
    "RUNE/USDT", "ZIL/USDT", "ENJ/USDT", "BAT/USDT", "LRC/USDT",
    "COMP/USDT", "SNX/USDT", "YFI/USDT", "SUSHI/USDT", "CRV/USDT",
    "1INCH/USDT", "DYDX/USDT", "KAVA/USDT", "ROSE/USDT", "CELO/USDT",
    "ANKR/USDT", "SUI/USDT", "SEI/USDT", "INJ/USDT", "TIA/USDT",
    "JUP/USDT", "RENDER/USDT", "FET/USDT", "ONDO/USDT", "PEPE/USDT",
    "WIF/USDT", "BONK/USDT", "FLOKI/USDT", "WLD/USDT", "PYTH/USDT",
    "JTO/USDT", "STRK/USDT", "PIXEL/USDT", "MANTA/USDT", "DYM/USDT",
    "ALT/USDT", "AI/USDT", "BOME/USDT", "ENA/USDT", "W/USDT",
    "TAO/USDT", "TON/USDT", "NOT/USDT", "IO/USDT", "ZK/USDT",
    "LISTA/USDT", "ZRO/USDT", "BLAST/USDT", "ETHFI/USDT", "REZ/USDT",
    "BB/USDT", "PEOPLE/USDT", "ORDI/USDT", "PENDLE/USDT",
    "MKR/USDT", "IMX/USDT", "GMT/USDT", "APE/USDT", "CFX/USDT",
]

# ─── Macro Symbols ───────────────────────────────────────────
MACRO_SYMBOLS = {
    "DXY": "DX-Y.NYB",      # Dollar Index
    "USDTRY": "USDTRY=X",   # USD/TRY
    "GOLD": "GC=F",         # Gold Futures
    "US10Y": "^TNX",        # US 10Y Treasury Yield
    "VIX": "^VIX",          # Volatility Index
    "SP500": "^GSPC",       # S&P 500
}

# ─── Signal Tier Definitions ────────────────────────────────
SIGNAL_TIERS = {
    1: {"name": "EXTREME", "emoji": "🔥", "min_indicators": 5},
    2: {"name": "STRONG", "emoji": "💪", "min_indicators": 4},
    3: {"name": "MODERATE", "emoji": "📊", "min_indicators": 3},
    4: {"name": "SPECULATIVE", "emoji": "🎲", "min_indicators": 2},
    5: {"name": "DIVERGENCE", "emoji": "🔀", "min_indicators": 1},
    6: {"name": "CONTRARIAN", "emoji": "🔄", "min_indicators": 1},
}

# ─── Trailing Stop Config ────────────────────────────────────
TRAILING_STOP_ENABLED = True
TRAILING_STOP_ATR_MULT = 2.0      # Trailing SL = price - 2*ATR
TRAILING_STOP_ACTIVATION = 1      # Activate after T1 hit

# ─── Kademeli Kar Alma (Partial Take Profit) ─────────────────
PARTIAL_TP_ENABLED = True
PARTIAL_TP_RATIOS = {
    "t1": 0.33,   # Close 33% at T1
    "t2": 0.33,   # Close 33% at T2
    "t3": 0.34,   # Close 34% at T3 (or trailing stop)
}

# ─── Circuit Breaker (Otomatik Koruma) ───────────────────────
CIRCUIT_BREAKER_ENABLED = True
CIRCUIT_BREAKER_MAX_CONSECUTIVE_LOSSES = 3   # Pause after 3 consecutive SL hits
CIRCUIT_BREAKER_MAX_DAILY_LOSS_PCT = 5.0     # Pause if daily loss > 5%
CIRCUIT_BREAKER_COOLDOWN_HOURS = 4           # How long to pause

# ─── Portfolio Risk Budget ───────────────────────────────────
MAX_OPEN_SIGNALS = 10                 # Max simultaneous open positions
MAX_TOTAL_RISK_PCT = 100.0            # Max total portfolio risk % (price-distance based, position sizing handles capital risk)
MAX_SINGLE_RISK_PCT = 100.0           # Disabled — position sizing already limits capital risk via RISK_PERCENT
MAX_CORRELATED_POSITIONS = 5          # Max positions in same direction

# ─── Funding Rate Config (Crypto) ────────────────────────────
FUNDING_RATE_ENABLED = True
FUNDING_RATE_THRESHOLD = 0.01         # 1% — high funding rate
FUNDING_RATE_PENALTY = 10             # Confidence penalty for high funding

# ─── Scoring Weights (total = 100) ──────────────────────────
SCORE_WEIGHTS = {
    "technical": 40,       # RSI, MACD, BB, indicators
    "mtf_confluence": 20,  # Multi-timeframe alignment
    "volume_profile": 15,  # Volume analysis
    "momentum": 5,         # Price momentum / trend strength
    "sentiment": 5,        # News + Fear/Greed
    "smart_money": 10,     # Whale/volume anomaly
    "macro": 5,            # DXY, USDTRY correlation
}

# ─── Session Killzone (ICT) ──────────────────────────────────
SESSION_FILTER_ENABLED   = True
LONDON_KILLZONE_START    = 2     # UTC hour
LONDON_KILLZONE_END      = 5
NY_KILLZONE_START        = 13
NY_KILLZONE_END          = 16
ASIA_KILLZONE_START      = 0
ASIA_KILLZONE_END        = 2
SESSION_MIN_QUALITY      = 3     # 1=OFF … 6=OVERLAP

# ─── Market Regime Detection ────────────────────────────────
REGIME_DETECTION_ENABLED  = True
ADX_TREND_THRESHOLD       = 25    # ADX > 25 = trending
ATR_VOLATILE_THRESHOLD    = 2.5   # ATR% > 2.5 = volatile

# ─── Kelly Criterion Position Sizing ────────────────────────
KELLY_SIZING_ENABLED   = True
KELLY_FRACTION         = 0.5      # Half-Kelly
KELLY_MAX_PCT          = 5.0      # Max 5% capital per trade

# ─── Correlation Management ─────────────────────────────────
CORRELATION_ENABLED           = True
MAX_CORRELATION_THRESHOLD     = 0.75
CORRELATION_LOOKBACK_HOURS    = 24

# ─── Enhanced Circuit Breaker ───────────────────────────────
CB_CONSECUTIVE_LOSSES    = 3
CB_HOURLY_LOSS_LIMIT     = 0.03
CB_BTC_DUMP_THRESHOLD    = -0.05  # BTC -5% → halt all
CB_MAX_SPREAD_PCT        = 0.001  # 0.1% max spread
NEWS_KILL_WINDOW_MINUTES = 30

# ─── On-Chain / Fear-Greed ───────────────────────────────────
ONCHAIN_ENABLED      = True
FEAR_GREED_MIN_BUY   = 20
FEAR_GREED_MAX_BUY   = 70

# ─── CVD (Cumulative Volume Delta) ───────────────────────────
CVD_ENABLED     = True
CVD_LOOKBACK    = 50
CVD_SCORE_BOOST = 8   # max score contribution

# ─── Market Structure BOS+CHoCH ──────────────────────────────
MS_ENABLED          = True
MS_SWING_LOOKBACK   = 5

# ─── Order Blocks ────────────────────────────────────────────
OB_ENABLED          = True
OB_MIN_IMPULSE_PCT  = 0.5
OB_TOUCH_TOLERANCE  = 0.003

# ─── Liquidity Sweep ────────────────────────────────────────
SWEEP_ENABLED       = True
SWEEP_MIN_RECOVERY  = 0.3

# ─── Scanner Flood Control ───────────────────────────────────
# Her tarama turunda gönderilecek maksimum sinyal sayısı.
# Aynı turda çok fazla sinyal gönderilmesini engeller.
MAX_SIGNALS_PER_CRYPTO_RUN = 5   # Kripto: max 5 sinyal / tarama
MAX_SIGNALS_PER_BIST_RUN   = 3   # BIST: max 3 sinyal / tarama

# SL yedikten sonra aynı sembol için ek confidence gereksinimi
# (MIN_CONFIDENCE + SL_HIT_CONFIDENCE_BOOST >= erişim eşiği)
SL_HIT_CONFIDENCE_BOOST    = 10  # +10 puan gereksinimi
SL_HIT_LOOKBACK_HOURS      = 24  # Bu süre içinde SL yendiği varsa boost uygulanır

# ─── Paper Trading (Demo Simülasyon) ────────────────────────
PAPER_TRADING_ENABLED         = True
PAPER_TRADING_CAPITAL         = 10000.0   # Başlangıç demo bakiyesi ($)
PAPER_TRADE_MAX_SLIPPAGE_PCT  = 1.0       # İzin verilen max kayma (%)

# ─── Adaptive Confidence Thresholds ─────────────────────────
ADAPTIVE_THRESHOLD_ENABLED    = True
ADAPTIVE_THRESHOLD_HIGH_WR    = 65        # Win rate > 65% → eşiği 5 düşür
ADAPTIVE_THRESHOLD_LOW_WR     = 40        # Win rate < 40% → eşiği 10 arttır
ADAPTIVE_THRESHOLD_RELAX      = 5
ADAPTIVE_THRESHOLD_TIGHTEN    = 10

# ─── Drawdown Recovery Guard ─────────────────────────────────
DRAWDOWN_CAUTION_PCT          = 10.0      # %10 DD → CAUTION
DRAWDOWN_DEFENSIVE_PCT        = 20.0      # %20 DD → DEFENSIVE
DRAWDOWN_HALT_PCT             = 30.0      # %30 DD → HALT

# ─── Order Book Imbalance ─────────────────────────────────────
ORDERBOOK_ENABLED             = True
ORDERBOOK_DEPTH               = 20        # Analiz edilecek seviye sayısı
ORDERBOOK_IMBALANCE_THRESHOLD = 2.0       # bid/ask oranı üstü = güçlü imbalance
ORDERBOOK_MAX_BOOST           = 8         # Max confidence boost

# ─── On-Chain Data (CoinGecko free) ──────────────────────────
ONCHAIN_FEED_ENABLED          = True
BTC_DOMINANCE_HIGH_THRESHOLD  = 62.0      # BTC.D > 62% = alt coinler için caution
BTC_DOMINANCE_LOW_THRESHOLD   = 40.0      # BTC.D < 40% = alt coinler için boost

# ─── Post-Trade AI Journal ────────────────────────────────────
JOURNAL_ENABLED               = True

# ─── KAP Calendar (BIST) ──────────────────────────────────────
KAP_FILTER_ENABLED            = True
KAP_BLACKOUT_MINUTES          = 30        # Açıklama öncesi/sonrası blok süresi

# ─── Limit Order Simulation ──────────────────────────────────
LIMIT_ORDER_SIMULATION        = False     # Açık: limit emir simülasyonu (15dk bekleme)
LIMIT_ORDER_PULLBACK_PCT      = 0.3       # Sinyal fiyatından %0.3 geri çekilme bekle
LIMIT_ORDER_TIMEOUT_MINUTES   = 15        # Bu süre içinde dolmazsa market fiyatından gir

# ─── Multi-Exchange Price Aggregation ────────────────────────
MULTI_EXCHANGE_AGGREGATION    = True      # Birden fazla borsadan median fiyat

# ══════════════════════════════════════════════════════════════
# PRECISION MODE — %95 Directional Accuracy Configuration
# ══════════════════════════════════════════════════════════════

# BIST is disabled — focus exclusively on crypto for maximum accuracy
BIST_ENABLED = False

# ─── Ultra-Strict Signal Gate ─────────────────────────────────
ULTRA_FILTER_ENABLED          = True      # Enable all mandatory gates
ULTRA_CONFIDENCE_MIN          = 80        # Only Grade A signals (was 45/55)
ULTRA_ADX_MIN                 = 25        # Minimum trend strength
ULTRA_VOLUME_RATIO_MIN        = 1.5       # Minimum volume confirmation
ULTRA_MTF_ALIGNED_MIN         = 3         # Min timeframes aligned (of 4)
ULTRA_RR_MIN                  = 2.5       # Minimum Risk:Reward ratio

# ─── Consensus Engine ─────────────────────────────────────────
CONSENSUS_ENGINE_ENABLED      = True
CONSENSUS_REQUIRED            = 8         # Min FOR votes (of 12 systems)
CONSENSUS_AGAINST_MAX         = 1         # Max AGAINST votes allowed

# ─── BTC Trend Bias (Altcoin Filter) ─────────────────────────
BTC_TREND_FILTER_ENABLED      = True
BTC_TREND_STRICT_MODE         = True      # Block alts against strong BTC trend

# ─── Top 20 Liquid Symbols (highest reliability for TA) ──────
# Restricted from 100+ to 20 most liquid — noise reduction
ULTRA_CRYPTO_SYMBOLS = [
    "BTC/USDT",    # Market leader
    "ETH/USDT",    # #2 — high liquidity, reliable TA
    "BNB/USDT",    # Exchange token — strong volume
    "SOL/USDT",    # High momentum coin
    "XRP/USDT",    # Very high liquidity
    "ADA/USDT",    # Stable alt
    "AVAX/USDT",   # L1 leader
    "DOT/USDT",    # Polkadot
    "LINK/USDT",   # Oracle — strong trend behavior
    "MATIC/USDT",  # Polygon — high volume
    "UNI/USDT",    # DeFi leader
    "ATOM/USDT",   # Cosmos
    "LTC/USDT",    # Long history, reliable TA
    "NEAR/USDT",   # L1 growing
    "APT/USDT",    # New L1 with good volume
    "ARB/USDT",    # L2 high volume
    "OP/USDT",     # L2 Optimism
    "INJ/USDT",    # High momentum
    "SUI/USDT",    # New L1
    "TON/USDT",    # Telegram coin — high volume
]

# Override CRYPTO_SYMBOLS when precision mode is active
# (scan_crypto.py uses ULTRA_CRYPTO_SYMBOLS when ULTRA_FILTER_ENABLED)

# ─── Signal Flood Control (stricter in precision mode) ────────
MAX_SIGNALS_PER_CRYPTO_RUN    = 5   # Max 5 per scan (scalp+swing combined)

# ══════════════════════════════════════════════════════════════
# SCALP MODE — Shorter Timeframe Signals (5m/15m primary)
# Runs BEFORE swing scan on each symbol — more notifications
# ══════════════════════════════════════════════════════════════
SCALP_MODE_ENABLED            = True
SCALP_CONFIDENCE_MIN          = 72        # Lower than swing (80)
SCALP_CONSENSUS_REQUIRED      = 7         # Lower than swing (8)
SCALP_ADX_MIN                 = 20        # Minimum trend strength for scalp
SCALP_VOLUME_RATIO_MIN        = 1.3       # Lower than swing (1.5)
SCALP_RR_MIN                  = 1.8       # Minimum R:R for scalp (swing: 2.5)

# BTC trend now uses 1h data for faster/more responsive detection
BTC_TREND_TIMEFRAME           = "1h"      # Changed from 4h to 1h
BTC_TREND_CACHE_TTL           = 300       # 5 min cache (was 900/15 min)
