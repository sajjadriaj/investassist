import yfinance as yf
import pandas as pd
import numpy as np
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("pandas_ta not available, using simplified technical indicators")

try:
    from py_vollib.black_scholes.greeks.analytical import theta, gamma
    HAS_VOLLIB = True
except ImportError:
    HAS_VOLLIB = False
    print("py_vollib not available, Greeks calculations will be skipped")

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
from database import init_database, is_data_fresh, get_market_sentiment_data, save_market_sentiment_data
import sys
import os

# Add the parent directory to the path to import database module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import (
    init_database, is_data_fresh, get_options_data, save_options_data,
    get_market_sentiment_data, save_market_sentiment_data, cleanup_old_data
)

# --- Shared Data Models ---

class PriceData(BaseModel):
    Date: str
    Open: float
    High: float
    Low: float
    Close: float
    Volume: int

# --- Market Sentiment Models ---

class TechnicalIndicatorsData(BaseModel):
    Date: str
    SMA20: Optional[float] = None
    SMA50: Optional[float] = None
    EMA20: Optional[float] = None
    EMA50: Optional[float] = None
    RSI: Optional[float] = None
    MACD: Optional[float] = None
    MACD_signal: Optional[float] = None
    MACD_hist: Optional[float] = None
    BB_lower: Optional[float] = None
    BB_middle: Optional[float] = None
    BB_upper: Optional[float] = None
    VWAP: Optional[float] = None
    ATR: Optional[float] = None
    STOCHk: Optional[float] = None
    STOCHd: Optional[float] = None
    ADX: Optional[float] = None
    DMP: Optional[float] = None
    DMN: Optional[float] = None
    OBV: Optional[float] = None
    Doji: Optional[bool] = None
    Support: Optional[float] = None
    Resistance: Optional[float] = None

class SentimentSignal(BaseModel):
    indicator: str
    signal: str
    description: str
    info: str
    value: Optional[Any] = None

class KeyFinancial(BaseModel):
    label: str
    value: str
    info: str

class FundamentalAnalysis(BaseModel):
    key_financials: List[KeyFinancial]
    recommendation_summary: Optional[str] = None

class MarketSentimentOutput(BaseModel):
    ticker: str
    technical_summary: List[SentimentSignal]
    fundamental_analysis: FundamentalAnalysis
    current_price: float
    price_history: List[PriceData]
    technical_indicators: List[Dict[str, Any]]  # Changed to accept raw dictionaries
    pe_ratio: Optional[float]
    market_cap: Optional[int]
    dividend_yield: Optional[float]

# --- Options Analysis Models ---

class OptionsDataEntry(BaseModel):
    contractSymbol: str
    strike: float
    lastPrice: float
    volume: Optional[int]
    openInterest: Optional[int]
    impliedVolatility: float
    theta: Optional[float] = None
    gamma: Optional[float] = None

class OptionAnalysisPoint(BaseModel):
    title: str
    description: str
    info: str

class ExpirationGroup(BaseModel):
    expiration_date: str
    days_to_expiry: int
    calls: List[OptionsDataEntry]
    puts: List[OptionsDataEntry]

class OptionsAnalysisOutput(BaseModel):
    ticker: str
    put_call_ratio: Optional[float]
    max_pain: Optional[float]
    analysis_points: List[OptionAnalysisPoint]
    calls: List[OptionsDataEntry]  # Keep for backward compatibility
    puts: List[OptionsDataEntry]   # Keep for backward compatibility
    expiration_groups: List[ExpirationGroup] = []
    options_analytics: Optional[Dict[str, Any]] = {}

# --- Portfolio Risk Models ---

class AssetRiskDetails(BaseModel):
    ticker: str
    var: float
    cvar: float
    correlation: float
    beta: Optional[float] = None
    volatility: Optional[float] = None
    weight: Optional[float] = None
    contribution_to_var: Optional[float] = None

class PortfolioRiskOutput(BaseModel):
    tickers: List[str]
    summary: str
    correlation_matrix: Dict[str, Dict[str, float]]
    betas: Dict[str, float]
    portfolio_var: Optional[float] = None
    conditional_var: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    portfolio_volatility: Optional[float] = None
    diversification_ratio: Optional[float] = None
    risk_metrics: Optional[Dict[str, Any]] = {}
    asset_details: List[AssetRiskDetails] = []
    portfolio_performance: Optional[Dict[str, Any]] = {}
    chart_data: Optional[Dict[str, Any]] = {}

# --- Backtesting Models ---

class TradeEntry(BaseModel):
    date: str
    type: str  # 'buy' or 'sell'
    price: float
    size: int
    value: float
    signal: Optional[str] = None
    reason: Optional[str] = None

class PerformanceMetrics(BaseModel):
    total_return: float
    annualized_return: float
    max_drawdown: float
    win_rate: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    total_trades: int
    avg_trade_return: float
    profit_factor: float
    volatility: float

class EquityCurveData(BaseModel):
    date: str
    equity: float
    benchmark: float
    drawdown: float

class BacktestingStrategy(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class BacktestingOutput(BaseModel):
    ticker: str
    strategy: BacktestingStrategy
    summary: str
    equity_curve: List[EquityCurveData]
    performance_metrics: PerformanceMetrics
    trades: List[TradeEntry]
    monthly_returns: Optional[Dict[str, float]] = {}
    risk_metrics: Optional[Dict[str, float]] = {}
    chart_data: Optional[Dict[str, Any]] = {}


# --- Data Fetching & Analysis Functions ---

def get_price_history(ticker, period="1y"):
    return yf.Ticker(ticker).history(period=period)

def get_current_price(ticker):
    todays_data = yf.Ticker(ticker).history(period='1d')
    return todays_data['Close'].iloc[0] if not todays_data.empty else None

def find_support_resistance(hist):
    support = hist['Low'].rolling(window=14, min_periods=1).min()
    resistance = hist['High'].rolling(window=14, min_periods=1).max()
    return support, resistance

def get_technical_indicators(hist):
    hist['Open'] = pd.to_numeric(hist['Open'], errors='coerce')
    hist['High'] = pd.to_numeric(hist['High'], errors='coerce')
    hist['Low'] = pd.to_numeric(hist['Low'], errors='coerce')
    hist['Close'] = pd.to_numeric(hist['Close'], errors='coerce')
    hist['Volume'] = pd.to_numeric(hist['Volume'], errors='coerce')

    if HAS_PANDAS_TA:
        # Use pandas_ta if available
        hist['SMA20'] = ta.sma(hist['Close'], length=20)
        hist['SMA50'] = ta.sma(hist['Close'], length=50)
        hist['EMA20'] = ta.ema(hist['Close'], length=20)
        hist['EMA50'] = ta.ema(hist['Close'], length=50)
        hist['RSI'] = ta.rsi(hist['Close'], length=14)
        macd = ta.macd(hist['Close'], fast=12, slow=26, signal=9)
        hist['MACD'] = macd['MACD_12_26_9']
        hist['MACD_signal'] = macd['MACDs_12_26_9']
        hist['MACD_hist'] = macd['MACDh_12_26_9']
        bbands = ta.bbands(hist['Close'], length=20, std=2)
        hist['BB_lower'] = bbands['BBL_20_2.0']
        hist['BB_middle'] = bbands['BBM_20_2.0']
        hist['BB_upper'] = bbands['BBU_20_2.0']
        hist['VWAP'] = ta.vwap(hist['High'], hist['Low'], hist['Close'], hist['Volume'])
        hist['ATR'] = ta.atr(hist['High'], hist['Low'], hist['Close'], length=14)
        stoch = ta.stoch(hist['High'], hist['Low'], hist['Close'], k=14, d=3)
        hist['STOCHk'] = stoch['STOCHk_14_3_3']
        hist['STOCHd'] = stoch['STOCHd_14_3_3']
        adx = ta.adx(hist['High'], hist['Low'], hist['Close'], length=14)
        hist['ADX'] = adx['ADX_14']
        hist['DMP'] = adx['DMP_14']
        hist['DMN'] = adx['DMN_14']
        hist['OBV'] = ta.obv(hist['Close'], hist['Volume'])
        hist['Doji'] = ta.cdl_doji(hist['Open'], hist['High'], hist['Low'], hist['Close'])
    else:
        # Use simplified calculations if pandas_ta is not available
        hist['SMA20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA50'] = hist['Close'].rolling(window=50).mean()
        
        # Simple EMA calculation
        alpha_20 = 2 / (20 + 1)
        alpha_50 = 2 / (50 + 1)
        hist['EMA20'] = hist['Close'].ewm(alpha=alpha_20).mean()
        hist['EMA50'] = hist['Close'].ewm(alpha=alpha_50).mean()
        
        # Simple RSI calculation
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # Simple MACD
        ema12 = hist['Close'].ewm(span=12).mean()
        ema26 = hist['Close'].ewm(span=26).mean()
        hist['MACD'] = ema12 - ema26
        hist['MACD_signal'] = hist['MACD'].ewm(span=9).mean()
        hist['MACD_hist'] = hist['MACD'] - hist['MACD_signal']
        
        # Simple Bollinger Bands
        sma20 = hist['Close'].rolling(window=20).mean()
        std20 = hist['Close'].rolling(window=20).std()
        hist['BB_middle'] = sma20
        hist['BB_upper'] = sma20 + (std20 * 2)
        hist['BB_lower'] = sma20 - (std20 * 2)
        
        # Set other indicators to None
        hist['VWAP'] = None
        hist['ATR'] = None
        hist['STOCHk'] = None
        hist['STOCHd'] = None
        hist['ADX'] = None
        hist['DMP'] = None
        hist['DMN'] = None
        hist['OBV'] = None
        hist['Doji'] = False
    
    support, resistance = find_support_resistance(hist)
    hist['Support'] = support
    hist['Resistance'] = resistance

    indicator_cols = [field for field in TechnicalIndicatorsData.__annotations__.keys() if field != 'Date']
    for col in indicator_cols:
        if col in hist.columns:
            hist[col] = hist[col].replace({np.nan: None})
    
    if 'Doji' in hist.columns:
        hist['Doji'] = hist['Doji'].apply(lambda x: True if x != 0 else False)
    
    return hist

def get_fundamental_analysis(stock) -> FundamentalAnalysis:
    key_financials = []
    rec_summary = "N/A" # This will remain N/A as recommendations are discarded

    info = stock.info
    key_financials.append(KeyFinancial(label="Revenue Growth (yoy)", value=f"{info.get('revenueGrowth', 0) * 100:.2f}%" if info.get('revenueGrowth') else "N/A", info="Year-over-year revenue growth."))
    key_financials.append(KeyFinancial(label="Earnings Growth (yoy)", value=f"{info.get('earningsGrowth', 0) * 100:.2f}%" if info.get('earningsGrowth') else "N/A", info="Year-over-year earnings growth."))
    debt_to_equity = info.get('debtToEquity')
    key_financials.append(KeyFinancial(label="Debt to Equity", value=f"{debt_to_equity:.2f}" if debt_to_equity is not None and isinstance(debt_to_equity, (int, float)) else "N/A", info="Measures a company's financial leverage."))
    
    forward_pe = info.get('forwardPE')
    key_financials.append(KeyFinancial(label="Forward P/E", value=f"{forward_pe:.2f}" if forward_pe is not None and isinstance(forward_pe, (int, float)) else "N/A", info="Price-to-earnings ratio based on future earnings estimates."))
    
    price_to_book = info.get('priceToBook')
    key_financials.append(KeyFinancial(label="Price-to-Book", value=f"{price_to_book:.2f}" if price_to_book is not None and isinstance(price_to_book, (int, float)) else "N/A", info="Compares a company's market value to its book value."))
    
    enterprise_to_revenue = info.get('enterpriseToRevenue')
    key_financials.append(KeyFinancial(label="EV-to-Revenue", value=f"{enterprise_to_revenue:.2f}" if enterprise_to_revenue is not None and isinstance(enterprise_to_revenue, (int, float)) else "N/A", info="Compares a company's enterprise value to its revenue."))
    key_financials.append(KeyFinancial(label="Profit Margins", value=f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get('profitMargins') else "N/A", info="Measures how much profit a company makes from its sales."))
    key_financials.append(KeyFinancial(label="Return on Equity", value=f"{info.get('returnOnEquity', 0) * 100:.2f}%" if info.get('returnOnEquity') else "N/A", info="Measures a company's profitability in relation to the equity invested by its shareholders."))
    
    return FundamentalAnalysis(
        key_financials=key_financials,
        recommendation_summary=rec_summary
    )

def analyze_market_sentiment(ticker) -> MarketSentimentOutput:
    """Analyze market sentiment with smart incremental data loading"""
    # Initialize database if not exists
    init_database()
    
    # Check if we have fresh sentiment analysis (within 5 minutes for frequent updates)
    if is_data_fresh(ticker, 'market_sentiment', hours=0.083):  # 5 minutes = 0.083 hours
        print(f"Loading cached market sentiment data for {ticker}")
        cached_data = get_market_sentiment_data(ticker)
        if cached_data and cached_data.get('price_history'):
            # Convert technical_summary from dict format to SentimentSignal objects if needed
            technical_summary = []
            for summary in cached_data.get('technical_summary', []):
                if isinstance(summary, dict):
                    technical_summary.append(SentimentSignal(**summary))
                else:
                    technical_summary.append(summary)
            
            # Convert fundamental_analysis from dict to FundamentalAnalysis object if needed
            fundamental_analysis = cached_data.get('fundamental_analysis', {})
            if isinstance(fundamental_analysis, dict) and 'key_financials' in fundamental_analysis:
                key_financials = []
                for financial in fundamental_analysis['key_financials']:
                    if isinstance(financial, dict):
                        key_financials.append(KeyFinancial(**financial))
                    else:
                        key_financials.append(financial)
                fundamental_analysis = FundamentalAnalysis(
                    key_financials=key_financials,
                    recommendation_summary=fundamental_analysis.get('recommendation_summary', 'N/A')
                )
            
            return MarketSentimentOutput(
                ticker=ticker,
                technical_summary=technical_summary,
                fundamental_analysis=fundamental_analysis,
                current_price=cached_data.get('current_price'),
                price_history=cached_data.get('price_history', []),
                technical_indicators=cached_data.get('technical_indicators', []),
                pe_ratio=cached_data.get('pe_ratio'),
                market_cap=cached_data.get('market_cap'),
                dividend_yield=cached_data.get('dividend_yield')
            )
    
    print(f"Fetching/updating market sentiment data for {ticker}")
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Use smart incremental loading
        hist = get_price_history_smart(ticker)
        current_price = get_current_price(ticker)
        tech_indicators = get_technical_indicators(hist.copy())
        
        fundamental_analysis = get_fundamental_analysis(stock)

        pe_ratio = info.get('trailingPE')
        market_cap = info.get('marketCap')
        dividend_yield = info.get('dividendYield')

        price_data = []
        tech_data = []

        if not hist.empty:
            # Convert price data
            price_df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
            
            # Ensure the first column (index) is named 'Date'
            if price_df.columns[0] != 'Date':
                price_df.rename(columns={price_df.columns[0]: 'Date'}, inplace=True)
            
            # Format Date column
            if hasattr(price_df['Date'].iloc[0], 'strftime'):
                price_df['Date'] = price_df['Date'].dt.tz_localize(None).dt.strftime('%Y-%m-%d')
            
            price_data = price_df.to_dict(orient='records')

            # Convert technical indicators data
            tech_df = tech_indicators.reset_index()
            
            # Ensure Date column is properly formatted
            tech_df['Date'] = tech_df['Date'].dt.tz_localize(None).dt.strftime('%Y-%m-%d')
            
            # Get available columns from TechnicalIndicatorsData model
            tech_data_cols = list(TechnicalIndicatorsData.__annotations__.keys())
            available_cols = [col for col in tech_data_cols if col in tech_df.columns]
            
            print(f"Available columns for tech data: {available_cols}")
            
            # Select only the available columns
            tech_df_filtered = tech_df[available_cols].copy()
            
            # Convert to dictionary records manually to ensure proper format
            tech_data = []
            for _, row in tech_df_filtered.iterrows():
                record = {}
                for col in available_cols:
                    value = row[col]
                    # Handle NaN values
                    if pd.isna(value):
                        record[col] = None
                    else:
                        record[col] = float(value) if col != 'Date' else str(value)
                tech_data.append(record)
            
            print(f"Tech data sample after manual conversion: {tech_data[0] if tech_data else 'No tech data'}")
            print(f"Tech data length: {len(tech_data)}")
            
            # Final validation
            for i, record in enumerate(tech_data[:3]):  # Check first 3 records
                print(f"Record {i} has Date field: {'Date' in record}")
                if 'Date' in record:
                    print(f"Record {i} Date value: {record['Date']} (type: {type(record['Date'])})")

        key_points = []
        
        # Handle both DataFrame (fresh data) and list of dicts (cached data)
        if isinstance(tech_indicators, pd.DataFrame):
            last_tech = tech_indicators.iloc[-1] if not tech_indicators.empty else None
            # Convert to dict-like access for consistency
            if last_tech is not None:
                last_tech_dict = last_tech.to_dict()
            else:
                last_tech_dict = None
        elif isinstance(tech_indicators, list) and len(tech_indicators) > 0:
            # tech_indicators is a list of dictionaries from cached data
            last_tech_dict = tech_indicators[-1] if tech_indicators else None
        else:
            last_tech_dict = None

        if last_tech_dict is not None:
            # Use dictionary access instead of attribute access
            if last_tech_dict.get('SMA20') is not None and last_tech_dict.get('SMA50') is not None:
                signal = "Bullish" if last_tech_dict['SMA20'] > last_tech_dict['SMA50'] else "Bearish"
                key_points.append(SentimentSignal(
                    indicator="Moving Averages", 
                    signal=signal, 
                    description=f"The 20-day SMA is {('above' if signal == 'Bullish' else 'below')} the 50-day SMA.", 
                    info="Compares short-term and long-term price momentum."
                ))
            
            if last_tech_dict.get('RSI') is not None:
                rsi_value = last_tech_dict['RSI']
                if rsi_value > 70: signal, desc = "Overbought", "may be overvalued"
                elif rsi_value < 30: signal, desc = "Oversold", "may be undervalued"
                else: signal, desc = "Neutral", "is in neutral territory"
                key_points.append(SentimentSignal(
                    indicator="RSI", 
                    signal=signal, 
                    description=f"RSI is {rsi_value:.2f}, suggesting the asset {desc}.", 
                    info="Measures the speed and change of price movements.", 
                    value=rsi_value
                ))

            if last_tech_dict.get('MACD') is not None and last_tech_dict.get('MACD_signal') is not None:
                signal = "Bullish" if last_tech_dict['MACD'] > last_tech_dict['MACD_signal'] else "Bearish"
                key_points.append(SentimentSignal(
                    indicator="MACD", 
                    signal=signal, 
                    description=f"The MACD line is {('above' if signal == 'Bullish' else 'below')} its signal line.", 
                    info="Shows the relationship between two moving averages of a security's price."
                ))

            if last_tech_dict.get('BB_upper') is not None and last_tech_dict.get('BB_lower') is not None and current_price is not None:
                if current_price > last_tech_dict['BB_upper']: signal, desc = "Overextended", "potential pullback"
                elif current_price < last_tech_dict['BB_lower']: signal, desc = "Potential Buy", "potential buying opportunity"
                else: signal, desc = "Neutral", "no immediate signal"
                key_points.append(SentimentSignal(
                    indicator="Bollinger Bands", 
                    signal=signal, 
                    description=f"Price is trading {('above the upper' if signal == 'Overextended' else 'below the lower' if signal == 'Potential Buy' else 'within the')} band, suggesting a {desc}.", 
                    info="Measure market volatility and identify overbought/oversold conditions."
                ))

            if last_tech_dict.get('ADX') is not None:
                adx_value = last_tech_dict['ADX']
                signal = "Strong Trend" if adx_value > 25 else "Weak Trend"
                key_points.append(SentimentSignal(
                    indicator="ADX", 
                    signal=signal, 
                    description=f"ADX is {adx_value:.2f}, indicating a {signal.lower()} is in place.", 
                    info="Identifies the strength of a trend.", 
                    value=adx_value
                ))

        # Prepare data for database storage
        data_to_save = {
            'current_price': current_price,
            'market_cap': market_cap,
            'pe_ratio': pe_ratio,
            'dividend_yield': dividend_yield,
            'price_history': price_data,
            'technical_indicators': tech_data,
            'technical_summary': [point.dict() for point in key_points],
            'fundamental_analysis': fundamental_analysis.dict() if fundamental_analysis else {}
        }
        
        # Save to database
        save_market_sentiment_data(ticker, data_to_save)
        print(f"Saved market sentiment data for {ticker} to database")

        return MarketSentimentOutput(
            ticker=ticker,
            technical_summary=key_points,
            fundamental_analysis=fundamental_analysis,
            current_price=current_price,
            price_history=price_data,
            technical_indicators=tech_data,
            pe_ratio=pe_ratio,
            market_cap=market_cap,
            dividend_yield=dividend_yield
        )
        
    except Exception as e:
        print(f"Error fetching market sentiment for {ticker}: {e}")
        # Try to return cached data even if it's older
        cached_data = get_market_sentiment_data(ticker)
        if cached_data:
            # Convert technical_summary from dict format to SentimentSignal objects if needed
            technical_summary = []
            for summary in cached_data.get('technical_summary', []):
                if isinstance(summary, dict):
                    technical_summary.append(SentimentSignal(**summary))
                else:
                    technical_summary.append(summary)
            
            # Convert fundamental_analysis from dict to FundamentalAnalysis object if needed
            fundamental_analysis = cached_data.get('fundamental_analysis', {})
            if isinstance(fundamental_analysis, dict) and 'key_financials' in fundamental_analysis:
                key_financials = []
                for financial in fundamental_analysis['key_financials']:
                    if isinstance(financial, dict):
                        key_financials.append(KeyFinancial(**financial))
                    else:
                        key_financials.append(financial)
                fundamental_analysis = FundamentalAnalysis(
                    key_financials=key_financials,
                    recommendation_summary=fundamental_analysis.get('recommendation_summary', 'N/A')
                )
            
            return MarketSentimentOutput(
                ticker=ticker,
                technical_summary=technical_summary,
                fundamental_analysis=fundamental_analysis,
                current_price=cached_data.get('current_price'),
                price_history=cached_data.get('price_history', []),
                technical_indicators=cached_data.get('technical_indicators', []),
                pe_ratio=cached_data.get('pe_ratio'),
                market_cap=cached_data.get('market_cap'),
                dividend_yield=cached_data.get('dividend_yield')
            )
        
        # Return empty data if no cache available
        return MarketSentimentOutput(
            ticker=ticker,
            technical_summary=[],
            fundamental_analysis=FundamentalAnalysis(key_financials=[]),
            current_price=None,
            price_history=[],
            technical_indicators=[],
            pe_ratio=None,
            market_cap=None,
            dividend_yield=None
        )

def get_risk_free_rate():
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="1d")
        return hist['Close'].iloc[-1] / 100
    except:
        return 0.04

def calculate_greeks(chain, underlying_price, risk_free_rate):
    print(f"Starting Greeks calculation. HAS_VOLLIB: {HAS_VOLLIB}")
    print(f"Chain shape: {chain.shape}")
    print(f"Sample row: {chain.iloc[0][['strike', 'lastPrice', 'impliedVolatility']].to_dict()}")
    
    if not HAS_VOLLIB:
        print("py_vollib not available, using simplified Greek calculations")
        # Use more realistic simplified calculations if py_vollib is not available
        today = datetime.now()
        
        # Convert expiration to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(chain['expiration']):
            chain['expiration'] = pd.to_datetime(chain['expiration'])
        
        # Calculate time to expiration in years
        chain['T'] = (chain['expiration'] - today).dt.days / 365.25
        print(f"Time to expiration calculation:")
        print(f"  Today: {today}")
        print(f"  Sample expiration: {chain['expiration'].iloc[0]}")
        print(f"  Days difference: {(chain['expiration'].iloc[0] - today).days}")
        print(f"  Time to expiration range: {chain['T'].min():.4f} to {chain['T'].max():.4f} years")
        
        # If all options have expired or expire today, use a minimum time value
        chain['T'] = chain['T'].apply(lambda x: max(x, 1/365.25))  # Minimum 1 day
        print(f"After minimum adjustment: {chain['T'].min():.4f} to {chain['T'].max():.4f} years")
        
        def simple_theta(row):
            try:
                if row['T'] <= 0 or row['lastPrice'] <= 0:
                    return None
                # More realistic theta: roughly -option_price/(days_to_expiry/365) * time_decay_factor
                time_factor = max(row['T'], 0.01)  # Avoid division by zero
                theta_val = -row['lastPrice'] / (time_factor * 100)  # Simplified daily theta
                return theta_val
            except Exception as e:
                T_val = row['T'] if 'T' in row else 'N/A'
                lastPrice_val = row['lastPrice'] if 'lastPrice' in row else 'N/A'
                print(f"Error in simple_theta: {e}, T={T_val}, lastPrice={lastPrice_val}")
                return None
        
        def simple_gamma(row):
            try:
                if row['T'] <= 0 or row['lastPrice'] <= 0:
                    return None
                # Simplified gamma based on implied volatility and moneyness
                vol = row['impliedVolatility'] if 'impliedVolatility' in row and pd.notna(row['impliedVolatility']) else 0.2
                moneyness = underlying_price / row['strike'] if row['strike'] > 0 else 1
                # Gamma is highest near ATM options
                gamma_factor = 1 - abs(moneyness - 1) * 2  # Higher near ATM
                gamma_factor = max(gamma_factor, 0.1)  # Minimum gamma
                gamma_val = vol * gamma_factor * 0.05  # Scaled gamma
                return gamma_val
            except Exception as e:
                print(f"Error in simple_gamma: {e}")
                return None
        
        print("Applying simple theta calculation...")
        chain['theta'] = chain.apply(simple_theta, axis=1)
        print("Applying simple gamma calculation...")
        chain['gamma'] = chain.apply(simple_gamma, axis=1)
        
        # Check results
        theta_count = chain['theta'].notna().sum()
        gamma_count = chain['gamma'].notna().sum()
        print(f"Calculated {theta_count} theta values and {gamma_count} gamma values")
        
        return chain
        
    today = datetime.now()
    # Convert expiration to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(chain['expiration']):
        chain['expiration'] = pd.to_datetime(chain['expiration'])
        
    chain['T'] = (chain['expiration'] - today).dt.days / 365.25
    # Apply minimum time value to avoid division by zero and handle expired options
    chain['T'] = chain['T'].apply(lambda x: max(x, 1/365.25))  # Minimum 1 day
    print(f"py_vollib section - Time to expiration range: {chain['T'].min():.4f} to {chain['T'].max():.4f} years")
    
    def safe_theta(row):
        try:
            if row['T'] > 0 and row['impliedVolatility'] > 0:
                theta_val = theta('p' if row['optionType'] == 'put' else 'c', 
                           underlying_price, row['strike'], row['T'], 
                           risk_free_rate, row['impliedVolatility'])
                return theta_val
            else:
                print(f"Skipping theta: T={row['T']}, IV={row['impliedVolatility']}")
                return None
        except Exception as e:
            print(f"Error calculating theta for row: {e}, T={row['T']}, IV={row['impliedVolatility']}, Strike={row['strike']}")
            # Fallback calculation
            if row['T'] > 0 and row['lastPrice'] > 0:
                return -row['lastPrice'] / (row['T'] * 100)
            return None
    
    def safe_gamma(row):
        try:
            if row['T'] > 0 and row['impliedVolatility'] > 0:
                gamma_val = gamma('p' if row['optionType'] == 'put' else 'c', 
                           underlying_price, row['strike'], row['T'], 
                           risk_free_rate, row['impliedVolatility'])
                return gamma_val
            else:
                print(f"Skipping gamma: T={row['T']}, IV={row['impliedVolatility']}")
                return None
        except Exception as e:
            print(f"Error calculating gamma for row: {e}, T={row['T']}, IV={row['impliedVolatility']}, Strike={row['strike']}")
            # Fallback calculation
            if row['T'] > 0:
                vol = row['impliedVolatility'] if 'impliedVolatility' in row and pd.notna(row['impliedVolatility']) else 0.2
                moneyness = underlying_price / row['strike'] if row['strike'] > 0 else 1
                gamma_factor = 1 - abs(moneyness - 1) * 2
                gamma_factor = max(gamma_factor, 0.1)
                return vol * gamma_factor * 0.05
            return None
    
    try:
        print(f"Applying safe_theta and safe_gamma functions...")
        print(f"Sample values - T: {chain['T'].iloc[0]:.4f}, IV: {chain['impliedVolatility'].iloc[0]:.4f}, underlying: {underlying_price}, risk_free: {risk_free_rate:.4f}")
        chain['theta'] = chain.apply(safe_theta, axis=1)
        chain['gamma'] = chain.apply(safe_gamma, axis=1)
        
        # Check results after calculation
        theta_values = chain['theta'].dropna()
        gamma_values = chain['gamma'].dropna()
        print(f"After py_vollib calculation - Theta values: {len(theta_values)}, Gamma values: {len(gamma_values)}")
        if len(theta_values) > 0:
            print(f"Theta range: {theta_values.min():.6f} to {theta_values.max():.6f}")
        if len(gamma_values) > 0:
            print(f"Gamma range: {gamma_values.min():.6f} to {gamma_values.max():.6f}")
            
    except Exception as e:
        print(f"Error calculating Greeks: {e}")
        # Fallback to simplified calculations
        chain['theta'] = chain.apply(lambda row: -row['lastPrice'] / (max(row['T'] if 'T' in row and pd.notna(row['T']) else 0.1, 0.01) * 100) if row['lastPrice'] > 0 else None, axis=1)
        chain['gamma'] = chain.apply(lambda row: (row['impliedVolatility'] if 'impliedVolatility' in row and pd.notna(row['impliedVolatility']) else 0.2) * 0.05 if row['lastPrice'] > 0 else None, axis=1)
    
    return chain

def calculate_max_pain(chain):
    strikes = sorted(chain['strike'].unique())
    loss = {}
    for s in strikes:
        calls_loss = (chain[chain['optionType'] == 'call']['openInterest'] * (s - chain[chain['optionType'] == 'call']['strike'])).clip(lower=0).sum()
        puts_loss = (chain[chain['optionType'] == 'put']['openInterest'] * (chain[chain['optionType'] == 'put']['strike'] - s)).clip(lower=0).sum()
        loss[s] = calls_loss + puts_loss
    return min(loss, key=loss.get) if loss else None

def get_options_analytics(options_chain, underlying_price):
    """Calculate advanced analytics for options"""
    analytics = {}
    
    if options_chain.empty:
        return analytics
    
    # Ensure we have time to expiry data
    from datetime import datetime
    if 'T' not in options_chain.columns:
        # Calculate T from days_to_expiry if available
        if 'days_to_expiry' in options_chain.columns:
            options_chain['T'] = options_chain['days_to_expiry'] / 365.0
        elif 'expiration' in options_chain.columns:
            today = datetime.now()
            options_chain['T'] = (pd.to_datetime(options_chain['expiration']) - today).dt.days / 365.0
        else:
            # Default time to expiry of 30 days if no data available
            options_chain['T'] = 30 / 365.0
    
    # Ensure minimum time to expiry to avoid division by zero
    options_chain['T'] = options_chain['T'].fillna(0.1).clip(lower=0.001)
    
    # Calculate risk/reward metrics
    options_chain['moneyness'] = underlying_price / options_chain['strike']
    options_chain['profit_potential'] = np.where(
        options_chain['optionType'] == 'call',
        np.maximum(underlying_price * 1.1 - options_chain['strike'] - options_chain['lastPrice'], 0),
        np.maximum(options_chain['strike'] - underlying_price * 0.9 - options_chain['lastPrice'], 0)
    )
    
    # Risk score (lower is safer)
    options_chain['risk_score'] = (
        abs(options_chain['moneyness'] - 1) * 50 +  # Distance from ATM
        (options_chain['impliedVolatility'].fillna(0.3) * 100) +  # High IV = higher risk
        np.where(options_chain['T'] < 0.1, 30, 0)  # Short expiry = higher risk
    )
    
    # Probability of profit (simplified estimate)
    options_chain['prob_profit'] = np.where(
        options_chain['optionType'] == 'call',
        np.maximum(0.5 - abs(options_chain['moneyness'] - 1) * 0.5, 0.1),
        np.maximum(0.5 - abs(options_chain['moneyness'] - 1) * 0.5, 0.1)
    )
    
    # Find safest options (lowest risk score)
    safest_calls = options_chain[
        (options_chain['optionType'] == 'call') & 
        (options_chain['lastPrice'] > 0.05)
    ].nsmallest(5, 'risk_score')
    
    safest_puts = options_chain[
        (options_chain['optionType'] == 'put') & 
        (options_chain['lastPrice'] > 0.05)
    ].nsmallest(5, 'risk_score')
    
    # Find highest profit potential options
    highest_profit_calls = options_chain[
        (options_chain['optionType'] == 'call') & 
        (options_chain['lastPrice'] > 0.05)
    ].nlargest(5, 'profit_potential')
    
    highest_profit_puts = options_chain[
        (options_chain['optionType'] == 'put') & 
        (options_chain['lastPrice'] > 0.05)
    ].nlargest(5, 'profit_potential')
    
    analytics['safest_calls'] = safest_calls[['strike', 'lastPrice', 'expiration', 'risk_score', 'prob_profit']].to_dict('records')
    analytics['safest_puts'] = safest_puts[['strike', 'lastPrice', 'expiration', 'risk_score', 'prob_profit']].to_dict('records')
    analytics['highest_profit_calls'] = highest_profit_calls[['strike', 'lastPrice', 'expiration', 'profit_potential', 'prob_profit']].to_dict('records')
    analytics['highest_profit_puts'] = highest_profit_puts[['strike', 'lastPrice', 'expiration', 'profit_potential', 'prob_profit']].to_dict('records')
    
    # Chart data for risk/reward analysis
    chart_data = options_chain[options_chain['lastPrice'] > 0.05].copy()
    analytics['risk_reward_data'] = {
        'calls': chart_data[chart_data['optionType'] == 'call'][
            ['strike', 'lastPrice', 'risk_score', 'profit_potential', 'prob_profit', 'expiration']
        ].to_dict('records'),
        'puts': chart_data[chart_data['optionType'] == 'put'][
            ['strike', 'lastPrice', 'risk_score', 'profit_potential', 'prob_profit', 'expiration']
        ].to_dict('records')
    }
    
    return analytics

def analyze_options_data(ticker) -> OptionsAnalysisOutput:
    """Analyze options data with database caching"""
    # Initialize database if not exists
    init_database()
    
    # Check if we have fresh options data (within 30 minutes)
    if is_data_fresh(ticker, 'options_data', hours=0.5):
        print(f"Loading cached options data for {ticker}")
        cached_options = get_options_data(ticker)
        if cached_options:
            # Convert cached data back to the expected format
            return convert_cached_options_to_output(ticker, cached_options)
    
    print(f"Fetching fresh options data for {ticker}")
    
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            # Return demo data to show how the interface works
            demo_data = OptionsAnalysisOutput(
                ticker=ticker, 
                analysis_points=[
                    OptionAnalysisPoint(
                        title="No Options Available", 
                        description="This ticker does not have options data available", 
                        info="Options may not be available for this security"
                    ),
                    OptionAnalysisPoint(
                        title="Demo Mode", 
                        description="Try tickers like AAPL, MSFT, or GOOGL for real options data", 
                        info="Large cap stocks typically have active options markets"
                    )
                ], 
                calls=[], 
                puts=[], 
                put_call_ratio=None, 
                max_pain=None,
                expiration_groups=[],
                options_analytics={}
            )
            return demo_data
        
        # Filter expirations to those within 1 year from now
        from datetime import datetime, timedelta
        today = datetime.now()
        one_year_from_now = today + timedelta(days=365)
        
        valid_expirations = []
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
                if exp_date <= one_year_from_now:
                    valid_expirations.append(exp_str)
            except ValueError:
                continue
        
        if not valid_expirations:
            return OptionsAnalysisOutput(
                ticker=ticker, 
                analysis_points=[OptionAnalysisPoint(
                    title="No Valid Expirations", 
                    description="No options with expiration within 1 year found", 
                    info="Try another ticker or check if options are available"
                )], 
                calls=[], 
                puts=[], 
                put_call_ratio=None, 
                max_pain=None,
                expiration_groups=[],
                options_analytics={}
            )

        # Limit to first 8 expirations to avoid overwhelming the UI
        valid_expirations = valid_expirations[:8]
        
        # Fetch and combine options data for all valid expiration dates
        all_options_data = []
        expiration_groups = []
        options_for_db = []  # Store options data for database
        
        current_price = get_current_price(ticker)
        if current_price is None:
            current_price = 100  # Default fallback
            
        risk_free_rate = get_risk_free_rate()
        
        for exp_str in valid_expirations:
            try:
                opt = stock.option_chain(exp_str)
                exp_date = datetime.strptime(exp_str, '%Y-%m-%d')
                days_to_expiry = (exp_date - today).days
                
                calls = opt.calls.copy()
                calls['optionType'] = 'call'
                calls['expiration'] = pd.to_datetime(exp_str)
                calls['days_to_expiry'] = days_to_expiry
                
                puts = opt.puts.copy()
                puts['optionType'] = 'put'
                puts['expiration'] = pd.to_datetime(exp_str)
                puts['days_to_expiry'] = days_to_expiry
                
                exp_options = pd.concat([calls, puts], ignore_index=True)
                
                if not exp_options.empty:
                    # Normalize implied volatility (convert from percentage to decimal if needed)
                    exp_options['impliedVolatility'] = exp_options['impliedVolatility'].apply(lambda x: x / 100 if x > 1 else x)
                    
                    # Calculate Greeks for this expiration
                    try:
                        exp_options = calculate_greeks(exp_options, current_price, risk_free_rate)
                    except Exception as e:
                        print(f"Error calculating Greeks for {exp_str}: {e}")
                        # Set simplified values if Greeks calculation fails
                        exp_options['theta'] = exp_options.apply(lambda row: -0.01 * row['lastPrice'] if row['lastPrice'] > 0 else None, axis=1)
                        exp_options['gamma'] = exp_options.apply(lambda row: 0.005 if row['lastPrice'] > 0 else None, axis=1)
                    
                    all_options_data.append(exp_options)
                    
                    # Process calls and puts for this expiration
                    exp_calls = exp_options[exp_options['optionType'] == 'call'].copy()
                    exp_puts = exp_options[exp_options['optionType'] == 'put'].copy()
                    
                    # Convert to records and handle NaN values
                    calls_records = []
                    for _, row in exp_calls.iterrows():
                        record = {
                            'contractSymbol': str(row.get('contractSymbol', '')),
                            'strike': float(row.get('strike', 0)),
                            'lastPrice': float(row.get('lastPrice', 0)),
                            'volume': int(row['volume']) if pd.notna(row.get('volume')) else None,
                            'openInterest': int(row['openInterest']) if pd.notna(row.get('openInterest')) else None,
                            'impliedVolatility': float(row.get('impliedVolatility', 0)),
                            'theta': float(row['theta']) if pd.notna(row.get('theta')) else None,
                            'gamma': float(row['gamma']) if pd.notna(row.get('gamma')) else None
                        }
                        calls_records.append(record)
                        
                    puts_records = []
                    for _, row in exp_puts.iterrows():
                        record = {
                            'contractSymbol': str(row.get('contractSymbol', '')),
                            'strike': float(row.get('strike', 0)),
                            'lastPrice': float(row.get('lastPrice', 0)),
                            'volume': int(row['volume']) if pd.notna(row.get('volume')) else None,
                            'openInterest': int(row['openInterest']) if pd.notna(row.get('openInterest')) else None,
                            'impliedVolatility': float(row.get('impliedVolatility', 0)),
                            'theta': float(row['theta']) if pd.notna(row.get('theta')) else None,
                            'gamma': float(row['gamma']) if pd.notna(row.get('gamma')) else None
                        }
                        puts_records.append(record)
                    
                    # Create expiration group
                    exp_group = ExpirationGroup(
                        expiration_date=exp_str,
                        days_to_expiry=days_to_expiry,
                        calls=calls_records,
                        puts=puts_records
                    )
                    expiration_groups.append(exp_group)
                    
            except Exception as e:
                print(f"Error processing expiration {exp_str}: {e}")
                continue

        if not all_options_data:
            return OptionsAnalysisOutput(
                ticker=ticker, 
                analysis_points=[OptionAnalysisPoint(
                    title="No Data", 
                    description="No options data found for this ticker", 
                    info="Try another ticker or check if options are available"
                )], 
                calls=[], 
                puts=[], 
                put_call_ratio=None, 
                max_pain=None,
                expiration_groups=[],
                options_analytics={}
            )

        # Combine all options data for overall analysis
        options_chain = pd.concat(all_options_data, ignore_index=True)
        
        # Calculate max pain with error handling
        try:
            max_pain = calculate_max_pain(options_chain)
        except Exception as e:
            print(f"Error calculating max pain: {e}")
            max_pain = None

        # Calculate put/call ratio
        total_put_oi = options_chain[options_chain['optionType'] == 'put']['openInterest'].fillna(0).sum()
        total_call_oi = options_chain[options_chain['optionType'] == 'call']['openInterest'].fillna(0).sum()
        put_call_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else None

        analysis_points = []
        
        # Add analysis points
        if put_call_ratio is not None:
            sentiment = "Bearish" if put_call_ratio > 0.7 else "Bullish"
            analysis_points.append(OptionAnalysisPoint(
                title="Market Sentiment", 
                description=f"Put/Call Ratio: {put_call_ratio:.2f} - {sentiment}", 
                info="Ratio > 0.7 suggests bearish sentiment, < 0.7 suggests bullish sentiment"
            ))
        
        if max_pain is not None:
            analysis_points.append(OptionAnalysisPoint(
                title="Max Pain Point", 
                description=f"${max_pain:.2f}", 
                info="Price level where most options expire worthless, often acts as a magnet"
            ))
        
        # Find highest volume call (with error handling)
        call_options = options_chain[options_chain['optionType'] == 'call'].copy()
        if not call_options.empty and 'volume' in call_options.columns:
            call_volumes = call_options['volume'].fillna(0)
            if call_volumes.max() > 0:
                high_vol_call_idx = call_volumes.idxmax()
                high_vol_call = call_options.loc[high_vol_call_idx]
                analysis_points.append(OptionAnalysisPoint(
                    title="Highest Call Volume", 
                    description=f"Strike: ${high_vol_call['strike']:.2f} (Volume: {int(high_vol_call['volume'])})", 
                    info="Strike with most call contracts traded today"
                ))
        
        # Find highest open interest put (with error handling)
        put_options = options_chain[options_chain['optionType'] == 'put'].copy()
        if not put_options.empty and 'openInterest' in put_options.columns:
            put_oi = put_options['openInterest'].fillna(0)
            if put_oi.max() > 0:
                high_oi_put_idx = put_oi.idxmax()
                high_oi_put = put_options.loc[high_oi_put_idx]
                analysis_points.append(OptionAnalysisPoint(
                    title="Highest Put Open Interest", 
                    description=f"Strike: ${high_oi_put['strike']:.2f} (OI: {int(high_oi_put['openInterest'])})", 
                    info="Strike with most open put contracts, potential support level"
                ))

        # Calculate advanced analytics
        try:
            analytics = get_options_analytics(options_chain, current_price)
        except Exception as e:
            print(f"Error calculating analytics: {e}")
            analytics = {}

        # For backward compatibility, include first expiration data in calls/puts
        first_exp_calls = expiration_groups[0].calls if expiration_groups else []
        first_exp_puts = expiration_groups[0].puts if expiration_groups else []

        return OptionsAnalysisOutput(
            ticker=ticker,
            put_call_ratio=put_call_ratio,
            max_pain=max_pain,
            analysis_points=analysis_points,
            calls=first_exp_calls,  # Backward compatibility
            puts=first_exp_puts,    # Backward compatibility
            expiration_groups=expiration_groups,
            options_analytics=analytics
        )
        
    except Exception as e:
        print(f"Error in analyze_options_data: {e}")
        return OptionsAnalysisOutput(
            ticker=ticker, 
            analysis_points=[OptionAnalysisPoint(
                title="Error", 
                description=f"Could not retrieve options data: {str(e)}", 
                info="Please try again later or check if the ticker is valid"
            )], 
            calls=[], 
            puts=[], 
            put_call_ratio=None, 
            max_pain=None,
            expiration_groups=[],
            options_analytics={}
        )

def analyze_portfolio_risk(tickers) -> PortfolioRiskOutput:
    """Comprehensive portfolio risk analysis with enhanced metrics"""
    try:
        # Download data
        data = yf.download(tickers, period="1y")['Close']
        
        if data.empty:
            # Return empty structure if no data
            return PortfolioRiskOutput(
                tickers=tickers,
                summary="No data available for the specified tickers",
                correlation_matrix={},
                betas={},
                asset_details=[],
                risk_metrics={},
                portfolio_performance={},
                chart_data={}
            )
            
        returns = data.pct_change().dropna()
        
        # Equal weight portfolio (can be enhanced to support custom weights)
        weights = np.array([1/len(tickers)] * len(tickers))
        
        # Portfolio-level metrics
        portfolio_returns = returns.mean(axis=1)
        portfolio_var = portfolio_returns.quantile(0.05)
        conditional_var = portfolio_returns[portfolio_returns <= portfolio_var].mean()
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_returns.mean() * 252 - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Maximum drawdown
        portfolio_cumulative = (1 + portfolio_returns).cumprod()
        running_max = portfolio_cumulative.expanding().max()
        drawdown = (portfolio_cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Diversification ratio
        individual_volatilities = returns.std() * np.sqrt(252)
        weighted_avg_vol = np.dot(weights, individual_volatilities)
        diversification_ratio = weighted_avg_vol / portfolio_volatility if portfolio_volatility > 0 else 1
        
        # Asset-level metrics
        correlation_matrix = returns.corr()
        betas = {}
        asset_details = []
        
        for i, ticker in enumerate(tickers):
            try:
                beta = get_beta(ticker)
                betas[ticker] = beta
                
                asset_returns = returns[ticker]
                var = asset_returns.quantile(0.05)
                cvar = asset_returns[asset_returns <= var].mean()
                correlation_with_portfolio = asset_returns.corr(portfolio_returns)
                volatility = asset_returns.std() * np.sqrt(252)
                
                # Contribution to portfolio VaR (simplified)
                marginal_var = correlation_matrix.loc[ticker].values @ weights * portfolio_var
                contribution_to_var = weights[i] * marginal_var
                
                asset_details.append(AssetRiskDetails(
                    ticker=ticker,
                    var=var,
                    cvar=cvar,
                    correlation=correlation_with_portfolio,
                    beta=beta,
                    volatility=volatility,
                    weight=weights[i],
                    contribution_to_var=contribution_to_var
                ))
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                # Add basic data even if some calculations fail
                asset_returns = returns[ticker]
                var = asset_returns.quantile(0.05)
                cvar = asset_returns[asset_returns <= var].mean()
                correlation_with_portfolio = asset_returns.corr(portfolio_returns)
                
                asset_details.append(AssetRiskDetails(
                    ticker=ticker,
                    var=var,
                    cvar=cvar,
                    correlation=correlation_with_portfolio,
                    beta=None,
                    volatility=None,
                    weight=weights[i],
                    contribution_to_var=None
                ))
        
        # Risk metrics
        risk_metrics = {
            'volatility_clustering': calculate_volatility_clustering(portfolio_returns),
            'tail_risk': calculate_tail_risk(portfolio_returns),
            'concentration_risk': calculate_concentration_risk(weights),
            'correlation_risk': calculate_correlation_risk(correlation_matrix)
        }
        
        # Portfolio performance over time
        portfolio_performance = {
            'cumulative_returns': portfolio_cumulative.tolist(),
            'dates': portfolio_cumulative.index.strftime('%Y-%m-%d').tolist(),
            'drawdowns': drawdown.tolist(),
            'rolling_volatility': calculate_rolling_volatility(portfolio_returns).tolist(),
            'rolling_sharpe': calculate_rolling_sharpe(portfolio_returns, risk_free_rate).tolist()
        }
        
        # Chart data
        chart_data = prepare_portfolio_chart_data(
            correlation_matrix, 
            asset_details, 
            portfolio_performance,
            returns
        )
        
        return PortfolioRiskOutput(
            tickers=tickers,
            summary=f"Portfolio risk analysis for {len(tickers)} assets completed. "
                   f"Portfolio volatility: {portfolio_volatility:.2%}, Sharpe ratio: {sharpe_ratio:.2f}",
            correlation_matrix=correlation_matrix.to_dict(),
            betas=betas,
            portfolio_var=portfolio_var,
            conditional_var=conditional_var,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            portfolio_volatility=portfolio_volatility,
            diversification_ratio=diversification_ratio,
            risk_metrics=risk_metrics,
            asset_details=asset_details,
            portfolio_performance=portfolio_performance,
            chart_data=chart_data
        )
        
    except Exception as e:
        print(f"Error in analyze_portfolio_risk: {e}")
        import traceback
        traceback.print_exc()
        return PortfolioRiskOutput(
            tickers=tickers,
            summary=f"Error analyzing portfolio risk: {str(e)}",
            correlation_matrix={},
            betas={},
            asset_details=[],
            risk_metrics={},
            portfolio_performance={},
            chart_data={}
        )

def get_beta(ticker, benchmark='^GSPC'):
    data = yf.download([ticker, benchmark], period="1y")['Close']
    log_returns = np.log(data / data.shift(1))
    cov = log_returns.cov().iloc[0, 1]
    var = log_returns[benchmark].var()
    return cov / var

def calculate_performance_metrics(returns, trades_df, equity_curve, risk_free_rate=0.02):
    """Calculate comprehensive performance metrics"""
    try:
        # Basic returns
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        annualized_return = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1) * 100
        
        # Drawdown calculation
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak * 100
        max_drawdown = drawdown.min()
        
        # Volatility and risk metrics
        returns_pct = returns.dropna()
        volatility = returns_pct.std() * np.sqrt(252) * 100
        
        # Sharpe ratio
        excess_returns = returns_pct.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / (returns_pct.std() * np.sqrt(252)) if returns_pct.std() > 0 else 0
        
        # Sortino ratio (uses downside deviation)
        downside_returns = returns_pct[returns_pct < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else returns_pct.std() * np.sqrt(252)
        sortino_ratio = excess_returns / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio
        calmar_ratio = (annualized_return / 100) / abs(max_drawdown / 100) if max_drawdown != 0 else 0
        
        # Trade statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0]) if 'pnl' in trades_df.columns else 0
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_trade_return = trades_df['pnl'].mean() if 'pnl' in trades_df.columns and len(trades_df) > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if 'pnl' in trades_df.columns else 0
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if 'pnl' in trades_df.columns else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            avg_trade_return=avg_trade_return,
            profit_factor=profit_factor,
            volatility=volatility
        )
    except Exception as e:
        print(f"Error calculating performance metrics: {e}")
        return PerformanceMetrics(
            total_return=0, annualized_return=0, max_drawdown=0, win_rate=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, total_trades=0,
            avg_trade_return=0, profit_factor=0, volatility=0
        )

def implement_sma_crossover_strategy(data, short_window=20, long_window=50):
    """Simple Moving Average Crossover Strategy"""
    strategy_data = data.copy()
    
    # Calculate moving averages
    strategy_data['SMA_short'] = strategy_data['Close'].rolling(window=short_window).mean()
    strategy_data['SMA_long'] = strategy_data['Close'].rolling(window=long_window).mean()
    
    # Generate signals
    strategy_data['signal'] = 0
    strategy_data['signal'][short_window:] = np.where(
        strategy_data['SMA_short'][short_window:] > strategy_data['SMA_long'][short_window:], 1, 0
    )
    strategy_data['position'] = strategy_data['signal'].diff()
    
    return strategy_data

def implement_rsi_strategy(data, rsi_period=14, oversold=30, overbought=70):
    """RSI Mean Reversion Strategy"""
    strategy_data = data.copy()
    
    # Calculate RSI
    delta = strategy_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    strategy_data['RSI'] = 100 - (100 / (1 + rs))
    
    # Generate signals
    strategy_data['signal'] = 0
    strategy_data.loc[strategy_data['RSI'] < oversold, 'signal'] = 1  # Buy signal
    strategy_data.loc[strategy_data['RSI'] > overbought, 'signal'] = -1  # Sell signal
    strategy_data['position'] = strategy_data['signal'].diff()
    
    return strategy_data

def implement_bollinger_bands_strategy(data, period=20, std_dev=2):
    """Bollinger Bands Mean Reversion Strategy"""
    strategy_data = data.copy()
    
    # Calculate Bollinger Bands
    strategy_data['SMA'] = strategy_data['Close'].rolling(window=period).mean()
    strategy_data['BB_std'] = strategy_data['Close'].rolling(window=period).std()
    strategy_data['BB_upper'] = strategy_data['SMA'] + (strategy_data['BB_std'] * std_dev)
    strategy_data['BB_lower'] = strategy_data['SMA'] - (strategy_data['BB_std'] * std_dev)
    
    # Generate signals
    strategy_data['signal'] = 0
    strategy_data.loc[strategy_data['Close'] < strategy_data['BB_lower'], 'signal'] = 1  # Buy
    strategy_data.loc[strategy_data['Close'] > strategy_data['BB_upper'], 'signal'] = -1  # Sell
    strategy_data['position'] = strategy_data['signal'].diff()
    
    return strategy_data

def implement_macd_strategy(data, fast=12, slow=26, signal_period=9):
    """MACD Strategy"""
    strategy_data = data.copy()
    
    # Calculate MACD
    ema_fast = strategy_data['Close'].ewm(span=fast).mean()
    ema_slow = strategy_data['Close'].ewm(span=slow).mean()
    strategy_data['MACD'] = ema_fast - ema_slow
    strategy_data['MACD_signal'] = strategy_data['MACD'].ewm(span=signal_period).mean()
    strategy_data['MACD_histogram'] = strategy_data['MACD'] - strategy_data['MACD_signal']
    
    # Generate signals
    strategy_data['signal'] = 0
    strategy_data['signal'][1:] = np.where(
        (strategy_data['MACD'][1:] > strategy_data['MACD_signal'][1:]) & 
        (strategy_data['MACD'][:-1].values <= strategy_data['MACD_signal'][:-1].values), 1, 0
    )
    strategy_data.loc[
        (strategy_data['MACD'] < strategy_data['MACD_signal']) & 
        (strategy_data['MACD'].shift(1) >= strategy_data['MACD_signal'].shift(1)), 'signal'
    ] = -1
    strategy_data['position'] = strategy_data['signal'].diff()
    
    return strategy_data

def simulate_trades(strategy_data, initial_capital=10000, commission=0.001):
    """Simulate trading based on strategy signals"""
    trades = []
    positions = []
    cash = initial_capital
    shares = 0
    
    for i, row in strategy_data.iterrows():
        if pd.isna(row['position']) or row['position'] == 0:
            continue
            
        if row['position'] > 0:  # Buy signal
            if cash > 0:
                cost = cash * commission
                shares_to_buy = int((cash - cost) / row['Close'])
                if shares_to_buy > 0:
                    total_cost = shares_to_buy * row['Close'] + cost
                    trades.append({
                        'date': i.strftime('%Y-%m-%d') if hasattr(i, 'strftime') else str(i),
                        'type': 'buy',
                        'price': row['Close'],
                        'size': shares_to_buy,
                        'value': total_cost,
                        'signal': 'Long Entry',
                        'reason': 'Strategy signal triggered'
                    })
                    cash -= total_cost
                    shares += shares_to_buy
                    
        elif row['position'] < 0 and shares > 0:  # Sell signal
            revenue = shares * row['Close']
            cost = revenue * commission
            trades.append({
                'date': i.strftime('%Y-%m-%d') if hasattr(i, 'strftime') else str(i),
                'type': 'sell',
                'price': row['Close'],
                'size': shares,
                'value': revenue - cost,
                'signal': 'Long Exit',
                'reason': 'Strategy signal triggered'
            })
            cash += revenue - cost
            shares = 0
    
    # Add PnL calculation to trades
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        trades_df['pnl'] = 0
        for i in range(1, len(trades_df), 2):  # Pair buy/sell trades
            if i < len(trades_df):
                buy_trade = trades_df.iloc[i-1]
                sell_trade = trades_df.iloc[i]
                pnl = (sell_trade['price'] - buy_trade['price']) * buy_trade['size']
                trades_df.iloc[i, trades_df.columns.get_loc('pnl')] = pnl
    
    return trades, trades_df

def perform_backtesting(ticker, strategy='sma_crossover') -> BacktestingOutput:
    """Enhanced backtesting with multiple strategies"""
    try:
        # Get historical data (2 years for better backtesting)
        hist = get_price_history(ticker, period="2y")
        if hist.empty:
            raise ValueError(f"No historical data available for {ticker}")
        
        # Get benchmark data (SPY)
        benchmark = get_price_history("SPY", period="2y")['Close']
        
        # Strategy implementations
        strategies = {
            'sma_crossover': {
                'func': implement_sma_crossover_strategy,
                'params': {'short_window': 20, 'long_window': 50},
                'name': 'Simple Moving Average Crossover',
                'description': 'Buy when 20-day SMA crosses above 50-day SMA, sell when it crosses below'
            },
            'rsi_mean_reversion': {
                'func': implement_rsi_strategy,
                'params': {'rsi_period': 14, 'oversold': 30, 'overbought': 70},
                'name': 'RSI Mean Reversion',
                'description': 'Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought)'
            },
            'bollinger_bands': {
                'func': implement_bollinger_bands_strategy,
                'params': {'period': 20, 'std_dev': 2},
                'name': 'Bollinger Bands Mean Reversion',
                'description': 'Buy when price touches lower band, sell when price touches upper band'
            },
            'macd': {
                'func': implement_macd_strategy,
                'params': {'fast': 12, 'slow': 26, 'signal_period': 9},
                'name': 'MACD Strategy',
                'description': 'Buy on MACD bullish crossover, sell on bearish crossover'
            }
        }
        
        # Default to SMA crossover if strategy not found
        if strategy not in strategies:
            strategy = 'sma_crossover'
        
        strategy_config = strategies[strategy]
        
        # Apply strategy
        strategy_data = strategy_config['func'](hist, **strategy_config['params'])
        
        # Simulate trades
        trades, trades_df = simulate_trades(strategy_data)
        
        # Calculate equity curve
        initial_capital = 10000
        equity_curve = []
        cash = initial_capital
        shares = 0
        
        for i, row in strategy_data.iterrows():
            # Update portfolio value
            portfolio_value = cash + (shares * row['Close'])
            
            # Calculate benchmark value
            if not benchmark.empty and i in benchmark.index:
                benchmark_value = initial_capital * (benchmark.loc[i] / benchmark.iloc[0])
            else:
                benchmark_value = initial_capital
            
            # Calculate drawdown
            if len(equity_curve) > 0:
                peak = max([point['equity'] for point in equity_curve])
                drawdown = (portfolio_value - peak) / peak * 100 if peak > 0 else 0
            else:
                drawdown = 0
            
            equity_curve.append({
                'date': i.strftime('%Y-%m-%d') if hasattr(i, 'strftime') else str(i),
                'equity': portfolio_value,
                'benchmark': benchmark_value,
                'drawdown': drawdown
            })
            
            # Execute trades
            if not pd.isna(row.get('position', 0)) and row.get('position', 0) != 0:
                if row['position'] > 0 and cash > 0:  # Buy
                    cost = cash * 0.001  # Commission
                    shares_to_buy = int((cash - cost) / row['Close'])
                    if shares_to_buy > 0:
                        cash -= shares_to_buy * row['Close'] + cost
                        shares += shares_to_buy
                elif row['position'] < 0 and shares > 0:  # Sell
                    revenue = shares * row['Close']
                    cost = revenue * 0.001
                    cash += revenue - cost
                    shares = 0
        
        # Calculate performance metrics
        equity_series = pd.Series([point['equity'] for point in equity_curve])
        returns = equity_series.pct_change().dropna()
        performance_metrics = calculate_performance_metrics(returns, trades_df, equity_series)
        
        # Calculate monthly returns
        monthly_returns = {}
        if len(equity_curve) > 30:
            equity_df = pd.DataFrame(equity_curve)
            equity_df['date'] = pd.to_datetime(equity_df['date'])
            equity_df.set_index('date', inplace=True)
            monthly_equity = equity_df['equity'].resample('M').last()
            monthly_rets = monthly_equity.pct_change().dropna()
            monthly_returns = {
                f"{date.strftime('%Y-%m')}": ret * 100 
                for date, ret in monthly_rets.items()
            }
        
        # Risk metrics
        risk_metrics = {
            'value_at_risk_95': returns.quantile(0.05) * 100,
            'conditional_var_95': returns[returns <= returns.quantile(0.05)].mean() * 100,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'best_month': max(monthly_returns.values()) if monthly_returns else 0,
            'worst_month': min(monthly_returns.values()) if monthly_returns else 0
        }
        
        # Chart data for visualization
        chart_data = {
            'equity_curve': equity_curve,
            'monthly_returns': list(monthly_returns.items()) if monthly_returns else [],
            'drawdown_curve': [point['drawdown'] for point in equity_curve],
            'strategy_name': strategy_config['name']
        }
        
        return BacktestingOutput(
            ticker=ticker,
            strategy=BacktestingStrategy(
                name=strategy_config['name'],
                description=strategy_config['description'],
                parameters=strategy_config['params']
            ),
            summary=f"Backtesting completed for {ticker} using {strategy_config['name']} strategy. "
                   f"Total return: {performance_metrics.total_return:.2f}%, "
                   f"Sharpe ratio: {performance_metrics.sharpe_ratio:.2f}, "
                   f"Max drawdown: {performance_metrics.max_drawdown:.2f}%",
            equity_curve=[EquityCurveData(**point) for point in equity_curve],
            performance_metrics=performance_metrics,
            trades=[TradeEntry(**trade) for trade in trades],
            monthly_returns=monthly_returns,
            risk_metrics=risk_metrics,
            chart_data=chart_data
        )
        
    except Exception as e:
        print(f"Error in backtesting: {e}")
        import traceback
        traceback.print_exc()
        
        # Return minimal data on error
        return BacktestingOutput(
            ticker=ticker,
            strategy=BacktestingStrategy(
                name="Error",
                description=f"Error occurred: {str(e)}",
                parameters={}
            ),
            summary=f"Backtesting failed for {ticker}: {str(e)}",
            equity_curve=[],
            performance_metrics=PerformanceMetrics(
                total_return=0, annualized_return=0, max_drawdown=0, win_rate=0,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, total_trades=0,
                avg_trade_return=0, profit_factor=0, volatility=0
            ),
            trades=[],
            monthly_returns={},
            risk_metrics={},
            chart_data={}
        )

def calculate_volatility_clustering(returns):
    """Calculate volatility clustering metric"""
    try:
        # Use GARCH-like measure: correlation between absolute returns
        abs_returns = returns.abs()
        return abs_returns.autocorr(lag=1) if len(abs_returns) > 1 else 0
    except:
        return 0

def calculate_tail_risk(returns):
    """Calculate tail risk metrics"""
    try:
        # Use 1st percentile as extreme tail risk
        return returns.quantile(0.01)
    except:
        return 0

def calculate_concentration_risk(weights):
    """Calculate concentration risk using Herfindahl index"""
    try:
        # Higher values indicate more concentration
        return np.sum(weights ** 2)
    except:
        return 1

def calculate_correlation_risk(correlation_matrix):
    """Calculate average correlation as a risk measure"""
    try:
        # Remove diagonal and calculate average correlation
        corr_values = correlation_matrix.values
        mask = ~np.eye(corr_values.shape[0], dtype=bool)
        return np.mean(corr_values[mask])
    except:
        return 0

def calculate_rolling_volatility(returns, window=30):
    """Calculate rolling volatility"""
    try:
        return returns.rolling(window=window).std() * np.sqrt(252)
    except:
        return pd.Series([0] * len(returns), index=returns.index)

def calculate_rolling_sharpe(returns, risk_free_rate=0.02, window=30):
    """Calculate rolling Sharpe ratio"""
    try:
        rolling_returns = returns.rolling(window=window).mean() * 252
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        return (rolling_returns - risk_free_rate) / rolling_vol
    except:
        return pd.Series([0] * len(returns), index=returns.index)

def prepare_portfolio_chart_data(correlation_matrix, asset_details, portfolio_performance, returns):
    """Prepare data for portfolio visualization charts"""
    try:
        chart_data = {}
        
        # Correlation matrix data for heatmap
        tickers = correlation_matrix.index.tolist()
        correlation_data = []
        for i, ticker1 in enumerate(tickers):
            for j, ticker2 in enumerate(tickers):
                correlation_data.append({
                    'x': i,
                    'y': j,
                    'v': correlation_matrix.loc[ticker1, ticker2],
                    'ticker1': ticker1,
                    'ticker2': ticker2
                })
        
        chart_data['correlation_matrix'] = {
            'data': correlation_data,
            'labels': tickers
        }
        
        # Risk decomposition data
        chart_data['risk_decomposition'] = {
            'tickers': [asset.ticker for asset in asset_details],
            'var_contributions': [asset.contribution_to_var or 0 for asset in asset_details],
            'volatilities': [asset.volatility or 0 for asset in asset_details],
            'betas': [asset.beta or 0 for asset in asset_details]
        }
        
        # Risk vs return scatter
        risk_return_data = []
        for asset in asset_details:
            if asset.volatility and returns is not None:
                try:
                    asset_return = returns[asset.ticker].mean() * 252
                    risk_return_data.append({
                        'x': asset.volatility,
                        'y': asset_return,
                        'ticker': asset.ticker,
                        'weight': asset.weight
                    })
                except:
                    pass
        
        chart_data['risk_return_scatter'] = risk_return_data
        
        return chart_data
        
    except Exception as e:
        print(f"Error preparing chart data: {e}")
        return {}

def convert_cached_options_to_output(ticker, cached_options):
    """Convert cached options data back to OptionsAnalysisOutput format"""
    try:
        # Group options by expiration
        expiration_groups = {}
        
        for option in cached_options:
            exp_date = option['expiration_date']
            if exp_date not in expiration_groups:
                expiration_groups[exp_date] = {'calls': [], 'puts': []}
            
            option_record = {
                'contractSymbol': f"{ticker}_{option['expiration_date']}_{option['option_type'][0].upper()}{option['strike']}",
                'strike': option['strike'],
                'lastPrice': option['last_price'],
                'volume': option['volume'],
                'openInterest': option['open_interest'],
                'impliedVolatility': option['implied_volatility'],
                'theta': option['theta'],
                'gamma': option['gamma']
            }
            
            if option['option_type'] == 'call':
                expiration_groups[exp_date]['calls'].append(option_record)
            else:
                expiration_groups[exp_date]['puts'].append(option_record)
        
        # Convert to ExpirationGroup objects
        groups = []
        for exp_date, options in expiration_groups.items():
            from datetime import datetime
            exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
            days_to_expiry = (exp_dt - datetime.now()).days
            
            groups.append(ExpirationGroup(
                expiration_date=exp_date,
                days_to_expiry=days_to_expiry,
                calls=options['calls'],
                puts=options['puts']
            ))
        
        # Create analysis points
        analysis_points = [
            OptionAnalysisPoint(
                title="Cached Data", 
                description="Displaying cached options data", 
                info="Data retrieved from local database cache"
            )
        ]
        
        return OptionsAnalysisOutput(
            ticker=ticker,
            analysis_points=analysis_points,
            calls=[],  # Individual calls/puts not needed when using expiration groups
            puts=[],
            put_call_ratio=None,
            max_pain=None,
            expiration_groups=groups,
            options_analytics={}
        )
        
    except Exception as e:
        print(f"Error converting cached options data: {e}")
        return OptionsAnalysisOutput(
            ticker=ticker,
            analysis_points=[OptionAnalysisPoint(
                title="Cache Error", 
                description="Error loading cached data", 
                info="Fetching fresh data instead"
            )],
            calls=[], puts=[], put_call_ratio=None, max_pain=None,
            expiration_groups=[], options_analytics={}
        )

def get_price_history_smart(ticker, period=None, interval=None):
    """
    Smart price history fetching with incremental loading and intraday support:
    - If no data exists: fetch 1 year of data
    - If data exists: fetch only new data since last update
    - For short periods (1d, 5d): use intraday intervals for more granular data
    """
    from database import get_latest_data_date, has_any_data
    from datetime import datetime, timedelta
    
    # Auto-select appropriate interval based on period for optimal granularity
    if period and interval is None:
        if period == '1d':
            interval = '2m'   # 2-minute intervals for 1 day (good balance: ~195 points)
        elif period == '5d':
            interval = '5m'   # 5-minute intervals for 5 days (manageable data points)
        elif period == '30d':
            interval = '30m'  # 30-minute intervals for 30 days
        elif period in ['6m', '1y']:
            interval = '1d'   # Daily intervals for longer periods
        else:
            interval = '1d'   # Default to daily
    
    # For intraday data, always fetch fresh (don't use incremental loading)
    if interval and interval != '1d':
        print(f"Fetching intraday data for {ticker} with {interval} interval for period {period}")
        try:
            data = yf.Ticker(ticker).history(period=period, interval=interval)
            if not data.empty:
                print(f"Successfully fetched {len(data)} intraday data points")
                print(f"Data range: {data.index[0]} to {data.index[-1]}")
                return data
            else:
                print(f"No intraday data returned for {ticker}")
        except Exception as e:
            print(f"Error fetching intraday data: {e}, falling back to daily data")
            # Fallback to daily data if intraday fails
            pass
    
    # For daily data or fallback, use the original incremental loading logic
    if not has_any_data(ticker):
        print(f"No existing data for {ticker}, fetching 1 year of historical data")
        return yf.Ticker(ticker).history(period=period or "1y", interval=interval or "1d")
    else:
        latest_date = get_latest_data_date(ticker)
        if latest_date:
            # Parse the date string to datetime
            try:
                if isinstance(latest_date, str):
                    latest_dt = datetime.strptime(latest_date, '%Y-%m-%d')
                else:
                    latest_dt = latest_date
                
                # Fetch data from the day after our latest date
                start_date = latest_dt + timedelta(days=1)
                
                # Only fetch if we need new data (not on same day)
                if start_date.date() <= datetime.now().date():
                    print(f"Fetching incremental data for {ticker} from {start_date.strftime('%Y-%m-%d')}")
                    new_data = yf.Ticker(ticker).history(start=start_date.strftime('%Y-%m-%d'), interval=interval or "1d")
                    
                    if not new_data.empty:
                        # Combine with existing data from database
                        existing_data = get_historical_price_data(ticker)
                        if existing_data:
                            # Convert existing data to DataFrame
                            existing_df = pd.DataFrame(existing_data)
                            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
                            existing_df.set_index('Date', inplace=True)
                            existing_df.rename(columns={
                                'Open': 'Open', 'High': 'High', 'Low': 'Low', 
                                'Close': 'Close', 'Volume': 'Volume'
                            }, inplace=True)
                            
                            # Combine old and new data
                            combined_data = pd.concat([existing_df, new_data])
                            combined_data = combined_data.sort_index()
                            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                            
                            print(f"Combined {len(existing_df)} existing + {len(new_data)} new records = {len(combined_data)} total")
                            return combined_data
                    else:
                        print(f"No new data available for {ticker}")
                        # Return existing data as DataFrame
                        existing_data = get_historical_price_data(ticker)
                        if existing_data:
                            existing_df = pd.DataFrame(existing_data)
                            existing_df['Date'] = pd.to_datetime(existing_df['Date'])
                            existing_df.set_index('Date', inplace=True)
                            existing_df.rename(columns={
                                'Open': 'Open', 'High': 'High', 'Low': 'Low', 
                                'Close': 'Close', 'Volume': 'Volume'
                            }, inplace=True)
                            return existing_df
                else:
                    print(f"Data for {ticker} is already up to date")
                    # Return existing data as DataFrame
                    existing_data = get_historical_price_data(ticker)
                    if existing_data:
                        existing_df = pd.DataFrame(existing_data)
                        existing_df['Date'] = pd.to_datetime(existing_df['Date'])
                        existing_df.set_index('Date', inplace=True)
                        existing_df.rename(columns={
                            'Open': 'Open', 'High': 'High', 'Low': 'Low', 
                            'Close': 'Close', 'Volume': 'Volume'
                        }, inplace=True)
                        return existing_df
            except Exception as e:
                print(f"Error processing incremental data for {ticker}: {e}")
                print("Falling back to full 1-year fetch")
                return yf.Ticker(ticker).history(period=period or "1y", interval=interval or "1d")
        
        # Fallback to full fetch if something went wrong
        print(f"Fallback: fetching 1 year of data for {ticker}")
        return yf.Ticker(ticker).history(period=period or "1y", interval=interval or "1d")

def get_historical_price_data(ticker):
    """Get historical price data from database as list of dicts"""
    from database import get_db
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT date as "Date", open_price as "Open", high_price as "High", 
                   low_price as "Low", close_price as "Close", volume as "Volume"
            FROM price_history 
            WHERE ticker = ? 
            ORDER BY date
        ''', (ticker,))
        return [dict(row) for row in cursor.fetchall()]

# Real-time data configuration
# Note: yfinance provides delayed data (15-20 minutes behind)
# For real-time data, consider these alternatives:
# 1. Alpha Vantage API (free tier with limitations)
# 2. IEX Cloud API (good free tier)
# 3. Polygon.io (real-time with paid plans)
# 4. Interactive Brokers API (for trading accounts)
# 5. TD Ameritrade API (for account holders)

REAL_TIME_DATA_ENABLED = False  # Set to True when using real-time APIs
REAL_TIME_UPDATE_INTERVAL = 30  # seconds between updates

def get_real_time_price(ticker):
    """
    Get real-time price data. Currently uses yfinance (delayed).
    Replace this function with real-time API calls when available.
    """
    if REAL_TIME_DATA_ENABLED:
        # TODO: Implement real-time API calls here
        # Example for Alpha Vantage:
        # api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        # url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}'
        # response = requests.get(url)
        # data = response.json()
        # return float(data['Global Quote']['05. price'])
        pass
    
    # Fallback to yfinance (delayed data)
    return get_current_price(ticker)

# --- Additional Helper Functions for CrewAI Integration ---

def get_financial_news(query: str, max_articles: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch financial news for a given query/ticker
    """
    try:
        import requests
        from datetime import datetime, timedelta
        
        # For demonstration, we'll use a simple news API approach
        # In production, you might want to use NewsAPI, Alpha Vantage News, or similar
        
        # Try to get news from yfinance if query looks like a ticker
        if len(query) <= 5 and query.isalpha():
            try:
                ticker_obj = yf.Ticker(query.upper())
                news = ticker_obj.news
                
                if news:
                    formatted_news = []
                    for item in news[:max_articles]:
                        formatted_news.append({
                            'title': item.get('title', 'No title'),
                            'link': item.get('link', ''),
                            'publisher': item.get('publisher', 'Unknown'),
                            'providerPublishTime': item.get('providerPublishTime', ''),
                            'type': item.get('type', 'article'),
                            'thumbnail': item.get('thumbnail', {}).get('resolutions', [{}])[0].get('url', '') if item.get('thumbnail') else '',
                            'relatedTickers': item.get('relatedTickers', [])
                        })
                    return formatted_news
            except Exception as e:
                print(f"Error fetching news from yfinance: {e}")
        
        # Fallback: return sample news structure
        return [
            {
                'title': f'Market Analysis: {query}',
                'link': 'https://example.com',
                'publisher': 'Financial Times',
                'providerPublishTime': datetime.now().isoformat(),
                'type': 'article',
                'thumbnail': '',
                'relatedTickers': [query.upper()] if query.isalpha() else []
            }
        ]
        
    except Exception as e:
        print(f"Error in get_financial_news: {e}")
        return []


def get_stock_info(ticker: str) -> Dict[str, Any]:
    """
    Get basic stock information for a ticker
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price and basic metrics
        current_price = get_current_price(ticker)
        
        return {
            'ticker': ticker.upper(),
            'company_name': info.get('longName', ticker),
            'current_price': current_price,
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('forwardPE'),
            'dividend_yield': info.get('dividendYield'),
            'beta': info.get('beta'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
            'volume': info.get('volume'),
            'avg_volume': info.get('averageVolume'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'exchange': info.get('exchange'),
            'currency': info.get('currency', 'USD'),
            'price_change': None,  # Could calculate from historical data
            'price_change_percent': None,  # Could calculate from historical data
        }
        
    except Exception as e:
        print(f"Error getting stock info for {ticker}: {e}")
        return {
            'ticker': ticker.upper(),
            'company_name': ticker,
            'current_price': 0.0,
            'error': str(e)
        }