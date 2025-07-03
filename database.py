import sqlite3
import json
import os
from datetime import datetime, timedelta
from contextlib import contextmanager

DATABASE_FILE = 'investassist.db'

def get_db_connection():
    """Get database connection with row factory for dict-like access"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """Initialize the database with required tables"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Options data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS options_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                expiration_date TEXT NOT NULL,
                option_type TEXT NOT NULL,  -- 'call' or 'put'
                strike REAL NOT NULL,
                last_price REAL,
                bid REAL,
                ask REAL,
                volume INTEGER,
                open_interest INTEGER,
                implied_volatility REAL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                rho REAL,
                intrinsic_value REAL,
                time_value REAL,
                moneyness TEXT,  -- 'ITM', 'OTM', 'ATM'
                risk_score REAL,
                profit_potential REAL,
                probability_of_profit REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, expiration_date, option_type, strike)
            )
        ''')
        
        # Market sentiment data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                current_price REAL,
                market_cap REAL,
                pe_ratio REAL,
                dividend_yield REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker)
            )
        ''')
        
        # Price history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
        ''')
        
        # Technical indicators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                sma20 REAL,
                sma50 REAL,
                macd REAL,
                macd_signal REAL,
                macd_hist REAL,
                rsi REAL,
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                obv REAL,
                support REAL,
                resistance REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
        ''')
        
        # Technical summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                indicator TEXT NOT NULL,
                signal TEXT NOT NULL,
                description TEXT,
                info TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, indicator)
            )
        ''')
        
        # Fundamental analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fundamental_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                metric_label TEXT NOT NULL,
                metric_value TEXT NOT NULL,
                metric_info TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, metric_label)
            )
        ''')
        
        conn.commit()
        print("Database initialized successfully!")

def is_data_fresh(ticker, table_name, hours=1):
    """Check if data for a ticker is fresh (within specified hours)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        cursor.execute(f'''
            SELECT COUNT(*) as count FROM {table_name} 
            WHERE ticker = ? AND last_updated > ?
        ''', (ticker, cutoff_time))
        
        result = cursor.fetchone()
        return result['count'] > 0

def get_options_data(ticker):
    """Get options data from database"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM options_data 
            WHERE ticker = ? 
            ORDER BY expiration_date, option_type, strike
        ''', (ticker,))
        return [dict(row) for row in cursor.fetchall()]

def save_options_data(ticker, options_data):
    """Save options data to database"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Clear existing data for this ticker
        cursor.execute('DELETE FROM options_data WHERE ticker = ?', (ticker,))
        
        # Insert new data
        for option in options_data:
            cursor.execute('''
                INSERT OR REPLACE INTO options_data (
                    ticker, expiration_date, option_type, strike, last_price, bid, ask,
                    volume, open_interest, implied_volatility, delta, gamma, theta, vega, rho,
                    intrinsic_value, time_value, moneyness, risk_score, profit_potential, probability_of_profit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker, option['expiration'], option['type'], option['strike'],
                option.get('lastPrice'), option.get('bid'), option.get('ask'),
                option.get('volume'), option.get('openInterest'), option.get('impliedVolatility'),
                option.get('delta'), option.get('gamma'), option.get('theta'), option.get('vega'), option.get('rho'),
                option.get('intrinsicValue'), option.get('timeValue'), option.get('moneyness'),
                option.get('riskScore'), option.get('profitPotential'), option.get('probabilityOfProfit')
            ))
        
        conn.commit()

def get_market_sentiment_data(ticker):
    """Get market sentiment data from database"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get basic market data
        cursor.execute('SELECT * FROM market_sentiment WHERE ticker = ?', (ticker,))
        market_data = cursor.fetchone()
        
        if not market_data:
            return None
            
        # Get price history
        cursor.execute('''
            SELECT date as "Date", open_price as "Open", high_price as "High", 
                   low_price as "Low", close_price as "Close", volume as "Volume"
            FROM price_history 
            WHERE ticker = ? 
            ORDER BY date
        ''', (ticker,))
        price_history = [dict(row) for row in cursor.fetchall()]
        
        # Get technical indicators
        cursor.execute('''
            SELECT date as "Date", sma20 as "SMA20", sma50 as "SMA50", macd as "MACD", 
                   macd_signal as "MACD_signal", macd_hist as "MACD_hist",
                   rsi as "RSI", bb_upper as "BB_upper", bb_middle as "BB_middle", 
                   bb_lower as "BB_lower", obv as "OBV", support as "Support", 
                   resistance as "Resistance"
            FROM technical_indicators 
            WHERE ticker = ? 
            ORDER BY date
        ''', (ticker,))
        technical_indicators = [dict(row) for row in cursor.fetchall()]
        
        # Get technical summary
        cursor.execute('''
            SELECT indicator, signal, description, info 
            FROM technical_summary 
            WHERE ticker = ?
        ''', (ticker,))
        technical_summary = [dict(row) for row in cursor.fetchall()]
        
        # Get fundamental analysis
        cursor.execute('''
            SELECT metric_label as "label", metric_value as "value", metric_info as "info"
            FROM fundamental_analysis 
            WHERE ticker = ?
        ''', (ticker,))
        key_financials = [dict(row) for row in cursor.fetchall()]
        
        return {
            'current_price': market_data['current_price'],
            'market_cap': market_data['market_cap'],
            'pe_ratio': market_data['pe_ratio'],
            'dividend_yield': market_data['dividend_yield'],
            'price_history': price_history,
            'technical_indicators': technical_indicators,
            'technical_summary': technical_summary,
            'fundamental_analysis': {
                'key_financials': key_financials
            }
        }

def save_market_sentiment_data(ticker, data):
    """Save market sentiment data to database with efficient incremental updates"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Save basic market data (always update these)
        cursor.execute('''
            INSERT OR REPLACE INTO market_sentiment (
                ticker, current_price, market_cap, pe_ratio, dividend_yield
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            ticker, data.get('current_price'), data.get('market_cap'),
            data.get('pe_ratio'), data.get('dividend_yield')
        ))
        
        # For price history and technical indicators, only update if we have new data
        if data.get('price_history'):
            # Get existing latest date
            cursor.execute('SELECT MAX(date) as latest FROM price_history WHERE ticker = ?', (ticker,))
            latest_existing = cursor.fetchone()
            latest_date = latest_existing['latest'] if latest_existing and latest_existing['latest'] else None
            
            # Only insert new records
            new_records = 0
            for price in data['price_history']:
                if not latest_date or price['Date'] > latest_date:
                    cursor.execute('''
                        INSERT OR REPLACE INTO price_history (
                            ticker, date, open_price, high_price, low_price, close_price, volume
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ticker, price['Date'], price['Open'], price['High'],
                        price['Low'], price['Close'], price['Volume']
                    ))
                    new_records += 1
            
            if new_records > 0:
                print(f"Added {new_records} new price records for {ticker}")
        
        # Save technical indicators (only new ones)
        if data.get('technical_indicators'):
            # Get existing latest date for technical indicators
            cursor.execute('SELECT MAX(date) as latest FROM technical_indicators WHERE ticker = ?', (ticker,))
            latest_existing = cursor.fetchone()
            latest_date = latest_existing['latest'] if latest_existing and latest_existing['latest'] else None
            
            new_tech_records = 0
            for tech in data['technical_indicators']:
                if not latest_date or tech.get('Date') > latest_date:
                    cursor.execute('''
                        INSERT OR REPLACE INTO technical_indicators (
                            ticker, date, sma20, sma50, macd, macd_signal, macd_hist,
                            rsi, bb_upper, bb_middle, bb_lower, obv, support, resistance
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        ticker, tech.get('Date'), tech.get('SMA20'), tech.get('SMA50'),
                        tech.get('MACD'), tech.get('MACD_signal'), tech.get('MACD_hist'),
                        tech.get('RSI'), tech.get('BB_upper'), tech.get('BB_middle'),
                        tech.get('BB_lower'), tech.get('OBV'), tech.get('Support'), tech.get('Resistance')
                    ))
                    new_tech_records += 1
            
            if new_tech_records > 0:
                print(f"Added {new_tech_records} new technical indicator records for {ticker}")
        
        # Always update technical summary (replace all)
        if data.get('technical_summary'):
            cursor.execute('DELETE FROM technical_summary WHERE ticker = ?', (ticker,))
            for summary in data['technical_summary']:
                cursor.execute('''
                    INSERT OR REPLACE INTO technical_summary (
                        ticker, indicator, signal, description, info
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    ticker, summary['indicator'], summary['signal'],
                    summary['description'], summary['info']
                ))
        
        # Always update fundamental analysis (replace all)
        if data.get('fundamental_analysis', {}).get('key_financials'):
            cursor.execute('DELETE FROM fundamental_analysis WHERE ticker = ?', (ticker,))
            for financial in data['fundamental_analysis']['key_financials']:
                cursor.execute('''
                    INSERT OR REPLACE INTO fundamental_analysis (
                        ticker, metric_label, metric_value, metric_info
                    ) VALUES (?, ?, ?, ?)
                ''', (
                    ticker, financial['label'], financial['value'], financial['info']
                ))
        
        conn.commit()
        
        # Save technical summary
        if data.get('technical_summary'):
            cursor.execute('DELETE FROM technical_summary WHERE ticker = ?', (ticker,))
            for summary in data['technical_summary']:
                cursor.execute('''
                    INSERT OR REPLACE INTO technical_summary (
                        ticker, indicator, signal, description, info
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    ticker, summary['indicator'], summary['signal'],
                    summary['description'], summary['info']
                ))
        
        # Save fundamental analysis
        if data.get('fundamental_analysis', {}).get('key_financials'):
            cursor.execute('DELETE FROM fundamental_analysis WHERE ticker = ?', (ticker,))
            for financial in data['fundamental_analysis']['key_financials']:
                cursor.execute('''
                    INSERT OR REPLACE INTO fundamental_analysis (
                        ticker, metric_label, metric_value, metric_info
                    ) VALUES (?, ?, ?, ?)
                ''', (
                    ticker, financial['label'], financial['value'], financial['info']
                ))
        
        conn.commit()

def cleanup_old_data(days=7):
    """Remove data older than specified days"""
    with get_db() as conn:
        cursor = conn.cursor()
        cutoff_date = datetime.now() - timedelta(days=days)
        
        tables = ['options_data', 'market_sentiment', 'price_history', 
                 'technical_indicators', 'technical_summary', 'fundamental_analysis']
        
        for table in tables:
            cursor.execute(f'DELETE FROM {table} WHERE last_updated < ?', (cutoff_date,))
        
        conn.commit()
        print(f"Cleaned up data older than {days} days")

def get_latest_data_date(ticker):
    """Get the latest date we have data for a ticker"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT MAX(date) as latest_date FROM price_history 
            WHERE ticker = ?
        ''', (ticker,))
        
        result = cursor.fetchone()
        return result['latest_date'] if result and result['latest_date'] else None

def has_any_data(ticker):
    """Check if we have any data for a ticker"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) as count FROM price_history 
            WHERE ticker = ?
        ''', (ticker,))
        
        result = cursor.fetchone()
        return result['count'] > 0

def save_incremental_price_data(ticker, new_price_data):
    """Save only new price data to database (for incremental updates)"""
    if not new_price_data:
        return
        
    with get_db() as conn:
        cursor = conn.cursor()
        
        for price in new_price_data:
            cursor.execute('''
                INSERT OR REPLACE INTO price_history (
                    ticker, date, open_price, high_price, low_price, close_price, volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker, price['Date'], price['Open'], price['High'],
                price['Low'], price['Close'], price['Volume']
            ))
        
        conn.commit()
        print(f"Saved {len(new_price_data)} new price records for {ticker}")

if __name__ == "__main__":
    init_database()
