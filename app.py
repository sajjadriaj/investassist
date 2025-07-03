from flask import Flask, render_template, jsonify, request, redirect, url_for
import os
import requests
import pandas as pd
from tools.financial_tools import (
    analyze_market_sentiment,
    analyze_options_data,
    analyze_portfolio_risk,
    perform_backtesting,
)
from database import init_database, cleanup_old_data

# Load environment variables from keys.txt if it exists
def load_env_from_file():
    try:
        with open('/home/sajjad/workspace/keys.txt', 'r') as f:
            for line in f:
                if line.strip() and not line.strip().startswith('#') and '=' in line:
                    key, value = line.strip().split('export ', 1)[1].split('=', 1)
                    os.environ[key] = value.strip('"')
    except FileNotFoundError:
        pass

load_env_from_file()

app = Flask(__name__)

# Initialize database
init_database()

# Clean up old data (older than 7 days) on startup
cleanup_old_data()

# In-memory data store (for simplicity)
tickers = ['AAPL', 'GOOGL', 'MSFT']
selected_ticker = tickers[0] if tickers else None

# --- Routes ---

@app.route('/')
def index():
    """Redirects to the default tab."""
    return redirect(url_for('market_sentiment'))

@app.route('/market_sentiment')
def market_sentiment():
    try:
        raw_data = analyze_market_sentiment(selected_ticker)
        
        # Prepare data for charts
        sentiment_over_time = []
        if raw_data.technical_indicators:
            for item in raw_data.technical_indicators:
                # Handle both dict and object formats
                date_val = item.get('Date') if isinstance(item, dict) else getattr(item, 'Date', None)
                rsi_val = item.get('RSI') if isinstance(item, dict) else getattr(item, 'RSI', None)
                if date_val and rsi_val is not None:
                    sentiment_over_time.append({'date': date_val, 'sentiment': rsi_val})

        price_vs_sentiment = []
        if raw_data.price_history and raw_data.technical_indicators:
            # Create sentiment map handling both dict and object formats
            sentiment_map = {}
            for item in raw_data.technical_indicators:
                date_val = item.get('Date') if isinstance(item, dict) else getattr(item, 'Date', None)
                rsi_val = item.get('RSI') if isinstance(item, dict) else getattr(item, 'RSI', None)
                if date_val and rsi_val is not None:
                    sentiment_map[date_val] = rsi_val
                    
            for price_item in raw_data.price_history:
                # Handle both dict and object formats for price data
                price_date = price_item.get('Date') if isinstance(price_item, dict) else getattr(price_item, 'Date', None)
                price_close = price_item.get('Close') if isinstance(price_item, dict) else getattr(price_item, 'Close', None)
                if price_date in sentiment_map:
                    price_vs_sentiment.append({
                        'date': price_date,
                        'price': price_close,
                        'sentiment': sentiment_map[price_date]
                    })

        # Convert Pydantic model to dict and add chart data
        data = raw_data.model_dump()
        data['sentiment_over_time'] = sentiment_over_time
        data['price_vs_sentiment'] = price_vs_sentiment
        
        return render_template('market_sentiment.html', tickers=tickers, selected_ticker=selected_ticker, data=data)
    except Exception as e:
        print(f"Error in market_sentiment: {e}")
        # Return minimal data structure on error
        data = {
            'ticker': selected_ticker,
            'current_price': 0,
            'pe_ratio': None,
            'market_cap': None,
            'dividend_yield': None,
            'technical_summary': [],
            'fundamental_analysis': {'key_financials': []},
            'price_history': [],
            'technical_indicators': [],
            'sentiment_over_time': [],
            'price_vs_sentiment': []
        }
        return render_template('market_sentiment.html', tickers=tickers, selected_ticker=selected_ticker, data=data)

@app.route('/options_analysis')
def options_analysis():
    try:
        raw_data = analyze_options_data(selected_ticker)
        
        # If no data found, try to get data for a popular ticker to demonstrate functionality
        if not raw_data.calls and not raw_data.puts and selected_ticker not in ['AAPL', 'MSFT', 'GOOGL']:
            print(f"No options data for {selected_ticker}, trying AAPL for demo...")
            demo_data = analyze_options_data('AAPL')
            if demo_data.calls or demo_data.puts:
                # Add a note that this is demo data
                from tools.financial_tools import OptionAnalysisPoint
                demo_data.analysis_points.insert(0, OptionAnalysisPoint(
                    title="Demo Data", 
                    description=f"Showing AAPL options data as {selected_ticker} has no options available", 
                    info="This is demonstration data to show how the options analysis works"
                ))
                raw_data = demo_data
        
        # Convert Pydantic model to dict for template rendering
        data = raw_data.model_dump()
        return render_template('options_analysis.html', tickers=tickers, selected_ticker=selected_ticker, data=data)
    except Exception as e:
        print(f"Error in options_analysis: {e}")
        import traceback
        traceback.print_exc()
        # Return empty data structure on error
        data = {
            'ticker': selected_ticker,
            'analysis_points': [{'title': 'Error', 'description': f'Could not retrieve options data: {str(e)}', 'info': 'Please try again later or check if the ticker has options available.'}],
            'calls': [],
            'puts': [],
            'put_call_ratio': None,
            'max_pain': None
        }
        return render_template('options_analysis.html', tickers=tickers, selected_ticker=selected_ticker, data=data)

@app.route('/portfolio_risk')
def portfolio_risk():
    try:
        raw_data = analyze_portfolio_risk(tickers)
        # Convert Pydantic model to dict for template rendering
        data = raw_data.model_dump()
        return render_template('portfolio_risk.html', tickers=tickers, selected_ticker=selected_ticker, data=data)
    except Exception as e:
        print(f"Error in portfolio_risk: {e}")
        import traceback
        traceback.print_exc()
        # Return empty data structure on error
        data = {
            'tickers': tickers,
            'summary': f"Error analyzing portfolio risk: {str(e)}",
            'correlation_matrix': {},
            'betas': {},
            'portfolio_var': None,
            'conditional_var': None,
            'sharpe_ratio': None,
            'max_drawdown': None,
            'portfolio_volatility': None,
            'diversification_ratio': None,
            'risk_metrics': {},
            'asset_details': [],
            'portfolio_performance': {},
            'chart_data': {}
        }
        return render_template('portfolio_risk.html', tickers=tickers, selected_ticker=selected_ticker, data=data)

@app.route('/backtesting')
def backtesting():
    try:
        # Get strategy from query parameters, default to SMA crossover
        strategy = request.args.get('strategy', 'sma_crossover')
        
        # Validate strategy
        valid_strategies = ['sma_crossover', 'rsi_mean_reversion', 'bollinger_bands', 'macd']
        if strategy not in valid_strategies:
            strategy = 'sma_crossover'
        
        raw_data = perform_backtesting(selected_ticker, strategy=strategy)
        
        # Convert Pydantic model to dict for template rendering
        data = raw_data.model_dump()
        
        return render_template('backtesting.html', 
                             tickers=tickers, 
                             selected_ticker=selected_ticker, 
                             data=data,
                             available_strategies=valid_strategies,
                             current_strategy=strategy)
    except Exception as e:
        print(f"Error in backtesting: {e}")
        import traceback
        traceback.print_exc()
        
        # Return minimal data structure on error
        data = {
            'ticker': selected_ticker,
            'strategy': {'name': 'Error', 'description': f'Error: {str(e)}', 'parameters': {}},
            'summary': f'Backtesting failed: {str(e)}',
            'equity_curve': [],
            'performance_metrics': {
                'total_return': 0, 'annualized_return': 0, 'max_drawdown': 0,
                'win_rate': 0, 'sharpe_ratio': 0, 'sortino_ratio': 0,
                'calmar_ratio': 0, 'total_trades': 0, 'avg_trade_return': 0,
                'profit_factor': 0, 'volatility': 0
            },
            'trades': [],
            'monthly_returns': {},
            'risk_metrics': {},
            'chart_data': {}
        }
        
        return render_template('backtesting.html', 
                             tickers=tickers, 
                             selected_ticker=selected_ticker, 
                             data=data,
                             available_strategies=['sma_crossover'],
                             current_strategy='sma_crossover')

@app.route('/news')
def news():
    return render_template('news.html', tickers=tickers, selected_ticker=selected_ticker)

@app.route('/chat')
def chat():
    return render_template('chat.html', tickers=tickers, selected_ticker=selected_ticker)


@app.route('/add_ticker', methods=['POST'])
def add_ticker():
    """Adds a new ticker to the list."""
    data = request.get_json()
    ticker = data.get('ticker', '').upper()
    if ticker and ticker not in tickers:
        tickers.append(ticker)
        return jsonify(success=True, ticker=ticker)
    return jsonify(success=False, message='Invalid or duplicate ticker.')

@app.route('/select_ticker/<ticker>')
def select_ticker_route(ticker):
    """Selects a ticker to view its data."""
    global selected_ticker
    if ticker in tickers:
        selected_ticker = ticker
        return jsonify(success=True)
    return jsonify(success=False, message='Ticker not found.')

@app.route('/api/news/<ticker>')
def get_news(ticker):
    """Fetches news for a given ticker from NewsAPI with enhanced categorization."""
    api_key = os.environ.get('NEWS_API_KEY')
    
    if not api_key or api_key == 'YOUR_NEWS_API_KEY':
        # Return mock data for demonstration if no API key is set
        return jsonify([
            {
                "title": f"{ticker} Reports Strong Quarterly Results",
                "description": f"{ticker} announced better-than-expected quarterly earnings, driving investor confidence.",
                "url": "https://example.com/news1",
                "source": {"name": "Financial Times"},
                "publishedAt": "2024-12-20T10:00:00Z",
                "urlToImage": "https://via.placeholder.com/400x200/10b981/ffffff?text=Positive+News",
                "category": "earnings",
                "sentiment": "positive"
            },
            {
                "title": f"Analysts Upgrade {ticker} Rating Following Innovation Announcement",
                "description": f"Leading investment firms have upgraded their rating for {ticker} following recent product innovations.",
                "url": "https://example.com/news2",
                "source": {"name": "Bloomberg"},
                "publishedAt": "2024-12-20T08:30:00Z",
                "urlToImage": "https://via.placeholder.com/400x200/3b82f6/ffffff?text=Analyst+Report",
                "category": "analyst",
                "sentiment": "positive"
            },
            {
                "title": f"{ticker} Faces Regulatory Challenges in New Market",
                "description": f"Regulatory authorities are reviewing {ticker}'s expansion plans amid compliance concerns.",
                "url": "https://example.com/news3",
                "source": {"name": "Reuters"},
                "publishedAt": "2024-12-20T06:15:00Z",
                "urlToImage": "https://via.placeholder.com/400x200/ef4444/ffffff?text=Regulatory+News",
                "category": "regulatory",
                "sentiment": "negative"
            }
        ])
    
    # Build search query with company name variations
    search_queries = [ticker]
    if ticker == 'AAPL':
        search_queries.extend(['Apple Inc', 'Apple'])
    elif ticker == 'GOOGL':
        search_queries.extend(['Google', 'Alphabet'])
    elif ticker == 'MSFT':
        search_queries.extend(['Microsoft'])
    
    all_articles = []
    
    for query in search_queries[:2]:  # Limit to 2 queries to avoid rate limits
        url = f'https://newsapi.org/v2/everything?q={query}&pageSize=20&sortBy=publishedAt&language=en&apiKey={api_key}'
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            news_data = response.json()
            articles = news_data.get('articles', [])
            
            # Enhance articles with categorization and sentiment analysis
            for article in articles:
                article['category'] = categorize_article(article.get('title', ''), article.get('description', ''))
                article['sentiment'] = analyze_sentiment(article.get('title', ''), article.get('description', ''))
                article['isBreaking'] = is_breaking_news(article.get('publishedAt', ''))
            
            all_articles.extend(articles)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news for query '{query}': {e}")
            continue
    
    # Remove duplicates and sort by publication date
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        title = article.get('title', '')
        if title not in seen_titles:
            seen_titles.add(title)
            unique_articles.append(article)
    
    # Sort by publication date (most recent first)
    unique_articles.sort(key=lambda x: x.get('publishedAt', ''), reverse=True)
    
    # Limit to top 15 articles
    return jsonify(unique_articles[:15])

def categorize_article(title, description):
    """Categorize news articles based on content."""
    text = (title + ' ' + description).lower()
    
    if any(keyword in text for keyword in ['earnings', 'revenue', 'profit', 'quarter', 'financial results']):
        return 'earnings'
    elif any(keyword in text for keyword in ['analyst', 'rating', 'upgrade', 'downgrade', 'price target']):
        return 'analyst'
    elif any(keyword in text for keyword in ['merger', 'acquisition', 'deal', 'buyout', 'takeover']):
        return 'regulatory'
    elif any(keyword in text for keyword in ['regulation', 'regulatory', 'compliance', 'sec', 'government']):
        return 'regulatory'
    else:
        return 'general'

def analyze_sentiment(title, description):
    """Simple sentiment analysis based on keywords."""
    text = (title + ' ' + description).lower()
    
    positive_words = ['profit', 'growth', 'increase', 'strong', 'beat', 'exceed', 'success', 'upgrade', 'buy', 'bullish']
    negative_words = ['loss', 'decline', 'decrease', 'weak', 'miss', 'fail', 'downgrade', 'sell', 'bearish', 'concern']
    
    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)
    
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'

def is_breaking_news(published_at):
    """Determine if news is recent enough to be considered breaking."""
    try:
        from datetime import datetime, timedelta
        article_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        current_time = datetime.now(article_time.tzinfo)
        return (current_time - article_time) < timedelta(hours=6)
    except:
        return False

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """Handles chat messages using CrewAI agentic architecture with conversation history."""
    data = request.get_json()
    query = data.get('query', '')
    conversation_history = data.get('conversation_history', [])
    
    if not query:
        return jsonify({"response": "Please provide a query."})

    try:
        # Import CrewAI crew
        from crew import get_financial_crew
        
        # Get the financial crew instance
        crew = get_financial_crew()
        
        if crew is None:
            raise Exception("Failed to initialize CrewAI agents")
        
        # Prepare context for the crew including conversation history
        context = {
            'selected_ticker': selected_ticker,
            'available_tickers': tickers,
            'portfolio_tickers': tickers,  # Use available tickers as default portfolio
            'conversation_history': conversation_history,  # Add conversation history
        }
        
        # Route the query through the agentic system
        response = crew.route_query(query, context)
        
        if response:
            return jsonify({"response": response})
        else:
            return jsonify({"response": "I'm sorry, I couldn't generate a response at the moment. Please try again."})
            
    except Exception as e:
        print(f"Error in agentic chat API: {e}")
        
        # Fallback to simple Gemini response if CrewAI fails
        try:
            # Get Gemini API key from environment
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key:
                return jsonify({"response": "AI service not available. Please configure GEMINI_API_KEY."})

            # Import Google Generative AI
            import google.generativeai as genai
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            # Create the model
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Build conversation history context
            history_context = ""
            if conversation_history:
                history_context = "\n\nConversation History:\n"
                for i, msg in enumerate(conversation_history[-5:]):  # Include last 5 messages for context
                    role = "User" if msg.get('isUser') else "Assistant"
                    history_context += f"{role}: {msg.get('message', '')}\n"
                history_context += "\n"
            
            # Create a financial context prompt with conversation history
            context_prompt = f"""You are a knowledgeable financial advisor and investment analyst assistant for InvestAssist. 
            
            Current context:
            - Available tickers: {', '.join(tickers)}
            - Selected ticker: {selected_ticker}
            - Platform features: Market sentiment analysis, options analysis, portfolio risk assessment, backtesting, and financial news
            {history_context}
            Please provide helpful, accurate, and professional financial advice and insights. Be concise but informative.
            Consider the conversation history for context when answering.
            
            User question: {query}"""
            
            # Generate response
            response = model.generate_content(context_prompt)
            
            if response and response.text:
                return jsonify({"response": response.text})
            else:
                return jsonify({"response": "I'm sorry, I couldn't generate a response at the moment. Please try again."})
                
        except Exception as fallback_error:
            print(f"Fallback error: {fallback_error}")
            return jsonify({"response": f"I'm experiencing technical difficulties. Please try again later."})

@app.route('/api/market_sentiment/<ticker>/<period>')
def get_market_sentiment_period(ticker, period):
    """Get market sentiment data for a specific time period with appropriate granularity"""
    try:
        # Validate period
        valid_periods = ['1d', '5d', '30d', '6m', '1y']
        if period not in valid_periods:
            period = '1y'
        
        # Import here to avoid circular imports
        from tools.financial_tools import get_price_history_smart, get_technical_indicators, get_current_price
        import yfinance as yf
        
        # Get data with appropriate interval for the period
        hist = get_price_history_smart(ticker, period=period)
        current_price = get_current_price(ticker)
        
        if hist.empty:
            return jsonify({
                'success': False,
                'error': 'No data available for this ticker and period'
            }), 404
        
        # Get technical indicators
        tech_indicators = get_technical_indicators(hist.copy())
        
        print(f"Technical indicators shape: {tech_indicators.shape}")
        print(f"Technical indicators columns: {tech_indicators.columns.tolist()}")
        print(f"Technical indicators index: {tech_indicators.index}")
        
        # Convert price data
        price_df = hist[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
        
        # Ensure the first column (index) is named 'Date'
        if price_df.columns[0] != 'Date':
            price_df.rename(columns={price_df.columns[0]: 'Date'}, inplace=True)
        
        # Format Date column (handle both daily and intraday timestamps)
        if hasattr(price_df['Date'].iloc[0], 'strftime'):
            if period in ['1d', '5d']:
                # For intraday data, include time
                price_df['Date'] = price_df['Date'].dt.tz_localize(None).dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                # For daily data, just date
                price_df['Date'] = price_df['Date'].dt.tz_localize(None).dt.strftime('%Y-%m-%d')
        
        price_data = price_df.to_dict(orient='records')
        
        # Convert technical indicators data
        tech_df = tech_indicators.reset_index()
        
        print(f"Tech df after reset_index columns: {tech_df.columns.tolist()}")
        print(f"Tech df shape: {tech_df.shape}")
        
        # Ensure the first column (index) is named 'Date' for technical indicators too
        if tech_df.columns[0] != 'Date':
            print(f"Renaming column {tech_df.columns[0]} to Date")
            tech_df.rename(columns={tech_df.columns[0]: 'Date'}, inplace=True)
        
        # Format Date column for technical indicators
        if len(tech_df) > 0 and hasattr(tech_df['Date'].iloc[0], 'strftime'):
            if period in ['1d', '5d']:
                # For intraday data, include time
                tech_df['Date'] = tech_df['Date'].dt.tz_localize(None).dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                # For daily data, just date
                tech_df['Date'] = tech_df['Date'].dt.tz_localize(None).dt.strftime('%Y-%m-%d')
        
        # Get available columns from TechnicalIndicatorsData model
        from tools.financial_tools import TechnicalIndicatorsData
        tech_data_cols = list(TechnicalIndicatorsData.__annotations__.keys())
        available_cols = [col for col in tech_data_cols if col in tech_df.columns]
        
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
        
        return jsonify({
            'success': True,
            'data': {
                'price_history': price_data,
                'technical_indicators': tech_data,
                'current_price': current_price,
                'data_points': len(price_data),
                'interval_used': '2m' if period == '1d' else '5m' if period == '5d' else '30m' if period == '30d' else '1d'
            },
            'period': period
        })
        
    except Exception as e:
        print(f"Error getting market sentiment for period {period}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream_api():
    """Handles streaming chat messages with status updates using Server-Sent Events."""
    import time
    import json
    from flask import Response
    
    data = request.get_json()
    query = data.get('query', '')
    conversation_history = data.get('conversation_history', [])
    
    if not query:
        return jsonify({"error": "Please provide a query."})

    def generate_response():
        """Generator function for streaming response"""
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Analyzing your query...'})}\n\n"
            time.sleep(0.5)
            
            # Import CrewAI crew
            from crew import get_financial_crew
            
            # Get the financial crew instance
            crew = get_financial_crew()
            
            if crew is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Failed to initialize AI agents'})}\n\n"
                return
            
            # Prepare context for the crew including conversation history
            context = {
                'selected_ticker': selected_ticker,
                'available_tickers': tickers,
                'portfolio_tickers': tickers,
                'conversation_history': conversation_history,
            }
            
            # Determine routing and send appropriate status
            routing_decision = crew._determine_routing(query.lower())
            agent_name = routing_decision['agent']
            
            agent_messages = {
                'market': 'Market analyst is analyzing technical indicators and sentiment...',
                'options': 'Options strategist is analyzing options data and Greeks...',
                'news': 'News analyst is gathering and analyzing recent financial news...',
                'portfolio': 'Risk analyst is calculating portfolio metrics and correlations...'
            }
            
            if agent_name in agent_messages:
                yield f"data: {json.dumps({'type': 'status', 'message': agent_messages[agent_name]})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Financial advisor is processing your query...'})}\n\n"
            
            time.sleep(1)
            yield f"data: {json.dumps({'type': 'status', 'message': 'Gathering financial data and running analysis...'})}\n\n"
            time.sleep(1)
            yield f"data: {json.dumps({'type': 'status', 'message': 'Synthesizing insights with AI...'})}\n\n"
            
            # Get the actual response (no streaming callbacks needed)
            response = crew.route_query(query, context)
            
            if response:
                # Stream the response character by character
                for chunk in response:
                    yield f"data: {json.dumps({'type': 'delta', 'message': chunk})}\n\n"
                # Send a final 'response' type with an empty message to signal completion
                yield f"data: {json.dumps({'type': 'response', 'message': ''})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': 'I could not generate a response at this time.'})}\n\n"
                
        except Exception as e:
            print(f"Error in streaming chat API: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': 'I encountered an error processing your request.'})}\n\n"
        
        # Send completion signal
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"

    return Response(generate_response(), mimetype='text/plain')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)