"""
CrewAI-based agentic architecture for InvestAssist.
Defines 3 specialized agents (market, option, news) and a chat orchestrator.
"""

import os
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from tools.financial_tools import (
    analyze_market_sentiment,
    analyze_options_data,
    analyze_portfolio_risk,
    perform_backtesting,
    get_financial_news,
    get_stock_info
)
import json

# Import Google Generative AI
try:
    import google.generativeai as genai
    HAS_GOOGLE_AI = True
except ImportError:
    HAS_GOOGLE_AI = False

# Simple approach: Don't use LLM with CrewAI agents, let them use tools only
# We'll handle the LLM responses at the crew level instead

class FinancialToolWrapper:
    """Wrapper class to convert our existing tools into CrewAI compatible tools"""
    
    @staticmethod
    def create_market_analysis_tool():
        class MarketAnalysisTool(BaseTool):
            name: str = "market_sentiment_analyzer"
            description: str = "Analyzes market sentiment, technical indicators, and fundamentals for a given stock ticker"
            
            def _run(self, ticker: str) -> str:
                try:
                    result = analyze_market_sentiment(ticker)
                    # Convert to string for CrewAI
                    return f"Market analysis for {ticker}: {json.dumps(result.dict(), indent=2, default=str)}"
                except Exception as e:
                    return f"Error analyzing market sentiment for {ticker}: {str(e)}"
        
        return MarketAnalysisTool()
    
    @staticmethod
    def create_options_analysis_tool():
        class OptionsAnalysisTool(BaseTool):
            name: str = "options_analyzer"
            description: str = "Analyzes options data including put/call ratios, max pain, and Greeks for a given stock ticker"
            
            def _run(self, ticker: str) -> str:
                try:
                    result = analyze_options_data(ticker)
                    return f"Options analysis for {ticker}: {json.dumps(result.dict(), indent=2, default=str)}"
                except Exception as e:
                    return f"Error analyzing options for {ticker}: {str(e)}"
        
        return OptionsAnalysisTool()
    
    @staticmethod
    def create_portfolio_risk_tool():
        class PortfolioRiskTool(BaseTool):
            name: str = "portfolio_risk_analyzer"
            description: str = "Analyzes portfolio risk metrics including VaR, correlations, and risk contributions for multiple tickers"
            
            def _run(self, tickers: str) -> str:
                try:
                    # Parse tickers from string (comma-separated)
                    ticker_list = [t.strip() for t in tickers.split(',')]
                    result = analyze_portfolio_risk(ticker_list)
                    return f"Portfolio risk analysis: {json.dumps(result.dict(), indent=2, default=str)}"
                except Exception as e:
                    return f"Error analyzing portfolio risk: {str(e)}"
        
        return PortfolioRiskTool()
    
    @staticmethod
    def create_backtesting_tool():
        class BacktestingTool(BaseTool):
            name: str = "backtesting_analyzer"
            description: str = "Performs backtesting analysis with various trading strategies for a given stock ticker"
            
            def _run(self, ticker: str, strategy: str = "sma_crossover") -> str:
                try:
                    result = perform_backtesting(ticker, strategy)
                    return f"Backtesting analysis for {ticker} using {strategy}: {json.dumps(result.dict(), indent=2, default=str)}"
                except Exception as e:
                    return f"Error performing backtesting for {ticker}: {str(e)}"
        
        return BacktestingTool()
    
    @staticmethod
    def create_news_tool():
        class NewsTool(BaseTool):
            name: str = "financial_news_analyzer"
            description: str = "Fetches and analyzes recent financial news for a given stock ticker or general market news"
            
            def _run(self, query: str) -> str:
                try:
                    result = get_financial_news(query)
                    return f"Financial news for '{query}': {json.dumps(result, indent=2, default=str)}"
                except Exception as e:
                    return f"Error fetching news for '{query}': {str(e)}"
        
        return NewsTool()
    
    @staticmethod
    def create_stock_info_tool():
        class StockInfoTool(BaseTool):
            name: str = "stock_info_fetcher"
            description: str = "Fetches basic stock information including current price, market cap, and key metrics"
            
            def _run(self, ticker: str) -> str:
                try:
                    result = get_stock_info(ticker)
                    return f"Stock info for {ticker}: {json.dumps(result, indent=2, default=str)}"
                except Exception as e:
                    return f"Error fetching stock info for {ticker}: {str(e)}"
        
        return StockInfoTool()


class FinancialCrew:
    """Main CrewAI orchestrator for financial analysis"""
    
    def __init__(self):
        self.gemini_model = self._setup_gemini()
        self.tools = self._setup_tools()
        self.agents = self._setup_agents()
    
    def _setup_gemini(self):
        """Setup Gemini for direct use"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        if HAS_GOOGLE_AI:
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-2.0-flash')
        else:
            raise ImportError("google-generativeai package is required")
    
    def _setup_tools(self):
        """Setup all financial analysis tools"""
        return {
            'market_analysis': FinancialToolWrapper.create_market_analysis_tool(),
            'options_analysis': FinancialToolWrapper.create_options_analysis_tool(),
            'portfolio_risk': FinancialToolWrapper.create_portfolio_risk_tool(),
            'backtesting': FinancialToolWrapper.create_backtesting_tool(),
            'news': FinancialToolWrapper.create_news_tool(),
            'stock_info': FinancialToolWrapper.create_stock_info_tool(),
        }
    
    def _setup_agents(self):
        """Setup specialized financial agents without LLM (tools only)"""
        
        # Market Analysis Agent
        market_agent = Agent(
            role='Market Analyst',
            goal='Provide comprehensive market sentiment analysis and technical insights',
            backstory="""You are an experienced market analyst with deep expertise in technical analysis, 
                        fundamental analysis, and market sentiment. You specialize in interpreting market data, 
                        technical indicators, and providing actionable investment insights.""",
            tools=[self.tools['market_analysis'], self.tools['stock_info']],
            verbose=False,
            allow_delegation=False
        )
        
        # Options Analysis Agent
        options_agent = Agent(
            role='Options Strategist',
            goal='Analyze options data and provide options trading insights',
            backstory="""You are a specialized options trader and strategist with extensive knowledge of 
                        options pricing, Greeks, volatility analysis, and options strategies. You excel at 
                        interpreting put/call ratios, max pain analysis, and identifying optimal options plays.""",
            tools=[self.tools['options_analysis'], self.tools['stock_info']],
            verbose=False,
            allow_delegation=False
        )
        
        # News Analysis Agent
        news_agent = Agent(
            role='Financial News Analyst',
            goal='Analyze financial news and market sentiment from news sources',
            backstory="""You are a financial journalist and news analyst with expertise in interpreting 
                        market-moving news, earnings reports, and macroeconomic events. You excel at 
                        connecting news events to market implications and investment opportunities.""",
            tools=[self.tools['news'], self.tools['stock_info']],
            verbose=False,
            allow_delegation=False
        )
        
        # Portfolio Risk Agent
        portfolio_agent = Agent(
            role='Risk Management Specialist',
            goal='Analyze portfolio risk and provide risk management insights',
            backstory="""You are a quantitative risk analyst with expertise in portfolio theory, VaR modeling, 
                        correlation analysis, and risk metrics. You specialize in helping investors understand 
                        and manage portfolio risk through diversification and hedging strategies.""",
            tools=[self.tools['portfolio_risk'], self.tools['backtesting']],
            verbose=False,
            allow_delegation=False
        )
        
        return {
            'market': market_agent,
            'options': options_agent,
            'news': news_agent,
            'portfolio': portfolio_agent
        }
    
    def _build_conversation_context(self, context: Dict) -> str:
        """Build conversation history context string for prompts"""
        conversation_history = context.get('conversation_history', [])
        if not conversation_history:
            return ""
        
        history_context = "\n\nConversation History (for context):\n"
        # Include last 5 messages for context, but don't make it too long
        for msg in conversation_history[-5:]:
            role = "User" if msg.get('isUser') else "Assistant"
            message = msg.get('message', '')[:200]  # Truncate very long messages
            if len(msg.get('message', '')) > 200:
                message += "..."
            history_context += f"{role}: {message}\n"
        history_context += "\nCurrent question (respond to this):\n"
        return history_context

    def route_query(self, user_query: str, context: Dict[str, Any] = None) -> str:
        """
        Route user query to appropriate agent(s) and orchestrate the response
        """
        context = context or {}
        user_query_lower = user_query.lower()
        
        # Simple routing logic based on keywords
        routing_decision = self._determine_routing(user_query_lower)
        
        if routing_decision['agent'] == 'market':
            return self._handle_market_query(user_query, routing_decision, context)
        elif routing_decision['agent'] == 'options':
            return self._handle_options_query(user_query, routing_decision, context)
        elif routing_decision['agent'] == 'news':
            return self._handle_news_query(user_query, routing_decision, context)
        elif routing_decision['agent'] == 'portfolio':
            return self._handle_portfolio_query(user_query, routing_decision, context)
        else:
            return self._handle_general_query(user_query, context)
    
    def route_query_with_streaming(self, user_query: str, context: Dict[str, Any] = None, status_callback=None) -> str:
        """
        Route user query to appropriate agent(s) with streaming status updates
        """
        context = context or {}
        user_query_lower = user_query.lower()
        
        # Simple routing logic based on keywords
        routing_decision = self._determine_routing(user_query_lower)
        
        if routing_decision['agent'] == 'market':
            return self._handle_market_query_streaming(user_query, routing_decision, context, status_callback)
        elif routing_decision['agent'] == 'options':
            return self._handle_options_query_streaming(user_query, routing_decision, context, status_callback)
        elif routing_decision['agent'] == 'news':
            return self._handle_news_query_streaming(user_query, routing_decision, context, status_callback)
        elif routing_decision['agent'] == 'portfolio':
            return self._handle_portfolio_query_streaming(user_query, routing_decision, context, status_callback)
        else:
            return self._handle_general_query_streaming(user_query, context, status_callback)

    def _determine_routing(self, query_lower: str) -> Dict[str, Any]:
        """Determine which agent should handle the query"""
        
        # Options-related keywords
        options_keywords = ['option', 'call', 'put', 'strike', 'expiry', 'volatility', 'greek', 'theta', 'gamma', 'delta', 'vega']
        
        # Market analysis keywords
        market_keywords = ['technical', 'sentiment', 'rsi', 'macd', 'moving average', 'sma', 'ema', 'chart', 'price', 'trend']
        
        # News keywords
        news_keywords = ['news', 'earnings', 'announcement', 'report', 'event', 'headline']
        
        # Portfolio/risk keywords
        portfolio_keywords = ['portfolio', 'risk', 'var', 'correlation', 'diversification', 'beta', 'sharpe', 'backtest']
        
        # Count keyword matches
        options_score = sum(1 for keyword in options_keywords if keyword in query_lower)
        market_score = sum(1 for keyword in market_keywords if keyword in query_lower)
        news_score = sum(1 for keyword in news_keywords if keyword in query_lower)
        portfolio_score = sum(1 for keyword in portfolio_keywords if keyword in query_lower)
        
        # Determine primary agent
        scores = {
            'options': options_score,
            'market': market_score,
            'news': news_score,
            'portfolio': portfolio_score
        }
        
        primary_agent = max(scores, key=scores.get)
        
        # If no clear winner, default to market analysis
        if scores[primary_agent] == 0:
            primary_agent = 'market'
        
        return {
            'agent': primary_agent,
            'scores': scores,
            'multi_agent': sum(1 for score in scores.values() if score > 0) > 1
        }
    
    def _handle_market_query(self, query: str, routing: Dict, context: Dict) -> str:
        """Handle market analysis queries"""
        ticker = self._extract_ticker(query, context)
        
        try:
            # Get market analysis data using tools directly
            market_tool = self.tools['market_analysis']
            stock_tool = self.tools['stock_info']
            
            market_data = market_tool._run(ticker)
            stock_data = stock_tool._run(ticker)
            
            # Build conversation history context
            history_context = self._build_conversation_context(context)
            
            # Use Gemini to synthesize a response
            prompt = f"""As an experienced market analyst, analyze the following data and answer the user's question.
            {history_context}
            User question: {query}
            Ticker: {ticker}
            
            Market Analysis Data:
            {market_data}
            
            Stock Information:
            {stock_data}
            
            Please provide a comprehensive analysis including:
            1. Current technical sentiment and key indicators
            2. Price trends and momentum  
            3. Support/resistance levels if available
            4. Investment recommendations based on the analysis
            
            Consider the conversation history for context when answering.
            Keep your response professional, actionable, and focused on the user's specific question."""
            
            response = self.gemini_model.generate_content(prompt)
            return response.text if response and response.text else "I couldn't analyze the market data at this time."
            
        except Exception as e:
            return f"I encountered an error while analyzing the market data: {str(e)}"
    
    def _handle_options_query(self, query: str, routing: Dict, context: Dict) -> str:
        """Handle options analysis queries"""
        ticker = self._extract_ticker(query, context)
        
        try:
            # Get options analysis data using tools directly
            options_tool = self.tools['options_analysis']
            stock_tool = self.tools['stock_info']
            
            options_data = options_tool._run(ticker)
            stock_data = stock_tool._run(ticker)
            
            # Build conversation history context
            history_context = self._build_conversation_context(context)
            
            # Use Gemini to synthesize a response
            prompt = f"""As a specialized options strategist, analyze the following data and answer the user's question.
            {history_context}
            User question: {query}
            Ticker: {ticker}
            
            Options Analysis Data:
            {options_data}
            
            Stock Information:
            {stock_data}
            
            Please provide analysis including:
            1. Put/call ratio and sentiment
            2. Options volume and open interest trends
            3. Implied volatility analysis
            4. Potential options strategies
            
            Consider the conversation history for context when answering.
            Keep your response professional, actionable, and focused on options trading insights."""
            
            response = self.gemini_model.generate_content(prompt)
            return response.text if response and response.text else "I couldn't analyze the options data at this time."
            
        except Exception as e:
            return f"I encountered an error while analyzing options data: {str(e)}"
    
    def _handle_news_query(self, query: str, routing: Dict, context: Dict) -> str:
        """Handle news analysis queries"""
        search_term = self._extract_news_search_term(query, context)
        
        try:
            # Get news data using tools directly
            news_tool = self.tools['news']
            stock_tool = self.tools['stock_info']
            
            news_data = news_tool._run(search_term)
            stock_data = stock_tool._run(search_term) if len(search_term) <= 5 else ""
            
            # Build conversation history context
            history_context = self._build_conversation_context(context)
            
            # Use Gemini to synthesize a response
            prompt = f"""As a financial news analyst, analyze the following data and answer the user's question.
            {history_context}
            User question: {query}
            Search term: {search_term}
            
            News Data:
            {news_data}
            
            Stock Information:
            {stock_data}
            
            Please provide analysis including:
            1. Key news developments and their market impact
            2. Sentiment analysis from recent headlines
            3. Potential market implications
            4. Investment considerations based on news flow
            
            Consider the conversation history for context when answering.
            Keep your response professional, insightful, and focused on market implications."""
            
            response = self.gemini_model.generate_content(prompt)
            return response.text if response and response.text else "I couldn't analyze the news data at this time."
            
        except Exception as e:
            return f"I encountered an error while analyzing news: {str(e)}"
    
    def _handle_portfolio_query(self, query: str, routing: Dict, context: Dict) -> str:
        """Handle portfolio risk queries"""
        tickers = self._extract_tickers_for_portfolio(query, context)
        
        try:
            # Get portfolio risk data using tools directly
            portfolio_tool = self.tools['portfolio_risk']
            tickers_str = ','.join(tickers)
            
            portfolio_data = portfolio_tool._run(tickers_str)
            
            # Build conversation history context
            history_context = self._build_conversation_context(context)
            
            # Use Gemini to synthesize a response
            prompt = f"""As a quantitative risk analyst, analyze the following data and answer the user's question.
            {history_context}
            User question: {query}
            Portfolio tickers: {', '.join(tickers)}
            
            Portfolio Risk Analysis:
            {portfolio_data}
            
            Please provide analysis including:
            1. Portfolio risk metrics (VaR, correlations)
            2. Diversification analysis
            3. Risk contribution by asset
            4. Risk management recommendations
            
            Consider the conversation history for context when answering.
            Keep your response professional, quantitative, and focused on risk management insights."""
            
            response = self.gemini_model.generate_content(prompt)
            return response.text if response and response.text else "I couldn't analyze the portfolio risk at this time."
            
        except Exception as e:
            return f"I encountered an error while analyzing portfolio risk: {str(e)}"
    
    def _handle_general_query(self, query: str, context: Dict) -> str:
        """Handle general financial queries that don't fit specific categories"""
        try:
            prompt = f"""You are a financial advisor AI assistant. Answer the following question with helpful, 
                       accurate financial advice and insights. Keep your response professional and actionable.
                       
                       User question: {query}
                       
                       Context: You have access to advanced financial analysis tools including market sentiment analysis,
                       options analysis, portfolio risk management, and financial news analysis. You can help with
                       investment research, risk assessment, and trading strategies.
                       
                       Available tickers: {context.get('available_tickers', ['AAPL', 'GOOGL', 'MSFT'])}
                       Currently selected ticker: {context.get('selected_ticker', 'AAPL')}
                       """
            
            response = self.gemini_model.generate_content(prompt)
            return response.text if response and response.text else "I couldn't generate a response at this time."
        except Exception as e:
            return f"I encountered an error processing your request: {str(e)}"
    
    def _extract_ticker(self, query: str, context: Dict) -> str:
        """Extract ticker symbol from query using LLM for intelligent parsing"""
        try:
            # Use LLM to intelligently extract ticker from the query
            extraction_prompt = f"""You are a financial data parser. Extract the stock ticker symbol from the following query.
            
            Query: "{query}"
            
            Available context:
            - Currently selected ticker: {context.get('selected_ticker', 'AAPL')}
            - Available tickers: {context.get('available_tickers', ['AAPL', 'GOOGL', 'MSFT'])}
            
            Instructions:
            1. If a well-known company name is mentioned (like "Apple", "Microsoft", "Tesla", "Amazon", etc.), return the corresponding stock ticker symbol
            2. If a ticker symbol is explicitly mentioned (like "AAPL", "MSFT", "TSLA"), return it as-is
            3. If no specific company/ticker is mentioned, return the currently selected ticker
            4. Use your knowledge of public companies to map company names to their ticker symbols
            5. For ambiguous cases, prefer the currently selected ticker
            6. Return ONLY the ticker symbol (2-5 uppercase letters), nothing else
            
            Examples:
            - "Apple stock" -> AAPL
            - "Microsoft performance" -> MSFT
            - "Tesla analysis" -> TSLA
            - "Amazon earnings" -> AMZN
            - "NVDA options" -> NVDA
            - "How is the market?" -> {context.get('selected_ticker', 'AAPL')}
            
            Ticker:"""
            
            response = self.gemini_model.generate_content(extraction_prompt)
            extracted_ticker = response.text.strip().upper() if response and response.text else None
            
            # Validate the extracted ticker (should be 1-5 uppercase letters)
            if extracted_ticker and 1 <= len(extracted_ticker) <= 5 and extracted_ticker.isalpha():
                return extracted_ticker
            else:
                # Fall back to context or default
                return context.get('selected_ticker', 'AAPL')
                
        except Exception as e:
            print(f"Error in LLM ticker extraction: {e}")
            # Fall back to context or default
            return context.get('selected_ticker', 'AAPL')
    
    def _extract_news_search_term(self, query: str, context: Dict) -> str:
        """Extract search term for news queries"""
        ticker = self._extract_ticker(query, context)
        
        # If query mentions specific company/ticker, search for that
        if ticker != context.get('selected_ticker', 'AAPL'):
            return ticker
        
        # Otherwise use the query itself for broader news search
        return query
    
    def _extract_tickers_for_portfolio(self, query: str, context: Dict) -> List[str]:
        """Extract ticker list for portfolio analysis using LLM"""
        try:
            # Use LLM to intelligently extract tickers from the query
            extraction_prompt = f"""You are a financial data parser. Extract ALL stock ticker symbols mentioned in the following query for portfolio analysis.
            
            Query: "{query}"
            
            Available context:
            - Currently selected ticker: {context.get('selected_ticker', 'AAPL')}
            - Available tickers: {context.get('available_tickers', ['AAPL', 'GOOGL', 'MSFT'])}
            - Default portfolio tickers: {context.get('portfolio_tickers', ['AAPL', 'GOOGL', 'MSFT'])}
            
            Instructions:
            1. Extract ALL company names and ticker symbols mentioned in the query
            2. Convert company names to their corresponding stock ticker symbols using your knowledge
            3. If multiple companies/tickers are mentioned, return all of them
            4. If no specific companies/tickers are mentioned, return the default portfolio tickers
            5. Use your knowledge of public companies to map names to ticker symbols
            6. Return only valid stock ticker symbols (2-5 uppercase letters)
            7. Format as comma-separated list (e.g., "AAPL,GOOGL,MSFT")
            8. Maximum 10 tickers
            
            Examples:
            - "Compare Apple, Microsoft, and Google" -> AAPL,MSFT,GOOGL
            - "Portfolio risk for AAPL, TSLA, NVDA" -> AAPL,TSLA,NVDA
            - "Tech stocks analysis" -> {','.join(context.get('portfolio_tickers', ['AAPL', 'GOOGL', 'MSFT']))}
            - "Diversification analysis" -> {','.join(context.get('portfolio_tickers', ['AAPL', 'GOOGL', 'MSFT']))}
            
            Tickers:"""
            
            response = self.gemini_model.generate_content(extraction_prompt)
            extracted_text = response.text.strip().upper() if response and response.text else ""
            
            if extracted_text:
                # Parse the comma-separated tickers
                tickers = [t.strip() for t in extracted_text.split(',') if t.strip()]
                
                # Validate tickers (should be 1-5 letter codes)
                valid_tickers = []
                for ticker in tickers:
                    if 1 <= len(ticker) <= 5 and ticker.isalpha():
                        valid_tickers.append(ticker)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_tickers = []
                for ticker in valid_tickers:
                    if ticker not in seen:
                        seen.add(ticker)
                        unique_tickers.append(ticker)
                
                if len(unique_tickers) >= 2:
                    return unique_tickers[:10]  # Limit to 10 tickers
            
            # Fall back to default portfolio tickers
            return context.get('portfolio_tickers', ['AAPL', 'GOOGL', 'MSFT'])
            
        except Exception as e:
            print(f"Error in LLM portfolio ticker extraction: {e}")
            # Fall back to default portfolio tickers
            return context.get('portfolio_tickers', ['AAPL', 'GOOGL', 'MSFT'])

    def _handle_market_query_streaming(self, query: str, routing: Dict, context: Dict, status_callback) -> str:
        """Streaming handler for market analysis queries"""
        ticker = self._extract_ticker(query, context)
        try:
            if status_callback:
                status_callback(f"Extracted ticker: {ticker}")
            market_tool = self.tools['market_analysis']
            stock_tool = self.tools['stock_info']
            if status_callback:
                status_callback("Analyzing market data...")
            market_data = market_tool._run(ticker)
            if status_callback:
                status_callback("Fetching stock info...")
            stock_data = stock_tool._run(ticker)
            history_context = self._build_conversation_context(context)
            prompt = f"""As an experienced market analyst, analyze the following data and answer the user's question.
            {history_context}
            User question: {query}
            Ticker: {ticker}
            
            Market Analysis Data:
            {market_data}
            
            Stock Information:
            {stock_data}
            
            Please provide a comprehensive analysis including:
            1. Current technical sentiment and key indicators
            2. Price trends and momentum  
            3. Support/resistance levels if available
            4. Investment recommendations based on the analysis
            
            Consider the conversation history for context when answering.
            Keep your response professional, actionable, and focused on the user's specific question."""
            if status_callback:
                status_callback("Synthesizing answer with Gemini...")
            response = self.gemini_model.generate_content(prompt)
            if status_callback:
                status_callback("done")
            return response.text if response and response.text else "I couldn't analyze the market data at this time."
        except Exception as e:
            if status_callback:
                status_callback(f"Error: {str(e)}")
            return f"I encountered an error while analyzing the market data: {str(e)}"

    def _handle_options_query_streaming(self, query: str, routing: Dict, context: Dict, status_callback) -> str:
        """Streaming handler for options analysis queries"""
        ticker = self._extract_ticker(query, context)
        try:
            if status_callback:
                status_callback(f"Extracted ticker: {ticker}")
            options_tool = self.tools['options_analysis']
            stock_tool = self.tools['stock_info']
            if status_callback:
                status_callback("Analyzing options data...")
            options_data = options_tool._run(ticker)
            if status_callback:
                status_callback("Fetching stock info...")
            stock_data = stock_tool._run(ticker)
            history_context = self._build_conversation_context(context)
            prompt = f"""As a specialized options strategist, analyze the following data and answer the user's question.
            {history_context}
            User question: {query}
            Ticker: {ticker}
            
            Options Analysis Data:
            {options_data}
            
            Stock Information:
            {stock_data}
            
            Please provide analysis including:
            1. Put/call ratio and sentiment
            2. Options volume and open interest trends
            3. Implied volatility analysis
            4. Potential options strategies
            
            Consider the conversation history for context when answering.
            Keep your response professional, actionable, and focused on options trading insights."""
            if status_callback:
                status_callback("Synthesizing answer with Gemini...")
            response = self.gemini_model.generate_content(prompt)
            if status_callback:
                status_callback("done")
            return response.text if response and response.text else "I couldn't analyze the options data at this time."
        except Exception as e:
            if status_callback:
                status_callback(f"Error: {str(e)}")
            return f"I encountered an error while analyzing options data: {str(e)}"

    def _handle_news_query_streaming(self, query: str, routing: Dict, context: Dict, status_callback) -> str:
        """Streaming handler for news analysis queries"""
        search_term = self._extract_news_search_term(query, context)
        try:
            if status_callback:
                status_callback(f"Search term: {search_term}")
            news_tool = self.tools['news']
            stock_tool = self.tools['stock_info']
            if status_callback:
                status_callback("Fetching news data...")
            news_data = news_tool._run(search_term)
            stock_data = stock_tool._run(search_term) if len(search_term) <= 5 else ""
            history_context = self._build_conversation_context(context)
            prompt = f"""As a financial news analyst, analyze the following data and answer the user's question.
            {history_context}
            User question: {query}
            Search term: {search_term}
            
            News Data:
            {news_data}
            
            Stock Information:
            {stock_data}
            
            Please provide analysis including:
            1. Key news developments and their market impact
            2. Sentiment analysis from recent headlines
            3. Potential market implications
            4. Investment considerations based on news flow
            
            Consider the conversation history for context when answering.
            Keep your response professional, insightful, and focused on market implications."""
            if status_callback:
                status_callback("Synthesizing answer with Gemini...")
            response = self.gemini_model.generate_content(prompt)
            if status_callback:
                status_callback("done")
            return response.text if response and response.text else "I couldn't analyze the news data at this time."
        except Exception as e:
            if status_callback:
                status_callback(f"Error: {str(e)}")
            return f"I encountered an error while analyzing news: {str(e)}"

    def _handle_portfolio_query_streaming(self, query: str, routing: Dict, context: Dict, status_callback) -> str:
        """Streaming handler for portfolio risk queries"""
        tickers = self._extract_tickers_for_portfolio(query, context)
        try:
            if status_callback:
                status_callback(f"Extracted tickers: {', '.join(tickers)}")
            portfolio_tool = self.tools['portfolio_risk']
            tickers_str = ','.join(tickers)
            if status_callback:
                status_callback("Analyzing portfolio risk...")
            portfolio_data = portfolio_tool._run(tickers_str)
            history_context = self._build_conversation_context(context)
            prompt = f"""As a quantitative risk analyst, analyze the following data and answer the user's question.
            {history_context}
            User question: {query}
            Portfolio tickers: {', '.join(tickers)}
            
            Portfolio Risk Analysis:
            {portfolio_data}
            
            Please provide analysis including:
            1. Portfolio risk metrics (VaR, correlations)
            2. Diversification analysis
            3. Risk contribution by asset
            4. Risk management recommendations
            
            Consider the conversation history for context when answering.
            Keep your response professional, quantitative, and focused on risk management insights."""
            if status_callback:
                status_callback("Synthesizing answer with Gemini...")
            response = self.gemini_model.generate_content(prompt)
            if status_callback:
                status_callback("done")
            return response.text if response and response.text else "I couldn't analyze the portfolio risk at this time."
        except Exception as e:
            if status_callback:
                status_callback(f"Error: {str(e)}")
            return f"I encountered an error while analyzing portfolio risk: {str(e)}"

    def _handle_general_query_streaming(self, query: str, context: Dict, status_callback) -> str:
        """Streaming handler for general queries"""
        try:
            if status_callback:
                status_callback("Synthesizing answer with Gemini...")
            prompt = f"""You are a financial advisor AI assistant. Answer the following question with helpful, 
                       accurate financial advice and insights. Keep your response professional and actionable.
                       
                       User question: {query}
                       
                       Context: You have access to advanced financial analysis tools including market sentiment analysis,
                       options analysis, portfolio risk management, and financial news analysis. You can help with
                       investment research, risk assessment, and trading strategies.
                       
                       Available tickers: {context.get('available_tickers', ['AAPL', 'GOOGL', 'MSFT'])}
                       Currently selected ticker: {context.get('selected_ticker', 'AAPL')}
                       """
            response = self.gemini_model.generate_content(prompt)
            if status_callback:
                status_callback("done")
            return response.text if response and response.text else "I couldn't generate a response at this time."
        except Exception as e:
            if status_callback:
                status_callback(f"Error: {str(e)}")
            return f"I encountered an error processing your request: {str(e)}"


# Global crew instance
financial_crew = None

def get_financial_crew():
    """Get or create the global financial crew instance"""
    global financial_crew
    if financial_crew is None:
        try:
            financial_crew = FinancialCrew()
        except Exception as e:
            print(f"Error initializing financial crew: {e}")
            # Return None so the fallback mechanism in app.py can handle it
            return None
    return financial_crew


