# InvestAssist: Your AI-Powered Financial Co-pilot

InvestAssist is an intelligent financial analysis platform designed to provide comprehensive market insights and investment recommendations. Built with an **agentic architecture** using CrewAI, it leverages the collaborative power of specialized AI agents to deliver real-time, actionable intelligence.

## Agentic Workflow

InvestAssist is a **multi-agent system** orchestrated by the `FinancialCrew`. This setup ensures that your financial queries are handled with precision and expertise.

Here's a simplified look at how it all comes together:

1.  **Input Processing:** Query is received and analyzed, considering past conversations for context.
2.  **Smart Routing:** The `FinancialCrew` intelligently routes your request to the most suitable specialized agent (or agents) based on its understanding of your intent.
3.  **Deep Dive Analysis:** The selected agent(s) spring into action, utilizing their specific tools and expertise to gather and process relevant financial data. You'll see real-time status updates as they work!
4.  **Actionable Insights:** Using advanced AI, the agent(s) synthesize their findings into clear, human-readable, and actionable insights tailored to user query.

This collaborative approach means you get specialized, context-aware, and efficient financial analysis, every time.

## Specialized AI Agents

InvestAssist features four dedicated AI agents:

*   **Market Analyst:** Responsible for technical and fundamental analysis, market sentiment, and price trends.
*   **Options Strategist:** Specializes in options data, volatility, Greeks, and strategic recommendations.
*   **Financial News Analyst:** Fetches real-time financial news, earnings, and market-moving events using NewsAPI.
*   **Portfolio Risk Manager:** Focuses on portfolio optimization, risk assessment (VaR, correlation), and backtesting.

## ✨ Key Features

*   **Multi-Modal Analysis:** Comprehensive insights covering technicals, options, news, and risk.
*   **Intelligent Agent Collaboration:** Seamless routing and context-aware responses for complex queries.

## Get Started

### Prerequisites

Make sure you have Python installed. Then, install the necessary packages:

```bash
pip install flask crewai google-generativeai yfinance pandas numpy
```

### Environment Configuration

Set your API keys. It's recommended to use a `.env` file or environment variables. For example, create a `keys.txt` (or similar) file:

```
export GEMINI_API_KEY="your_gemini_api_key"
export NEWS_API_KEY="your_news_api_key"
```

### Running the Application

1.  Navigate to the `InvestAssist` directory.
2.  Run the application:
    ```bash
    python app.py
    ```
3.  Access InvestAssist in your browser at `http://localhost:5000`.

## Docker Deployment

For easy deployment, you can containerize InvestAssist:

1.  **Build the Docker Image:**
    ```bash
    docker build -t investassist .
    ```
2.  **Run the Container:**
    ```bash
    docker run -p 5000:5000 -e NEWS_API_KEY="YOUR_NEWS_API_KEY" -e GEMINI_API_KEY="YOUR_GEMINI_API_KEY" investassist
    ```
    (Remember to replace placeholder API keys with your actual ones.)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).