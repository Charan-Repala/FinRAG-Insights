import os
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import yfinance as yf
from typing import Type

class YahooFinanceToolInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol to fetch data for (e.g., AAPL).")

class YahooFinanceTool(BaseTool):
    name: str = "YahooFinanceTool"
    description: str = "Fetch stock data from Yahoo Finance."
    args_schema: Type[BaseModel] = YahooFinanceToolInput

    def _run(self, symbol: str) -> str:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            return f"Stock: {symbol}\nPrice: {info.get('currentPrice', 'N/A')}\nMarket Cap: {info.get('marketCap', 'N/A')}"
        except Exception as e:
            return f"Error fetching stock data: {str(e)}"

class NewsAPIToolInput(BaseModel):
    query: str = Field(..., description="Query to search news for.")

class NewsAPITool(BaseTool):
    name: str = "NewsAPITool"
    description: str = "Fetch news articles based on a query using NewsAPI."
    args_schema: Type[BaseModel] = NewsAPIToolInput

    def _run(self, query: str) -> str:
        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            return "NewsAPI key not found in environment variables."
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}"
        try:
            response = requests.get(url)
            data = response.json()
            if data['status'] == 'ok':
                articles = data['articles'][:3]  # Limit to top 3 articles
                result = ""
                for article in articles:
                    result += f"Title: {article['title']}\nDescription: {article['description']}\nURL: {article['url']}\n\n"
                return result.strip()
            else:
                return "No news found."
        except Exception as e:
            return f"Error fetching news: {str(e)}"

class CryptoAPIToolInput(BaseModel):
    coin: str = Field(..., description="Cryptocurrency to fetch data for (e.g., bitcoin).")

class CryptoAPITool(BaseTool):
    name: str = "CryptoAPITool"
    description: str = "Fetch cryptocurrency data from CoinGecko."
    args_schema: Type[BaseModel] = CryptoAPIToolInput

    def _run(self, coin: str) -> str:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd"
        try:
            response = requests.get(url)
            data = response.json()
            if coin in data:
                price = data[coin]['usd']
                return f"{coin.capitalize()} price: ${price}"
            else:
                return f"Cryptocurrency {coin} not found."
        except Exception as e:
            return f"Error fetching crypto data: {str(e)}"