from dotenv import load_dotenv
from pydantic import BaseModel  # For Schema to define what data the end points accepts
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI  # type: ignore

import yfinance as yf  # type: ignore


load_dotenv() # Load Key Value Pairs


app = FastAPI()

model = ChatOpenAI(
    model='c1/openai/gpt-5/v-20250930',
    base_url='https://api.thesys.dev/v1/embed/'

)

checkpointer = InMemorySaver()


@tool('get_stock_price', description='A function that returns the current stock price based on a ticker symbol.')
def get_stock_price(ticker: str):
    print('get_stock_price tool is being used.')
    stock = yf.Ticker(ticker)
    return stock.history()['Close'].iloc[-1]


@tool('get_historical_stock_price', description='A function that returns the current stock price over time based on a ticker symbol and a start and end date.')
def get_historical_stock_price(ticker: str, start_date: str, end_date: str):
    print('get_historical_stock_price tool is being used.')
    stock = yf.Ticker(ticker)
    return stock.history(start=start_date, end=end_date).to_dict()


@tool('get_balance_sheet', description='A function that returns the balance sheet based on a ticker symbol and a given year.')
def get_balance_sheet(ticker: str, year: int):
    print('get_balance_sheet tool is being used.')
    stock = yf.Ticker(ticker)
    return stock.balance_sheet


@tool('get_stock_news', description='A function that returns news based on a ticker symbol.')
def get_stock_news(ticker: str):
    print('get_stock_news tool is being used.')
    stock = yf.Ticker(ticker)
    return stock.news


agent = create_agent(
    model=model,
    checkpointer=checkpointer,
    tools=[get_stock_price, get_historical_stock_price, get_balance_sheet, get_stock_news]
)


class PromptObject(BaseModel):
    content: str  # The Message
    id: str
    role: str  # The role of the user/sender


class RequestObject(BaseModel):
    prompt: PromptObject
    threadId: str
    responseId: str


# Route to handle conversation
@app.post('/api/chat')
async def chat(request: RequestObject):
    # Passing threadID ensures context remains throughout conversation history
    config = {'configurable': {'thread_id': request.threadId}}

    def generate():
        for token, _ in agent.stream(
                # instructs model it is a financial assistant
                {'messages': [
                    SystemMessage('You are a stock analysis assistant. You have the '
                                  'ability to get real-time stock prices (given a date range), historical stock prices, '
                                  'news and balance sheet data for a given ticker symbol.'),
                    HumanMessage(request.prompt.content)
                ]},
                stream_mode='messages',
                config=config
        ):
            yield token.content

    return StreamingResponse(generate(), media_type='text/event-stream', headers={
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive'
    })

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8888)
