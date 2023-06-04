#!/usr/bin/env python
from datetime import datetime, timedelta
from newsapi import NewsApiClient
import sqlite3

api = NewsApiClient(api_key='6e4633ea353f41ee806d21ea559001a4')

def get_news(querry, page=1):
    final_time = datetime.now()
    initial_time = final_time - timedelta(days=6)
    final_time = final_time - timedelta(days=0)
    print(f"final_time: {final_time} initial_time {initial_time}")
    response = api.get_everything(
        q=querry, 
        from_param=initial_time, 
        to=final_time,
        sort_by='relevancy', 
        language='en', 
        page_size=10,
        page=page
    )
    return response


con = sqlite3.connect("news.db")
cur = con.cursor()

# stock_tickers = [
#                 "AAPL",
#                 "AMC",
#                 "AMD",
#                 "AMZN",
#                 "BAC",
#                 "DIS",
#                 "GD",
#                 "GM",
#                 "FORD",
#                 "GOOGL",
#                 "JNJ",
#                 "JPM",
#                 "LMT",
#                 "META",
#                 "MSFT",
#                 "NVDA",
#                 "PLTR",
#                 "RTX",
#                 "SNAP",
#                 "TSLA"
#                 ]
stock_tickers = [
                "ACB",
                "AMGN",
                "ALLT",
                "ATRI",
                "BMY",
                "CARV",
                "EMKR",
                "EXPR",
                "HIBB",
                "IFF",
                "IROQ",
                "KDP",
                "MBRX",
                "MOS",
                "PYPL",
                "SNBR",
                "WBA",
                ]

for stock_ticker in stock_tickers:
    response = get_news(stock_ticker)
    # print(f"response {response}")
    for article in response['articles']:
        dt = datetime.strptime(
            article['publishedAt'], 
            '%Y-%m-%dT%H:%M:%SZ'
        )
        title = article['title'].replace('\'', '\'\'')
        try:
            description = article['description'].replace('\'', '\'\'')
            content = article['content'].replace('\'', '\'\'')
        except AttributeError:
            continue
        command = f"""
        INSERT INTO news VALUES
            (
                '{stock_ticker}', 
                '{dt}', 
                '{title}', 
                '{description}',
                '{content}',
                '{article['url']}',
                "" 
                )
        """
        print(command)
        cur.execute(command)
    con.commit()

