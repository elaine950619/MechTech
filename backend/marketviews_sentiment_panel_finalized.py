# This version was revised on text errors already

''' Outline
1. Set Up and Data Gathering
2. Data Processing: Parse and Retrieve the News Data
3. Sentiment Analysis on News Data from Finviz Website with Vader
4. Data Aggregation With Wikipedia Data
5. Interactive Visualization <br>
Each section has subsections with a general description on each part of the code's funcationalities. More detailed explaination is in the comment accompnied with certain lines of code.
'''

# 1. Set Up and Data Gathering
# 1.1 Importing packages to extract data from webpage and neccesary libraries to complete the basic data manipulation, analyzation(sentiment analysis), and visualization.

# libraries for webscraping, parsing and getting stock data
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

import plotly.graph_objs as go
# for plotting and data manipulation
import pandas as pd
import numpy as np
# import plotly
import plotly.express as px
from datetime import timedelta,datetime
import requests
import json
# NLTK VADER for sentiment analysis
import nltk
import os
# from datetime import date

# Lexicon-based approach to do sentiment analysis for of social media texts
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
panel_dir = os.path.join(BASE_DIR, 'panel_data/')
directory = panel_dir

# 1.2 Accessing a Wikipedia page containing information of S&P 500 companies and extracting their tickers into a list.
def get_tickers(debug):
    # Get all dataframes from a given URL, note this only access data frame of the website not the html file
    wikipedia_url = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

    stocks = None
    # Iterate through all tables in the webpage and find the one with 'Symbol' column
    for t in wikipedia_url:
        if 'Symbol' in t.columns:
            stocks = t
            break
    if stocks is None:
        raise ValueError("No table with 'Symbol' column found.")
    # Get the 'Symbol' column
    tickers = stocks['Symbol']
    # Convert it to a list
    tickers = tickers.tolist() if debug==False else tickers.tolist()[:50]
    return tickers

# 1.3 Accessing Finviz website to concurrently retrieve news data of each S&P 500 companies

# (1) Preparation
import time
# import random
# Libraries for executing concurrent tasks
from concurrent.futures import ThreadPoolExecutor, as_completed
# from urllib.request import Request, urlopen
# from bs4 import BeautifulSoup

#fetchs data from finwiz and retrun a dict

# URL for fetching financial data from the website "https://finviz.com/"

# (2) Method Definition to Retrieve News Data
# Helper method to fetch news data for each stock ticker from finviz website
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError


def fetch_news_table(ticker):
    finwiz_url = 'https://finviz.com/quote.ashx?t='
    # A random delay of 5 to 10 seconds before making the web request
    #time.sleep(np.random.uniform(5, 10))
    time.sleep(np.random.uniform(5, 10))
    # Replaces any '.' in the ticker symbol with '-' to ensure compatibility with the URL structure
    ticker = ticker.replace(".", "-")
    print(ticker)

    # Construct the URL for a specific stock on the Finviz website
    url = finwiz_url + ticker

    ###### Try to make the request 3 times before giving up ######
    for _ in range(3):
        try:
            # Request object
            # User-Agent header is to mimic a web browser and avoid any restrictions or blocks from the website
            req = Request(url=url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'})

            response = urlopen(req)
            # Parsing the HTML content
            html = BeautifulSoup(response, 'html.parser')

            # Get the table containing news data related to the stock
            news_table = html.find(id='news-table')
            return ticker, news_table
        except (IncompleteRead, HTTPError, URLError) as e:
            print("Error occurred while fetching data for {}: {}".format(ticker, str(e)))
            time.sleep(4)  # Wait for 2 seconds before trying again
            continue

    print("Failed to fetch data for {} after 3 attempts".format(ticker))
    return ticker, None


# Method to implement the helper method **fetch_news_table(ticker)** with rate limiting
def process_ticker(index, ticker):
    # 1 second sleep for delay between processing each ticker
    sleep_duration = 1
    # A longer sleep duration between 5 and 10 seconds after certain amount of consecutive requests to prevent overwhelming the website's server
    if index % 50 == 0:
        time.sleep(np.random.uniform(5, 10))
    # Call the method fetch_news_table(ticker)
    return fetch_news_table(ticker)


# (3) Concurrent tasks to get multiple tickers' news data(table) parallelly
#  A maximum of 8 worker threads

# An dictionary used to store news data fetched from web pages.
def get_news_table(tickers):
    news_tables = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Iterate over the tickers list
        # Submit process_ticker(index, ticker) method to the executor for execution
        # Create a dictionary, keys are futures returned by the executor.submit(),which is a tuple of ticker and news_table, which is a beautiful soup object and values are ticker
        futures = {executor.submit(process_ticker, index, ticker): ticker for index, ticker in enumerate(tickers)}
        # Iterate over each future in the futures dictionary as they are completed, the as_completed means the completed future
        for future in as_completed(futures):
            ticker, news_table = future.result()  # Retrieve the result of a completed future, future.result is the tuple of ticker and news_table
            news_tables[ticker] = news_table  # Add the retrieved news table to the news_tables dictionary: Tickers serve as the key, and values are html object

    return news_tables
# 2. Data Processing: Parse and Retrieve the News Data
# Iterate through each news table obtained from the website, parse the news text to extract the relevant information (ticker symbol, date, time, and headline) and append them to the **parsed_news** list.
def parse_news_table(news_tables):
    parsed_news = []  # To store the parsed news data
    # Iterate through the news data
    # the element of parsed_news are a list of [stockname,date,time]
    for file_name, news_table in news_tables.items():
        # Iterate through all tr tags in 'news_table', note we need to use.items() to iterate through the dictionary, and this way will return key-value tuple
        for x in news_table.findAll('tr'):
            # Skip if we never got a valid news_table (e.g., 429 error)
            if news_table is None:
                continue
            
            for row in news_table.findAll('tr'):
                # Make sure the row has at least one <td> ...
                tds = row.findAll('td')
                if not tds:
                    continue

            try:
                # get text from tag <a> only to extract news headline
                if len(x.findAll('a')) == 0:
                    continue
                text = x.a.get_text()

                # split text in the td (usually contains date and time info) tag into a list
                date_scrape = x.td.text.split()
                # if the length of 'date_scrape' is 1 (only the time information is available.), load 'time' as the only element
                if len(date_scrape) == 1:
                    # if there is only the time information, it's probably not the first news at that day
                    # date = parsed_news[-1][1]
                    time = date_scrape[0]
                # else load 'date' as the 1st element and 'time' as the second
                else:
                    date = date_scrape[0]
                    time = date_scrape[1]
                # Extract the ticker from the file name, get the string up to the 1st '_'
                ###### ticker = file_name.split('_')[0] ######
                # Append ticker, date, time and headline to the 'parsed_news' list
                ###### parsed_news.append([ticker, date, time, text]) ######
                # date here might be undefined: since date will not be updated if we can only retrieve time information,
                # date will automatically be the same as previous news date. To remove warning here,
                # we can use date=parsed_news[-1][1]
                parsed_news.append([file_name, date, time, text])
            # Catches any exceptions that occur during the process
            except Exception as e:
                print(e)
    # print(error_files)
    return parsed_news
# 3. Sentiment Analysis on News Data from Finviz Website with Vader

'''
3.1 Analyze the sentiment of all news data with VADER and get each company's overall sentiment scores<br>
Operate sentiment analysis on news' headline with Vader and create a dataframe called **parsed_and_scored_news** which includes necessary
information about each news (tickers, date, time, headline, and sentiment scores). Then, get companies' overall sentiment scores by averaging the sentiment of all news.
'''

# Instantiate the sentiment intensity analyzer
def sentiment_analysis(parsed_news):

    vader = SentimentIntensityAnalyzer()
    # Set column names
    columns = ['ticker', 'date', 'time', 'headline']
    # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
    parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)

    # Iterate through the headlines and get the polarity scores using vader
    scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()
    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)

    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')
    # Convert the date column from string to datetime
    today = datetime.today().date()  # Get the current date
    parsed_and_scored_news['date'] = parsed_and_scored_news['date'].replace('Today', today)


    # Group by each ticker and get the mean of all sentiment scores （it might have repetitive tickers in the data frame, this line incorporate all the ticker and tget the average value. Groupby() need to follow a aggregate function like .ean() or .count())
    mean_scores = parsed_and_scored_news.groupby(['ticker'])[['neg', 'neu', 'pos', 'compound']].mean()
    parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news['date'], errors='coerce')

    return parsed_and_scored_news

# stores the most recent date of each ticker


# 3.2 Analyze the sentiment of the-most-2-recent-days news data and get each company's overall recent sentiment scores
# A new dataframe to store the recent news data and sentiment scores
def get_recent_data(parsed_and_scored_news):
    frames = []
    max_dates = parsed_and_scored_news.groupby('ticker')['date'].max()
    # Find recent news data for each ticker frame is a datafram whose column matches the condition ticker column matches to ticker and date column matches the most recent date and date before this date, other unlatching rows get deleted.
    for ticker, max_date in max_dates.items():
        # Filter recent news data in the last two days
        frame = parsed_and_scored_news[(parsed_and_scored_news['ticker'] == ticker) &
                                       ((parsed_and_scored_news['date'] == max_date) |
                                        (parsed_and_scored_news['date'] == max_date - timedelta(days=1)))]
        frames.append(frame)

    # Combine data for all tickers， note frame in frames are rows of ticker and its date are different concat is essentially  make a datraframe that only keep the rows of the most recent two days of the parse_stock_news
    recent_two_days_news = pd.concat(frames)

    # Calculate the average sentiment score for each ticker in the last two days
    mean_scores = recent_two_days_news.groupby(['ticker'])[['neg', 'neu', 'pos', 'compound']].mean()

    # Remove tickers that have no data in the last two days
    mean_scores = mean_scores.dropna()
    # Reindex
    mean_scores = mean_scores.reset_index()
    #Change ticker column to Ticker column
    mean_scores = mean_scores.rename(columns={'ticker': 'Ticker'})
    return mean_scores

# 4. Data Aggregation With Wikipedia Data
'''
Retrieve the S&P 500 company tickers and their respective sectors from Wikipedia, merge the tickers with the mean sentiment scores
to get a new dataframe, and identify the top 5 and bottom 5 companies with the highest and lowest sentiment scores for each sector.
'''

def get_wiki_data(mean_scores):
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    # Retrieve and parsing
    #handle the connection error
    try:
        response = requests.get(url)
    except requests.exceptions.RequestException as req_error:
        print(f"Request Exception in getting wiki data : {req_error}")
        if isinstance(req_error, requests.exceptions.ConnectionError):
            print("Connection Error. Retrying...")
            time.sleep(30)
            return get_wiki_data(mean_scores)

    soup = BeautifulSoup(response.text, 'html.parser')
    # Find the table with the CSS class 'wikitable sortable' within the HTML content, the 'table' argument in soup.find() specifies that we are looking for an HTML table element, and the second argument {'class': 'wikitable sortable'} defines the attributes (in this case, the CSS class) that the table element should have in order to be considered a match.
    table = soup.find('table', {'class': 'wikitable sortable'})
    # Retrieve all the rows of table, excluding the first one which is header row. Table is a tag  of beautiful soup, and we can apply .findall() and it returns a list of tag object
    rows = table.findAll('tr')[1:]
    # To store the extracted tickers and sectors
    tickers_and_sectors = []

    # Iterate all rows/stocks
    for row in rows:
        # Extract the stock's ticker and sector
        cells = row.findAll('td')
        ticker, sector = None, None
        if len(cells) > 0:
            ticker = cells[0].text.strip()
            sector = row.findAll('td')[2].text.strip()
        # ticker = row.findAll('td')[0].text.strip()
        # sector = row.findAll('td')[2].text.strip()
        # Add the tuple (ticker, sector) to a list
        tickers_and_sectors.append((ticker, sector))

    # Convert list to dataframe
    tickers = pd.DataFrame(tickers_and_sectors, columns=['Ticker', 'Sector'])

    df = tickers.merge(mean_scores, on='Ticker')
    df = df.rename(columns={"compound": "Sentiment Score", "neg": "Negative", "neu": "Neutral", "pos": "Positive"})
    df = df.reset_index()
    grouped = df.groupby('Sector')
    return grouped
# Selects the top 5 stocks with the highest sentiment score for each sector
# The apply() will apply the lambda function on each group(a sector correspond to a group of values)
# The nlargest(5, 'Sentiment Score') method retrieve 5 rows with the highest 'Sentiment Score' within each group. It returns a DataFrame containing the top 5 rows for each group.
# but group.apply will return a data frame
# lamda x takes an input  x, which is a group of data in sector, and applies the nlargest() method on that group. nlargest is a panda function applied to

def get_top_five(grouped):

    top_5_each_sector = grouped.apply(lambda x: x.nlargest(5, 'Sentiment Score')).reset_index(drop=True)
    # Selects the top 5 stocks with the lowest sentiment score
    low_5_each_sector = grouped.apply(lambda x: x.nsmallest(5, 'Sentiment Score')).reset_index(drop=True)
    return  top_5_each_sector,low_5_each_sector


def get_data_to_draw(debug):
    tickers = get_tickers(debug)
    news_tables = get_news_table(tickers)
    parsed_news = parse_news_table(news_tables)
    parsed_news_scores = sentiment_analysis(parsed_news)
    mean_scores = get_recent_data(parsed_news_scores)
    grouped = get_wiki_data(mean_scores)
    top_5_each_sector, low_5_each_sector = get_top_five(grouped)
    return top_5_each_sector, low_5_each_sector

# 5. Interactive Visualization
'''
Display the tree map of sentiment scores for the top 5 or low 5 companies in each sector based on the users' input.<br>
The first figure is for the top 5 of each sector, and the second figure is for the low 5 of each sector
'''
# Check the valid input


def draw_sentiment_panel(top_5_each_sector, low_5_each_sector):


# For the most positive

    # Visualization attributes. Sectors serve as the root level of the treemap hierarchy. The color of each treemap cell based on the 'Sentiment Score' column
    fig = px.treemap(top_5_each_sector, path=[px.Constant("Sectors"), 'Sector', 'Ticker'],
                     color='Sentiment Score', hover_data=['Negative', 'Neutral', 'Positive', 'Sentiment Score'],
                     color_continuous_scale=['#FF0000', "#000000", '#00FF00'],
                     color_continuous_midpoint=0)

    # Customize the hover tooltip text
    fig.data[0].customdata = top_5_each_sector[['Negative', 'Neutral', 'Positive', 'Sentiment Score']].round(
        3)  # round to 3 decimal places
    fig.data[0].texttemplate = "%{label}<br>%{customdata[3]}"
    fig.update_traces(textposition="middle center")
    fig.update_layout(margin=dict(t=30, l=10, r=10, b=10), font_size=20)

    # plotly.offline.plot(fig, filename='stock_sentiment.html') # this writes the plot into a html file and opens it

    # For the most negative


    fig1 = px.treemap(low_5_each_sector, path=[px.Constant("Sectors"), 'Sector', 'Ticker'],
                     color='Sentiment Score', hover_data=['Negative', 'Neutral', 'Positive', 'Sentiment Score'],
                     color_continuous_scale=['#FF0000', "#000000", '#00FF00'],
                     color_continuous_midpoint=0)

    fig1.data[0].customdata = low_5_each_sector[['Negative', 'Neutral', 'Positive', 'Sentiment Score']].round(
        3)  # round to 3 decimal places
    fig1.data[0].texttemplate = "%{label}<br>%{customdata[3]}"
    fig1.update_traces(textposition="middle center")
    fig1.update_layout(margin=dict(t=30, l=10, r=10, b=10), font_size=20)

    # plotly.offline.plot(fig, filename='stock_sentiment.html') # this writes the plot into a html file and opens it
    return fig.to_json(),fig1.to_json()

def store_json(fig1,fig2,now_time,absolute_path):
    with open(f"{absolute_path}Top5-{now_time}.json", 'w') as file:
        file.write(fig1)
    with open(f"{absolute_path}Low5-{now_time}.json", 'w') as file:
        file.write(fig2)


def read_json(file1,file2):
    with open(file1, 'r') as json_file:
        fig1_data = json.load(json_file)
    fig1 = go.Figure(data=fig1_data['data'])
    with open(file2, 'r') as json_file:
        fig2_data = json.load(json_file)
    fig2 = go.Figure(data=fig2_data['data'])
    return fig1,fig2

#top_5_each_sector, low_5_each_sector = get_data_to_draw()
#fig1, fig2 =draw_sentiment_panel(top_5_each_sector, low_5_each_sector)
#store_json(fig1, fig2, "temp","/home/MarketMonitor/Webpage_Tutorial_2/panel_data/")
