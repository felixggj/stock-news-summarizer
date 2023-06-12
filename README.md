# Stock News Scraper & Summarizer with BeautifulSoup and NewsAPI | Felix GG

## Description
This is a simple Python script that scrapes the latest news articles from the [Yahoo Finance](https://finance.yahoo.com/) website and summarizes them using the [NewsAPI](https://newsapi.org/) API. The script is designed to be run on a daily basis to provide a quick summary of the latest news articles for a given stock ticker.

## Technologies used

- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) - Python library for pulling data out of HTML and XML files
- [NewsAPI](https://newsapi.org/) - API for searching and retrieving live articles from all over the web
- [Transformer](https://huggingface.co/transformers/) - Python library for Natural Language Processing (NLP) tasks using state-of-the-art Transformer models

## Installation
1. Clone the repository
2. Install the required packages using `requirements.yml`
3. TensorFlow Installation will change depending on your machine. See [TensorFlow](https://www.tensorflow.org/install) for more information. Set-up in the .yml works for M1 Macs. 
4. Enter file NewsScript.py and edit the following:
    - `monitored_tickets` to the desired stock tickers
    - `api_key` to your NewsAPI API key
    - `file_name` where it says "YOURFILENAME.csv"
5. Run the script

