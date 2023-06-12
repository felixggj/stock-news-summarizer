from transformers import PegasusTokenizer, TFPegasusForConditionalGeneration
from bs4 import BeautifulSoup
import requests
from newsapi import NewsApiClient

model_name = "human-centered-summarization/financial-summarization-pegasus"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = TFPegasusForConditionalGeneration.from_pretrained(model_name)

monitored_tickets = ['NVDA', 'ASML', 'AAPL']

def search_for_news_urls(ticker):
    newsapi = NewsApiClient(api_key='YOURAPIKEY')

    # Get articles related to the ticker and specific sources
    all_articles = newsapi.get_everything(q=f'{ticker} "Motley Fool" OR "Yahoo Finance" OR "Marketwatch"',
                                          language='en')

    unwanted_substrings = ['biztoc', 'consent', 'bossip', 'sparkfun', 'avclub', 'rapradar', 'archdaily', 'chaos2ch', 'huffpost', 'lwn', 'spring', 'comics', 'libsyn']

    hrefs = []
    for article in all_articles['articles']:
        url = article['url']
        # Only include the URL if it doesn't contain any of the unwanted substrings
        if not any(substring in url for substring in unwanted_substrings):
            hrefs.append(url)

    return hrefs[:10]


raw_urls = {ticker:search_for_news_urls(ticker) for ticker in monitored_tickets}

def scrape_and_process(URLs):
    ARTICLES = []
    for url in URLs:
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = [paragraph.text for paragraph in paragraphs]
        words = ' '.join(text).split(' ')[:350]
        ARTICLE = ' '.join(words)
        ARTICLES.append(ARTICLE)
    return ARTICLES

articles = {ticker:scrape_and_process(raw_urls[ticker]) for ticker in monitored_tickets}

# Function to summarize all of the scraped articles
def summarize(articles):
    summaries = []
    for article in articles:
        input_ids = tokenizer.encode(article, return_tensors="tf")
        output= model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries


summaries = {ticker:summarize(articles[ticker]) for ticker in monitored_tickets}

from transformers import pipeline
sentiment = pipeline("sentiment-analysis")

scores = {ticker:sentiment(summaries[ticker]) for ticker in monitored_tickets}


print(summaries["NVDA"][0], scores["NVDA"][0]["label"], scores["NVDA"][0]["score"])

# creating a function to output an array of all final summaries with their sentiment scores as well.
def create_output_array(summaries, scores, urls):
    output_array = []
    for ticker in monitored_tickets:
        for counter in range(len(summaries[ticker])):
            output = [
                ticker,
                summaries[ticker][counter],
                scores[ticker][counter]["label"],
                scores[ticker][counter]["score"],
                urls[ticker][counter],
            ]
            output_array.append(output)
    return output_array


final_output = create_output_array(summaries, scores, raw_urls)

final_output.insert(0, ["Ticker", "Summary", "Label", "Confidence", "URL"])

# Downloading constructed array into a csv file
import csv
with open('YOURFILENAME.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerows(final_output)





