# Import Libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Step 1: Scrape Data from a Financial News Website
def scrape_headlines(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    headlines = soup.find_all('h3', class_='Mb(5px)')  # Yahoo Finance example
    news_list = [headline.text for headline in headlines]
    return news_list

# Step 2: Preprocess Text
def preprocess_text(text):
    # Tokenization and removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    cleaned_text = ' '.join(word for word in tokens if word.isalnum() and word not in stop_words)
    return cleaned_text

# Step 3: Perform Sentiment Analysis
def analyze_sentiment(news_list):
    sentiment_scores = []
    for news in news_list:
        cleaned_news = preprocess_text(news)
        score = sia.polarity_scores(cleaned_news)
        sentiment_scores.append(score['compound'])  # Compound score for overall sentiment
    return sentiment_scores

# Step 4: Visualize Sentiment
def visualize_sentiment(sentiment_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(sentiment_scores, marker='o', label='Sentiment Score')
    plt.axhline(0, color='red', linestyle='--', label='Neutral Sentiment')
    plt.title('Sentiment Analysis of Stock Market News')
    plt.xlabel('News Articles')
    plt.ylabel('Sentiment Score')
    plt.legend()
    plt.show()

# Main Program Execution
if __name__ == "__main__":
    # Step 1: Scrape Data
    url = "https://finance.yahoo.com/"  # Change to your desired financial news site
    news_list = scrape_headlines(url)
    
    # Display Scraped Headlines
    print("\nScraped Headlines:")
    for news in news_list:
        print(f"- {news}")
    
    # Step 2 & 3: Preprocess and Analyze Sentiment
    sentiment_scores = analyze_sentiment(news_list)
    
    # Combine Headlines and Sentiment Scores into a DataFrame
    df = pd.DataFrame({'Headline': news_list, 'Sentiment Score': sentiment_scores})
    print("\nSentiment Analysis Results:")
    print(df)
    
    # Step 4: Visualize Results
    visualize_sentiment(sentiment_scores)
