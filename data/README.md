# EDT dataset

The EDT dataset is designed for corporate event detection and text-based stock prediction (trading strategy) benchmark. It includes `9721​` news articles with token-level event labels and `303893​` news articles with minute-level timestamps and comprehensive stock price labels. 



The dataset is available [here](https://drive.google.com/drive/folders/1xKjd9hzA8UTn2DXVIYYnX5TngNAMom19?usp=sharing).



## Detail Information

EDT contains data for three purposes: 1. corporate event detection; 2. news-based trading strategy benchmark; 3. financial domain adaptation. 



#### 1. Corporate Event Detection

EDT contains 11 types of corporate events: Acquisition(A), Clinical Trial(CT), Regular Dividend(RD), Dividend Cut(DC), Dividend Increase(DI), Guidance Increase(GI), New Contract(NC), Reverse Stock Split(RSS), Special Dividend(SD), Stock Repurchase(SR) and Stock Split(SS).

A special category "O" stands for no event.



`train.txt` contains the training data and `dev.txt` contains the validation data. Each article/document is seperated by a white line (`\n`).





#### 2. Trading Benchmark

The benchmark dataset contains `303893​` news articles range from `2020/03/01` to `2021/05/06`. The articles are downloaded from the [PRNewswire](https://www.prnewswire.com/) and [Businesswire](https://www.businesswire.com/).



Each article is saved as a dictionary with the following keys:

```
'title': Title and possible subtitle of the news article.
'text': Main text of the news article.
'pub_time': Adjusted minute-level timestamp of the article's publish time.
'labels': [
	'ticker': An automatically recognized ticker of the company that occurs most in the article. 
	          The following price labels comes from the price data of this ticker. 
	          If not ticker is recognized, the value is empty and there is no price labels.
	'start_time': The first available trading time of the ticker after the news is published.
	'start_price_open': The "Open" price of the ticker at 'start_time'.
	'start_price_close': The "Close" price of the ticker at 'start_time'.
	'end_price_1day': The "Close" price at the last minute of the following 1 trading day.
	                  The "following 1 trading day" refers to the same day as the "start_time"
	                  if 'start_time' if early than 4pm ET. Otherwise, it refers to the next 
	                  trading day. And so on for "following n trading day"
	'end_price_2day': The "Close" price at the last minute of the following 2 trading days.
	'end_price_3day': The "Close" price at the last minute of the following 3 trading days.
	'end_time_1day': The time corresponds to 'end_price_1day'.
	'end_time_2day': The time corresponds to 'end_price_2day'.
	'end_time_3day': The time corresponds to 'end_price_1day'.
	'highest_price_1day': The highest price in the following 1 trading day.
	'highest_price_2day': The highest price in the following 2 trading days.
	'highest_price_3day': The highest price in the following 3 trading days.
	'highest_time_1day': The time corresponds to 'highest_price_1day'.
	'highest_time_2day': The time corresponds to 'highest_price_2day'.
	'highest_time_3day': The time corresponds to 'highest_price_3day'.
	'lowest_price_1day': The lowest price in the following 1 trading day.
	'lowest_price_2day': The lowest price in the following 2 trading days.
	'lowest_price_2day': The lowest price in the following 3 trading days.
	'lowest_time_1day': The time corresponds to 'lowest_price_1day'.
	'lowest_time_2day': The time corresponds to 'lowest_price_2day'.
	'lowest_time_3day': The time corresponds to 'lowest_price_3day'.
] 
```



<font color=red>**IMPORTANT: Ticker of each article is automatically assigned. So there are errors. Please be aware of this if you are using the 'labels' for model evaluation.**</font>

As described in the paper, for news articles that describe a specific company (e.g., [this](https://www.globenewswire.com/news-release/2019/06/03/1863089/0/en/Avenue-Therapeutics-Announces-Positive-Topline-Data-from-Second-Pivotal-Phase-3-Study-of-Intravenous-Tramadol-in-the-Management-of-Postoperative-Pain.html) and [this](https://www.prnewswire.com/news-releases/flowers-foods-increases-quarterly-dividend-301063646.html)), the ticker recognizer achieves an accuracy of 98% in recoginizing the ticker. However, recently, we realize that some articles that are not related to any company are assigned with random tickers due to the greedy nature of the ticker recognizer. 

You may either label the ticker and price changes by yourself or stick with our original labels to directly compare with our paper. A naive way to filter out (almost) all the mistakenly labeled item is:

```python
import json

with open('PATH_TO_DATA', 'r') as f:
  original_data = json.load(f)
 
filtered_data = []
for item in origianal_data:
  # only take the articles that are assigned with a ticker
  if len(item['labels']) > 10:
    # if the ticker of keywords like 'NYSE' and 'NASDAQ' exist in the article
    # then the label is very likely to be correct
    if 'nyse' in item['text'].lower() or 'nasdaq' in item['text'].lower() or item['labels']['ticker'] in item['text']:
      filtered_data.append(item)
```









#### 3. Domain Adaptation

The corpus for domain adaptation contains financial news articles and a financial terms encyclopedia. The encyclopediais downloaded from [Investopedia](https://www.investopedia.com/).



Similar to the data for event detetcion, `train.txt` contains the training data and `dev.txt` contains the validation data. Each article/document is seperated by a white line (`\n`).

