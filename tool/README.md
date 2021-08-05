# Tool for News Scraping and Interactive News Analysis

We provide tools that:

1. Interactively analyze a news article's influence to the market based on web scraping, event detection and sentiment analysis.
2. Scrapes [Reuters](https://www.reuters.com/) to download all company-specific news articles from 2017.





### 1. Environment

#### 1.1 Python Environment

```
cd tool
python3 -m pip install -r requirements.txt
```



#### 1.2 Database

We use the [Mongodb](https://www.mongodb.com/) to save the downloaded news articles, so please install mongodb (see [the official tutorial](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/)) and start it.

For example, to install Mongodb on Ubuntu18.04, run

```
sudo apt-get install libcurl4 openssl liblzma5 -y 
wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu1804-4.4.2.tgz 
tar -zxvf mongodb-linux-x86_64-ubuntu1804-4.4.2.tgz 
cp mongodb-linux-x86_64-ubuntu1804-4.4.2/bin/* /usr/local/bin/ 
sudo mkdir -p /var/lib/mongo
sudo mkdir -p /var/log/mongodb
sudo chown `whoami` /var/lib/mongo     # Or substitute another user
sudo chown `whoami` /var/log/mongodb   # Or substitute another user
mongod --dbpath /var/lib/mongo --logpath /var/log/mongodb/mongod.log --fork
```



#### 1.3 Download Trained Event Detection Model and NLTK Sentiment Model

Please download the model [here](https://drive.google.com/file/d/1PmiVjVsJe5_K28e7s2bweGujZp5kUOo0/view?usp=sharing) and copy the zip file it into `TradeTheEvent/models`.

```
cd ../models
unzip model_seed24.zip
python -c "import nltk;nltk.download('vader_lexicon')"
```


### 2. Interactive Analysis

Type `python Analyze_news.py` in command line to open the interactive tools. This tool will ask you to provide the link to the news article (e.g., [this](https://www.globenewswire.com/en/news-release/2021/04/20/2213717/11536/en/Lydall-Announces-Stock-Repurchase-Program.html)). After the link is provide, the tool will automatically download and parse the news article from the given link, and then perform sentiment analysis and event detection.

Press `Control + C` to end the tool.

![Tool_example](data/Tool_example.jpeg)


### 3. Scrape Reuters

#### 3.1 Download links to the news articles

To download all of the news articles:
```
python Download_links.py --download_all
```

To download news articles for a specific company(e.g., Apple):
```
python Download_links.py --ticker AAPL
```


#### 3.2 Download news articles from the links
```
python Download_news.py
```
