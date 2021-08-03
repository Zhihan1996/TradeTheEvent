# Tool for News Scraping and Interactive News Analysis

We provide tools that:

1. Scrapes [Reuters](https://www.reuters.com/) to download company-specific news articles 
2. Interactively analyze a news article's influence to the market based on web scraping, event detection and sentiment analysis.





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

Please download the model [here](https://drive.google.com/file/d/1v5Gk9zAADZ-f-3nTpt0YZBeD4juQGHLB/view?usp=sharing) and copy the zip file it into `TradeTheEvent/model`.

```
cd ../model
unzip model_seed24.zip
```

