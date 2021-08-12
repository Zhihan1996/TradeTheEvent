from pymongo import MongoClient

import sys
sys.path.append('../')
from utils.tool import download_article, download_article_seperate

DB_NAME = "TTE"
client = MongoClient('localhost', 27017)
db = client[DB_NAME]
collection_news = db["news"]

results = collection_news.find({"text": {"$exists": False}})
url_to_download = []
for res in results:
    url_to_download.append(res)
print("Find {} news articles to download".format(len(url_to_download)))

for i, item in enumerate(url_to_download):
    if i > 0 and i % 20 == 0:
        print("Successfully downloading {} news".format(i))

    title, text, pub_time = download_article_seperate(item['url'])
    item['title'] = title
    item['text'] = text
    item['pub_time'] = pub_time

    condition = {"_id": item["_id"]}
    collection_news.replace_one(condition, item)
