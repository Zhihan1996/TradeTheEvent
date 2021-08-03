from newspaper import Article

def download_article(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.title + " " + article.text

def download_article_seperate(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.title, article.text, article.publish_date