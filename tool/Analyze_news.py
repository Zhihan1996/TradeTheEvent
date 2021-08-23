import os
import logging
import torch
import numpy as np
from transformers import BertConfig, BertTokenizerFast
from termcolor import colored

import sys
sys.path.append('../')
from utils.tool import download_article
from utils.model import BertForBilevelClassification
from run_backtest import _get_positive_for_event_single
from nltk.sentiment.vader import SentimentIntensityAnalyzer


MODEL_DIR = '../models/model_seed24'
MAX_LEN = 256
Event2color = {
    'Acquisitions': ["31m", "41m"],
    'Clinical Trials': ["35m", "45m"],
    'Dividend Cut': ["31m", "41m"],
    'Dividend Increase': ["31m", "41m"],
    'Guidance Change': ["32m", "42m"],
    'New Contract': ["32m", "42m"],
    'Regular Dividend': ["34m", "44m"],
    'Reverse Stock Split': ["36m", "46m"],
    'Special Dividend': ["35m", "45m"],
    'Stock Repurchase': ["33m", "43m"],
    'Stock Split': ["36m", "46m"],
}


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


logger.info("Loading Model From: {}".format(MODEL_DIR))
config = BertConfig.from_pretrained(MODEL_DIR)
config.num_labels = 12
config.max_seq_length = MAX_LEN
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
model = BertForBilevelClassification.from_pretrained(MODEL_DIR, config=config)
model.eval()

sid = SentimentIntensityAnalyzer()

def get_continuous(array, event=""):
    if len(array) == 1:
        return [[array[0], 1, event]]

    res = []
    last = array[0]
    last_start = array[0]
    last_len = 1
    for cur in array[1:]:
        if cur - last > 1:
            res.append([last_start, last_len, event])
            last_start = cur
            last_len = 0
        last_len += 1
        last = cur

    if last_len > 0:
        res.append([last_start, last_len, event])
    
    # smooth
    if len(res) > 1:
        smoothed_res = []
        smoothed_res.append(res[0])
        last_end = res[0][0] + res[0][1]
        for span in res[1:]:
            if (span[0] - last_end) < 2:
                smoothed_res[-1][1] += 1 + span[1]
            else:
                smoothed_res.append(span)
            last_end = span[1]
        return smoothed_res
    
    return res


while True:
    try:
        with torch.no_grad():
            logger.info("Please type the link to the news article")
            url = input("Link: ")

            if len(url) < 5:
                continue

            logger.info("Downloading and parsing article")
            article = download_article(url)

            # predict events
            logger.info("Processing article")
            model_input = tokenizer.encode_plus(article, add_special_tokens=True, max_length=256, truncation=True, padding=True)["input_ids"]
            model_input = torch.tensor(model_input, dtype=torch.long).unsqueeze(0)
            output = model(model_input)[0].squeeze(0).cpu().numpy()
            pred = np.argmax(output, axis=1)
            results = _get_positive_for_event_single(pred)

            # calculate sentiment with vader
            sentiment = sid.polarity_scores(article)
            score = sentiment['compound']
            if score > 0.2:
                logger.info(colored("Article sentiment is Positive", "green"))
            elif score < -0.2:
                logger.info(colored("Article sentiment is Negative", "red"))
            else:
                logger.info(colored("Article sentiment is Neutral", "yellow"))
            logger.info("Sentiment score is: {}".format(score))
            print("\n")
            
            # for print
            model_input = model_input.squeeze(0).cpu().numpy()
            entire_ids = list(range(MAX_LEN))


            if len(results) > 0:
                event_spans = []
                for event in results:
                    evidence = results[event]
                    cur_span = get_continuous(evidence, event)
                    event_spans.extend(cur_span)
                event_spans.sort()

                text = ""
                last = 0
                for i, s in enumerate(event_spans):
                    if s[0] > last:
                        text += " " + tokenizer.decode(model_input[entire_ids[last : s[0]]])
                    text += " \033[" + Event2color[s[2]][1] + tokenizer.decode(model_input[entire_ids[s[0]: s[0] + s[1]]]) + "\033[0m"
                    last = s[0]+s[1]
                if last < MAX_LEN:
                    text += " " + tokenizer.decode(model_input[entire_ids[last:]])
                logger.info(("{} corporate events are detected".format(len(results))))
                for event in results:
                    logger.info(" \033[" + Event2color[event][0] + event + "\033[0m")
            else:
                text = tokenizer.decode(model_input[entire_ids])
                logger.info("No event is detected")

            logger.info(text.replace("[CLS] ", "").replace("[SEP]", "") + "......")
            print("\n\n\n")


    except KeyboardInterrupt:
        print("\n")
        print("Ending program")
        break

