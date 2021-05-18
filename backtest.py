import os
import json
import time
import pickle
import numpy as np
from dateutil import parser
from datetime import timedelta
from copy import deepcopy
from nltk.sentiment.vader import SentimentIntensityAnalyzer

index2event = {
    '0': 'Acquisitions',
    '1': 'Clinical Trials',
    '2': 'Dividend Cut',
    '3': 'Dividend Increase',
    '4': 'Guidance Change',
    '5': 'New Contract',
    '6': 'Regular Dividend',
    '7': 'Reverse Stock Split',
    '8': 'Special Dividend',
    '9': 'Stock Repurchase',
    '10': 'Stock Split',
    '11': 'NoEvent',
}

event2index = {v: k for k, v in index2event.items()}
NUM_EVENTS = len(event2index) - 1
NOEVENT_ID = int(event2index['NoEvent'])

IS_POSITIVE = {
    'Acquisitions': True,
    'Clinical Trials': True,
    'Dividend Cut': False,
    'Dividend Increase': True,
    'Guidance Change': True,
    'New Contract': True,
    'Regular Dividend': True,
    'Reverse Stock Split': False,
    'Special Dividend': True,
    'Stock Repurchase': True,
    'Stock Split': True,
    'Sentiment': True,
}


def load_trading_dates():
    with open("data/all_trading_dates.json", "r") as f:
        trading_dates = json.load(f)

    return trading_dates


def load_ticker2comp():
    with open("data/generic.pickle", "rb") as f:
        ticker_info = pickle.load(f)
    ticker2comp = dict()
    for item in ticker_info.symbols.keys():
        if ticker_info.symbols[item].exchangeDisplay in ['NASDAQ', 'NYSE']:
            ticker2comp[ticker_info.symbols[item].ticker] = ticker_info.symbols[item].name

    with open('data/ticker2com.json', 'r') as f:
        old_ticker2comp = json.load(f)
    for key in old_ticker2comp.keys():
        if key not in ticker2comp.keys():
            ticker2comp[key] = old_ticker2comp[key]

    comp2ticker = {}
    for item in ticker2comp.items():
        if item[1] not in comp2ticker.keys():
            comp2ticker[item[1]] = [item[0]]
        else:
            comp2ticker[item[1]].append(item[0])

    for item in list(comp2ticker.keys()):
        clean_item = item.replace('Inc.', '').replace('Inc', '').replace('Ltd.', '').replace('Ltd', '').replace(',', '') \
            .replace('.', '').replace(' plc', '').replace('Corporation', '').replace(' Plc', '').replace(' ltd', '') \
            .replace(' inc', '')
        clean_item = ' '.join(clean_item.split())
        if clean_item != item and len(clean_item) >= 5:
            comp2ticker[clean_item] = comp2ticker[item]

    return ticker2comp, comp2ticker


def load_evaluation_news(data_dir):
    print("Loading data from {}".format(data_dir))

    with open(data_dir, "r") as f:
        evaluation_news = json.load(f)

    return evaluation_news


def get_positive_for_keyword(evaluation_news):
    print('Finding trading signals for keyword matching')
    all_positive = {}
    for label in range(NUM_EVENTS):
        all_positive[index2event[str(label)]] = []

    for index, item in enumerate(evaluation_news):
        text = (item['title'] + " " + item['text']).lower()
        if 'acquire' in text or 'acquisition' in text or 'merge' in text:
            all_positive[index2event[str(0)]].append(index)
        elif 'clinic' in text or 'fda ' in text:
            all_positive[index2event[str(1)]].append(index)
        elif 'dividend' in text and 'cut' in text:
            all_positive[index2event[str(2)]].append(index)
        elif 'dividend' in text and 'increase' in text:
            all_positive[index2event[str(3)]].append(index)
        elif 'guidance' in text or 'outlook' in text:
            all_positive[index2event[str(4)]].append(index)
        elif 'contract' in text or 'award' in text:
            all_positive[index2event[str(5)]].append(index)
        elif 'dividend' in text and 'cut' not in text and 'special' not in text and 'increase' not in text:
            all_positive[index2event[str(6)]].append(index)
        elif 'reverse stock split' in text:
            all_positive[index2event[str(7)]].append(index)
        elif 'dividend' in text and 'special' in text:
            all_positive[index2event[str(8)]].append(index)
        elif 'buyback' in text or 'repurchase' in text:
            all_positive[index2event[str(9)]].append(index)
        elif 'stock split' in text and 'reverse' not in text:
            all_positive[index2event[str(10)]].append(index)

    count = 0
    for key in all_positive.keys():
        count += len(all_positive[key])

    print('Find {} trading signals with keyword matching'.format(count))

    return all_positive


def get_positive_for_vader_sentiment(evaluation_news, threshold=0.2, save_dir='data/vader_scores.json'):
    print('Finding trading signals for vader sentiment')
    all_positive = {'Sentiment': []}
    if os.path.exists(save_dir):
        print("Loading cached vader sentiment scores at {}".format(save_dir))
        with open(save_dir, "r") as f:
            vader_scores = json.load(f)

        for index, score in enumerate(vader_scores):
            positive = score > threshold
            if positive:
                all_positive['Sentiment'].append(index)

    else:
        sid = SentimentIntensityAnalyzer()
        vader_scores = []
        for index, item in enumerate(evaluation_news):
            if index > 0 and index % 10000 == 0:
                print("Finished: {}".format(index))

            text = item['title'] + " " + item["text"]
            sentiment = sid.polarity_scores(text)
            score = sentiment['pos']
            vader_scores.append(score)
            positive = score > threshold
            if positive:
                item['vader_sentiment'] = sentiment
                all_positive['Sentiment'].append(index)

        print("Saving vader sentiment scores at {} for potential future usage".format(save_dir))
        with open(save_dir, "w") as f:
            json.dump(vader_scores, f)

    print('Find {} trading signals with vader sentiment'.format(str(len(all_positive['Sentiment']))))

    return all_positive


def get_positive_for_bertsst_sentiment(BERT_SENTIMENT_PRED_DIR, threshold=0.995):
    print('Finding trading signals for BERT SST sentiment')
    all_positive = {'Sentiment': []}
    bert_sentiment = np.load(BERT_SENTIMENT_PRED_DIR)
    for index, x in enumerate(bert_sentiment):
        positive = (np.exp(x) / sum(np.exp(x)))[1]
        if positive > threshold:
            all_positive['Sentiment'].append(index)


    print('Find {} trading signals with BERT SST sentiment'.format(str(len(all_positive['Sentiment']))))

    return all_positive


def get_positive_for_event_sent_split(pred_dir, seq_threshold=0, ignore_event_list=('Clinical Trials', 'Regular Dividend')):
    ignore_list = []
    if len(ignore_event_list) > 0:
        for event in ignore_event_list:
            ignore_list.append(int(event2index[event]))
    
    all_positive = {}
    for label in range(NUM_EVENTS):
        all_positive[index2event[str(label)]] = []
    
    starts_dir = os.path.join(pred_dir, 'starts.json')
    with open(starts_dir,'r') as f:
        starts = json.load(f)
    
    seq_path = os.path.join(pred_dir, 'seq_pred.npy')
    seq_preds = np.load(seq_path)
    seq_preds = seq_preds[1:,:]

    for i in range(len(starts)):
        if i == len(starts) - 1:
            current_seq_preds = seq_preds[starts[i]:len(seq_preds), :]
        else:
            current_seq_preds = seq_preds[starts[i]:starts[i + 1]] 

        tags = set()
        for pred in current_seq_preds:
            pos_label = list(np.where(pred > 0)[0])
            if len(pos_label) > 0 and NOEVENT_ID not in pos_label:
                for tag in pos_label:
                    tags.add(tag)

        for tag in tags:
            if tag not in ignore_list:
                all_positive[index2event[str(tag)]].append(i)
    
    return all_positive



def get_positive_for_event(pred_dir, NER=False, SEQ=False, max_seq_len=256, seq_threshold=0,
                           ignore_event_list=('Clinical Trials', 'Regular Dividend')):
    print('Finding trading signals for events with NER={}, SEQ={}, MAX_SEQUENCE_LEN={}, seq_threshold={}'.format(NER, SEQ, max_seq_len, seq_threshold))
    count = 0

    ignore_list = []
    if len(ignore_event_list) > 0:
        for event in ignore_event_list:
            ignore_list.append(int(event2index[event]))

    all_positive = {}
    for label in range(NUM_EVENTS):
        all_positive[index2event[str(label)]] = []

    ner_path = os.path.join(pred_dir, 'ner_pred.npy')
    seq_path = os.path.join(pred_dir, 'seq_pred.npy')

    if NER:
        ner_preds = np.load(os.path.join(ner_path))
        ner_preds = ner_preds.reshape([-1, max_seq_len])
        ner_preds = ner_preds[:, 1:]

    if SEQ:
        seq_preds = np.load(seq_path)
        seq_preds = seq_preds[1:, :]

    if NER:
        for index, pred in enumerate(ner_preds):
            pred[pred == -100] = NOEVENT_ID
            tags = set(pred)
            if SEQ:
                seq_tags = set(list(np.where(seq_preds[index] > seq_threshold)[0]))
                tags = tags.union(seq_tags)

            if len(tags) == 1:
                continue

            tags.remove(NOEVENT_ID)
            for tag in list(tags):
                tag = int(tag)
                if tag not in ignore_list:
                    # if len(np.where(pred == tag)[0]) < 2:
                    #     continue
                    all_positive[index2event[str(tag)]].append(index)
                    count += 1
    elif SEQ:
        for index, pred in enumerate(seq_preds):
            pos_label = set(list(np.where(pred > seq_threshold)[0]))
            if len(pos_label) == 0:
                pass
            elif NOEVENT_ID not in pos_label:
                for pos in pos_label:
                    if pos not in ignore_list:
                        all_positive[index2event[str(pos)]].append(index)
                        count += 1

    print('Find {} trading signals with events'.format(count))

    return all_positive


def _initialize_dicts_for_data_storage(event_list):
    results = {}
    enriched_event_list = list(event_list) + ['All']

    for start_type in ['open', 'close']:
        results[start_type] = {}

    for start_type in ['open', 'close']:
        for policy in ['end', 'best']:
            results[start_type][policy] = {}

    for start_type in ['open', 'close']:
        for policy in ['end', 'best']:
            for period in ['1', '2', '3']:
                results[start_type][policy][period] = {}

    for start_type in ['open', 'close']:
        for policy in ['end', 'best']:
            for period in ['1', '2', '3']:
                for event in enriched_event_list:
                    results[start_type][policy][period][event] = {}

    for start_type in ['open', 'close']:
        for policy in ['end', 'best']:
            for period in ['1', '2', '3']:
                for event in enriched_event_list:
                    for metric in ['win_count', 'loss_count', 'total_count', 'win_rate', 'win_change_rate',
                                   'loss_change_rate', 'total_change_rate']:
                        results[start_type][policy][period][event][metric] = 0
                    for index in ['win_index', 'loss_index']:
                        results[start_type][policy][period][event][index] = {}

    return results


def _update_backtest_results_with_change_rate(index, change_rate, result_dict):
    result_dict['win_count'] += (change_rate >= 0)
    result_dict['loss_count'] += (change_rate < 0)
    result_dict['total_count'] += 1
    result_dict['win_change_rate'] += (change_rate >= 0) * change_rate
    result_dict['loss_change_rate'] += (change_rate < 0) * change_rate
    result_dict['total_change_rate'] += change_rate

    if change_rate >= 0:
        result_dict['win_index'][index] = change_rate
    else:
        result_dict['loss_index'][index] = change_rate


def backtest(all_positive, evaluation_news, save_dir, buy_pub_same_time=False, stoploss=0.0):
    ticker2comp, _ = load_ticker2comp()
    # trading_dates = load_trading_dates()

    print("Perform backtesting with buy_pub_same_time={}, stoploss={}".format(buy_pub_same_time, stoploss))

    event_list = all_positive.keys()
    results = _initialize_dicts_for_data_storage(event_list)

    for event in event_list:
        positive = IS_POSITIVE[event]
        all_signals = all_positive[event]

        for index in all_signals:
            item = evaluation_news[index]
            labels = item['labels']
            if len(labels) <= 1:
                continue

            if buy_pub_same_time:
                '''
                skip the signal if the stock buy time is different from the article publish time. On the one hand,  
                all the news articles that are not published in the market hours are ignored. On the other hand, 
                since there are missing values in our historical stock data, some market hour signals whose historical
                data are imcomplete are also ignored
                '''
                # if labels['start_time'] != item['pub_time']:
                #     continue
                # else:
                #     start_hour = int(labels['start_time'].split()[1].split(":")[0])
                #     if  9 < start_hour < 16:
                #         continue
                if parser.parse(labels['start_time']) != parser.parse(item['pub_time']):
                    continue
            

            open_price = labels['start_price_open']
            close_price = labels['start_price_close']

            if positive:
                change_rate_close_end_1 = (labels['end_price_1day'] - close_price) / close_price
                change_rate_close_end_2 = (labels['end_price_2day'] - close_price) / close_price
                change_rate_close_end_3 = (labels['end_price_3day'] - close_price) / close_price

                change_rate_open_end_1 = (labels['end_price_1day'] - open_price) / open_price
                change_rate_open_end_2 = (labels['end_price_2day'] - open_price) / open_price
                change_rate_open_end_3 = (labels['end_price_3day'] - open_price) / open_price

                change_rate_close_best_1 = (labels['highest_price_1day'] - close_price) / close_price
                change_rate_close_best_2 = (labels['highest_price_2day'] - close_price) / close_price
                change_rate_close_best_3 = (labels['highest_price_3day'] - close_price) / close_price

                change_rate_open_best_1 = (labels['highest_price_1day'] - open_price) / open_price
                change_rate_open_best_2 = (labels['highest_price_2day'] - open_price) / open_price
                change_rate_open_best_3 = (labels['highest_price_3day'] - open_price) / open_price

            else:
                change_rate_close_end_1 = (close_price - labels['end_price_1day']) / close_price
                change_rate_close_end_2 = (close_price - labels['end_price_2day']) / close_price
                change_rate_close_end_3 = (close_price - labels['end_price_3day']) / close_price

                change_rate_open_end_1 = (open_price - labels['end_price_1day']) / open_price
                change_rate_open_end_2 = (open_price - labels['end_price_2day']) / open_price
                change_rate_open_end_3 = (open_price - labels['end_price_3day']) / open_price

                change_rate_close_best_1 = (close_price - labels['lowest_price_1day']) / close_price
                change_rate_close_best_2 = (close_price - labels['lowest_price_2day']) / close_price
                change_rate_close_best_3 = (close_price - labels['lowest_price_3day']) / close_price

                change_rate_open_best_1 = (open_price - labels['lowest_price_1day']) / open_price
                change_rate_open_best_2 = (open_price - labels['lowest_price_2day']) / open_price
                change_rate_open_best_3 = (open_price - labels['lowest_price_3day']) / open_price
            

            if stoploss:
                if positive:
                    max_loss_close_end_1 = (labels['lowest_price_1day'] - close_price) / close_price
                    max_loss_close_end_2 = (labels['lowest_price_2day'] - close_price) / close_price
                    max_loss_close_end_3 = (labels['lowest_price_3day'] - close_price) / close_price

                    max_loss_open_end_1 = (labels['lowest_price_1day'] - open_price) / open_price
                    max_loss_open_end_2 = (labels['lowest_price_2day'] - open_price) / open_price
                    max_loss_open_end_3 = (labels['lowest_price_3day'] - open_price) / open_price

                else:
                    max_loss_close_end_1 = (close_price - labels['highest_price_1day']) / close_price
                    max_loss_close_end_2 = (close_price - labels['highest_price_2day']) / close_price
                    max_loss_close_end_3 = (close_price - labels['highest_price_3day']) / close_price

                    max_loss_open_end_1 = (open_price - labels['highest_price_1day']) / open_price
                    max_loss_open_end_2 = (open_price - labels['highest_price_2day']) / open_price
                    max_loss_open_end_3 = (open_price - labels['highest_price_3day']) / open_price
                

                change_rate_close_end_1 = -stoploss if max_loss_close_end_1 < -stoploss else change_rate_close_end_1
                change_rate_close_end_2 = -stoploss if max_loss_close_end_2 < -stoploss else change_rate_close_end_2
                change_rate_close_end_3 = -stoploss if max_loss_close_end_3 < -stoploss else change_rate_close_end_3

                change_rate_open_end_1 = -stoploss if max_loss_open_end_1 < -stoploss else change_rate_open_end_1
                change_rate_open_end_2 = -stoploss if max_loss_open_end_2 < -stoploss else change_rate_open_end_2
                change_rate_open_end_3 = -stoploss if max_loss_open_end_3 < -stoploss else change_rate_open_end_3


            _update_backtest_results_with_change_rate(index, change_rate_close_end_1, results['close']['end']['1'][event])
            _update_backtest_results_with_change_rate(index, change_rate_close_end_2, results['close']['end']['2'][event])
            _update_backtest_results_with_change_rate(index, change_rate_close_end_3, results['close']['end']['3'][event])

            _update_backtest_results_with_change_rate(index, change_rate_open_end_1, results['open']['end']['1'][event])
            _update_backtest_results_with_change_rate(index, change_rate_open_end_2, results['open']['end']['2'][event])
            _update_backtest_results_with_change_rate(index, change_rate_open_end_3, results['open']['end']['3'][event])

            _update_backtest_results_with_change_rate(index, change_rate_close_best_1, results['close']['best']['1'][event])
            _update_backtest_results_with_change_rate(index, change_rate_close_best_2, results['close']['best']['2'][event])
            _update_backtest_results_with_change_rate(index, change_rate_close_best_3, results['close']['best']['3'][event])

            _update_backtest_results_with_change_rate(index, change_rate_open_best_1, results['open']['best']['1'][event])
            _update_backtest_results_with_change_rate(index, change_rate_open_best_2, results['open']['best']['2'][event])
            _update_backtest_results_with_change_rate(index, change_rate_open_best_3, results['open']['best']['3'][event])


    for start_type in ['open', 'close']:
        for policy in ['end', 'best']:
            for period in ['1', '2', '3']:
                for event in event_list:
                    for metric in ['win_count', 'loss_count', 'total_count', 'win_rate', 'win_change_rate',
                                   'loss_change_rate', 'total_change_rate']:
                        results[start_type][policy][period]['All'][metric] += results[start_type][policy][period][event][metric]
                    # for index in ['win_index', 'loss_index']:
                    #     results[start_type][policy][period]['All'][index].extend(results[start_type][policy][period][event][index])

    for start_type in ['open', 'close']:
        for policy in ['end', 'best']:
            for period in ['1', '2', '3']:
                for event in (list(event_list) + ['All']):
                    results[start_type][policy][period][event]['win_rate'] = results[start_type][policy][period][event]['win_count'] \
                                                                             / max(1, results[start_type][policy][period][event]['total_count'])
                    results[start_type][policy][period][event]['win_change_rate'] = results[start_type][policy][period][event]['win_change_rate'] \
                                                                                    / max(1, results[start_type][policy][period][event]['win_count'])
                    results[start_type][policy][period][event]['loss_change_rate'] = results[start_type][policy][period][event]['loss_change_rate'] \
                                                                                     / max(1, results[start_type][policy][period][event]['loss_count'])
                    results[start_type][policy][period][event]['total_change_rate'] = results[start_type][policy][period][event]['total_change_rate'] \
                                                                                      / max(1, results[start_type][policy][period][event]['total_count'])


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir = os.path.join(save_dir, "backtest_results.json")

    print("Saving backtesting results in {}".format(save_dir))

    with open(save_dir, "w") as f:
        json.dump(results, f)


    print(results['open']['end']['1']['All'])
    print(results['open']['end']['2']['All'])
    print(results['open']['end']['3']['All'])
    print(results['open']['best']['1']['All'])
    print(results['open']['best']['2']['All'])
    print(results['open']['best']['3']['All'])
    # print(results['close']['end']['1']['All'])
    # print(results['close']['end']['2']['All'])
    # print(results['close']['end']['3']['All'])
    # print(results['close']['best']['1']['All'])
    # print(results['close']['best']['2']['All'])
    # print(results['close']['best']['3']['All'])
    for event in event_list:
        print("{}: {} {}".format(event, results['open']['end']['1'][event]['total_change_rate'], results['open']['end']['1'][event]['total_count']))

    sequential_backtest(results, event_list, evaluation_news)

    return results



def sequential_backtest(results, event_list, evaluation_news, start=10000, each=2000, commission_fee=0.003, market_earning=4404):
    for start_type in ['open', 'close']:
        for policy in ['end', 'best']:
            for period in ['1', '2', '3']:
                all_trades = []
                for event in event_list:
                    positive = IS_POSITIVE[event]
                    for ind in ['win_index', 'loss_index']:
                        for index in results[start_type][policy][period][event][ind]:
                            change_rate = results[start_type][policy][period][event][ind][index]
                            labels = evaluation_news[index]['labels']
                            start_time = labels['start_time']
                            if policy == "end":
                                end_time = labels['end_time_' + period + "day"]
                            else:
                                if positive:
                                    end_time = labels['highest_time_' + period + "day"]
                                else:
                                    end_time = labels['lowest_time_' + period + "day"]
                            all_trades.append([change_rate, parser.parse(start_time), parser.parse(end_time)])
                
                all_trades.sort(key=lambda item: item[1])

                holdings = []
                total = deepcopy(start)
                no_money_count = 0
                trade_count = 0
                for event in all_trades:
                    # sell all the stock that should be sold current time
                    current_datetime = event[1]
                    if len(holdings) > 0:
                        copy_holdings = deepcopy(holdings)
                        for stock in copy_holdings:
                            if current_datetime > stock[1]:
                                holdings.remove(stock)
                                total += stock[0]

                    start_money = each
                    end_money = float(start_money)*(1+event[0])*(1-commission_fee)

                    if total > start_money:
                        total -= start_money
                        holdings.append([end_money, event[2]])
                        trade_count += 1
                    else:
                        no_money_count += 1

                # sell all the remained stocks
                for stock in holdings:
                    total += stock[0]

                earning = total-start-market_earning
                print("Earning of {} {} {} is {}, no money: {}, trade: {}".format(start_type, policy, period, earning, no_money_count, trade_count))



if __name__ == "__main__":
    # evaluation_news = load_evaluation_news("/Users/ZZH/Northwestern/Research/er/data/test/evaluate_news_ACL_large/filtered_ticker_time_price_pr_20200301_2021_0430_busi_20200816_20210506_news.json")
    evaluation_news = load_evaluation_news("data/evaluate_news/filtered_ticker_time_price_fixbestname_pr_20200301_2021_0430_busi_20200816_20210506_news.json")[:]
    # all_positive = get_positive_for_event(pred_dir='acl_preds/stack_4', SEQ=False, NER=True, seq_threshold=5, ignore_event_list=('Regular Dividend',))
    # all_positive = get_positive_for_bertsst_sentiment("acl_preds/bertsst.npy", threshold=0.99)
    all_positive = get_positive_for_event_sent_split(pred_dir='acl_preds/seq_256_sent_split_4', seq_threshold=0)
    results = backtest(all_positive, evaluation_news, save_dir='acl_results/stack_4', buy_pub_same_time=True, stoploss=0.2)