import json
import pickle

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

ticker2comp, _ = load_ticker2comp()
