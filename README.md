# TradeTheEvent

This repository contains

1. Official implementation of the following paper:

Zhihan Zhou, Liqian Ma, Han Liu. [Trade the Event: Corporate Events Detection for News-Based Event-Driven Trading](https://aclanthology.org/2021.findings-acl.186.pdf). In Findings of ACL 2021.

2. The EDT dataset for corporate event detection and news-based stock prediction benchmark.
3. Tools for scraping all the news articles from [Reuters](https://www.reuters.com/) back to 2017 and interactively analyze any online news articles (Including event detection and sentiment analysis).



If you use any of them in your work, please [cite](#reference) our paper.



## Dataset

We release the EDT dataset for corporate event detection and news-based stock prediction benchmark. Please refer to the `data/` folder for access and detailed information about this dataset. The dataset can be found [here](https://drive.google.com/drive/folders/1xKjd9hzA8UTn2DXVIYYnX5TngNAMom19?usp=sharing).



## Tool

We shall the tool for scraping news article and interactively analyze news articles on how it may influence the stock market. Please refer to the `tool/` folder for access and detailed information about our tool.



## Environment

We recommand to use a Python virtual environment with Python >= 3.6. The requirements can be installed with:

```
git clone https://github.com/Zhihan1996/TradeTheEvent
cd TradeTheEvent
pip install -r requirements.txt
```







## Official Implementation

### 0. Download Data

Please download data from [here](https://drive.google.com/drive/folders/1xKjd9hzA8UTn2DXVIYYnX5TngNAMom19?usp=sharing) and put all the three data folder into a folder. Suppose the data dir is `/home/user/data`, set the environment variable as:

```
export DIR_TO_EDT_DATASET=/home/user/data
```





### 1. Domain Adaptation

```
export TRAIN_FILE=DIR_TO_EDT_DATASET/Domain_adaptation/train.txt
export TEST_FILE=DIR_TO_EDT_DATASET/Domain_adaptation/dev.txt
export ADA_MODEL_DIR=models/bert_bc_adapted

python run_domain_adapt.py \
    --output_dir=ADA_MODEL_DIR \
    --model_type=bert \
    --model_name_or_path=bert-base-cased \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --warmup_steps 500 \
    --learning_rate 3e-5 \
    --evaluate_during_training \
    --eval_steps 500 \
    --num_train_epochs 20 \
    --max_steps 10000
```





### 2. Train the Model for Event Detection

```
export ADA_MODEL_DIR=models/bert_bc_adapted
export DATA_DIR=DIR_TO_EDT_DATASET/Event_detection
export OUTPUT_DIR=models/bilevel

python run_event.py \
    --TASK bilevel \
    --data_dir DATA_DIR \
    --epoch 5 \
    --model_type ADA_MODEL_DIR \
    --output_dir OUTPUT_DIR \
    --bert_lr 5e-5  \
    --per_gpu_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --max_seq_length 256 
```



### 3. Detect Events on the Evaluation News

```
export OUTPUT_DIR=models/bilevel
export PRED_DIR=preds/bilevel
export DATA_DIR=DIR_TO_EDT_DATASET/Trading_benchmark/evaluate_news.json


python run_event.py \
    --TASK bilevel \
    --output_dir OUTPUT_DIR \
    --per_gpu_batch_size 64 \
    --predict_dir PRED_DIR \
    --do_predict \
    --max_seq_length 256 \
    --data_dir DATA_DIR

```





### 4. Backtest

```
export PRED_DIR=preds/bilevel
export DATA_DIR=DIR_TO_EDT_DATASET/Trading_benchmark/evaluate_news.json
export RESULTS_DIR=results/

python run_backtest.py \
    --evaluate_news_dir DATA_DIR \
    --pred_dir PRED_DIR \
    --save_dir RESULTS_DIR \
    --model_type bilevel \
    --seq_threshold 5 \
    --stoploss 0.2 \
    --buy_pub_same_time
```

The final backtest results will appear at the `RESULTS_DIR`.







## Reference

If you use the codes, tool, or the dataset, please kindly cite our paper.

```
@inproceedings{zhou-etal-2021-trade,
    title = "Trade the Event: Corporate Events Detection for News-Based Event-Driven Trading",
    author = "Zhou, Zhihan  and
      Ma, Liqian  and
      Liu, Han",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.186",
    doi = "10.18653/v1/2021.findings-acl.186",
    pages = "2114--2124",
}
```