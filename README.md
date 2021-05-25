# TradeTheEvent

This repository is the official implementation of the following paper:

Zhihan Zhou, Liqian Ma, Han Liu. "Trade the Event: Corporate Events Detection for News-Based Event-Driven Trading." In Findings of ACL 2021.

Please see our paper for more details.



## Data

We release the EDT dataset for corporate event detection and news-based stock prediction benchmark. Please refer to the `data/` folder for access and detailed information about this dataset. The dataset can be found [here](https://drive.google.com/drive/folders/1xKjd9hzA8UTn2DXVIYYnX5TngNAMom19?usp=sharing).



## Environment

We recommand to use a Python virtual environment with Python >= 3.6. The requirements can be installed with:

```
git clone https://github.com/Zhihan1996/TradeTheEvent
cd TradeTheEvent
pip install -r requirements.txt
```



## Run and Backtest Our Model

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

If you use the codes or dataset in your work, please kindly cite our paper.

