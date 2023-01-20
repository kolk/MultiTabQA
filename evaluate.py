from rouge_score import rouge_scorer, scoring
from evaluate import load as evaluate_load
import re
from collections import Counter
from datasets import load_metric
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoConfig, AutoTokenizer
from transformers.models.bart.modeling_bart import shift_tokens_right
from datasets import load_dataset, load_from_disk
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoConfig
from transformers.models.auto import AutoModelForSeq2SeqLM
import glob
from collections import defaultdict
import argparse
import torch
from torch.utils.data import random_split
from datasets import load_from_disk
from rouge_score import rouge_scorer, scoring
from evaluate import load as evaluate_load
from SpiderDataset import SpiderDataset, SpiderProcessor
from utility import clean_query

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, help="name of dataset to evaluate on [atis_test, geo_test, spider_nq, spider_sql]")
parser.add_argument("--pretrained_model_name", type=str, help="name of model to evaluate on")
parser.add_argument("--batch_size", type=str, help="batch size")
args = parser.parse_args()
dataset = args.dataset_name

def get_rows_columns_cells(line):
  line=line.lower()
  line=line.split("col :")[1].strip()
  lines=re.split("\s+row\s+[0-9]+\s+:\s+",line)
  rows=[" | ".join([cell.strip() for cell in row.split("|")]) for row in lines[1:]]
  cells=[cell.strip() for row in lines[1:] for cell in row.split("|")]
  columns=[" | ".join([elem.strip() for elem in elems]) for elems in list(zip(*[row.split(" | ") for row in lines]))]
  return rows,columns,cells


def get_correct_total_prediction(target_str,pred_str):
  target_rows,target_columns,target_cells=get_rows_columns_cells(target_str)
  prediction_rows,prediction_columns,prediction_cells=get_rows_columns_cells(pred_str)
  common_rows = Counter(target_rows) & Counter(prediction_rows)
  common_rows = list(common_rows.elements())
  common_columns = Counter(target_columns) & Counter(prediction_columns)
  common_columns = list(common_columns.elements())
  common_cells = Counter(target_cells) & Counter(prediction_cells)
  common_cells = list(common_cells.elements())
  return {"target_rows":target_rows,
          "target_columns":target_columns,
          "target_cells":target_cells,
          "pred_rows":prediction_rows,
          "pred_columns":prediction_columns,
          "pred_cells":prediction_cells,
          "correct_rows":common_rows,
          "correct_columns":common_columns,
          "correct_cells":common_cells}

batch_size = args.batch_size
model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
use_cuda = True 
device = torch.device("cuda")
model.to(device).eval()
outputs = defaultdict(list)
config = AutoConfig.from_pretrained(pretrained_model_name)


if dataset.lower() == "spider_sql":
    print('Loading bart-base tokenized test dataset for Spider ')
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    config = AutoConfig.from_pretrained(pretrained_model_name)
    test_dataset = load_from_disk("data/spider/spider_sql_valid.hf")
    print(f"Evaluating on {len(test_dataset)} samples")
    test_processor = SpiderProcessor(test_dataset=test_dataset,
                                          batch_size=batch_size,
                                          decoder_max_length=1024,
                                          tokenizer=tokenizer,
                                          decoder_start_token_id=config.decoder_start_token_id,
                                          is_test=True)
elif dataset.lower() == "spider_nq":
    print('Loading bart-base tokenized test dataset for Spider Natural Questions')
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    config = AutoConfig.from_pretrained(pretrained_model_name)
    test_dataset = load_from_disk("data/spider/tokenized_spider_nq_valid_with_answer.hf")
    print(f"Evaluating on {len(test_dataset)} samples")
    test_processor = SpiderProcessor(test_dataset=test_dataset,
                                          batch_size=batch_size,
                                          decoder_max_length=1024,
                                          tokenizer=tokenizer,
                                          decoder_start_token_id=config.decoder_start_token_id,
                                          is_test=True)

elif dataset.lower() in "atis_test":
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base") 
    config = AutoConfig.from_pretrained(pretrained_model_name)
    test_dataset = load_from_disk("data/atis/atis_nq_test_with_answer.hf")
    test_processor = SpiderProcessor(test_dataset=test_dataset,
                                          batch_size=batch_size,
                                          decoder_max_length=1024,
                                          tokenizer=tokenizer,
                                          decoder_start_token_id=config.decoder_start_token_id,
                                          is_test=True)
elif dataset.lower() in "geo_test":
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base") 
    config = AutoConfig.from_pretrained(pretrained_model_name)
    test_dataset = load_from_disk("data/geoquery/geoquery_nq_test_with_answer.hf"")
    test_processor = SpiderProcessor(test_dataset=test_dataset,
                                          batch_size=batch_size,
                                          decoder_max_length=1024,
                                          tokenizer=tokenizer,
                                          decoder_start_token_id=config.decoder_start_token_id,
                                          is_test=True)


total_columns_in_dataset = 0
total_rows_in_dataset = 0
total_cells_in_dataset = 0
total_correct_rows = 0
total_correct_columns = 0
total_correct_cells = 0
total_prediced_rows_in_dataset = 0
total_predicted_columns_in_dataset = 0
total_predicted_cells_in_dataset = 0
rouge_types = ["rouge1", "rouge2", "rougeL"]
scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
exact_match_metric = evaluate_load("exact_match")
aggregator = scoring.BootstrapAggregator()
aggregator_em = scoring.BootstrapAggregator()
print('Starting Inference')
predictions, references =[],[]
for i, batch in enumerate(test_processor.test_generator):
    batch_sz = len(batch["input_ids"])
    question = [tokenizer.decode(samp.to("cuda"), skip_special_tokens=True, clean_up_tokenizatimodon_spaces=False) for
                samp in batch["input_ids"]]
    prediction = model.generate(batch["input_ids"].to("cuda"), decoder_start_token_id=config.decoder_start_token_id,
                                eos_token_id=model.config.eos_token_id,
                                length_penalty=0.5, num_beams=5, return_dict_in_generate=True,
                                output_scores=True)
    seq_len = prediction["sequences"].shape[1]
    answer = [tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True) for pred in
              prediction['sequences']]
    target = [tokenizer.decode(samp, skip_special_tokens=True, clean_up_tokenization_spaces=False) for samp in
              batch["labels"]]
    assert len(target) == len(answer)
    predictions.extend(prediction)
    references.extend(target)
    em_results = exact_match_metric.compute(predictions=answer, references=target)
    for ques, tgt, ans in zip(question, target, answer):
        score = scorer.score(tgt.strip(), ans.strip())
        em_score = exact_match_metric.compute(predictions=[ans.strip()], references=[tgt.strip()])
        aggregator.add_scores(score)
        aggregator_em.add_scores(em_score)

        print("question:", ques)
        print("target:", tgt)
        print("prediction:", ans)
        print()
        statistics = get_correct_total_prediction(tgt.strip().lower(), ans.strip().lower())
        total_columns_in_dataset += len(statistics['target_columns'])
        total_rows_in_dataset += len(statistics['target_rows'])
        total_cells_in_dataset += len(statistics['target_cells'])
        total_correct_columns += len(statistics['correct_columns'])
        total_correct_rows += len(statistics['correct_rows'])
        total_correct_cells += len(statistics['correct_cells'])
        total_prediced_rows_in_dataset += len(statistics['pred_rows'])
        total_predicted_columns_in_dataset += len(statistics['pred_columns'])
        total_predicted_cells_in_dataset += len(statistics['pred_cells'])

em_result = aggregator_em.aggregate()
print(f"exact_match: {round(em_result['exact_match'].mid,4)}")

row_precision = total_correct_rows / total_prediced_rows_in_dataset
row_recall = total_correct_rows / total_rows_in_dataset
print(f"row precision {row_precision}")
print(f"row recall {row_recall}")
print(f"row F1 {(2*row_precision*row_recall)/(row_precision+row_recall)}")
print()
column_precision = total_correct_columns / total_predicted_columns_in_dataset
column_recall = total_correct_columns / total_columns_in_dataset
print(f"column_precision {column_precision}")
print(f"column_recall {column_recall}")
print(f"column F1 {(2*column_precision*column_recall)/(column_precision+column_recall)}")
print()
cell_precision = total_correct_cells / total_predicted_cells_in_dataset
cell_recall = total_correct_cells / total_cells_in_dataset
print(f"cell_precision {cell_precision}")
print(f"cell_recall {cell_recall}")
print(f"cell F1 {(2*cell_precision*cell_recall)/(cell_precision+cell_recall)}")

