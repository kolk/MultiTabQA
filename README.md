# MultiTabQA: Generating Tabular Answers for Multi-Table Question Answering
Details of dataset generation and results can be found in our [paper](https://arxiv.org/abs/2305.12820).

Finetuning datasets present in data directory:

Datasets present in data directory:
1. geoquery: Natural questions, context tables, and target table of GeoQuery for multi-table QA
2. atis:  Natural questions, context tables, and target table  of Atis for multi-table QA
3. data.txt: contains url to download all data. The zip file contains Tapex single-table pre-training data, Multi-Table pre-training data, spider dataset where each sample is comprised of the natural question/SQL query, context table/(s), and target table  for multi-table QA.

+ **keys in datasets**:  
    + 'source': flattened input sequence comprising of natural question and context input tables as concatenated string
    + 'target': flattened target table
    
Loading the Spider dataset:
 ```
 from datasets import load_from_disk
 spider_natural_questions_data = load_from_disk(f"data/spider/tokenized_spider_nq_train_with_answer.hf")
 spider_sql_query_data = load_from_disk(f"data/spider/tokenized_spider_sql_train.hf")
 ```

Arguments for pre-training:
```
python train.py --dataset_name "multitab_pretraining" 
                --pretrained_model_name "microsoft/tapex-base" \
                --learning_rate 1e-4 --train_batch_size 4 --eval_batch_size 4 \
                --gradient_accumulation_steps 64 --eval_gradient_accumulation 64 \
                --num_train_epochs 60 --use_multiprocessing False \
                --num_workers 2 --decoder_max_length 1024 \
                --local_rank -1  --seed 47 \ 
                --output_dir "experiments/tapex_base_pretraining"
```


Arguments for fine-tuning:
```
python train.py --dataset_name "spider_nq" 
                --pretrained_model_name "experiments/tapex_base_pretraining" \
                --learning_rate 1e-4 --train_batch_size 4 --eval_batch_size 4 \
                --gradient_accumulation_steps 64 --eval_gradient_accumulation 64 \
                --num_train_epochs 60 --use_multiprocessing False \
                --num_workers 2 --decoder_max_length 1024 \
                --local_rank -1  --seed 47 \ 
                --output_dir "experiments/tapex_base_finetuning_on_spiderNQ"
```

To evaluate:
```
python evaluate.py --batch_size 2 \
                   --pretrained_model_name "experiments/tapex_base_finetuning_on_spiderNQ" \
                   --dataset_name "spider_nq"
```

**Citation**

Please cite our work if you use our code or dataset:
```
@misc{pal2023multitabqa,
      title={MultiTabQA: Generating Tabular Answers for Multi-Table Question Answering}, 
      author={Vaishali Pal and Andrew Yates and Evangelos Kanoulas and Maarten de Rijke},
      year={2023},
      eprint={2305.12820},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
