# MultiTabQA: Generating Tabular Answers for Multi-Table Question Answering

Finetuning datasets present in data directory:

Datasets present in data directory:
1. geoquery: Natural questions, context tables, and target table of GeoQuery for multi-table QA
2. atis:  Natural questions, context tables, and target table  of Atis for multi-table QA
3. spider: Natural questions, context tables, and target table  of spider for multi-table QA

+ **keys in datasets**:  
    + 'source': flattened input sequecne comprising of natural question and context input tables as concatenated string
    + 'target': flattened target table

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
