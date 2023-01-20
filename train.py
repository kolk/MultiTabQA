import os
import glob
import transformers
import argparse
import torch
from datasets import load_dataset, load_metric, load_from_disk, concatenate_datasets
from transformers import AutoModelForSeq2SeqLM, AutoConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer
from transformers.optimization import AdamW
from transformers import set_seed
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from multitabqa_processor import MultiTabQAProcessor

os.environ["WANDB_DISABLED"] = "true"
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, help="name of dataset to adapter-tune on")
parser.add_argument("--max_length", default=1024, type=int, help="encoder sequence max length")
parser.add_argument("--decoder_max_length", default=1024, type=int, help="decoder sequence max length")
parser.add_argument("--pretrained_model_name", type=str, default=None, help="prtrained model name")
parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--lr_scheduler", default="polynomial", choices=arg_to_scheduler_choices,
                    metavar=arg_to_scheduler_metavar, type=str, help="Learning rate scheduler", )
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight Decay for AdamW optimizer.")
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--max_grad_norm", default=0.1, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
parser.add_argument("--use_multiprocessing", default=True, type=bool, help="use multiple processes for data loading")
parser.add_argument("--num_train_epochs", default=30, type=int)
parser.add_argument("--train_batch_size", default=4, type=int)
parser.add_argument("--eval_batch_size", default=4, type=int)
parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
parser.add_argument("--eval_gradient_accumulation", default=64, type=int)
parser.add_argument("--adafactor", action="store_true")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cpu", action="store_true", help="train using cpu")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="resume training from a checkpoint")
parser.add_argument("--gradient_checkpointing", action="store_true")
parser.add_argument("--local_rank", type=int, default=-1)


def tokenize_sample(sample):
    from transformers.models.bart.modeling_bart import shift_tokens_right
    config = AutoConfig.from_pretrained(tokenizer.name_or_path)
    input_encoding = tokenizer(sample['source'].strip().lower().replace('"', ''),
                               return_tensors="pt",
                               padding='max_length',
                               max_length=args.max_length,
                               truncation='longest_first',
                               add_special_tokens=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            text=sample['target'].strip().lower().replace('"', ''),
            add_special_tokens=True,
            return_tensors="pt",
            padding='max_length',
            max_length=args.decoder_max_length,
            truncation='longest_first',
        )

    decoder_input_ids = shift_tokens_right(labels['input_ids'], tokenizer.pad_token_id,
                                           config.decoder_start_token_id)
    return {"input_ids": input_encoding["input_ids"],
            "attention_mask": input_encoding["attention_mask"],
            "labels": labels['input_ids'],
            "decoder_input_ids": decoder_input_ids}


args = parser.parse_args()
use_cuda = False if args.cpu else True
device = torch.device("cuda" if use_cuda else "cpu")
seed = args.seed


def model_init():
    set_seed(args.seed)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_name)
    model.config.max_length = args.decoder_max_length
    model = model.to(device)
    return model


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
config = AutoConfig.from_pretrained(args.pretrained_model_name)


def get_dataset(dataset_name):
    # stage 2
    if dataset_name in "spider_sql":
        train_set = load_from_disk("data/spider/sql/spider_sql_train.hf")
        valid_set = load_from_disk("data/spider/sql/spider_sql_valid.hf")
        test_set = None
    # Stage 1
    elif dataset_name == "tapex_pretraining":
        print("Processing Tapex Pretrainng dataset ...")
        train_set = load_from_disk("data/tapex_pretraining/tapex_pretraining_train.hf")
        valid_set = load_from_disk("data/tapex_pretraining/tapex_pretraining_valid.hf")
        test_set = None
    # stage 2 + 3
    elif dataset_name == "multitable_pretraining":
        print("Processing MultiTable Pretraining dataset ...")
        valid_set = load_from_disk("data/raw_data/spider/sql/spider_sql_valid.hf")
        synthetic_train_set = load_from_disk("data/pretraining_synthetic_dataset")
        spider_train_set = load_from_disk("data/spider/sql/spider_sql_train.hf")
        train_set = concatenate_datasets([synthetic_train_set, spider_train_set])
        train_set = train_set.shuffle(seed=args.seed)
        test_set = None
        print(f"Training with {len(train_set)} samples")
    # spider natural question fine-tuning dataset
    elif dataset_name == "spider_nq":
        print("Loading Spider natural questions tokenized with bart-base")
        train_set = load_from_disk("data/spider/natural_questions/spider_nq_train_with_answer.hf")
        valid_set = load_from_disk("data/spider/natural_questions/spider_nq_valid_with_answer.hf")
        train_set = valid_set
        test_set = None
        print(f"Training with {len(train_set)} samples, evaluating with {len(valid_set)} samples")
    # atis natural question finetuning dataset
    elif dataset_name == "atis":
        print("Loading Atis subset natural questions tokenized with bart-base")
        train_set = load_from_disk("data/atis/atis_nq_train_with_answer.hf")
        valid_set = load_from_disk("data/atis/atis_nq_dev_with_answer.hf")
        test_set = load_from_disk("data/atis/atis_nq_test_with_answer.hf")
        print(f"Training with {len(train_set)} samples, evaluating with {len(valid_set)} samples")
    # geoquery natural question finetuning dataset
    elif dataset_name == "geoquery":
        print("Loading Geo natural questions tokenized with bart-base")
        train_set = load_from_disk("data/geoquery/geoquery_nq_train_with_answer.hf")
        valid_set = load_from_disk("data/geoquery/geoquery_nq_dev_with_answer.hf")
        test_set = load_from_disk("data/geoquery/geoquery_nq_test_with_answer.hf")
        print(f"Training with {len(train_set)} samples, evaluating with {len(valid_set)} samples")
    train_set = train_set.map(tokenize_sample)
    valid_set = valid_set.map(tokenize_sample)
    processor = MultiTabQAProcessor(training_dataset=train_set, eval_dataset=valid_set, tokenizer=tokenizer,
                                    decoder_start_token_id=config.decoder_start_token_id,
                                    decoder_max_length=args.decoder_max_length)
    return train_set, valid_set, test_set, processor


train_dataset, valid_dataset, test_dataset, processor = get_dataset(args.dataset_name)
print("#############Data loading done!#############")


def em_metric_builder(tokenizer):
    def compute_em_metrics(pred):
        """utility to compute Exact Match during training."""
        # All special tokens are removed.
        pred_ids, labels_ids = pred.predictions, pred.label_ids
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        em = load_metric("exact_match")
        scores = em.compute(predictions=pred_str, references=label_str, ignore_case=True)
        print(f"Exact Match Scores: {scores}")
        return {
            "exact_match": round(scores['exact_match'], 4),
        }

    return compute_em_metrics


em_metric_fn = em_metric_builder(tokenizer)

train_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    no_cuda=args.cpu,
    fp16=True if use_cuda else False,
    save_strategy="epoch",
    save_total_limit=5,
    logging_steps=100,
    eval_accumulation_steps=args.eval_gradient_accumulation,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_train_epochs,
    seed=seed,
    disable_tqdm=False,
    predict_with_generate=True,
    generation_max_length=args.decoder_max_length,
    generation_num_beams=4,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    dataloader_num_workers=args.num_workers,
    metric_for_best_model="exact_match",
    dataloader_drop_last=True,
    adam_epsilon=args.adam_epsilon,
    weight_decay=args.weight_decay,
    max_grad_norm=args.max_grad_norm,
    lr_scheduler_type=args.lr_scheduler,
    warmup_steps=args.warmup_steps,
    gradient_checkpointing=args.gradient_checkpointing,
    local_rank=args.local_rank,
)

transformers.logging.set_verbosity_info()
trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=processor.collate_tokenized,
    compute_metrics=em_metric_fn,
)

print("Starting Training...")
if args.resume_from_checkpoint:
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
else:
    trainer.train()
trainer.save_state()
