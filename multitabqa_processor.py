import os
import json
import sqlite3
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


class IndexedRowTableLinearize:
    """
    FORMAT: col: col1 | col2 | col 3 row 1 : val1 | val2 | val3 row 2 : ...
    """

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        assert "header" in table_content and "rows" in table_content, self.PROMPT_MESSAGE
        # process header
        table_names = self.process_name(table_content["name"]) if table_content["name"] else ""
        table_str = table_names + " " + self.process_header(table_content["header"]) + " "

        # process rows
        for i, row_example in enumerate(table_content["rows"]):
            # NOTE: the row should start from row 1 instead of 0
            table_str += self.process_row(row_example, row_index=i + 1) + " "
        return table_str.strip()

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        return "col : " + " | ".join(headers)

    def process_name(self, names: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        return "<table_name> : " + names

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        row_str = ""
        row_cell_values = []
        for cell_value in row:
            if isinstance(cell_value, int) or isinstance(cell_value, float):
                row_cell_values.append(str(cell_value))
            else:
                row_cell_values.append(cell_value)
        row_str += " | ".join(row_cell_values)
        return "row " + str(row_index) + " : " + row_str


class MultiTabQAProcessor:
    def __init__(self,
                 training_dataset: Dataset = None,
                 eval_dataset: Dataset = None,
                 test_dataset: Dataset = None,
                 tokenizer: AutoTokenizer = None,
                 max_length: int = 1024,
                 decoder_start_token_id=0,
                 decoder_max_length: int = 1024,
                 is_test: bool = False,
                 **params):
        """
        Generated tokenized batches for training, evaluation and testing of seq2seq task
        :param training_dataset: tokenized training samples of Spider
        :param eval_dataset: tokenized evaluation samples of Spider
        :param docder_max_length: maximum sequence length of the decoder
        :param tokenizer: tokenizer for tokenizing the samples of dataset
        :param test_dataset: Optional tokenized test samples of Spider
        """
        self.max_length = max_length
        self.decoder_start_token_id = decoder_start_token_id
        self.tokenizer = tokenizer
        self.config = AutoConfig.from_pretrained(self.tokenizer.name_or_path)
        self.decoder_max_length = decoder_max_length
        self.table_linearize = IndexedRowTableLinearize()

        if not is_test:
            self.training_dataset = training_dataset
            self.eval_dataset = eval_dataset

        if is_test:
            self.test_dataset = test_dataset
            self.test_generator = DataLoader(self.test_dataset,
                                             sampler=SequentialSampler(data_source=self.test_dataset),
                                             collate_fn=self.collate_tokenized,
                                             drop_last=True,
                                             **params)
    def prepare_table_query(
            self,
            tables,
            table_names,
            query,
    ):
        """
        This method can be used to linearize a table and add a corresponding query.
        """
        # num_tables = len(tables)
        # lens_tables = [len(table) for table in tables]
        # total_length = sum(lens_tables)
        # calculate the tokens of header, special tokens will only be pre-prepended into question
        # query_tokens = tokenizer(query)#, add_special_tokens=True)
        # used_token_length = len(query_tokens)
        linear_tables = []
        assert len(tables) == len(table_names), "Number of table names and tables must be same!"
        for table, table_name in zip(tables, table_names):
            if not table.empty:
                # step 1: create table dictionary
                table_content = {"name": table_name, "header": list(table.columns),
                                 "rows": [list(row.values) for i, row in table.iterrows()]}

                # step 3: linearize table
                linear_table = self.table_linearize.process_table(table_content)
                linear_tables.append(linear_table)
            else:
                linear_table = ""
                logger.warning(
                    f"You provide an empty table {table_name}"
                    + f"Please carefully check the corresponding table with the query : {query}."
                )

        if query == "":
            logger.warning("You provide nothing to query with respect to the table.")
        # step 4: concatenate query with linear_table
        query = query.replace("<>", "!=")
        separator = " " if query and len(linear_tables) > 0 else ""
        tables_with_separator = " ".join(linear_tables)
        joint_input = (query + separator + tables_with_separator) if query else tables_with_separator
        return joint_input

    def clean_table(self, table):
        table.fillna("", inplace=True)
        if len(table) > 0:
            # rename all duplicate column names
            if len(set(table.columns.to_list())) != len(table.columns):
                column_name_indices = defaultdict(list)
                for i, col_name in enumerate(table.columns):
                    column_name_indices[col_name].append(i)
                new_column_names = [''] * len(table.columns)
                for col_name, indices in column_name_indices.items():
                    if len(indices) > 1:
                        for j, index in enumerate(indices):
                            new_column_names[index] = f"{col_name} -{j + 1}"
                            # table.rename(inplace=True,columns={f'{index+1}col':f"{col_name}-{j+1}"})
                    else:
                        new_column_names[indices[0]] = col_name
                table.columns = new_column_names

            # check all values of table and map them to str
            table = table.applymap(
                lambda x: pd.to_datetime(x, infer_datetime_format=True).strftime('%Y-%m-%d %H:%M:%S')
                if isinstance(x, pd.Timestamp) or isinstance(x, np.datetime64) else x)
            table = table.applymap(lambda x: str(x))
            table.columns = table.columns.astype(str)
        return table

    def flatten_dataset(self, sample):
        if "tables" not in sample.keys() and "db_name" in sample.keys():
            database_path = 'data/database'
            con = sqlite3.connect(os.path.join(database_path, line["db_name"], line["db_name"] + '.sqlite'))
            encoding = "latin1"
            con.text_factory = lambda x: str(x, encoding)
            sample["tables"] = [pd.read_sql_query(f'SELECT * FROM {table_name}', con) for table_name in line["tables"]]
        return self.preprocess_sample(
            {"query": saple["question"], "table_names": sample["table_names"], "tables": sample["tables"],
             "answer": sample["answer"]})

    def preprocess_sample(self, sample):
        tables = sample["tables"]
        table_names = sample["table_names"]
        query = sample["query"]
        answer = sample["answer"]
        do_lower_case = False
        if answer is None:
            logger.warning(f"Answer None!!")
        tables = [clean_table(table) if isinstance(table, pd.core.frame.DataFrame) else clean_table(
            pd.read_json(table, orient='split')) for table in tables]
        text = prepare_table_query(
            tables, table_names, query, answer, truncation_strategy=None)  # "drop_rows_to_fit")
        answer = clean_table(answer) if isinstance(answer, pd.core.frame.DataFrame) else clean_table(
            pd.read_json(answer, orient='split'))
        if not answer.empty:
            # step 1: create table dictionary
            table_content = {"name": None, "header": list(answer.columns),
                             "rows": [list(row.values) for i, row in answer.iterrows()]}
            # step 2: linearize table
            answer_text = table_linearize.process_table(table_content)
        else:
            answer_text = ""
            if answer_text == "":
                logger.warning(
                    "Your answer is an empty table. "
                    + f"Please carefully check the answer table with the query : {query}."
                )
        if do_lower_case:
            answer_text = answer_text.lower()
            text = text.lower()
        return {"source": text, "target": answer_text}

    def collate_tokenized(self, batch):
        """
        Generates tokenized batches
        """
        batch_input_ids, batch_attention_mask, batch_labels, batch_decoder_input_ids = [], [], [], []
        for sample in batch:
            # print("Sample keys", sample.keys())
            batch_input_ids.append(torch.tensor(sample['input_ids']))
            batch_attention_mask.append(torch.tensor(sample['attention_mask']))
            batch_labels.append(torch.tensor(sample['labels']))
            batch_decoder_input_ids.append(torch.tensor(sample["decoder_input_ids"]))

        return {"input_ids": torch.stack(batch_input_ids).squeeze(),
                "attention_mask": torch.stack(batch_attention_mask).squeeze(),
                "labels": torch.stack(batch_labels).squeeze(),
                "decoder_input_ids": torch.stack(
                    batch_decoder_input_ids).squeeze()}
