import random
import string
import re
import numpy as np
import pandas as pd
from collections import defaultdict

agg_set = ['SUM', 'AVG', 'COUNT', 'MIN', "MAX"]
numeric_relational_op = ["<", ">", "=", "!=", "<=", ">="]
string_relational_op = ["=", "!="]

def clean_query(query):
    new_query = []
    query_words = query.split()
    skip_next = False
    final_word = query_words[-1]
    for i in range(len(query_words)-1):
        if skip_next:
            skip_next = False
            continue
        if ((query_words[i] == '!' and query_words[i+1] == '=') or
            (query_words[i] == '>' and query_words[i+1] == '=') or
            (query_words[i] == '<' and query_words[i+1] == '=')
        ):
            new_query.append(query_words[i]+query_words[i+1])
            skip_next = True
        else:
            new_query.append(query_words[i])
    new_query.append(final_word)
    return " ".join(new_query)

def get_select_indices(query):
    pattern = re.compile('[S|s][E|e][L|l][E|e][C|c][T|t] ')
    matches = re.finditer(pattern, query)
    indices = [match.span() for match in matches]
    return indices

def get_from_indices(query):
    pattern = re.compile('[F|f][R|r][O|o][M|m]')
    matches = re.finditer(pattern, query)
    indices = [match.span() for match in matches]
    return indices

def get_where_conditions_from_2_tables(total_columns, table1, table2, with_aggs=False):    
    number_of_where_conditions = random.randint(1, len(total_columns))
    where_conditional_columns = random.sample(total_columns, k=number_of_where_conditions)
    where_conditional_values = [random.choice(table1[col].to_list()) if col in table1.columns else random.choice(table2[col].to_list()) for col in where_conditional_columns]
    where_relational_op = [random.choice(numeric_relational_op) if isinstance(cond_val, int) or isinstance(cond_val, float) else random.choice(string_relational_op) for cond_val in where_conditional_values]
    where_condition_and = " AND ".join([f"{where_col} {where_rel_op} {where_val}" for where_col, where_rel_op, where_val in zip(where_conditional_columns, where_relational_op, where_conditional_values)])
    where_condition_or = " OR ".join([f"{where_col} {where_rel_op} {where_val}" for where_col, where_rel_op, where_val in zip(where_conditional_columns, where_relational_op, where_conditional_values)])
    where_conditions = []
    where_conditions_with_aggs = []
    where_aggs = [random.choice(agg_set) if random.random() > 0.5 and (isinstance(cond_val, int) or isinstance(cond_val, float)) else "" for cond_val in where_conditional_values]

    for i, (where_col, where_rel_op, where_val, where_agg) in enumerate(zip(where_conditional_columns, where_relational_op, where_conditional_values, where_aggs)):
            if isinstance(where_val, int) or isinstance(where_val, float) or (type(where_val) == np.int_) or (type(where_val) == np.float_):
                where_conditions.append(f'"{where_col}" {where_rel_op} {where_val}')
                where_conditions.append(random.choice(['AND', 'OR']))
                
                if where_agg:
                    if where_agg == "COUNT":
                        where_val = random.randint(1, len(table1[where_col])) if where_col in table1.columns else random.randint(1, len(table2[where_col]))
                    where_conditions_with_aggs.append(f'{where_agg}({where_col}) {where_rel_op} {where_val}')
                else:
                    where_conditions_with_aggs.append(f'{where_col} {where_rel_op} {where_val}')

                where_conditions_with_aggs.append(random.choice(['AND', 'OR']))
            else:
                where_conditions.append(f'{where_col} {where_rel_op} "{where_val}"')
                where_conditions.append(random.choice(['AND', 'OR']))
                
                if where_agg:
                    if where_agg == "COUNT":
                        where_val = random.randint(1, len(table1[where_col])) if where_col in table1.columns else random.randint(1, len(table2[where_col]))
                    where_conditions_with_aggs.append(f'{where_agg}({where_col}) {where_rel_op} "{where_val}"')
                else:
                    where_conditions_with_aggs.append(f'{where_col} {where_rel_op} "{where_val}"')
                where_conditions_with_aggs.append(random.choice(['AND', 'OR']))
    where_conditions = " ".join(where_conditions[:-1])
    where_conditions_with_aggs = " ".join(where_conditions_with_aggs[:-1])
    if with_aggs:
        return where_conditions_with_aggs
    return where_conditions

def get_where_conditions_from_1_table(table, with_aggs=False):
    total_columns = table.columns    
    number_of_where_conditions = random.randint(1, len(total_columns))
    where_conditional_columns = random.sample(total_columns.to_list(), k=number_of_where_conditions)
    where_conditional_values = [random.choice(table[col].to_list()) for col in where_conditional_columns]
    where_relational_op = []
    for i, cond_val in enumerate(where_conditional_values):
        if isinstance(cond_val, int) or isinstance(cond_val, float):
            where_relational_op.append(random.choice(numeric_relational_op))
        else:
            #if random.random() > 0.5:
            where_relational_op.append(random.choice(numeric_relational_op))
            #else:
            #    where_relational_op.append(random.choice(string_relational_op))
                
    where_aggs = []
    for i, cond_val in enumerate(where_conditional_values):
        if isinstance(cond_val, int) or isinstance(cond_val, float):
            if random.random() > 0.5:
                where_aggs.append(random.choice(agg_set))
            else:
                where_aggs.append("")
        else:
            #if where_relational_op[i] in numeric_relational_op:
            #    where_aggs.append('COUNT')
            #else:
            where_aggs.append("")
    
    where_conditions = []
    where_conditions_with_aggs = []
    for i, (where_col, where_rel_op, where_val, where_agg) in enumerate(zip(where_conditional_columns, where_relational_op, where_conditional_values, where_aggs)):
            if isinstance(where_val, int) or isinstance(where_val, float) or (type(where_val) == np.int_) or (type(where_val) == np.float_):
                where_conditions.append(f'"{where_col}" {where_rel_op} {where_val}')
                where_conditions.append(random.choice(['AND', 'OR']))
                
                if where_agg:
                    if where_agg == 'COUNT':
                        where_val = random.randint(1, len(table[where_conditional_columns[i]]))
                    where_conditions_with_aggs.append(f'{where_agg}({where_col}) {where_rel_op} {where_val}')
                else:
                    where_conditions_with_aggs.append(f'{where_col} {where_rel_op} {where_val}')

                where_conditions_with_aggs.append(random.choice(['AND', 'OR']))
            else:
                if where_rel_op not in string_relational_op:
                    temp_rel_op = random.choice(string_relational_op)
                    where_conditions.append(f'{where_col} {temp_rel_op} "{where_val}"')
                else:
                    where_conditions.append(f'{where_col} {where_rel_op} "{where_val}"')
                where_conditions.append(random.choice(['AND', 'OR']))
            
                if where_agg:
                    if where_agg == 'COUNT':
                        where_val = random.randint(1, len(table[where_conditional_columns[i]]))
                        where_conditions_with_aggs.append(f'{where_agg}({where_col}) {where_rel_op} {where_val}')
                else:
                    temp_rel_op = random.choice(string_relational_op)
                    where_conditions_with_aggs.append(f'{where_col} {temp_rel_op} "{where_val}"')
                where_conditions_with_aggs.append(random.choice(['AND', 'OR']))
    where_conditions = " ".join(where_conditions[:-1])
    where_conditions_with_aggs = " ".join(where_conditions_with_aggs[:-1])
    if with_aggs:
        return where_conditions_with_aggs
    return where_conditions

def get_having_conditions(total_columns, table1, table2, with_aggs=False):
    group_by_column = random.choice(total_columns)
    number_of_having_conditions = random.randint(1, len(total_columns))
    having_conditional_columns = random.sample(total_columns, k=number_of_having_conditions) 
    having_conditional_values = [random.choice(table1[having_col].to_list()) if having_col in table1.columns else random.choice(table2[having_col].to_list())
                        for having_col in having_conditional_columns]
    having_relational_op = [random.choice(numeric_relational_op) if isinstance(cond_val, int) or isinstance(cond_val, float) else random.choice(string_relational_op) for cond_val in having_conditional_values]
    having_conditions = []
    having_condition_with_aggs = []
    having_aggs = [random.choice(agg_set) if random.random() > 0.5 and (isinstance(cond_val, int) or isinstance(cond_val, float)) or (type(cond_val) == np.int_) or (type(cond_val) == np.float_) \
                                            else "" for cond_val in having_conditional_values]
    for i, (having_col, having_rel_op, having_val, having_agg) in enumerate(zip(having_conditional_columns, having_relational_op, having_conditional_values, having_aggs)):
            if isinstance(having_val, int) or isinstance(having_val, float):
                having_conditions.append(f'{having_col} {having_rel_op} {having_val}')
                having_conditions.append(random.choice(['AND', 'OR']))
                
                if having_agg:
                    having_condition_with_aggs.append(f'{having_agg}({having_col}) {having_rel_op} {having_val}')
                else:
                    having_condition_with_aggs.append(f'{having_col} {having_rel_op} {having_val}')
                having_condition_with_aggs.append(random.choice(['AND', 'OR']))
            else:
                having_conditions.append(f'{having_col} {having_rel_op} "{having_val}"')
                having_conditions.append(random.choice(['AND', 'OR']))
                
                if having_agg:
                    having_condition_with_aggs.append(f'{having_agg}({having_col}) {having_rel_op} "{having_val}"')
                else:
                    having_condition_with_aggs.append(f'{having_col} {having_rel_op} "{having_val}"')
                having_condition_with_aggs.append(random.choice(['AND', 'OR']))
    having_conditions = " ".join(having_conditions[:-1])
    having_condition_with_aggs = " ".join(having_condition_with_aggs[:-1])
    if with_aggs:
        return having_condition_with_aggs
    return having_conditions

def get_having_conditions_from_1_table(total_columns, table, with_aggs=False):
    group_by_column = random.choice(total_columns)
    number_of_having_conditions = random.randint(1, len(total_columns))
    having_conditional_columns = random.sample(total_columns, k=number_of_having_conditions) 
    having_conditional_values = [random.choice(table[having_col].to_list()) for having_col in having_conditional_columns]
    having_relational_op = [random.choice(numeric_relational_op) if isinstance(cond_val, int) or isinstance(cond_val, float) else random.choice(string_relational_op) for cond_val in having_conditional_values]
    having_conditions = []
    having_condition_with_aggs = []
    having_aggs = [random.choice(agg_set) if random.random() > 0.5 and (isinstance(cond_val, int) or isinstance(cond_val, float)) or (type(cond_val) == np.int_) or (type(cond_val) == np.float_) \
                                            else "" for cond_val in having_conditional_values]
    for i, (having_col, having_rel_op, having_val, having_agg) in enumerate(zip(having_conditional_columns, having_relational_op, having_conditional_values, having_aggs)):
            if isinstance(having_val, int) or isinstance(having_val, float):
                having_conditions.append(f'{having_col} {having_rel_op} {having_val}')
                having_conditions.append(random.choice(['AND', 'OR']))
                
                if having_agg:
                    having_condition_with_aggs.append(f'{having_agg}({having_col}) {having_rel_op} {having_val}')
                else:
                    having_condition_with_aggs.append(f'{having_col} {having_rel_op} {having_val}')
                having_condition_with_aggs.append(random.choice(['AND', 'OR']))
            else:
                having_conditions.append(f'{having_col} {having_rel_op} "{having_val}"')
                having_conditions.append(random.choice(['AND', 'OR']))
                
                if having_agg:
                    having_condition_with_aggs.append(f'{having_agg}({having_col}) {having_rel_op} "{having_val}"')
                else:
                    having_condition_with_aggs.append(f'{having_col} {having_rel_op} "{having_val}"')
                having_condition_with_aggs.append(random.choice(['AND', 'OR']))
    having_conditions = " ".join(having_conditions[:-1])
    having_condition_with_aggs = " ".join(having_condition_with_aggs[:-1])
    if with_aggs:
        return having_condition_with_aggs
    return having_conditions

def get_group_by_with_1_table(select_common_columns, table):
    # GROUP BY condition
    group_by_column = random.choice(select_common_columns)
    agg_functions = []
    
    # all select columns are aggregated apart from group by column
    for col in select_common_columns:
        val = table[col][0]
        if col == group_by_column:
            agg_functions.append("")
        else:
            if isinstance(val, int) or isinstance(val, float) or (type(val) == np.int_) or (type(val) == np.float_):         
                agg_functions.append(random.choice(agg_set))
            else:
                agg_functions.append('COUNT')

    select_common_columns_with_aggs = [f"{agg}({col})" if agg != "" else col for agg, col in zip(agg_functions, select_common_columns)]
    return select_common_columns_with_aggs, group_by_column
                                
def get_having_from_groups_1_table(table_name, select_common_columns, con, group_by_column, select_common_columns_with_aggs=None, where_clause=None):
    if len(select_common_columns) == 1:
        number_of_having_conditions = random.randint(1, len(select_common_columns))
    else:
        number_of_having_conditions = random.randint(1, len(select_common_columns)-1) # do not include group by column in having
    possible_having_columns = select_common_columns_with_aggs.copy() if select_common_columns_with_aggs else select_common_columns.copy()
    if len(select_common_columns) > 1:
        possible_having_columns.remove(group_by_column)
    having_conditional_columns = random.sample(possible_having_columns, k=number_of_having_conditions) 
    sql_template = f'SELECT {", ".join(select_common_columns_with_aggs)} FROM {table_name}'
    if where_clause:
        sql_template += f' WHERE {where_clause}'
    group_by_template = sql_template + f' GROUP BY {group_by_column}'
    group_by_answer = pd.read_sql(group_by_template, con)
    having_conditional_values = [random.choice(group_by_answer[col].to_list()) for col in having_conditional_columns]
    having_relational_op = [random.choice(numeric_relational_op) if isinstance(cond_val, int) or isinstance(cond_val, float) else random.choice(string_relational_op) for cond_val in having_conditional_values]
    having_conditions = []

    assert len(having_conditional_columns) == len(having_relational_op) == len(having_conditional_values), "number of HAVING columns, operations and values must match!"
    for having_col, having_rel_op, having_val in zip(having_conditional_columns, having_relational_op, having_conditional_values):
            if isinstance(having_val, int) or isinstance(having_val, float):
                having_conditions.append(f'{having_col} {having_rel_op} {having_val}')
                having_conditions.append(random.choice(['AND', 'OR']))
                
    having_conditions = " ".join(having_conditions[:-1])
    
    template_with_having = group_by_template + f' HAVING {having_conditions}'
    try:
        answer = pd.read_sql(template_with_having, con)
    except:
        return ""
    return template_with_having
    

def clean_table(table):
    table.fillna("", inplace=True)
    if len(table) > 0:
        # rename all duplicate column names 
        if len(set(table.columns.to_list())) != len(table.columns):
            column_name_indices = defaultdict(list)
            for i, col_name in enumerate(table.columns):
                column_name_indices[col_name].append(i)
            new_column_names = ['']*len(table.columns)
            for col_name, indices in column_name_indices.items():
                if len(indices) > 1:
                    for j, index in enumerate(indices):
                        new_column_names[index]=f"{col_name} -{j+1}"
                        #table.rename(inplace=True,columns={f'{index+1}col':f"{col_name}-{j+1}"})
                else:
                    new_column_names[indices[0]] = col_name 
            table.columns = new_column_names        
        # check all values of table and map them to str
        table = table.applymap(lambda x: str(pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S')) if isinstance(x, pd.Timestamp) or isinstance(x, np.datetime64) else str(x))
        #table = table.applymap(lambda x: str(x))
    return table
    
def preprocess_sample_pretraining(sample):
    """
    Each sample of the preprocessed dataset comprises of a dictionary of format
                                        {
                                        "tables": pd.Dataframe,
                                        "table_names": list,
                                        "question": str,
                                        "query": str,
                                        "answer": str,
                                        }
    Each sample of Spider data comprises of a question and its associated list of answers. We split each answer into multiple samples by repeating the question with its unique answer.
    :param sample: A Spider sample
    :return: A pre-prcessed Spider sample
    """
    tables = [clean_table(sample['tables'][0])]
    table_names = [""]
    answer = sample["answer"] #clean_table(pd.read_json(sample["answer"], orient='split'))
    question = clean_query(sample['question'])
    answer.columns = answer.columns.map(str)
    #print(f'answer headers: {answer.columns} {type(answer.columns)}')
    preprocessed = {
        "tables": tables,
        "table_names": table_names,
        "question": question,
        "answer": answer,
        }       
    return preprocessed
