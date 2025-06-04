import pandas as pd
from typing import List, Tuple

def apply_condition(df: pd.DataFrame, 
                     conditions: List[Tuple[str, any, str]], 
                     logic: str = 'and'):
    operator_options = ['==', '!=','<', '<=', '>', '>=', 'in', '!in', 'contains']
    logic_options    = ['and', 'or']
    
    if logic not in logic_options:
        raise ValueError(f'{logic} Unsupported logic. Supported operators are in {logic_options}')
    
    operation_dict = {
            "==":       lambda col, val: df[col] == val,
            "<":        lambda col, val: df[col] < val,
            ">":        lambda col, val: df[col] > val,
            "<=":       lambda col, val: df[col] <= val,
            ">=":       lambda col, val: df[col] >= val,
            "!=":       lambda col, val: df[col] != val,
            "in":       lambda col, val: df[col].isin(_input_to_list(val)),
            "!in":      lambda col, val: df[col].isin(_input_to_list(val)),
            "contains": lambda col, val: df[col].str.contains(val, case=False, na=False)
        } 
    
    logic_dict = {
            'and': lambda cond1, cond2: cond1 & cond2,
            'or':  lambda cond1, cond2: cond1 | cond2
        }

    for n, (colname, value, operator) in enumerate(conditions):
        if operator not in operator_options:
            raise ValueError(f'{operator} Unsupported operator. Supported operators are in {operator_options}')
       
        condition = operation_dict[operator](colname, value)

        if n == 0:
            combined_condition = condition
        else:
            combined_condition = logic_dict[logic](combined_condition, condition)
        
    return combined_condition

def _input_to_list(input):
    """changes the input into a list of the input (only if it not already is, otherwise the input is returned)"""
    if not isinstance(input, list):
        input = [input]   
    return input