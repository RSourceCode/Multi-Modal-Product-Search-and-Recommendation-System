import pandas as pd

def Recall_K(K : int, retrieved_prods: pd.DataFrame, actual_prods: pd.DataFrame):

    # TODO: Check the column names from the table. The current names are just placeholder
    relevant_prods = actual_prods[actual_prods['Label'] != 'I']
    total_relevant_prods = (actual_prods['Label'] != 'I').sum()
    correct_retrievals = pd.merge(retrieved_prods[ : K], relevant_prods, how='inner', on='prod_id')
    num_correct_retrieval = len(correct_retrievals) 

    return num_correct_retrieval / total_relevant_prods

def Precision_K():
    pass

def NDCG_K():
    pass
