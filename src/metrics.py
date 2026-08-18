import pandas as pd
import numpy as np
import time

def Recall_K(K : int, retrieved_prods: pd.DataFrame, actual_prods: pd.DataFrame):
    relevant_prods = actual_prods[actual_prods['esci_label'] != 'I']
    total_relevant_prods = (actual_prods['esci_label'] != 'I').sum()
    correct_retrievals = pd.merge(retrieved_prods[ : K], relevant_prods, how='inner', on='product_id')
    num_correct_retrieval = len(correct_retrievals) 

    return num_correct_retrieval / total_relevant_prods

def Precision_K(K : int, retrieved_prods: pd.DataFrame, actual_prods: pd.DataFrame):
    relevant_prods = actual_prods[actual_prods['esci_label'] != 'I']
    correct_retrievals = pd.merge(retrieved_prods[ : K], relevant_prods, how='inner', on='product_id')
    num_correct_retrieval = len(correct_retrievals) 

    return num_correct_retrieval / K

def DCG_K(K : int, retrieved_prods: pd.DataFrame, actual_prods: pd.DataFrame):
    relevance_map = {'E': 3, 'S': 2, 'C': 1, 'I': 0}
    merged = pd.merge(retrieved_prods[:K], actual_prods, how='left', on='product_id')
    relevance = merged['esci_label_y'].map(relevance_map).fillna(0).values

    return sum(rel / np.log2(rank + 1) for rank, rel in enumerate(relevance, 1))

def NDCG_K(K : int, retrieved_prods: pd.DataFrame, actual_prods: pd.DataFrame):
    dcg = DCG_K(K, retrieved_prods, actual_prods)
    relevance_map = {'E': 3, 'S': 2, 'C': 1, 'I': 0}
    ideal = actual_prods['esci_label'].map(relevance_map).fillna(0).sort_values(ascending=False)[:K].values
    idcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(ideal, 1))

    return dcg / idcg if idcg != 0 else 0

def query_latency(query_func, *args, **kwargs):
    start = time.perf_counter()
    result = query_func(*args, **kwargs)
    latency = time.perf_counter() - start

    return latency