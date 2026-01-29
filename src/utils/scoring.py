from sklearn.metrics import mean_absolute_error,accuracy_score,precision_score,recall_score,roc_auc_score,average_precision_score,r2_score,mean_squared_error

import numpy as np
from collections import defaultdict
import heapq
from copy import deepcopy
from collections import Counter

def score_predictions(task_type,y_test,y_pred,y_probs):
   res = dict()
   if task_type == 'bin':
      res['Accuracy'] = accuracy_score(y_test, y_pred)
      res['Precision'] = precision_score(y_test, y_pred)
      res['Recall'] = recall_score(y_test, y_pred)
      try: res['AUROC'] = roc_auc_score(y_test, y_probs)
      except ValueError: res['AUROC'] = float('nan')
      res['AUPRC'] = average_precision_score(y_test, y_probs)
   elif task_type == 'reg':
      if len(y_pred) > 1: res['R2'] = r2_score(y_test, y_pred)
      res['MAE'] = mean_absolute_error(y_test, y_pred)
      res['RMSE'] = mean_squared_error(y_test, y_pred, squared=False)
   return res

def get_average_metrics(res_list):
   res_agg, res_trials = defaultdict(list), defaultdict(list)
   for res in res_list:
      for key,value in res.items():
         res_trials[key].append(value)

   for key,value in res_trials.items():
      avg, std = np.mean(value), np.std(value)
      res_agg[key].append(avg)
      res_agg[key].append(std)
   return dict(res_agg), dict(res_trials)

def get_top_enzymes_groundtruth(enzymes, vals):
    vals = [abs(v) for v in vals] # adjust vals in case of ddg (should not affect ee)
    maxval = np.nanmax(vals)
    top_idxs = [i for i,v in enumerate(vals) if v == maxval] # account for ties
    top_enzymes = [enzymes[i] for i in top_idxs]
    return top_enzymes

def get_top_enzymes_predictions(enzymes, vals, k):
    nan_val = -1000
    vals = [abs(v) for v in vals] # adjust vals in case of ddg (should not affect ee)
    vals = [nan_val if np.isnan(v) else v for v in vals] # replace NaNs with large neg value
    if all(v == nan_val for v in vals): return [] # if all vals are NaN, return empty list
    
    k_vals = heapq.nlargest(k,vals)
    top_k_idxs = [i for i,v in enumerate(vals) if v in k_vals] # account for ties
    top_enzymes = [enzymes[i] for i in top_k_idxs]
    return top_enzymes

def calc_regret(enzymes, groundtruth_vals, predictions):
    assert len(enzymes) == len(groundtruth_vals) == len(predictions)
    groundtruth_vals = [abs(v) for v in groundtruth_vals] # adjust vals in case of ddg (should not affect ee)
    predictions = [abs(v) for v in predictions] # adjust vals in case of ddg (should not affect ee)
    max_groundtruth_val = np.nanmax(groundtruth_vals)
    max_pred = np.nanmax(predictions)
    max_pred_idx = predictions.index(max_pred)
    max_groundtruth_val_pred = groundtruth_vals[max_pred_idx]
    #print(max_groundtruth_val, max_groundtruth_val_pred)
    regret = max_groundtruth_val - max_groundtruth_val_pred
    assert regret >= 0
    return regret

def top_k_accuracy_score(res_df, target_type, feature_type, k):
    res_df = deepcopy(res_df)
    unique_ketones = list(res_df['Ketone_Smiles'].unique())
    top_k_flags = []
    regrets = []

    for ketone in unique_ketones:
        sub_df = res_df[res_df['Ketone_Smiles'] == ketone]
        enzymes = sub_df['Enzyme'].to_list()
        groundtruth_vals = sub_df[target_type].to_list()
        pred_vals = sub_df[f'Predicted_{target_type}_{feature_type}'].to_list()

        top_enzymes_groundtruth = get_top_enzymes_groundtruth(enzymes, groundtruth_vals)
        top_enzymes_predictions = get_top_enzymes_predictions(enzymes, pred_vals, k)
        flag = 1 if len(set(top_enzymes_groundtruth).intersection(top_enzymes_predictions)) >= 1 else 0
        top_k_flags.append(flag)
        regrets.append(calc_regret(enzymes, groundtruth_vals, pred_vals))
    
    assert len(top_k_flags) == len(unique_ketones)
    assert len(regrets) == len(unique_ketones)
    top_k_acc = sum(top_k_flags)/len(top_k_flags)
    return top_k_acc, regrets

def get_top_enzymes_predictions_naive(df, target_type, k): # get the top-k performing enzymes (by experiment) from the overall data
    all_top_k_enzymes = []
    top_k_enzymes_exp = []
    unique_ketones = list(df['Ketone_Smiles'].unique())
    for ketone in unique_ketones:
        sub_df = df[df['Ketone_Smiles'] == ketone]
        enzymes = sub_df['Enzyme'].to_list()
        groundtruth_vals = sub_df[target_type].to_list()
        #print(get_top_enzymes_predictions(enzymes,groundtruth_vals,1))
        all_top_k_enzymes.extend(get_top_enzymes_predictions(enzymes,groundtruth_vals,1)) # get top-performing enzyme(s) for this ketone

    most_common = Counter(all_top_k_enzymes).most_common()
    #print(most_common)
    assert len(most_common) == len(list(df['Enzyme'].unique()))
    for i in most_common:#[0:k]:
        top_k_enzymes_exp.append(i[0])
    return top_k_enzymes_exp

def top_k_accuracy_score_naive(df, target_type, k):
    df = deepcopy(df)
    unique_ketones = list(df['Ketone_Smiles'].unique())
    top_k_flags = []

    # get naive prediction from full dataset
    top_enzymes_prediction_naive = get_top_enzymes_predictions_naive(df, target_type, k)
    #print(top_enzymes_prediction_naive)

    for ketone in unique_ketones:
        sub_df = df[df['Ketone_Smiles'] == ketone]
        enzymes = sub_df['Enzyme'].to_list()
        groundtruth_vals = sub_df[target_type].to_list()
        
        # pick the k enzymes that naively would have been predicted

        # order available enzymes by the naive order
        naive_enzymes_ordered = []
        for enz in top_enzymes_prediction_naive:
            if enz in enzymes: naive_enzymes_ordered.append(enz)
        naive_k_enzymes = naive_enzymes_ordered[:k] # take the top-k enzymes from the naive order
        
        # compare to groundtruth
        top_enzymes_groundtruth = get_top_enzymes_groundtruth(enzymes, groundtruth_vals)
        flag = 1 if len(set(top_enzymes_groundtruth).intersection(naive_k_enzymes)) >= 1 else 0
        top_k_flags.append(flag)
    
    assert len(top_k_flags) == len(unique_ketones)
    top_k_acc = sum(top_k_flags)/len(top_k_flags)
    return top_k_acc