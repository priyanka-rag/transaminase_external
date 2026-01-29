from models.main_models import *

from sklearn.metrics import r2_score
from heapq import nlargest
import pdb

def pick_k_descriptors(k, enzyme, corr_df):
    all_descs = corr_df['Descriptor'].to_list()
    all_coeffs = [abs(i) for i in corr_df[enzyme].to_list()]
    descs_idxs = nlargest(3, range(len(all_coeffs)), key=lambda idx: all_coeffs[idx])
    descs = [i for i in range(len(all_descs)) if i in descs_idxs]
    return descs

def get_k_feature_lists(init_feature_list, all_dft_descs):
    feature_lists = []
    items_to_add = [i for i in all_dft_descs if i not in init_feature_list]
    for i in items_to_add:
        feature_lists.append(init_feature_list + [i])
    return feature_lists

def perform_loocv_on_enzyme(enzyme_df, model_type, task_type, feature_type, target_type):
    split_type = 'loocv' 

    preds = []

    for i,row in enzyme_df.iterrows():
        test_df = enzyme_df.loc[i,:].to_frame().transpose()
        train_df = enzyme_df.drop(enzyme_df.index[i])

        assert len(train_df) == len(enzyme_df) - 1
        assert len(test_df) == 1
        assert test_df['dummy'].to_list()[0]  not in train_df['dummy'].to_list()
        assert len(train_df['Ketone_Smiles'].unique()) == len(train_df)
        assert test_df['Ketone_Smiles'].to_list()[0] not in train_df['Ketone_Smiles'].to_list()

        targets = enzyme_df[target_type].to_list()
        
        model = Model_Trainer(model_type, split_type, task_type, feature_type, target_type, train_df, test_df)
        model.train_test_model()

        assert model.y_pred.shape[0] == 1

        if task_type == 'bin': preds.extend(model.y_probs[:,1])
        else: preds.extend(model.y_pred)
        #break
        
    if task_type == 'reg' and (target_type == 'conversion' or target_type == 'ee'): preds = [i*100 for i in preds]
    return targets, preds

def perform_forward_mvlr(master_df, corr_df, enzyme, target_type):

    model_type = 'lin'
    task_type = 'reg'

    # read data, subset by enzyme type
    enzyme_df = master_df[master_df['Enzyme'] == enzyme].reset_index(drop=True)

    # get correlations
    all_dft_descs = corr_df['Descriptor'].to_list()
    all_enzyme_corrs = list(zip(all_dft_descs, corr_df[enzyme].to_list()))

    k = 1
    curr_best_r2, best_overall_r2 = float('inf'), -1*float('inf') #some random initializations just to enter the loop

    # get single most correlated descriptor
    top_descs = [max(all_enzyme_corrs, key=lambda x: abs(x[1]))[0]]
    print(f'{enzyme}:')
    print(f'The {k} most correlated descriptor is: {top_descs[0]}\n')

    # enter forward MVLR loop
    while curr_best_r2 >= best_overall_r2:
        k = k+1
        #if k == 2: best_overall_r2 = curr_best_r2
        #elif curr_best_r2 > best_overall_r2: best_overall_r2 = curr_best_r2 

        print(f'Now running all {k}-parameter models...')
        feature_lists = get_k_feature_lists(top_descs, all_dft_descs) # get all k-parameter feature lists
        r2_list = []
        targets_lists, preds_lists = [], []

        for feature_list in feature_lists: # train models and evaluate
            targets, preds = perform_loocv_on_enzyme(enzyme_df, model_type, task_type, feature_list, target_type)
            #targets_lists.append(targets)
            preds_lists.append(preds)
            r2_list.append(r2_score(targets, preds))

        curr_best_r2 = max(r2_list) #find best model performance for this k
        top_descs = feature_lists[np.argmax(r2_list)] #find associated features for this k
        curr_best_preds = preds_lists[np.argmax(r2_list)] 
        print(f'The best features were {top_descs}, giving an R2 of {curr_best_r2}\n')

        if curr_best_r2 >= best_overall_r2:
            overall_top_descs = top_descs 
            best_overall_r2 = curr_best_r2
            #best_targets = curr_best_targets
            best_preds = curr_best_preds
        
    #enzyme_df[f'Predicted_{target_type}'] = best_preds
    print(f'The overall selected features were {overall_top_descs}, giving an R2 of {best_overall_r2}\n')
    return overall_top_descs, best_overall_r2, best_preds
