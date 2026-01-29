import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from copy import deepcopy

def calc_molecule_correlations(df, enzymes, target):
    df = deepcopy(df) # we add columns to the df later so make a copy

    # define desired features
    feature_names = [key for key in df['MoleculeFeatures'].to_list()[0].keys()]
    feature_names = [key for key in feature_names if not(key in ['number_of_atoms','charge','multiplicity','converged','stoichiometry'])]

    for feat in feature_names:
        vals = [i[feat] for i in df['MoleculeFeatures'].to_list()]
        df[feat] = vals
    
    corr_arr = np.zeros((len(enzymes),len(feature_names)))

    for i, enzyme in enumerate(enzymes):
        sub_df = df[df['Enzyme'] == enzyme]
        for j, feat in enumerate(feature_names):
            #print(i,j,feat)
            dft_feats = np.array([float(k) for k in sub_df[feat].to_list()])
            targets = np.array(sub_df[target].to_list())
            #print(dft_feats)
            #print(targets)
            corr_coeff = pearsonr(dft_feats, targets).statistic
            corr_arr[i][j] = corr_coeff

    #print(corr_arr.shape)
    corr_df = pd.DataFrame(corr_arr, columns = feature_names, index=enzymes)
    return corr_df

def calc_atom_correlations(df, enzymes, target, atom):
    df = deepcopy(df) # we add columns to the df later so make a copy

    # define desired features
    feature_names = [key for key in df[f'{atom}_AtomFeatures'].to_list()[0].keys()]
    feature_names = [key for key in feature_names if not(key in ['X','Y','Z'])]

    for feat in feature_names:
        vals = [i[feat] for i in df[f'{atom}_AtomFeatures'].to_list()]
        df[feat] = vals
    
    corr_arr = np.zeros((len(enzymes),len(feature_names)))

    for i, enzyme in enumerate(enzymes):
        sub_df = df[df['Enzyme'] == enzyme]
        for j, feat in enumerate(feature_names):
            #print(i,j,feat)
            dft_feats = np.array([float(k) for k in sub_df[feat].to_list()])
            targets = np.array(sub_df[target].to_list())
            #print(dft_feats)
            #print(targets)
            corr_coeff = pearsonr(dft_feats, targets).statistic
            corr_arr[i][j] = corr_coeff

    #print(corr_arr.shape)
    corr_df = pd.DataFrame(corr_arr, columns = feature_names, index=enzymes)
    return corr_df

def calc_sterimol_correlations(df, enzymes, target, atom):
    df = deepcopy(df) # we add columns to the df later so make a copy

    # define desired features
    feature_names = [key for key in df[f'{atom}_Sterimol'].to_list()[0].keys()]

    for feat in feature_names:
        vals = [i[feat] for i in df[f'{atom}_Sterimol'].to_list()]
        df[feat] = vals
    
    corr_arr = np.zeros((len(enzymes),len(feature_names)))

    for i, enzyme in enumerate(enzymes):
        sub_df = df[df['Enzyme'] == enzyme]
        for j, feat in enumerate(feature_names):
            #print(i,j,feat)
            dft_feats = np.array([float(k) for k in sub_df[feat].to_list()])
            targets = np.array(sub_df[target].to_list())
            #print(dft_feats)
            #print(targets)
            corr_coeff = pearsonr(dft_feats, targets).statistic
            corr_arr[i][j] = corr_coeff

    #print(corr_arr.shape)
    corr_df = pd.DataFrame(corr_arr, columns = feature_names, index=enzymes)
    return corr_df

def calc_all_correlations(df, enzymes, target):
    dft_atoms = ['Carbon1', 'Carbonyl', 'Carbon2']
    sterimol_atoms = ['CC1', 'CC2']
    corr_dfs = []

    # get molecule-level correlations
    corr_df = calc_molecule_correlations(df, enzymes, target).T
    corr_df.insert(0,'Descriptor',corr_df.index.to_list())
    corr_dfs.append(corr_df)

    # get atom-level correlations
    for i,atom in enumerate(dft_atoms):
        corr_df = calc_atom_correlations(df, enzymes, target, atom)
        corr_df = corr_df.T
        desc_names = [f"{i}.{atom}" for i in corr_df.index.to_list()]
        corr_df.insert(0,'Descriptor',desc_names)
        corr_dfs.append(corr_df)

    # get sterimol correlations
    for i,atom in enumerate(sterimol_atoms):
        corr_df = calc_sterimol_correlations(df, enzymes, target, atom)
        corr_df = corr_df.T 
        desc_names = [f"{i}.{atom}" for i in corr_df.index.to_list()]
        corr_df.insert(0,'Descriptor',desc_names)
        corr_dfs.append(corr_df)

    all_corr_df = pd.concat(corr_dfs)
    return all_corr_df.reset_index(drop=True)

