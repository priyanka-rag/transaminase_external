import numpy as np
from morfeus import Sterimol, read_xyz
from morfeus.conformer import ConformerEnsemble
from extract_multiconf_dft_descriptors import *
import pdb

def get_confs_and_weights(mol_id):
    filepath = f'/data/abbvie/transaminase/boltzmann_molecule_descriptors/{mol_id}.json'
    with open(filepath) as project_file:
        dft_descriptors = json.load(project_file)

    conf_ids_used = dft_descriptors['conf_ids_used']
    weights = dft_descriptors['weights']

    filepath_list = [f'/data/abbvie/transaminase/molecule_descriptors/{mol_id}/conf_{i}.json' for i in conf_ids_used]

    conformations = []
    for filepath in filepath_list:
        with open(filepath) as project_file:
            conf = json.load(project_file)
        conformations.append(conf)

    return conformations, weights

def calc_sterimol_parameters(dft_descriptors, carbon1_idx, carbonyl_idx, carbon2_idx):
    # get optimized geometry coordinates
    atom_descs = dft_descriptors['atom_descriptors']
    elements = dft_descriptors['labels']
    x_coords, y_coords, z_coords = np.array(atom_descs['X']).reshape(-1,1), np.array(atom_descs['Y']).reshape(-1,1), np.array(atom_descs['Z']).reshape(-1,1)
    coordinates = np.hstack((x_coords,y_coords,z_coords))
    #pdb.set_trace()
    #print(elements,coordinates.shape)

    # calculate carbonyl --> carbon1 parameters
    sterimol1 = Sterimol(elements,coordinates,carbonyl_idx,carbon1_idx)
    cc1_dict = {'L': sterimol1.L_value, 'B1': sterimol1.B_1_value, 'B5': sterimol1.B_5_value}

    # calculate carbonyl --> carbon2 parameters
    sterimol2 = Sterimol(elements,coordinates,carbonyl_idx,carbon2_idx)
    cc2_dict = {'L': sterimol2.L_value, 'B1': sterimol2.B_1_value, 'B5': sterimol2.B_5_value}
    return cc1_dict, cc2_dict

def calc_boltzmann_sterimol_parameters(conformations, weights, carbon1_idx, carbonyl_idx, carbon2_idx):
    cc1_sterimol_list, cc2_sterimol_list = [], []
    for conf in conformations:
        cc1_dict, cc2_dict = calc_sterimol_parameters(conf, carbon1_idx, carbonyl_idx, carbon2_idx)
        cc1_sterimol_list.append(cc1_dict)
        cc2_sterimol_list.append(cc2_dict)
    
    keys = ['L','B1','B5']
    cc1_boltzmann_dict = {key: np.average([cc1[key] for cc1 in cc1_sterimol_list], weights=weights) for key in keys}
    cc2_boltzmann_dict = {key: np.average([cc2[key] for cc2 in cc2_sterimol_list], weights=weights) for key in keys}
    return cc1_boltzmann_dict, cc2_boltzmann_dict

def add_sterimol_params_to_df(df):
    cc1_sterimol_list, cc2_sterimol_list = [], []
    for i,row in df.iterrows():
        smi, c1_idx, carb_idx, c2_idx = row['Ketone_Smiles'], row['Carbon1_Idx'], row['Carbonyl_Idx'], row['Carbon2_Idx']
        mol_id = smiles_to_mol_id(smi)

        conformations, weights = get_confs_and_weights(mol_id)
        cc1_boltzmann_dict, cc2_boltzmann_dict = calc_boltzmann_sterimol_parameters(conformations, weights, c1_idx, carb_idx, c2_idx)
        cc1_sterimol_list.append(cc1_boltzmann_dict)
        cc2_sterimol_list.append(cc2_boltzmann_dict)
    df['CC1_Sterimol'] = cc1_sterimol_list
    df['CC2_Sterimol'] = cc2_sterimol_list
    return df