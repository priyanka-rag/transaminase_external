import numpy as np
import math
import traceback
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.preprocessing import OneHotEncoder,MultiLabelBinarizer

def ee_to_ddg(df):
    ddg_list = []
    R = 0.0019872036 #kcal/mol K
    T = 308.15
    for i,row in df.iterrows():
        ee, enzyme = row['ee']/100, row['Enzyme_Type']
        if ee == 1: ee = 0.9999
        ddg = R*T*math.log((1+ee)/(1-ee))
        if enzyme == 'R': ddg = -1*ddg
        ddg_list.append(ddg)
    df['ddg'] = ddg_list
    return df

def ddg_to_ee(ddg):
    R = 0.0019872036 #kcal/mol K
    T = 308.15
    scaled_ddg = abs(ddg)/(R*T)
    ee = (math.exp(scaled_ddg)-1)/(math.exp(scaled_ddg)+1)
    return 100*ee #unscale

def get_target_data(target_list, task_type, target_type):
    if target_type == 'conversion':
        if task_type == 'bin':
            target_list = np.array([0 if i <= 25 else 1 for i in target_list])
        elif task_type == 'reg':
            target_list = np.array([i/100 for i in target_list])
    elif target_type == 'ee':
        if task_type == 'bin':
            target_list = np.array([0 if i < 95 else 1 for i in target_list])
        elif task_type == 'reg':
            target_list = np.array([i/100 for i in target_list])
    elif target_type == 'ddg':
        if task_type == 'bin':
            target_list = np.array([0 if abs(i) < 1.803 else 1 for i in target_list])
    return target_list

def one_hot_encode(molecule_list):
    molecule_ohe = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=2)
    molecule_list_dummy = np.concatenate((molecule_list,molecule_list)).reshape(-1,1)
    molecule_list_dummy = np.append(molecule_list_dummy,'blah').reshape(-1,1)
    molecule_ohe.fit(molecule_list_dummy)
    return molecule_ohe

def genFingerprints(smiles_list,radius,nBits): # generates Morgan fingerprints given a list of smiles strings
    fp_list = []
    for i,smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,radius=radius, nBits=nBits)
        arr = np.array((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_list.append(arr.tolist())
    return fp_list

def get_fgp_data(ketone_list,radius,nBits):
    fp_list = np.vstack(genFingerprints(ketone_list,radius,nBits))
    return fp_list

# this function was taken from https://greglandrum.github.io/rdkit-blog/posts/2022-12-23-descriptor-tutorial.html
def getMolDescriptors(mol, missingVal=None):
    res = {}
    for nm,fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return list(res.values())

def get_physchem_descriptors(molecule_list):
    descriptor_list = [getMolDescriptors(Chem.MolFromSmiles(smi)) for smi in molecule_list]
    return np.array(descriptor_list)

def get_dft_data(df, features):
    mol_list = df['MoleculeFeatures'].to_list()
    carbon1_list,carbonyl_list,carbon2_list = df['Carbon1_AtomFeatures'].to_list(), \
                                              df['Carbonyl_AtomFeatures'].to_list(), \
                                              df['Carbon2_AtomFeatures'].to_list()
    cc1_sterimol_list,cc2_sterimol_list = df['CC1_Sterimol'].to_list(), df['CC2_Sterimol'].to_list()
    
    dft_descriptors = np.zeros((len(df),len(features)))
    for i, feat in enumerate(features):
        if 'Carbon' in feat: # if atom-level DFT feature
            feat_mod = feat.split('.')[0]
            if 'Carbon1' in feat: feat_vals = [float(j[feat_mod]) for j in carbon1_list]
            elif 'Carbonyl' in feat: feat_vals = [float(j[feat_mod]) for j in carbonyl_list]
            elif 'Carbon2' in feat: feat_vals = [float(j[feat_mod]) for j in carbon2_list]
        elif 'CC' in feat: # if Sterimol parameter
            feat_mod = feat.split('.')[0]
            if 'CC1' in feat: feat_vals = [float(j[feat_mod]) for j in cc1_sterimol_list]
            elif 'CC2' in feat: feat_vals = [float(j[feat_mod]) for j in cc2_sterimol_list]
        else: feat_vals = [float(j[feat]) for j in mol_list] # if molecule-level feature
        dft_descriptors[:,i] =  np.array(feat_vals)
    return dft_descriptors

def get_reactive_site_dft_data(df):
    mol_list = df['MoleculeFeatures'].to_list()
    carbon1_list,carbonyl_list,carbon2_list = df['Carbon1_AtomFeatures'].to_list(), \
                                              df['Carbonyl_AtomFeatures'].to_list(), \
                                              df['Carbon2_AtomFeatures'].to_list()
    mol_desc_labels = ['dipole','molar_mass','electronic_spatial_extent','E_scf','G','homo_energy','lumo_energy','electronegativity']
    atom_desc_labels = ['VBur','Mulliken_charge','APT_charge','NPA_charge','NPA_valence','NMR_anisotropy','ES_root_Mulliken_charge','ES_root_NPA_charge']
    mol_descs,carbon1_descs,carbonyl_descs,carbon2_descs = [],[],[],[]

    for mol,carbon1,carbonyl,carbon2 in zip(mol_list,carbon1_list,carbonyl_list,carbon2_list):
        mol_descs.append([float(mol[key]) for key in mol_desc_labels])
        carbon1_descs.append([float(carbon1[key]) for key in atom_desc_labels])
        carbonyl_descs.append([float(carbonyl[key]) for key in atom_desc_labels])
        carbon2_descs.append([float(carbon2[key]) for key in atom_desc_labels])

    dft_descriptors = np.hstack((mol_descs,carbon1_descs,carbonyl_descs,carbon2_descs))
    #dft_descriptors = np.hstack((carbon1_descs,carbonyl_descs,carbon2_descs))
    return dft_descriptors