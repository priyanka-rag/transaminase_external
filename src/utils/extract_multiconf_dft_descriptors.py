import json
from collections import namedtuple
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdqueries
RDLogger.DisableLog('rdApp.*')

from data.read_data import *
import pdb

smiles_id_df = pd.read_csv(f'./dataset_files/exp_transaminase_unique_molecules.csv')
ketone_patt = Chem.MolFromSmarts('[#6][#6X3](=O)[#6]')

# get the molecule ID of a molecule given its SMILES
def smiles_to_mol_id(smi):
    # read df of smiles-molecule_id mapping
    mol_id = smiles_id_df.loc[smiles_id_df['SMILES'] == smi, 'Molecule_ID'].iloc[0]
    return mol_id

# read .json file of DFT descriptors for a certain molecule_id
def read_json_file(molecule_id):
    desc_filepath = f"/data/abbvie/transaminase/boltzmann_molecule_descriptors/{molecule_id}.json"
    with open(desc_filepath) as project_file:    
        dft_descriptors = json.load(project_file)
    return dft_descriptors

def map_ketone(smi):
    # atom map ketone
    unmapped = 1
    mol = Chem.MolFromSmiles(smi)
    for a in mol.GetAtoms():
        if not a.HasProp('molAtomMapNumber'): a.SetIntProp('molAtomMapNumber', unmapped)
        unmapped += 1

    # tag ketone
    for idx in range(mol.GetNumAtoms()):
        mol.GetAtomWithIdx(idx).SetBoolProp("tagged", True)
    return mol

def find_reactive_atoms(mol):
    # get ketone atomx
    hasprop = rdqueries.HasPropQueryAtom("tagged")
    for atom in ketone_patt.GetAtoms():
        atom.ExpandQuery(hasprop)
    reactive_atoms_idx = list(mol.GetSubstructMatch(ketone_patt))

    # remove oxygen atom
    reactive_atoms = []
    for idx in reactive_atoms_idx:
        if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6: reactive_atoms.append(idx)
    reactive_atoms = [a+1 for a in reactive_atoms]

    #sanity check that we got the right atoms by checking how many
    assert len(reactive_atoms) == 3

    #assign atoms to their roles and return
    carbon1 = reactive_atoms[0]
    carbonyl = reactive_atoms[1]
    carbon2 = reactive_atoms[2]
    return carbon1, carbonyl, carbon2

# extracts all molecule-level descriptors from the .json file associated with this molecule
def extract_molecule_descriptors(smi):
   # get the molecule ID
    molid = smiles_to_mol_id(smi)

    # read corresponding .json file
    descriptors = read_json_file(molid)

    # read molecule-level features from "descriptors" 
    molecule_descriptors = descriptors['descriptors']
    return molecule_descriptors

# extracts all atom-level (reactive site specific) descriptors for the ketone molecule from the .json files associated with these molecules
def extract_atom_descriptors(smi):
   # get the molecule ID
    molid = smiles_to_mol_id(smi)

    # get reactive atoms from this reaction
    mol = map_ketone(smi)
    carbon1_idx, carbonyl_idx, carbon2_idx = find_reactive_atoms(mol)

    # get atom-level (reactive site specific) descriptors, using the atom indices, for the boronic acid and aryl halide
    descriptors = read_json_file(molid)
    atom_descriptors = descriptors['atom_descriptors']
    carbon1_features = dict([(key,value[carbon1_idx-1]) for key,value in atom_descriptors.items()])
    carbonyl_features = dict([(key,value[carbonyl_idx-1]) for key,value in atom_descriptors.items()])
    carbon2_features = dict([(key,value[carbon2_idx-1]) for key,value in atom_descriptors.items()])
    #print(Chem.MolToSmiles(mol), carbon1_idx, carbonyl_idx, carbon2_idx)
    #pdb.set_trace()
    return carbon1_idx, carbonyl_idx, carbon2_idx, carbon1_features, carbonyl_features, carbon2_features, Chem.MolToSmiles(mol)

# adds columns of molecule-level and atom-level, reactive site DFT descriptors to a dataset df
def extract_descriptors_all_reactions(df):
    failed_reactions = [] #keep track of reactions that couldn't be atom-mapped, or involved molecules without computed descriptors
    mapped_ketones_list, molfeatures_list, carbon1_features_list, carbonyl_features_list, carbon2_features_list = [], [], [], [], []
    carbon1_idx_list, carbonyl_idx_list, carbon2_idx_list = [], [], []

    for i,row in df.iterrows():
        try:
            smi = row['Ketone_Smiles']

            # get the molecule-level features
            moleculefeatures = extract_molecule_descriptors(smi)
            molfeatures_list.append(moleculefeatures)

            # get the atom-level features
            c1_idx, carb_idx, c2_idx, carbon1_features, carbonyl_features, carbon2_features, mapped_ketone = extract_atom_descriptors(smi)
            carbonyl_idx_list.append(carb_idx)
            carbonyl_features_list.append(carbonyl_features)
            mapped_ketones_list.append(mapped_ketone)

            if carbon1_features['VBur'] > carbon2_features['VBur']:
                carbon1_idx_list.append(c2_idx)
                carbon2_idx_list.append(c1_idx)
                carbon1_features_list.append(carbon2_features)
                carbon2_features_list.append(carbon1_features)
            else:
                carbon1_idx_list.append(c1_idx)
                carbon2_idx_list.append(c2_idx)
                carbon1_features_list.append(carbon1_features)
                carbon2_features_list.append(carbon2_features)
            #pdb.set_trace()
        except: failed_reactions.append(i)
    
    df.drop(index=failed_reactions,inplace=True)
    df['Mapped_Ketone'] = mapped_ketones_list
    df['Carbon1_Idx'] = carbon1_idx_list
    df['Carbonyl_Idx'] = carbonyl_idx_list
    df['Carbon2_Idx'] = carbon2_idx_list
    df['MoleculeFeatures'] = molfeatures_list
    df['Carbon1_AtomFeatures'] = carbon1_features_list
    df['Carbonyl_AtomFeatures'] = carbonyl_features_list
    df['Carbon2_AtomFeatures'] = carbon2_features_list
    pdb.set_trace()
    return df

def reassign_alpha_carbons(df):
    c1_features, c2_features = [], []
    c1_idxs, c2_idxs = [], []
    for i,row in df.iterrows():
        c1_vbur = row['Carbon1_AtomFeatures']['VBur']
        c2_vbur = row['Carbon2_AtomFeatures']['VBur']
        if c1_vbur > c2_vbur: 
            print(i)
            c1_features.append(row['Carbon2_AtomFeatures'])
            c2_features.append(row['Carbon1_AtomFeatures'])
            c1_idxs.append(row['Carbon2_Idx'])
            c2_idxs.append(row['Carbon1_Idx'])
        else:
            c1_features.append(row['Carbon1_AtomFeatures'])
            c2_features.append(row['Carbon2_AtomFeatures'])
            c1_idxs.append(row['Carbon1_Idx'])
            c2_idxs.append(row['Carbon2_Idx'])
    
    df['Carbon1_AtomFeatures'] = c1_features
    df['Carbon2_AtomFeatures'] = c2_features
    df['Carbon1_Idx'] = c1_idxs
    df['Carbon2_Idx'] = c2_idxs
    return df

