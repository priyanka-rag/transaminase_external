import pandas as pd
from ast import literal_eval

main_datasets_dir = './dataset_files'

# some general converters to help read in various datasets
converters = {'MoleculeFeatures': literal_eval, 'Carbon1_AtomFeatures': literal_eval,
              'Carbonyl_AtomFeatures': literal_eval, 'Carbon2_AtomFeatures': literal_eval,
              'CC1_Sterimol': literal_eval, 'CC2_Sterimol': literal_eval}

def read_data():
    df = pd.read_csv(f'{main_datasets_dir}/transaminase_dataset_dft_descriptors_boltzmann_final.csv', converters=converters)
    return df