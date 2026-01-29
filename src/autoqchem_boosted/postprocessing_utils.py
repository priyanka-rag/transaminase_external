from autoqchem_boosted.guassian_log_extractor import *
import math

def custom_sort(filepath_list):
    conf_ids = []
    for fp in filepath_list:
        conf_id = fp.split('/')[-1].split('.gjf.log')[0]
        conf_ids.append(int(conf_id.split('_')[-1]))
    
    sorted_idxs = sorted(range(len(conf_ids)), key=conf_ids.__getitem__)
    filepath_list = [filepath_list[i] for i in sorted_idxs]
    conf_ids = [conf_ids[i] for i in sorted_idxs]
    return filepath_list, conf_ids

def extract_from_rdmol(mol: Chem.Mol) -> tuple:
    """Extract information from RDKit Mol object with conformers.

    :param mol: rdkit molecule
    :type mol: rdkit.Chem.Mol
    :return: tuple(elements, conformer_coordinates, connectivity_matrix, charges)
    """

    # Take out elements, coordinates and connectivity matrix
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    charges = np.array([atom.GetFormalCharge() for atom in mol.GetAtoms()])

    conformer_coordinates = []
    for conformer in mol.GetConformers():
        coordinates = conformer.GetPositions()
        conformer_coordinates.append(coordinates)
    conformer_coordinates = np.array(conformer_coordinates)

    connectivity_matrix = Chem.GetAdjacencyMatrix(mol, useBO=True)

    return elements, conformer_coordinates, connectivity_matrix, charges


def get_rdkit_mol(elements: list, conformer_coordinates: np.ndarray,
                  connectivity_matrix: np.ndarray, charges: np.ndarray) -> Chem.Mol:
    """Construct an rdkit molecule from elements, positions, connectivity matrix and charges.

    :param elements: list of elements
    :type elements: list
    :param conformer_coordinates: list of conformers 3D coordinate arrays
    :type conformer_coordinates: np.ndarray
    :param connectivity_matrix:  connectivity matrix
    :type connectivity_matrix: np.ndarray
    :param charges: list of formal charges for each element
    :type charges: np.ndarray
    :return: rdkit.Chem.Mol
    """
    _RDKIT_BOND_TYPES = {
        1.0: Chem.BondType.SINGLE,
        1.5: Chem.BondType.AROMATIC,
        2.0: Chem.BondType.DOUBLE,
        3.0: Chem.BondType.TRIPLE,
        4.0: Chem.BondType.QUADRUPLE,
        5.0: Chem.BondType.QUINTUPLE,
        6.0: Chem.BondType.HEXTUPLE,
    }

    mol = Chem.RWMol()

    # Add atoms
    for element, charge in zip(elements, charges):
        atom = Chem.Atom(element)
        atom.SetFormalCharge(int(charge))
        mol.AddAtom(atom)

    # Add bonds
    for i, j in zip(*np.tril_indices_from(connectivity_matrix)):
        if i != j:
            bo = connectivity_matrix[i, j]
            if bo != 0:
                bond_type = _RDKIT_BOND_TYPES[float(bo)]
                mol.AddBond(int(i), int(j), bond_type)

    # Add conformers
    add_conformers_to_rdmol(mol, conformer_coordinates)

    mol = mol.GetMol()
    Chem.SanitizeMol(mol)

    return mol


def add_conformers_to_rdmol(mol: Chem.Mol, conformer_coordinates: np.ndarray) -> None:
    """Add conformers to RDKit Mol object.

    :param mol: rdkit mol object
    :type mol: rdkit.Chem.Mol
    :param conformer_coordinates: list of conformers 3D coordinate arrays
    :type conformer_coordinates: np.ndarray
    """

    conformer_coordinates = np.array(conformer_coordinates)
    if len(conformer_coordinates.shape) == 2:
        conformer_coordinates.reshape(-1, conformer_coordinates.shape[0], 3)

    for coordinates in conformer_coordinates:
        conformer = Chem.Conformer()
        for i, coord in enumerate(coordinates):
            point = Geometry.Point3D(*coord)
            conformer.SetAtomPosition(i, point)
        mol.AddConformer(conformer, assignId=True)


def rdmol_from_jobs(smi, filepath_list) -> Chem.Mol:
    """Create and rdkit molecule from a set of slurm jobs.

    :param jobs: list of jobs for the molecule
    :type jobs: list
    :param postDFT: if the DFT calculations are already available, the optimized geomtries will be used, if not the initial geometries.
    :type postDFT: bool
    :return: rdkit.Chem.Mol
    """

    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
    charges = np.array([atom.GetFormalCharge() for atom in mol.GetAtoms()])
    connectivity_matrix = Chem.GetAdjacencyMatrix(mol, useBO=True)

    conformer_coordinates = []
    energies = []
    labels_ok = True
    for fp in filepath_list:
        #print(fp)
        le = gaussian_log_extractor(smi, fp)

        le.get_descriptors()
        energies.append(le.descriptors['G'] * Hartree_in_kcal_per_mol)
        conformer_coordinates.append(le.geom[list('XYZ')].values)

        # verify that the labels are in the same order in gaussian after running it
        assert tuple(le.labels) == tuple(elements)
    
    rdmol = get_rdkit_mol(elements, conformer_coordinates, connectivity_matrix, charges)

    return rdmol, energies


def get_rmsd_rdkit(rdmol) -> np.ndarray:
    """Calculate RMSD row-wise with RDKit. This is a computationally slower version, but takes symmetry into account (explores permutations of atom order).

    :param rdmol: rdkit molecule
    :type rdmol: rdkit.Chem.Mol
    :return: np.ndarray
    """

    N = rdmol.GetNumConformers()
    rmsds = np.zeros(shape=(N, N))
    for i, j in itertools.combinations(range(N), 2):
        rms = AllChem.GetBestRMS(rdmol, rdmol, i, j)
        rmsds[i][j] = rms
        rmsds[j][i] = rms
    
    return rmsds


def prune_rmsds(rdmol, thres):
    """Get a list of conformer indices to keep

    :param rdmol: rdkit molecule
    :type rdmol: rdkit.Chem.Mol
    :param thres: RMSD threshold below which conformers are considered identical (in Angstrom)
    :type thres: float
    :return: list(indices of conformers to keep)
    """
    
    rmsds = get_rmsd_rdkit(rdmol)

    working_array = rmsds
    candidates = np.array(range(rdmol.GetNumConformers()))

    keep_list = []
    while len(working_array) > 0:
        keeper = candidates[0]
        keep_list.append(keeper)
        rmsd = working_array[0]
        mask = rmsd > thres
        candidates = candidates[mask]
        working_array = working_array[mask, :][:, mask]

    return keep_list

def read_json_file(desc_filepath):
    with open(desc_filepath) as project_file:    
        dft_descriptors = json.load(project_file)
    return dft_descriptors

def get_conf_descriptors(molecule_id, conf_ids_list):
    directory = f'{main_datasets_dir}/molecule_descriptors/{molecule_id}'
    filepaths = [f'{directory}/conf_{i}.json' for i in conf_ids_list]
    conformations = []

    for fp in filepaths:
        conformations.append(read_json_file(fp))
    return conformations

def get_lowest_energy_conformation(conformations, conf_ids_list):
    assert len(conformations) == len(conf_ids_list)
    energies = [c['descriptors']['G'] for c in conformations]
    lowest_energy_idx = np.argmin(energies)

    lowest_energy_geometry = {}
    lowest_energy_conf_id = conf_ids_list[lowest_energy_idx]
    lowest_energy_geometry['X'] = conformations[lowest_energy_idx]['atom_descriptors']['X']
    lowest_energy_geometry['Y'] = conformations[lowest_energy_idx]['atom_descriptors']['Y']
    lowest_energy_geometry['Z'] = conformations[lowest_energy_idx]['atom_descriptors']['Z']

    return lowest_energy_geometry, lowest_energy_conf_id

def get_metadata(conformation):
    metadata = {}
    metadata['smiles'] = conformation['smiles']
    metadata['labels'] = conformation['labels']
    metadata['atommaps'] = conformation['atommaps']
    return metadata

def compute_boltzmann_weights(conformations):
    free_energies = np.array(
        [Hartree_in_kcal_per_mol * c['descriptors']['G'] for c in conformations])  # in kcal_mol
    #print(free_energies)
    free_energies -= free_energies.min()  # to avoid huge exponentials
    weights = np.exp(-free_energies / (k_in_kcal_per_mol_K * T))
    weights /= weights.sum()

    # Check if weights sum to 1
    assert math.isclose(weights.sum(), 1.0, abs_tol=1e-5), "Weights do not sum to 1"
    return weights

def get_aggregate_descriptors(conformations, weights):
    avg_molecule_descriptors = {}
    avg_atom_descriptors = {}

    molecule_keys = [key for key,value in conformations[0]['descriptors'].items() if isinstance(value, (int, float))]
    atom_keys = [key for key,_ in conformations[0]['atom_descriptors'].items()]

    # molecule-level descriptors
    for key in molecule_keys:
        descriptor_vals = [float(c['descriptors'][key]) for c in conformations]
        avg_molecule_descriptors[key] = np.average(descriptor_vals, weights=weights)
    
    # atom-level descriptors
    for key in atom_keys:
        descriptor_vals = [c['atom_descriptors'][key] for c in conformations]
        assert len(set([len(i) for i in descriptor_vals])) == 1 # check that all conformations have the same atom descriptor length
        avg_atom_descriptors[key] = [np.average([float(val_list[i]) for val_list in descriptor_vals], weights=weights) for i in range(len(descriptor_vals[0]))]

    return avg_molecule_descriptors, avg_atom_descriptors

def extract_aggregate_descriptors():
    df = pd.read_csv(f'{main_datasets_dir}/exp_transaminase_unique_molecules.csv')

    for i,row in df.iterrows():
        smi, molecule_id = row['SMILES'], row['Molecule_ID']
        average_descriptors = {}

        # get unique conformations, based on defined rmsd threshold
        mol_directory = f'{directory}/multiconf_output_files/{molecule_id}'
        filepath_list, conf_ids = custom_sort(glob.glob(f'{mol_directory}/*.log'))
        rdmol, energies = rdmol_from_jobs(smi, filepath_list)
        keep = prune_rmsds(rdmol, thres=0.35)
        conf_ids_to_keep = [conf_ids[i] for i in keep]
        conformations = get_conf_descriptors(molecule_id, conf_ids_to_keep)
        lowest_energy_geometry, lowest_energy_conf_id = get_lowest_energy_conformation(conformations, conf_ids_to_keep)

        # extract and write boltzmann-averaged descriptors
        metadata = get_metadata(conformations[0])
        weights = compute_boltzmann_weights(conformations)
        avg_molecule_descriptors, avg_atom_descriptors = get_aggregate_descriptors(conformations, weights)

        for key,value in metadata.items(): average_descriptors[key] = value
        average_descriptors['lowest_energy_conformer'] = lowest_energy_conf_id
        average_descriptors['conf_ids_used'] = conf_ids_to_keep
        average_descriptors['weights'] = list(weights)
        average_descriptors['descriptors'] = avg_molecule_descriptors
        average_descriptors['atom_descriptors'] = avg_atom_descriptors

        # replace boltzmann-averaged geometry (meaningless) with lowest-energy geometry
        average_descriptors['atom_descriptors']['X'] = lowest_energy_geometry['X']
        average_descriptors['atom_descriptors']['Y'] = lowest_energy_geometry['Y']
        average_descriptors['atom_descriptors']['Z'] = lowest_energy_geometry['Z']

        descs_str = json.dumps(average_descriptors, indent=4)
        with open(f"{main_datasets_dir}/boltzmann_molecule_descriptors/{molecule_id}.json", "w") as outfile:
            outfile.write(descs_str)
    return