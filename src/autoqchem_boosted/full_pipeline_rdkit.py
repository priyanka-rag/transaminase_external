from autoqchem_boosted.imports import *
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# DEFINE CONFORMER GENERATION AND GAUSSIAN WORKFLOW TYPE AND PARAMETERS
workflow_type="equilibrium"
theory="B3LYP"
solvent="None"
light_basis_set="6-31G*"
heavy_basis_set="LANL2DZ"
generic_basis_set="genecp"
max_light_atomic_number=36
num_conf = 20

# Define list of molecules and IDs
unique_molecules_df = pd.read_csv(f'{main_datasets_dir}/exp_transaminase_unique_molecules.csv')
smiles_list = unique_molecules_df['SMILES'].to_list()
molecule_ids = unique_molecules_df['Molecule_ID'].to_list()
total_num_confs = 0

for smiles_str,molecule_id in zip(smiles_list,molecule_ids):
    # GENERATE CONFORMERS
    n_threads = os.cpu_count() - 1 #num available processors

    # initialize rdmol
    rdmol = Chem.AddHs(Chem.MolFromSmiles(smiles_str))

    # some parameters for conformer generation
    params = AllChem.EmbedParameters()
    params.useSymmetryForPruning = True
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.ETversion = 2
    params.pruneRmsThresh = 0.35 # diversity distance-based metric for conformer gen
    params.numThreads = n_threads
    params.randomSeed = 42

    # embed and optimize conformers
    cids = AllChem.EmbedMultipleConfs(rdmol, num_conf, params)
    results_MMFF = AllChem.MMFFOptimizeMoleculeConfs(rdmol, mmffVariant="MMFF94", numThreads=n_threads)
    elements = [atom.GetSymbol() for atom in rdmol.GetAtoms()]
    charges = np.array([atom.GetFormalCharge() for atom in rdmol.GetAtoms()])

    total_num_confs += len(cids)

    conformer_coordinates = [c.GetPositions() for c in rdmol.GetConformers()]
    conformer_coordinates = np.array(conformer_coordinates)

    assert conformer_coordinates.shape[0] == len(cids)
    assert conformer_coordinates.shape[1] == len(elements)
    assert conformer_coordinates.shape[2] == 3
    assert len(cids) == len(results_MMFF)
    
    connectivity_matrix = Chem.GetAdjacencyMatrix(rdmol, useBO=True)
    labeled_connectivity_matrix = pd.DataFrame(connectivity_matrix, index=elements, columns=elements)

    can = smiles_str
    #inchi = Chem.MolToInchi(rdmol)
    #inchikey = Chem.MolToInchiKey(rdmol)

    # add configuration info
    max_num_conformers = num_conf
    conformer_engine = "rdkit"

    # add charge and spin
    charge = sum(charges)
    spin = Descriptors.NumRadicalElectrons(rdmol) + 1

    #pdb.set_trace()
    molecule_workdir = os.path.join(directory, "multiconf_input_files", molecule_id)

    # GENERATE GAUSSIAN INPUT FILE 
    atomic_nums = set(atom.GetAtomicNum() for atom in rdmol.GetAtoms())
    light_elements = [GetSymbol(n) for n in atomic_nums if n <= max_light_atomic_number]
    heavy_elements = [GetSymbol(n) for n in atomic_nums if n > max_light_atomic_number]
    #pdb.set_trace()
    heavy_block = ""

    if heavy_elements:
        basis_set = generic_basis_set
        heavy_block += f"{' '.join(light_elements + ['0'])}\n"
        heavy_block += f"{light_basis_set}\n****\n"
        heavy_block += f"{' '.join(heavy_elements + ['0'])}\n"
        heavy_block += f"{heavy_basis_set}\n****\n"
        heavy_block += f"\n"
        heavy_block += f"{' '.join(heavy_elements + ['0'])}\n"
        heavy_block += f"{heavy_basis_set}\n"
    else:
        basis_set = light_basis_set

    # define type of job and parameters
    #pdb.set_trace()
    solvent_input = f"SCRF=(Solvent={solvent}) " if solvent.lower() != "none" else ""

    if workflow_type == "equilibrium":
        tasks = (
            f"opt=CalcFc {theory}/{basis_set} {solvent_input}scf=xqc ",
            f"freq {theory}/{basis_set} {solvent_input}volume NMR pop=NPA density=current Geom=AllCheck Guess=Read",
            f"TD(NStates=10, Root=1) {theory}/{basis_set} {solvent_input}volume pop=NPA density=current Geom=AllCheck Guess=Read"
        )

    gaussian_config = {'theory': theory,
                        'solvent': solvent,
                        'light_basis_set': light_basis_set,
                        'heavy_basis_set': heavy_basis_set,
                        'generic_basis_set': generic_basis_set,
                        'max_light_atomic_number': max_light_atomic_number}
    
    os.makedirs(molecule_workdir, exist_ok=True)

    # resources configuration
    n_processors = max(1, min(config['remote']['max_processors'],
                                rdmol.GetNumAtoms() // config['remote']['atoms_per_processor']))
    ram = n_processors * config['remote']['ram_per_processor']
    resource_block = f"%nprocshared={n_processors}\n%Mem={ram}GB\n"

    logger.info(f"Generating Gaussian input files for {rdmol.GetNumConformers()} conformations for {molecule_id}.")

    for conf_id, conf_coord in enumerate(conformer_coordinates):
        # set conformer
        #conf_name = f"{self.molecule.can}_conf_{conf_id}"
        conf_name = f"conf_{conf_id}"

        # coordinates block
        geom_np_array = np.concatenate((np.array([elements]).T, conf_coord), axis=1)
        coords_block = "\n".join(map(" ".join, geom_np_array))
    
        gau_output = ""

        # loop through the tasks in the workflow and create input file
        for i, task in enumerate(tasks):
            if i == 0:  # first task is special, coordinates follow
                gau_output += resource_block
                gau_output += f"%Chk={molecule_id}_{conf_name}_{i}.chk\n"
                gau_output += f"# {task}\n\n"
                gau_output += f"{conf_name}\n\n"
                gau_output += f"{charge} {spin}\n"
                gau_output += f"{coords_block.strip()}\n"
                gau_output += f"\n"
            else:
                gau_output += "\n--Link1--\n"
                gau_output += resource_block
                gau_output += f"%Oldchk={molecule_id}_{conf_name}_{i - 1}.chk\n"
                gau_output += f"%Chk={molecule_id}_{conf_name}_{i}.chk\n"
                gau_output += f"# {task}\n"
                gau_output += f"\n"

            gau_output += heavy_block  # this is an empty string if no heavy elements are in the molecule

        gau_output += f"\n\n"

        file_path = f"{molecule_workdir}/{conf_name}.gjf"
        with open(file_path, "w") as file:
            file.write(gau_output)
    #pdb.set_trace()        
print(total_num_confs)