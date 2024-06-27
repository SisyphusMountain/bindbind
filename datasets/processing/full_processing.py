import sys
sys.path.append("/fs/pool/pool-marsot")
import pickle
import os
from tqdm import tqdm
from itertools import chain
import pandas as pd
import logging
from multiprocessing import Pool

import torch
from torch_geometric.data import HeteroData

from bindbind.datasets.processing.ligand_features.diffdock_ligand_features import create_diffdock_ligand_features
from bindbind.datasets.processing.protein_features.diffdock_protein_features import create_diffdock_protein_features
from bindbind.datasets.processing.ligand_features.tankbind_ligand_features import create_tankbind_ligand_features, reorder_compound_smiles
from bindbind.datasets.processing.protein_features.tankbind_protein_features import create_tankbind_protein_features, get_chains_in_contact_with_compound


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def make_full_dict(directory="/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/PDBBind_processed",
                   output_path_shape="/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/PDBBind_processed/{}/{}_full_data.pkl",
                   overwrite=False,):
    logger.info("Starting to retrieve protein paths and names.")
    protein_paths, protein_names = get_pdbbind_paths_and_names(directory)
    # logger.info("Retrieving ligand paths.")
    # sdf_paths, mol2_paths = get_ligand_paths(protein_names)
    # # reordering tankbind ligand atoms (no idea why we need to do that but ok)
    # logger.info("Reordering tankbind ligand atoms.")
    # for sdf_path, mol2_path in zip(sdf_paths.values(), mol2_paths.values()):
    #     reorder_compound_smiles(sdf_path, mol2_path)
    # # cutting off the protein chains far from the ligand
    # logger.info("Cutting off protein chains far from the ligand.")
    # indices_in_contact_with_compound_dict = {}
    # for protein_path, protein_name in tqdm(zip(protein_paths, protein_names)):
        # assert "_protein_processed.pdb" in protein_path
        # protein_chains_in_contact_with_ligand_path = protein_path.replace("_protein_processed.pdb", "_protein_chains_in_contact_with_ligand.pdb")
        # ligand_path = protein_path.replace("_protein_processed.pdb", "_ligand_renumbered.sdf")
        # _chains_in_contact_with_compound, indexes_in_contact_with_compound = get_chains_in_contact_with_compound(protein_path=protein_path,
        #                                     target_path=protein_chains_in_contact_with_ligand_path,
        #                                     ligand_path=ligand_path,
        #                                     cutoff=10.0)
        # indices_in_contact_with_compound_dict[protein_name] = indexes_in_contact_with_compound
    # logger.info("Performing P2rank pocket prediction.")
    # predictions_folder = "/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/p2rank_predictions_new"
    # with open(f"{predictions_folder}/p2rank_predictions.ds", "w") as f:
    #     for protein_name in protein_names:
    #         if protein_name != "2r1w": #apparently has no chain near ligand
    #             f.write(f"../PDBBind_processed/{protein_name}/{protein_name}_protein_chains_in_contact_with_ligand.pdb\n")
    # cmd = f"/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/p2rank_2.3/prank predict -o {predictions_folder} -threads 24 {predictions_folder}/p2rank_predictions.ds"
    # os.system(cmd)

    # protein_data = [(protein_name, protein_path, sdf_paths, mol2_paths, overwrite, output_path_shape.format(protein_name, protein_name)) for protein_name, protein_path in zip(protein_names, protein_paths)]
    # logger.info("Processing proteins in parallel.")

    # with Pool(24) as pool:
    #     list(tqdm(pool.starmap(process_protein, protein_data), total=len(protein_data)))
    

    # logger.info("Saving pockets in a dataframe.")
    # list_pockets_df = []
    # for protein_name in tqdm(protein_names):
    #     p2rank_file = f"/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/p2rank_predictions_new/{protein_name}_protein_chains_in_contact_with_ligand.pdb_predictions.csv"
    #     df = pd.read_csv(p2rank_file)
    #     df.columns = df.columns.str.strip()
    #     list_pockets_df.append(df.assign(name=protein_name))
    # pockets_df = pd.concat(list_pockets_df).reset_index(drop=True)
    # pockets_df.to_feather("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/p2rank_predictions.feather")
    # best_pockets_df = pockets_df[pockets_df["rank"]<=10]
    # csv_best_pockets_df = best_pockets_df[["name", "rank", "center_x", "center_y", "center_z"]]
    # csv_best_pockets_df.to_csv("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/p2rank_predictions.csv", index=False)
    csv_best_pockets_df = pd.read_csv("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/p2rank_predictions.csv")
    pockets_dict = csv_best_pockets_df.groupby("name").apply(lambda x: torch.tensor(x[['center_x', 'center_y', 'center_z']].values).to(torch.float32)).to_dict()
    logger.info("Updating feature dictionaries with pocket center coordinates.")
    compound_coordinates_dict = {}
    for protein_name in tqdm(protein_names):
        with open(output_path_shape.format(protein_name, protein_name), "rb") as f:
            full_dict = pickle.load(f)
        compound_coordinates_dict[protein_name] = full_dict["tankbind_ligand_atom_coordinates"]
        ligand_coordinates = full_dict["tankbind_ligand_atom_coordinates"]
        compound_center = ligand_coordinates.mean(dim=0, keepdim=True)
        full_dict["tankbind_compound_center"] = compound_center
        try:
            full_dict["tankbind_protein_pocket_center_coordinates"] = pockets_dict[protein_name]
        except:
            full_dict["tankbind_protein_pocket_center_coordinates"] = torch.zeros((0, 3), dtype=torch.float32)
        assert "tankbind_protein_pocket_center_coordinates" in full_dict
        protein_nodes_close_to_ligand = (torch.cdist(full_dict["tankbind_protein_alpha_carbon_coordinates"], ligand_coordinates) <= 10.0)
        protein_nodes_close_to_compound_center = (torch.cdist(full_dict["tankbind_protein_alpha_carbon_coordinates"], compound_center) <= 10.0)
        full_dict["tankbind_num_protein_nodes_close_to_ligand_and_in_contact_with_ligand"] = torch.logical_and(protein_nodes_close_to_ligand, protein_nodes_close_to_compound_center).sum()
        with open(output_path_shape.format(protein_name, protein_name), "wb") as f:
            pickle.dump(full_dict, f)
        full_dict["tankbind_residue_indices_in_contact_with_compound"] = torch.tensor(indices_in_contact_with_compound_dict[protein_name], dtype = torch.long)
    with open("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/compound_coordinates_dict.pkl", "wb") as f:
        pickle.dump(compound_coordinates_dict, f)

    logger.info("Done!")
    return None


def list_folders(directory):
    """Find all protein names in the dataset folder.
    
    The pdbbind folder contains folders named after the proteins contained in it.
    
    Parameters
    ----------
    directory: str
        the path of the pdbbind dataset folder.
    """
    entries = os.listdir(directory)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
    return folders

def get_pdbbind_paths_and_names(folder_path):
    protein_names = list_folders(folder_path)
    protein_paths = [f"{folder_path}/{protein}/{protein}_protein_processed.pdb" for protein in protein_names]
    
    # Filter out the paths that don't exist
    existing_protein_paths = [path for path in protein_paths if os.path.exists(path)]
    
    # Corresponding protein names for the existing paths
    existing_protein_names = [protein_names[i] for i in range(len(protein_paths)) if os.path.exists(protein_paths[i])]
    
    return existing_protein_paths, existing_protein_names

def get_ligand_paths(protein_names):
    dict_sdf_paths = {}
    dict_mol2_paths = {}
    for protein in protein_names:
        sdf_path = "/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/PDBBind_processed/{}/{}_ligand.sdf".format(protein, protein)
        mol2_path = "/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/PDBBind_processed/{}/{}_ligand.mol2".format(protein, protein) 
        dict_sdf_paths[protein] = sdf_path
        dict_mol2_paths[protein] = mol2_path
    return dict_sdf_paths, dict_mol2_paths




def process_protein(protein_name, protein_path, sdf_paths, mol2_paths, overwrite, output_path):
    """Obtain protein and ligand features in a dictionary.
    
    We get features as constructed in the TankBind paper, as well as DiffDock features.
    
    Parameters
    ----------
    protein_name: str
        the name of the protein currently processed
    protein_path: str
        the path of the protein currently processed
    sdf_paths: dict[str]
        all paths of the ligand sdf files
    mol2_paths: dict[str]
        all paths of the ligand mol2 files
    overwrite: bool
        whether to overwrite the output file if it already exists
    output_path: str
        the path where the output dictionary will be saved

    Returns
    -------
    None        
    """
    print("Processing protein", protein_name)
    if not overwrite and os.path.exists(output_path):
        return None

    ligand_path_sdf = sdf_paths[protein_name]
    ligand_path_mol2 = mol2_paths[protein_name]
    diffdock_ligand_features = create_diffdock_ligand_features(ligand_path_sdf, ligand_path_mol2)
    diffdock_protein_features = create_diffdock_protein_features(protein_path)
    assert "_protein_processed.pdb" in protein_path, "The protein path should contain '_protein_processed.pdb' in its name."
    tankbind_ligand_features = create_tankbind_ligand_features(ligand_path_sdf.replace(".sdf", "_renumbered.sdf"), None)
    tankbind_protein_features = create_tankbind_protein_features(protein_path.replace("_protein_processed.pdb", "_protein_chains_in_contact_with_ligand.pdb"))

    full_dict = dict(chain(diffdock_ligand_features.items(),
                           diffdock_protein_features.items(),
                           tankbind_ligand_features.items(),
                           tankbind_protein_features.items(),
                           {"protein_name": protein_name}.items()),
                            )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(full_dict, f)