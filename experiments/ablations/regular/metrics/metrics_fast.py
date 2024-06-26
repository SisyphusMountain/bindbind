import sys
sys.path.append("/fs/pool/pool-marsot")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
from tqdm import tqdm
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from torch_geometric.data import HeteroData
import torch_geometric
import glob
import torch
torch.set_num_threads(1)
torch.multiprocessing.set_sharing_strategy('file_system')
import logging
from bindbind.torch_datasets.tankbind_dataset import TankBindTestDataset, TankBindValDataset
from torch_geometric.loader import DataLoader
from bindbind.models.model import TankBindModel
from bindbind.torch_datasets.tankbind_dataloader import TankBindDataLoader
result_folder_test = "/fs/pool/pool-marsot/bindbind/experiments/ablations/regular/tankbind_predictions"
result_folder_val = "/fs/pool/pool-marsot/bindbind/experiments/ablations/regular/tankbind_predictions_val"
os.system(f"mkdir -p {result_folder_test}")
rdkit_folder_test = f"{result_folder_test}/rdkit/"
os.system(f"mkdir -p {rdkit_folder_test}")
rdkit_folder_val = f"{result_folder_val}/rdkit/"
os.system(f"mkdir -p {rdkit_folder_val}")
from bindbind.datasets.processing.ligand_features.tankbind_ligand_features import read_molecule, create_tankbind_ligand_features, get_LAS_distance_constraint_mask
from bindbind.experiments.ablations.regular.metrics.helper import compute_RMSD, write_with_new_coords, generate_sdf_from_smiles_using_rdkit, get_info_pred_distance, simple_custom_description, distribute_function




def evaluate_model_test(model,
                   batch_size=2,
                   num_workers=8,
                   result_folder=result_folder_test,
                   rdkit_folder=rdkit_folder_test):
    device = model.device
    compound_dict = {}
    test = np.loadtxt("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/tankbind_splits/timesplit_test", dtype=str)
    print("loading compound dict")
    if not os.path.exists(f"{result_folder}/pdbbind_test_compound_dict_based_on_rdkit.pt"):
        for protein_name in tqdm(test):
            mol, _ = read_molecule(f"/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/PDBBind_processed/{protein_name}/{protein_name}_ligand_renumbered.sdf", None)
            smiles = Chem.MolToSmiles(mol)

            rdkit_mol_path = f"{rdkit_folder}/{protein_name}_ligand.sdf"
            generate_sdf_from_smiles_using_rdkit(smiles, rdkit_mol_path, shift_dis=0)

            mol, _ = read_molecule(rdkit_mol_path, None)
            compound_dict[protein_name] = create_tankbind_ligand_features(rdkit_mol_path, None, has_LAS_mask=True)
        torch.save(compound_dict, f"{result_folder}/pdbbind_test_compound_dict_based_on_rdkit.pt")
    else:
        compound_dict = torch.load(f"{result_folder}/pdbbind_test_compound_dict_based_on_rdkit.pt")



    dataset = TankBindTestDataset("/fs/pool/pool-marsot/bindbind/datasets/tankbind_test",)
    dataset.compound_dict = torch.load(f"{result_folder}/pdbbind_test_compound_dict_based_on_rdkit.pt")
    data_loader = TankBindDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


    logging.basicConfig(level=logging.INFO)

    affinity_pred_list = []
    y_pred_list = []
    print("Predicting affinities and distances...")
    for data in tqdm(data_loader):
        data = data.to(device)
        previous_index_start=0
        this_index_start=0
        protein_sizes = torch.diff(data["protein"].ptr)
        compound_sizes = torch.diff(data["compound"].ptr)

        with torch.no_grad():
            y_pred, affinity_pred = model(data)
        affinity_pred_list.append(affinity_pred.detach().cpu())
        for i in range(data.batch_n):
            this_index_start += protein_sizes[i] * compound_sizes[i]
            y_pred_list.append((y_pred[previous_index_start:this_index_start]).detach().cpu())
            previous_index_start = this_index_start.clone()
    affinity_pred_list = torch.cat(affinity_pred_list)



    output_info_chosen = dataset.pockets_df
    output_info_chosen['affinity'] = affinity_pred_list
    output_info_chosen['dataset_index'] = range(len(output_info_chosen))
 
    chosen = output_info_chosen.loc[output_info_chosen.groupby(['name'], sort=False)['affinity'].agg('idxmax')].reset_index()


    # TODO: Create compound_coordinates_dict
    device = "cpu"
    with open("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/compound_coordinates_dict.pkl", "rb") as f:
        compound_coordinates_dict = pkl.load(f)
    max_compound_nodes = 0
    max_protein_nodes = 0
    list_mols = []
    list_complexes = []
    for idx, line in tqdm(chosen.iterrows(), total=chosen.shape[0]):
        protein_name = line['name']
        dataset_index = line['dataset_index']

        coords = compound_coordinates_dict[protein_name]
        protein_node_coordinates = dataset[dataset_index]["protein"].coordinates
        n_compound = coords.shape[0]
        n_protein = protein_node_coordinates.shape[0]
        y_pred = y_pred_list[dataset_index]
        y = dataset[dataset_index]["protein", "distance_to", "compound"].edge_attr
        rdkit_mol_path = f"{rdkit_folder}/{protein_name}_ligand.sdf"
        mol, _ = read_molecule(rdkit_mol_path, None)
        LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol).bool().flatten()
        max_compound_nodes = max(max_compound_nodes, n_compound)
        max_protein_nodes = max(max_protein_nodes, n_protein)
        complex = HeteroData()
        complex.protein_name = protein_name
        complex.protein_nodes_xyz = protein_node_coordinates
        complex.coords = coords
        complex.y_pred = y_pred
        complex.y = y
        complex.LAS_distance_constraint_mask = LAS_distance_constraint_mask
        list_complexes.append(complex)
        list_mols.append(mol)
    
    dataloader = DataLoader(list_complexes, batch_size=chosen.shape[0], shuffle=False,
                            follow_batch=["protein_nodes_xyz", "coords", "y_pred", "y", "LAS_distance_constraint_mask"])

    batch = next(iter(dataloader))
    coords_batched, coords_mask = torch_geometric.utils.to_dense_batch(batch.coords, batch.coords_batch)
    coords_pair_mask = torch.einsum("ij,ik->ijk", coords_mask, coords_mask)
    compound_pair_dis_constraint = torch.cdist(coords_batched, coords_batched)[coords_pair_mask]
    batch.compound_pair_dis_constraint = compound_pair_dis_constraint
    pred_dist_info = get_info_pred_distance(batch,
                                n_repeat=1, show_progress=False)
    
    for idx, line in tqdm(chosen.iterrows(), total=chosen.shape[0]):
        protein_name = line['name']
        toFile = f'{result_folder}/{protein_name}_tankbind_chosen.sdf'
        new_coords = pred_dist_info['coords'].iloc[idx].astype(np.double)
        write_with_new_coords(list_mols[idx], new_coords, toFile)





    ligand_metrics = []
    for idx, line in tqdm(chosen.iterrows(), total=chosen.shape[0]):
        protein_name = line['name']
        mol, _ = read_molecule(f"/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/PDBBind_processed/{protein_name}/{protein_name}_ligand_renumbered.sdf", None)
        mol_pred, _ = read_molecule(f"{result_folder}/{protein_name}_tankbind_chosen.sdf", None) # tankbind_chosen is the compound with predicted coordinates assigned by write_with_new_coords

        sm = Chem.MolToSmiles(mol)
        mol_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
        mol = Chem.RenumberAtoms(mol, mol_order)
        mol = Chem.RemoveHs(mol)
        true_ligand_pos = np.array(mol.GetConformer().GetPositions())

        sm = Chem.MolToSmiles(mol_pred)
        mol_order = list(mol_pred.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
        mol_pred = Chem.RenumberAtoms(mol_pred, mol_order)
        mol_pred = Chem.RemoveHs(mol_pred)
        mol_pred_pos = np.array(mol_pred.GetConformer().GetPositions())

        rmsd = np.sqrt(((true_ligand_pos - mol_pred_pos) ** 2).sum(axis=1).mean(axis=0))
        com_dist = compute_RMSD(mol_pred_pos.mean(axis=0), true_ligand_pos.mean(axis=0))
        ligand_metrics.append([protein_name, rmsd, com_dist,])




    d = pd.DataFrame(ligand_metrics, columns=['pdb', 'TankBind_RMSD', 'TankBind_COM_DIST',])
    return simple_custom_description(d)


def evaluate_model_val(model,
                   batch_size=2,
                   num_workers=8,
                   result_folder=result_folder_val,
                   rdkit_folder=rdkit_folder_val):
    device = model.device
    model.eval()
    compound_dict = {}
    val = np.loadtxt("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/tankbind_splits/timesplit_no_lig_overlap_val", dtype=str)
    print("loading compound dict")
    if not os.path.exists(f"{result_folder}/pdbbind_test_compound_dict_based_on_rdkit.pt"):
        list_inputs = []
        for protein_name in tqdm(val):
            mol, _ = read_molecule(f"/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/PDBBind_processed/{protein_name}/{protein_name}_ligand_renumbered.sdf", None)
            smiles = Chem.MolToSmiles(mol)

            rdkit_mol_path = f"{rdkit_folder}/{protein_name}_ligand.sdf"
            list_inputs.append((smiles, rdkit_mol_path))
        generate = lambda x: generate_sdf_from_smiles_using_rdkit(x[0], x[1], shift_dis=0)
        distribute_function(generate, list_inputs, n_jobs=8, description="creating coordinates")
        for protein_name in tqdm(val):
            mol, _ = read_molecule(rdkit_mol_path, None)
            compound_dict[protein_name] = create_tankbind_ligand_features(rdkit_mol_path, None, has_LAS_mask=True)
        torch.save(compound_dict, f"{result_folder}/pdbbind_test_compound_dict_based_on_rdkit.pt")
    else:
        compound_dict = torch.load(f"{result_folder}/pdbbind_test_compound_dict_based_on_rdkit.pt")



    dataset = TankBindValDataset("/fs/pool/pool-marsot/bindbind/datasets/tankbind_val",)
    dataset.compound_dict = torch.load(f"{result_folder}/pdbbind_test_compound_dict_based_on_rdkit.pt")
    data_loader = TankBindDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


    logging.basicConfig(level=logging.INFO)

    affinity_pred_list = []
    y_pred_list = []
    print("Predicting affinities and distances...")
    for data in tqdm(data_loader):
        data = data.to(device)
        previous_index_start=0
        this_index_start=0
        protein_sizes = torch.diff(data["protein"].ptr)
        compound_sizes = torch.diff(data["compound"].ptr)

        with torch.no_grad():
            y_pred, affinity_pred = model(data)
        affinity_pred_list.append(affinity_pred.detach().cpu())
        for i in range(data.batch_n):
            this_index_start += protein_sizes[i] * compound_sizes[i]
            y_pred_list.append((y_pred[previous_index_start:this_index_start]).detach().cpu())
            previous_index_start = this_index_start.clone()
    affinity_pred_list = torch.cat(affinity_pred_list)



    output_info_chosen = dataset.pockets_df
    output_info_chosen['affinity'] = affinity_pred_list
    output_info_chosen['dataset_index'] = range(len(output_info_chosen))
 
    chosen = output_info_chosen.loc[output_info_chosen.groupby(['name'], sort=False)['affinity'].agg('idxmax')].reset_index()


    # TODO: Create compound_coordinates_dict
    device = "cpu"
    with open("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/compound_coordinates_dict.pkl", "rb") as f:
        compound_coordinates_dict = pkl.load(f)
    max_compound_nodes = 0
    max_protein_nodes = 0
    list_mols = []
    list_complexes = []
    for idx, line in tqdm(chosen.iterrows(), total=chosen.shape[0]):
        protein_name = line['name']
        dataset_index = line['dataset_index']

        coords = compound_coordinates_dict[protein_name]
        protein_node_coordinates = dataset[dataset_index]["protein"].coordinates
        n_compound = coords.shape[0]
        n_protein = protein_node_coordinates.shape[0]
        y_pred = y_pred_list[dataset_index]
        y = dataset[dataset_index]["protein", "distance_to", "compound"].edge_attr
        rdkit_mol_path = f"{rdkit_folder}/{protein_name}_ligand.sdf"
        mol, _ = read_molecule(rdkit_mol_path, None)
        LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol).bool().flatten()
        max_compound_nodes = max(max_compound_nodes, n_compound)
        max_protein_nodes = max(max_protein_nodes, n_protein)
        complex = HeteroData()
        complex.protein_name = protein_name
        complex.protein_nodes_xyz = protein_node_coordinates
        complex.coords = coords
        complex.y_pred = y_pred
        complex.y = y
        complex.LAS_distance_constraint_mask = LAS_distance_constraint_mask
        list_complexes.append(complex)
        list_mols.append(mol)
    
    dataloader = DataLoader(list_complexes, batch_size=chosen.shape[0], shuffle=False,
                            follow_batch=["protein_nodes_xyz", "coords", "y_pred", "y", "LAS_distance_constraint_mask"])

    batch = next(iter(dataloader))
    coords_batched, coords_mask = torch_geometric.utils.to_dense_batch(batch.coords, batch.coords_batch)
    coords_pair_mask = torch.einsum("ij,ik->ijk", coords_mask, coords_mask)
    compound_pair_dis_constraint = torch.cdist(coords_batched, coords_batched)[coords_pair_mask]
    batch.compound_pair_dis_constraint = compound_pair_dis_constraint
    pred_dist_info = get_info_pred_distance(batch,
                                n_repeat=1, show_progress=False)
    
    for idx, line in tqdm(chosen.iterrows(), total=chosen.shape[0]):
        protein_name = line['name']
        toFile = f'{result_folder}/{protein_name}_tankbind_chosen.sdf'
        new_coords = pred_dist_info['coords'].iloc[idx].astype(np.double)
        write_with_new_coords(list_mols[idx], new_coords, toFile)





    ligand_metrics = []
    for idx, line in tqdm(chosen.iterrows(), total=chosen.shape[0]):
        protein_name = line['name']
        mol, _ = read_molecule(f"/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/PDBBind_processed/{protein_name}/{protein_name}_ligand_renumbered.sdf", None)
        mol_pred, _ = read_molecule(f"{result_folder}/{protein_name}_tankbind_chosen.sdf", None) # tankbind_chosen is the compound with predicted coordinates assigned by write_with_new_coords

        sm = Chem.MolToSmiles(mol)
        mol_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
        mol = Chem.RenumberAtoms(mol, mol_order)
        mol = Chem.RemoveHs(mol)
        true_ligand_pos = np.array(mol.GetConformer().GetPositions())

        sm = Chem.MolToSmiles(mol_pred)
        mol_order = list(mol_pred.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
        mol_pred = Chem.RenumberAtoms(mol_pred, mol_order)
        mol_pred = Chem.RemoveHs(mol_pred)
        mol_pred_pos = np.array(mol_pred.GetConformer().GetPositions())

        rmsd = np.sqrt(((true_ligand_pos - mol_pred_pos) ** 2).sum(axis=1).mean(axis=0))
        com_dist = compute_RMSD(mol_pred_pos.mean(axis=0), true_ligand_pos.mean(axis=0))
        ligand_metrics.append([protein_name, rmsd, com_dist,])




    d = pd.DataFrame(ligand_metrics, columns=['pdb', 'TankBind_RMSD', 'TankBind_COM_DIST',])
    return simple_custom_description(d)
