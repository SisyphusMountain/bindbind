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

import glob
import torch
torch.set_num_threads(1)
torch.multiprocessing.set_sharing_strategy('file_system')

import sys


import logging

from bindbind.torch_datasets.tankbind_dataset import TankBindTestDataset
from torch_geometric.loader import DataLoader
from bindbind.models.model import TankBindModel
from bindbind.torch_datasets.tankbind_dataloader import TankBindDataLoader
result_folder = "/fs/pool/pool-marsot/bindbind/experiments/ablations/regular/tankbind_predictions"
os.system(f"mkdir -p {result_folder}")

rdkit_folder = f"{result_folder}/rdkit/"
os.system(f"mkdir -p {rdkit_folder}")

from bindbind.datasets.processing.ligand_features.tankbind_ligand_features import read_molecule, create_tankbind_ligand_features, get_LAS_distance_constraint_mask

def compute_RMSD(a, b):
    # correct rmsd calculation.
    return torch.sqrt(((a-b)**2).sum(dim=-1)).mean()

def generate_conformation(mol):
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv2()
    try:
        rid = AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500, confId=0)
    except:
        mol.Compute2DCoords()
    mol = Chem.RemoveHs(mol)
    return mol
def write_with_new_coords(mol, new_coords, toFile):
    # put this new coordinates into the sdf file.
    w = Chem.SDWriter(toFile)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        x,y,z = new_coords[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))
    # w.SetKekulize(False)
    w.write(mol)
    w.close()
def generate_sdf_from_smiles_using_rdkit(smiles, rdkitMolFile, shift_dis=30, fast_generation=False):
    mol_from_rdkit = Chem.MolFromSmiles(smiles)
    if fast_generation:
        # conformation generated using Compute2DCoords is very fast, but less accurate.
        mol_from_rdkit.Compute2DCoords()
    else:
        mol_from_rdkit = generate_conformation(mol_from_rdkit)
    coords = mol_from_rdkit.GetConformer().GetPositions()
    new_coords = coords + np.array([shift_dis, shift_dis, shift_dis])
    write_with_new_coords(mol_from_rdkit, new_coords, rdkitMolFile)

def distance_loss_function(epoch, y_pred, x, protein_nodes_xyz, compound_pair_dis_constraint, LAS_distance_constraint_mask=None, mode=0):
    dis = torch.cdist(protein_nodes_xyz, x)
    dis_clamp = torch.clamp(dis, max=10)
    if mode == 0:
        interaction_loss = ((dis_clamp - y_pred).abs()).sum()
    elif mode == 1:
        interaction_loss = ((dis_clamp - y_pred)**2).sum()
    elif mode == 2:
        # probably not a good choice. x^0.5 has infinite gradient at x=0. added 1e-5 for numerical stability.
        interaction_loss = (((dis_clamp - y_pred).abs() + 1e-5)**0.5).sum()
    config_dis = torch.cdist(x, x)
    if LAS_distance_constraint_mask is not None:
        configuration_loss = 1 * (((config_dis-compound_pair_dis_constraint).abs())[LAS_distance_constraint_mask]).sum()
        # basic exlcuded-volume. the distance between compound atoms should be at least 1.22Å
        configuration_loss += 2 * ((1.22 - config_dis).relu()).sum()
    else:
        configuration_loss = 1 * ((config_dis-compound_pair_dis_constraint).abs()).sum()
    if epoch < 500:
        loss = interaction_loss
    else:
        loss = 1 * (interaction_loss + 5e-3 * (epoch - 500) * configuration_loss)
    return loss, (interaction_loss.item(), configuration_loss.item())


def distance_optimize_compound_coords(coords, y_pred, protein_nodes_xyz, 
                        compound_pair_dis_constraint, total_epoch=5000, loss_function=distance_loss_function, LAS_distance_constraint_mask=None, mode=0, show_progress=False):
    # random initialization. center at the protein center.
    c_pred = protein_nodes_xyz.mean(axis=0)
    x = (5 * (2 * torch.rand(coords.shape) - 1) + c_pred.reshape(1, 3).detach())
    x.requires_grad = True
    x_cuda = x.to("cuda:0").clone().detach().requires_grad_(True)
    y_pred_cuda = y_pred.to("cuda:0")
    protein_nodes_xyz_cuda = protein_nodes_xyz.to("cuda:0")
    compound_pair_dis_constraint_cuda = compound_pair_dis_constraint.to("cuda:0")
    if LAS_distance_constraint_mask is not None:
        LAS_distance_constraint_mask_cuda = LAS_distance_constraint_mask.to("cuda:0")
    else:
        LAS_distance_constraint_mask_cuda = None
    #optimizer = torch.optim.Adam([x], lr=0.1)
    optimizer = torch.optim.Adam([x_cuda], lr=0.1)
    #     optimizer = torch.optim.LBFGS([x], lr=0.01)
    loss_list = []
    rmsd_list = []
    if show_progress:
        it = tqdm(range(total_epoch))
    else:
        it = range(total_epoch)
    for epoch in it:
        optimizer.zero_grad()
        #loss, (interaction_loss, configuration_loss) = loss_function(epoch, y_pred, x, protein_nodes_xyz, compound_pair_dis_constraint, LAS_distance_constraint_mask=LAS_distance_constraint_mask, mode=mode)
        loss, (interaction_loss, configuration_loss) = loss_function(epoch, y_pred_cuda, x_cuda, protein_nodes_xyz_cuda, compound_pair_dis_constraint_cuda, LAS_distance_constraint_mask=LAS_distance_constraint_mask_cuda, mode=mode)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        rmsd = compute_RMSD(coords, x.detach())
        rmsd_list.append(rmsd.item())
        # break
    return x, loss_list, rmsd_list

def get_info_pred_distance(coords, y_pred, protein_nodes_xyz, compound_pair_dis_constraint, n_repeat=1, LAS_distance_constraint_mask=None, mode=0, show_progress=False):
    info = []
    if show_progress:
        it = tqdm(range(n_repeat))
    else:
        it = range(n_repeat)
    for repeat in it:
        # random initialization.
        # x = torch.rand(coords.shape, requires_grad=True)
        x, loss_list, rmsd_list = distance_optimize_compound_coords(coords, y_pred, protein_nodes_xyz, 
                            compound_pair_dis_constraint, LAS_distance_constraint_mask=LAS_distance_constraint_mask, mode=mode, show_progress=False)
        # rmsd = compute_rmsd(coords.detach().cpu().numpy(), movable_coords.detach().cpu().numpy())
        # print(coords, movable_coords)
        # rmsd = compute_rmsd(coords, x.detach())
        rmsd = rmsd_list[-1]
        try:
            info.append([repeat, rmsd, float(loss_list[-1]), x.detach().cpu().numpy()])
        except:
            info.append([repeat, rmsd, 0, x.detach().cpu().numpy()])
    info = pd.DataFrame(info, columns=['repeat', 'rmsd', 'loss', 'coords'])
    return info



def evaluate_model(model):
    device = model.device
    compound_dict = {}
    test = np.loadtxt("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/tankbind_splits/timesplit_test", dtype=str)
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
    data_loader = TankBindDataLoader(dataset, batch_size=2, shuffle=False, num_workers=8, pin_memory=True)


    logging.basicConfig(level=logging.INFO)

    affinity_pred_list = []
    y_pred_list = []
    if not os.path.exists(f"{result_folder}/affinity_pred_list.pkl"):

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
        with open(f"{result_folder}/affinity_pred_list.pkl", "wb") as f:
            pkl.dump(affinity_pred_list, f)
        with open(f"{result_folder}/y_pred_list.pkl", "wb") as f:
            pkl.dump(y_pred_list, f)
    else:
        with open(f"{result_folder}/affinity_pred_list.pkl", "rb") as f:
            affinity_pred_list = pkl.load(f)
        with open(f"{result_folder}/y_pred_list.pkl", "rb") as f:
            y_pred_list = pkl.load(f)


    output_info_chosen = dataset.pockets_df
    output_info_chosen['affinity'] = affinity_pred_list
    output_info_chosen['dataset_index'] = range(len(output_info_chosen))

 
    chosen = output_info_chosen.loc[output_info_chosen.groupby(['name'], sort=False)['affinity'].agg('idxmax')].reset_index()


    # TODO: Create compound_coordinates_dict
    device = "cpu"
    with open("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/compound_coordinates_dict.pkl", "rb") as f:
        compound_coordinates_dict = pkl.load(f)

    for idx, line in tqdm(chosen.iterrows(), total=chosen.shape[0]):
        protein_name = line['name']
        dataset_index = line['dataset_index']

        coords = compound_coordinates_dict[protein_name].to(device)
        protein_node_coordinates = dataset[dataset_index]["protein"].coordinates.to(device)
        n_compound = coords.shape[0]
        n_protein = protein_node_coordinates.shape[0]
        y_pred = y_pred_list[dataset_index].reshape(n_protein, n_compound).to(device)
        y = dataset[dataset_index]["protein", "distance_to", "compound"].edge_attr.reshape(n_protein, n_compound).to(device)
        compound_pair_dis_constraint = torch.cdist(coords, coords)
        rdkit_mol_path = f"{rdkit_folder}/{protein_name}_ligand.sdf"
        mol, _ = read_molecule(rdkit_mol_path, None)
        LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol).bool()
        pred_dist_info = get_info_pred_distance(coords, y_pred, protein_node_coordinates, compound_pair_dis_constraint,
                                    LAS_distance_constraint_mask=LAS_distance_constraint_mask,
                                    n_repeat=1, show_progress=False)

        toFile = f'{result_folder}/{protein_name}_tankbind_chosen.sdf'
        new_coords = pred_dist_info.sort_values("loss")['coords'].iloc[0].astype(np.double)
        write_with_new_coords(mol, new_coords, toFile)

    def rigid_transform_3D(A, B, correct_reflection=True):
        assert A.shape == B.shape

        num_rows, num_cols = A.shape
        if num_rows != 3:
            raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

        num_rows, num_cols = B.shape
        if num_rows != 3:
            raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

        # find mean column wise
        centroid_A = np.mean(A, axis=1)
        centroid_B = np.mean(B, axis=1)

        # ensure centroids are 3x1
        centroid_A = centroid_A.reshape(-1, 1)
        centroid_B = centroid_B.reshape(-1, 1)

        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B

        H = Am @ np.transpose(Bm)

        # sanity check
        #if linalg.matrix_rank(H) < 3:
        #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

        # find rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # special reflection case
        if np.linalg.det(R) < 0 and correct_reflection:
            print("det(R) < R, reflection detected!, correcting for it ...")
            Vt[2,:] *= -1
            R = Vt.T @ U.T

        t = -R @ centroid_A + centroid_B

        return R, t

    def compute_RMSD(a, b):
        # correct rmsd calculation.
        return np.sqrt((((a-b)**2).sum(axis=-1)).mean())

    def kabsch_RMSD(new_coords, coords):
        out = new_coords.T
        target = coords.T
        ret_R, ret_t = rigid_transform_3D(out, target, correct_reflection=False)
        out = (ret_R@out) + ret_t
        return compute_RMSD(target.T, out.T)


    ligand_metrics = []
    test = np.loadtxt("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/tankbind_splits/timesplit_test", dtype=str)
    for protein_name in test:
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
        kabsch = kabsch_RMSD(mol_pred_pos, true_ligand_pos)
        com_dist = compute_RMSD(mol_pred_pos.mean(axis=0), true_ligand_pos.mean(axis=0))
        ligand_metrics.append([protein_name, rmsd, com_dist, kabsch])


    # custom description function.
    def below_threshold(x, threshold=5):
        return 100 * (x < threshold).sum() / len(x)
    def custom_description(data):
        t1 = data
        t2 = t1.describe()
        t3 = t1.iloc[:,1:].apply(below_threshold, threshold=2, axis=0).reset_index(name='2A').set_index('index').T
        t31 = t1.iloc[:,1:].apply(below_threshold, threshold=5, axis=0).reset_index(name='5A').set_index('index').T
        t32 = t1.iloc[:,1:].median().reset_index(name='median').set_index('index').T
        t4 = pd.concat([t2, t3, t31, t32]).loc[['mean', '25%', '50%', '75%', '5A', '2A', 'median']]
        t5 = t4.T.reset_index()
        t5[['Methods', 'Metrics']] = t5['index'].str.split('_', 1, expand=True)
        t6 = pd.pivot(t5, values=['mean', 'median', '25%', '50%', '75%', '5A', '2A'], index=['Methods'], columns=['Metrics'])
        t6_col = t6.columns
        t6.columns = t6_col.swaplevel(0, 1)
        t7 = t6[sorted(t6.columns)]
        my_MultiIndex = [
                    (    'RMSD',  'mean'),
                    (    'RMSD',   '25%'),
                    (    'RMSD',  '50%'),
                    (    'RMSD',   '75%'),
                    (    'RMSD',  '5A'),
                    (    'RMSD', '2A'),
                    ('COM_DIST',  'mean'),
                    ('COM_DIST',   '25%'),
                    ('COM_DIST',  '50%'),
                    ('COM_DIST',   '75%'),
                    ('COM_DIST',  '5A'),
                    ('COM_DIST', '2A'),
                    (  'KABSCH',  'mean'),
                    (  'KABSCH',   'median'),
                    ]
        t8 = t7[my_MultiIndex]

        my_MultiIndex_fancy = [
                    (    'Ligand RMSD $\downarrow$', ' ', 'mean'),
                    (    'Ligand RMSD $\downarrow$', 'Percentiles $\downarrow$', '25%'),
                    (    'Ligand RMSD $\downarrow$', 'Percentiles $\downarrow$',  '50%'),
                    (    'Ligand RMSD $\downarrow$', 'Percentiles $\downarrow$',   '75%'),
                    (    'Ligand RMSD $\downarrow$', r'% Below Threshold $\uparrow$',  '5A'),
                    (    'Ligand RMSD $\downarrow$', r'% Below Threshold $\uparrow$', '2A'),
                    ('Centroid Distance $\downarrow$', ' ',  'mean'),
                    ('Centroid Distance $\downarrow$', 'Percentiles $\downarrow$',   '25%'),
                    ('Centroid Distance $\downarrow$', 'Percentiles $\downarrow$',  '50%'),
                    ('Centroid Distance $\downarrow$', 'Percentiles $\downarrow$',   '75%'),
                    ('Centroid Distance $\downarrow$', r'% Below Threshold $\uparrow$', '5A'),
                    ('Centroid Distance $\downarrow$', r'% Below Threshold $\uparrow$', '2A'),
                    (  'KABSCH', 'RMSD $\downarrow$',  'mean'),
                    (  'KABSCH', 'RMSD $\downarrow$',   'median'),
                    ]

        t8.columns = pd.MultiIndex.from_tuples(my_MultiIndex_fancy)
        return t8.round(2)

    d = pd.DataFrame(ligand_metrics, columns=['pdb', 'TankBind_RMSD', 'TankBind_COM_DIST', 'TankBind_KABSCH'])
    print(custom_description(d))
    return d