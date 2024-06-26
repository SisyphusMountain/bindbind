import sys
sys.path.append('/fs/pool/pool-marsot')

import numpy as np

import torch
from scipy.spatial.distance import cdist

from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs
from rdkit.Geometry import Point3D
import prody as pr
from bindbind.datasets.processing.constants import ATOM_ORDER_DIFFDOCK, THREE_TO_ONE_DIFFDOCK, ONE_TO_NUMBER_DIFFDOCK, NUMBER_TO_ONE_DIFFDOCK, ONE_TO_THREE_DIFFDOCK,CHI
from torch_geometric.nn.pool import knn_graph
from collections import defaultdict
from itertools import chain

def create_diffdock_protein_features(protein_path):
    node_features = create_node_features(protein_path)
    edge_features = create_edge_features(node_features)
    return dict(chain(node_features.items(), edge_features.items()))


def create_node_features(protein_path):
    """
    Create a dictionary of node features for the protein, according to the DiffDock featurization found in https://github.com/gcorso/DiffDock/blob/5238b18d4a4036f6cdf530ff08fb0fbb8c508614/datasets/process_mols.py
    

    Parameters
    ----------
    protein_path : str
        The path to the protein file.

    Returns
    -------
    dict
        A dictionary containing the following
        - diffdock_sequences : list of str
            The sequences of the protein, separated by chains.
        - diffdock_node_features : torch.tensor : shape [N, 1]
            The node features of the protein.
        - diffdock_chain_ids : torch.tensor : shape [N]
            The chain ids of the residues in the protein.
        - diffdock_alpha_carbon_coordinates : torch.tensor : shape [N, 3]
            The coordinates of the alpha carbons of the residues in the protein.
        - diffdock_residue_atom_coordinates : torch.tensor : shape [N, 14, 3]
            The coordinates of the atoms of the residues in the protein (not just backbone atoms, but also
            the other atoms of the residue).
        - diffdock_sidechain_vector_features : torch.tensor : shape [N, 12]
            The side chain vector features of the residues in the protein.


    """
    pdb, seq = read_protein_file(protein_path)
    residue_atom_coordinates = get_atom_coordinates(pdb)
    one_hot = get_onehot_sequence(seq)
    chain_ids = np.zeros(len(one_hot))
    residue_chain_ids = pdb.ca.getChids() # There are different chains with different ids in some proteins.
    res_segment_ids = pdb.ca.getSegnames()
    residue_chain_ids = np.array([s + c for s, c in zip(res_segment_ids, residue_chain_ids)])
    unique_residue_indexes = np.unique(residue_chain_ids)
    sequences = []

    for i, id in enumerate(unique_residue_indexes):
        chain_ids[residue_chain_ids == id] = i

        s = np.argmax(one_hot[residue_chain_ids == id], axis=1)
        s = ''.join([NUMBER_TO_ONE_DIFFDOCK[residue_index] for residue_index in s])
        sequences.append(s)
    chi_angles = get_chi_angles(residue_atom_coordinates, seq)

    n_rel_pos, c_rel_pos = residue_atom_coordinates[:, 0, :] - residue_atom_coordinates[:, 1, :], residue_atom_coordinates[:, 2, :] - residue_atom_coordinates[:, 1, :]
    side_chain_vecs = np.concatenate([chi_angles / 360, n_rel_pos, c_rel_pos], axis=1)
    # -10 is an impossible angle, so we can use it to replace NaNs in the chi_angles
    side_chain_vecs = np.nan_to_num(side_chain_vecs, nan=-10)
    # Build the k-NN graph
    residue_alpha_carbon_coordinates = np.array(residue_atom_coordinates[:, 1, :])


    res_names_list = [ONE_TO_THREE_DIFFDOCK[seq[i]] if seq[i] in ONE_TO_THREE_DIFFDOCK else 'misc' for i in range(len(seq))]
    feature_list = [[safe_index(allowable_features['possible_amino_acids'], res)] for res in res_names_list]
    node_feat = np.array(feature_list)

    return {
            "diffdock_protein_sequences": sequences,
            "diffdock_protein_node_features": torch.tensor(node_feat, dtype=torch.float),
            "diffdock_protein_chain_ids": torch.tensor(chain_ids, dtype=torch.long),
            "diffdock_protein_alpha_carbon_coordinates": torch.tensor(residue_atom_coordinates[:, 1, :], dtype=torch.float),
            "diffdock_protein_residue_atom_coordinates": torch.tensor(residue_atom_coordinates, dtype=torch.float),
            "diffdock_protein_sidechain_vector_features": torch.tensor(side_chain_vecs, dtype=torch.float),
    }

def create_edge_features(node_features, max_neighbors=30):
    alpha_carbon_coordinates = node_features["diffdock_protein_alpha_carbon_coordinates"]
    edge_index = knn_graph(alpha_carbon_coordinates, k=max_neighbors if max_neighbors else 32)
    return {
        'diffdock_protein_alpha_carbon_knn_edge_index': edge_index
    }

#region helper

periodic_table = GetPeriodicTable()


allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}
def read_protein_file(protein_path):
    """Read a protein file using prody, and return the prody object and the sequence of the protein.
    
    Parameters
    ----------
    protein_path : str
        The path to the protein file.

    Returns
    -------
    prody.AtomGroup
        The protein object.

    """
    pdb = pr.parsePDB(protein_path)
    seq = pdb.ca.getSequence()
    return pdb, seq

def safe_index(l, e):
    """ Return index of element e in list l. If e is not present, return the last index."""
    try:
        return l.index(e)
    except:
        return len(l) - 1

def get_atom_coordinates(pdb):
    """
    
    Parameters
    ----------
    pdb : prody.AtomGroup
        The protein, opened with prody.

    Returns
    -------
    residue_coordinates : np.array
        The coordinates of the atoms of the residues in the protein (not just backbone atoms, but also
        the other atoms of the residue).
    """
    residue_indices = sorted(set(pdb.ca.getResindices()))
    residue_coordinates = np.zeros((len(residue_indices), 14, 3))
    for i, residue_index in enumerate(residue_indices):
        selected_residue = pdb.select(f'resindex {residue_index}')
        residue_name = selected_residue.getResnames()[0]
        for j, name in enumerate(ATOM_ORDER_DIFFDOCK[THREE_TO_ONE_DIFFDOCK[residue_name] if residue_name in THREE_TO_ONE_DIFFDOCK else 'X']):
            selected_residue_name = selected_residue.select(f'name {name}')
            if selected_residue_name is not None:
                residue_coordinates[i, j, :] = np.array(selected_residue_name.getCoords()[0])
            else:
                residue_coordinates[i, j, :] = np.array([float("nan"), float("nan"), float("nan")])
    return residue_coordinates

def get_onehot_sequence(protein_sequence):
    """Create a one-hot encoding of a protein sequence.
    Parameters
    ----------
    protein_sequence : str
        The protein sequence.
    Returns
    -------
    np.array
        The one-hot encoding of the protein sequence (shape: (len(protein_sequence), 20)).
        """
    onehot = np.zeros((len(protein_sequence), 20))
    for i, residue in enumerate(protein_sequence):
        index = ONE_TO_NUMBER_DIFFDOCK[residue] if residue in ONE_TO_NUMBER_DIFFDOCK else 7 # 7 is the index for GLY
        onehot[i, index] = 1
    return onehot

def dihedral_angle_batch(residue_atom_coordinates):
    """Backbone dihedral angles for a batch of protein residues.
    



    Parameters
    ----------
    residue_atom_coordinates : np.array
        The coordinates of the atoms of the residues in the protein (not just backbone atoms, but also
        the other atoms of the residue).
        
    Returns
    -------
    np.array
        The dihedral angles of the residues in the protein.
    
    
    """
    # indexes 0, 1, 2, 3 are the N, CA, C, O atoms of the residue
    b0 = residue_atom_coordinates[:, 0] - residue_atom_coordinates[:, 1]  # N-CA
    b1 = residue_atom_coordinates[:, 1] - residue_atom_coordinates[:, 2]  # CA-C
    b2 = residue_atom_coordinates[:, 2] - residue_atom_coordinates[:, 3]  # C-O
    
    n1 = np.cross(b0, b1)  # N-CA-C
    n2 = np.cross(b1, b2)  # CA-C-O
    
    m1 = np.cross(n1, b1/(np.linalg.norm(b1, axis=1, keepdims=True)+np.where(np.linalg.norm(b1, axis=1, keepdims=True)==0, 1, 0)))
    
    x = np.sum(n1 * n2, axis=1)
    y = np.sum(m1 * n2, axis=1)
    
    deg = np.degrees(np.arctan2(y, x))
    
    deg[deg < 0] += 360
    
    return deg

def get_dihedral_atom_indices(residue_name, dihedral_angle_index):
    """Return the atom indices for the specified dihedral angle.

    There are several dihedral angles in the residue. Each atom composing a dihedral angles has
    a certain index in the residue. We try to find the indices of the atoms composing the dihedral
    angle in the residue.

    Parameters
    ----------
    residue_name : str
        The residue name (one letter code).
    dihedral_angle_index : int
        The number of the dihedral angle. Up to 4 dihedral angles in prolin.


    Returns
    -------
    np.array : dtype=int : shape depends on the residue.
        The indices of the atoms composing the dihedral angle in the residue.        
    """
    # To compute a dihedral angle, we use the coordinates of 4 atoms. We obtain the atom names
    # corresponding to any dihedral angle of any residue in the CHI dictionary, which contains quadruples.
    if residue_name not in CHI:
        return np.array([float("nan")]*4)
    if dihedral_angle_index not in CHI[residue_name]:
        return np.array([float("nan")]*4)
    # CHI[residue_name][dihedral_angle_index] will give the names of the atoms for a given residue and a dihedral angle index
    # of this residue.  ATOM_ORDER_DIFFDOCK[residue_name].index(x) recovers the index corresponding to the atom name x.
    return np.array([ATOM_ORDER_DIFFDOCK[residue_name].index(x) for x in CHI[residue_name][dihedral_angle_index]], dtype=int)

dihedral_atom_indices = defaultdict(list)
for residue_name in ATOM_ORDER_DIFFDOCK.keys():
    for i in range(1, 5): # dihedral angles are numbered from 1 to 4 in the dictionary
        dihedral_atom_indices[residue_name].append(get_dihedral_atom_indices(residue_name, i))
    dihedral_atom_indices[residue_name] = np.stack(dihedral_atom_indices[residue_name])

def get_chi_angles(residue_atom_coordinates, protein_sequence):
    """
    Compute the chi angles of the sidechains of a protein.
    Parameters
    ----------
    residue_atom_coordinates : np.array
        The coordinates of the atoms of the residues in the protein (not just backbone atoms, but also
        the other atoms of the residue).
    sequence 

    Returns
    -------
    numpy array of shape (N, 4)
        Array contains chi angles of sidechains in row-order of residue indices in prody_pdb.
        If a chi angle is not defined for a residue, due to missing atoms or GLY / ALA, it is set to float("nan").
    """
    residue_onehot_encoding = get_onehot_sequence(protein_sequence)
    dihedral_indices = np.array([dihedral_atom_indices[NUMBER_TO_ONE_DIFFDOCK[residue_index]] for residue_index in np.argmax(residue_onehot_encoding, axis=-1)])
    dihedral_indices_long = dihedral_indices.astype(int)
    N = residue_atom_coordinates.shape[0]
    # there can be 0 to 4 dihedral angles in a residue. The atom indices for non-existing dihedral angles are set to nan. We remove them now.
    mask = np.isnan(dihedral_indices)
    # setting the indices of the atoms for non-existing dihedral angles to 0
    dihedral_indices_long[mask] = 0
    Z = residue_atom_coordinates[np.arange(N).reshape(N,1,1), dihedral_indices_long, :]
    Z[mask] = float("nan")
    chi_angles = dihedral_angle_batch(Z.reshape((-1, 4, 3))).reshape(N, 4)
    return chi_angles

#endregion helper