import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from scipy.spatial.distance import cdist
from bindbind.datasets.processing.constants import LETTER_TO_NUM_TANKBIND, THREE_TO_ONE_TANKBIND
from Bio.PDB.PDBIO import Select, PDBIO
from Bio.PDB.PDBParser import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import chain
import math
#region helper

def create_tankbind_protein_features(protein_path):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("", protein_path) # no name required to parse the structure
    residue_list = clean_residues(structure.get_residues())
    residue_list = [residue for residue in residue_list if (('N' in residue) and (
        'CA' in residue) and ('C' in residue) and ('O' in residue))]
    sequence = "".join([THREE_TO_ONE_TANKBIND.get(
                                residue.resname) for residue in residue_list])
    all_backbone_atom_coordinates = []
    for residue in residue_list:
        all_backbone_atom_coordinates.append([list(residue['N'].get_coord()),
                                            list(residue['CA'].get_coord()),
                                            list(residue['C'].get_coord()),
                                            list(residue['O'].get_coord())])
    all_backbone_atom_coordinates = torch.tensor(all_backbone_atom_coordinates, dtype=torch.float32)

    sequence = torch.tensor(([LETTER_TO_NUM_TANKBIND[a] for a in sequence]), dtype=torch.long)
    mask = torch.isfinite(all_backbone_atom_coordinates.sum(dim=(1,2)))
    all_backbone_atom_coordinates = all_backbone_atom_coordinates[mask]
    node_features = create_node_features(all_backbone_atom_coordinates)
    edge_features = create_edge_features(all_backbone_atom_coordinates,top_k=30,num_rbf=16,num_embeddings=16)
    dict_tankbind_protein_features = dict(chain(node_features.items(),
                                                edge_features.items(),
                                                {"tankbind_one_letter_sequence":sequence}.items()),
                                                )
    return dict_tankbind_protein_features


def create_node_features(all_residue_atoms_coordinates):
    """
    Create node features for the protein graph.

    Parameters
    ----------
    node_coordinates: torch.Tensor
        Tensor of node coordinates.

    Returns
    -------

    """
    dihedrals = dihedrals_fn(all_residue_atoms_coordinates)
    assert dihedrals.shape[-1] == 6, f"Expected 6 dihedral angles, got {dihedrals.shape[-1]}"
    assert dihedrals.dim() == 2, f"Expected 2 dimensions, got {dihedrals.dim()}"
    alpha_carbon_coordinates = all_residue_atoms_coordinates[:, 1]
    orientations = orientations_fn(alpha_carbon_coordinates)
    assert orientations.shape[-1] == 3, f"Expected 3 dimensions, got {orientations.shape[-1]}"
    assert orientations.dim() == 3, f"Expected 3 dimensions, got shape {orientations.shape}"
    assert orientations.shape[-2] == 2, f"Expected 2 orientations, got {orientations.shape[-2]}"
    sidechains = sidechains_fn(all_residue_atoms_coordinates).unsqueeze(1)

    node_scalar_features = dihedrals
    node_vector_features = torch.cat([orientations, sidechains], dim=-2)

    return {"tankbind_protein_alpha_carbon_coordinates": alpha_carbon_coordinates,
            "tankbind_protein_node_scalar_features": node_scalar_features, 
            "tankbind_protein_node_vector_features": node_vector_features}

def create_edge_features(
    all_backbone_atom_coordinates, top_k, num_rbf, num_embeddings
):
    """
    Create edge features for the protein graph.

    Parameters
    ----------
    node_coordinates : torch.Tensor
        Tensor of node coordinates.
    top_k : int
        Number of nearest neighbors for the KNN graph.
    num_rbf : int
        Number of radial basis functions.
    num_embeddings : int
        Number of positional embeddings.

    Returns
    -------
    tuple
        Tuple containing edge index, edge scalar features, and edge vector features.
    """
    alpha_carbon_coordinates = all_backbone_atom_coordinates[:, 1]
    edge_index = torch_geometric.nn.knn_graph(alpha_carbon_coordinates, k=top_k)

    edge_positional_embeddings = edge_positional_embeddings_fn(
        edge_index, num_embeddings
    )
    edge_embeddings = (
        alpha_carbon_coordinates[edge_index[0]]
        - alpha_carbon_coordinates[edge_index[1]]
    )

    distance_matrix = edge_embeddings.norm(dim=-1)
    rbf = rbf_fn(distance_matrix, D_count=num_rbf)

    edge_scalar_features = torch.cat([rbf, edge_positional_embeddings], dim=-1)
    edge_vector_features = normalize_fn(edge_embeddings).unsqueeze(-2)

    return {"tankbind_protein_edge_index": edge_index,
            "tankbind_protein_edge_scalar_features": edge_scalar_features,
            "tankbind_protein_edge_vector_features": edge_vector_features}






#region helper

def clean_residues(residue_list):
    """We want to obtain the position of the different residues in the protein sequence,
    and we assimilate the position of the alpha carbon of the residue to the position of the residue.
    We therefore need to keep only the alpha carbons of the residues, which are indicated by the
    atom name 'CA'.
    Each residue should have a single alpha carbon atom."""
    clean_residue_list = []
    for residue in residue_list:
        # sequence name is the name of the protein we parsed (in our case,
        # we gave it the generic name "protein" when we parsed the structure)
        # model is the model number (0 in our case), which stands for the
        # different conformations of the protein.
        # chain is the chain identifier, which is a string: a protein can have
        # multiple chains, and each chain is identified by a letter.
        # residue_id is a tuple (hetero_flag, sequence_number, insertion_code)
        # hetero_flag is a boolean indicating whether the residue is a heteroatom,
        # We do not use the sequence number or the insertion code in this function.
        sequence_name, model, chain, residue_id = residue.full_id
        hetero_flag, sequence_number, insertion_code = residue_id
        if hetero_flag == " ":
            # residue.resname is a three-letter code for the residue name.
            if residue.resname not in THREE_TO_ONE_TANKBIND:
                continue
            if "CA" in residue:
                clean_residue_list.append(residue)
    return clean_residue_list


def get_chains_in_contact_with_compound(protein_path, target_path, ligand_path, cutoff):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("", protein_path)
    residues = list(structure.get_residues())
    protein_atoms = [atom for residue in residues for atom in residue.get_atoms()]
    protein_atom_coordinates = np.array([atom.coord for atom in protein_atoms])
    chains = np.array([atom.full_id[2] for atom in protein_atoms])
    ligand = Chem.MolFromMolFile(ligand_path)
    ligand_atom_coordinates = np.array(ligand.GetConformer().GetPositions())
    protein_ligand_distances = cdist(protein_atom_coordinates, ligand_atom_coordinates)
    protein_atom_is_near_a_ligand_atom = np.any(protein_ligand_distances < cutoff, axis=-1)
    chains_in_contact_with_compound = set(chains[protein_atom_is_near_a_ligand_atom])
    class SelectChainsInContactWithCompound(Select):
        def accept_residue(self, residue, chains_in_contact_with_compound=chains_in_contact_with_compound):
            _, _, chain, (_, _, _) = residue.full_id
            return chain in chains_in_contact_with_compound
    io = PDBIO()
    io.set_structure(structure)
    io.save(target_path, SelectChainsInContactWithCompound())
    indexes_in_contact_with_compound = []
    for residue in residues:
        _, _, chain, (_, _, _) = residue.full_id
        if chain in chains_in_contact_with_compound:
            indexes_in_contact_with_compound.append(residue.id[1])
    return chains_in_contact_with_compound, indexes_in_contact_with_compound
def get_pockets():
    """Create pockets using p2rank. We don't need it for now since I have already
    run p2rank on the data, so the output files obtained suffice."""
    raise NotImplementedError
def get_pockets_dict():
    """Obtain the pockets from the output files of p2rank"""
    raise NotImplementedError
def normalize_fn(tensor, dim=-1):
    """
    Normalize a `torch.Tensor` along dimension `dim` without `nan`s.
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )


def rbf_fn(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    """
    Compute an embedding of a pairwise distance matrix, by composing the values with
    gaussian functions of different means.

    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF


def orientations_fn(alpha_carbon_coordinates):
    """
    Compute differences between successive alpha carbon positions.

    Obtain the vectors representing differences between successive carbon positions, in a tensor.

    Parameters
    ----------
    alpha_carbon_coordinates : shape [n_residues, 3]

    Returns
    -------
    result : shape [n_residues, 2, 3]
    """
    forward = normalize_fn(alpha_carbon_coordinates[1:] - alpha_carbon_coordinates[:-1])
    backward = normalize_fn(
        alpha_carbon_coordinates[:-1] - alpha_carbon_coordinates[1:]
    )
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    assert forward.dim() == 2, f"Expected 2 dimensions, got shape {forward.shape}"
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def sidechains_fn(residue_atoms_coordinates):
    """
    Compute a feature related to the orientation of the amino acid in 3d space.

    An amino acid polymerizes with other amino acids by losing an oxygen from its
    carboxyl group to form a protein. As part of the protein, an amino acid has:
        an alpha carbon atom, which carries the side-chain of the protein.
        a beta carbon atom, carried by the alpha carbon atom
        a nitrogen atom, carried by the alpha carbon atom
        an oxygen atom, carried by the beta carbon atom
    We use the positions of the first three of these atoms to compute a bisector
    and a normal vector for the angle (beta, alpha, nitrogen).
    This computation comes from the AlphaFold 2 code.
    Parameters
    ----------
    residue_atoms_coordinates : shape [n_residues, 4, 3]
        The coordinates of N, CA, C, O for each residue

    Returns
    -------
    vec : shape [n_residues, 3]
        a weird vector.
    """
    nitrogen_coordinates, alpha_carbon_coordinates, beta_carbon_coordinates = (
        residue_atoms_coordinates[:, 0],
        residue_atoms_coordinates[:, 1],
        residue_atoms_coordinates[:, 2],
    )
    beta_carbon_coordinates_normalized, nitrogen_coordinates_normalized = (
        normalize_fn(beta_carbon_coordinates - alpha_carbon_coordinates),
        normalize_fn(nitrogen_coordinates - alpha_carbon_coordinates),
    )
    bisector = normalize_fn(
        beta_carbon_coordinates_normalized + nitrogen_coordinates_normalized
    )
    normal_vector = normalize_fn(
        torch.linalg.cross(beta_carbon_coordinates_normalized, nitrogen_coordinates_normalized, dim=-1)
    )
    vec = -bisector * (1 / 3) ** 0.5 - normal_vector * (2 / 3) ** 0.5

    return vec


def dihedrals_fn(protein_node_coordinates, eps=1e-7):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    """Obtain normal vectors for angles, and the cosine of
    dihedral angles.

    Parameters
    ----------
    protein_node_coordinates : tensor, shape [n_residues, 4, 3]
        The coordinates of the residues (coordinates of N, CA, C, O)

    Returns
    -------
    D_features : tensor, shape [n_residues, 6]
        a tensor encoding the cos and the sine of all 3 torsion angles along the chain
        N, CA, C, O
    """
    protein_node_coordinates = torch.reshape(
        protein_node_coordinates[:, :3], [3 * protein_node_coordinates.shape[0], 3]
    )  # shape [3*n_residue, 3]
    dX = protein_node_coordinates[1:] - protein_node_coordinates[:-1]
    U = normalize_fn(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    # shape [3(n_residues-1), 3]
    n_2 = normalize_fn(torch.linalg.cross(u_2, u_1, dim=-1), dim=-1)
    n_1 = normalize_fn(torch.linalg.cross(u_1, u_0, dim=-1), dim=-1)

    # Angle between normals (torsion angle) between successive bonds
    cosD = torch.sum(
        n_2 * n_1, -1
    )  # elementwise multiplication then sum is dot product. shape [3(n_residues-1)]
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(
        cosD
    )  # getting back to the torsion angles

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


def edge_positional_embeddings_fn(
    edge_index,
    num_embeddings,
    ):
    """
    Create a positional embedding based on the distance between alpha carbons of the protein.

    In the original Attention is All You Need paper, the positional embedding is based on the
    distances between tokens. Here, we use the distances between the indexes of the sequence residues,
    and transform this quantity into a sine and cosine vector which we concatenate.
    Parameters
    ----------
    edge_index : shape [2, num_edges_protein], dtype=LongTensor
    num_embeddings :
        How many different frequencies to use for the Attention is All You Need-style positional embedding.

    Returns
    -------
    positional_embedding : shape [n_edges_protein, 2*num_embeddings]
    """
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32)*(-math.log(10000.0)/num_embeddings)
    )
    # cos/sin positional embeddings, which have nothing to do with geometry but are inspired by Attention is All You Need
    phase = d.unsqueeze(-1) * frequency  # shape [n_edges_protein, num_embeddings]
    positional_embedding = torch.cat((torch.cos(phase), torch.sin(phase)), -1)
    return positional_embedding
#endregion helper