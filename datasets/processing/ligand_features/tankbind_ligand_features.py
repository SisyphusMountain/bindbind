import torch
import scipy
import sys
from io import StringIO
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from bindbind.datasets.processing.constants import ATOM_VOCAB_TANKBIND, DEGREE_VOCAB, NUM_HS_VOCAB, TOTAL_VALENCE_VOCAB, FORMAL_CHARGE_VOCAB, BOND_TYPE_VOCAB, BOND_DIR_VOCAB, BOND_STEREO_VOCAB



def create_tankbind_ligand_features(ligand_path_sdf, ligand_path_mol2, has_LAS_mask=True):
    """
    Create the molecule features.

    
    Parameters
    ----------
    mol: rdkit.Chem.rdchem.Mol
        The molecule to featurize
    has_LAS_mask:
        Whether to use the LAS mask (local atomic structures) to encode the known information about the structure.

    """
    mol, problem = read_molecule(ligand_path_sdf, ligand_path_mol2)

    if has_LAS_mask:
        LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol)
    else:
        LAS_distance_constraint_mask = None
    # Obtain the coordinates of the atoms
    coords = mol.GetConformer().GetPositions()
    compound_node_coordinates = torch.tensor(coords, dtype=torch.float32)
    # Enumerate over atoms to obtain atom features
    atom_features_list = []
    list_atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())]
    for atom in list_atoms:
        # atom_embedding(atom) is a list, so atom_features_list is a list of lists
        atom_features_list.append(atom_embedding(atom))
    atom_features = torch.tensor(atom_features_list)
    # Enumerate over bonds to obtain bond features
    bond_features_list = []
    edge_list = []  # list of edges to be used in torch_geometric.data.HeteroData
    list_bonds = [mol.GetBondWithIdx(i) for i in range(mol.GetNumBonds())]
    for bond in list_bonds:
        edge_type = bond.GetBondType()
        if edge_type not in BOND_TYPE_VOCAB:
            continue
        feature = bond_embedding(bond)
        bond_features_list += [feature, feature]
        edge_list.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_list.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
    edge_index = torch.tensor(edge_list).T
    edge_features = torch.tensor(bond_features_list)

    # Obtain the pairwise distance distribution
    pairwise_distance_distribution = get_compound_pairwise_distance_distribution(coords,
                                                                        LAS_distance_constraint_mask)
    # BUG The function Chem.RemoveHs from rdkit.Chem may leave a hydrogen, for example try to open 3kqs.sdf and apply RemoveHs.
    # However, performing Chem.MolFromSmiles(Chem.MolToSmiles(mol)) will remove this hydrogen. I do not understand this behavior.
    assert len(compound_node_coordinates) == len(
        atom_features), "There may have been a problem in the hydrogen removing process in the function read_molecule."
    return {"tankbind_ligand_atom_coordinates":compound_node_coordinates, 
            "tankbind_ligand_atom_features":atom_features,
            "tankbind_ligand_edge_index":edge_index,
            "tankbind_ligand_edge_features":edge_features,
            "tankbind_ligand_pairwise_distance_distribution":pairwise_distance_distribution
            }


def onehot(x, vocab, add_unknown_possibility=False):
    if x in vocab:
        if isinstance(vocab, dict):
            index = vocab[x]
        else:
            index = vocab.index(x)
    else:
        index = -1
    if add_unknown_possibility:
        feature = [0] * (len(vocab) + 1)
        feature[index] = 1
    else:
        feature = [0] * len(vocab)
        if index == -1:
            raise ValueError(
                "Unknown value `%s`. Available vocabulary is `%s`" % (x, vocab))
        feature[index] = 1
    return feature

def atom_embedding(atom):
    """The atom property_prediction embedding from torchdrug 0.1.2.:
    https://github.com/DeepGraphLearning/torchdrug/blob/489677cbaa60171f0de7d80605185ac26cf12232/torchdrug/data/feature.py#L168
    Takes as input an atom from an rdkit Molecule"""
    return onehot(atom.GetSymbol(), ATOM_VOCAB_TANKBIND, add_unknown_possibility=True) + \
        onehot(atom.GetDegree(), DEGREE_VOCAB, add_unknown_possibility=True) + \
        onehot(atom.GetTotalNumHs(), NUM_HS_VOCAB, add_unknown_possibility=True) + \
        onehot(atom.GetTotalValence(), TOTAL_VALENCE_VOCAB, add_unknown_possibility=True) + \
        onehot(atom.GetFormalCharge(), FORMAL_CHARGE_VOCAB, add_unknown_possibility=True) + \
        [atom.GetIsAromatic()]


def bond_length(bond):
    """Function obtained from torchdrug 0.1.2.:
    https://github.com/DeepGraphLearning/torchdrug/blob/489677cbaa60171f0de7d80605185ac26cf12232/torchdrug/data/feature.py#L242C1-L250C27
    """
    mol = bond.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    h = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
    t = conformer.GetAtomPosition(bond.GetEndAtomIdx())
    return [h.Distance(t)]

def reorder_compound_smiles(ligand_path_sdf, ligand_path_mol2):
    mol, problem = read_molecule(ligand_path_sdf, ligand_path_mol2)
    # rewriting molecule with canonical smiles order
    mol_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"])
    mol = Chem.RenumberAtoms(mol, mol_order)
    ligand_path_sdf_renumbered = ligand_path_sdf.replace(".sdf", "_renumbered.sdf")
    w = Chem.SDWriter(ligand_path_sdf_renumbered)
    w.write(mol)
    w.close()

def bond_embedding(bond):
    """The default torchdrug 0.1.2. bond embedding:
    https://github.com/DeepGraphLearning/torchdrug/blob/489677cbaa60171f0de7d80605185ac26cf12232/torchdrug/data/feature.py#L220
    """
    return onehot(bond.GetBondType(), BOND_TYPE_VOCAB, add_unknown_possibility=False) + \
        onehot(bond.GetBondDir(), BOND_DIR_VOCAB, add_unknown_possibility=False) + \
        onehot(bond.GetStereo(), BOND_STEREO_VOCAB, add_unknown_possibility=False) + \
        [int(bond.GetIsConjugated())] + \
        bond_length(bond)

def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

def n_hops_adj(adjacency_matrix, n_hops):
    """
    Create the n-hop adjacency matrix.
    At each point in time, adjacency_matrices[i] is the adjacency matrix of the i-hop graph.
    Therefore, the sum over all (adjacency_matrix[i] - adjacency_matrix[i-1]) gives the nodes that 
    can be reached in i hops but not in i-1 hops.
    The sum over all (adjacency_matrix[i] - adjacency_matrix[i-1]) for i in [1, n_hops] gives the
    adjacency matrix for n hops, weighted by the minimum number of hops needed to get from
    node i to node j.
    """
    matrix_dim = adjacency_matrix.size(0)
    device = adjacency_matrix.device
    adjacency_matrices = [torch.eye(matrix_dim, dtype=torch.long, device=device),
                          binarize(adjacency_matrix + torch.eye(matrix_dim, dtype=torch.long, device=device))]

    for i in range(2, n_hops+1):
        adjacency_matrices.append(
            binarize(adjacency_matrices[i-1] @ adjacency_matrices[1]))
    extend_mat = torch.zeros_like(adjacency_matrix)

    for i in range(1, n_hops+1):
        extend_mat += (adjacency_matrices[i] - adjacency_matrices[i-1]) * i

    return extend_mat

def get_LAS_distance_constraint_mask(mol):
    """
    As in the TankBind article, we use a LAS (local atomic structures) mask.
    In the so-called self-docking setting, the ligand has no known geometric structure,
    and we only know the bonds between atoms locally. We therefore use a LAS mask to encode the 
    know information about the structure: it is known that bond lengths are relatively fixed,
    whereas torsion angles are not.

    We take into account the following information:
    1) The 1-hop neighborhoods
    2) The 2-hop neighborhoods
    3) Belonging to aromatic rings 

    BUG: We encode belonging to aromatic rings by adding 1 to the adjacency matrix
    but this will not allow us to differentiate between belonging to the same
    aromatic ring, and being a distance of 1 or 2 apart.
    This is a weird approach, but it does not pose a problem in the code since mol_mask
    is only used as a binary mask eventually, to encode the condition of belonging to the
    same ring or being 1 or 2 apart.

    FIXME: GetSymSSSR does not actually encode being part of aromatic rings,
    but simply being part of a ring, which is very different.
    Also, not differentiating between 2-hop neighborhoods and aromatic rings feels
    very weak, as these structures have very different properties.
    """
    # Get the adj
    adj = Chem.GetAdjacencyMatrix(mol)
    adj = torch.from_numpy(adj)
    extend_adj = n_hops_adj(adj, 2)
    #
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        for i in ring:
            for j in ring:
                if i == j:
                    continue
                else:
                    extend_adj[i][j] += 1
    mol_mask = binarize(extend_adj)
    return mol_mask


def get_compound_pairwise_distance_distribution(compound_node_coordinates, LAS_distance_constraint_mask=None):
    """
    Compute pairwise distances between atoms of the compound, with an
    optional mask.
    
    The pairwise distances are then classified into bins, 
    
    """
    compound_atom_pairwise_distances = scipy.spatial.distance.cdist(compound_node_coordinates, compound_node_coordinates)
    bin_size = 1
    bin_min = -0.5
    bin_max = 15
    if LAS_distance_constraint_mask is not None:
        compound_atom_pairwise_distances[LAS_distance_constraint_mask == 0] = bin_max
        # diagonal is zero.
        for i in range(compound_atom_pairwise_distances.shape[0]):
            compound_atom_pairwise_distances[i, i] = 0
    compound_atom_pairwise_distances = torch.tensor(compound_atom_pairwise_distances, dtype=torch.float)
    compound_atom_pairwise_distances[compound_atom_pairwise_distances > bin_max] = bin_max # Could be done with torch.clamp
    compound_pairwise_distance_bin_index = torch.div(
        compound_atom_pairwise_distances - bin_min, bin_size, rounding_mode='floor').long()
    compound_pairwise_distance_one_hot = torch.nn.functional.one_hot(
        compound_pairwise_distance_bin_index, num_classes=16)
    compound_pairwise_distance_distribution = compound_pairwise_distance_one_hot.float()
    return compound_pairwise_distance_distribution


def read_molecule(sdf_fileName, mol2_fileName, verbose=False):
    """
    Read molecule file from sdf file, or mol2 if it fails.

    Parameters
    ----------
    sdf_fileName: str
        The path to the sdf file.
    mol2_fileName: str
        The path to the mol2 file.
    verbose: bool
        Whether to print the warnings.

    
    BUG: Produces a bunch of "Can't kekulize mol". Does it matter or not? We should check the
    molecule representations at the end. It is probably because we don't use Chem.WrapLogs() to
    suppress the warnings."""
    stderr = sys.stderr
    sio = sys.stderr = StringIO()
    mol = Chem.MolFromMolFile(sdf_fileName, sanitize=False)
    problem = False
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)

        sm = Chem.MolToSmiles(mol)
    except Exception as e:
        sm = str(e)
        problem = True
    if problem:
        mol = Chem.MolFromMol2File(mol2_fileName, sanitize=False)
        problem = False
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            sm = Chem.MolToSmiles(mol)
            problem = False
        except Exception as e:
            sm = str(e)
            problem = True

    if verbose:
        print(sio.getvalue())
    sys.stderr = stderr
    return mol, problem