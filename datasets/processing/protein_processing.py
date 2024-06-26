import torch
import torch.nn.functional as F
import torch_geometric
from bindbind.datasets.processing.constants import THREE_TO_ONE_TANKBIND, THREE_TO_ONE_DIFFDOCK, LETTER_TO_NUM_TANKBIND, THREE_TO_NUMBER_DIFFDOCK
from Bio.PDB import PDBParser
from esm import pretrained, FastaBatchedDataset
import bindbind.datasets.processing.protein_features.diffdock_protein_features as diffdock_protein_features
import bindbind.datasets.processing.protein_features.tankbind_protein_features as tankbind_protein_features

# region features
##################### FEATURES #####################
def create_protein_node_coordinates(residue_list):
    """
    Create node coordinates for the protein graph.
    Independent of any choices: returns positions of all atoms of the backbone.

    Parameters
    ----------
    residue_list : list
        List of residues in the protein.

    Returns
    -------
    list
        List of coordinates for each residue.
    """
    coordinates = []
    for residue in residue_list:
        residue_coords = [list(residue[atom].coord) for atom in ("N", "CA", "C", "O")]
        coordinates.append(residue_coords)
    return coordinates


##################### END FEATURES #####################
# endregion features
# region esm_embeddings
##################### ESM EMBEDDINGS #####################
def get_sequences_from_pdbfile(file_path):
    biopython_parser = PDBParser()
    structure = biopython_parser.get_structure("random_id", file_path)
    structure = structure[0]
    sequence = None
    for i, chain in enumerate(structure):
        seq = ""
        for residue in chain:
            if residue.get_resname() == "HOH":
                continue
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == "CA":
                    c_alpha = list(atom.get_vector())
                if atom.name == "N":
                    n = list(atom.get_vector())
                if atom.name == "C":
                    c = list(atom.get_vector())
            if (
                c_alpha is not None and n is not None and c is not None
            ):  # only append residue if it is an amino acid
                try:
                    seq += THREE_TO_ONE_DIFFDOCK[residue.get_resname()]
                except Exception:
                    seq += "-"
                    print(
                        "encountered unknown AA: ",
                        residue.get_resname(),
                        " in the complex. Replacing it with a dash - .",
                    )

        if sequence is None:
            sequence = seq
        else:
            sequence += ":" + seq

    return sequence

def get_sequences(protein_files, protein_sequences):
    new_sequences = []
    for i in range(len(protein_files)):
        if protein_files[i] is not None:
            new_sequences.append(get_sequences_from_pdbfile(protein_files[i]))
        else:
            new_sequences.append(protein_sequences[i])
    return new_sequences

def compute_ESM_embeddings(model, alphabet, labels, sequences):
    # settings used
    toks_per_batch = 4096
    repr_layers = [33]
    truncation_seq_length = 1022

    dataset = FastaBatchedDataset(labels, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length),
        batch_sampler=batches,
    )

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [
        (i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers
    ]
    embeddings = {}

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                embeddings[label] = representations[33][i, 1 : truncate_len + 1].clone()
    return embeddings

def add_ESM_embeddings(labels, sequences):
    """

    Parameters
    ----------
    labels : list
        List of labels.
    sequences : list
        List of sequences.

    Returns
    -------
    lm_embedding : dict[str: torch.Tensor]
        List of ESM embeddings, indexed by label.
    """
    model_location = "esm2_t33_650M_UR50D"
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    toks_per_batch = 4096
    repr_layers = [33]
    truncation_seq_length = 1022

    dataset = FastaBatchedDataset(labels, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length),
        batch_sampler=batches,
    )

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [
        (i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers
    ]
    embeddings = {}

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                embeddings[label] = representations[33][i, 1 : truncate_len + 1].clone()
    return embeddings

def file_to_residues(protein_path):

    parser = PDBParser()
    structure = parser.get_structure("no_name", protein_path)
    
##################### END ESM EMBEDDINGS #####################
# endregion esm_embeddings
# region main_function
##################### MAIN FUNCTION #####################
def get_protein_features(
    protein_path,
    protein_name,
    top_k_nn_tankbind=30,
    num_rbf_tankbind=16,
    num_embeddings_tankbind=16,
    ):
    """
    Create a Data object for a protein.

    Parameters
    ----------
    protein_path: str
        Path to the protein file.
    protein_name: str
        Name of the protein.
    top_k_nn_tankbind : int, optional
        Number of nearest neighbors for the KNN graph which is the protein graph used in tankbind (default is 30).
    num_rbf : int, optional
        Number of radial basis functions (default is 16).
    num_embeddings : int, optional
        Number of positional embeddings (default is 16).
    Returns
    -------
    protein_data : dict
        Dictionary containing the protein data.
    """
    # Read protein files


    residue_list_diffdock = diffdock_protein_features.read_protein_file(protein_path)
    # Filter out residues that do not contain all required atoms (N, CA, C, O)

    
    residue_list_diffdock = [
        residue
        for residue in residue_list_diffdock
        if all(atom in residue for atom in ("N", "CA", "C", "O"))
    ]
    # Generate protein sequence from residue list. There are more accepted residues for diffdock than TankBind.

    tankbind_sequence = "".join(
        [
            THREE_TO_ONE_TANKBIND.get(residue.resname, "?")
            for residue in residue_list_tankbind
        ]
    )

    diffdock_sequence = "".join(
        [
            THREE_TO_ONE_DIFFDOCK.get(residue.resname, "?")
            for residue in residue_list_diffdock
        ]
    )

    # Create node coordinates
    protein_node_coordinates = create_protein_node_coordinates(residue_list)
    protein_node_coordinates = torch.as_tensor(
        protein_node_coordinates, dtype=torch.float32
    )

    # Encode protein sequence as integers
    # possible values from 0 to 20, 20 is a placeholder for unknown residues
    tankbind_numeric_sequence = torch.as_tensor(
        [LETTER_TO_NUM_TANKBIND.get(residue, 20) for residue in tankbind_sequence],
        dtype=torch.long,
    )

    diffdock_numeric_sequence = torch.as_tensor(
        [THREE_TO_NUMBER_DIFFDOCK.get(residue, 20) for residue in diffdock_sequence],
        dtype=torch.long,
    )

    # Create edge features
    edge_index_tankbind, edge_scalar_features_tankbind, edge_vector_features_tankbind = (
        tankbind_protein_features.create_edge_features(
            protein_node_coordinates, top_k_nn_tankbind, num_rbf_tankbind, num_embeddings_tankbind
        )
    )

    # Create node features
    node_scalar_features_tankbind, node_vector_features_tankbind = tankbind_protein_features.create_node_features(
        protein_node_coordinates
    )

    # Handle NaN values
    (
        node_scalar_features_tankbind,
        node_vector_features_tankbind,
        edge_scalar_features_tankbind,
        edge_vector_features_tankbind,
    ) = map(
        torch.nan_to_num,
        (
            node_scalar_features_tankbind,
            node_vector_features_tankbind,
            edge_scalar_features_tankbind,
            edge_vector_features_tankbind,
        ),
    )

    # Mask for valid node coordinates
    mask = torch.isfinite(protein_node_coordinates.sum(dim=(1, 2)))
    protein_node_coordinates[~mask] = float("Inf")
    alpha_carbon_coordinates = protein_node_coordinates[:, 1]

    # Create dictionary
    protein_data = {
        "residues": residue_list,
        "residue_backbone_coordinates": protein_node_coordinates,
        "alpha_carbon_coordinates": alpha_carbon_coordinates,
        "tankbind_sequence":tankbind_sequence, # One-letter encoding of protein residues for tankbind
        "diffdock_sequence":diffdock_sequence, # One-letter encoding of protein residues for diffdock
        "name":protein_name,
        "node_scalar_features_tankbind":node_scalar_features_tankbind,
        "node_vector_features_tankbind":node_vector_features_tankbind,
        "edge_scalar_features_tankbind":edge_scalar_features_tankbind,
        "edge_vector_features_tankbind":edge_vector_features_tankbind,
        "edge_index_tankbind":edge_index_tankbind,
        "mask":mask,
    }

    return protein_data
