import os
import sys
sys.path.append('/fs/pool/pool-marsot')

import torch
from bindbind.datasets.processing.constants import THREE_TO_ONE_DIFFDOCK
from esm import pretrained, FastaBatchedDataset
from Bio.PDB import PDBParser

import os
import pickle
from multiprocessing import Pool


def print_cuda_memory_usage():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**2:.2f} MB")
    print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")

def get_sequences_from_pdbfile(file_path):
    # Define the path for the result file
    folder_path = os.path.dirname(file_path)
    result_file_path = os.path.join(folder_path, "sequence_result.txt")
    
    # Check if the result file already exists
    if os.path.exists(result_file_path):
        with open(result_file_path, 'r') as file:
            sequence = file.read().strip()
            return sequence
    
    # Parse the PDB file to extract sequences
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
            if c_alpha is not None and n is not None and c is not None:  # only append residue if it is an amino acid
                try:
                    seq += THREE_TO_ONE_DIFFDOCK[residue.get_resname()]
                except Exception:
                    seq += "-"
                    print("encountered unknown AA: ", residue.get_resname(), " in the complex. Replacing it with a dash - .")
        
        if sequence is None:
            sequence = seq
        else:
            sequence += ":" + seq
    
    # Save the result to a file
    with open(result_file_path, 'w') as file:
        file.write(sequence)
    
    return sequence

def create_ESM_embeddings(labels, sequences, model_dim="650m"):
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
    if model_dim == "650m":
        model_location = "esm2_t33_650M_UR50D"
        toks_per_batch = 12288
    elif model_dim == "15B":
        model_location = "esm2_t48_15B_UR50D"
        toks_per_batch = 4096
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

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
            print_cuda_memory_usage()
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                embeddings[label] = representations[33][i, 1 : truncate_len + 1].clone()
    return embeddings

def get_all_ESM_embeddings(protein_paths, protein_names, model="650m"):
    sequences_dict = {protein_names[i]: get_sequences_from_pdbfile(protein_paths[i]) for i in range(len(protein_paths))}
    labels_cleaned, sequences_cleaned = [], []
    for name, sequence in sequences_dict.items():
        s = sequence.split(':')
        sequences_cleaned.extend(s)
        labels_cleaned.extend([name + '_chain_' + str(j) for j in range(len(s))])
    results = create_ESM_embeddings(labels_cleaned, sequences_cleaned, model)
    embeddings = {}
    for name in protein_names:
        embeddings[name] = torch.cat([results[name + '_chain_' + str(j)] for j in range(len(sequences_dict[name].split(':')))], dim=0)

    return embeddings

def list_folders(directory):
    entries = os.listdir(directory)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
    return folders

def get_pdbbind_paths_and_names(folder_path):
    protein_names = list_folders(folder_path)
    protein_paths = [f"{folder_path}/{protein}/{protein}_protein_chains_in_contact_with_ligand.pdb" for protein in protein_names]
    
    # Filter out the paths that don't exist
    existing_protein_paths = [path for path in protein_paths if os.path.exists(path)]
    
    # Corresponding protein names for the existing paths
    existing_protein_names = [protein_names[i] for i in range(len(protein_paths)) if os.path.exists(protein_paths[i])]
    
    return existing_protein_paths, existing_protein_names

def populate_sequences_dict(folder_path):
    protein_names = list_folders(folder_path)
    protein_paths = [f"{folder_path}/{protein}/{protein}_protein_chains_in_contact_with_ligand.pdb" for protein in protein_names]
    
    # Filter out the paths that don't exist
    protein_paths = [path for path in protein_paths if os.path.exists(path)]
    with Pool(processes=32) as pool:
        pool.map(get_sequences_from_pdbfile, protein_paths)
    return None

def get_pdbbind_ESM_embeddings(folder_path="/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/PDBBind_processed", model="650m"):
    import time
    start = time.time()
    protein_paths, protein_names = get_pdbbind_paths_and_names(folder_path)
    result = get_all_ESM_embeddings(protein_paths, protein_names, model=model)
    end = time.time()
    print(f"Time taken: {end - start}")

    with open(f"{folder_path}/esm_embeddings_{model}.pkl", "wb") as file:
        pickle.dump(result, file)

    return result