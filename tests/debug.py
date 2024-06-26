import torch
from torch_geometric.data import Dataset
from bindbind.torch_datasets.tankbind_dataset import make_tankbind_data_object, restrict_tankbind_data_to_pocket
from bindbind.datasets.processing.pocket_constructors import get_all_nodes_to_keep, get_all_noised_pairwise_distances
import pickle
import pandas as pd
from tqdm import tqdm


class TankBindDebugDataset(Dataset):
    def __init__(self, root="/fs/pool/pool-marsot/bindbind/datasets/tankbind_debug", add_esm_embeddings=None, noise_range=0.0, contact_threshold=8.0):
        """
        DEBUG VERSION OF THE DATASET
        
        Parameters
        ----------
        root: str
            dataset files, with paths root/processed/proteins_csv.pt, root/processed/pockets_csv.pt, root/processed/data.pt
            are expected there. If they are not there, the function processing will be called to create these files.
        esm_embeddings: Optional[str]
            can be either "650m" or "15B". If None, we do not use ESM embeddings.
        noise_range: float
            magnitude of the uniform noise that is added to the compound and protein node coordinates.
        contact_threshold: float
            threshold below which we consider that a compound node is touching a protein node.
            """
        print("DEBUG DATASET")
        self.add_esm_embeddings = add_esm_embeddings
        self.noise_range = noise_range
        self.contact_threshold = contact_threshold
        super().__init__(root)

        self.proteins_df = torch.load(self.processed_paths[0])
        self.pockets_df = torch.load(self.processed_paths[1])
        self.affinity_dict = self.proteins_df.set_index('protein_names')['affinity'].to_dict()

        self.data = torch.load(self.processed_paths[2])
        self.esm_embeddings = torch.load(self.processed_paths[3]) if add_esm_embeddings else None

        self.add_esm_embeddings = add_esm_embeddings
        self.noise_range = noise_range
        self.contact_threshold = contact_threshold

        self.noised_pairwise_distances = get_all_noised_pairwise_distances(self.data, batch_size=256, noise_level=self.noise_range)
        self.protein_nodes_to_keep = get_all_nodes_to_keep(self.data, batch_size=256)
    
    def recompute_for_new_epoch(self):
        self.noised_pairwise_distances = get_all_noised_pairwise_distances(self.data, batch_size=256, noise_level=self.noise_range)
        self.protein_nodes_to_keep = get_all_nodes_to_keep(self.data, batch_size=256)

    def get(self, idx):
        # looking at protein name
        protein_name = self.pockets_df.iloc[idx]['name']
        pocket_index = self.pockets_df.iloc[idx]["rank"] - 1 # ranks start at 1 in the dataframe. 
        complex_features = self.data[protein_name]
        data = make_tankbind_data_object(complex_features)
        # concatenate esm embeddings just before 
        if self.esm_embeddings is not None:
             # concatenate the ESM embeddings along the embedding dimension
            data["protein"].node_scalar_features = torch.cat([data["protein"].node_scalar_features, self.esm_embeddings[protein_name]], dim=-1) # concatenate the ESM embeddings
       
        data["protein", "distance_to", "compound"].edge_attr = self.noised_pairwise_distances[protein_name]
        mask = self.protein_nodes_to_keep[protein_name][pocket_index]
        if mask.sum()<5:
            mask = torch.zeros_like(mask, dtype=torch.bool)
            mask[:100] = True
        # Removing protein residues outside of the pocket 
      # Removing protein residues outside of the pocket 
        data, _ligand_is_mostly_contained_in_pocket_sum = restrict_tankbind_data_to_pocket(data, mask, self.contact_threshold)


        # Make ligand_is_mostly_contained_in_protein_mask
        ligand_is_mostly_contained_in_pocket = _ligand_is_mostly_contained_in_pocket_sum/complex_features["tankbind_num_protein_nodes_close_to_ligand_and_in_contact_with_ligand"] >= 0.9
        data.ligand_is_mostly_contained_in_pocket = ligand_is_mostly_contained_in_pocket
        data.ligand_in_pocket_mask = torch.ones(data["protein", "distance_to", "compound"].edge_attr.shape).bool() if ligand_is_mostly_contained_in_pocket else torch.zeros(data["protein", "distance_to", "compound"].edge_attr.shape).bool()
        data.affinity = self.affinity_dict[protein_name]
        return data

    def len(self):
        return len(self.pockets_df)

    @property
    def processed_file_names(self):
        return ["proteins_csv.pt", "pockets_csv.pt", "data.pt",]+(["esm_embeddings_650m.pt"] if self.add_esm_embeddings=="650m" else [])+(["esm_embeddings_15B.pt"] if self.add_esm_embeddings=="15B" else [])

    def process(self):
        proteins_df = pd.read_csv('/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/protein_paths_and_names.csv').iloc[:100]
        pockets_df = pd.read_csv("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/p2rank_predictions.csv")
        pockets_df = pockets_df[pockets_df["name"].isin(proteins_df["protein_names"])]
        data = {}
        if self.add_esm_embeddings=="650m":
            with open("/fs/pool/pool-marsot/bindbind/datasets/processing/esmcheckpoints/esm_embeddings_650m.pkl", "rb") as f:
                esm_embeddings = pickle.load(f)
        elif self.add_esm_embeddings=="15B":
            with open("/fs/pool/pool-marsot/bindbind/datasets/processing/esmcheckpoints/esm_embeddings_15B.pkl", "rb") as f:
                esm_embeddings = pickle.load(f)
        for protein_name in tqdm(proteins_df["protein_names"]):
            with open(f"/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/PDBBind_processed/{protein_name}/{protein_name}_full_data.pkl", "rb") as f:
                data[protein_name] = pickle.load(f)
        torch.save(proteins_df, self.processed_paths[0])
        torch.save(pockets_df, self.processed_paths[1])
        torch.save(data, self.processed_paths[2])
        if self.add_esm_embeddings:
            torch.save(esm_embeddings, self.processed_paths[3])
    
def example_complex_dict():
    with open("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/PDBBind_processed/1a0q/1a0q_full_data.pkl", "rb") as f:
        complex_dict = pickle.load(f)
    return complex_dict

def info_df():
    info_df = pd.read_csv("/fs/pool/pool-marsot/tankbind_enzo/bind/preprocessing/info.csv")
    return info_df

def protein_names_df():
    df_names = pd.read_csv("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/protein_paths_and_names.csv")
    return df_names