import torch
from torch_geometric.data import Dataset, HeteroData
from bindbind.datasets.processing.pocket_constructors import get_all_nodes_to_keep, get_all_noised_pairwise_distances
import pickle
import pandas as pd
from lightning.pytorch.callbacks import Callback





class TankBindDataset(Dataset):
    def __init__(self, root="/fs/pool/pool-marsot/bindbind/datasets/tankbind_processed", add_esm_embeddings=None, noise_range=0.0, contact_threshold=8.0, pocket_radius=20.0):
        """
        
        
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
        self.add_esm_embeddings = add_esm_embeddings
        self.noise_range = noise_range
        self.contact_threshold = contact_threshold
        self.esm_embeddings = None
        super().__init__(root)
        self.esm_embeddings = torch.load(self.processed_paths[3]) if add_esm_embeddings else None

        self.proteins_df = torch.load(self.processed_paths[0])
        self.pockets_df = torch.load(self.processed_paths[1])
        self.affinity_dict = self.proteins_df.set_index('protein_names')['affinity'].to_dict()

        self.data = torch.load(self.processed_paths[2])
        self.pocket_radius = pocket_radius
        self.noised_pairwise_distances = get_all_noised_pairwise_distances(self.data, batch_size=256, noise_level=self.noise_range)
        self.protein_nodes_to_keep = get_all_nodes_to_keep(self.data, batch_size=256, pocket_radius=self.pocket_radius)
    
    def recompute_for_new_epoch(self):
        self.noised_pairwise_distances = get_all_noised_pairwise_distances(self.data, batch_size=256, noise_level=self.noise_range)
        self.protein_nodes_to_keep = get_all_nodes_to_keep(self.data, batch_size=256, pocket_radius=self.pocket_radius)

    def get(self, idx):
        # looking at protein name
        protein_name = self.pockets_df.iloc[idx]['name']
        pocket_index = self.pockets_df.iloc[idx]["rank"] - 1 # ranks start at 1 in the dataframe. 
        complex_features = self.data[protein_name]
        complex_features['affinity'] = self.affinity_dict[protein_name]
        data = make_tankbind_data_object(complex_features)
        # concatenate esm embeddings just before 
        if self.esm_embeddings is not None:
            # concatenate the ESM embeddings along the embedding dimension
            esm_value = (self.esm_embeddings[protein_name])[complex_features["tankbind_residue_indices_in_contact_with_compound"]]
            data["protein"].node_scalar_features = torch.cat([data["protein"].node_scalar_features, esm_value], dim=-1) # concatenate the ESM embeddings
       
        data["protein", "distance_to", "compound"].edge_attr = self.noised_pairwise_distances[protein_name].clamp(0,10)
        mask = self.protein_nodes_to_keep[protein_name][pocket_index]
        if mask.sum()<5:
            mask = torch.zeros_like(mask, dtype=torch.bool)
            mask[:100] = True
        # Removing protein residues outside of the pocket 
        data, _ligand_is_mostly_contained_in_pocket_sum = restrict_tankbind_data_to_pocket(data, mask, self.contact_threshold)


        # Make ligand_is_mostly_contained_in_protein_mask
        ligand_is_mostly_contained_in_pocket = _ligand_is_mostly_contained_in_pocket_sum/complex_features["tankbind_num_protein_nodes_close_to_ligand_and_in_contact_with_ligand"] >= 0.9
        data.ligand_is_mostly_contained_in_pocket = ligand_is_mostly_contained_in_pocket.bool()
        data.ligand_in_pocket_mask = torch.ones(data["protein", "distance_to", "compound"].edge_attr.shape).bool() if ligand_is_mostly_contained_in_pocket else torch.zeros(data["protein", "distance_to", "compound"].edge_attr.shape).bool()
        data.affinity = self.affinity_dict[protein_name] 
        return data
    
    def len(self):
        return len(self.pockets_df)
    
    @property
    def processed_file_names(self):
        return ["proteins_csv.pt", "pockets_csv.pt", "data.pt",]+(["esm_embeddings_650m.pt"] if self.add_esm_embeddings=="650m" else [])+(["esm_embeddings_15B.pt"] if self.add_esm_embeddings=="15B" else [])

    def process(self):
        proteins_df = pd.read_csv('/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/protein_paths_and_names.csv')
        pockets_df = pd.read_csv("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/p2rank_predictions.csv")
        data = {}
        if self.add_esm_embeddings=="650m":
            with open("/fs/pool/pool-marsot/bindbind/datasets/processing/esmcheckpoints/esm_embeddings_650m.pkl", "rb") as f:
                esm_embeddings = pickle.load(f)
        elif self.add_esm_embeddings=="15B":
            with open("/fs/pool/pool-marsot/bindbind/datasets/processing/esmcheckpoints/esm_embeddings_15B.pkl", "rb") as f:
                esm_embeddings = pickle.load(f)
        for protein_name in proteins_df["protein_names"].tolist():
            with open(f"/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/PDBBind_processed/{protein_name}/{protein_name}_full_data.pkl", "rb") as f:
                data[protein_name] = pickle.load(f)
        
        torch.save(proteins_df, self.processed_paths[0])
        torch.save(pockets_df, self.processed_paths[1])
        torch.save(data, self.processed_paths[2])
        if self.add_esm_embeddings is not None:
            torch.save(esm_embeddings, self.processed_paths[3])

class TankBindDatasetWithoutp2rank(Dataset):
    def __init__(self, root="/fs/pool/pool-marsot/bindbind/datasets/tankbind_processed_no_p2rank", add_esm_embeddings=None, noise_range=0.0, contact_threshold=8.0, pocket_radius=20.0):
        """
        Dataset where the pockets are epsilon-neighborhoods around
        the ligand center of mass, rather than p2rank-predicted pockets
        
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
        self.add_esm_embeddings = add_esm_embeddings
        self.noise_range = noise_range
        self.contact_threshold = contact_threshold
        super().__init__(root)

        self.proteins_df = torch.load(self.processed_paths[0])
        self.affinity_dict = self.proteins_df.set_index('protein_names')['affinity'].to_dict()

        self.data = torch.load(self.processed_paths[1])
        self.esm_embeddings = torch.load(self.processed_paths[2]) if add_esm_embeddings else None

        self.add_esm_embeddings = add_esm_embeddings
        self.noise_range = noise_range
        self.contact_threshold = contact_threshold
        self.pocket_radius = pocket_radius
        self.noised_pairwise_distances = get_all_noised_pairwise_distances(self.data, batch_size=256, noise_level=self.noise_range)
        self.protein_nodes_to_keep = get_all_nodes_to_keep(self.data,
                                                            key_1="tankbind_protein_alpha_carbon_coordinates",
                                                            key_2 = "tankbind_ligand_atom_coordinates",
                                                            batch_size=256,
                                                            pocket_radius=self.pocket_radius)
    
    def recompute_for_new_epoch(self):
        self.noised_pairwise_distances = get_all_noised_pairwise_distances(self.data, batch_size=256, noise_level=self.noise_range)
        self.protein_nodes_to_keep = get_all_nodes_to_keep(self.data,
                                                            key_1="tankbind_protein_alpha_carbon_coordinates",
                                                            key_2 = "tankbind_ligand_atom_coordinates",
                                                            batch_size=256)

    def get(self, idx):
        # looking at protein name
        protein_name = self.proteins_df.iloc[idx]['protein_names']

        complex_features = self.data[protein_name]

        complex_features['affinity'] = self.affinity_dict[protein_name]
        data = make_tankbind_data_object(complex_features)
        # concatenate esm embeddings just before 
        if self.esm_embeddings is not None:
            # concatenate the ESM embeddings along the embedding dimension
            esm_value = (self.esm_embeddings[protein_name])[complex_features["tankbind_residue_indices_in_contact_with_compound"]]
            data["protein"].node_scalar_features = torch.cat([data["protein"].node_scalar_features, esm_value], dim=-1) # concatenate the ESM embeddings
       
        data["protein", "distance_to", "compound"].edge_attr = self.noised_pairwise_distances[protein_name]
        mask = self.protein_nodes_to_keep[protein_name][0]
        if mask.sum()<5:
            mask = torch.zeros_like(mask, dtype=torch.bool)
            mask[:100] = True
        # Removing protein residues outside of the pocket 
        data, _ligand_is_mostly_contained_in_pocket_sum = restrict_tankbind_data_to_pocket(data, mask, self.contact_threshold)


        # Make ligand_is_mostly_contained_in_protein_mask
        ligand_is_mostly_contained_in_pocket = True
        data.ligand_is_mostly_contained_in_pocket = ligand_is_mostly_contained_in_pocket
        data.ligand_in_pocket_mask = torch.ones(data["protein", "distance_to", "compound"].edge_attr.shape).bool()

        return data
    
    def len(self):
        return len(self.proteins_df)
    
    @property
    def processed_file_names(self):
        return ["proteins_csv.pt", "data.pt",]+(["esm_embeddings_650m.pt"] if self.add_esm_embeddings=="650m" else [])+(["esm_embeddings_15B.pt"] if self.add_esm_embeddings=="15B" else [])

    def process(self):
        proteins_df = pd.read_csv('/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/protein_paths_and_names.csv')
        data = {}
        if self.add_esm_embeddings=="650m":
            with open("/fs/pool/pool-marsot/bindbind/datasets/processing/esmcheckpoints/esm_embeddings_650m.pkl", "rb") as f:
                esm_embeddings = pickle.load(f)
        elif self.add_esm_embeddings=="15B":
            with open("/fs/pool/pool-marsot/bindbind/datasets/processing/esmcheckpoints/esm_embeddings_15B.pkl", "rb") as f:
                esm_embeddings = pickle.load(f)
        for protein_name in proteins_df["protein_names"].tolist():
            with open(f"/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/PDBBind_processed/{protein_name}/{protein_name}_full_data.pkl", "rb") as f:
                data[protein_name] = pickle.load(f)
        
        torch.save(proteins_df, self.processed_paths[0])
        torch.save(data, self.processed_paths[1])
        if self.add_esm_embeddings is not None:
            torch.save(esm_embeddings, self.processed_paths[2])

class TankBindTestDataset(Dataset):
    def __init__(self, root="/fs/pool/pool-marsot/bindbind/datasets/tankbind_test", add_esm_embeddings=None, noise_range=0.0, contact_threshold=8.0):
        """
        
        
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
        complex_features['affinity'] = self.affinity_dict[protein_name]
        data = make_tankbind_data_object(complex_features)
        # concatenate esm embeddings just before 
        if self.esm_embeddings is not None:
            # concatenate the ESM embeddings along the embedding dimension
            esm_value = (self.esm_embeddings[protein_name])[complex_features["tankbind_residue_indices_in_contact_with_compound"]]
            data["protein"].node_scalar_features = torch.cat([data["protein"].node_scalar_features, esm_value], dim=-1) # concatenate the ESM embeddings
       
        data["protein", "distance_to", "compound"].edge_attr = self.noised_pairwise_distances[protein_name]
        mask = self.protein_nodes_to_keep[protein_name][pocket_index]
        if mask.sum()<5:
            mask = torch.zeros_like(mask, dtype=torch.bool)
            mask[:100] = True
        # Removing protein residues outside of the pocket 
        data, _ligand_is_mostly_contained_in_pocket_sum = restrict_tankbind_data_to_pocket(data, mask, self.contact_threshold)


        # Make ligand_is_mostly_contained_in_protein_mask
        ligand_is_mostly_contained_in_pocket = _ligand_is_mostly_contained_in_pocket_sum/complex_features["tankbind_num_protein_nodes_close_to_ligand_and_in_contact_with_ligand"] >= 0.9
        data.ligand_is_mostly_contained_in_pocket = ligand_is_mostly_contained_in_pocket
        data.ligand_in_pocket_mask = torch.ones(data["protein", "distance_to", "compound"].edge_attr.shape).bool()*ligand_is_mostly_contained_in_pocket.bool()
        data.affinity = self.affinity_dict[protein_name] 
        return data
    
    def len(self):
        return len(self.pockets_df)
    
    @property
    def processed_file_names(self):
        return ["proteins_csv.pt", "pockets_csv.pt", "data.pt",]+(["esm_embeddings_650m.pt"] if self.add_esm_embeddings=="650m" else [])+(["esm_embeddings_15B.pt"] if self.add_esm_embeddings=="15B" else [])

    def process(self):
        proteins_df = pd.read_csv('/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/protein_paths_and_names.csv')
        pockets_df = pd.read_csv("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/p2rank_predictions.csv")
        with open("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/tankbind_splits/timesplit_test", "r") as f:
            test_proteins = set(f.read().splitlines())
        data = {}
        if self.add_esm_embeddings=="650m":
            with open("/fs/pool/pool-marsot/bindbind/datasets/processing/esmcheckpoints/esm_embeddings_650m.pkl", "rb") as f:
                esm_embeddings = pickle.load(f)
        elif self.add_esm_embeddings=="15B":
            with open("/fs/pool/pool-marsot/bindbind/datasets/processing/esmcheckpoints/esm_embeddings_15B.pkl", "rb") as f:
                esm_embeddings = pickle.load(f)
        for protein_name in set(proteins_df["protein_names"].tolist()).intersection(test_proteins):
            with open(f"/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/PDBBind_processed/{protein_name}/{protein_name}_full_data.pkl", "rb") as f:
                data[protein_name] = pickle.load(f)
        
        proteins_df = proteins_df[proteins_df["protein_names"].isin(test_proteins)]
        pockets_df = pockets_df[pockets_df["name"].isin(test_proteins)]
        torch.save(proteins_df, self.processed_paths[0])
        torch.save(pockets_df, self.processed_paths[1])
        torch.save(data, self.processed_paths[2])
        if self.add_esm_embeddings is not None:
            torch.save(esm_embeddings, self.processed_paths[3])

class TankBindValDataset(Dataset):
    def __init__(self, root="/fs/pool/pool-marsot/bindbind/datasets/tankbind_val", add_esm_embeddings=None, noise_range=0.0, contact_threshold=8.0):
        """
        
        
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
        complex_features['affinity'] = self.affinity_dict[protein_name]
        data = make_tankbind_data_object(complex_features)
        # concatenate esm embeddings just before 
        if self.esm_embeddings is not None:
            # concatenate the ESM embeddings along the embedding dimension
            esm_value = (self.esm_embeddings[protein_name])[complex_features["tankbind_residue_indices_in_contact_with_compound"]]
            data["protein"].node_scalar_features = torch.cat([data["protein"].node_scalar_features, esm_value], dim=-1) # concatenate the ESM embeddings
       
        data["protein", "distance_to", "compound"].edge_attr = self.noised_pairwise_distances[protein_name]
        mask = self.protein_nodes_to_keep[protein_name][pocket_index]
        if mask.sum()<5:
            mask = torch.zeros_like(mask, dtype=torch.bool)
            mask[:100] = True
        # Removing protein residues outside of the pocket 
        data, _ligand_is_mostly_contained_in_pocket_sum = restrict_tankbind_data_to_pocket(data, mask, self.contact_threshold)


        # Make ligand_is_mostly_contained_in_protein_mask
        ligand_is_mostly_contained_in_pocket = _ligand_is_mostly_contained_in_pocket_sum/complex_features["tankbind_num_protein_nodes_close_to_ligand_and_in_contact_with_ligand"] >= 0.9
        data.ligand_is_mostly_contained_in_pocket = ligand_is_mostly_contained_in_pocket
        data.ligand_in_pocket_mask = torch.ones(data["protein", "distance_to", "compound"].edge_attr.shape).bool()*ligand_is_mostly_contained_in_pocket.bool()
        data.affinity = self.affinity_dict[protein_name] 
        return data
    
    def len(self):
        return len(self.pockets_df)
    
    @property
    def processed_file_names(self):
        return ["proteins_csv.pt", "pockets_csv.pt", "data.pt",]+(["esm_embeddings_650m.pt"] if self.add_esm_embeddings=="650m" else [])+(["esm_embeddings_15B.pt"] if self.add_esm_embeddings=="15B" else [])

    def process(self):
        proteins_df = pd.read_csv('/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/protein_paths_and_names.csv')
        pockets_df = pd.read_csv("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/p2rank_predictions.csv")
        with open("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/tankbind_splits/timesplit_no_lig_overlap_val", "r") as f:
            val_proteins = set(f.read().splitlines())
        data = {}
        if self.add_esm_embeddings=="650m":
            with open("/fs/pool/pool-marsot/bindbind/datasets/processing/esmcheckpoints/esm_embeddings_650m.pkl", "rb") as f:
                esm_embeddings = pickle.load(f)
        elif self.add_esm_embeddings=="15B":
            with open("/fs/pool/pool-marsot/bindbind/datasets/processing/esmcheckpoints/esm_embeddings_15B.pkl", "rb") as f:
                esm_embeddings = pickle.load(f)
        for protein_name in set(proteins_df["protein_names"].tolist()).intersection(val_proteins):
            with open(f"/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/PDBBind_processed/{protein_name}/{protein_name}_full_data.pkl", "rb") as f:
                data[protein_name] = pickle.load(f)
        
        proteins_df = proteins_df[proteins_df["protein_names"].isin(val_proteins)]
        pockets_df = pockets_df[pockets_df["name"].isin(val_proteins)]
        torch.save(proteins_df, self.processed_paths[0])
        torch.save(pockets_df, self.processed_paths[1])
        torch.save(data, self.processed_paths[2])
        if self.add_esm_embeddings is not None:
            torch.save(esm_embeddings, self.processed_paths[3])







class NoisyCoordinates(Callback):
    def __init__(self, dataset):
        self.dataset = dataset
    def on_train_epoch_end(self, trainer, pl_module):
        self.dataset.recompute_for_new_epoch()
        print("Recomputed noisy coordinates for new epoch")


def restrict_tankbind_data_to_pocket(data, mask, contact_threshold):
    new_node_index = torch.zeros_like(mask, dtype=torch.long)
    new_node_index[mask] = torch.arange(mask.sum())
    data["protein"].coordinates = data["protein"].coordinates[mask]
    data["protein"].node_scalar_features = data["protein"].node_scalar_features[mask].contiguous()
    data["protein"].node_vector_features = data["protein"].node_vector_features[mask].contiguous()
    edges = data["protein", "to", "protein"].edge_index
    data.seq = data.seq[mask]
    edges_to_keep = mask[edges[0]] & mask[edges[1]]
    data["protein", "to", "protein"].edge_index = edges[:, edges_to_keep]
    data["protein", "to", "protein"].edge_index = new_node_index[data["protein", "to", "protein"].edge_index]
    data["protein", "to", "protein"].edge_scalar_features = data["protein", "to", "protein"].edge_scalar_features[edges_to_keep].contiguous()
    data["protein", "to", "protein"].edge_vector_features = data["protein", "to", "protein"].edge_vector_features[edges_to_keep].contiguous()
    data["protein", "distance_to", "compound"].edge_attr = data["protein", "distance_to", "compound"].edge_attr[mask]
    # count the number of protein residues that are in contact with the ligand and in the ligand
    _ligand_is_mostly_contained_in_pocket_sum = (data["protein", "distance_to", "compound"].edge_attr < contact_threshold).sum()
    data["protein", "distance_to", "compound"].edge_attr = data["protein", "distance_to", "compound"].edge_attr.flatten().contiguous()
    return data, _ligand_is_mostly_contained_in_pocket_sum



def make_tankbind_data_object(complex_dict):
    """Create the input for the model, without the target affinity and target pairwise distances"""
    data = HeteroData()
    data.complex_name = complex_dict["protein_name"]
    data.seq = complex_dict["tankbind_one_letter_sequence"]
    data["protein"].coordinates = complex_dict["tankbind_protein_alpha_carbon_coordinates"]
    data["protein"].node_scalar_features = complex_dict["tankbind_protein_node_scalar_features"]
    data["protein"].node_vector_features = complex_dict["tankbind_protein_node_vector_features"]
    data["protein", "to", "protein"].edge_index = complex_dict["tankbind_protein_edge_index"]
    data["protein", "to", "protein"].edge_scalar_features = complex_dict["tankbind_protein_edge_scalar_features"]
    data["protein", "to", "protein"].edge_vector_features = complex_dict["tankbind_protein_edge_vector_features"]

    data["compound"].x = complex_dict["tankbind_ligand_atom_features"]
    data["compound", "to", "compound"].edge_index = complex_dict["tankbind_ligand_edge_index"]
    data["compound", "to", "compound"].edge_attr = complex_dict["tankbind_ligand_edge_features"]
    data["compound", "distance_to", "compound"].edge_attr = complex_dict["tankbind_ligand_pairwise_distance_distribution"].flatten(0,1)

    data.affinity = complex_dict["affinity"] 
    # add assertions for data shape and presence of nans.
    assert data["protein"].node_scalar_features.shape[0] == data["protein"].node_vector_features.shape[0]
    assert data["protein"].node_scalar_features.dim() == 2
    assert data["protein"].node_vector_features.dim() == 3
    assert data["protein", "to", "protein"].edge_index.dim() == 2
    assert data["protein", "to", "protein"].edge_scalar_features.dim() == 2
    assert data["protein", "to", "protein"].edge_vector_features.dim() == 3
    assert data["compound"].x.dim() == 2
    assert data["compound", "to", "compound"].edge_index.dim() == 2
    assert data["compound", "to", "compound"].edge_attr.dim() == 2
    assert torch.isnan(data["protein"].coordinates).sum() == 0
    assert torch.isnan(data["protein"].node_scalar_features).sum() == 0
    assert torch.isnan(data["protein"].node_vector_features).sum() == 0
    assert torch.isnan(data["protein", "to", "protein"].edge_scalar_features).sum() == 0
    assert torch.isnan(data["protein", "to", "protein"].edge_vector_features).sum() == 0
    assert torch.isnan(data["compound"].x).sum() == 0
    assert torch.isnan(data["compound", "to", "compound"].edge_attr).sum() == 0
    assert torch.isnan(data["compound", "distance_to", "compound"].edge_attr).sum() == 0
    return normalize_data_features(data)


means_dict = {"protein_node_coordinates": torch.tensor([17.631010055541992, 14.444746971130371, 21.412504196166992]),
              "protein_node_scalar_features": torch.tensor([0.07262682914733887, 0.1329594999551773, -0.9804213643074036, -0.760931670665741, 0.02471655048429966, 0.011947994120419025]),
              "protein_node_vector_features": torch.tensor([[-0.0005944063304923475, 0.0004770602972712368, 0.0003047776408493519], [0.0005944063304923475, -0.0004770602972712368, -0.0003047776408493519], [-9.438615961698815e-05, 0.0009057653369382024, -0.00030406430596485734]]),
              "protein_edge_scalar_features": torch.tensor([6.944148026377661e-06, 0.001373058999888599, 0.03128135949373245, 0.10510130971670151, 0.1631690412759781, 0.17957991361618042, 0.2117183655500412, 0.3174160420894623, 0.30822935700416565, 0.18642766773700714, 0.09413681924343109, 0.039788469672203064, 0.014366302639245987, 0.005030795466154814, 0.001948254881426692, 0.0008591284276917577, -0.03676437586545944, -0.03928199037909508, -0.027324488386511803, -0.04502237215638161, -0.03382807597517967, -0.0309743732213974, -0.03358544036746025, -0.00877009890973568, -0.0006605424568988383, -0.0003721231478266418, 2.121787292708177e-05, 8.425789928878658e-06, 4.6160774218151346e-05, -9.526106623525266e-06, 0.0002609408984426409, 4.474521119846031e-05]),
              "protein_edge_vector_features": torch.tensor([[-0.0002645916829351336, -0.0002201295574195683, -3.9455248042941093e-05]]),
              "protein_distance_to_compound": torch.tensor(8.333386421203613),
              "compound_to_compound": torch.tensor([0.5834429264068604, 0.09144987910985947, 0.001274818554520607, 0.32383236289024353, 0.9850631952285767, 0.0, 0.0, 0.004669341258704662, 0.010267456993460655, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5256147384643555, 1.4313571453094482]),
              "affinity": torch.tensor(6.3677),
              }

vars_dict = {"protein_node_coordinates": torch.tensor([2417.467529296875, 2081.125244140625, 2309.912353515625]),
             "protein_node_scalar_features": torch.tensor([0.2213876098394394, 0.6002227067947388, 0.02893569879233837, 0.19432076811790466, 0.3814883232116699, 0.009695463813841343]),
                "protein_node_vector_features": torch.tensor([[0.33243584632873535, 0.33213087916374207, 0.33270949125289917], [0.33243584632873535, 0.33213087916374207, 0.33270949125289917], [0.3334696292877197, 0.3342677056789398, 0.33226191997528076]]),
                "protein_edge_scalar_features": torch.tensor([3.820823657463279e-09, 3.113720958936028e-05, 0.011859781108796597, 0.06986543536186218, 0.09459588676691055, 0.0935622900724411, 0.0947701707482338, 0.13434819877147675, 0.1342504769563675, 0.09262754023075104, 0.054759006947278976, 0.02425190433859825, 0.008764220401644707, 0.0031063344795256853, 0.0012501657474786043, 0.0005616022972390056, 0.48492634296417236, 0.48378145694732666, 0.4840312898159027, 0.48184001445770264, 0.4788280129432678, 0.47806236147880554, 0.48250657320022583, 0.5102618336677551, 0.5137215852737427, 0.5146753191947937, 0.5152220726013184, 0.5161329507827759, 0.5200276374816895, 0.5209782123565674, 0.5163654088973999, 0.4896612763404846]),
                "protein_edge_vector_features": torch.tensor([[0.33269432187080383, 0.33325907588005066, 0.33404645323753357]]),
                "protein_distance_to_compound": torch.tensor(31.070316314697266),
                "compound_to_compound": torch.tensor([0.24303746223449707, 0.08308686316013336, 0.0012731943279504776, 0.21896512806415558, 0.014713701792061329, 0.0, 0.0, 0.004647541791200638, 0.010162044316530228, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24934406578540802, 0.011662687174975872]),
                "affinity": torch.tensor(3.4767),
                }

stds_dict = {key: torch.sqrt(value) for key, value in vars_dict.items()}

def normalize_data_features(data, epsilon = 1e-3):
    data["protein"].coordinates = torch.nan_to_num((data["protein"].coordinates - means_dict["protein_node_coordinates"])/(stds_dict["protein_node_coordinates"]+epsilon))
    data["protein"].node_scalar_features = torch.nan_to_num((data["protein"].node_scalar_features - means_dict["protein_node_scalar_features"])/(stds_dict["protein_node_scalar_features"]+epsilon))
    data["protein"].node_vector_features = torch.nan_to_num((data["protein"].node_vector_features - means_dict["protein_node_vector_features"])/(stds_dict["protein_node_vector_features"]+epsilon))
    data["protein", "to", "protein"].edge_scalar_features = torch.nan_to_num((data["protein", "to", "protein"].edge_scalar_features - means_dict["protein_edge_scalar_features"])/(stds_dict["protein_edge_scalar_features"]+epsilon))
    data["protein", "to", "protein"].edge_vector_features = torch.nan_to_num((data["protein", "to", "protein"].edge_vector_features - means_dict["protein_edge_vector_features"])/(stds_dict["protein_edge_vector_features"]+epsilon))
    data["compound", "to", "compound"].edge_attr = torch.nan_to_num((data["compound", "to", "compound"].edge_attr - means_dict["compound_to_compound"])/(stds_dict["compound_to_compound"]+epsilon))
    data.affinity = torch.nan_to_num((data.affinity - means_dict["affinity"])/(stds_dict["affinity"]+epsilon))
    return data


