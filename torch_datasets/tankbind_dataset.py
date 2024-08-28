import os
import torch
from torch_geometric.data import Dataset, HeteroData
from bindbind.datasets.processing.pocket_constructors import get_all_nodes_to_keep, get_all_noised_pairwise_distances
import pickle
import pandas as pd
from lightning.pytorch.callbacks import Callback
from bindbind.datasets.processing.ligand_features.tankbind_ligand_features import create_tankbind_ligand_features

from bindbind.datasets.processing.ligand_features.tankbind_ligand_features import read_molecule, create_tankbind_ligand_features, get_LAS_distance_constraint_mask
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from bindbind.experiments.ablations.regular.metrics.helper import generate_sdf_from_smiles_using_rdkit

class TankBindDataset(Dataset):
    def __init__(self, root="/fs/pool/pool-marsot/bindbind/datasets/tankbind_processed",
                 add_esm_embeddings=None,
                 noise_range=0.0,
                 contact_threshold=8.0,
                 pocket_radius=20.0,
                 normalize_features=True,
                 **kwargs):
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
        self.normalize_features = normalize_features
        super().__init__(root, **kwargs)
        self.esm_embeddings = torch.load(self.processed_paths[3]) if add_esm_embeddings else None

        self.proteins_df = torch.load(self.processed_paths[0])
        self.pockets_df = torch.load(self.processed_paths[1])
        self.affinity_dict = self.proteins_df.set_index('protein_names')['affinity'].to_dict()

        self.data = torch.load(self.processed_paths[2])
        self.pocket_radius = pocket_radius

    def get(self, idx):
        # looking at protein name
        protein_name = self.pockets_df.iloc[idx]['name']
        pocket_index = self.pockets_df.iloc[idx]["rank"] - 1 # ranks start at 1 in the dataframe.
        complex_features = self.data[protein_name]
        pocket_center_coordinates = (complex_features["tankbind_protein_pocket_center_coordinates"][pocket_index]).unsqueeze(0)

        complex_features['affinity'] = self.affinity_dict[protein_name]
        noised_pocket_center_coordinates = pocket_center_coordinates + torch.randn_like(pocket_center_coordinates)*self.noise_range 

        data = make_tankbind_data_object(complex_features, normalize_features=self.normalize_features)
        protein_node_coordinates = complex_features["tankbind_protein_alpha_carbon_coordinates"]
        compound_node_coordinates = complex_features["tankbind_ligand_atom_coordinates"]
        noised_protein_node_coordinates = protein_node_coordinates + torch.randn_like(protein_node_coordinates) * self.noise_range
        noised_compound_node_coordinates = compound_node_coordinates + torch.randn_like(compound_node_coordinates) * self.noise_range
        noised_pairwise_distance = torch.cdist(noised_protein_node_coordinates, noised_compound_node_coordinates, compute_mode="donot_use_mm_for_euclid_dist")
        data["protein", "distance_to", "compound"].edge_attr = noised_pairwise_distance.clamp(0,10)
        mask = get_pocket_mask(noised_protein_node_coordinates, noised_pocket_center_coordinates, self.pocket_radius) 

        
        if self.normalize_features:
            data["protein", "distance_to", "compound"].edge_attr = torch.nan_to_num((data["protein", "distance_to", "compound"].edge_attr - means_dict["protein_distance_to_compound"])/(stds_dict["protein_distance_to_compound"]+1e-3))


        # concatenate esm embeddings just before 
        if self.esm_embeddings is not None:
            # concatenate the ESM embeddings along the embedding dimension
            esm_value = (self.esm_embeddings[protein_name])[complex_features["tankbind_residue_indices_in_contact_with_compound"]]
            data["protein"].node_scalar_features = torch.cat([data["protein"].node_scalar_features, esm_value], dim=-1) # concatenate the ESM embeddings
       

        if mask.sum()<5:
            mask = torch.zeros_like(mask, dtype=torch.bool)
            mask[:100] = True
        # Removing protein residues outside of the pocket 
        data, _ligand_is_mostly_contained_in_pocket_sum = restrict_tankbind_data_to_pocket(data, mask, self.contact_threshold)


        # Make ligand_is_mostly_contained_in_protein_mask
        ligand_is_mostly_contained_in_pocket = _ligand_is_mostly_contained_in_pocket_sum/complex_features["tankbind_num_protein_nodes_close_to_ligand_and_in_contact_with_ligand"] >= 0.9
        data.ligand_is_mostly_contained_in_pocket = ligand_is_mostly_contained_in_pocket.bool()
        data.ligand_in_pocket_mask = torch.ones(data["protein", "distance_to", "compound"].edge_attr.shape).bool() if ligand_is_mostly_contained_in_pocket else torch.zeros(data["protein", "distance_to", "compound"].edge_attr.shape).bool()
   
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
    def __init__(self,
                 root="/fs/pool/pool-marsot/bindbind/datasets/tankbind_processed_no_p2rank",
                 add_esm_embeddings=None,
                 noise_range=0.0,
                 contact_threshold=8.0,
                 pocket_radius=20.0,
                 normalize_features=True,
                 **kwargs):
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
        self.normalize_features = normalize_features
        super().__init__(root, **kwargs)

        self.proteins_df = torch.load(self.processed_paths[0])
        self.affinity_dict = self.proteins_df.set_index('protein_names')['affinity'].to_dict()

        self.data = torch.load(self.processed_paths[1])
        self.esm_embeddings = torch.load(self.processed_paths[2]) if add_esm_embeddings else None

        self.pocket_radius = pocket_radius
    def get(self, idx):
        # looking at protein name
        protein_name = self.proteins_df.iloc[idx]['protein_names']
        complex_features = self.data[protein_name]
        complex_features['affinity'] = self.affinity_dict[protein_name]
        protein_node_coordinates = complex_features["tankbind_protein_alpha_carbon_coordinates"]
        compound_node_coordinates = complex_features["tankbind_ligand_atom_coordinates"]
        noised_protein_node_coordinates = protein_node_coordinates + torch.randn_like(protein_node_coordinates) * self.noise_range
        noised_compound_node_coordinates = compound_node_coordinates + torch.randn_like(compound_node_coordinates) * self.noise_range
        noised_pairwise_distance = torch.cdist(noised_protein_node_coordinates, noised_compound_node_coordinates, compute_mode="donot_use_mm_for_euclid_dist")
        compound_center_coordinates = compound_node_coordinates.mean(dim=0, keepdim=True)
        noised_compound_center_of_mass_coordinates = compound_center_coordinates + torch.randn_like(compound_center_coordinates) * self.noise_range
        mask = get_pocket_mask(protein_node_coordinates, noised_compound_center_of_mass_coordinates, self.pocket_radius) 


        data = make_tankbind_data_object(complex_features, normalize_features=self.normalize_features)

        data["protein", "distance_to", "compound"].edge_attr = noised_pairwise_distance.clamp(0,10)

        # concatenate esm embeddings just before 
        if self.esm_embeddings is not None:
            esm_value = (self.esm_embeddings[protein_name])[complex_features["tankbind_residue_indices_in_contact_with_compound"]]
            data["protein"].node_scalar_features = torch.cat([data["protein"].node_scalar_features, esm_value], dim=-1) # concatenate the ESM embeddings

        if self.normalize_features:
            data["protein", "distance_to", "compound"].edge_attr = torch.nan_to_num((data["protein", "distance_to", "compound"].edge_attr - means_dict["protein_distance_to_compound"])/(stds_dict["protein_distance_to_compound"]+1e-3))

        if mask.sum() < 5:
            mask = torch.zeros_like(mask, dtype=torch.bool)
            mask[:100] = True
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
        return ["proteins_csv.pt", "data.pt"] + (["esm_embeddings_650m.pt"] if self.add_esm_embeddings == "650m" else []) + (["esm_embeddings_15B.pt"] if self.add_esm_embeddings == "15B" else [])

    def process(self):
        proteins_df = pd.read_csv('/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/protein_paths_and_names.csv')
        data = {}
        if self.add_esm_embeddings == "650m":
            with open("/fs/pool/pool-marsot/bindbind/datasets/processing/esmcheckpoints/esm_embeddings_650m.pkl", "rb") as f:
                esm_embeddings = pickle.load(f)
        elif self.add_esm_embeddings == "15B":
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
    def __init__(self,
                 root="/fs/pool/pool-marsot/bindbind/datasets/tankbind_test",
                 add_esm_embeddings=None,
                 noise_range=0.0,
                 contact_threshold=8.0,
                 pocket_radius=20.0,
                 normalize_features=True,
                 **kwargs):
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
        self.normalize_features = normalize_features
        super().__init__(root, **kwargs)

        self.proteins_df = torch.load(self.processed_paths[0])
        self.pockets_df = torch.load(self.processed_paths[1])
        self.affinity_dict = self.proteins_df.set_index('protein_names')['affinity'].to_dict()

        self.data = torch.load(self.processed_paths[2])
        self.esm_embeddings = torch.load(self.processed_paths[3]) if add_esm_embeddings else None
        self.pocket_radius = pocket_radius
    def get(self, idx):
        protein_name = self.pockets_df.iloc[idx]['name']
        pocket_index = self.pockets_df.iloc[idx]["rank"] - 1 # ranks start at 1 in the dataframe.
        complex_features = self.data[protein_name]
        pocket_center_coordinates = (complex_features["tankbind_protein_pocket_center_coordinates"][pocket_index]).unsqueeze(0)
        complex_features['affinity'] = self.affinity_dict[protein_name]
        noised_pocket_center_coordinates = pocket_center_coordinates + torch.randn_like(pocket_center_coordinates)*self.noise_range 

        data = make_tankbind_data_object(complex_features, normalize_features=self.normalize_features)
        protein_node_coordinates = complex_features["tankbind_protein_alpha_carbon_coordinates"]
        compound_node_coordinates = complex_features["tankbind_ligand_atom_coordinates"]
        noised_protein_node_coordinates = protein_node_coordinates + torch.randn_like(protein_node_coordinates) * self.noise_range
        noised_compound_node_coordinates = compound_node_coordinates + torch.randn_like(compound_node_coordinates) * self.noise_range
        noised_pairwise_distance = torch.cdist(noised_protein_node_coordinates, noised_compound_node_coordinates, compute_mode="donot_use_mm_for_euclid_dist")
        data["protein", "distance_to", "compound"].edge_attr = noised_pairwise_distance.clamp(0,10)
        mask = get_pocket_mask(protein_node_coordinates, noised_pocket_center_coordinates, self.pocket_radius) 

    
        # concatenate esm embeddings just before 
        if self.esm_embeddings is not None:
            esm_value = (self.esm_embeddings[protein_name])[complex_features["tankbind_residue_indices_in_contact_with_compound"]]
            data["protein"].node_scalar_features = torch.cat([data["protein"].node_scalar_features, esm_value], dim=-1)  # concatenate the ESM embeddings

        if self.normalize_features:
            data["protein", "distance_to", "compound"].edge_attr = torch.nan_to_num((data["protein", "distance_to", "compound"].edge_attr - means_dict["protein_distance_to_compound"]) / (stds_dict["protein_distance_to_compound"] + 1e-3))


        if mask.sum() < 5:
            mask = torch.zeros_like(mask, dtype=torch.bool)
            mask[:100] = True
        data, _ligand_is_mostly_contained_in_pocket_sum = restrict_tankbind_data_to_pocket(data, mask, self.contact_threshold)

        # Make ligand_is_mostly_contained_in_protein_mask
        ligand_is_mostly_contained_in_pocket = _ligand_is_mostly_contained_in_pocket_sum / complex_features["tankbind_num_protein_nodes_close_to_ligand_and_in_contact_with_ligand"] >= 0.9
        data.ligand_is_mostly_contained_in_pocket = ligand_is_mostly_contained_in_pocket
        data.ligand_in_pocket_mask = torch.ones(data["protein", "distance_to", "compound"].edge_attr.shape).bool() * ligand_is_mostly_contained_in_pocket.bool()
        return data

    def len(self):
        return len(self.pockets_df)

    @property
    def processed_file_names(self):
        return ["proteins_csv.pt", "pockets_csv.pt", "data.pt"] + (["esm_embeddings_650m.pt"] if self.add_esm_embeddings == "650m" else []) + (["esm_embeddings_15B.pt"] if self.add_esm_embeddings == "15B" else [])

    def process(self):
        proteins_df = pd.read_csv('/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/protein_paths_and_names.csv')
        pockets_df = pd.read_csv("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/p2rank_predictions.csv")
        with open("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/tankbind_splits/timesplit_test", "r") as f:
            test_proteins = set(f.read().splitlines())
        data = {}
        if self.add_esm_embeddings == "650m":
            with open("/fs/pool/pool-marsot/bindbind/datasets/processing/esmcheckpoints/esm_embeddings_650m.pkl", "rb") as f:
                esm_embeddings = pickle.load(f)
        elif self.add_esm_embeddings == "15B":
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
    def __init__(self,
                 root="/fs/pool/pool-marsot/bindbind/datasets/tankbind_val",
                 add_esm_embeddings=None,
                 noise_range=0.0,
                 contact_threshold=8.0,
                 pocket_radius=20.0,
                 normalize_features=True,
                 **kwargs):
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
        self.normalize_features = normalize_features
        super().__init__(root, **kwargs)

        self.proteins_df = torch.load(self.processed_paths[0])
        self.pockets_df = torch.load(self.processed_paths[1])
        self.affinity_dict = self.proteins_df.set_index('protein_names')['affinity'].to_dict()

        self.data = torch.load(self.processed_paths[2])
        self.esm_embeddings = torch.load(self.processed_paths[3]) if add_esm_embeddings else None
        self.pocket_radius = pocket_radius
    def get(self, idx):
        protein_name = self.pockets_df.iloc[idx]['name']
        pocket_index = self.pockets_df.iloc[idx]["rank"] - 1 # ranks start at 1 in the dataframe.
        complex_features = self.data[protein_name]
        pocket_center_coordinates = (complex_features["tankbind_protein_pocket_center_coordinates"][pocket_index]).unsqueeze(0)

        complex_features['affinity'] = self.affinity_dict[protein_name]
        noised_pocket_center_coordinates = pocket_center_coordinates + torch.randn_like(pocket_center_coordinates)*self.noise_range 

        data = make_tankbind_data_object(complex_features, normalize_features=self.normalize_features)
        protein_node_coordinates = complex_features["tankbind_protein_alpha_carbon_coordinates"]
        compound_node_coordinates = complex_features["tankbind_ligand_atom_coordinates"]
        noised_protein_node_coordinates = protein_node_coordinates + torch.randn_like(protein_node_coordinates) * self.noise_range
        noised_compound_node_coordinates = compound_node_coordinates + torch.randn_like(compound_node_coordinates) * self.noise_range
        noised_pairwise_distance = torch.cdist(noised_protein_node_coordinates, noised_compound_node_coordinates, compute_mode="donot_use_mm_for_euclid_dist")
        data["protein", "distance_to", "compound"].edge_attr = noised_pairwise_distance.clamp(0,10)
        mask = get_pocket_mask(protein_node_coordinates, noised_pocket_center_coordinates, self.pocket_radius) 

        # concatenate esm embeddings just before 
        if self.esm_embeddings is not None:
            esm_value = (self.esm_embeddings[protein_name])[complex_features["tankbind_residue_indices_in_contact_with_compound"]]
            data["protein"].node_scalar_features = torch.cat([data["protein"].node_scalar_features, esm_value], dim=-1)  # concatenate the ESM embeddings


        if self.normalize_features:
            data["protein", "distance_to", "compound"].edge_attr = torch.nan_to_num((data["protein", "distance_to", "compound"].edge_attr - means_dict["protein_distance_to_compound"]) / (stds_dict["protein_distance_to_compound"] + 1e-3))

        if mask.sum() < 5:
            mask = torch.zeros_like(mask, dtype=torch.bool)
            mask[:100] = True
        data, _ligand_is_mostly_contained_in_pocket_sum = restrict_tankbind_data_to_pocket(data, mask, self.contact_threshold)

        # Make ligand_is_mostly_contained_in_protein_mask
        ligand_is_mostly_contained_in_pocket = _ligand_is_mostly_contained_in_pocket_sum / complex_features["tankbind_num_protein_nodes_close_to_ligand_and_in_contact_with_ligand"] >= 0.9
        data.ligand_is_mostly_contained_in_pocket = ligand_is_mostly_contained_in_pocket
        data.ligand_in_pocket_mask = torch.ones(data["protein", "distance_to", "compound"].edge_attr.shape).bool() * ligand_is_mostly_contained_in_pocket.bool()
        return data

    def len(self):
        return len(self.pockets_df)

    @property
    def processed_file_names(self):
        return ["proteins_csv.pt", "pockets_csv.pt", "data.pt"] + (["esm_embeddings_650m.pt"] if self.add_esm_embeddings == "650m" else []) + (["esm_embeddings_15B.pt"] if self.add_esm_embeddings == "15B" else [])

    def process(self):
        proteins_df = pd.read_csv('/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/protein_paths_and_names.csv')
        pockets_df = pd.read_csv("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/p2rank_predictions.csv")
        with open("/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/tankbind_splits/timesplit_no_lig_overlap_val", "r") as f:
            val_proteins = set(f.read().splitlines())
        data = {}
        if self.add_esm_embeddings == "650m":
            with open("/fs/pool/pool-marsot/bindbind/datasets/processing/esmcheckpoints/esm_embeddings_650m.pkl", "rb") as f:
                esm_embeddings = pickle.load(f)
        elif self.add_esm_embeddings == "15B":
            with open("/fs/pool/pool-marsot/bindbind/datasets/processing/esmcheckpoints/esm_embeddings_15B.pkl", "rb") as f:
                esm_embeddings = pickle.load(f)

        result_folder = "/fs/pool/pool-marsot/bindbind/experiments/ablations/regular/tankbind_predictions_2"
        os.system(f"mkdir -p {result_folder}")

        rdkit_folder = f"{result_folder}/rdkit/"
        os.system(f"mkdir -p {rdkit_folder}")
        for protein_name in set(proteins_df["protein_names"].tolist()).intersection(val_proteins):
            with open(f"/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/PDBBind_processed/{protein_name}/{protein_name}_full_data.pkl", "rb") as f:
                mol, _ = read_molecule(f"/fs/pool/pool-marsot/bindbind/datasets/data/equibind_dataset/PDBBind_processed/{protein_name}/{protein_name}_ligand_renumbered.sdf", None)
                smiles = Chem.MolToSmiles(mol)
                rdkit_mol_path = f"{rdkit_folder}/{protein_name}_ligand.sdf"
                generate_sdf_from_smiles_using_rdkit(smiles, rdkit_mol_path, shift_dis=0)

                mol, _ = read_molecule(rdkit_mol_path, None)

                tankbind_ligand_features = create_tankbind_ligand_features(rdkit_mol_path, None, has_LAS_mask=True)
                data[protein_name] = pickle.load(f)
                for key, item in tankbind_ligand_features.items():
                    data[protein_name][key] = item
        proteins_df = proteins_df[proteins_df["protein_names"].isin(val_proteins)]
        pockets_df = pockets_df[pockets_df["name"].isin(val_proteins)]
        torch.save(proteins_df, self.processed_paths[0])
        torch.save(pockets_df, self.processed_paths[1])
        torch.save(data, self.processed_paths[2])
        if self.add_esm_embeddings is not None:
            torch.save(esm_embeddings, self.processed_paths[3])


def get_pocket_mask(protein_node_coordinates, pocket_center_coordinates, pocket_radius):
    """Get the mask of the protein nodes that are in the pocket."""
    assert pocket_center_coordinates.dim() == 2
    return (torch.norm(protein_node_coordinates - pocket_center_coordinates, dim=-1) < pocket_radius)


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
    data["protein", "distance_to", "compound"].edge_attr = data["protein", "distance_to", "compound"].edge_attr.flatten()
    return data, _ligand_is_mostly_contained_in_pocket_sum



def make_tankbind_data_object(complex_dict, normalize_features=True):
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
    if normalize_features:
        return normalize_data_features(data)
    else:
        return data

from torch import tensor
means_dict = {'protein_node_scalar_features': tensor([ 0.0727457255,  0.1325012892, -0.9804269075, -0.7606167793,
         0.0248864293,  0.0118933702]), 
        'protein_node_vector_features': tensor([[-0.0006267764,  0.0006361017,  0.0003718575],
        [ 0.0006267764, -0.0006361017, -0.0003718575],
        [-0.0001250555,  0.0010360993, -0.0001783383]]),
    'protein_edge_scalar_features': tensor([ 6.9658167376e-06,  1.3754346874e-03,  3.1302969903e-02,
         1.0506523401e-01,  1.6300360858e-01,  1.7937441170e-01,
         2.1146914363e-01,  3.1705018878e-01,  3.0818232894e-01,
         1.8676058948e-01,  9.4478376210e-02,  4.0007371455e-02,
         1.4457522891e-02,  5.0572101027e-03,  1.9602947868e-03,
         8.6626538541e-04, -3.6794535816e-02,  9.7907505929e-02,
         2.7249491215e-01,  4.7467842698e-01,  7.6569545269e-01,
         9.5167309046e-01,  9.9420434237e-01,  9.9940609932e-01,
        -6.3369655982e-04,  2.6633599191e-04,  7.6850854384e-05,
         5.2463921020e-04, -1.2254845351e-03,  1.4618394198e-04,
         1.0501391807e-04,  3.4044867789e-05]), 
         'protein_edge_vector_features': tensor([[-2.4531493546e-04, -2.2749346681e-04, -6.0290578404e-05]]),
         'protein_distance_to_compound': tensor(9.8255100250),
         'affinity': tensor(6.3676585872)}

stds_dict = {'protein_node_scalar_features': tensor([0.4711185396, 0.7746803761, 0.1705850214, 0.4407012165, 0.6178143024,
        0.0975829586]),
        'protein_node_vector_features': tensor([[0.5764503479, 0.5763314366, 0.5768814683],
        [0.5764503479, 0.5763314366, 0.5768814683],
        [0.5775108337, 0.5781271458, 0.5764107108]]),
        'protein_edge_scalar_features': tensor([6.2377876020e-05, 5.5922851898e-03, 1.0898678005e-01, 2.6428866386e-01,
        3.0745354295e-01, 3.0576357245e-01, 3.0775597692e-01, 3.6643519998e-01,
        3.6639633775e-01, 3.0458340049e-01, 2.3440334201e-01, 1.5616969764e-01,
        9.3920931220e-02, 5.5855795741e-02, 3.5477731377e-02, 2.3793805391e-02,
        6.9635939598e-01, 6.8488472700e-01, 7.2912204266e-01, 6.6055721045e-01,
        4.5727527142e-01, 1.5990658104e-01, 2.7558168396e-02, 3.0823557172e-03,
        7.1674913168e-01, 7.2204363346e-01, 6.2779581547e-01, 5.8167368174e-01,
        4.5233646035e-01, 2.6219883561e-01, 1.0391493887e-01, 3.4320969135e-02]),
        'protein_edge_vector_features': tensor([[0.5767907500, 0.5773115158, 0.5779478550]]),
        'protein_distance_to_compound': tensor(0.7166844010),
        'affinity': tensor(1.8645911818)}


def normalize_data_features(data, epsilon = 1e-3):
    # Should not normalize protein coordinates because they are binned.
    #data["protein"].coordinates = torch.nan_to_num((data["protein"].coordinates - means_dict["protein_node_coordinates"])/(stds_dict["protein_node_coordinates"]+epsilon))
    data["protein"].node_scalar_features = torch.nan_to_num((data["protein"].node_scalar_features - means_dict["protein_node_scalar_features"])/(stds_dict["protein_node_scalar_features"]+epsilon))
    data["protein"].node_vector_features = torch.nan_to_num((data["protein"].node_vector_features - means_dict["protein_node_vector_features"])/(stds_dict["protein_node_vector_features"]+epsilon))
    data["protein", "to", "protein"].edge_scalar_features = torch.nan_to_num((data["protein", "to", "protein"].edge_scalar_features - means_dict["protein_edge_scalar_features"])/(stds_dict["protein_edge_scalar_features"]+epsilon))
    data["protein", "to", "protein"].edge_vector_features = torch.nan_to_num((data["protein", "to", "protein"].edge_vector_features - means_dict["protein_edge_vector_features"])/(stds_dict["protein_edge_vector_features"]+epsilon))
    data.affinity = torch.nan_to_num((data.affinity - means_dict["affinity"])/(stds_dict["affinity"]+epsilon))
    return data

def denormalize_feature(tensor, feature, epsilon=1e-3):
    unstandardized = tensor*(epsilon + stds_dict[feature])
    return unstandardized + means_dict[feature]


