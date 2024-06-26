import torch
from torch_geometric.utils import to_dense_batch
from itertools import chain


def get_all_nodes_to_keep(complex_dict_dataset,
                          key_1="tankbind_protein_alpha_carbon_coordinates",
                          key_2="tankbind_protein_pocket_center_coordinates",
                          pocket_radius=20.0,
                          batch_size=1024,
                          ):
    """Obtain tensors indicating which nodes to keep, corresponding to the different pockets
    predicted by p2rank."""
    protein_nodes_to_keep_dict = {}
    inputs = get_batched_input_for_pairwise_distances(complex_dict_dataset=complex_dict_dataset,
                                                      key_1=key_1,
                                                      key_2=key_2,
                                                      batch_size=batch_size)

    for sample in inputs:
        names, coordinates_1, coordinates_2, batch_1, batch_2, ptr_1, ptr_2 = sample
        pairwise_distances, mask = add_noise_to_batch_and_compute_pairwise_distances(coordinates_1, coordinates_2, batch_1, batch_2, noise_level=0)
        protein_nodes_to_keep = get_nodes_to_keep(pairwise_distances, mask, pocket_radius, ptr_1, ptr_2)
        protein_nodes_to_keep_dict.update(dict(zip(names, protein_nodes_to_keep)))
    return protein_nodes_to_keep_dict

def get_all_noised_pairwise_distances(complex_dict_dataset,
                                      noise_level=0.1,
                                      key_1="tankbind_protein_alpha_carbon_coordinates",
                                      key_2="tankbind_ligand_atom_coordinates",
                                      batch_size=1024):
    """Obtain pairwise distances between two kinds of elements of the complex dataset, with added noise."""
    noised_pairwise_distances_dict = {}
    iterator_batches = get_batched_input_for_pairwise_distances(complex_dict_dataset=complex_dict_dataset,
                                                      key_1=key_1,
                                                      key_2=key_2,
                                                      batch_size=batch_size)
    for batched_input in iterator_batches:
        names, coordinates_1, coordinates_2, batch_1, batch_2, ptr_1, ptr_2 = batched_input
        pairwise_distances, mask = add_noise_to_batch_and_compute_pairwise_distances(coordinates_1, coordinates_2, batch_1, batch_2, noise_level)
        list_pairwise_distances = unbatch_pairwise_distances(pairwise_distances, ptr_1, ptr_2, mask)
        noised_pairwise_distances_dict.update(dict(zip(names, list_pairwise_distances)))

    return noised_pairwise_distances_dict








def add_noise_to_batch_and_compute_pairwise_distances(concatenated_coordinates_1, concatenated_coordinates_2, batch_1, batch_2, noise_level, device=None):
    """Add noise to the coordinates of the atoms and compute the pairwise distances between the atoms.
    
    Parameters
    ----------

    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        concatenated_coordinates_1 = concatenated_coordinates_1.to(device)
        concatenated_coordinates_2 = concatenated_coordinates_2.to(device)
        batch_1 = batch_1.to(device)
        batch_2 = batch_2.to(device)
        if noise_level != 0:
            concatenated_coordinates_1 += noise_level * (2*torch.randn_like(concatenated_coordinates_1, device=device)-torch.ones_like(concatenated_coordinates_1, device=device))
            concatenated_coordinates_2 += noise_level * (2*torch.randn_like(concatenated_coordinates_2, device=device)-torch.ones_like(concatenated_coordinates_2, device=device))
        batched_coordinates_1, mask_1 = to_dense_batch(concatenated_coordinates_1, batch_1)
        batched_coordinates_2, mask_2 = to_dense_batch(concatenated_coordinates_2, batch_2)
        mask = torch.einsum("bi,bj->bij", mask_1, mask_2)
        batched_coordinates_1 = batched_coordinates_1
        batched_coordinates_2 = batched_coordinates_2
        pairwise_distances = torch.cdist(batched_coordinates_1.to(torch.double), batched_coordinates_2.to(torch.double)).to(torch.float32)
    return pairwise_distances, mask

def unbatch_pairwise_distances(pairwise_distances, ptr_1, ptr_2, mask, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        flat_pairwise_distances = pairwise_distances[mask]
        list_dimensions_1 = [ptr_1[i+1]-ptr_1[i] for i in range(len(ptr_1)-1)]
        list_dimensions_2 = [ptr_2[i+1]-ptr_2[i] for i in range(len(ptr_2)-1)]
        list_products = [0] + [a*b for (a,b) in zip(list_dimensions_1, list_dimensions_2)]
        cumulative_products = torch.cumsum(torch.tensor(list_products, device=device), dim=0)
        list_pairwise_distances = [flat_pairwise_distances[cumulative_products[i]:cumulative_products[i+1]].reshape(list_dimensions_1[i], list_dimensions_2[i]).cpu() for i in range(len(list_dimensions_1))]
        return list_pairwise_distances

def get_batched_input_for_pairwise_distances(complex_dict_dataset, key_1, key_2, batch_size):
    n_batches = len(complex_dict_dataset) // batch_size + 1
    protein_names = list(complex_dict_dataset.keys())
    all_coordinates_1 = [complex_dict_dataset[name][key_1] for name in protein_names]

    all_coordinates_2 = [complex_dict_dataset[name][key_2] for name in protein_names]

    for i in range(n_batches):
        start_batch = i*batch_size
        end_batch = min((i+1)*batch_size, len(protein_names))

        coordinates_1 = all_coordinates_1[start_batch:end_batch]
        coordinates_2 = all_coordinates_2[start_batch:end_batch]

        batch_1 = torch.tensor(list(chain(*[[i]*coords.shape[0] for i, coords in enumerate(coordinates_1)])))
        batch_2 = torch.tensor(list(chain(*[[i]*coords.shape[0] for i, coords in enumerate(coordinates_2)])))
        assert batch_1.dim() == 1
        assert batch_2.dim() == 1

        ptr_1 = torch.cumsum(torch.tensor([0] + [coords.shape[0] for coords in coordinates_1]), dim=0)
        ptr_2 = torch.cumsum(torch.tensor([0] + [coords.shape[0] for coords in coordinates_2]), dim=0)

        names = protein_names[start_batch:end_batch]

        coordinates_1 = torch.cat(coordinates_1, dim=0)
        coordinates_2 = torch.cat(coordinates_2, dim=0)
        yield names, coordinates_1, coordinates_2, batch_1, batch_2, ptr_1, ptr_2





def get_nodes_to_keep(distances_to_center_of_pockets, mask, pocket_radius, ptr_1, ptr_2, device=None):
    """
    
    Parameters
    ----------
    distances_to_center_of_pockets:torch.Tensor:shape [batch_size, n_protein_nodes, n_pocket_centers]

    mask: torch.Tensor[bool]:shape [batch_size, n_protein_nodes, n_pocket_centers]
    Returns
    -------
    list[torch.Tensor[bool]: shape [n_pocket_centers, n_protein_nodes]]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        mask_is_close_and_real = torch.logical_and(mask, distances_to_center_of_pockets<pocket_radius)
        flat_inclusion_values = mask_is_close_and_real[mask]
        list_dimensions_1 = [ptr_1[i+1]-ptr_1[i] for i in range(len(ptr_1)-1)]
        list_dimensions_2 = [ptr_2[i+1]-ptr_2[i] for i in range(len(ptr_2)-1)]
        list_products = [0] + [a*b for (a,b) in zip(list_dimensions_1, list_dimensions_2)]
        cumulative_products = torch.cumsum(torch.tensor(list_products, device=device), dim=0)
        list_inclusion_values = [flat_inclusion_values[cumulative_products[i]:cumulative_products[i+1]].reshape(list_dimensions_1[i], list_dimensions_2[i]).transpose(-1,-2).cpu() for i in range(len(list_dimensions_1))]
        return list_inclusion_values