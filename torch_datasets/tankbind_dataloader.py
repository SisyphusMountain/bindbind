from torch_geometric.loader.dataloader import Collater
from torch_geometric.utils import to_dense_batch
import torch


class TankBindDataLoader(torch.utils.data.DataLoader):
    """Subclass of the torch DataLoader, in order to apply the collate function TankBindCollater."""
    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 follow_batch=None,
                 exclude_keys=None,
                 make_divisible_by_8=True,
                 **kwargs):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.make_divisible_by_8=make_divisible_by_8
        super().__init__(dataset,
                         batch_size,
                         shuffle,
                         collate_fn=TankBindCollater(dataset, follow_batch, exclude_keys, make_divisible_by_8=self.make_divisible_by_8),
                         **kwargs)



class TankBindCollater(Collater):
    """Applies batching operations and computations of masks in place of the model, in order to avoid having to recompute it in the
    forward pass on GPU."""
    def __init__(self, dataset,
                 follow_batch=None,
                 exclude_keys=None,
                 make_divisible_by_8=True):
        super().__init__(dataset, follow_batch, exclude_keys)
        self.make_divisible_by_8 = make_divisible_by_8
    def __call__(self, batch):
        data = super().__call__(batch)
        if self.make_divisible_by_8:
            max_dim_divisible_by_8_protein = 8 * (torch.diff(data["protein"].ptr).max() // 8 + 1)
            max_dim_divisible_by_8_compound = 8 * (torch.diff(data["compound"].ptr).max() // 8 + 1)
        else:
            max_dim_divisible_by_8_protein = torch.diff(data["protein"].ptr).max()
            max_dim_divisible_by_8_compound = torch.diff(data["compound"].ptr).max()
        protein_coordinates_batched, _ = to_dense_batch(
            data["protein"].coordinates, data["protein"].batch,
            max_num_nodes=max_dim_divisible_by_8_protein,
            )
        protein_pairwise_representation = get_pair_dis_index(
            protein_coordinates_batched,
            bin_size=2,
            bin_min=-1,
            bin_max=protein_bin_max,
            ) # shape [batch_n, max_protein_size, max_protein_size, 16]
        _compound_lengths = (data["compound"].ptr[1:] - data["compound"].ptr[:-1]) ** 2
        _total = torch.cumsum(_compound_lengths, 0)
        compound_pairwise_distance_batch = torch.zeros(
                _total[-1], dtype=torch.long
            )
        for i in range(len(_total) - 1):
            compound_pairwise_distance_batch[_total[i] : _total[i + 1]] = i + 1
        compound_pair_batched, compound_pair_batched_mask = to_dense_batch(
            data["compound", "distance_to", "compound"].edge_attr,
            compound_pairwise_distance_batch,
            )
        compound_pairwise_representation = torch.zeros(
            (len(batch), max_dim_divisible_by_8_compound, max_dim_divisible_by_8_compound, 16),
            dtype=torch.float32,
            )
        for i in range(len(batch)):
            one = compound_pair_batched[i]
            # collate
            compound_size_square = (compound_pairwise_distance_batch == i).sum()
            compound_size = int(compound_size_square**0.5)
            compound_pairwise_representation[i, :compound_size, :compound_size] = one[
                :compound_size_square
                ].reshape((compound_size, compound_size, -1))
        data.batch_n = len(batch)
        data.max_dim_divisible_by_8_protein = max_dim_divisible_by_8_protein
        data.max_dim_divisible_by_8_compound = max_dim_divisible_by_8_compound
        data["protein", "to", "protein"].pairwise_representation = protein_pairwise_representation
        data["compound", "to", "compound"].pairwise_representation = compound_pairwise_representation
        data["compound", "to", "compound"].pairwise_representation_mask = compound_pair_batched_mask
        return data




def get_pair_dis_index(d, bin_size=2, bin_min=-1, bin_max=30):
    """
    Computing pairwise distances and binning.
    """
    pair_dis = torch.cdist(d, d, compute_mode='donot_use_mm_for_euclid_dist')
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    return pair_dis_bin_index

protein_bin_max = 30