import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from .models.tankbind_layers import TankBindProteinEmbedding, TankBindUnembedding, GINE, TankBindTrigonometry, get_pair_dis_index
from torch_geometric.utils import to_dense_batch
from bindbind.models.targets.losses import total_loss
import logging
from lightning.pytorch.utilities import grad_norm

#logger = logging.getLogger(__name__)
#logger.setLevel(logging.WARN)


class TankBindModel(L.LightningModule):
    def __init__(
        self,
        cfg=None,
        logger=None,
        use_gvp=True,
        in_channels_compound_nodes=56,
        in_channels_compound_edges=19,
        embedding_channels=128,
        c=128,
        dropout=0.0,
        n_trigonometry_module_stack=5,
        protein_bin_max=30,
        fast_attention=True,
        esm_features=None,
    ):
        self.esm_features = esm_features
        self.use_gvp = use_gvp
        if esm_features == "15B":
            embedding_channels_protein = 5120+6
        elif esm_features == "650m":
            embedding_channels_protein = 1280+6
        else:
            embedding_channels_protein = 6
        self.cfg=cfg
        super().__init__()
        if use_gvp:
            self.protein_embedding = TankBindProteinEmbedding(
                node_in_dim=(embedding_channels_protein, 3),
                node_h_dim=(embedding_channels, 16),
                edge_in_dim=(32, 1),
                edge_h_dim=(32, 1),
            )
        else:
            self.protein_embedding = nn.Linear(embedding_channels_protein, embedding_channels)

        self.compound_node_linear = nn.Linear(in_channels_compound_nodes, embedding_channels)
        self.compound_edge_linear = nn.Linear(in_channels_compound_edges, embedding_channels)
        self.compound_embedding = GINE(
          emb_dim=embedding_channels, num_layer=5, drop_ratio=dropout,
        )
        self.protein_pair_embedding = nn.Embedding(16, c)
        self.compound_pair_embedding = nn.Linear(16, c)
        self.trigonometry_blocks = nn.ModuleList(
            [
                TankBindTrigonometry(embedding_channels=embedding_channels, c=c, fast_attention=fast_attention)
                for _ in range(n_trigonometry_module_stack)
            ]
        )
        self.unembedding = TankBindUnembedding(embedding_channels, 1)
        self.layernorm = nn.LayerNorm(embedding_channels)
        self.n_heads=4
        self.protein_bin_max = protein_bin_max
        self.logger_fn = logger

    def forward(self, data):
        max_dim_divisible_by_8_protein = data.max_dim_divisible_by_8_protein
        max_dim_divisible_by_8_compound = data.max_dim_divisible_by_8_compound
        if self.use_gvp:
            protein = self.protein_embedding(
                h_V=(
                    data["protein"]["node_scalar_features"],
                    data["protein"]["node_vector_features"],
                ),
                edge_index=data[("protein", "to", "protein")]["edge_index"],
                h_E=(
                    data[("protein", "to", "protein")]["edge_scalar_features"],
                    data[("protein", "to", "protein")]["edge_vector_features"],
                ),
                seq=data.seq,
            )
        else:
            protein = self.protein_embedding(data["protein"].x)

        compound_node_features = self.compound_node_linear(data["compound"].x.to(self.dtype))
        compound_edge_features = self.compound_edge_linear(data["compound", "to", "compound"].edge_attr)
        compound_embedding_tensor = self.compound_embedding(
            x=compound_node_features,
            edge_index=data["compound", "to", "compound"].edge_index,
            edge_attr=compound_edge_features,
        )
        
        protein_batched, protein_mask = to_dense_batch(protein, data["protein"].batch, max_num_nodes=max_dim_divisible_by_8_protein)
        compound_batched, compound_mask = to_dense_batch(
            compound_embedding_tensor, data["compound"].batch,
            max_num_nodes=max_dim_divisible_by_8_compound,
        )

        protein_pairwise_representation = data["protein", "to", "protein"].pairwise_representation # shape [batch_n, max_protein_size, max_protein_size, 16]
        compound_pairwise_representation = data["compound", "to", "compound"].pairwise_representation # shape [batch_n, max_compound_size, max_compound_size, 16]
        ######## MASKING ########
        batch_n = data.batch_n
        z_mask = torch.einsum("bi,bj->bij", protein_mask, compound_mask)
        z_mask_attention = torch.einsum("bik, bq-> biqk", z_mask, compound_mask).reshape(batch_n*protein_batched.shape[1], max_dim_divisible_by_8_compound, max_dim_divisible_by_8_compound).unsqueeze(1).expand(-1, self.n_heads, -1, -1).contiguous()
        z_mask_attention = torch.where(z_mask_attention, 0.0, -10.0**6)
        z_mask_flat = torch.arange(
            start=0, end=z_mask.numel(), device=self.device
        ).view(z_mask.shape)[z_mask]
        protein_square_mask = torch.einsum("bi,bj->bij", protein_mask, protein_mask)
        

        # Fake protein nodes have been put at the origin, with coordinates 0,0,0
        # After computing pairwise distances, some embeddings can now be completely meaningless. We need to multiply by the mask to avoid it.
        protein_pairwise_representation = self.protein_pair_embedding(
            protein_pairwise_representation
        )*protein_square_mask.unsqueeze(-1)
        # This embedding is safe for now: no fake values
        compound_pairwise_representation = self.compound_pair_embedding(
            compound_pairwise_representation
        )

        protein_batched = self.layernorm(protein_batched)
        compound_batched = self.layernorm(compound_batched)
        # Here z acts as a mask for the fake values in protein_batched, and sets them to 0.
        # For real nodes, we still get regular elementwise products.
        z = torch.einsum("bik,bjk->bijk", protein_batched, compound_batched)


        for block in self.trigonometry_blocks:
            z = block(
                z,
                protein_pairwise_representation,
                compound_pairwise_representation,
                z_mask,
                z_mask_attention.to(z.dtype)
            )
        result = self.unembedding(z, z_mask, z_mask_flat)
        #logger.info(f"[Forward] Result Nans:{torch.isnan(result[0]).sum().detach().cpu()} {torch.isnan(result[1]).sum().detach().cpu()}")
        #logger.info(f"[Forward] Result norms:{torch.norm(result[0]).detach().cpu()} {torch.norm(result[1]).detach().cpu()}")
        return result


    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     #logger.info(f"[On before optimizer step] Computing norms")
    #     norms = grad_norm(self, norm_type=2)
    #     # Compute the max of norms over all keys
    #     max_norm = max([torch.norm(norm) for norm in norms.values()])
    #     # Log the max norm
    #     self.log_dict({"max_grad_norm": max_norm})
    #     self.log_dict(norms)

    #def on_before_backward(self, loss):
    #    logging.info(f"[On before backward] Computing norms")
         #from IPython import embed; embed(); exit(

    def training_step(self, data, batch_idx):
        #logger.info(f"[Beginning Batch - training step] Batch {batch_idx}")
        y_pred, affinity_pred = self(data)
        #logger.info(f"[Beginning Batch - model output] y_pred Nans:{torch.isnan(y_pred).sum().detach().cpu()}; affinity_pred Nans:{torch.isnan(affinity_pred).sum().detach().cpu()}")

        affinity_target = data.affinity
        y_target = data['protein', 'distance_to', 'compound'].edge_attr
        e = self.current_epoch
        affinity_coeff = 0.01 
        affinity_mask = data.ligand_is_mostly_contained_in_pocket
        pocket_mask = data.ligand_in_pocket_mask
        mse, affinity_loss = total_loss(y_pred, y_target, affinity_pred, affinity_target, pocket_mask, affinity_mask)
        #logger.info(f"[Beginning Batch - losses output] MSE {mse.detach().cpu()}; Affinity Loss {affinity_loss.detach().cpu()}; Affinity coef {affinity_coeff}; Epoch {e}")
        loss = mse + affinity_coeff*affinity_loss
        self.log('train_mse', mse, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_affinity_loss', affinity_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('affinity_coeff', affinity_coeff, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("y_pred_mean", y_pred.clamp(0, 10).mean(), on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("y_pred_std", y_pred.clamp(0,10).std(), on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("y_target_mean", y_target.clamp(0, 10).mean(), on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("y_target_std", y_target.clamp(0, 10).std(), on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("affinity_pred_mean", affinity_pred.mean(), on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("affinity_pred_std", affinity_pred.std(), on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("affinity_target_mean", affinity_target.mean(), on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("affinity_target_std", affinity_target.std(), on_step=True, on_epoch=False, prog_bar=False, logger=True)
        # if torch.isnan(loss):
        #     from IPython import embed; embed(); exit()
        #from IPython import embed; embed(); exit()
        return loss

    def validation_step(self, data, batch_idx):
        y_pred, affinity_pred = self(data)


        affinity_target = data.affinity
        y_target = data['protein', 'distance_to', 'compound'].edge_attr
        affinity_mask = data.ligand_is_mostly_contained_in_pocket
        pocket_mask = data.ligand_in_pocket_mask
        mse, affinity_loss = total_loss(y_pred, y_target, affinity_pred, affinity_target, pocket_mask, affinity_mask)
        loss = mse # No affinity loss in validation
        self.log('val_mse', mse, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('val_affinity_loss', affinity_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    
    def predict_step(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.training.lr)

        if self.cfg.training.scheduler is not None:
            raise NotImplementedError("Scheduler not implemented yet")
            return {"optimizer":optimizer, "lr_scheduler":scheduler}

        else:
            return optimizer
    

def halve_batch(batch):
    new_size = batch.batch_size // 2

    # First half
    first_half = batch.clone()
    second_half = batch.clone()

    first_half["complex_name"] = first_half["complex_name"][:new_size]
    first_half["ligand_is_mostly_contained_in_pocket"] = first_half["ligand_is_mostly_contained_in_pocket"][:new_size]
    first_half["affinity"] = first_half["affinity"][:new_size]

    second_half["complex_name"] = second_half["complex_name"][new_size:]
    second_half["ligand_is_mostly_contained_in_pocket"] = second_half["ligand_is_mostly_contained_in_pocket"][new_size:]
    second_half["affinity"] = second_half["affinity"][new_size:]

    protein_ptr = batch["protein"].ptr
    compound_ptr = batch["compound"].ptr

    new_protein_ptr_first_half = protein_ptr[:new_size+1]
    new_compound_ptr_first_half = compound_ptr[:new_size+1]

    new_protein_ptr_second_half = protein_ptr[new_size:]
    new_compound_ptr_second_half = compound_ptr[new_size:]

    first_half["protein"].ptr = new_protein_ptr_first_half
    first_half["compound"].ptr = new_compound_ptr_first_half

    second_half["protein"].ptr = new_protein_ptr_second_half - new_protein_ptr_second_half[0]
    second_half["compound"].ptr = new_compound_ptr_second_half - new_compound_ptr_second_half[0]

    protein_max_first_half = new_protein_ptr_first_half[-1]
    compound_max_first_half = new_compound_ptr_first_half[-1]

    protein_max_second_half = new_protein_ptr_second_half[-1] - new_protein_ptr_second_half[0]
    compound_max_second_half = new_compound_ptr_second_half[-1] - new_compound_ptr_second_half[0]

    first_half["protein"].coordinates = first_half["protein"].coordinates[:protein_max_first_half]
    first_half["protein"].node_scalar_features = first_half["protein"].node_scalar_features[:protein_max_first_half]
    first_half.seq = first_half.seq[:protein_max_first_half]
    first_half["protein"].node_vector_features = first_half["protein"].node_vector_features[:protein_max_first_half]
    first_half["protein"].batch = first_half["protein"].batch[:protein_max_first_half]

    second_half["protein"].coordinates = second_half["protein"].coordinates[protein_max_first_half:]
    second_half["protein"].node_scalar_features = second_half["protein"].node_scalar_features[protein_max_first_half:]
    second_half.seq = second_half.seq[protein_max_first_half:]
    second_half["protein"].node_vector_features = second_half["protein"].node_vector_features[protein_max_first_half:]
    second_half["protein"].batch = second_half["protein"].batch[protein_max_first_half:] - new_protein_ptr_second_half[0]

    first_half["compound"].x = first_half["compound"].x[:compound_max_first_half]
    first_half["compound"].batch = first_half["compound"].batch[:compound_max_first_half]

    second_half["compound"].x = second_half["compound"].x[compound_max_first_half:]
    second_half["compound"].batch = second_half["compound"].batch[compound_max_first_half:] - new_compound_ptr_second_half[0]

    protein_edges_to_keep_first_half = (first_half["protein", "to", "protein"].edge_index[0] < protein_max_first_half) & (first_half["protein", "to", "protein"].edge_index[1] < protein_max_first_half)
    first_half["protein", "to", "protein"].edge_index = first_half["protein", "to", "protein"].edge_index[:, protein_edges_to_keep_first_half]
    first_half["protein", "to", "protein"].edge_scalar_features = first_half["protein", "to", "protein"].edge_scalar_features[protein_edges_to_keep_first_half]
    first_half["protein", "to", "protein"].edge_vector_features = first_half["protein", "to", "protein"].edge_vector_features[protein_edges_to_keep_first_half]

    protein_edges_to_keep_second_half = (second_half["protein", "to", "protein"].edge_index[0] >= protein_max_first_half) & (second_half["protein", "to", "protein"].edge_index[1] >= protein_max_first_half)
    second_half["protein", "to", "protein"].edge_index = second_half["protein", "to", "protein"].edge_index[:, protein_edges_to_keep_second_half] - protein_max_first_half
    second_half["protein", "to", "protein"].edge_scalar_features = second_half["protein", "to", "protein"].edge_scalar_features[protein_edges_to_keep_second_half]
    second_half["protein", "to", "protein"].edge_vector_features = second_half["protein", "to", "protein"].edge_vector_features[protein_edges_to_keep_second_half]

    compound_edges_to_keep_first_half = (first_half["compound", "to", "compound"].edge_index[0] < compound_max_first_half) & (first_half["compound", "to", "compound"].edge_index[1] < compound_max_first_half)
    first_half["compound", "to", "compound"].edge_index = first_half["compound", "to", "compound"].edge_index[:, compound_edges_to_keep_first_half]
    first_half["compound", "to", "compound"].edge_attr = first_half["compound", "to", "compound"].edge_attr[compound_edges_to_keep_first_half]

    compound_edges_to_keep_second_half = (second_half["compound", "to", "compound"].edge_index[0] >= compound_max_first_half) & (second_half["compound", "to", "compound"].edge_index[1] >= compound_max_first_half)
    second_half["compound", "to", "compound"].edge_index = second_half["compound", "to", "compound"].edge_index[:, compound_edges_to_keep_second_half] - compound_max_first_half
    second_half["compound", "to", "compound"].edge_attr = second_half["compound", "to", "compound"].edge_attr[compound_edges_to_keep_second_half]





    compound_differences = torch.diff(batch["compound"].ptr)
    compound_differences_squared = compound_differences**2
    compound_squared_ptr = torch.cumsum(torch.cat((torch.tensor([0], device=self.device),compound_differences_squared), 0), 0)
    protein_differences = torch.diff(batch["protein"].ptr)
    protein_compound_ptr = torch.cumsum(torch.cat((torch.tensor([0], device=self.device), protein_differences*compound_differences),0), 0)
    first_half["compound", "distance_to", "compound"].edge_attr = first_half["compound", "distance_to", "compound"].edge_attr[:compound_squared_ptr[new_size]]
    second_half["compound", "distance_to", "compound"].edge_attr = second_half["compound", "distance_to", "compound"].edge_attr[compound_squared_ptr[new_size]:]

    first_half["protein", "distance_to", "compound"].edge_attr = first_half["protein", "distance_to", "compound"].edge_attr[:protein_compound_ptr[new_size]]
    second_half["protein", "distance_to", "compound"].edge_attr = second_half["protein", "distance_to", "compound"].edge_attr[protein_compound_ptr[new_size]:]
    first_half.ligand_in_pocket_mask = first_half.ligand_in_pocket_mask[:protein_compound_ptr[new_size]]
    second_half.ligand_in_pocket_mask = second_half.ligand_in_pocket_mask[protein_compound_ptr[new_size]:]
    return first_half, second_half