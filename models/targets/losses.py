import torch
import torch.nn.functional as F

def mse_loss(y_pred, y_target):
    """Measure the accuracy of the binding site prediction.
    We use MSE for the atoms that are supposed to be close to residues,
    but for atoms that are further than 10A, we only penalize the prediction being lower than 10A.
    Parameters
    ----------
    y_pred: torch.Tensor: shape [n_batch*n_residues*n_atoms]
        Predicted pairwise distances between atoms.
    y_target: torch.Tensor: shape [n_batch*n_residues*n_atoms]
    """
    return F.mse_loss(y_pred.clamp(0, 10), y_target.clamp(0, 10))

def affinity_loss(affinity_pred, affinity_target):
    """#FIXME probably does not make sense from a physical point of view
    to choose an MSE loss: affinities may not be distributed like a gaussian
    but rather in a logarithmic fashion"""
    return F.mse_loss(affinity_pred, affinity_target)

def contrastive_affinity_loss(affinity_pred, affinity_target, ligand_is_mostly_contained_in_pocket, contrastive_epsilon=1.0):
    affinity_loss = torch.zeros(affinity_pred.shape, device=affinity_pred.device)
    affinity_loss[ligand_is_mostly_contained_in_pocket] = ((affinity_pred-affinity_target)**2)[ligand_is_mostly_contained_in_pocket]
    affinity_loss[~ligand_is_mostly_contained_in_pocket] = ((affinity_pred-(affinity_target-contrastive_epsilon)).relu()**2)[~ligand_is_mostly_contained_in_pocket]
    return affinity_loss.mean()

def total_loss(y_pred, y_target, affinity_pred, affinity_target, pocket_mask, affinity_mask):
    if pocket_mask is not None:
        y_pred = y_pred[pocket_mask]
        y_target = y_target[pocket_mask]
    
    mse = mse_loss(y_pred, y_target) if y_pred.numel() > 0 else torch.tensor([0], device=y_pred.device)
    aff = contrastive_affinity_loss(affinity_pred, affinity_target, affinity_mask)
    return (mse, aff)