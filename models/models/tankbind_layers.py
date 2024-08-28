import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import GINEConv
import functools
from torch_scatter import scatter_add
import xformers.ops as xops
from torch_geometric.nn import MessagePassing
import logging

#logger = logging.getLogger(__name__)
#logger.setLevel(logging.WARN)

def get_pair_dis_index(d, bin_size=2, bin_min=-1, bin_max=30):
    """
    Computing pairwise distances and binning. 
    """
    pair_dis = torch.cdist(d, d, compute_mode='donot_use_mm_for_euclid_dist')
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    return pair_dis_bin_index


class TankBindTransition(nn.Module):
    def __init__(self, embedding_channels, transition_expansion_factor):
        super().__init__()
        self.layernorm = nn.LayerNorm(embedding_channels)
        self.linear1 = nn.Linear(
            embedding_channels, transition_expansion_factor * embedding_channels
        )
        self.linear2 = nn.Linear(
            transition_expansion_factor * embedding_channels, embedding_channels
        )

    def forward(self, z):
        z = self.layernorm(z)
        z = F.relu(self.linear2(self.linear1(z)))
        return z

class FastTriangleSelfAttention(nn.Module):
    def __init__(self, embedding_channels, num_attention_heads):
        super().__init__()
        self.layernorm = nn.LayerNorm(embedding_channels, bias=False)
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = embedding_channels // num_attention_heads
        self.linear_qkv = nn.Linear(embedding_channels, 3*embedding_channels, bias=False)
        self.output_linear = nn.Linear(embedding_channels, embedding_channels)
        self.g = nn.Linear(embedding_channels, embedding_channels)
    def forward(self, z, z_mask_attention_float, z_mask):
        """
        Parameters
        ----------
        z: torch.Tensor of shape [batch, n_protein, n_compound, embedding_channels]
        z_mask: torch.Tensor of shape [batch*n_protein*num_attention_heads, n_compound, n_compound] saying which coefficients
            correspond to actual data. (we take this weird shape because scaled_dot_product_attention
            requires it). We take it to be float("-inf") where we want to mask.
        Returns
        -------
        """
        z = self.layernorm(z)
        batch_size, n_protein, n_compound, embedding_channels = z.shape
        z = z.reshape(batch_size*n_protein, n_compound, embedding_channels)
        q, k, v = self.linear_qkv(z).chunk(3, dim=-1)
        q = q.view(batch_size*n_protein, n_compound, self.num_attention_heads, self.attention_head_size).contiguous()
        k = k.view(batch_size*n_protein, n_compound, self.num_attention_heads, self.attention_head_size).contiguous()
        v = v.view(batch_size*n_protein, n_compound, self.num_attention_heads, self.attention_head_size).contiguous()
        attention_coefficients = xops.memory_efficient_attention(query=q,
                                                key=k,
                                                value=v,
                                                attn_bias=z_mask_attention_float.to("cuda:0")) # shape [batch*protein_nodes, compound_nodes, n_heads, embedding//n_heads]        

        attention_output = attention_coefficients.view(batch_size, n_protein, n_compound, embedding_channels)
        g = self.g(z).sigmoid()
        output = g * attention_output.view(batch_size*n_protein, n_compound, embedding_channels)

        output = self.output_linear(output.view(batch_size, n_protein, n_compound, embedding_channels))*z_mask.unsqueeze(-1).to('cuda:0')
        return output

class TankBindTriangleSelfAttention(torch.nn.Module):
    # use the protein-compound matrix only.
    def __init__(self, embedding_channels, num_attention_heads=4):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = embedding_channels // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.layernorm = torch.nn.LayerNorm(embedding_channels)

        self.linear_q = nn.Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_k = nn.Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_v = nn.Linear(embedding_channels, self.all_head_size, bias=False)
        self.g = nn.Linear(embedding_channels, self.all_head_size)
        self.final_linear = nn.Linear(self.all_head_size, embedding_channels)

    def reshape_last_dim(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, z, z_mask_float, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        # z_mask of shape b, i, j

        z = self.layernorm(z)
   

        # q, k, v of shape b, j, h, c
        q = self.reshape_last_dim(self.linear_q(z))  #  * (self.attention_head_size**(-0.5))
        k = self.reshape_last_dim(self.linear_k(z))
        v = self.reshape_last_dim(self.linear_v(z))
        #import IPython; IPython.embed()
        logits = torch.einsum("biqhc,bikhc->bihqk", q, k)/self.attention_head_size**0.5
        logits += z_mask_float.view(logits.shape)
        weights = nn.Softmax(dim=-1)(logits)
        # weights of shape b, h, j, j
        # attention_probs = self.dp(attention_probs)
        weighted_avg = torch.einsum("bihqk,bikhc->biqhc", weights, v)
        g = self.reshape_last_dim(self.g(z)).sigmoid()
        output = g * weighted_avg
        new_output_shape = output.size()[:-2] + (self.all_head_size,)
        output = output.view(*new_output_shape)
        # output of shape b, j, embedding.
        # z[:, i] = output
        z = output
        # print(g.shape, block1.shape, block2.shape)
        #import IPython; IPython.embed()
        z = self.final_linear(z) * z_mask.float().unsqueeze(-1)
        return z


class TankBindTriangleProteinToCompound(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels, c):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels, bias=False)#BUG: in original code, this has a bias. It should not be the case, otherwise the bias can propagate from false entries that should be 0 and destroy our masking
        self.layernorm_c = torch.nn.LayerNorm(c, bias=False)

        self.gate_linear1 = nn.Linear(embedding_channels, c, bias=False) 
        self.gate_linear2 = nn.Linear(embedding_channels, c, bias=False)

        self.linear1 = nn.Linear(embedding_channels, c)
        self.linear2 = nn.Linear(embedding_channels, c)

        self.ending_gate_linear = nn.Linear(embedding_channels, embedding_channels)
        self.linear_after_sum = nn.Linear(c, embedding_channels)

    def forward(self, z, protein_pair, compound_pair, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.

        z = self.layernorm(z)
        protein_pair = self.layernorm(protein_pair)
        compound_pair = self.layernorm(compound_pair)

        ab1 = self.gate_linear1(z).sigmoid() * self.linear1(z) * z_mask
        ab2 = self.gate_linear2(z).sigmoid() * self.linear2(z) * z_mask
        # Weird parameter reuse
        protein_pair = self.gate_linear2(protein_pair).sigmoid() * self.linear2(
            protein_pair
        )
        compound_pair = self.gate_linear1(compound_pair).sigmoid() * self.linear1(
            compound_pair
        )
        g = self.ending_gate_linear(z).sigmoid()
        block1 = torch.einsum("bikc,bkjc->bijc", protein_pair, ab1)
        block2 = torch.einsum("bikc,bjkc->bijc", ab2, compound_pair)
        z = g * self.linear_after_sum(self.layernorm_c(block1 + block2)) * z_mask
        return z



class AtomEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        self.linear = nn.Linear(emb_dim, emb_dim)
    def forward(self, x):
        return self.linear(x)
    
class BondEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        self.linear = nn.Linear(emb_dim, emb_dim)
    def forward(self, x):
        return self.linear(x)

class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, drop_ratio = 0.5, JK = "last", residual = True, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr,):
        ### computing input node embedding
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]

        return node_representation

class GINE(torch.nn.Module):

    def __init__(self, num_layer = 5, emb_dim = 128, 
                    gnn_type = 'gin', residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):


        super(GINE, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.graph_pooling = graph_pooling
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


    def forward(self, x, edge_index, edge_attr,):
        h_node = self.gnn_node(x, edge_index, edge_attr,)
        return h_node

"""
class GINE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        edge_input_dim,
        num_mlp_layer=2,
        eps=0,
        learn_eps=False,
    ):
        super().__init__()
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]
        self.dims = [input_dim] + list(hidden_dims)
        self.layers = nn.ModuleList()

        for i in range(len(self.dims) - 1):
            layer_hidden_dims = [self.dims[i + 1]] * (num_mlp_layer - 1)
            layer_dims = [self.dims[i]] + layer_hidden_dims + [self.dims[i + 1]]
            mlp = pyg.nn.MLP(channel_list=layer_dims)
            self.layers.append(
                GINEConv(nn=mlp, eps=eps, train_eps=learn_eps, edge_dim=edge_input_dim)
            )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
    ):
        for layer in self.layers:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return x
"""

# region tankbindproteinembedding
def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out





class GVP(nn.Module):
    """
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.

    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(
        self,
        in_dims,
        out_dims,
        h_dim=None,
        activations=(F.relu, torch.sigmoid),
        vector_gate=False,
    ):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate:
                    self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        """
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(_norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3, device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)

        return (s, v) if self.vo else s


def tuple_index(x, idx):
    """
    Indexes into a tuple (s, V) along the first dimension.

    :param idx: any object which can be used to index into a `torch.Tensor`
    """
    return x[0][idx], x[1][idx]


def _merge(s, v):
    """
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    """
    v = torch.reshape(v, v.shape[:-2] + (3 * v.shape[-2],))
    return torch.cat([s, v], -1)


def _split(x, nv):
    """
    Splits a merged representation of (s, V) back into a tuple.
    Should be used only with `_merge(s, V)` and only if the tuple
    representation cannot be used.

    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    """
    v = torch.reshape(x[..., -3 * nv :], x.shape[:-1] + (nv, 3))
    s = x[..., : -3 * nv]
    return s, v


def tuple_sum(*args):
    """
    Sums any number of tuples (s, V) elementwise.
    """
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim=-1):
    """
    Concatenates any number of tuples (s, V) elementwise.

    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    """
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


class LayerNorm(nn.Module):
    """
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        """
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


from torch_geometric.nn import MessagePassing


class GVPConv(MessagePassing):
    """
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.

    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.

    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(
        self,
        in_dims,
        out_dims,
        edge_dims,
        n_layers=3,
        module_list=None,
        aggr="mean",
        activations=(F.relu, torch.sigmoid),
        vector_gate=False,
    ):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims

        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)

        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_(
                        (2 * self.si + self.se, 2 * self.vi + self.ve),
                        (self.so, self.vo),
                        activations=(None, None),
                    )
                )
            else:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims, activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        """
        x_s, x_v = x
        message = self.propagate(
            edge_index,
            s=x_s,
            v=x_v.reshape(x_v.shape[0], 3 * x_v.shape[1]),
            edge_attr=edge_attr,
        )
        return _split(message, self.vo)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)


class _VDropout(nn.Module):
    """
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    """

    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class Dropout(nn.Module):
    """
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        """
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


class GVPConvLayer(nn.Module):
    """
    Full graph convolution / message passing layer with
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward
    network to node embeddings, and returns updated node embeddings.

    To only compute the aggregated messages, see `GVPConv`.

    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(
        self,
        node_dims,
        edge_dims,
        n_message=3,
        n_feedforward=2,
        drop_rate=0.1,
        autoregressive=False,
        activations=(F.relu, torch.sigmoid),
        vector_gate=False,
    ):
        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(
            node_dims,
            node_dims,
            edge_dims,
            n_message,
            aggr="add" if autoregressive else "mean",
            activations=activations,
            vector_gate=vector_gate,
        )
        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4 * node_dims[0], 2 * node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward - 2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr, autoregressive_x=None, node_mask=None):
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`.
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The corrent node
                embeddings `x` will still be the base of the update and the
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        """

        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)

            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward),
            )

            count = (
                scatter_add(torch.ones_like(dst), dst, dim_size=dh[0].size(0))
                .clamp(min=1)
                .unsqueeze(-1)
            )

            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            dh = self.conv(x, edge_index, edge_attr)

        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))

        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))

        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x


class TankBindProteinEmbedding(nn.Module):
    """
    Modified based on https://github.com/drorlab/gvp-pytorch/blob/main/gvp/models.py
    GVP-GNN for Model Quality Assessment as described in manuscript.

    Takes in protein structure graphs of type `torch_geometric.data.Data`
    or `torch_geometric.data.Batch` and returns a scalar score for
    each graph in the batch in a `torch.Tensor` of shape [n_nodes]

    Should be used with `gvp.data.ProteinGraphDataset`, or with generators
    of `torch_geometric.data.Batch` objects with the same attributes.

    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    """

    def __init__(
        self,
        node_in_dim,
        node_h_dim,
        edge_in_dim,
        edge_h_dim,
        seq_in=True,
        num_layers=3,
        drop_rate=0.1,
    ):
        super().__init__()

        if seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim = (node_in_dim[0] + 20, node_in_dim[1])

        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None)),
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None)),
        )

        self.layers = nn.ModuleList(
            GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers)
        )

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(LayerNorm(node_h_dim), GVP(node_h_dim, (ns, 0)))

    def forward(self, h_V, edge_index, h_E, seq):
        """
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        """
        seq = self.W_s(seq)
        h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)

        return out


# endregion tankbindproteinembedding


class TankBindGLULinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.glu = nn.GLU(dim=-1)
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.glu(torch.cat((self.linear1(x), self.linear2(x)), dim=-1))


class TankBindUnembedding(nn.Module):
    def __init__(self, embedding_channels, distogram_bins=None):
        super().__init__()
        self.glu = TankBindGLULinear(embedding_channels, embedding_channels)
        self.linear = nn.Linear(embedding_channels, 1)
        self.bias = nn.Parameter(torch.tensor(0.27))
        self.batch_norm = nn.BatchNorm1d(embedding_channels, )
        self.energy_linear = nn.Linear(embedding_channels, 1)
        if distogram_bins is not None:
            self.distogram_bins = distogram_bins
            self.histogram_linear = nn.Linear(embedding_channels, distogram_bins)
    def forward(self, z, z_mask, z_mask_flat):
        y_pred = self.linear(z).flatten()[z_mask_flat]
        y_pred = 10*nn.functional.tanh(y_pred)+self.bias

        z_glu = self.glu(z)*z_mask.unsqueeze(-1)
        z_glu = self.batch_norm(z_glu.view(-1, z_glu.shape[-1])).view(z_glu.shape)
        z_glu = F.relu(z_glu)*z_mask.unsqueeze(-1)
        z_glu = self.energy_linear(z_glu).squeeze(-1) * z_mask
        z_glu = z_glu.sum(dim=(-1, -2))/100
        affinity_pred = 5*torch.tanh(z_glu)
        if hasattr(self, "histogram_linear"):
            histogram_pred = self.histogram_linear(z).view(-1, self.distogram_bins)[z_mask_flat]
            return y_pred, affinity_pred, histogram_pred
        return y_pred, affinity_pred

class NoSigmoidUnembedding(nn.Module):
    def __init__(self, embedding_channels, c):
        super().__init__()
        self.glu = TankBindGLULinear(embedding_channels, c)
        self.linear = nn.Linear(embedding_channels, 1)
    def forward(self, z, z_mask, z_mask_flat):
        y_pred = self.linear(z).flatten()[z_mask_flat]
        pair_energy = self.glu(z).squeeze(-1) * z_mask
        affinity_pred = 10*torch.tanh(pair_energy.sum(axis=(-1, -2)))
        return y_pred, affinity_pred

class TankBindTrigonometry(nn.Module):
    def __init__(self, embedding_channels, c, dropout=0.25, fast_attention=False):
        super().__init__()
        self.protein_to_compound = TankBindTriangleProteinToCompound(
            embedding_channels=embedding_channels, c=c
        )
        if fast_attention:
            self.triangle_self_attention = FastTriangleSelfAttention(
                embedding_channels=embedding_channels, num_attention_heads=4
            )
        else:
            self.triangle_self_attention = TankBindTriangleSelfAttention(
                embedding_channels=embedding_channels
            )

        self.transition = TankBindTransition(
            embedding_channels=embedding_channels, transition_expansion_factor=4
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, protein_pair, compound_pair, z_mask, z_mask_attention_float):
        z = z + self.dropout(
            self.protein_to_compound(
                z, protein_pair, compound_pair, z_mask.unsqueeze(-1)
            )
        )
        #if torch.isnan(z).any():
        #    logger.info(f"[Beginning Batch - trigonometry output] NaNs in protein_to_compound")
        #z = z + self.dropout(torch.nan_to_num(self.triangle_self_attention(z, z_mask_attention_float),nan=0))
        z = z + self.dropout(self.triangle_self_attention(z, z_mask_attention_float, z_mask))
        z = self.transition(z)
        return z
