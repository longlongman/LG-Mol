# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params
from .transformer_encoder_with_pair import TransformerEncoderWithPair
from typing import Dict, Any, List
from .models.encoder_shape import EGNN_VN_Encoder_point_cloud
from .models.GNN import GCN

import torch_geometric
import lmdb
import pickle
import numpy as np
from torch_geometric.nn import knn
logger = logging.getLogger(__name__)


@register_model("shape4classify")
class Shape4classify(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--emb-dropout",
            type=float,
            metavar="D",
            help="dropout probability for embeddings",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-seq-len", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--post-ln", type=bool, help="use post layernorm or pre layernorm"
        )
        parser.add_argument(
            "--masked-token-loss",
            type=float,
            metavar="D",
            help="mask loss ratio",
        )
        parser.add_argument(
            "--masked-dist-loss",
            type=float,
            metavar="D",
            help="masked distance loss ratio",
        )
        parser.add_argument(
            "--masked-coord-loss",
            type=float,
            metavar="D",
            help="masked coord loss ratio",
        )
        parser.add_argument(
            "--x-norm-loss",
            type=float,
            metavar="D",
            help="x norm loss ratio",
        )
        parser.add_argument(
            "--delta-pair-repr-norm-loss",
            type=float,
            metavar="D",
            help="delta encoder pair repr norm loss ratio",
        )
        parser.add_argument(
            "--masked-coord-dist-loss",
            type=float,
            metavar="D",
            help="masked coord dist loss ratio",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "infer"],
        )

    def __init__(self, args, dictionary):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.padding_idx = dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(dictionary), args.encoder_embed_dim, self.padding_idx
        )
        self._num_updates = None
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=args.encoder_layers,
            embed_dim=args.encoder_embed_dim,
            ffn_embed_dim=args.encoder_ffn_embed_dim,
            attention_heads=args.encoder_attention_heads,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_seq_len,
            activation_fn=args.activation_fn,
            no_final_head_layer_norm=args.delta_pair_repr_norm_loss < 0,
        )
        if args.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=len(dictionary),
                activation_fn=args.activation_fn,
                weight=None,
            )

        K = 128
        n_edge_type = len(dictionary) * len(dictionary)
        self.gbf_proj = NonLinearHead(
            K, args.encoder_attention_heads, args.activation_fn
        )
        self.gbf = GaussianLayer(K, n_edge_type)

        if args.masked_coord_loss > 0:
            self.pair2coord_proj = NonLinearHead(
                args.encoder_attention_heads, 1, args.activation_fn
            )
        if args.masked_dist_loss > 0:
            self.dist_head = DistanceHead(
                args.encoder_attention_heads, args.activation_fn
            )
        self.classification_heads = nn.ModuleDict()
        self.cloudNet = nn.ModuleDict()

        self.apply(init_bert_params)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args, task.dictionary)

    def forward(
        self,
        src_tokens,
        src_distance,
        src_coord,
        src_edge_type,
        src_cloud,
        encoder_masked_tokens=None,
        features_only=False,
        classification_head_name=None,
        couldNet_name=None,
        **kwargs
    ):

        if classification_head_name is not None:
            features_only = True

        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            encoder_pair_rep,
            delta_encoder_pair_rep,
            x_norm,
            delta_encoder_pair_rep_norm,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

        encoder_distance = None
        encoder_coord = None

        if not features_only:
            if self.args.masked_token_loss > 0:
                logits = self.lm_head(encoder_rep, encoder_masked_tokens)
            if self.args.masked_coord_loss > 0:
                coords_emb = src_coord
                if padding_mask is not None:
                    atom_num = (torch.sum(1 - padding_mask.type_as(x), dim=1) - 1).view(
                        -1, 1, 1, 1
                    )
                else:
                    atom_num = src_coord.shape[1] - 1
                delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                attn_probs = self.pair2coord_proj(delta_encoder_pair_rep)
                coord_update = delta_pos / atom_num * attn_probs
                coord_update = torch.sum(coord_update, dim=2)
                encoder_coord = coords_emb + coord_update
            if self.args.masked_dist_loss > 0:
                encoder_distance = self.dist_head(encoder_pair_rep)


        Z_equivariants, Z_invariants = None, None
        x_batch, edge_index_batch, edge_attr_batch,pos_batch, cloud_batch, cloud_index_batch, cloud_indices_batch = [], [], [], [], [], [], []
        for i in range(src_cloud.shape[0]):
            
            esp_shape = src_cloud[i]
            column_means = torch.nanmean(esp_shape, dim=0)
            nan_locations = torch.isnan(esp_shape)
            esp_shape[nan_locations] = column_means.repeat(esp_shape.shape[0], 1)[nan_locations]


            esp_node_features = torch.zeros((esp_shape.shape[0], 1), dtype=torch.float32)
            esp_node_features[:, :] = esp_shape[:, -1].unsqueeze(dim=-1)

            # esp_node_features = torch.zeros_like(esp_node_features) # ablation esp [w/o esp]
            # esp_shape = torch.zeros_like(esp_shape) # ablation shapes [w/o shape]
            x = esp_shape[:, :-1].float()

            batch_x = torch.zeros(x.shape[0], dtype=torch.long)
            # esp_edge_index = knn(x, x, 2 + 1, batch_x, batch_x) + (i * x.shape[0])
            # esp_edge_index = knn(x, x, 5 + 1, batch_x, batch_x) + (i * x.shape[0])
            # esp_edge_index = knn(x, x, 8 + 1, batch_x, batch_x) + (i * x.shape[0])
            esp_edge_index = knn(x, x, 10 + 1, batch_x, batch_x) + (i * x.shape[0])
            esp_edge_features = torch.zeros((esp_edge_index.shape[1], 5), dtype=torch.float32)
            esp_positions = esp_shape[:, :-1]

            esp_positions = esp_positions - (torch.sum(esp_positions, dim=0) / esp_positions.shape[0])
            esp_cloud = esp_positions
            esp_cloud_batch_indices = torch.arange(0, esp_cloud.shape[0])
            esp_atom_fragment_associations = torch.zeros((esp_node_features.shape[0]), dtype=torch.int)

            data_x = esp_node_features
            data_edge_index = esp_edge_index
            data_edge_attr = esp_edge_features
            data_pos = esp_positions
            data_cloud = esp_cloud 
            data_cloud_index = esp_cloud_batch_indices
            data_cloud_indices = esp_cloud_batch_indices
            

            '''
            data_batch = torch.zeros(data_x.size(0), dtype=torch.int64)
            device = "cuda:0"
            data_x = data_x.to(device)
            data_edge_index = data_edge_index.to(device)
            data_batch = data_batch.to(device)
            tensor_mean = torch.nanmean(data_x)
            data_x[torch.isnan(data_x)] = tensor_mean
            
            Z_invariant = self.cloudNet(
                data_x,
                data_edge_index,
                data_batch,
            )
            Z_invariant = Z_invariant

            if Z_invariants is None:
                Z_invariants = Z_invariant
            else:
                Z_invariants = torch.cat([Z_invariants, Z_invariant], dim=0)
            '''
        
        # mini-batch
            x_batch.append(data_x)
            edge_index_batch.append(data_edge_index)
            edge_attr_batch.append(data_edge_attr)
            pos_batch.append(data_pos)
            cloud_batch.append(data_cloud)
            cloud_index_batch.append(data_cloud_index)
            cloud_indices_batch.append(data_cloud_indices)
        
        x_batch = torch.concat(x_batch, dim=0)
        edge_index_batch = torch.concat(edge_index_batch, dim=-1)
        edge_attr_batch = torch.concat(edge_attr_batch, dim=0)
        pos_batch = torch.concat(pos_batch, dim=0)
        cloud_batch = torch.concat(cloud_batch, dim=0)
        cloud_index_batch = torch.concat(cloud_index_batch, dim=0)
        cloud_indices_batch = torch.concat(cloud_indices_batch, dim=0)
        
        # --- debug ---
        # x_batch = torch.zeros_like(x_batch)
        # pos_batch = torch.zeros_like(pos_batch)
        # cloud_batch = torch.zeros_like(cloud_batch)
        
        # x_batch = x_batch * 1e-12
        # pos_batch = pos_batch * 1e-12
        # cloud_batch = cloud_batch * 1e-12

        # x_batch = x_batch - x_batch
        # pos_batch = pos_batch - pos_batch
        # cloud_batch = cloud_batch - cloud_batch

        
        # origin-EGNN
        cloud_features = self.cloudNet(
            # h = torch.cat((data_x, torch.zeros((data_x.shape[0], 64))), dim = 1),
            h = x_batch,
            edge_index = edge_index_batch, 
            edge_attr = edge_attr_batch,
            pos = pos_batch,
            points = cloud_batch,
            points_atom_index = cloud_indices_batch,
            batch_size = src_cloud.shape[0],
            select_indices = None,
            select_indices_batch = None,
            use_variational_GNN = False, 
            variational_GNN_factor = 1.0, 
            interpolate_to_GNN_prior = 1.0,
            h_interpolate = None,
        )

        Z_equivariants, Z_invariants = cloud_features[1], cloud_features[2]
        

        # if Z_equivariants is None:
        #     Z_equivariants = Z_equivariant
        # else:
        #     Z_equivariants = torch.cat([Z_equivariants, Z_equivariant], dim=0)

        # if Z_invariants is None:
        #     Z_invariants = Z_invariant
        # else:
        #     Z_invariants = torch.cat([Z_invariants, Z_invariant], dim=0)
        # if(torch.any(torch.isnan(Z_invariants))):
        #     breakpoint()
        if classification_head_name is not None:
            # Z_invariants = torch.zeros_like(Z_invariants) # without ESP
            # encoder_rep = torch.zeros_like(encoder_rep) # without atomic
            # breakpoint()
            logits = self.classification_heads[classification_head_name](encoder_rep, Z_invariants)
        if self.args.mode == 'infer':
            return encoder_rep, encoder_pair_rep
        else:
            return (
                logits,
                encoder_distance,
                encoder_coord,
                x_norm,
                delta_encoder_pair_rep_norm,
            )         

    def get_cloud(self, idx, lmdb_path):
        env_mol = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        mol_pickled = env_mol.begin().get(f"{idx}".encode("ascii"))
        esp_shape = pickle.loads(mol_pickled)['charge']

        esp_node_features = np.zeros((esp_shape.shape[0], 45), dtype=np.float32)
        esp_node_features[:, :] = np.expand_dims(esp_shape[:, -1], axis=-1)

        x = torch.from_numpy(esp_shape[:, :-1])
        batch_x = torch.zeros(x.shape[0], dtype=torch.long)
        esp_edge_index = knn(x, x, 5 + 1, batch_x, batch_x)
        esp_edge_index = esp_edge_index.numpy()
        esp_edge_features = np.zeros((esp_edge_index.shape[1], 5), dtype=np.float32)
        esp_positions = esp_shape[:, :-1]

        esp_positions = esp_positions - (np.sum(esp_positions, axis = 0) / esp_positions.shape[0])
        esp_cloud = esp_positions
        esp_cloud_batch_indices = np.arange(0, esp_cloud.shape[0])
        esp_atom_fragment_associations = np.zeros((esp_node_features.shape[0]), dtype=np.int)

        data = torch_geometric.data.Data(
            x = torch.from_numpy(esp_node_features),
            edge_index = torch.from_numpy(esp_edge_index).type(torch.long),
            edge_attr = torch.from_numpy(esp_edge_features),
            pos = torch.from_numpy(esp_positions),
            cloud = torch.from_numpy(esp_cloud),
            cloud_index = torch.from_numpy(esp_cloud_batch_indices).type(torch.long),
            cloud_indices = torch.from_numpy(esp_cloud_batch_indices).type(torch.long),
            atom_fragment_associations = torch.from_numpy(esp_atom_fragment_associations).type(torch.long),
        )
        return data

    def register_cloud_feature_gnn(self, name):
        self.cloudNet = GCN(
            hidden_channels = 64
        )

    def register_cloud_feature(self, name):

        input_nf = 45,
        edges_in_d = 5,
        n_knn = 5, 
        conv_dims = [32, 32, 64, 128], 
        num_components = 64, 
        fragment_library_dim = 64, 
        N_fragment_layers = 3, 
        append_noise = False, 
        N_members = 125 - 1, 
        EGNN_layer_dim = 64, 
        N_EGNN_layers = 4, 
        output_MLP_hidden_dim = 64, 
        pooling_MLP = False, 
        shared_encoders = False, 
        subtract_latent_space = True,
        variational = False,
        variational_mode = 'inv', # not used
        # variational_GNN = True,
        variational_GNN = False,
        
        ###
        mix_node_inv_to_equi = True,
        mix_shape_to_nodes = True,
        ###
        ablate_HvarCat = False,
        
        predict_pairwise_properties = False,
        predict_mol_property = False,
    
        
        old_EGNN = False,

        self.cloudNet = EGNN_VN_Encoder_point_cloud(
            node_input_dim = 1,
            edges_in_d = 5, 
            # EGNN_layer_dim = 16, 
            # EGNN_layer_dim = 32, 
            EGNN_layer_dim = 64,
            # EGNN_layer_dim = 128, 
            n_knn = 5, 
            conv_dims = [32, 32, 64, 128], 
            pooling_MLP = pooling_MLP,
            N_EGNN_layers = 1,
            # N_EGNN_layers = 2,
            # N_EGNN_layers = 3,
            # N_EGNN_layers = 4,
            variational_GNN = False,
            variational_GNN_mol = False,
            mix_node_inv_to_equi = True,
            mix_shape_to_nodes = True,
            ablate_HvarCat = False,
            
            old_EGNN = False,
        )

    def register_mergeclassification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = MergeClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class MaskLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x

class MergeClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        # self.cloud_dense = nn.Linear(16, inner_dim)
        # self.cloud_dense = nn.Linear(32, inner_dim)
        self.cloud_dense = nn.Linear(64, inner_dim)
        # self.cloud_dense = nn.Linear(128, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(2*inner_dim, num_classes)

    def forward(self, features, cloud_features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        y = self.cloud_dense(cloud_features)
        merge_features = torch.cat([x, y], dim=1) 
        x = self.activation_fn(merge_features)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class NonLinearHead(nn.Module):
    """Head for simple classification tasks."""

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class DistanceHead(nn.Module):
    def __init__(
        self,
        heads,
        activation_fn,
    ):
        super().__init__()
        self.dense = nn.Linear(heads, heads)
        self.layer_norm = nn.LayerNorm(heads)
        self.out_proj = nn.Linear(heads, 1)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def forward(self, x):
        bsz, seq_len, seq_len, _ = x.size()
        # x[x == float('-inf')] = 0
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.out_proj(x).view(bsz, seq_len, seq_len)
        x = (x + x.transpose(-1, -2)) * 0.5
        return x


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=1024):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        mul = self.mul(edge_type).type_as(x) # [batch_size, atom_num, atom_num, emb_dim]
        bias = self.bias(edge_type).type_as(x) # [batch_size, atom_num, atom_num, emb_dim]
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


@register_model_architecture("shape4classify", "shape4classify_init")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)


@register_model_architecture("shape4classify", "shape4classify")
def unimol_base_architecture(args):
    base_architecture(args)
