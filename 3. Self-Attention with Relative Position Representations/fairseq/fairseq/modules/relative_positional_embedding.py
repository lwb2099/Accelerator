# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from torch import Tensor


class RelativePositionalEmbedding(nn.Module):
    """
    relative positional embeddings up to a fixed maximum size.
    """

    def __init__(self, embedding_dim: int, max_relative_position = None):
        super().__init__()
        self.max_relative_position = max_relative_position
        self.num_embeddings = 2*max_relative_position + 1
        self.embedding_dim = embedding_dim
        #* [2*window_size, d_model]
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, embedding_dim))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(
        self,
        src_len: int = None,
        tgt_len: int = None,
    ):
        range_vec_q = torch.arange(tgt_len)
        range_vec_k = torch.arange(src_len)
        #* [tgt_len, src_len]
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]
        return embeddings
