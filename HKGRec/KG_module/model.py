import torch, math, itertools, os, psutil
from torch.nn import functional as F, Parameter
from torch.autograd import Variable
from itertools import permutations, product

from torch.nn.init import xavier_normal_, xavier_uniform_, uniform_, zeros_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

'''将输入的张量进行截断正态分布初始化'''
def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

class KG(torch.nn.Module):

    def __init__(self, n_rel_keys, num_values, embedding_size, num_filters=200):
        super(KG, self).__init__()
        self.embedding_size = embedding_size
        self.num_filters = num_filters

        self.f_FCN_net = torch.nn.Linear(num_filters*(embedding_size-2), 1)
        xavier_normal_(self.f_FCN_net.weight.data)
        zeros_(self.f_FCN_net.bias.data)

        self.emb_relations_keys = torch.nn.Embedding(n_rel_keys, self.embedding_size, padding_idx=0)
        self.emb_entities_values = torch.nn.Embedding(num_values, self.embedding_size, padding_idx=0)
        self.conv1 = torch.nn.Conv2d(1, num_filters, (3, 3))
        zeros_(self.conv1.bias.data)
        self.batchNorm1 = torch.nn.BatchNorm2d(num_filters, momentum=0.1)
        truncated_normal_(self.conv1.weight, mean=0.0, std=0.1)
        self.conv2 = torch.nn.Conv2d(1, num_filters, (5, 3))
        zeros_(self.conv2.bias.data)
        self.batchNorm2 = torch.nn.BatchNorm2d(num_filters, momentum=0.1)
        truncated_normal_(self.conv2.weight, mean=0.0, std=0.1)

        self.loss = torch.nn.Softplus()

    def init(self):
        bound = math.sqrt(1.0/self.embedding_size)
        uniform_(self.emb_relations_keys.weight.data, -bound, bound)
        uniform_(self.emb_entities_values.weight.data, -bound, bound)

    def forward(self, x_batch, arity, mode, device=None, id2rel_key=None, id2entity_value=None):
        relation_key_id = torch.LongTensor(np.array(x_batch[:, 0::2][:, 0:2]).flatten())
        head_tail_value_id = torch.LongTensor(np.array(x_batch[:, 1::2][:, 0:2]).flatten())

        relation_emb = self.emb_relations_keys(relation_key_id)
        relation_embedding = relation_emb.view(len(x_batch), 2, self.embedding_size)
        head_tail_emb = self.emb_entities_values(head_tail_value_id)
        head_tail_embedding = head_tail_emb.view(len(x_batch), 2, self.embedding_size)
        hrt_embedding = torch.cat((head_tail_embedding[:, 0, :].unsqueeze(1), relation_embedding[:, 0, :].unsqueeze(1), head_tail_embedding[:, 1, :].unsqueeze(1)), 1).unsqueeze(1)
        hrt_vector = self.conv1(hrt_embedding)
        hrt_vector = self.batchNorm1(hrt_vector)
        hrt_vector = F.relu(hrt_vector).squeeze(3)
        all_vector = hrt_vector.view(hrt_vector.size(0), -1).unsqueeze(2)

        if arity > 2:
            kv_key_id = torch.LongTensor(np.array(x_batch[:, 0::2][:, 2:]).flatten())
            kv_value_id = torch.LongTensor(np.array(x_batch[:, 1::2][:, 2:]).flatten())
            kv_key_emb = self.emb_relations_keys(kv_key_id)
            kv_key_embedding = kv_key_emb.view(len(x_batch), arity-2, self.embedding_size)
            kv_value_emb = self.emb_entities_values(kv_value_id)
            kv_value_embedding = kv_value_emb.view(len(x_batch), arity-2, self.embedding_size)
            hrt_embedding = torch.cat((head_tail_embedding[:, 0, :].unsqueeze(1), relation_embedding[:, 0, :].unsqueeze(1), head_tail_embedding[:, 1, :].unsqueeze(1)), 1)
            hrt_kv_embedding = torch.cat((hrt_embedding, kv_key_embedding[:, 0, :].unsqueeze(1), kv_value_embedding[:, 0, :].unsqueeze(1)), 1).unsqueeze(1)
            hrt_kv_vector = self.conv2(hrt_kv_embedding)
            hrt_kv_vector = self.batchNorm2(hrt_kv_vector)
            hrt_kv_vector = F.relu(hrt_kv_vector).squeeze(3)
            hrt_kv_vector = hrt_kv_vector.view(hrt_kv_vector.size(0), -1).unsqueeze(2)
            fact_vector = torch.cat((all_vector, hrt_kv_vector), 2)

            for i in range(arity-3):
                hrt_nkv_embedding = torch.cat((hrt_embedding, kv_key_embedding[:, i+1, :].unsqueeze(1), kv_value_embedding[:, i+1, :].unsqueeze(1)), 1).unsqueeze(1)
                hrt_nkv_vector = self.conv2(hrt_nkv_embedding)
                hrt_nkv_vector = self.batchNorm2(hrt_nkv_vector)
                hrt_nkv_vector = F.relu(hrt_nkv_vector).squeeze(3)
                hrt_nkv_vector = hrt_nkv_vector.view(hrt_nkv_vector.size(0), -1).unsqueeze(2)
                all_vector = torch.cat((fact_vector, hrt_nkv_vector), 2)

        min_val, _ = torch.min(all_vector, 2)
        evaluation_score = self.f_FCN_net(min_val)

        length = int(1/2 * len(head_tail_value_id))
        head_tail_value_list = head_tail_value_id[:length].tolist()
        head_tail_list = head_tail_emb[:length].tolist()

        if arity > 2:
            kv_value_list = kv_value_id[:length].tolist()
            entity_id_list = head_tail_value_list + kv_value_list
            kv_value_list = kv_value_emb[:length].tolist()
            entity_emb_list = head_tail_list + kv_value_list
        else:
            entity_id_list = head_tail_value_list
            entity_emb_list = head_tail_list
        trained_emb = dict(zip(entity_id_list, entity_emb_list))



        return evaluation_score, trained_emb
