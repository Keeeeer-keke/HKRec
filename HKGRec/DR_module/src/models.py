import torch
# from torch._C import contiguous_format
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dnc import DNC
from layers import GraphConvolution, SelfAttend
import math
from torch.nn.parameter import Parameter
import dill
import pandas as pd

'''
Our model
'''
class HKGRec(nn.Module):
    def __init__(self, vocab_size, ehr_adj, ddi_adj, ICD_tensor_dict, ATC_tensor_dict, emb_dim=64, device=torch.device('cpu'), ddi_in_memory=True):

        super(HKGRec, self).__init__()
        self.vocab_size = vocab_size 
        self.emb_dim = emb_dim
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.nhead = 2
        self.ddi_in_memory = ddi_in_memory
        K = len(vocab_size)
        self.K = K
        self.ICD_tensor_dict = ICD_tensor_dict
        self.ATC_tensor_dict = ATC_tensor_dict

        # 初始embedding，dig_num * emb_dim
        self.diag_embedding = nn.Sequential(
            nn.Embedding(vocab_size[0], emb_dim),
            nn.Dropout(0.3)
        )

        # proc_num * emb_dim
        self.proc_embedding = nn.Sequential(
            nn.Embedding(vocab_size[1], emb_dim),
            nn.Dropout(0.3)
        )

        # med_num * emb_dim
        self.med_embedding = nn.Sequential(
            nn.Embedding(vocab_size[2], emb_dim),
            nn.Dropout(0.3)
        )

        # 用于visit级别的信息进行编码
        self.diagnoses_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, batch_first=True, dropout=0.2)
        self.procedure_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, batch_first=True, dropout=0.2)
        self.medication_encoder = nn.TransformerEncoderLayer(emb_dim, self.nhead, batch_first=True, dropout=0.2)
        
        # 聚合单个visit内的diag和proc得到visit-level的表达
        self.diag_self_attend = SelfAttend(emb_dim)
        self.proc_self_attend = SelfAttend(emb_dim)
        self.med_self_attend = SelfAttend(emb_dim)
        self.encoders_f = nn.ModuleList([nn.LSTM(emb_dim, emb_dim, batch_first=True, bidirectional=True) for _ in range(K)])  # 双向LSTM
        self.encoders_o = nn.ModuleList([nn.LSTM(emb_dim, emb_dim, batch_first=True, bidirectional=True) for _ in range(K)])  # 双向LSTM
        self.ehr_gcn = GCN(vocab_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device)
        self.ddi_gcn = GCN(vocab_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device)
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2]),
        )

        self.fo = nn.Linear(emb_dim * 2, 1)  # 双向LSTM
        self.op = nn.Linear(emb_dim * 2, emb_dim)  # 双向LSTM


    def forward(self, input, ICD_tensor_dict, ATC_tensor_dict):
        '''input: [[[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2], [0]],
                   [[1, 3, 9, 10, 6, 5], [0], [0]]]'''
        diag_seq = []
        proc_seq = []
        med_seq = []

        for adm in input:
            '''adm1: [[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2], [0]]'''
            '''adm2: [[1, 3, 9, 10, 6, 5], [0], [0]]'''
            diseases = torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
            procedures = torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
            medications = torch.LongTensor(adm[2]).unsqueeze(dim=0).to(self.device)

            max_diag_num = diseases.size()[1]
            max_proc_num = procedures.size()[1]
            max_med_num = medications.size()[1]

            # 生成d_mask_matrix
            d_mask_matrix = torch.full((max_diag_num,), -1e9).to(self.device)
            for i in range(len(adm[0])):
                d_mask_matrix[i] = 0.

            # 生成p_mask_matrix
            p_mask_matrix = torch.full((max_proc_num,), -1e9).to(self.device)
            for i in range(len(adm[1])):
                p_mask_matrix[i] = 0.

            # 生成m_mask_matrix
            m_mask_matrix = torch.full((max_med_num,), -1e9).to(self.device)
            for i in range(len(adm[2])):
                m_mask_matrix[i] = 0.

            '''Diagnosis'''
            input_diag_embedding = self.diag_embedding(diseases)
            d_enc_mask_matrix = d_mask_matrix.unsqueeze(dim=1).repeat(1, self.nhead, max_diag_num, 1).view(self.nhead, max_diag_num, max_diag_num)
            diag_embedding = self.diagnoses_encoder(input_diag_embedding, src_mask=d_enc_mask_matrix)
            copy_diseases = diseases.cpu().numpy()
            # 将list里的id映射为向量
            kg_diag = list(map(ICD_tensor_dict.get, copy_diseases.flatten()))
            kg_diag_list = [torch.zeros(64) if ele is None else ele for ele in kg_diag]
            kg_diag_tensor = torch.stack(kg_diag_list).view(max_diag_num, self.emb_dim).to(self.device)
            diag_embedding = torch.add(kg_diag_tensor, diag_embedding)
            # diag_embedding = diag_embedding + F.sigmoid(self.W_d(kg_diag_tensor)).view(max_diag_num, 1) * kg_diag_tensor
            diag_embedding = self.diag_self_attend(diag_embedding, d_mask_matrix)

            '''Procedure'''
            input_proc_embedding = self.proc_embedding(procedures)
            p_enc_mask_matrix = p_mask_matrix.unsqueeze(dim=1).repeat(1, self.nhead, max_proc_num, 1).view(self.nhead, max_proc_num, max_proc_num)
            proc_embedding = self.procedure_encoder(input_proc_embedding, src_mask=p_enc_mask_matrix)
            proc_embedding = self.proc_self_attend(proc_embedding, p_mask_matrix)

            '''Medication'''
            input_med_embedding = self.med_embedding(medications)
            m_enc_mask_matrix = m_mask_matrix.unsqueeze(dim=1).repeat(1, self.nhead, max_med_num, 1).view(self.nhead, max_med_num, max_med_num)
            med_embedding = self.medication_encoder(input_med_embedding, src_mask=m_enc_mask_matrix)  # 与input_med_embedding的size一样
            copy_medications = medications.cpu().numpy()

            # 将list里的id映射为向量
            kg_med = list(map(ATC_tensor_dict.get, copy_medications.flatten()))
            kg_med_list = [torch.zeros(64) if ele is None else ele for ele in kg_med]
            kg_med_tensor = torch.stack(kg_med_list).view(max_med_num, self.emb_dim).to(self.device)
            med_embedding = torch.add(kg_med_tensor, med_embedding)
            # med_embedding = med_embedding + F.sigmoid(self.W_m(kg_med_tensor)).view(max_med_num, 1) * kg_med_tensor
            
            if self.ddi_in_memory:
                drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)-(14,64)
            else:
                drug_memory = self.ehr_gcn()
            input_medication_memory = drug_memory[medications]
            
            med_embedding = med_embedding + input_medication_memory
            med_embedding = self.med_self_attend(med_embedding, m_mask_matrix)

            diag_seq.append(diag_embedding)
            proc_seq.append(proc_embedding)
            med_seq.append(med_embedding)

        diag_seq = torch.cat(diag_seq, dim=1)  # (1,visit,dim)，一个患者的多次visit
        proc_seq = torch.cat(proc_seq, dim=1)  # (1,visit,dim)，一个患者的多次visit
        med_seq = torch.cat(med_seq, dim=1)  # (1,visit-1,dim)，一个患者的多次visit

        diag_f, _ = self.encoders_f[0](diag_seq)  # diag_f:(1, visit, dim)
        proc_f, _ = self.encoders_f[1](proc_seq)
        med_f, _ = self.encoders_f[2](med_seq)
        diag_o, _ = self.encoders_o[0](diag_seq)  # diag_o:(1, visit, dim)
        proc_o, _ = self.encoders_o[1](proc_seq)
        med_o, _ = self.encoders_o[2](med_seq)

        weight_diag_f = F.softmax(self.fo(diag_f.squeeze(dim=0)), dim=-1)  # (visit, 1)
        weight_proc_f = F.softmax(self.fo(proc_f.squeeze(dim=0)), dim=-1)  # (visit, 1)
        weight_med_f = F.softmax(self.fo(med_f.squeeze(dim=0)), dim=-1)  # (visit-1, 1)
        weight_diag_o = F.tanh(self.op(diag_o.squeeze(dim=0)))  # (visit, dim)
        weight_proc_o = F.tanh(self.op(proc_o.squeeze(dim=0)))  # (visit, dim)
        weight_med_o = F.tanh(self.op(med_o.squeeze(dim=0)))  # (visit-1, dim)

        diag = torch.sum(weight_diag_f * weight_diag_o * diag_seq, dim=1)  # (1, dim)
        proc = torch.sum(weight_proc_f * weight_proc_o * proc_seq, dim=1)  # (1, dim)
        med = torch.sum(weight_med_f * weight_med_o * med_seq, dim=1)  # (1, dim)


        output = self.output(torch.cat([diag, proc, med], dim=-1))

        if self.training:
            neg_pred_prob = F.sigmoid(output)  # (1, 14)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (vocab_size, vocab_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()  # tensor值
            return output, batch_neg
        else:
            return output
        return output


class GCN(nn.Module):
    def __init__(self, vocab_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(GCN, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.device = device

        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(vocab_size).to(device)

        self.gcn1 = GraphConvolution(vocab_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.device = device
        self.embed_fingerprint = nn.Embedding(
            N_fingerprint, dim).to(self.device)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim).to(self.device)
                                            for _ in range(layer_hidden)])
        self.layer_hidden = layer_hidden

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch proc essing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, inputs):
        """Cat or pad each input data for batch processing."""
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)

        """MPNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            # fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.
            fingerprint_vectors = hs

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors


class SafeDrugModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            ddi_adj,
            ddi_mask_H,
            MPNNSet,
            N_fingerprints,
            average_projection,
            emb_dim=256,
            device=torch.device("cuda:0"),
    ):
        super(SafeDrugModel, self).__init__()

        self.device = device

        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(2)]
        )
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)]
        )
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim))

        # bipartite local embedding
        self.bipartite_transform = nn.Sequential(
            nn.Linear(emb_dim, ddi_mask_H.shape[1])
        )
        self.bipartite_output = MaskLinear(ddi_mask_H.shape[1], vocab_size[2], False)

        # MPNN global embedding
        self.MPNN_molecule_Set = list(zip(*MPNNSet))

        self.MPNN_emb = MolecularGraphNeuralNetwork(
            N_fingerprints, emb_dim, layer_hidden=2, device=device
        ).forward(self.MPNN_molecule_Set)
        self.MPNN_emb = torch.mm(
            average_projection.to(device=self.device),
            self.MPNN_emb.to(device=self.device),
        )
        self.MPNN_emb.to(device=self.device)
        # self.MPNN_emb = torch.tensor(self.MPNN_emb, requires_grad=True)
        self.MPNN_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.MPNN_layernorm = nn.LayerNorm(vocab_size[2])

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_H = torch.FloatTensor(ddi_mask_H).to(device)
        self.init_weights()

    def forward(self, input):

        # patient health representation
        i1_seq = []
        i2_seq = []

        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:
            i1 = sum_embedding(
                self.dropout(
                    self.embeddings[0](
                        torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )  # (1,1,dim)
            i2 = sum_embedding(
                self.dropout(
                    self.embeddings[1](
                        torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )
            i1_seq.append(i1)
            i2_seq.append(i2)
        i1_seq = torch.cat(i1_seq, dim=1)  # (1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)

        o1, h1 = self.encoders[0](i1_seq)
        o2, h2 = self.encoders[1](i2_seq)
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(
            dim=0
        )  # (seq, dim*2)
        query = self.query(patient_representations)[-1:, :]  # (seq, dim)

        # MPNN embedding
        MPNN_match = F.sigmoid(torch.mm(query, self.MPNN_emb.t()))
        MPNN_att = self.MPNN_layernorm(MPNN_match + self.MPNN_output(MPNN_match))

        # local embedding
        bipartite_emb = self.bipartite_output(
            F.sigmoid(self.bipartite_transform(query)), self.tensor_ddi_mask_H.t()
        )

        result = torch.mul(bipartite_emb, MPNN_att)

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (vocab_size, vocab_size)

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)


class DMNC(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device("cpu:0")):
        super(DMNC, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device

        self.token_start = vocab_size[2]
        self.token_end = vocab_size[2] + 1

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(vocab_size[i] if i != 2 else vocab_size[2] + 2, emb_dim)
                for i in range(K)
            ]
        )
        self.dropout = nn.Dropout(p=0.5)

        self.encoders = nn.ModuleList(
            [
                DNC(
                    input_size=emb_dim,
                    hidden_size=emb_dim,
                    rnn_type="gru",
                    num_layers=1,
                    num_hidden_layers=1,
                    nr_cells=16,
                    cell_size=emb_dim,
                    read_heads=1,
                    batch_first=True,
                    gpu_id=0,
                    independent_linears=False,
                )
                for _ in range(K - 1)
            ]
        )

        self.decoder = nn.GRU(
            emb_dim + emb_dim * 2, emb_dim * 2, batch_first=True
        )  # input: (y, r1, r2,) hidden: (hidden1, hidden2)
        self.interface_weighting = nn.Linear(
            emb_dim * 2, 2 * (emb_dim + 1 + 3)
        )  # 2 read head (key, str, mode)
        self.decoder_r2o = nn.Linear(2 * emb_dim, emb_dim * 2)

        self.output = nn.Linear(emb_dim * 2, vocab_size[2] + 2)

    def forward(self, input, i1_state=None, i2_state=None, h_n=None, max_len=20):
        # input (3, code)
        i1_input_tensor = self.embeddings[0](
            torch.LongTensor(input[0]).unsqueeze(dim=0).to(self.device)
        )  # (1, seq, codes)
        i2_input_tensor = self.embeddings[1](
            torch.LongTensor(input[1]).unsqueeze(dim=0).to(self.device)
        )  # (1, seq, codes)

        o1, (ch1, m1, r1) = self.encoders[0](
            i1_input_tensor, (None, None, None) if i1_state is None else i1_state
        )
        o2, (ch2, m2, r2) = self.encoders[1](
            i2_input_tensor, (None, None, None) if i2_state is None else i2_state
        )

        # save memory state
        i1_state = (ch1, m1, r1)
        i2_state = (ch2, m2, r2)

        predict_sequence = [self.token_start] + input[2]
        if h_n is None:
            h_n = torch.cat([ch1[0], ch2[0]], dim=-1)

        output_logits = []
        r1 = r1.unsqueeze(dim=0)
        r2 = r2.unsqueeze(dim=0)

        if self.training:
            for item in predict_sequence:
                # teacher force predict drug
                item_tensor = self.embeddings[2](
                    torch.LongTensor([item]).unsqueeze(dim=0).to(self.device)
                )  # (1, seq, codes)

                o3, h_n = self.decoder(torch.cat([item_tensor, r1, r2], dim=-1), h_n)
                read_keys, read_strengths, read_modes = self.decode_read_variable(
                    h_n.squeeze(0)
                )

                # read from i1_mem, i2_mem and i3_mem
                r1, _ = self.read_from_memory(
                    self.encoders[0],
                    read_keys[:, 0, :].unsqueeze(dim=1),
                    read_strengths[:, 0].unsqueeze(dim=1),
                    read_modes[:, 0, :].unsqueeze(dim=1),
                    i1_state[1],
                )

                r2, _ = self.read_from_memory(
                    self.encoders[1],
                    read_keys[:, 1, :].unsqueeze(dim=1),
                    read_strengths[:, 1].unsqueeze(dim=1),
                    read_modes[:, 1, :].unsqueeze(dim=1),
                    i2_state[1],
                )

                output = self.decoder_r2o(torch.cat([r1, r2], dim=-1))
                output = self.output(output + o3).squeeze(dim=0)
                output_logits.append(output)
        else:
            item_tensor = self.embeddings[2](
                torch.LongTensor([self.token_start]).unsqueeze(dim=0).to(self.device)
            )  # (1, seq, codes)
            for idx in range(max_len):
                # predict
                # teacher force predict drug
                o3, h_n = self.decoder(torch.cat([item_tensor, r1, r2], dim=-1), h_n)
                read_keys, read_strengths, read_modes = self.decode_read_variable(
                    h_n.squeeze(0)
                )

                # read from i1_mem, i2_mem and i3_mem
                r1, _ = self.read_from_memory(
                    self.encoders[0],
                    read_keys[:, 0, :].unsqueeze(dim=1),
                    read_strengths[:, 0].unsqueeze(dim=1),
                    read_modes[:, 0, :].unsqueeze(dim=1),
                    i1_state[1],
                )

                r2, _ = self.read_from_memory(
                    self.encoders[1],
                    read_keys[:, 1, :].unsqueeze(dim=1),
                    read_strengths[:, 1].unsqueeze(dim=1),
                    read_modes[:, 1, :].unsqueeze(dim=1),
                    i2_state[1],
                )

                output = self.decoder_r2o(torch.cat([r1, r2], dim=-1))
                output = self.output(output + o3).squeeze(dim=0)
                output = F.softmax(output, dim=-1)
                output_logits.append(output)

                input_token = torch.argmax(output, dim=-1)
                input_token = input_token.item()
                item_tensor = self.embeddings[2](
                    torch.LongTensor([input_token]).unsqueeze(dim=0).to(self.device)
                )  # (1, seq, codes)

        return torch.cat(output_logits, dim=0), i1_state, i2_state, h_n

    def read_from_memory(self, dnc, read_key, read_str, read_mode, m_hidden):
        read_vectors, hidden = dnc.memories[0].read(
            read_key, read_str, read_mode, m_hidden
        )
        return read_vectors, hidden

    def decode_read_variable(self, input):
        w = 64
        r = 2
        b = input.size(0)

        input = self.interface_weighting(input)
        # r read keys (b * w * r)
        read_keys = F.tanh(input[:, : r * w].contiguous().view(b, r, w))
        # r read strengths (b * r)
        read_strengths = F.softplus(input[:, r * w: r * w + r].contiguous().view(b, r))
        # read modes (b * 3*r)
        read_modes = F.softmax(input[:, (r * w + r):].contiguous().view(b, r, 3), -1)
        return read_keys, read_strengths, read_modes


class GAMENet(nn.Module):
    def __init__(
            self,
            vocab_size,
            ehr_adj,
            ddi_adj,
            emb_dim=64,
            device=torch.device("cpu:0"),
            ddi_in_memory=True,
    ):
        super(GAMENet, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(K - 1)]
        )
        self.dropout = nn.Dropout(p=0.5)

        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim * 2, batch_first=True) for _ in range(K - 1)]
        )

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        self.ehr_gcn = GCN(
            vocab_size=vocab_size[2], emb_dim=emb_dim, adj=ehr_adj, device=device
        )
        self.ddi_gcn = GCN(
            vocab_size=vocab_size[2], emb_dim=emb_dim, adj=ddi_adj, device=device
        )
        self.inter = nn.Parameter(torch.FloatTensor(1))

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2]),
        )

        self.init_weights()

    def forward(self, input):
        # input (adm, 3, codes)

        # generate medical embeddings and queries
        i1_seq = []
        i2_seq = []

        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:
            i1 = mean_embedding(
                self.dropout(
                    self.embeddings[0](
                        torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )  # (1,1,dim)
            i2 = mean_embedding(
                self.dropout(
                    self.embeddings[1](
                        torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )
            i1_seq.append(i1)
            i2_seq.append(i2)
        i1_seq = torch.cat(i1_seq, dim=1)  # (1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)

        o1, h1 = self.encoders[0](i1_seq)  # o1:(1, seq, dim*2) hi:(1,1,dim*2)
        o2, h2 = self.encoders[1](i2_seq)
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(
            dim=0
        )  # (seq, dim*4)
        queries = self.query(patient_representations)  # (seq, dim)

        # graph memory module
        """I:generate current input"""
        query = queries[-1:]  # (1,dim)

        """G:generate graph memory bank and insert history information"""
        if self.ddi_in_memory:
            drug_memory = self.ehr_gcn() - self.ddi_gcn() * self.inter  # (size, dim)
        else:
            drug_memory = self.ehr_gcn()

        if len(input) > 1:
            history_keys = queries[: (queries.size(0) - 1)]  # (seq-1, dim)

            history_values = np.zeros((len(input) - 1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input) - 1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(
                self.device
            )  # (seq-1, size)

        """O:read from global memory bank and dynamic memory bank"""
        key_weights1 = F.softmax(torch.mm(query, drug_memory.t()), dim=-1)  # (1, size)
        fact1 = torch.mm(key_weights1, drug_memory)  # (1, dim)

        if len(input) > 1:
            visit_weight = F.softmax(torch.mm(query, history_keys.t()))  # (1, seq-1)
            weighted_values = visit_weight.mm(history_values)  # (1, size)
            fact2 = torch.mm(weighted_values, drug_memory)  # (1, dim)
        else:
            fact2 = fact1
        """R:convert O and predict"""
        output = self.output(torch.cat([query, fact1, fact2], dim=-1))  # (1, dim)

        if self.training:
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (vocab_size, vocab_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        self.inter.data.uniform_(-initrange, initrange)


class Leap(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, device=torch.device("cpu:0")):
        super(Leap, self).__init__()
        self.vocab_size = vocab_size
        self.device = device
        self.SOS_TOKEN = vocab_size[2]
        self.END_TOKEN = vocab_size[2] + 1

        self.enc_embedding = nn.Sequential(
            nn.Embedding(
                vocab_size[0],
                emb_dim,
            ),
            nn.Dropout(0.3),
        )
        self.dec_embedding = nn.Sequential(
            nn.Embedding(
                vocab_size[2] + 2,
                emb_dim,
            ),
            nn.Dropout(0.3),
        )

        self.dec_gru = nn.GRU(emb_dim * 2, emb_dim, batch_first=True)

        self.attn = nn.Linear(emb_dim * 2, 1)

        self.output = nn.Linear(emb_dim, vocab_size[2] + 2)

    def forward(self, input, max_len=20):
        device = self.device
        # input (3, codes)
        input_tensor = torch.LongTensor(input[0]).to(device)
        # (len, dim)
        input_embedding = self.enc_embedding(input_tensor.unsqueeze(dim=0)).squeeze(
            dim=0
        )

        output_logits = []
        hidden_state = None
        if self.training:
            for med_code in [self.SOS_TOKEN] + input[2]:
                dec_input = torch.LongTensor([med_code]).unsqueeze(dim=0).to(device)
                dec_input = self.dec_embedding(dec_input).squeeze(dim=0)  # (1,dim)

                if hidden_state is None:
                    hidden_state = dec_input
                hidden_state_repeat = hidden_state.repeat(
                    input_embedding.size(0), 1
                )  # (len, dim)

                combined_input = torch.cat(
                    [hidden_state_repeat, input_embedding], dim=-1
                )  # (len, dim*2)
                attn_weight = F.softmax(
                    self.attn(combined_input).t(), dim=-1
                )  # (1, len)
                input_embedding = attn_weight.mm(input_embedding)  # (1, dim)

                _, hidden_state = self.dec_gru(
                    torch.cat([input_embedding, dec_input], dim=-1).unsqueeze(dim=0),
                    hidden_state.unsqueeze(dim=0),
                )
                hidden_state = hidden_state.squeeze(dim=0)  # (1,dim)

                output_logits.append(self.output(F.relu(hidden_state)))
            return torch.cat(output_logits, dim=0)

        else:
            for di in range(max_len):
                if di == 0:
                    dec_input = torch.LongTensor([[self.SOS_TOKEN]]).to(device)
                dec_input = self.dec_embedding(dec_input).squeeze(dim=0)  # (1,dim)
                if hidden_state is None:
                    hidden_state = dec_input
                hidden_state_repeat = hidden_state.repeat(
                    input_embedding.size(0), 1
                )  # (len, dim)
                combined_input = torch.cat(
                    [hidden_state_repeat, input_embedding], dim=-1
                )  # (len, dim*2)
                attn_weight = F.softmax(
                    self.attn(combined_input).t(), dim=-1
                )  # (1, len)
                input_embedding = attn_weight.mm(input_embedding)  # (1, dim)
                _, hidden_state = self.dec_gru(
                    torch.cat([input_embedding, dec_input], dim=-1).unsqueeze(dim=0),
                    hidden_state.unsqueeze(dim=0),
                )
                hidden_state = hidden_state.squeeze(dim=0)  # (1,dim)
                output = self.output(F.relu(hidden_state))
                topv, topi = output.data.topk(1)
                output_logits.append(F.softmax(output, dim=-1))
                dec_input = topi.detach()
            return torch.cat(output_logits, dim=0)


class Retain(nn.Module):
    def __init__(self, vocab_size, emb_size=64, device=torch.device("cpu:0")):
        super(Retain, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.input_len = vocab_size[0] + vocab_size[1] + vocab_size[2]
        self.output_len = vocab_size[2]

        self.embedding = nn.Sequential(
            nn.Embedding(self.input_len + 1, self.emb_size, padding_idx=self.input_len),
            nn.Dropout(0.5),
        )

        self.alpha_gru = nn.GRU(emb_size, emb_size, batch_first=True)
        self.beta_gru = nn.GRU(emb_size, emb_size, batch_first=True)

        self.alpha_li = nn.Linear(emb_size, 1)
        self.beta_li = nn.Linear(emb_size, emb_size)

        self.output = nn.Linear(emb_size, self.output_len)

    def forward(self, input):
        device = self.device
        # input: (visit, 3, codes )
        max_len = max([(len(v[0]) + len(v[1]) + len(v[2])) for v in input])
        input_np = []
        for visit in input:
            input_tmp = []
            input_tmp.extend(visit[0])
            input_tmp.extend(list(np.array(visit[1]) + self.vocab_size[0]))
            input_tmp.extend(
                list(np.array(visit[2]) + self.vocab_size[0] + self.vocab_size[1])
            )
            if len(input_tmp) < max_len:
                input_tmp.extend([self.input_len] * (max_len - len(input_tmp)))

            input_np.append(input_tmp)

        visit_emb = self.embedding(
            torch.LongTensor(input_np).to(device)
        )  # (visit, max_len, emb)
        visit_emb = torch.sum(visit_emb, dim=1)  # (visit, emb)

        g, _ = self.alpha_gru(visit_emb.unsqueeze(dim=0))  # g: (1, visit, emb)
        h, _ = self.beta_gru(visit_emb.unsqueeze(dim=0))  # h: (1, visit, emb)

        g = g.squeeze(dim=0)  # (visit, emb)
        h = h.squeeze(dim=0)  # (visit, emb)
        attn_g = F.softmax(self.alpha_li(g), dim=-1)  # (visit, 1)
        attn_h = F.tanh(self.beta_li(h))  # (visit, emb)

        c = attn_g * attn_h * visit_emb  # (visit, emb)
        c = torch.sum(c, dim=0).unsqueeze(dim=0)  # (1, emb)

        return self.output(c)


class PositionalEncodingLayer(nn.Module):

    def __init__(self, d_model, max_len=100):
        super(PositionalEncodingLayer, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def get_angles(self, positions, indexes):
        d_model_tensor = torch.FloatTensor([[self.d_model]]).to(positions.device)
        angle_rates = torch.pow(10000, (2 * (indexes // 2)) / d_model_tensor)
        return positions / angle_rates

    def forward(self, input_sequences):
        """
        :param Tensor[batch_size, seq_len] input_sequences
        :return Tensor[batch_size, seq_len, d_model] position_encoding
        """
        positions = torch.arange(input_sequences.size(1)).unsqueeze(1).to(input_sequences.device)  # [seq_len, 1]
        indexes = torch.arange(self.d_model).unsqueeze(0).to(input_sequences.device)  # [1, d_model]
        angles = self.get_angles(positions, indexes)  # [seq_len, d_model]
        angles[:, 0::2] = torch.sin(angles[:, 0::2])  # apply sin to even indices in the tensor; 2i
        angles[:, 1::2] = torch.cos(angles[:, 1::2])  # apply cos to odd indices in the tensor; 2i
        position_encoding = angles.unsqueeze(0).repeat(input_sequences.size(0), 1, 1)  # [batch_size, seq_len, d_model]
        return position_encoding


class MultiheadAttentionEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional

        self.attention = nn.MultiheadAttention(self.head_dim, num_heads)

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim * 2)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        # print(x.size())
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)
        # print(qkv.shape)
        values, att_map = self.attention(q.squeeze(), k.squeeze(), v.squeeze())
        # values = values.transpose(0, 1)
        # values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        # print(values.shape)
        values = values.reshape(batch_size, seq_length, embed_dim)
        # print(values.shape)
        o = self.o_proj(values)
        return o, att_map


class LGC(nn.Module):
    def __init__(self, vocab_size, emb_dim, adj, device=torch.device('cpu:0')):
        super(LGC, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.device = device
        adj = self.normalize(adj + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(vocab_size).to(device)

        self.gcn1 = GraphConvolution(vocab_size, emb_dim)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding1 = self.gcn1(self.x, self.adj)
        node_embedding2 = self.gcn2(node_embedding1, self.adj)
        node_embedding = 1/2 * node_embedding1 + 1/2 * node_embedding2
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


class MICRON(nn.Module):
    def __init__(self, vocab_size, ddi_adj, emb_dim=256, device=torch.device('cpu:0')):
        super(MICRON, self).__init__()

        self.device = device

        # pre-embedding
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(2)])
        self.dropout = nn.Dropout(p=0.5)
        
        self.health_net = nn.Sequential(
                nn.Linear(2 * emb_dim, emb_dim)
        )

        # 
        self.prescription_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, vocab_size[2])
        )

        # graphs, bipartite matrix
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.init_weights()

    def forward(self, input):

	    # patient health representation
        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        
        diag_emb = sum_embedding(self.dropout(self.embeddings[0](torch.LongTensor(input[-1][0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
        prod_emb = sum_embedding(self.dropout(self.embeddings[1](torch.LongTensor(input[-1][1]).unsqueeze(dim=0).to(self.device))))
        # diag_emb = torch.cat(diag_emb, dim=1) #(1,seq,dim)
        # prod_emb = torch.cat(prod_emb, dim=1) #(1,seq,dim)

        if len(input) < 2:
            diag_emb_last = diag_emb * torch.tensor(0.0)
            prod_emb_last = diag_emb * torch.tensor(0.0)
        else:
            diag_emb_last = sum_embedding(self.dropout(self.embeddings[0](torch.LongTensor(input[-2][0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            prod_emb_last = sum_embedding(self.dropout(self.embeddings[1](torch.LongTensor(input[-2][1]).unsqueeze(dim=0).to(self.device))))
            # diag_emb_last = torch.cat(diag_emb_last, dim=1) #(1,seq,dim)
            # prod_emb_last = torch.cat(prod_emb_last, dim=1) #(1,seq,dim)

        health_representation = torch.cat([diag_emb, prod_emb], dim=-1).squeeze(dim=0) # (seq, dim*2)
        health_representation_last = torch.cat([diag_emb_last, prod_emb_last], dim=-1).squeeze(dim=0) # (seq, dim*2)

        health_rep = self.health_net(health_representation)[-1:, :] # (seq, dim)
        health_rep_last = self.health_net(health_representation_last)[-1:, :] # (seq, dim)
        health_residual_rep = health_rep - health_rep_last

	    # drug representation
        drug_rep = self.prescription_net(health_rep)
        drug_rep_last = self.prescription_net(health_rep_last)
        drug_residual_rep = self.prescription_net(health_residual_rep)  

        # reconstructon loss
        rec_loss = 1 / self.tensor_ddi_adj.shape[0] * torch.sum(torch.pow((F.sigmoid(drug_rep) - F.sigmoid(drug_rep_last + drug_residual_rep)), 2))
        
        # ddi_loss
        neg_pred_prob = F.sigmoid(drug_rep)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (vocab_size, vocab_size)

        batch_neg = 1 / self.tensor_ddi_adj.shape[0] *  neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return drug_rep, drug_rep_last, drug_residual_rep, batch_neg, rec_loss

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
