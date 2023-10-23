from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss

import dgl
from dgl.nn import GATConv
from models.RoBERTa.SRoBERTa import RobertaPreTrainedModel, RobertaModel
from dataset import emotion_mapping


def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


class MPEG_RoBERTa(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, num_labels, num_graph_attention, gcn_layers=2, data_name="RECCON", activation='relu'):
        super().__init__(config)
        self.num_labels = num_labels
        self.roberta = RobertaModel(config)
        self.emotion_embeddings = nn.ParameterList([nn.Parameter(torch.zeros(1, config.hidden_size))
                                                    for _ in range(len(emotion_mapping[data_name]) + 1)])

        self.gcn_dim = config.hidden_size
        self.num_layers = gcn_layers

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            assert 1 == 2, "you should provide activation function."

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.gcn_dim * 2 * (self.num_layers + 1), self.num_labels)
        self.init_weights()

        self.graph_attention_size = int(self.gcn_dim / num_graph_attention)
        self.GCN_layers = nn.ModuleList([HANLayer(meta_paths=[['speaker', 'speaker'],
                                                              ['dialog', 'dialog'], ['entity', 'entity']],
                                                  in_size=self.gcn_dim, out_size=self.graph_attention_size,
                                                  layer_num_heads=num_graph_attention) for _ in range(self.num_layers)])

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.turnAttention = MultiHeadAttention(config.num_attention_heads, config.hidden_size,
                                                self.attention_head_size, self.attention_head_size,
                                                config.attention_probs_dropout_prob)
        self.emotionAttention = MultiHeadAttention(config.num_attention_heads, config.hidden_size*2,
                                                   self.attention_head_size*2, self.attention_head_size*2,
                                                   config.attention_probs_dropout_prob)
        self.transform = nn.Linear(config.hidden_size * 2, self.gcn_dim)

        self.ffn_layers = nn.ModuleList([PositionWiseFeedForward(self.gcn_dim, self.gcn_dim, 0.2) for _ in range(self.num_layers)])

    def init_emotions(self, tokenizer, data_name):
        for k, v in emotion_mapping[data_name].items():
            input_ids = torch.tensor(tokenizer.encode(v)).unsqueeze(0)
            speaker_ids = torch.zeros_like(input_ids)
            outputs = self.roberta(input_ids, speaker_ids=speaker_ids)
            self.emotion_embeddings[k+1] = nn.Parameter(outputs[1])

    def forward(
            self,
            input_ids=None,
            attention_masks=None,
            token_type_ids=None,
            position_ids=None,
            head_masks=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False,
            speaker_ids=None,
            graphs=None,
            mention_ids=None,
            emotion_ids=None,
            turn_masks=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        slen = input_ids.size(1)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_masks,
            token_type_ids=None,
            position_ids=position_ids,
            speaker_ids=speaker_ids,
            head_mask=head_masks,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_outputs = outputs[0]
        pooled_outputs = outputs[1]

        features = None
        sequence_outputs, _ = self.turnAttention(sequence_outputs, sequence_outputs, sequence_outputs, turn_masks)

        num_batch_turn = []

        # initialize graph nodes
        for i in range(len(graphs)):
            sequence_output = sequence_outputs[i]
            mention_num = torch.max(mention_ids[i])
            num_batch_turn.append(mention_num + 1)
            mention_index = get_cuda((torch.arange(mention_num) + 1).unsqueeze(1).expand(-1, slen))
            mentions = mention_ids[i].unsqueeze(0).expand(mention_num, -1)
            select_metrix = (mention_index == mentions).float()

            emotion_index = select_metrix.argmax(dim=1)
            emotions = [int(emotion_ids[i][j]) for j in emotion_index]
            emotion_metrix = [self.emotion_embeddings[k] for k in emotions]
            emotion_metrix = torch.stack(emotion_metrix).squeeze().to(select_metrix.device)

            word_total_numbers = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, slen)
            select_metrix = torch.where(word_total_numbers > 0, select_metrix / word_total_numbers, select_metrix)

            x = torch.mm(select_metrix, sequence_output)
            x = torch.cat((x, emotion_metrix), dim=1).unsqueeze(0)
            x, _ = self.emotionAttention(x, x, x)
            x = self.transform(x.squeeze())

            dialog = pooled_outputs[i].unsqueeze(0)
            emotion_dialog = self.emotion_embeddings[0].to(dialog.device)
            dialog = torch.cat((dialog, emotion_dialog), dim=1)
            dialog = self.transform(dialog)
            x = torch.cat((dialog, x), dim=0)

            if features is None:
                features = x
            else:
                features = torch.cat((features, x), dim=0)

        # construct big graph
        graph_big = dgl.batch(graphs)
        output_features = [features]

        # graph encoding
        for layer_num, GCN_layer in enumerate(self.GCN_layers):
            start = 0
            new_features = []
            for idx in num_batch_turn:
                new_features.append(features[start])
                ffn_out = self.ffn_layers[layer_num](features[start + 1: start + idx - 2].unsqueeze(1)).squeeze(1)
                new_features += ffn_out
                new_features.append(features[start + idx - 2])
                new_features.append(features[start + idx - 1])
                start += idx
            features = torch.stack(new_features)
            features = GCN_layer(graph_big, features)
            output_features.append(features)
        graphs = dgl.unbatch(graph_big)

        # get the output of each mini graph
        graph_output = list()
        fea_idx = 0
        for i in range(len(graphs)):
            node_num = graphs[i].number_of_nodes('node')
            integrated_output = None
            for j in range(self.num_layers + 1):
                if integrated_output is None:
                    integrated_output = output_features[j][fea_idx + node_num - 2]
                else:
                    integrated_output = torch.cat((integrated_output, output_features[j][fea_idx + node_num - 2]), dim=-1)
                integrated_output = torch.cat((integrated_output, output_features[j][fea_idx + node_num - 1]), dim=-1)
            fea_idx += node_num
            graph_output.append(integrated_output)
        graph_output = torch.stack(graph_output)

        # classify 
        pooled_output = self.dropout(graph_output)
        logits = self.classifier(pooled_output)
        logits = logits.view(-1, self.num_labels)
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.view(-1, self.num_labels)
            loss = loss_fct(logits, labels.float())
            return loss, logits
        else:
            return logits


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(in_channels=1, out_channels=input_size, kernel_size=1)
        self.w2 = nn.Conv1d(in_channels=hidden_size, out_channels=1, kernel_size=1)
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w2(F.elu(self.w1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x


class HANLayer(nn.Module):

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout=0.2):
        super(HANLayer, self).__init__()

        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads, dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(g, meta_path)
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        return self.semantic_attention(semantic_embeddings)


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)

        return (beta * z).sum(1)


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
