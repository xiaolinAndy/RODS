import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import config


def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)


def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)


def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)


class Encoder(nn.Module):
    def __init__(self, weight):
        super(Encoder, self).__init__()
        #self.embedding = nn.Embedding(config.args.vocab_size, config.emb_dim)
        self.embedding = nn.Embedding.from_pretrained(weight)
        for i in range(weight.shape[0]):
            if weight[i, 0] == 0:
                init_wt_normal(self.embedding.weight[i])

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

    def forward(self, input, seq_lens):
        embedded = self.embedding(input)
        #sorted_length, sorted_idx = torch.sort(seq_lens, descending=True)
        #embedded = embedded[sorted_idx]

        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True, enforce_sorted=False)
        output, hidden = self.lstm(packed)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        encoder_outputs = encoder_outputs.contiguous()
        #_, reversed_idx = torch.sort(sorted_idx)
        #encoder_outputs = encoder_outputs[reversed_idx]


        encoder_feature = encoder_outputs.view(-1, 2 * config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_feature = self.W_h(encoder_feature)

        return encoder_outputs, encoder_feature, hidden


class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden  # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))  # h, c dim = 1 x b x hidden_dim


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage, role_mask, role_type):
        b, t_k, n = list(encoder_outputs.size())
        if role_type == 'user':
            enc_padding_mask = (enc_padding_mask & (role_mask == 1)).float()
        else:
            enc_padding_mask = (enc_padding_mask & (role_mask == 2)).float()

        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()  # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = F.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask  # B x t_k

        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage, scores * enc_padding_mask

class AttentionInter(nn.Module):
    def __init__(self):
        super(AttentionInter, self).__init__()
        # attention
        self.feature_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim, bias=False)
        self.v = nn.Linear(config.hidden_dim, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, enc_padding_mask):
        b, t_k, n = list(encoder_outputs.size())
        encoder_feature = self.feature_proj(encoder_outputs.view(-1, n))
        dec_fea = self.decode_proj(s_t_hat)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()  # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim

        e = F.tanh(att_features)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask[:, :t_k]  # B x t_k

        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        return c_t


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.user_attention_network = Attention()
        self.agent_attention_network = Attention()
        self.inter_attention_network_1 = AttentionInter()
        self.inter_attention_network_2 = AttentionInter()
        # decoder
        self.embedding = nn.Embedding(config.args.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(config.hidden_dim * 3 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        # if config.pointer_gen:
        #     if config.args.merge == 'gate':
        #         self.p_gen_linear = nn.Linear(config.hidden_dim * 5 + config.emb_dim, 1)
        #     elif config.args.merge == 'tanh' or config.args.merge == 'linear':
        #         self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)
        #
        # # p_vocab
        # if config.args.merge == 'gate':
        #     self.out1 = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        # elif config.args.merge == 'tanh' or config.args.merge == 'linear':
        #     self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        #     self.activate = nn.Tanh()
        self.p_gen_linear = nn.Linear(config.hidden_dim * 7 + config.emb_dim, 1)
        self.p_role_linear = nn.Linear(config.hidden_dim * 7 + config.emb_dim, 1)
        self.out1 = nn.Linear(config.hidden_dim * 6, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.args.vocab_size)
        init_linear_wt(self.out2)
        self.trans = nn.Linear(config.hidden_dim, config.hidden_dim*2, bias=False)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step, int_t_1, inter_masks, inter_features, role_mask):

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t_user, _, coverage_next_user, _ = self.user_attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                                          enc_padding_mask, coverage, role_mask,
                                                                          role_type='user')
            c_t_agent, _, coverage_next_agent, _ = self.agent_attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                                             enc_padding_mask, coverage, role_mask,
                                                                             role_type='agent')
            if coverage_next_user is not None:
                coverage = 0.5 * coverage_next_user + 0.5 * coverage_next_agent
            else:
                coverage = None

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((int_t_1, c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t_user, attn_dist_user, coverage_next_user, score_user = self.user_attention_network(s_t_hat, encoder_outputs,
                                                                                   encoder_feature,
                                                                                   enc_padding_mask, coverage,
                                                                                   role_mask, role_type='user')
        c_t_agent, attn_dist_agent, coverage_next_agent, score_agent = self.agent_attention_network(s_t_hat, encoder_outputs,
                                                                                       encoder_feature,
                                                                                       enc_padding_mask, coverage,
                                                                                       role_mask, role_type='agent')
        # inter att 1
        if inter_features is not None:
            int_t = self.inter_attention_network_1(s_t_hat, inter_features, inter_masks)
        else:
            int_t = int_t_1


        state_input = torch.cat((c_t_user, c_t_agent, int_t, s_t_hat, x), 1)
        p_gen = self.p_gen_linear(state_input)
        p_gen = F.sigmoid(p_gen)
        p_role = self.p_role_linear(state_input)
        p_role = F.sigmoid(p_role)
        c_t = p_role * c_t_user + (1 - p_role) * c_t_agent
        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t_user, c_t_agent, int_t), 1)
        output = self.out1(output)
        output = self.out2(output)  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)
        vocab_dist_ = p_gen * vocab_dist
        attn_dist = p_role * attn_dist_user + (1 - p_role) * attn_dist_agent
        attn_dist_ = (1 - p_gen) * attn_dist

        if extra_zeros is not None:
            vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

        final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        if coverage_next_user is not None:
            coverage_next = p_role * coverage_next_user + (1 - p_role) * coverage_next_agent
        else:
            coverage_next = None

        if self.training or step > 0:
            coverage = coverage_next

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage, h_decoder.view(-1, config.hidden_dim).unsqueeze(1), int_t, score_user, score_agent

class FinalDecoder(nn.Module):
    def __init__(self):
        super(FinalDecoder, self).__init__()
        self.attention_network = Attention()
        # decoder
        self.embedding = nn.Embedding(config.args.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        # p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.args.vocab_size)
        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step, int_t_1, inter_masks, inter_features):

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                           enc_padding_mask, coverage)
            coverage = coverage_next

        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                               enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_next

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1)  # B x hidden_dim * 4
        output = self.out1(output)  # B x hidden_dim

        # output = F.relu(output)

        output = self.out2(output)  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage, h_decoder.view(-1, config.hidden_dim).unsqueeze(1), int_t_1


class PGNModel(object):
    def __init__(self, model_file_path=None, is_eval=False, device=None, embedding=None):
        encoder = Encoder(embedding)
        reduce_state = ReduceState()
        final_decoder = FinalDecoder()
        user_decoder = Decoder()
        agent_decoder = Decoder()

        # shared the embedding between encoder and decoder
        final_decoder.embedding.weight = encoder.embedding.weight
        user_decoder.embedding.weight = encoder.embedding.weight
        agent_decoder.embedding.weight = encoder.embedding.weight
        if is_eval:
            encoder = encoder.eval()
            reduce_state = reduce_state.eval()
            final_decoder = final_decoder.eval()
            user_decoder = user_decoder.eval()
            agent_decoder = agent_decoder.eval()

        encoder = encoder.to(device)
        reduce_state = reduce_state.to(device)
        final_decoder = final_decoder.to(device)
        user_decoder = user_decoder.to(device)
        agent_decoder = agent_decoder.to(device)

        self.encoder = encoder
        self.reduce_state = reduce_state
        self.final_decoder = final_decoder
        self.user_decoder = user_decoder
        self.agent_decoder = agent_decoder

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.reduce_state.load_state_dict(state['reduce_state_dict'])
            self.final_decoder.load_state_dict(state['final_state_dict'], strict=False)
            self.user_decoder.load_state_dict(state['user_state_dict'], strict=False)
            self.agent_decoder.load_state_dict(state['agent_state_dict'], strict=False)

    def eval(self):
        self.encoder = self.encoder.eval()
        self.reduce_state = self.reduce_state.eval()
        self.final_decoder = self.final_decoder.eval()
        self.user_decoder = self.user_decoder.eval()
        self.agent_decoder = self.agent_decoder.eval()

    def train(self):
        self.encoder = self.encoder.train()
        self.reduce_state = self.reduce_state.train()
        self.final_decoder = self.final_decoder.train()
        self.user_decoder = self.user_decoder.train()
        self.agent_decoder = self.agent_decoder.train()