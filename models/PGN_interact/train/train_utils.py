from torch.autograd import Variable
import numpy as np
import torch
from utils import config


def padded_sequence(seqs, pad):
    max_len = max([len(seq) for seq in seqs])
    padded_seqs = [seq + [pad] * (max_len - len(seq)) for seq in seqs]
    length = [len(seq) for seq in seqs]
    return padded_seqs, length

def get_input_from_batch(features, device, vocab):
    batch_size = len(features[0][0])
    enc_input_ids, enc_lens, enc_batch_extend_vocab, art_oovs, max_art_oovs, role_mask_ids = features[0]
    enc_padding_mask = enc_input_ids.ne(vocab.token2idx('<PAD>'))

    extra_zeros = None

    if config.pointer_gen:
        # max_art_oovs is the max over all the article oov list in the batch
        if max_art_oovs > 0:
          extra_zeros = torch.zeros((batch_size, max_art_oovs))

    c_t_1 = torch.zeros((batch_size, 2 * config.hidden_dim))

    coverage = None
    if config.is_coverage:
        coverage = torch.zeros(enc_input_ids.size())

    enc_input_ids = enc_input_ids.to(device)
    enc_padding_mask = enc_padding_mask.to(device)
    enc_lens = enc_lens.to(device)


    if enc_batch_extend_vocab is not None:
        enc_batch_extend_vocab = enc_batch_extend_vocab.to(device)
    if extra_zeros is not None:
        extra_zeros = extra_zeros.to(device)
    c_t_1 = c_t_1.to(device)

    if coverage is not None:
        coverage = coverage.to(device)

    role_mask_ids = role_mask_ids.to(device)

    return enc_input_ids, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, role_mask_ids

def get_final_from_batch(features, device, vocab):
    final_input_ids, final_output_ids, final_lens = features[1]
    final_padding_mask = final_input_ids.ne(vocab.token2idx('<PAD>')).float()
    max_final_len = max(final_lens)

    final_input_ids = final_input_ids.to(device)
    final_padding_mask = final_padding_mask.to(device)
    final_lens = final_lens.to(device)
    final_output_ids = final_output_ids.to(device)

    return final_input_ids, final_padding_mask, max_final_len, final_lens, final_output_ids

def get_user_from_batch(features, device, vocab):
    user_input_ids, user_output_ids, user_lens = features[2]
    user_padding_mask = user_input_ids.ne(vocab.token2idx('<PAD>')).float()
    max_user_len = max(user_lens)

    user_input_ids = user_input_ids.to(device)
    user_padding_mask = user_padding_mask.to(device)
    user_lens = user_lens.to(device)
    user_output_ids = user_output_ids.to(device)

    return user_input_ids, user_padding_mask, max_user_len, user_lens, user_output_ids

def get_agent_from_batch(features, device, vocab):
    agent_input_ids, agent_output_ids, agent_lens = features[3]
    agent_padding_mask = agent_input_ids.ne(vocab.token2idx('<PAD>')).float()
    max_agent_len = max(agent_lens)

    agent_input_ids = agent_input_ids.to(device)
    agent_padding_mask = agent_padding_mask.to(device)
    agent_lens = agent_lens.to(device)
    agent_output_ids = agent_output_ids.to(device)

    return agent_input_ids, agent_padding_mask, max_agent_len, agent_lens, agent_output_ids

def get_sums_from_batch(features):
    return features[4]

