import json
import torch
import nltk
import re
import jieba

from data_utils.vocab_gen import Vocab, article2ids, abstract2ids
from torch.utils.data import Dataset, RandomSampler, DataLoader, SequentialSampler
from utils.config import args
from utils import config
from data_utils.tokenizer import Tokenizer, get_dialogue_vocab, load_vocab, get_tencent_embedding
from pathlib import Path


dm_single_close_quote = '\u2019' # unicode
dm_double_close_quote = '\u201d'
# acceptable ways to end a sentence
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"',
              dm_single_close_quote, dm_double_close_quote, ")"]
# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '<PAD>' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '<UNK>' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '<START>' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '<END>' # This has a vocab id, which is used at the end of untruncated target sequences



class PGNDataset(Dataset):
    def __init__(self, features):
        self.data = features
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data = self.data[idx]
        return data

def get_data(data_pth):
    with open(data_pth, 'r') as f:
        data = json.load(f)
    return data


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + " ."

def get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id):
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len: # truncate
      inp = inp[:max_len]
      target = target[:max_len] # no end_token
    else: # no truncation
      target.append(stop_id) # end token
    assert len(inp) == len(target)
    return inp, target

def get_border_id(sample):
    borders = [[], []]
    for topic in sample['Topics']:
        for tri in topic['Triplets']:
            min_index = min(tri['QueSummUttIDs'] + tri['AnsSummUttIDs'])
            max_index = max(tri['QueSummUttIDs'] + tri['AnsSummUttIDs'])
            borders[0].append(min_index)
            borders[1].append(max_index)
    return borders


def convert_example_to_feature(data, max_seq_length, org_vocab, context_mode, get_vocab=False):
    features = []
    num_len_below, num_len_over = 0, 0
    final_sums, user_sums, agent_sums, contexts, contexts_joint = [], [], [], [], []
    role_masks = []
    if args.data_name == 'CSDS':
        for sample in data:
            final_sum = ''.join(sample['FinalSumm'])
            user_sum = ''.join(sample['UserSumm'])
            agent_sum = ''.join(sample['AgentSumm'])
            final_sums.append(final_sum)
            user_sums.append(user_sum)
            agent_sums.append(agent_sum)
            context = []
            role_mask = []
            for turn in sample['Dialogue']:
                tmp_utt = []
                # if turn['turn'] in border_ids[0]:
                #     tmp_utt.append('<BOP>')
                if turn['speaker'] == 'Q':
                    tmp_utt += [sample['QRole'], ':']
                else:
                    tmp_utt += ['客服', ':']
                if args.complete:
                    sent = jieba.lcut(''.join(turn['new_utterance'].split()))
                else:
                    sent = turn['utterance'].split()
                for word in sent:
                    if len(word) > 2 and word[0] == '[' and word[-1] == ']':
                        tmp_utt += ['[', word[1:-1], ']']
                    else:
                        tmp_utt.append(word)
                # if turn['turn'] in border_ids[1]:
                #     tmp_utt.append('<EOP>')
                tmp_utt = ' '.join(tmp_utt)
                context.append(tmp_utt)
                if turn['speaker'] == 'Q':
                    role_mask += [1] * (len(tmp_utt.split()) + 1)
                else:
                    role_mask += [2] * (len(tmp_utt.split()) + 1)
                # role_mask += [0]
            context_joint = " <EOU> ".join(context)
            contexts.append(context)
            contexts_joint.append(context_joint)
            role_masks.append(role_mask[:-1])
    elif args.data_name == 'MC':
        for sample in data:
            user_sum = ' '.join(sample['summary']['description'])
            agent_sum = ' '.join(sample['summary']['suggestion'])
            user_sums.append(user_sum)
            agent_sums.append(agent_sum)
            context = []
            role_mask = []
            for turn in sample['content']:
                tmp_utt = []
                tmp_utt += [turn['speaker'], ':']
                tmp_utt += turn['utterance']
                tmp_utt = ' '.join(tmp_utt)
                context.append(tmp_utt)
                if turn['speaker'] == '患者':
                    role_mask += [1] * (len(tmp_utt.split()) + 1)
                else:
                    role_mask += [2] * (len(tmp_utt.split()) + 1)

            context_joint = " <EOU> ".join(context)
            contexts.append(context)
            contexts_joint.append(context_joint)
            role_masks.append(role_mask[:-1])


    # get vocab
    if get_vocab:
        if args.new_vocab:
            get_dialogue_vocab(agent_sums, contexts, Path('data_utils/embeddings/'), 'word')
            dialogue_vocab = load_vocab(Path('data_utils/embeddings/dialogue_vocab_word'))
            get_tencent_embedding(dialogue_vocab, Path('../../pretrained/Tencent_AILab_ChineseEmbedding.txt'),
                                  Path(args.vocab_path))
        vocab = Tokenizer(args.vocab_path, args.vocab_dim, args.vocab_size)
    else:
        vocab = org_vocab

    start_decoding = vocab.token2idx(START_DECODING)
    stop_decoding = vocab.token2idx(STOP_DECODING)

    for user_sum, agent_sum, context, context_joint, role_mask in zip(user_sums, agent_sums, contexts, contexts_joint, role_masks):
        user_sum_ids = vocab.tokenize(user_sum, False)
        agent_sum_ids = vocab.tokenize(agent_sum, False)
        context_ids = vocab.tokenize(context_joint, True)
        if len(context_ids) > max_seq_length:
            num_len_over += 1
        else:
            num_len_below += 1
        context_ids = context_ids[:max_seq_length]
        role_mask = role_mask[:max_seq_length]



        user_input, user_output = get_dec_inp_targ_seqs(user_sum_ids, args.max_dec_steps, start_decoding,
                                                          stop_decoding)
        agent_input, agent_output = get_dec_inp_targ_seqs(agent_sum_ids, args.max_dec_steps, start_decoding,
                                                          stop_decoding)
        enc_input_extend, article_oovs = vocab.article2ids(context_joint.split()[:max_seq_length])

        user_extend = vocab.abstract2ids(vocab.split(user_sum), article_oovs)
        user_input_extend, user_output_extend = get_dec_inp_targ_seqs(user_extend, args.max_dec_steps, start_decoding,
                                                                        stop_decoding)
        agent_extend = vocab.abstract2ids(vocab.split(agent_sum), article_oovs)
        agent_input_extend, agent_output_extend = get_dec_inp_targ_seqs(agent_extend, args.max_dec_steps, start_decoding,
                                                                        stop_decoding)

        features.append(
            {'enc_input': context_ids,
             'enc_input_extend': enc_input_extend,
             'user_input': user_input,
             'user_output': user_output,
             'user_input_extend': user_input_extend,
             'user_output_extend': user_output_extend,
             'agent_input': agent_input,
             'agent_output': agent_output,
             'agent_input_extend': agent_input_extend,
             'agent_output_extend': agent_output_extend,
             'oovs': article_oovs,
             'pad': vocab.token2idx(PAD_TOKEN),
             'user_sum': user_sum,
             'agent_sum': agent_sum,
             'role_mask': role_mask}
            )
    print(num_len_below, num_len_over)

    return features, vocab

def padded_sequence(seqs, pad):
    max_len = max([len(seq) for seq in seqs])
    padded_seqs = [seq + [pad] * (max_len - len(seq)) for seq in seqs]
    length = [len(seq) for seq in seqs]
    return padded_seqs, length

def padded_sequence_len(seqs, pad, max_len):
    padded_seqs = [seq + [pad] * (max_len - len(seq)) for seq in seqs]
    length = [len(seq) for seq in seqs]
    return padded_seqs, length

def batchify_data(batch):
    enc_input_ids = [f['enc_input'] for f in batch]
    enc_input_ids, enc_lens = padded_sequence(enc_input_ids, batch[0]['pad'])
    enc_input_ids = torch.tensor(enc_input_ids, dtype=torch.long)
    enc_lens = torch.tensor(enc_lens, dtype=torch.long)
    enc_batch_extend_vocab = [f['enc_input_extend'] for f in batch]
    enc_batch_extend_vocab, _ = padded_sequence(enc_batch_extend_vocab, batch[0]['pad'])
    enc_batch_extend_vocab = torch.tensor(enc_batch_extend_vocab, dtype=torch.long)
    art_oovs = [f['oovs'] for f in batch]
    max_art_oovs = max([len(f['oovs']) for f in batch])
    role_mask_ids = [f['role_mask'] for f in batch]
    role_mask_ids, _ = padded_sequence(role_mask_ids, 0)
    role_mask_ids = torch.tensor(role_mask_ids, dtype=torch.long)

    user_input_ids = [f['user_input'] for f in batch]
    user_output_ids = [f['user_output'] for f in batch]
    if config.pointer_gen:
        user_output_ids = [f['user_output_extend'] for f in batch]
    agent_input_ids = [f['agent_input'] for f in batch]
    agent_output_ids = [f['agent_output'] for f in batch]
    if config.pointer_gen:
        agent_output_ids = [f['agent_output_extend'] for f in batch]

    max_user_len = max([len(seq) for seq in user_input_ids])
    max_agent_len = max([len(seq) for seq in agent_input_ids])
    max_len = max(max_user_len, max_agent_len)


    user_input_ids, user_lens = padded_sequence_len(user_input_ids, batch[0]['pad'], max_len)
    user_output_ids, user_lens = padded_sequence_len(user_output_ids, batch[0]['pad'], max_len)
    user_input_ids = torch.tensor(user_input_ids, dtype=torch.long)
    user_output_ids = torch.tensor(user_output_ids, dtype=torch.long)
    user_lens = torch.tensor(user_lens, dtype=torch.long)


    agent_input_ids, agent_lens = padded_sequence_len(agent_input_ids, batch[0]['pad'], max_len)
    agent_output_ids, agent_lens = padded_sequence_len(agent_output_ids, batch[0]['pad'], max_len)
    agent_input_ids = torch.tensor(agent_input_ids, dtype=torch.long)
    agent_output_ids = torch.tensor(agent_output_ids, dtype=torch.long)
    agent_lens = torch.tensor(agent_lens, dtype=torch.long)

    user_sums = [f['user_sum'] for f in batch]
    agent_sums = [f['agent_sum'] for f in batch]


    return (enc_input_ids, enc_lens, enc_batch_extend_vocab, art_oovs, max_art_oovs, role_mask_ids), \
           (None, None, None), \
           (user_input_ids, user_output_ids, user_lens), \
           (agent_input_ids, agent_output_ids, agent_lens), \
           (None, user_sums, agent_sums)

def get_train_dataloader(data_pth, max_seq_length, train_batch_size):
    print('processing training data-------------')
    data = get_data(data_pth)
    features, vocab = convert_example_to_feature(data, max_seq_length, None, context_mode=args.context_mode, get_vocab=True)
    train_data = PGNDataset(features)
    sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=sampler, batch_size=train_batch_size, collate_fn=batchify_data)
    return train_dataloader, features, vocab

def get_val_dataloader(data_pth, vocab, max_seq_length, val_batch_size, mode='eval', beam_size=0):
    data = get_data(data_pth)
    features, vocab = convert_example_to_feature(data, max_seq_length, vocab, context_mode=args.context_mode, get_vocab=False)
    if mode == 'eval':
        val_data = PGNDataset(features)
        sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=sampler, batch_size=val_batch_size, collate_fn=batchify_data)
    else:
        decode_features = []
        for f in features:
            decode_features.extend([f] * beam_size)
        val_data = PGNDataset(decode_features)
        sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=sampler, batch_size=beam_size, collate_fn=batchify_data)
    return val_dataloader, features




