import json
import random
import numpy as np
import torch
import os
import time
import torch.nn.functional as F
import files2rouge
import re
import math
import tqdm

from utils import config
from model import pgn
from data_utils.batch import get_train_dataloader, get_val_dataloader
from train.train_utils import get_input_from_batch, get_final_from_batch, get_user_from_batch, get_agent_from_batch, get_sums_from_batch
from data_utils.tokenizer import Tokenizer
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adagrad, Adam
from utils.metric import cal_mul_sums_n, cal_mul_sums_l

class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage, inter, history, mask, stop, attn=None):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage
        self.inter = inter
        self.history = history
        self.mask = mask
        self.stop = stop
        self.attn = attn

    def extend(self, token, log_prob, state, context, coverage, inter, history, mask, stop, attn=None):
        if self.history is None:
            new_history = history
            new_mask = mask
        else:
            #print(self.mask)
            new_history = torch.cat([self.history, history], 0)
            new_mask = torch.cat([self.mask, mask], 0)
        return Beam(tokens = self.tokens + [token],
                    log_probs = self.log_probs + [log_prob],
                    state = state,
                    context = context,
                    coverage = coverage,
                    inter = inter,
                    history = new_history,
                    mask = new_mask,
                    stop=self.stop or stop,
                    attn=None if attn is None else self.attn + [attn])

    def padding(self, length):
        pad_len = length - self.history.shape[0]
        new_history = torch.cat([self.history, torch.zeros(pad_len, self.history.shape[-1]).to(self.history)])
        new_mask = torch.cat([self.mask, torch.zeros(pad_len).to(self.mask)])
        return Beam(tokens=self.tokens,
                    log_probs=self.log_probs,
                    state=self.state,
                    context=self.context,
                    coverage=self.coverage,
                    inter=self.inter,
                    history=new_history,
                    mask=new_mask,
                    stop=self.stop,
                    attn=self.attn)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)

class MultiBeam(object):
    def __init__(self, user, agent):
        self.user = user
        self.agent = agent
        self.beams = [self.user, self.agent]

    @property
    def avg_log_prob(self):
        return (self.user.avg_log_prob + self.agent.avg_log_prob)

def change_word2id_split(ref, pred):
    ref_id, pred_id = [], []
    tmp_dict = {'%': 0}
    new_index = 1
    words = list(ref)
    for w in words:
        if w not in tmp_dict.keys():
            tmp_dict[w] = new_index
            ref_id.append(str(new_index))
            new_index += 1
        else:
            ref_id.append(str(tmp_dict[w]))
        if w == '。':
            ref_id.append(str(0))
    words = list(pred)
    for w in words:
        if w not in tmp_dict.keys():
            tmp_dict[w] = new_index
            pred_id.append(str(new_index))
            new_index += 1
        else:
            pred_id.append(str(tmp_dict[w]))
        if w == '。':
            pred_id.append(str(0))
    return ' '.join(ref_id), ' '.join(pred_id)


class BeamSearch(object):
    def __init__(self, model, vocab, dataloader, device, save, args):
        model.eval()
        self.vocab = vocab
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.save_dir = args.save_path
        self.save = save
        self.args = args

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


    def decode(self, index=None):
        refs, preds = [[], [], []], [[], [], []]
        attns = [[], []]
        dials = []
        if self.args.no_final:
            types = ['user', 'agent']
        else:
            types = ['final', 'user', 'agent']
        start_time = time.time()
        count = 0
        for batch in tqdm.tqdm(self.dataloader):
            if index is not None and count != index:
                count += 1
                continue
            count += 1

            # Run beam search to get best Hypothesis
            art_oovs = batch[0][3][0]
            final_refs, user_refs, agent_refs = get_sums_from_batch(batch)
            if self.args.no_final:
                best_summary_user, best_summary_agent = self.beam_search(batch)
                ref_seqs = [user_refs, agent_refs]
                pred_seqs = [best_summary_user, best_summary_agent]
            else:
                best_summary_final, best_summary_user, best_summary_agent = self.beam_search(batch)
                ref_seqs = [final_refs, user_refs, agent_refs]
                pred_seqs = [best_summary_final, best_summary_user, best_summary_agent]

            for i, (ref, pred) in enumerate(zip(ref_seqs, pred_seqs)):
                # Extract the output ids from the hypothesis and convert back to words
                output_ids = [int(t) for t in pred.tokens[1:]]
                decoded_words = self.vocab.outputids2words(output_ids, art_oovs)

                # Remove the [STOP] token from decoded_words, if necessary
                try:
                    fst_stop_idx = decoded_words.index('<END>')
                    decoded_words = decoded_words[:fst_stop_idx]
                except ValueError:
                    decoded_words = decoded_words

                # calulate based on character
                refs[i].append(''.join(ref[0].split()))
                preds[i].append(''.join(decoded_words))

                if index is not None:
                    attns[i].append(pred.attn)

            _, _, _, input_ids, _, _, _, _= get_input_from_batch(batch, self.device, self.vocab)
            input_ids = [int(t) for t in input_ids[0]]
            dial_words = self.vocab.outputids2words(input_ids, art_oovs)
            dials.append(dial_words)

        if index is not None:
            attn_data = {'user': None, 'agent': None, 'dial': None}
            for i in range(len(types)):
                # print('ref: ', refs[i][0])
                # print('pred: ', preds[i][0])
                # print('attn: ', attns[i][0])
                # print(len(attns[i][0]))
                attn = torch.mean(torch.stack(attns[i][0], 0), 0).cpu().numpy().tolist()
                attn_data[types[i]] = attn
            attn_data['dial'] = dials[0]
            print(len(attn_data['user']), len(attn_data['agent']), len(attn_data['dial']))
            with open('attn_score.json', 'w') as f:
                json.dump(attn_data, f, ensure_ascii=False)
            exit()



        for i in range(len(types)):
            print('ref: ', refs[i][0])
            print('pred: ', preds[i][0])
            #print('time: ', time.time() - start_time)
            # using files2rouge as calculating method, chinese
            ref_ids, pred_ids = [], []
            if self.save:
                with open(self.save_dir + types[i] + '_refs.txt', 'w') as f:
                    for ref in refs[i]:
                        f.write(ref + '\n')
                with open(self.save_dir + types[i] + '_preds.txt', 'w') as f:
                    for pred in preds[i]:
                        f.write(pred + '\n')
            for ref, pred in zip(refs[i], preds[i]):
                ref_id, pred_id = change_word2id_split(ref, pred)
                ref_ids.append(ref_id)
                pred_ids.append(pred_id)

            print("Decoder has finished reading dataset for single_pass.")
            print("Now starting ROUGE eval...")
            with open(self.save_dir + 'ref_ids.txt', 'w') as f:
                for ref in ref_ids:
                    f.write(ref + '\n')
            with open(self.save_dir + 'pred_ids.txt', 'w') as f:
                for pred in pred_ids:
                    f.write(pred + '\n')
            os.system('files2rouge %s/ref_ids.txt %s/pred_ids.txt -s rouge.txt -e 0' % (self.save_dir, self.save_dir))


    def get_ratio(self, attn, role_mask):
        user_attn = torch.sum(attn[role_mask == 1])
        agent_attn = torch.sum(attn[role_mask == 2])
        return user_attn.item(), agent_attn.item()


    def beam_search(self, batch):

        # batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0, role_mask = get_input_from_batch(
            batch, self.device, self.vocab)
        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0  # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        with torch.no_grad():
            # decode final
            if not self.args.no_final:
                decoder = self.model.final_decoder  #, self.model.user_decoder, self.model.agent_decoder]

                # decoder batch preparation, it has beam_size example initially everything is repeated
                final_beams = []
                for _ in range(config.args.beam_size):
                    final_beams.append(Beam(tokens=[self.vocab.token2idx('<START>')],
                                          log_probs=[0.0],
                                          state=(dec_h[0], dec_c[0]),
                                          context=c_t_0[0],
                                          coverage=(coverage_t_0[0] if config.is_coverage else None),
                                          inter=torch.zeros(config.hidden_dim).to(c_t_0),
                                          history=None,
                                          mask=None,
                                          stop=False))
                final_results = []
                steps = 0
                while steps < config.args.max_dec_steps and len(final_results) < config.args.beam_size:
                    latest_tokens = [h.latest_token for h in final_beams]
                    latest_tokens = [t if t < self.vocab.vocab_size else self.vocab.token2idx('[UNK]') \
                                     for t in latest_tokens]
                    y_t_1 = torch.LongTensor(latest_tokens)
                    y_t_1 = y_t_1.to(self.device)
                    all_state_h = []
                    all_state_c = []

                    all_context = []

                    for h in final_beams:
                        state_h, state_c = h.state
                        all_state_h.append(state_h)
                        all_state_c.append(state_c)

                        all_context.append(h.context)


                    s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
                    c_t_1 = torch.stack(all_context, 0)
                    int_t_1 = None

                    coverage_t_1 = None
                    if config.is_coverage:
                        all_coverage = []
                        for h in final_beams:
                            all_coverage.append(h.coverage)
                        coverage_t_1 = torch.stack(all_coverage, 0)

                    # inter mask & features
                    all_inter_feats = None
                    all_inter_masks = None
                    final_dist, s_t, c_t, attn_dist, p_gen, coverage_t, new_inter_feature, int_t = decoder(y_t_1, s_t_1,
                                                                                                               encoder_outputs,
                                                                                                               encoder_feature,
                                                                                                               enc_padding_mask,
                                                                                                               c_t_1,
                                                                                                               extra_zeros,
                                                                                                               enc_batch_extend_vocab,
                                                                                                               coverage_t_1,
                                                                                                               steps,
                                                                                                               int_t_1,
                                                                                                               all_inter_masks,
                                                                                                               all_inter_feats)
                    log_probs = torch.log(final_dist)
                    topk_log_probs, topk_ids = torch.topk(log_probs, config.args.beam_size * 2)

                    dec_h, dec_c = s_t
                    dec_h = dec_h.squeeze()
                    dec_c = dec_c.squeeze()

                    all_beams = []
                    num_orig_beams = 1 if steps == 0 else len(final_beams)
                    for i in range(num_orig_beams):
                        for k in range(config.args.beam_size * 2):  # for each of the top 2*beam_size hyps:
                            h = final_beams[i]
                            state_i = (dec_h[i], dec_c[i])
                            context_i = c_t[i]
                            coverage_i = (coverage_t[i] if config.is_coverage else None)
                            inter_i = None
                            history_i = None

                            if topk_ids[i, k].item() == self.vocab.token2idx('<END>') or h.stop:
                                stop_i = True
                            else:
                                stop_i = False
                            if h.stop:
                                mask_i = torch.zeros(1).to(self.device)
                            else:
                                mask_i = torch.ones(1).to(self.device)
                            new_beam = h.extend(token=topk_ids[i, k].item(),
                                                log_prob=topk_log_probs[i, k].item(),
                                                state=state_i,
                                                context=context_i,
                                                coverage=coverage_i,
                                                inter=inter_i,
                                                history=history_i,
                                                mask=mask_i,
                                                stop=stop_i)
                            all_beams.append(new_beam)
                    final_beams = []
                    for beam in self.sort_beams(all_beams):
                        if beam.stop:
                            if steps >= config.min_dec_steps:
                                final_results.append(beam)
                        else:
                            final_beams.append(beam)
                        if len(final_beams) == config.args.beam_size or len(final_results) == config.args.beam_size:
                            break

                    steps += 1

                if len(final_results) == 0:
                    final_results = final_beams
                final_results = self.sort_beams(final_results)

            # decode role sums
            dec_h, dec_c = s_t_0  # 1 x 2*hidden_size
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()
            decoders = [self.model.user_decoder, self.model.agent_decoder]
            multi_beams = []
            for _ in range(config.args.beam_size):
                tmp_beams = []
                for _ in range(2):
                    tmp_beams.append(Beam(tokens=[self.vocab.token2idx('<START>')],
                                          log_probs=[0.0],
                                          state=(dec_h[0], dec_c[0]),
                                          context=c_t_0[0],
                                          coverage=(coverage_t_0[0] if config.is_coverage else None),
                                          inter=torch.zeros(config.hidden_dim).to(c_t_0),
                                          history=None,
                                          mask=None,
                                          stop=False,
                                          attn=[]))
                multi_beams.append(MultiBeam(tmp_beams[0], tmp_beams[1]))

            results = [[], []]
            steps = 0
            while steps < config.args.max_dec_steps and (len(results[0]) < config.args.beam_size or
                                                         len(results[1]) < config.args.beam_size):
                update_infos = []
                for i in range(2):
                    beams = [h.beams[i] for h in multi_beams]
                    latest_tokens = [h.latest_token for h in beams]
                    latest_tokens = [t if t < self.vocab.vocab_size else self.vocab.token2idx('[UNK]') \
                                     for t in latest_tokens]
                    y_t_1 = torch.LongTensor(latest_tokens)
                    y_t_1 = y_t_1.to(self.device)
                    all_state_h = []
                    all_state_c = []

                    all_context = []
                    all_inter = []
                    all_history = []

                    for h in beams:
                        state_h, state_c = h.state
                        all_state_h.append(state_h)
                        all_state_c.append(state_c)

                        all_context.append(h.context)
                        all_inter.append(h.inter)
                        all_history.append(h.history)

                    s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
                    c_t_1 = torch.stack(all_context, 0)
                    int_t_1 = torch.stack(all_inter, 0)

                    coverage_t_1 = None
                    if config.is_coverage:
                        all_coverage = []
                        for h in beams:
                            all_coverage.append(h.coverage)
                        coverage_t_1 = torch.stack(all_coverage, 0)

                    # inter mask & features
                    all_inter_feats = []
                    all_inter_masks = []
                    if i == 0:
                        k = 1
                        beams = [h.beams[k] for h in multi_beams]
                        for h in beams:
                            all_inter_feats.append(h.history)
                            all_inter_masks.append(h.mask)
                        if steps != 0:
                            all_inter_feats = torch.stack(all_inter_feats, 0)
                            all_inter_masks = torch.stack(all_inter_masks, 0)
                        else:
                            all_inter_feats = None
                    else:
                        k = 0
                        beams = [h.beams[k] for h in multi_beams]
                        for h in beams:
                            all_inter_feats.append(h.history)
                            all_inter_masks.append(h.mask)
                        if steps != 0:
                            all_inter_feats = torch.stack(all_inter_feats, 0)
                            all_inter_masks = torch.stack(all_inter_masks, 0)
                        else:
                            all_inter_feats = None

                    final_dist, s_t, c_t, attn_dist, p_gen, coverage_t, new_inter_feature, int_t, _, _ = decoders[i](y_t_1, s_t_1,
                                                                                                               encoder_outputs,
                                                                                                               encoder_feature,
                                                                                                               enc_padding_mask,
                                                                                                               c_t_1,
                                                                                                               extra_zeros,
                                                                                                               enc_batch_extend_vocab,
                                                                                                               coverage_t_1,
                                                                                                               steps,
                                                                                                               int_t_1,
                                                                                                               all_inter_masks,
                                                                                                               all_inter_feats,
                                                                                                               role_mask)
                    #print(self.get_ratio(attn_dist[0, :], role_mask[0, :]))
                    log_probs = torch.log(final_dist)
                    topk_log_probs, topk_ids = torch.topk(log_probs, config.args.beam_size * 2)

                    dec_h, dec_c = s_t
                    dec_h = dec_h.squeeze()
                    dec_c = dec_c.squeeze()
                    update_infos.append(
                        [beams, dec_h, dec_c, c_t, coverage_t, int_t, new_inter_feature, topk_ids, topk_log_probs, attn_dist])

                all_beams = []
                num_orig_beams = 1 if steps == 0 else len(multi_beams)
                for i in range(num_orig_beams):
                    for k in range(config.args.beam_size * 2):  # for each of the top 2*beam_size hyps:
                        new_multi_beam = []
                        for j in range(2):
                            h = multi_beams[i].beams[j]
                            state_i = (update_infos[j][1][i], update_infos[j][2][i])
                            context_i = update_infos[j][3][i]
                            coverage_i = (update_infos[j][4][i] if config.is_coverage else None)
                            inter_i = update_infos[j][5][i]
                            history_i = update_infos[j][6][i]
                            attn_i = update_infos[j][9][i]

                            if update_infos[j][7][i, k].item() == self.vocab.token2idx('<END>') or h.stop:
                                stop_i = True
                            else:
                                stop_i = False
                            if h.stop:
                                mask_i = torch.zeros(1).to(self.device)
                            else:
                                mask_i = torch.ones(1).to(self.device)
                            new_beam = h.extend(token=update_infos[j][7][i, k].item(),
                                                log_prob=update_infos[j][8][i, k].item(),
                                                state=state_i,
                                                context=context_i,
                                                coverage=coverage_i,
                                                inter=inter_i,
                                                history=history_i,
                                                mask=mask_i,
                                                stop=stop_i,
                                                attn=attn_i)
                            new_multi_beam.append(new_beam)
                        new_multi_beam = MultiBeam(new_multi_beam[0], new_multi_beam[1])
                        all_beams.append(new_multi_beam)

                multi_beams_list = [[], []]
                for i in range(2):
                    beams = [multi.beams[i] for multi in all_beams]
                    if len(results[i]) < config.args.beam_size:
                        for beam in self.sort_beams(beams):
                            if beam.stop:
                                if steps >= config.min_dec_steps:
                                    results[i].append(beam)
                            else:
                                multi_beams_list[i].append(beam)
                            if len(multi_beams_list[i]) == config.args.beam_size:
                                break
                            if len(results[i]) == config.args.beam_size:
                                multi_beams_list[i] = results[i]
                                break
                    else:
                        multi_beams_list[i] = results[i]
                    # align results
                    if len(results[i]) != 0:
                        max_len = max([s.history.shape[0] for s in results[i]])
                        for j in range(len(results[i])):
                            if results[i][j].history.shape[0] < max_len:
                                results[i][j] = results[i][j].padding(max_len)

                multi_beams = []
                for i in range(len(multi_beams_list[0])):
                    multi_beams.append(MultiBeam(multi_beams_list[0][i], multi_beams_list[1][i]))

                steps += 1

            beams_sorted = [[], []]
            for i in range(2):
                if len(results[i]) == 0:
                    results[i] = [multi.beams[i] for multi in multi_beams]
                beams_sorted[i] = self.sort_beams(results[i])
            #exit()
            if self.args.no_final:
                return beams_sorted[0][0], beams_sorted[1][0]
            else:
                return final_results[0], beams_sorted[0][0], beams_sorted[1][0]

class PGNTrainer(object):
    def __init__(self, plot_path, gpu_id):
        self.args = config.args
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)

        self.device = torch.device('cuda', gpu_id) if gpu_id >= 0 else torch.device('cpu')
        self.gpu = True if gpu_id >= 0 else False
        self.model = None
        self.train_dataloader, self.train_data, self.vocab = get_train_dataloader(self.args.train_pth, self.args.max_seq_len,
                                                                      self.args.batch_size)
        self.val_dataloader, self.val_data = get_val_dataloader(self.args.val_pth, self.vocab, self.args.max_seq_len,
                                                                self.args.batch_size)
        self.val_decode_dataloader, _ = get_val_dataloader(self.args.val_pth, self.vocab, self.args.max_seq_len,
                                                                self.args.batch_size, mode='decode', beam_size=self.args.beam_size)
        self.test_dataloader, self.test_data = get_val_dataloader(self.args.test_pth, self.vocab, self.args.max_seq_len,
                                                                  self.args.batch_size, mode='decode', beam_size=self.args.beam_size)
        self.optimizer = None
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        if not os.path.exists(os.path.join(self.args.save_path, 'checkpoints')):
            os.makedirs(os.path.join(self.args.save_path, 'checkpoints'))
        self.summary_writer = SummaryWriter(plot_path)
        self.min_val_loss = 10
        self.embedding = torch.tensor(self.vocab.embedding, dtype=torch.float).to(self.device)


    def __init_model(self):
        self.model.to(self.device)

    def new_model(self):
        self.model = pgn.PGNModel()
        self.__init_model()


    def save_model(self, running_avg_loss, iter, val_loss):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'final_state_dict': self.model.final_decoder.state_dict(),
            'user_state_dict': self.model.user_decoder.state_dict(),
            'agent_state_dict': self.model.agent_decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(os.path.join(self.args.save_path, 'checkpoints'), '%.3f_model_%d' % (val_loss, iter))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = pgn.PGNModel(model_file_path, device=self.device, embedding=self.embedding)

        params = list(self.model.encoder.parameters()) + list(self.model.final_decoder.parameters()) + \
                 list(self.model.user_decoder.parameters()) + list(self.model.agent_decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)
        #self.optimizer = Adam(params)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if self.gpu:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t, coverage_zero, role_mask = \
            get_input_from_batch(batch, self.device, self.vocab)
        final_batch_input, final_padding_mask, max_final_len, final_lens_var, final_batch_output = \
            get_final_from_batch(batch, self.device, self.vocab)
        user_batch_input, user_padding_mask, max_user_len, user_lens_var, user_batch_output = \
            get_user_from_batch(batch, self.device, self.vocab)
        agent_batch_input, agent_padding_mask, max_agent_len, agent_lens_var, agent_batch_output = \
            get_agent_from_batch(batch, self.device, self.vocab)
        sum_inputs = [final_batch_input, user_batch_input, agent_batch_input]
        sum_paddings = [final_padding_mask, user_padding_mask, agent_padding_mask]
        max_sum_len = max_final_len
        max_lens_var = [final_lens_var, user_lens_var, agent_lens_var]
        sum_outputs = [final_batch_output, user_batch_output, agent_batch_output]
        decoders = [self.model.final_decoder, self.model.user_decoder, self.model.agent_decoder]

        self.optimizer.zero_grad()

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        enc_final = self.model.reduce_state(encoder_hidden)
        final_feature, user_feature, agent_feature = None, None, None
        decoder_feature = [final_feature, user_feature, agent_feature]
        inter_feature = [decoder_feature[0], decoder_feature[2], decoder_feature[1]]
        inter_mask = [final_padding_mask, agent_padding_mask, user_padding_mask]
        three_losses = [[], [], []]
        user_attn_loss, agent_attn_loss = [], []
        user_user, user_agent, agent_user, agent_agent = [], [], [], []
        s_t_1 = [(enc_final[0].clone(), enc_final[1].clone()), (enc_final[0].clone(), enc_final[1].clone()), (enc_final[0].clone(), enc_final[1].clone())]
        c_t_1 = [c_t.clone(), c_t.clone(), c_t.clone()]
        int_t_1 = [torch.zeros(c_t.shape[0], config.hidden_dim).to(c_t), torch.zeros(c_t.shape[0], config.hidden_dim).to(c_t), torch.zeros(c_t.shape[0], config.hidden_dim).to(c_t)]
        if config.is_coverage:
            coverage = [coverage_zero.clone(), coverage_zero.clone(), coverage_zero.clone()]
        else:
            coverage = [None, None, None]

        for di in range(min(max_sum_len, self.args.max_dec_steps)):
            attns = [[], []]
            for i, (dec_batch, dec_padding_mask, target_batch, decoder) in enumerate(zip(sum_inputs, sum_paddings, sum_outputs, decoders)):
                if self.args.no_final:
                    if i == 0:
                        continue
                y_t_1 = dec_batch[:, di]  # Teacher forcing
                final_dist, s_t_1[i], c_t_1[i], attn_dist, p_gen, next_coverage, new_inter_feature, int_t_1[i], attn_user, attn_agent = decoder(y_t_1, s_t_1[i],
                                                                                    encoder_outputs,
                                                                                    encoder_feature,
                                                                                    enc_padding_mask, c_t_1[i],
                                                                                    extra_zeros,
                                                                                    enc_batch_extend_vocab,
                                                                                    coverage[i], di, int_t_1[i], inter_mask[i], inter_feature[i], role_mask)
                target = target_batch[:, di]
                gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
                step_loss = -torch.log(gold_probs + config.eps)
                if config.is_coverage:
                    step_coverage_loss = torch.sum(torch.min(attn_dist, coverage[i]), 1)
                    step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                    coverage[i] = next_coverage

                step_mask = dec_padding_mask[:, di]
                step_loss = step_loss * step_mask
                three_losses[i].append(step_loss)
                if decoder_feature[i] is None:
                    decoder_feature[i] = new_inter_feature  # (bs, 1, hidden_dim)
                else:
                    decoder_feature[i] = torch.cat([decoder_feature[i], new_inter_feature], 1)  # (bs, k, hidden_dim)
                attns[0].append(attn_user)
                attns[1].append(attn_agent)
            inter_feature = [decoder_feature[0], decoder_feature[2], decoder_feature[1]]
            # user_loss = torch.sum(torch.nn.KLDivLoss(reduction='none')(F.log_softmax(attns[0][1], 1), F.softmax(attns[0][0], 1)), 1)
            # agent_loss = torch.sum(torch.nn.KLDivLoss(reduction='none')(F.log_softmax(attns[1][0], 1), F.softmax(attns[1][1], 1)), 1)
            # step_mask = sum_paddings[0][:, di] * sum_paddings[1][:, di]
            # user_attn_loss.append(user_loss * step_mask)
            # agent_attn_loss.append(agent_loss * step_mask)
            user_user.append(F.softmax(attns[0][0], 1) * sum_paddings[1][:, di].unsqueeze(-1))
            user_agent.append(F.softmax(attns[1][0], 1) * sum_paddings[1][:, di].unsqueeze(-1))
            agent_user.append(F.softmax(attns[0][1], 1) * sum_paddings[2][:, di].unsqueeze(-1))
            agent_agent.append(F.softmax(attns[1][1], 1) * sum_paddings[2][:, di].unsqueeze(-1))
        if self.args.no_final:
            for i in range(len(three_losses[1:])):
                sum_losses = torch.sum(torch.stack(three_losses[i+1], 1), 1)
                batch_avg_loss = sum_losses / max_lens_var[i+1]
                loss = torch.mean(batch_avg_loss)
                three_losses[i+1] = loss
            # user_sum_attn_loss = torch.sum(torch.stack(user_attn_loss, 1), 1)
            # agent_sum_attn_loss = torch.sum(torch.stack(agent_attn_loss, 1), 1)
            # lens = torch.min(torch.stack([max_lens_var[1], max_lens_var[2]], 1), 1)[0]
            # user_sum_attn_loss /= lens
            # agent_sum_attn_loss /= lens
            # user_sum_attn_loss = torch.mean(user_sum_attn_loss)
            # agent_sum_attn_loss = torch.mean(agent_sum_attn_loss)
            user_user = torch.sum(torch.stack(user_user, -1), -1) / max_lens_var[1].unsqueeze(-1)
            user_agent = torch.sum(torch.stack(user_agent, -1), -1) / max_lens_var[1].unsqueeze(-1)
            agent_user = torch.sum(torch.stack(agent_user, -1), -1) / max_lens_var[2].unsqueeze(-1)
            agent_agent = torch.sum(torch.stack(agent_agent, -1), -1) / max_lens_var[2].unsqueeze(-1)
            user_sum_attn_loss = torch.nn.KLDivLoss(reduction='batchmean')(torch.log(agent_user), user_user)
            agent_sum_attn_loss = torch.nn.KLDivLoss(reduction='batchmean')(torch.log(user_agent), agent_agent)
            loss = (three_losses[1] + three_losses[2]) / 2 + (user_sum_attn_loss + agent_sum_attn_loss) * config.args.kl_loss_weight / 2
            return_losses = [three_losses[1].item(), three_losses[2].item(), user_sum_attn_loss.item(), agent_sum_attn_loss.item()]
        else:
            for i in range(len(three_losses)):
                sum_losses = torch.sum(torch.stack(three_losses[i], 1), 1)
                batch_avg_loss = sum_losses / max_lens_var[i]
                loss = torch.mean(batch_avg_loss)
                three_losses[i] = loss
            loss = (three_losses[0] + three_losses[1] + three_losses[2]) / 3
            return_losses = [three_losses[0].item(), three_losses[1].item(), three_losses[2].item()]
        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.final_decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.user_decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.agent_decoder.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return return_losses

    def test_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t, coverage_zero, role_mask = \
            get_input_from_batch(batch, self.device, self.vocab)
        final_batch_input, final_padding_mask, max_final_len, final_lens_var, final_batch_output = \
            get_final_from_batch(batch, self.device, self.vocab)
        user_batch_input, user_padding_mask, max_user_len, user_lens_var, user_batch_output = \
            get_user_from_batch(batch, self.device, self.vocab)
        agent_batch_input, agent_padding_mask, max_agent_len, agent_lens_var, agent_batch_output = \
            get_agent_from_batch(batch, self.device, self.vocab)
        sum_inputs = [final_batch_input, user_batch_input, agent_batch_input]
        sum_paddings = [final_padding_mask, user_padding_mask, agent_padding_mask]
        max_sum_len = max_final_len
        max_lens_var = [final_lens_var, user_lens_var, agent_lens_var]
        sum_outputs = [final_batch_output, user_batch_output, agent_batch_output]
        decoders = [self.model.final_decoder, self.model.user_decoder, self.model.agent_decoder]

        with torch.no_grad():
            encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
            enc_final = self.model.reduce_state(encoder_hidden)
            final_feature, user_feature, agent_feature = None, None, None
            decoder_feature = [final_feature, user_feature, agent_feature]
            inter_feature = [decoder_feature[0], decoder_feature[2], decoder_feature[1]]
            inter_mask = [final_padding_mask, agent_padding_mask, user_padding_mask]
            three_losses = [[], [], []]
            user_attn_loss, agent_attn_loss = [], []
            user_user, user_agent, agent_user, agent_agent = [], [], [], []
            s_t_1 = [(enc_final[0].clone(), enc_final[1].clone()), (enc_final[0].clone(), enc_final[1].clone()),
                     (enc_final[0].clone(), enc_final[1].clone())]
            c_t_1 = [c_t.clone(), c_t.clone(), c_t.clone()]
            int_t_1 = [torch.zeros(c_t.shape[0], config.hidden_dim).to(c_t),
                       torch.zeros(c_t.shape[0], config.hidden_dim).to(c_t),
                       torch.zeros(c_t.shape[0], config.hidden_dim).to(c_t)]
            if config.is_coverage:
                coverage = [coverage_zero.clone(), coverage_zero.clone(), coverage_zero.clone()]
            else:
                coverage = [None, None, None]

            for di in range(min(max_sum_len, self.args.max_dec_steps)):
                attns = [[], []]
                for i, (dec_batch, dec_padding_mask, target_batch, decoder) in enumerate(
                        zip(sum_inputs, sum_paddings, sum_outputs, decoders)):
                    if self.args.no_final:
                        if i == 0:
                            continue
                    y_t_1 = dec_batch[:, di]  # Teacher forcing
                    final_dist, s_t_1[i], c_t_1[i], attn_dist, p_gen, next_coverage, new_inter_feature, int_t_1[
                        i], attn_user, attn_agent = decoder(y_t_1, s_t_1[i],
                                     encoder_outputs,
                                     encoder_feature,
                                     enc_padding_mask, c_t_1[i],
                                     extra_zeros,
                                     enc_batch_extend_vocab,
                                     coverage[i], di, int_t_1[i], inter_mask[i], inter_feature[i], role_mask)
                    target = target_batch[:, di]
                    gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
                    step_loss = -torch.log(gold_probs + config.eps)
                    if config.is_coverage:
                        step_coverage_loss = torch.sum(torch.min(attn_dist, coverage[i]), 1)
                        step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                        coverage[i] = next_coverage

                    step_mask = dec_padding_mask[:, di]
                    step_loss = step_loss * step_mask
                    three_losses[i].append(step_loss)
                    if decoder_feature[i] is None:
                        decoder_feature[i] = new_inter_feature  # (bs, 1, hidden_dim)
                    else:
                        decoder_feature[i] = torch.cat([decoder_feature[i], new_inter_feature], 1)  # (bs, k, hidden_dim)
                    attns[0].append(attn_user)
                    attns[1].append(attn_agent)
                inter_feature = [decoder_feature[0], decoder_feature[2], decoder_feature[1]]
                # user_loss = torch.sum(
                #     torch.nn.KLDivLoss(reduction='none')(F.log_softmax(attns[0][1], 1), F.softmax(attns[0][0], 1)), 1)
                # agent_loss = torch.sum(
                #     torch.nn.KLDivLoss(reduction='none')(F.log_softmax(attns[1][0], 1), F.softmax(attns[1][1], 1)), 1)
                # step_mask = sum_paddings[0][:, di] * sum_paddings[1][:, di]
                # user_attn_loss.append(user_loss * step_mask)
                # agent_attn_loss.append(agent_loss * step_mask)
                user_user.append(F.softmax(attns[0][0], 1) * sum_paddings[1][:, di].unsqueeze(-1))
                user_agent.append(F.softmax(attns[1][0], 1) * sum_paddings[1][:, di].unsqueeze(-1))
                agent_user.append(F.softmax(attns[0][1], 1) * sum_paddings[2][:, di].unsqueeze(-1))
                agent_agent.append(F.softmax(attns[1][1], 1) * sum_paddings[2][:, di].unsqueeze(-1))
            if self.args.no_final:
                for i in range(len(three_losses[1:])):
                    sum_losses = torch.sum(torch.stack(three_losses[i+1], 1), 1)
                    batch_avg_loss = sum_losses / max_lens_var[i+1]
                    loss = torch.mean(batch_avg_loss)
                    three_losses[i + 1] = loss
                # user_sum_attn_loss = torch.sum(torch.stack(user_attn_loss, 1), 1)
                # agent_sum_attn_loss = torch.sum(torch.stack(agent_attn_loss, 1), 1)
                # lens = torch.min(torch.stack([max_lens_var[1], max_lens_var[2]], 1), 1)[0]
                # user_sum_attn_loss /= lens
                # agent_sum_attn_loss /= lens
                # user_sum_attn_loss = torch.mean(user_sum_attn_loss)
                # agent_sum_attn_loss = torch.mean(agent_sum_attn_loss)
                user_user = torch.sum(torch.stack(user_user, -1), -1) / max_lens_var[1].unsqueeze(-1)
                user_agent = torch.sum(torch.stack(user_agent, -1), -1) / max_lens_var[1].unsqueeze(-1)
                agent_user = torch.sum(torch.stack(agent_user, -1), -1) / max_lens_var[2].unsqueeze(-1)
                agent_agent = torch.sum(torch.stack(agent_agent, -1), -1) / max_lens_var[2].unsqueeze(-1)
                user_sum_attn_loss = torch.nn.KLDivLoss(reduction='batchmean')(torch.log(agent_user), user_user)
                agent_sum_attn_loss = torch.nn.KLDivLoss(reduction='batchmean')(torch.log(user_agent), agent_agent)
                return [three_losses[1], three_losses[2], user_sum_attn_loss, agent_sum_attn_loss]
            else:
                for i in range(len(three_losses)):
                    sum_losses = torch.sum(torch.stack(three_losses[i], 1), 1)
                    batch_avg_loss = sum_losses / max_lens_var[i]
                    loss = torch.mean(batch_avg_loss)
                    three_losses[i] = loss
                return three_losses


    def calc_running_avg_loss(self, loss, running_avg_loss, summary_writer, step, decay=0.99):
        if running_avg_loss == 0:  # on the first iteration just take the loss
            running_avg_loss = loss
        else:
            running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
        running_avg_loss = min(running_avg_loss, 12)  # clip
        tag_name = 'running_avg_loss/decay=%f' % (decay)
        summary_writer.add_scalar(tag_name, running_avg_loss, step)
        return running_avg_loss

    def trainIters(self, epochs, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        if config.args.test_first:
            losses, save = self.validate(iter)
            self.test(save)
            self.save_model(running_avg_loss, iter, losses)
        for e in range(epochs):
            print("Epoch {}".format(e))
            for step, batch in enumerate(self.train_dataloader):
                self.model.train()
                if self.args.no_final:
                    user_loss, agent_loss, user_attn_loss, agent_attn_loss = self.train_one_batch(batch)
                    running_avg_loss = self.calc_running_avg_loss((user_loss + agent_loss + user_attn_loss + agent_attn_loss) / 2,
                                                                  running_avg_loss, self.summary_writer, iter)
                else:
                    final_loss, user_loss, agent_loss = self.train_one_batch(batch)
                    running_avg_loss = self.calc_running_avg_loss((final_loss + user_loss + agent_loss) / 3,
                                                                  running_avg_loss, self.summary_writer, iter)

                iter += 1

                if iter % self.args.log_freq == 0:
                    self.summary_writer.flush()
                print_interval = self.args.log_freq
                if iter % print_interval == 0:
                    if self.args.no_final:
                        print('steps %d, seconds for %d batch: %.2f, user_loss: %f, agent_loss: %f, user_attn_loss: %f, agent_attn_loss: %f' % (
                        iter, print_interval,
                        time.time() - start, user_loss, agent_loss, user_attn_loss, agent_attn_loss))
                    else:
                        print('steps %d, seconds for %d batch: %.2f , final_loss: %f, user_loss: %f, agent_loss: %f' % (iter, print_interval,
                                                                               time.time() - start, final_loss, user_loss, agent_loss))
                    start = time.time()
                if iter % self.args.val_freq == 0:
                    losses, save = self.validate(iter)
                    if save:
                        self.test(save)
                    self.save_model(running_avg_loss, iter, losses)
        print('min loss: ', self.min_val_loss)
        print('best step: ', self.best_step)

    def eval(self, model_file_path=None):
        _, _ = self.setup_train(model_file_path)
        beam_Search_processor = BeamSearch(self.model, self.vocab, self.test_dataloader, self.device, True,
                                           self.args)
        beam_Search_processor.decode(index=None)

    def predict(self, model_file_path, index):
        _, _ = self.setup_train(model_file_path)
        self.test(True, index)

    def validate(self, train_step):
        print('Begin Validating-------------------')
        final_losses, user_losses, agent_losses = 0., 0., 0.
        user_attn_losses, agent_attn_losses = 0., 0.
        batch_count = 0
        for step, batch in enumerate(self.val_dataloader):
            if self.args.no_final:
                user_loss, agent_loss, user_attn_loss, agent_attn_loss = self.test_one_batch(batch)
                user_losses += user_loss
                agent_losses += agent_loss
                user_attn_losses += user_attn_loss
                agent_attn_losses += agent_attn_loss
            else:
                final_loss, user_loss, agent_loss = self.test_one_batch(batch)
                final_losses += final_loss
                user_losses += user_loss
                agent_losses += agent_loss
            batch_count += 1
        final_losses /= batch_count
        user_losses /= batch_count
        agent_losses /= batch_count
        user_attn_losses /= batch_count
        agent_attn_losses /= batch_count
        if self.args.no_final:
            print(
                'validation user_loss: %f, agent_loss: %f, user_attn_loss: %f, agent_attn_loss: %f' % (user_losses, agent_losses, user_attn_losses, agent_attn_losses))
            avg_losses = (user_losses + agent_losses) / 2 #+ (user_attn_losses + user_attn_losses) * config.args.kl_loss_weight / 2
        else:
            print('validation final loss: %f, user_loss: %f, agent_loss: %f' % (final_losses, user_losses, agent_losses))
            avg_losses = (final_losses + user_losses + agent_losses) / 3
        if avg_losses < self.min_val_loss:
            self.min_val_loss = avg_losses
            self.best_step = train_step
            save = True
        else:
            save = False
        # print('Begin Decoding-------------------')
        # beam_Search_processor = BeamSearch(self.model, self.vocab, self.val_decode_dataloader, self.device, False, self.args)
        # beam_Search_processor.decode()
        return avg_losses, save

    def test(self, save, index=None):
        print('Begin Testing-------------------')
        beam_Search_processor = BeamSearch(self.model, self.vocab, self.test_dataloader, self.device, save, self.args)
        #beam_Search_processor = BeamSearch(self.model, self.vocab, self.val_decode_dataloader, self.device, save, self.args)
        beam_Search_processor.decode(index)


