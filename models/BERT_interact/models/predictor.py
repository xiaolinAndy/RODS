#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math

import torch
import tqdm

from tensorboardX import SummaryWriter

from others.utils import rouge_results_to_str, test_rouge, tile
from cal_rouge import cal_rouge_path
from translate.beam import GNMTGlobalScorer


def build_predictor(args, tokenizer, symbols, model, logger=None):
    scorer = GNMTGlobalScorer(args.alpha,length_penalty='wu')

    translator = Translator(args, model, tokenizer, symbols, global_scorer=scorer, logger=logger)
    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 args,
                 model,
                 vocab,
                 symbols,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'

        self.args = args
        self.model = model
        self.generator = self.model.generator
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']

        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.max_length = args.max_length

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        tensorboard_log_dir = args.model_path

        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def _build_target_tokens(self, pred):
        # vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_token:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t < len(self.vocab)]
        tokens = self.vocab.DecodeIds(tokens).split(' ')
        return tokens

    def from_batch(self, translation_batch, type):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
                len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, gold_score, src =  translation_batch["predictions"],translation_batch["scores"],translation_batch["gold_score"], batch.src
        if type == 'final':
            tgt_str = batch.tgt_str_final
        elif type == 'user':
            tgt_str = batch.tgt_str_user
        elif type == 'agent':
            tgt_str = batch.tgt_str_agent
        else:
            assert True

        translations = []
        for b in range(batch_size):
            pred_sents = self.vocab.convert_ids_to_tokens([int(n) for n in preds[b][0]])
            pred_sents = ' '.join(pred_sents).replace(' ##','')
            gold_sent = ' '.join(tgt_str[b].split())
            # translation = Translation(fname[b],src[:, b] if src is not None else None,
            #                           src_raw, pred_sents,
            #                           attn[b], pred_score[b], gold_sent,
            #                           gold_score[b])
            # src = self.spm.DecodeIds([int(t) for t in translation_batch['batch'].src[0][5] if int(t) != len(self.spm)])
            raw_src = [self.vocab.ids_to_tokens[int(t)] for t in src[b]][:500]
            raw_src = ' '.join(raw_src)
            translation = (pred_sents, gold_sent, raw_src)
            # translation = (pred_sents[0], gold_sent)
            translations.append(translation)

        return translations

    def translate(self,
                  data_iter, step,
                  attn_debug=False):

        self.model.eval()
        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')
        types = ['user', 'agent']
        self.gold_out_files, self.can_out_files = [], []
        gold_paths, can_paths = [], []
        for i in range(2):
            gold_paths.append(self.args.result_path + '.%d.%s.gold' % (step, types[i]))
            can_paths.append(self.args.result_path + '.%d.%s.candidate' % (step, types[i]))
            self.gold_out_files.append(codecs.open(gold_paths[i], 'w', 'utf-8'))
            self.can_out_files.append(codecs.open(can_paths[i], 'w', 'utf-8'))

        # pred_results, gold_results = [], []
        ct = 0
        with torch.no_grad():
            for batch in tqdm.tqdm(data_iter):
                if(self.args.recall_eval):
                    gold_tgt_len = batch.tgt.size(1)
                    self.min_length = gold_tgt_len + 20
                    self.max_length = gold_tgt_len + 60
                batch_data = self.translate_batch(batch)
                for i in range(2):
                    translations = self.from_batch(batch_data[i], types[i])

                    for trans in translations:
                        pred, gold, src = trans
                        pred_str = pred.replace('[unused99]', '').replace('[unused3]', '').replace('[PAD]', '').replace('[unused1]', '').replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]', '').strip()
                        gold_str = gold.strip()
                        if(self.args.recall_eval):
                            _pred_str = ''
                            gap = 1e3
                            for sent in pred_str.split('<q>'):
                                can_pred_str = _pred_str+ '<q>'+sent.strip()
                                can_gap = math.fabs(len(_pred_str.split())-len(gold_str.split()))
                                # if(can_gap>=gap):
                                if(len(can_pred_str.split())>=len(gold_str.split())+10):
                                    pred_str = _pred_str
                                    break
                                else:
                                    gap = can_gap
                                    _pred_str = can_pred_str



                            # pred_str = ' '.join(pred_str.split()[:len(gold_str.split())])
                        # self.raw_can_out_file.write(' '.join(pred).strip() + '\n')
                        # self.raw_gold_out_file.write(' '.join(gold).strip() + '\n')
                        self.can_out_files[i].write(pred_str + '\n')
                        self.gold_out_files[i].write(gold_str + '\n')
                        if i == 0:
                            self.src_out_file.write(src.strip() + '\n')
                        ct += 1
                    self.can_out_files[i].flush()
                    self.gold_out_files[i].flush()
                    if i == 0:
                        self.src_out_file.flush()
        self.src_out_file.close()
        for i in range(2):
            self.can_out_files[i].close()
            self.gold_out_files[i].close()
            cal_rouge_path(can_paths[i], gold_paths[i])

        # if (step != -1):
        #     rouges = self._report_rouge(gold_path, can_path)
        #     self.logger.info('Rouges at step %d \n%s' % (step, rouge_results_to_str(rouges)))
        #     if self.tensorboard_writer is not None:
        #         self.tensorboard_writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
        #         self.tensorboard_writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
        #         self.tensorboard_writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)

    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)
        return results_dict

    def translate_batch(self, batch, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(
                batch,
                self.max_length,
                min_length=self.min_length)

    def _fast_translate_batch(self,
                              batch,
                              max_length,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        #print(max_length)
        assert not self.dump_beam

        beam_size = self.beam_size
        batch_size = batch.batch_size
        src = batch.src
        segs = batch.segs
        mask_src = batch.mask_src
        role_mask = batch.role_mask

        src_features = self.model.bert(src, segs, mask_src)
        device = src_features.device
        decoders = [self.model.user_decoder, self.model.agent_decoder]
        results_full = []

        # decode role summaries
        dec_states_roles, src_features_p_roles = [], []
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        alive_seq_roles, topk_log_probs_roles, hypotheses_roles, results_roles = [], [], [], []
        end_states = []
        role_masks = []
        for k in range(2):
            dec_states_roles.append(decoders[k].init_decoder_state(src, src_features, with_cache=True))
            # Tile states and memory beam_size times.
            dec_states_roles[k].map_batch_fn(
                lambda state, dim: tile(state, beam_size, dim=dim))
            src_features_p_roles.append(tile(src_features, beam_size, dim=0))
            role_masks.append(tile(role_mask, beam_size, dim=0))
            alive_seq = torch.full(
                [batch_size * beam_size, 1],
                self.start_token,
                dtype=torch.long,
                device=device)
            alive_seq_roles.append(alive_seq)

            # Give full probability to the first beam on the first step.
            topk_log_probs = (
                torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                             device=device).repeat(batch_size))
            topk_log_probs_roles.append(topk_log_probs)

            # Structure that holds finished hypotheses.
            hypotheses = [[] for _ in range(batch_size)]  # noqa: F812
            hypotheses_roles.append(hypotheses)

            results = {}
            results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
            results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
            results["gold_score"] = [0] * batch_size
            results["batch"] = batch
            results_roles.append(results)
            end_states.append(torch.zeros(batch_size).bool().to(device))

        for step in range(max_length):
            # print(step)
            # print('--------------------')
            # print(end_states)
            # print(hypotheses_roles)
            decoder_input_user = alive_seq_roles[0][:, -1].view(1, -1)
            decoder_input_agent = alive_seq_roles[1][:, -1].view(1, -1)
            # Decoder forward.
            decoder_input_user = decoder_input_user.transpose(0,1)
            decoder_input_agent = decoder_input_agent.transpose(0, 1)
            user_outputs, agent_outputs, state_user, state_agent, _, _, _, _ = self.model.role_decoder(decoder_input_user, decoder_input_agent,
                                                                                     src_features_p_roles[0], dec_states_roles[0],
                                                                                     dec_states_roles[1], self.args.merge,
                                                                                     self.args.inter_weight, role_masks[0], step=step)

            # Generator forward.
            outputs = [user_outputs, agent_outputs]
            states = [state_user, state_agent]
            is_finished_roles, topk_score_roles = [], []
            batch_indexes = []
            for k in range(2):
                log_probs = self.generator.forward(outputs[k].transpose(0,1).squeeze(0))
                vocab_size = log_probs.size(-1)

                if step < min_length:
                    log_probs[:, self.end_token] = -1e20

                # Multiply probs by the beam probability.
                #print(topk_log_probs_roles[k].shape)
                log_probs += topk_log_probs_roles[k].view(-1).unsqueeze(1)  # [batch*beam, vocab]

                alpha = self.global_scorer.alpha
                length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

                # Flatten probs into a list of possibilities.
                curr_scores = log_probs / length_penalty  # [batch*beam, vocab]

                if(self.args.block_trigram):
                    cur_len = alive_seq_roles[k].size(1)
                    if(cur_len>5):
                        for i in range(alive_seq_roles[k].size(0)):
                            fail = False
                            words = [int(w) for w in alive_seq_roles[k][i]]
                            words = [self.vocab.ids_to_tokens[w] for w in words]
                            words = ' '.join(words).replace(' ##','').split()
                            if(len(words)<=5):
                                continue
                            trigrams = [(words[i-2],words[i-1],words[i],words[i+1], words[i+2]) for i in range(2,len(words)-2)]
                            trigram = tuple(trigrams[-1])
                            if trigram in trigrams[:-1]:
                                fail = True
                            if fail:
                                curr_scores[i] = -10e20

                curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
                topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

                # Recover log probs.
                topk_log_probs = topk_scores * length_penalty
                topk_log_probs_roles[k] = topk_log_probs

                # Resolve beam origin and true word ids.
                topk_beam_index = topk_ids.true_divide(vocab_size)
                topk_ids = topk_ids.fmod(vocab_size)

                # Map beam_index to batch_index in the flat representation.
                batch_index = (
                        topk_beam_index
                        + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
                batch_indexes.append(batch_index)
                select_indices = batch_index.view(-1)

                # Append last prediction.
                alive_seq_roles[k] = torch.cat(
                    [alive_seq_roles[k].to(torch.long).index_select(0, select_indices.to(torch.long)),
                     topk_ids.to(torch.long).view(-1, 1)], -1)

                is_finished = topk_ids.eq(self.end_token)  # [batch, beam]
                if step + 1 == max_length:
                    is_finished.fill_(1)
                # End condition is top beam is finished.
                end_states[k] = is_finished[:, 0].eq(1) | end_states[k]
                is_finished_roles.append(is_finished)
                topk_score_roles.append(topk_scores)

            end_condition = end_states[0] & end_states[1]
            non_finished = end_condition.eq(0).nonzero().view(-1)
            for k in range(2):
                # Save finished hypotheses.
                predictions = alive_seq_roles[k].view(-1, beam_size, alive_seq_roles[k].size(-1))
                for i in range(is_finished_roles[k].size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished_roles[k][i].fill_(1)
                    finished_hyp = is_finished_roles[k][i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses_roles[k][b].append((
                            topk_score_roles[k][i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses_roles[k][b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results_roles[k]["scores"][b].append(score)
                        results_roles[k]["predictions"][b].append(pred)
                # Remove finished batches for the next step.
                topk_log_probs_roles[k] = topk_log_probs_roles[k].index_select(0, non_finished)
                alive_seq_roles[k] = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq_roles[k].size(-1))

            #print(non_finished)
            # If all sentences are translated, no need to go further.
            if len(non_finished) == 0:
                break
            batch_offset = batch_offset.index_select(0, non_finished)
            # Reorder states.
            for k in range(2):
                batch_indexes[k] = batch_indexes[k].index_select(0, non_finished)
                end_states[k] = end_states[k].index_select(0, non_finished)
                #print(batch_indexes[k].to(torch.long))
                select_indices = batch_indexes[k].view(-1)
                src_features_p_roles[k] = src_features_p_roles[k].index_select(0, select_indices.to(torch.long))
                role_masks[k] = role_masks[k].index_select(0, select_indices.to(torch.long))
                dec_states_roles[k].map_batch_fn(
                    lambda state, dim: state.index_select(dim, select_indices.to(torch.long)))

        for k in range(2):
            #print(results_roles[k])
            results_full.append(results_roles[k])

        return results_full


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, fname, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.fname = fname
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
