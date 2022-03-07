import re
import os
import json
import files2rouge
import bert_score
import numpy as np
import math
from moverscore_v2 import get_idf_dict, word_mover_score
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

model_name = ['PGN', 'PGN_multi', 'PGN_enc', 'PGN_dec', 'PGN_both', 'bert_abs', 'bert_abs_multi', 'bert_abs_enc', 'bert_abs_dec', 'bert_abs_both']
modes = ['user', 'agent']
auto_metrics = ['rouge_1', 'rouge_2', 'rouge_l', 'bleu', 'bertscore', 'moverscore']

def get_sents_str(file_path):
    sents = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = re.sub(' ', '', line)
            line = re.sub('<q>', '', line)
            sents.append(line)
    return sents

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

def read_rouge_score(name):
    with open(name, 'r') as f:
        lines = f.readlines()
    r1 = lines[3][21:28]
    r2 = lines[7][21:28]
    rl = lines[11][21:28]
    rl_p = lines[10][21:28]
    rl_r = lines[9][21:28]
    return [float(r1), float(r2), float(rl), float(rl_p), float(rl_r)]

def calculate(pred_file, ref_file, mode, model):
    refs = get_sents_str(ref_file)
    preds = get_sents_str(pred_file)

    scores = []
    #get rouge scores
    print('Running ROUGE for ' + mode + ' ' + model + '-----------------------------')
    pred_ids, ref_ids = [], []
    for ref, pred in zip(refs, preds):
        ref_id, pred_id = change_word2id_split(ref, pred)
        pred_ids.append(pred_id)
        ref_ids.append(ref_id)
    with open('ref_ids.txt', 'w') as f:
        for ref_id in ref_ids:
            f.write(ref_id + '\n')
    with open('pred_ids.txt', 'w') as f:
        for pred_id in pred_ids:
            f.write(pred_id + '\n')
    os.system('files2rouge ref_ids.txt pred_ids.txt -s rouge.txt -e 0')
    rouge_scores = read_rouge_score('rouge.txt')
    scores.append(rouge_scores[0])
    scores.append(rouge_scores[1])
    scores.append(rouge_scores[2])

    # get bleu scores
    #print('Running BLEU for ' + mode + ' ' + model + '-----------------------------')
    bleu_preds, bleu_refs = [], []
    bleu_scores = []
    for ref, pred in zip(refs, preds):
        bleu_preds.append(list(pred))
        bleu_refs.append([list(ref)])
        bleu_score = sentence_bleu([list(ref)], list(pred))
        bleu_scores.append(bleu_score)
    bleu_score = corpus_bleu(bleu_refs, bleu_preds)
    #bleu_score = np.mean(np.array(bleu_scores))
    scores.append(bleu_score)

    # run bertscore
    print('Running BERTScore for ' + mode + ' ' + model + '-----------------------------')
    prec, rec, f1 = bert_score.score(preds, refs, lang='zh')
    scores.append(f1.numpy().mean().item())
    #
    # run moverscore
    #print('Running MoverScore for ' + mode + ' ' + model + '-----------------------------')
    idf_dict_hyp = get_idf_dict(preds)
    idf_dict_ref = get_idf_dict(refs)
    mover_scores = word_mover_score(refs, preds, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=2, batch_size=16)
    scores.append(np.array(mover_scores).mean().item())

    print('Model: %s, Mode: %s' % (model, mode))
    for i in range(len(auto_metrics)):
        print('%s: %.4f' % (auto_metrics[i], scores[i]))

    score_dict = {}
    for i, metric in enumerate(auto_metrics):
        score_dict[metric] = scores[i]
    return score_dict


if __name__ == '__main__':
    results = {}
    dataset_names = ['CSDS', 'MC']
    for dataset in dataset_names:
        results[dataset] = {}
        for model in model_name:
            results[model] = {}
            for mode in modes:
                pred_file = '../results/' + dataset + '/' + model + '_' + mode + '_pred.txt'
                if model[:3] == 'PGN':
                    ref_file = '../results/' + dataset + '/' + 'PGN_' + mode + '_ref.txt'
                else:
                    ref_file = '../results/' + dataset + '/' + model + '_' + mode + '_ref.txt'
                scores = calculate(pred_file, ref_file, mode, model)
                results[model][mode] = scores
