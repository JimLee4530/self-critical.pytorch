from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
from caption_eval.run_evaluations import evaluate
from caption_eval.val_reference import make_ref

def language_eval(preds, model_id, split, raw_val_anno_path, val_ref_path):
    # print( preds)
    # sys.exit(0)
    if not os.path.exists(val_ref_path):
        val_ref = make_ref(raw_val_anno_path, val_ref_path)
    else:
        val_ref = json.load(open(val_ref_path, 'r'))

    print(len(preds))
    print(len(val_ref))
    out = evaluate(preds, val_ref)
    if not os.path.exists('eval_results'):
        os.mkdir('eval_results')

    cache_path = os.path.join('eval_results/', model_id + '_' + split + 'chenchao.json')
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out}, outfile)
    return out


def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    # num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    val_ref_path = eval_kwargs.get('val_ref_path', '')
    raw_val_anno_path = eval_kwargs.get('raw_val_anno_path', '')
    num_images = loader.get_img_num(split)

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size


        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['num_bbox'][np.arange(loader.batch_size) * loader.seq_per_img]]
        tmp = [Variable(torch.from_numpy(_).float(), volatile=True).cuda() for _ in tmp]
        fc_feats, att_feats, num_bbox = tmp
        # forward the model to also get generated samples for each image
        seq, _ = model.sample(fc_feats, att_feats, num_bbox, eval_kwargs)
        # seq, _ = model.sample(fc_feats, att_feats, eval_kwargs)
        
        #set_trace()
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            # DEBUG
            entry = {'image_id': data['infos'][k]['id'].split('.')[0], 'caption': sent.replace(' ', '')} # .replace(' ', '')
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            # if verbose:
                # print('image %s: %s' %(entry['image_id'], entry['caption']))
        json.dump(predictions, open('val_res.json','w'))
        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            # print(ix0-1)
            print('evaluating validation preformance... %d/%d' %(ix0 - 1, ix1))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(predictions, eval_kwargs['id'], split, raw_val_anno_path, val_ref_path)

    # Switch back to training mode
    model.train()
    return predictions, lang_stats
