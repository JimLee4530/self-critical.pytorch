#coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import time
import os, sys
from six.moves import cPickle
import argparse
# import opts
import models_ensemble
from dataloadertest import *
import eval_utils
import misc.utils as utils
from misc.rewards import init_cider_scorer, get_self_critical_reward


parser = argparse.ArgumentParser()
# Data input settings
parser.add_argument('--gpu_id', type=str, default='0',
                help='gpu id')
parser.add_argument('--tensorboard', type=str, default=None,
                help='tensorboard')
parser.add_argument('--input_json', type=str, default='./data/ImageCaption.json',
                help='path to the json file containing additional info and vocab')
parser.add_argument('--input_fc_dir', type=str, default='/home/j1ml3e/data',
                help='path to the directory containing the preprocessed fc feats')
parser.add_argument('--input_att_dir', type=str, default='/home/j1ml3e/data',
                help='path to the directory containing the preprocessed att feats')
parser.add_argument('--input_label_h5', type=str, default='data/ImageCaption_label.h5',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--start_from', type=str, default=None,
                help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                    'infos.pkl'         : configuration;
                    'checkpoint'        : paths to model file(s) (created by tf).
                                          Note: this file contains absolute paths, be careful when moving files around;
                    'model.ckpt-*'      : file(s) with model definition (created by tf)
                """)
parser.add_argument('--cached_tokens', type=str, default='coco-train-idxs',
                help='Cached token file for calculating cider score during self critical training.')

# Model settings
parser.add_argument('--caption_model', type=str, default="topdown",
                help='show_tell, show_attend_tell, all_img, fc, att2in, att2in2, adaatt, adaattmo, topdown')

# ***********************************
parser.add_argument('--rnn_size', type=int, default=512,
                help='size of the rnn in number of hidden nodes in each layer')

parser.add_argument('--rnn_size1', type=int, default=512,
                help='size of the rnn in number of hidden nodes in each layer')
parser.add_argument('--rnn_size2', type=int, default=512,
                help='size of the rnn in number of hidden nodes in each layer')
# *********************
parser.add_argument('--num_layers', type=int, default=1,
                help='number of layers in the RNN')
parser.add_argument('--rnn_type', type=str, default='lstm',
                help='rnn, gru, or lstm')
parser.add_argument('--input_encoding_size', type=int, default=512,
                help='the encoding size of each token in the vocabulary, and the image.')
parser.add_argument('--att_hid_size', type=int, default=512,
                help='the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
parser.add_argument('--fc_feat_size', type=int, default=2048,
                help='2048 for resnet, 4096 for vgg')
parser.add_argument('--att_feat_size', type=int, default=2048,
                help='2048 for resnet, 512 for vgg')

# Optimization: General
parser.add_argument('--max_epochs', type=int, default=-1,
                help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64,
                help='minibatch size')
parser.add_argument('--grad_clip', type=float, default=0.1, #5.,
                help='clip gradients at this value')
parser.add_argument('--drop_prob_lm', type=float, default=0.1,
                help='strength of dropout in the Language Model RNN')
parser.add_argument('--dropout1', type=float, default=0.1,
                help='strength of dropout in the Language Model RNN')
parser.add_argument('--dropout2', type=float, default=0.1,
                help='strength of dropout in the Language Model RNN')
parser.add_argument('--self_critical_after', type=int, default=30,
                help='After what epoch do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
parser.add_argument('--seq_per_img', type=int, default=1,
                help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
parser.add_argument('--beam_size', type=int, default=3,
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

#Optimization: for the Language Model
parser.add_argument('--optim', type=str, default='adam',
                help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
parser.add_argument('--learning_rate', type=float, default=5e-5,
                help='learning rate')
parser.add_argument('--learning_rate_decay_start', type=int, default=-1, 
                help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
parser.add_argument('--learning_rate_decay_every', type=int, default=3, 
                help='every how many iterations thereafter to drop LR?(in epoch)')
parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8, 
                help='every how many iterations thereafter to drop LR?(in epoch)')
parser.add_argument('--optim_alpha', type=float, default=0.9,
                help='alpha for adam')
parser.add_argument('--optim_beta', type=float, default=0.999,
                help='beta used for adam')
parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                help='epsilon that goes into denominator for smoothing')
parser.add_argument('--weight_decay', type=float, default=0,
                help='weight_decay')

parser.add_argument('--scheduled_sampling_start', type=int, default=-1, 
                help='at what iteration to start decay gt probability')
parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5, 
                help='every how many iterations thereafter to gt probability')
parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05, 
                help='How much to update the prob')
parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25, 
                help='Maximum scheduled sampling prob.')


# Evaluation/Checkpointing
parser.add_argument('--raw_val_anno_path', type=str, default='/data/common/data/ai_challenger_caption/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json',
                help='raw_val_anno_path')
parser.add_argument('--val_ref_path', type=str, default='./data/val_ref.json',
                help='val_ref_path')
parser.add_argument('--val_images_use', type=int, default=30000,
                help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
parser.add_argument('--save_checkpoint_every', type=int, default=6000,
                help='how often to save a model checkpoint (in iterations)?')
parser.add_argument('--checkpoint_path', type=str, default='save',
                help='directory to store checkpointed models')
parser.add_argument('--language_eval', type=int, default=1,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--losses_log_every', type=int, default=25,
                help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')       
parser.add_argument('--load_best_score', type=int, default=1,
                help='Do we load previous best score when resuming training.')       

# misc
parser.add_argument('--id', type=str, default='topdown',
                help='an id identifying this run/job. used in cross-val and appended when writing progress files')
parser.add_argument('--train_only', type=int, default=0,
                help='if true then use 80k, else use 110k')

parser.add_argument('--seq_length', type=int, default=22,
                help='seq_length.')  

# parser.add_argument('--test_image_feat', type=str, default='/home/j1ml3e/data/test_b_20171120/faster_rcnn_pool5_features',
#                 help='test_image_feat.')
parser.add_argument('--test_image_feat', type=str, default='/home/j1ml3e/data/validation_20170910/faster_rcnn_pool5_features',
                help='validation_20170910_image_feat.')

# build MFB
parser.add_argument('--MFB_FACTOR_NUM', type=int, default=5,
                    help='MFB_FACTOR_NUM')
parser.add_argument('--MFB_OUT_DIM', type=int, default=1000,
                    help='MFB_FACTOR_NUM')
parser.add_argument('--MFB_DROPOUT_RATIO', type=float, default=0.1,
                    help='MFB_DROPOUT_RATIO')

parser.add_argument('--use_maxout', type=bool, default=False,
                    help='use_maxout')

opt = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id




from dataloadertest import DataLoader
# from models.TopDown import TopDownModel
from torch.autograd import Variable

caption_models = ['adaatt', 'top_down_adaatt'] # , 'top_down_mfb', 'topdown',   , 'adaatt', , 'top_down_mfb2'
model_paths = ['log_adaatt_39_2', 'log_top_down_adaatt_12010213'] # , 'log_top_down_mfb_32_512_0.1', 'log_top_down_34_512_0.1', , 'log_adaatt_train_val', , 'log_top_down_mfb2_41'



def beam_search(logprobs, model_list, opt):
    # args are the miscelleous inputs to the core in addition to embedded word and state
    # kwargs only accept opt

    def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state_list):
        # INPUTS:
        # logprobsf: probabilities augmented after diversity
        # beam_size: obvious
        # t        : time instant
        # beam_seq : tensor contanining the beams
        # beam_seq_logprobs: tensor contanining the beam logprobs
        # beam_logprobs_sum: tensor contanining joint logprobs
        # OUPUTS:
        # beam_seq : tensor containing the word indices of the decoded captions
        # beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
        # beam_logprobs_sum : joint log-probability of each beam

        ys, ix = torch.sort(logprobsf, 1, True)
        candidates = []
        cols = min(beam_size, ys.size(1))
        rows = beam_size
        if t == 0:
            rows = 1
        for c in range(cols):  # for each column (word, essentially)
            for q in range(rows):  # for each beam expansion
                # compute logprob of expanding beam q with word in (sorted) position c
                local_logprob = ys[q, c]
                candidate_logprob = beam_logprobs_sum[q] + local_logprob
                candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': local_logprob})
        candidates = sorted(candidates, key=lambda x: -x['p'])

        new_state_list = [[tmp.clone() for tmp in state] for state in state_list]
        # beam_seq_prev, beam_seq_logprobs_prev
        if t >= 1:
            # we''ll need these as reference when we fork beams around
            beam_seq_prev = beam_seq[:t].clone()
            beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
        for vix in range(beam_size):
            v = candidates[vix]
            # fork beam index q into index vix
            if t >= 1:
                beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
            # rearrange recurrent states
            for i in range(len(state_list)):
                for state_ix in range(len(new_state_list[i])):
                    #  copy over state in previous beam q to new beam at vix
                    new_state_list[i][state_ix][:, vix] = state_list[i][state_ix][:, v['q']]  # dimension one is time step
            # append new end terminal at the end of this beam
            beam_seq[t, vix] = v['c']  # c'th word is the continuation
            beam_seq_logprobs[t, vix] = v['r']  # the raw logprob here
            beam_logprobs_sum[vix] = v['p']  # the new (sum) logprob along this beam
        state_list = new_state_list
        return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state_list, candidates

    # start beam search
    beam_size = opt.get('beam_size', 10)
    seq_length = opt.get('seq_length')

    beam_seq = torch.LongTensor(seq_length, beam_size).zero_()
    beam_seq_logprobs = torch.FloatTensor(seq_length, beam_size).zero_()
    beam_logprobs_sum = torch.zeros(beam_size)  # running sum of logprobs for each beam
    done_beams = []

    state_list = []
    for model in model_list:
        state = model.mystate
        state_list.append(state)

    for t in range(seq_length):

        """pem a beam merge. that is,
        for every previous beam we now many new possibilities to branch out
        we need to resort our beams to maintain the loop invariant of keeping
        the top beam_size most likely sequences."""
        logprobsf = logprobs.data.float()  # lets go to CPU for more efficiency in indexing operations
        # suppress UNK tokens in the decoding
        logprobsf[:, logprobsf.size(1) - 1] = logprobsf[:, logprobsf.size(1) - 1] - 1000

        beam_seq, \
        beam_seq_logprobs, \
        beam_logprobs_sum, \
        state_list, \
        candidates_divm = beam_step(logprobsf,
                                    beam_size,
                                    t,
                                    beam_seq,
                                    beam_seq_logprobs,
                                    beam_logprobs_sum,
                                    state_list)

        for vix in range(beam_size):
            # if time's up... or if end token is reached then copy beams
            if beam_seq[t, vix] == 0 or t == seq_length - 1:
                final_beam = {
                    'seq': beam_seq[:, vix].clone(),
                    'logps': beam_seq_logprobs[:, vix].clone(),
                    'p': beam_logprobs_sum[vix]
                }
                done_beams.append(final_beam)
                # don't continue beams from finished sequences
                beam_logprobs_sum[vix] = -1000

        # encode as vectors
        it = beam_seq[t]
        tmp_state_list = []

        probs = None
        for i, model in enumerate(model_list):
            logprobs, state = model.get_logprobs_state(Variable(it).cuda(), model.tmp_att_feats, model.tmp_p_att_feats, model.tmp_avg_att_feats, state_list[i])
            tmp_state_list.append(state)
            if probs is None:
                probs = Variable(logprobs.data.new(logprobs.size()).zero_())
            probs += torch.exp(logprobs)
        state_list = tmp_state_list
        probs = probs.div(len(model_list))
        logprobs = torch.log(probs)


    done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
    return done_beams


def sample_beam(model_list, fc_feats, att_feats, num_bbox, opt):
    beam_size = opt.get('beam_size')
    vocab_size = opt.get('vocab_size')
    seq_length = opt.get('seq_length')

    batch_size = att_feats.size(0)

    assert beam_size <= vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
    seq = torch.LongTensor(seq_length, batch_size).zero_()
    seqLogprobs = torch.FloatTensor(seq_length, batch_size)

    done_beams = [[] for _ in range(batch_size)]
    for k in range(batch_size):
        probs = None
        for i, model in enumerate(model_list):
            state, t_logprobs = model.single_sample_beam(k, fc_feats, att_feats, num_bbox, opt)
            if probs is None:
                probs = Variable(t_logprobs.data.new(t_logprobs.size()).zero_())
            probs += torch.exp(t_logprobs)

        probs = probs.div(len(model_list))
        logprobs = torch.log(probs)
        done_beams[k] = beam_search(logprobs, model_list, opt=opt)
        seq[:, k] = done_beams[k][0]['seq']  # the first beam has highest cumulative score
        seqLogprobs[:, k] = done_beams[k][0]['logps']

    return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)




def test(opt):
    eval_kwargs = vars(opt)
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    val_ref_path = eval_kwargs.get('val_ref_path', '')
    raw_val_anno_path = eval_kwargs.get('raw_val_anno_path', '')
    kwargs = {'num_workers': 5, 'pin_memory': True}
    dataset = DataLoader(opt)
    opt.vocab_size = dataset.vocab_size
    eval_kwargs['vocab_size'] = dataset.vocab_size

    loader = torch.utils.data.DataLoader( dataset, 
                        batch_size=64, shuffle=False, **kwargs)

    # 模型融合参数
    seq_length = opt.seq_length



    imgids = dataset.imgids
    data_iter = iter(loader)
    # net = TopDownModel(opt)

    # if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"save/optimizer.pth")):
    #     net.load_state_dict(torch.load(os.path.join(opt.start_from, 'save/optimizer.pth')))

    model_list = []
    for i, m in enumerate(caption_models):
        opt.caption_model = m
        model = models_ensemble.setup(opt)
        model.load_state_dict(torch.load(model_paths[i]+'/model-best.pth'))
        model.cuda()
        model.eval()
        model_list.append(model)
    # model.set_mode('test')

    result = {}
    results = []
    num_images = len(dataset)

    for i, (fc_feats, att_feats, num_bbox) in enumerate(data_iter):
        fc_feats = Variable(fc_feats).cuda()
        att_feats = Variable(att_feats).cuda()
        num_bbox = Variable(num_bbox).cuda()
        num_bbox = num_bbox.view(-1)
        seq, _ = sample_beam(model_list, fc_feats, att_feats, num_bbox, eval_kwargs)

        sents = utils.decode_sequence(dataset.get_vocab(), seq)

        print("{}%".format((i*64.0)/300))
        for j in range(att_feats.size(0)):
            iid = imgids[(i * 64) + j]
            sent = sents[j]
            result = {'image_id':iid, 'caption':sent}
            results.append(result)
        # print("{}%".format((i*64.0)/30000)*100)
        # break
        sys.stdout.write('%.2f percent'%(i/float(num_images)))
        sys.stdout.flush()
    json.dump(results, open('results_ensemble_4_models_test_new.json', 'w'))
        
test(opt)








































