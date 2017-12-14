from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--gpu_id', type=str, default='1',
                    help='gpu id')
parser.add_argument('--model', type=str, default='save/model.pth',
                help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                help='path to infos to evaluate')
parser.add_argument('--caption_model', type=str, default="topdown",
                    help='show_tell, show_attend_tell, all_img, fc, att2in, att2in2, adaatt, adaattmo, topdown')
parser.add_argument('--start_from', type=str, default=None,
                    help="""continue training from saved model at this path. Path must contain files saved by previous training process: 
                        'infos.pkl'         : configuration;
                        'checkpoint'        : paths to model file(s) (created by tf).
                                              Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'      : file(s) with model definition (created by tf)
                    """)
parser.add_argument('--raw_val_anno_path', type=str, default='/data/common/data/ai_challenger_caption/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json',
                    help='raw_val_anno_path')
parser.add_argument('--val_ref_path', type=str, default='./data/val_ref.json',
                    help='val_ref_path')


parser.add_argument('--seq_per_img', type=int, default=5,
                    help='number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
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
parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                help='strength of dropout in the Language Model RNN')
parser.add_argument('--dropout1', type=float, default=0.5,
                help='strength of dropout in the Language Model RNN')
parser.add_argument('--dropout2', type=float, default=0.5,
                help='strength of dropout in the Language Model RNN')

# Basic options
parser.add_argument('--batch_size', type=int, default=1,
                help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=1,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=0,
                help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=0,
                help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--beam_size', type=int, default=1,
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--temperature', type=float, default=1.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='', 
                help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='', 
                help='In case the image paths have to be preprended with a root path to an image folder')


# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_fc_dir', type=str, default='data/ai_challenger_caption/image_feat',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_att_dir', type=str, default='data/ai_challenger_caption/image_feat',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_label_h5', type=str, default='data/ImageCaption_label.h5',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='./data/ImageCaption.json', 
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')


parser.add_argument('--split', type=str, default='val', 
                help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='', 
                help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
# misc
parser.add_argument('--id', type=str, default='topdown', 
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


# # Load infos
# with open(opt.infos_path) as f:
#     infos = cPickle.load(f)

# # override and collect parameters
# if len(opt.input_fc_dir) == 0:
#     opt.input_fc_dir = infos['opt'].input_fc_dir
#     opt.input_att_dir = infos['opt'].input_att_dir
#     opt.input_label_h5 = infos['opt'].input_label_h5
# if len(opt.input_json) == 0:
#     opt.input_json = infos['opt'].input_json
# if opt.batch_size == 0:
#     opt.batch_size = infos['opt'].batch_size
# if len(opt.id) == 0:
#     opt.id = infos['opt'].id
# ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval"]
# for k in vars(infos['opt']).keys():
#     if k not in ignore:
#         if k in vars(opt):
#             assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
#         else:
#             vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

# vocab = infos['vocab'] # ix -> word mapping
loader = DataLoader(opt)
opt.vocab_size = loader.vocab_size
opt.seq_length = loader.seq_length


# Setup the model
model = models.setup(opt)
model.load_state_dict(torch.load(opt.model))
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()
# Create the Data Loader instance
# if len(opt.image_folder) == 0:
#   loader = DataLoader(opt)
# else:
#   loader = DataLoaderRaw({'folder_path': opt.image_folder, 
#                             'coco_json': opt.coco_json,
#                             'batch_size': opt.batch_size,
#                             'cnn_model': opt.cnn_model})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
# loader.ix_to_word = infos['vocab']


# Set sample options
split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, 
    vars(opt))

# print('loss: ', loss)
if lang_stats:
  print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis.json', 'w'))

