# encoding: utf-8

import hashlib
import json
import sys
import jieba

def make_ref(raw_anno_path, ref_path):
    id_to_words = {}
    annotations = []
    images = []

    info =  {
        "contributor": "He Zheng",
        "description": "CaptionEval",
        "url": "https://github.com/AIChallenger/AI_Challenger.git",
        "version": "1",
        "year": 2017
    }
    licenses =[
        {
          "url": "https://challenger.ai"
        }
      ]
    type = "captions"

    imgs = json.load(open(raw_anno_path, 'r'))
    i = 1

    for img in imgs:
        # print(img['image_id'].split('.')[0])
        image_id = img['image_id'].split('.')[0]
        image_hash = int(int(hashlib.sha256(image_id.encode('utf-8')).hexdigest(), 16) % sys.maxint)
        # print(image_hash)
        for caption in img['caption']:
            image = {}
            annotation = {}
            image['file_name'] = image_id
            image['id'] = image_hash
            annotation['caption'] = ' '.join(jieba.lcut(caption))
            annotation['id'] = i
            i+=1
            annotation['image_id'] = image_hash
            annotations.append(annotation)
            images.append(image)

        # print(annotations)
        # print(images)
        # break
    id_to_words['annotations'] = annotations
    id_to_words['images'] = images
    id_to_words['info'] = info
    id_to_words['licenses'] = licenses
    id_to_words['type'] = type
    # print(id_to_words)
    json.dump(id_to_words, open(ref_path, 'w'))
    return id_to_words
