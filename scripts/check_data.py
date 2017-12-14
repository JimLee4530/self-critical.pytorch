import json
import jieba

filepath = '/home/j1ml3e/self-critical/data/dataset_coco.json'
file2path = '/home/j1ml3e/self-critical/data/caption_train_annotations_20170902.json'
save_dataset_ai = '/home/j1ml3e/self-critical/data/dataset_ai.json'

file = json.load(open(filepath, 'r'))

for i in file['images']:
    print(i)
    break

file2 = json.load(open(file2path, 'r'))

dataset_ai = {}
images = []

for i in file2:
    image = {}
    image['split'] = 'train'
    sentences = []
    tokens = {}
    for caption in i['caption']:
        tokens = {}
        tokens['tokens'] = jieba.lcut(caption)
        sentences.append(tokens)
    image['sentences'] = sentences
    images.append(image)
    # break

dataset_ai['images'] = images

json.dump(dataset_ai, open(save_dataset_ai, 'w'))