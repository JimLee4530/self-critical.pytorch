from caption_eval.val_reference import make_ref

raw_val_anno_path = '/data/common/data/ai_challenger_caption/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json'
val_ref_path = '/home/common/self-critical/data/val_ref.json'
make_ref(raw_val_anno_path, val_ref_path)