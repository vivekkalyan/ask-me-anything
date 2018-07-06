annotations_val_path = './data/v2_mscoco_train2014_annotations.json'
annotations_train_path = './data/v2_mscoco_val2014_annotations.json'
questions_train_path = './data/v2_OpenEnded_mscoco_train2014_questions.json'
questions_val_path = './data/v2_OpenEnded_mscoco_val2014_questions.json'

vocab_answers_path = './data/vocab.answers.json'
vocab_questions_path = './data/vocab.questions.json'
img_feature_path = './data/img_features.h5'

train_path = 'vqa/mscoco/train2014'
val_path = 'vqa/mscoco/val2014'

# image features config
image_size = 448
output_size = image_size // 32
output_features = 2048
scale_fraction = 0.875
image_batch_size = 64

data_workers = 4
