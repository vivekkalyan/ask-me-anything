annotations_val_path = './data/v2_mscoco_train2014_annotations.json'
annotations_train_path = './data/v2_mscoco_val2014_annotations.json'
questions_train_path = './data/v2_OpenEnded_mscoco_train2014_questions.json'
questions_val_path = './data/v2_OpenEnded_mscoco_val2014_questions.json'

vocab_answers_path = './data/vocab.answers.json'
vocab_questions_path = './data/vocab.questions.json'

img_feature_train_path = './data/img_features.train.h5'
img_feature_val_path = './data/img_features.val.h5'
train_path = './data/train2014'
val_path = './data/val2014'

# image features config
image_size = 448
output_size = image_size // 32
output_features = 2048
scale_fraction = 0.875
image_batch_size = 4

data_workers = 4

# data loader config
batch_size = 4

# training config
initial_learning_rate = 1e-3
lr_halflife = 5e4
epochs = 10
