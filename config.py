annotations_val_path = ''
annotations_train_path = ''

vocab_answers_path = ''
vocab_questions_path = ''

train_path = 'vqa/mscoco/train2014'
val_path = 'vqa/mscoco/val2014'

# image features config
image_size = 448
output_size = image_size // 32
output_features = 2048
scale_fraction = 0.875
image_batch_size = 64

data_workers = 4