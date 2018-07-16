import torch
import torch.nn.functional as F
from PIL import Image
import model
import vocab
import config
import image_features
import pdb

q_vocab, a_vocab = vocab.retrieve_vocab(
    config.vocab_questions_path, config.vocab_answers_path, config.questions_train_path, config.annotations_train_path)
imageNet = image_features.ImageFeaturesNet()
imageNet.eval()
mainModel = model.MainModel(len(q_vocab)+1)
mainModel.eval()
checkpoint = torch.load('weights', map_location='cpu')
mainModel.load_state_dict(checkpoint['state_dict'])
transform = image_features.get_transform(
    config.image_size, config.scale_fraction)


def predict(img_path, qn):
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    img.unsqueeze_(0)  # add batch size dim of 1
    img_features = imageNet(img)

    if qn[-1] == '?':
        qn = qn[:-1]
    qn = qn.lower().split(' ')
    qn, _ = process_question(qn, q_vocab)
    out = mainModel(img_features, qn.unsqueeze_(0), [qn.shape[0]])
    out = F.softmax(out[0], 0)
    topk_prob, topk_idx = torch.topk(out, 5)

    topk_ans = [next(k for k, v in a_vocab.items() if v == idx)
                for idx in topk_idx]
    predictions = [[topk_ans[i], topk_prob[i].item()] for i in range(5)]
    print(predictions)
    return predictions


def process_question(question, q_vocab):
    rtv = torch.zeros(len(question)).long()
    for i, tok in enumerate(question):
        if tok in q_vocab:
            idx = q_vocab[tok]
            rtv[i] = idx

    return rtv, len(question)


if __name__ == '__main__':
    predict('./data/val2014/COCO_val2014_000000000328.jpg',
            'how many people are there')
