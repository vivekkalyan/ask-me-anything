import os
import torch
import torch.utils.data as data
from PIL import Image
import json
import h5py
import random

import config
import vocab


class CocoImages(data.Dataset):
    def __init__(self, path, transform=None):
        super(CocoImages, self).__init__()
        self.path = path
        self.id_to_filename = self._crawl_images()
        # used for deterministic iteration order
        self.sorted_ids = sorted(self.id_to_filename.keys())
        print('found {} images in {}'.format(len(self), self.path))
        self.transform = transform

    def _crawl_images(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith('.jpg'):
                continue
            id_and_extension = filename.split('_')[-1]
            id = int(id_and_extension.split('.')[0])
            id_to_filename[id] = filename
        return id_to_filename

    def __getitem__(self, item):
        id = self.sorted_ids[item]
        path = os.path.join(self.path, self.id_to_filename[id])
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return id, img

    def __len__(self):
        return len(self.sorted_ids)


class VQAData(data.Dataset):
    def __init__(self, questions_path=config.questions_train_path, annotations_path=config.annotations_train_path,  vocab_q_path=config.vocab_questions_path, vocab_a_path=config.vocab_answers_path, img_feature_path=config.img_feature_train_path, only_with_answer=False):
        super(VQAData, self).__init__()

        self.img_features_path = img_feature_path
        print('Generating map of COCO ids to index. Please wait...')
        self.img_id_to_idx = self.create_map_coco_id_to_index()

        with open(questions_path, 'r') as f:
            print('Getting all questions. Please wait...')
            raw_questions = json.load(f)['questions']
            self.img_ids = [q['image_id'] for q in raw_questions]
            self.questions = vocab.get_all_questions(raw_questions)
            self.max_question_len = max(list(map(len, self.questions)))
            f.close()
        with open(annotations_path, 'r') as f:
            print('Getting all answers. Please wait...')
            self.answers = vocab.get_all_answers(json.load(f)['annotations'])
            f.close()

        if not os.path.isfile(vocab_q_path):
            print('Generating question vocab. Please wait...')
            all_questions = []
            with open(config.questions_train_path) as f:
                all_questions += vocab.get_all_questions(
                    json.load(f)['questions'])
                f.close()
            self.q_vocab = vocab.extract_vocab(all_questions)
            with open(vocab_q_path, 'w') as f:
                json.dump(self.q_vocab, f)
                f.close()
        else:
            with open(vocab_q_path) as f:
                self.q_vocab = json.load(f)
                f.close()
            print('Question vocab loaded!')

        if not os.path.isfile(vocab_a_path):
            print('Generating answer vocab. Please wait...')
            all_answers = []
            with open(config.annotations_train_path) as f:
                all_answers += vocab.get_all_answers(
                    json.load(f)['annotations'])
                f.close()
            self.a_vocab = vocab.extract_vocab(all_answers, top=3000)
            with open(vocab_a_path, 'w') as f:
                json.dump(self.a_vocab, f)
                f.close()
        else:
            with open(vocab_a_path) as f:
                self.a_vocab = json.load(f)
                f.close()
            print('Answer vocab loaded!')

        self.questions = list(map(self.process_question, self.questions))
        self.answers = list(map(self.one_hot_answer, self.answers))

        self.only_with_answer = only_with_answer
        if self.only_with_answer:
            self.qidx_with_answers = self.questions_with_answers()

    def create_map_coco_id_to_index(self):
        ids = None

        with h5py.File(self.img_features_path, 'r') as features_file:
            ids = features_file['ids'][()]
            features_file.close()


        return {identity: idx for idx, identity in enumerate(ids)}

    def process_question(self, question):
        rtv = torch.zeros(self.max_question_len).long()
        for i, tok in enumerate(question):
            if tok in self.q_vocab:
                idx = self.q_vocab[tok]
                rtv[i] = idx

        return rtv, len(question)

    def one_hot_answer(self, answers):
        rtv = torch.zeros(len(self.a_vocab))
        for a in answers:
            if a in self.a_vocab:
                idx = self.a_vocab[a]
                rtv[idx] += 1

        return rtv

    def questions_with_answers(self):
        '''return list of questions indices with answers that's in vocab'''
        rtv = []
        for i, answers in enumerate(self.answers):
            if len(answers.nonzero()) > 0:
                rtv.append(i)

        return rtv

    def get_img(self, img_id):
        if not hasattr(self, 'features_file'):
            self.features_file = h5py.File(self.img_features_path, 'r')

        # just to clear img id not found error during testing with less data
        # will need to be deleted later
        if img_id in self.img_id_to_idx:
            idx = self.img_id_to_idx[img_id]
        else:
            idx = self.img_id_to_idx[random.choice(
                list(self.img_id_to_idx.keys()))]
        ###############
        # and changed to the below line
        # idx = self.img_id_to_idx[img_id]
        images = self.features_file['features']
        img = images[idx].astype('float32')
        return torch.from_numpy(img)

    def num_tokens(self):
        return len(self.q_vocab) + 1

    def __len__(self):
        if self.only_with_answer:
            return len(self.qidx_with_answers)
        else:
            return len(self.questions)

    def __getitem__(self, idx):
        if self.only_with_answer:
            idx = self.qidx_with_answers[idx]

        q, q_len = self.questions[idx]
        a = self.answers[idx]
        img_id = self.img_ids[idx]
        img = self.get_img(img_id)
        return img, q, a, idx, q_len


def create_vqa_loader(train=True):
    questions_path = config.questions_train_path if train else config.questions_val_path
    annotations_path = config.annotations_train_path if train else config.annotations_val_path
    img_feature_path = config.img_feature_train_path if train else config.img_feature_val_path
    dset = VQAData(questions_path=questions_path,
                   annotations_path=annotations_path, img_feature_path=img_feature_path, only_with_answer=train)

    return data.DataLoader(dset, batch_size=config.batch_size, shuffle=train, pin_memory=True, num_workers=config.data_workers, collate_fn=descending_in_len)


def descending_in_len(batch):
    batch.sort(key=lambda x: x[-1], reverse=True)
    return data.dataloader.default_collate(batch)
