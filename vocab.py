import json
from collections import Counter
import itertools
import config
from os.path import isfile


def get_all_questions(questions_list):
    # remove the question mark at the end and make everything lowercase
    # return a list of list of words that correspond to question in question lists
    return [q["question"].lower()[:-1].split(' ') for q in questions_list]


def get_all_answers(annotations):
    return [[a['answer'] for a in ans_dict['answers']] for ans_dict in annotations]


def extract_vocab(list_of_list_of_token, top=None):
    all_tokens = itertools.chain.from_iterable(list_of_list_of_token)

    # counter example Counter({'what': 9, 'is': 8, 'this': 6}) {<word>: <frequency>}
    counter = Counter(all_tokens)

    if top:
        most_common = (word for word, _ in counter.most_common(top))
    else:
        most_common = counter.keys()

    tokens = sorted(most_common, key=lambda word: (
        counter[word], word), reverse=True)
    vocab = {word: idx for idx, word in enumerate(tokens)}

    return vocab


def main():
    # for testing purpose and demo of how to use
    vocab_answer = None

    if isfile(config.vocab_answers_path):
        with open(config.vocab_answers_path, 'r') as f:
            vocab_answer = json.load(f)
            f.close()
    else:
        with open(config.annotations_train_path, 'r') as f:
            annotations_train = get_all_answers(json.load(f)['annotations'])
            f.close()
        with open(config.annotations_val_path, 'r') as f:
            annotations_val = get_all_answers(json.load(f)['annotations'])
            f.close()

        annotations = annotations_train + annotations_val
        vocab_answer = extract_vocab(annotations)
        with open(config.vocab_answers_path, 'w') as f:
            json.dump(vocab_answer, f)
            f.close()

    print(vocab_answer)


if __name__ == "__main__":
    main()
