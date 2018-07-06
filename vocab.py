import json
from collections import Counter
import itertools
import config
from os.path import isfile
import re


def get_all_questions(questions_list):
    # remove the question mark at the end and make everything lowercase
    # return a list of list of words that correspond to question in question lists
    return [q["question"].lower()[:-1].split(' ') for q in questions_list]


def _handle_punctuation(s):
    punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
    punctuation = re.compile(r'([{}])'.format(re.escape(punctuation_chars)))
    punctuation_with_space = re.compile(
        r'(?<= )([{0}])|([{0}])(?= )'.format(punctuation_chars))

    if punctuation.search(s) is not None:
        s = punctuation_with_space.sub('', s)
        if re.search(re.compile(r'(\d)(,)(\d)'), s) is not None:
            s = s.replace(',', '')
        s = punctuation.sub(' ', s)
        s = re.compile(r'(?!<=\d)(\.)(?!\d)').sub('', s)

    return s


def get_all_answers(annotations):
    list_of_list_of_answers = [
        [a['answer'] for a in ans_dict['answers']] for ans_dict in annotations]

    for i, list_of_answers in enumerate(list_of_list_of_answers):
        list_of_list_of_answers[i] = list(
            map(_handle_punctuation, list_of_answers))

    return list_of_list_of_answers


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
        vocab_answer = extract_vocab(annotations, top=3000)
        with open(config.vocab_answers_path, 'w') as f:
            json.dump(vocab_answer, f)
            f.close()

    print(vocab_answer)


if __name__ == "__main__":
    main()
