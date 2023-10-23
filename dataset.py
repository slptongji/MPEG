from itertools import combinations
import os
import csv
import dgl
import json
import logging
import random
import pickle
import numpy as np
from collections import defaultdict
from torch.utils.data import IterableDataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

emotion_mapping = {
    'RECCON': {0: 'neutral', 1: 'anger', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'sadness', 6: 'surprise'},
    'convCEPE': {0: 'neutral', 1: 'sad', 2: 'happy', 3: 'angry', 4: 'excited', 5: 'frustrated'}
}


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, text_c, label, emotion_a, emotion_b, emotion_c):
        """Constructs a InputExample."""
        self.guid = guid
        self.text_a = text_a
        self.emotion_a = emotion_a
        self.text_b = text_b
        self.emotion_b = emotion_b
        self.text_c = text_c
        self.emotion_c = emotion_c
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, speaker_ids, mention_ids, emotion_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.speaker_ids = speaker_ids
        self.mention_ids = mention_ids
        self.emotion_ids = emotion_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MyProcessor(DataProcessor):
    """Processor for the RECCON data set."""

    def __init__(self, input_dir, max_seq_length, data_name):
        random.seed(1)
        self.dataset = [[], [], []]
        self.emotions = emotion_mapping[data_name]
        self.emotion_indexes = {v: k + 1 for k, v in emotion_mapping[data_name].items()}

        for idx, data_type in enumerate(['train', 'valid', 'test']):
            input_file = os.path.join(input_dir, '{}.json'.format(data_type))
            neg_examples = []

            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data_type == 'train':
                random.shuffle(data)

            for dialog in data:
                speakers = {}
                context = []
                emotions = []
                context_num = 0
                for ut in dialog:
                    speaker_id = speakers.get(ut['speaker'])
                    if speaker_id is None:
                        speaker_id = "S{}".format(len(speakers) + 1)
                        speakers[ut['speaker']] = speaker_id

                    info = speaker_id + " " + self.emotions[ut['emotion']] + " " + ut["text"].lower()
                    context.append(info)
                    emotions.append(int(ut['emotion']) + 1)
                    context_num += 1
                    assert context_num == len(context), "{} \n {}".format(context_num, context)

                    if 'cause' not in ut.keys():
                        continue

                    cur_len = 0
                    text_a = []
                    emotion_a = []
                    i = len(context) - 1
                    while i >= 0 and cur_len <= max_seq_length:
                        text_a.insert(0, context[i])
                        emotion_a.insert(0, emotions[i])
                        i -= 1
                        cur_len += len(context[i])

                    if data_name == "convCEPE":
                        start = 0
                        for i in range(start, len(context)):
                            emotion_b = int(ut['emotion']) + 1
                            emotion_c = self.emotion_indexes[context[i].split(' ')[1]]
                            if i + 1 in list(ut['cause']):
                                example = [text_a, emotion_a, info, emotion_b, context[i], emotion_c, [1]]
                                self.dataset[idx].append(example)
                            else:
                                example = [text_a, emotion_a, info, emotion_b,  context[i], emotion_c, [0]]
                                neg_examples.append(example)
                    else:
                        for i in range(len(context)):
                            emotion_b = int(ut['emotion']) + 1
                            emotion_c = self.emotion_indexes[context[i].split(' ')[1]]
                            if i + 1 in list(ut['cause']):
                                example = [text_a, emotion_a, info, emotion_b, context[i], emotion_c, [1]]
                            else:
                                example = [text_a, emotion_a, info, emotion_b, context[i], emotion_c, [0]]
                            self.dataset[idx].append(example)

            if data_name == "convCEPE":
                random.shuffle(neg_examples)
                self.dataset[idx] += neg_examples[:len(self.dataset[idx])]
                random.shuffle(self.dataset[idx])

        logger.info(
            "Train set:{} \t validation set:{} \t test set:{}.".format(len(self.dataset[0]), len(self.dataset[1]),
                                                                       len(self.dataset[2])))

    def _create_examples(self, data, data_type):
        examples = []
        for (i, d) in enumerate(data):
            guid = "{}-{}".format(data_type, i)
            text_a = ' '.join(d[0])
            emotion_a = d[1]
            text_b = d[2]
            emotion_b = d[3]
            text_c = d[4]
            emotion_c = d[5]
            label = d[6]
            examples.append(InputExample(guid=guid, text_a=text_a, emotion_a=emotion_a, text_b=text_b,
                                         emotion_b=emotion_b, text_c=text_c, emotion_c=emotion_c, label=label))
        return examples

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        return self._create_examples(self.dataset[0], "train")

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(self.dataset[1], "valid")

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        return self._create_examples(self.dataset[2], "test")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return [str(x) for x in range(2)]


class MyDataset(IterableDataset):

    def __init__(self, input_dir, saved_file, max_seq_length, max_speaker_num, tokenizer, encoder_type="BERT",
                 data_name="RECCON"):
        super(MyDataset, self).__init__()
        self.data = None
        self.max_seq_length = max_seq_length

        logger.info("Reading data from directory {}.".format(input_dir))

        if os.path.exists(saved_file):
            with open(saved_file, 'rb') as f:
                self.data = pickle.load(f)
            logger.info("Loading feature data from {}.".format(saved_file))
        else:
            self.data = []
            processor = MyProcessor(input_dir, max_seq_length, data_name)

            if 'train' in saved_file:
                examples = processor.get_train_examples(saved_file)
            elif 'dev' in saved_file:
                examples = processor.get_dev_examples(saved_file)
            elif 'test' in saved_file:
                examples = processor.get_test_examples(saved_file)
            else:
                logging.error("Invalid output file:{}".format(saved_file))

            logger.info("{} examples are constructed.".format(len(examples)))

            if encoder_type == 'BERT':
                features = convert_examples_to_features(examples, tokenizer, max_seq_length, max_speaker_num)
            elif encoder_type == 'RoBERTa':
                features = convert_examples_to_features_roberta(examples, tokenizer, max_seq_length, max_speaker_num)
            else:
                raise ValueError("The encoder type is invalid.")

            for idx, f in enumerate(features):
                turn_node_num = max(f[0].mention_ids) - 2
                head_mention_id = max(f[0].mention_ids) - 1
                tail_mention_id = max(f[0].mention_ids)
                speaker_edge = self.get_speaker_edge(f[0].speaker_ids, f[0].mention_ids)
                context_edge = self.get_context_edge(f[0].input_ids, f[0].mention_ids)
                graph, used_mention = self.create_graph(speaker_edge, context_edge, turn_node_num, head_mention_id,
                                                        tail_mention_id)
                assert len(used_mention) == (max(f[0].mention_ids) + 1)

                self.data.append({
                    'input_id': np.array(f[0].input_ids),
                    'label_id': np.array(f[0].label_ids),
                    'mention_id': np.array(f[0].mention_ids),
                    'segment_id': np.array(f[0].segment_ids),
                    'speaker_id': np.array(f[0].speaker_ids),
                    'input_mask': np.array(f[0].input_mask),
                    'turn_mask': mention2mask(np.array(f[0].mention_ids)),
                    'emotion_id': np.array(f[0].emotion_ids),
                    'graph': graph

                })

                if idx < 2:
                    logger.info("-------- Input Feature --------")
                    logger.info("tokens: %s" % graph)

            with open(saved_file, mode='wb') as f:
                pickle.dump(self.data, f)
            logger.info("Finish reading {} and save preprocessed data to {}.".format(input_dir, saved_file))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def get_speaker_edge(self, speaker_ids, mention_ids):
        tmp = defaultdict(set)
        for i in range(1, len(speaker_ids)):
            if speaker_ids[i] == 0:
                break
            tmp[speaker_ids[i]].add(mention_ids[i])

        speaker2turn_dict = dict()
        for k, v in tmp.items():
            speaker2turn_dict[k] = list(v)

        return speaker2turn_dict

    def KMP(self, text, pattern):
        next_matrix = [-1 for _ in range(len(pattern) + 1)]
        i, j = 0, -1
        while i < len(pattern):
            if j == -1 or pattern[i] == pattern[j]:
                i += 1
                j += 1
                next_matrix[i] = j
            else:
                j = next_matrix[j]

        i, j = 0, 0
        while i < len(text) and j < len(pattern):
            if j == -1 or text[i] == pattern[j]:
                i += 1
                j += 1
            else:
                j = next_matrix[j]

        if j == len(pattern):
            return i - j
        else:
            return -1

    def get_context_edge(self, input_ids, mention_ids, window=2):
        head_tokens = list()
        tail_tokens = list()
        head_mention_id = max(mention_ids) - 1
        tail_mention_id = max(mention_ids)
        edges = list()

        for i, men_id in enumerate(mention_ids):
            if men_id == head_mention_id:
                head_tokens.append(input_ids[i])
            elif men_id == tail_mention_id:
                tail_tokens.append(input_ids[i])
            if i + 1 < len(mention_ids) and men_id == 0 and mention_ids[i + 1] == 0:
                break

        tmp1 = self.KMP(input_ids, head_tokens)
        tmp2 = self.KMP(input_ids, tail_tokens)

        if tmp1 != -1 and mention_ids[tmp1] != head_mention_id:
            head_idx = mention_ids[tmp1]
            start = head_idx - window if head_idx > window else 1
            for i in range(start, head_idx + 1):
                edges.append((i, head_mention_id))
        if tmp2 != -1 and mention_ids[tmp2] != tail_mention_id:
            tail_idx = mention_ids[tmp2]
            start = tail_idx - window if tail_idx > window else 1
            for i in range(start, tail_idx + 1):
                edges.append((i, tail_mention_id))

        return edges

    def create_graph(self, speaker_edge, context_edge, turn_num, head_mention_id, tail_mention_id):

        edges = defaultdict(list)
        used_mention = set()

        for _, turns in speaker_edge.items():
            for h, t in combinations(turns, 2):
                edges[('node', 'speaker', 'node')].append((h, t))
                used_mention.add(h)
                used_mention.add(t)
        if not edges[('node', 'speaker', 'node')]:
            edges[('node', 'speaker', 'node')].append((1, 0))
            used_mention.add(1)
            used_mention.add(0)

        for h, t in context_edge:
            edges[('node', 'entity', 'node')].append((h, t))
            used_mention.add(h)
            used_mention.add(t)
        if not edges[('node', 'entity', 'node')]:
            edges[('node', 'entity', 'node')].append((0, head_mention_id))
            edges[('node', 'entity', 'node')].append((0, tail_mention_id))
            used_mention.add(head_mention_id)
            used_mention.add(tail_mention_id)
            used_mention.add(0)

        for i in range(1, turn_num + 1):
            edges[('node', 'dialog', 'node')].append((i, 0))
            edges[('node', 'dialog', 'node')].append((0, i))
            used_mention.add(i)
            used_mention.add(0)

        if head_mention_id not in used_mention:
            edges[('node', 'dialog', 'node')].append((head_mention_id, 0))
            edges[('node', 'dialog', 'node')].append((0, head_mention_id))
            used_mention.add(head_mention_id)
            used_mention.add(0)
        if tail_mention_id not in used_mention:
            edges[('node', 'dialog', 'node')].append((tail_mention_id, 0))
            edges[('node', 'dialog', 'node')].append((0, tail_mention_id))
            used_mention.add(tail_mention_id)
            used_mention.add(0)

        graph = dgl.heterograph(edges)

        return graph, used_mention


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_speaker_num=6):
    """ Converts a set of InputExamples to a list of InputFeatures."""

    logger.info("Converting {} examples to features.".format(len(examples)))
    features = [[]]

    for (idx, example) in enumerate(examples):
        tokens_a, tokens_a_speaker_ids, tokens_a_mention_ids = tokenize(example.text_a, tokenizer, max_speaker_num)
        tokens_b, tokens_b_speaker_ids, _ = tokenize(example.text_b, tokenizer, max_speaker_num)
        tokens_c, tokens_c_speaker_ids, _ = tokenize(example.text_c, tokenizer, max_speaker_num)

        _truncate_seq_pair(tokens_a, tokens_b, tokens_c, max_seq_length - 4, tokens_a_speaker_ids,
                           tokens_b_speaker_ids, tokens_c_speaker_ids, tokens_a_mention_ids)
        tokens_b_mention_ids = [max(tokens_a_mention_ids) + 1 for _ in range(len(tokens_b))]
        tokens_c_mention_ids = [max(tokens_a_mention_ids) + 2 for _ in range(len(tokens_c))]

        tokens_bc = tokens_b + ["[SEP]"] + tokens_c
        tokens_bc_speaker_ids = tokens_b_speaker_ids + [0] + tokens_c_speaker_ids
        tokens_bc_mention_ids = tokens_b_mention_ids + [0] + tokens_c_mention_ids

        tokens = ["[CLS]"]
        segment_ids = [0]
        speaker_ids = [0]
        mention_ids = [0]
        tokens += tokens_a
        speaker_ids += tokens_a_speaker_ids
        mention_ids += tokens_a_mention_ids
        segment_ids += [0] * len(tokens_a)
        tokens.append("[SEP]")
        speaker_ids.append(0)
        mention_ids.append(0)
        segment_ids.append(0)
        tokens += tokens_bc
        speaker_ids += tokens_bc_speaker_ids
        mention_ids += tokens_bc_mention_ids
        segment_ids += [1] * len(tokens_bc)
        tokens.append("[SEP]")
        speaker_ids.append(0)
        mention_ids.append(0)
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        label_ids = example.label

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            speaker_ids.append(0)
            mention_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(speaker_ids) == max_seq_length
        assert len(mention_ids) == max_seq_length

        if idx < 2:
            logger.info("-------- Input Example --------")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([x for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("speaker_ids: %s" % " ".join([str(x) for x in speaker_ids]))
            logger.info("mention_ids: %s" % " ".join([str(x) for x in mention_ids]))

        features[-1].append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                speaker_ids=speaker_ids,
                mention_ids=mention_ids
            )
        )
        if len(features[-1]) == 1:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]

    logging.info("Feature: {}".format(len(features)))
    return features


def convert_examples_to_features_roberta(examples, tokenizer, max_seq_length, max_speaker_num=6):
    """ Converts a set of InputExamples to a list of InputFeatures."""

    logger.info("Converting %d examples to features." % len(examples))

    features = [[]]

    for (idx, example) in enumerate(examples):
        tokens_a, tokens_a_speaker_ids, tokens_a_mention_ids, tokens_a_emotion_ids = tokenize(example.text_a,
                                                                                              example.emotion_a,
                                                                                              tokenizer,
                                                                                              max_speaker_num)
        tokens_b, tokens_b_speaker_ids, _, _ = tokenize(example.text_b, [example.emotion_b], tokenizer, max_speaker_num)
        tokens_c, tokens_c_speaker_ids, _, _ = tokenize(example.text_c, [example.emotion_c], tokenizer, max_speaker_num)

        _truncate_seq_pair(tokens_a, tokens_b, tokens_c, max_seq_length - 6, tokens_a_speaker_ids,
                           tokens_b_speaker_ids, tokens_c_speaker_ids, tokens_a_mention_ids, tokens_a_emotion_ids)
        tokens_b_mention_ids = [max(tokens_a_mention_ids) + 1 for _ in range(len(tokens_b))]
        tokens_c_mention_ids = [max(tokens_a_mention_ids) + 2 for _ in range(len(tokens_c))]
        tokens_b_emotion_ids = [example.emotion_b for _ in range(len(tokens_b))]
        tokens_c_emotion_ids = [example.emotion_c for _ in range(len(tokens_c))]

        tokens_bc = tokens_b + ['</s>', '</s>'] + tokens_c
        tokens_bc_speaker_ids = tokens_b_speaker_ids + [0, 0] + tokens_c_speaker_ids
        tokens_bc_mention_ids = tokens_b_mention_ids + [0, 0] + tokens_c_mention_ids
        tokens_bc_emotion_ids = tokens_b_emotion_ids + [0, 0] + tokens_c_emotion_ids

        tokens = ['<s>']
        segment_ids = [0]
        speaker_ids = [0]
        mention_ids = [0]
        emotion_ids = [0]
        tokens += tokens_a
        speaker_ids += tokens_a_speaker_ids
        mention_ids += tokens_a_mention_ids
        emotion_ids += tokens_a_emotion_ids
        segment_ids += [0] * len(tokens_a)
        tokens.append('</s>')
        speaker_ids.append(0)
        mention_ids.append(0)
        emotion_ids.append(0)
        segment_ids.append(0)
        tokens.append('</s>')
        speaker_ids.append(0)
        mention_ids.append(0)
        emotion_ids.append(0)
        segment_ids.append(0)
        tokens += tokens_bc
        speaker_ids += tokens_bc_speaker_ids
        mention_ids += tokens_bc_mention_ids
        emotion_ids += tokens_bc_emotion_ids

        segment_ids += [1] * len(tokens_bc)
        tokens.append('</s>')
        speaker_ids.append(0)
        mention_ids.append(0)
        emotion_ids.append(0)
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        label_ids = example.label

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            speaker_ids.append(0)
            mention_ids.append(0)
            emotion_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(speaker_ids) == max_seq_length
        assert len(mention_ids) == max_seq_length
        assert len(emotion_ids) == max_seq_length

        if idx < 2:
            logger.info("-------- Input Example --------")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([x for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("speaker_ids: %s" % " ".join([str(x) for x in speaker_ids]))
            logger.info("mention_ids: %s" % " ".join([str(x) for x in mention_ids]))
            logger.info("emotion_ids: %s" % " ".join([str(x) for x in emotion_ids]))

        features[-1].append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                speaker_ids=speaker_ids,
                mention_ids=mention_ids,
                emotion_ids=emotion_ids,
            )
        )
        if len(features[-1]) == 1:
            features.append([])

    if len(features[-1]) == 0:
        features = features[:-1]

    logging.info("Feature: %d" % len(features))
    return features


def tokenize(context, emotions, tokenizer, max_speaker_num):
    speaker2id = {}
    for i in range(1, max_speaker_num + 1):
        token = "S{}".format(i)
        speaker2id[token] = i

    speaker_ids = []
    mention_ids = []
    emotion_ids = []
    speaker_id = 0
    mention_id = 0
    tokens = tokenizer.tokenize(context)

    for token in tokens:
        if token in speaker2id.keys():
            speaker_id = speaker2id[token]
            mention_id += 1
        speaker_ids.append(speaker_id)
        mention_ids.append(mention_id)
        emotion_ids.append(emotions[mention_id - 1])

    return tokens, speaker_ids, mention_ids, emotion_ids


def _truncate_seq_pair(tokens_a, tokens_b, tokens_c, max_seq_length, tokens_a_speaker_ids, tokens_b_speaker_ids,
                       tokens_c_speaker_ids, tokens_a_mention_ids, tokens_a_emotion_ids):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_seq_length:
            break

        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop(0)
            tokens_a_speaker_ids.pop(0)
            tokens_a_mention_ids.pop(0)
            tokens_a_emotion_ids.pop(0)

        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
            tokens_b_speaker_ids.pop()
        else:
            tokens_c.pop()
            tokens_c_speaker_ids.pop()


def mention2mask(mention_id, window=1):
    slen = len(mention_id)
    mask = []
    turn_mention_ids = [i for i in range(1, np.max(mention_id) - 1)]
    for j in range(slen):
        if mention_id[j] not in turn_mention_ids:
            tmp = np.zeros(slen, dtype=bool)
            tmp[j] = 1
        else:
            start = mention_id[j]
            end = mention_id[j]
            if mention_id[j] - window in turn_mention_ids:
                start = mention_id[j] - window

            if mention_id[j] + window in turn_mention_ids:
                end = mention_id[j] + window
            tmp = (mention_id >= start) & (mention_id <= end)
        mask.append(tmp)
    mask = np.stack(mask)
    return mask
