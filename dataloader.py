import math
import torch
import random
from torch.utils.data import DataLoader


class MPEGDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, max_length=512):
        super(MPEGDataLoader, self).__init__(dataset, batch_size=batch_size)
        self.shuffle = shuffle
        self.length = len(self.dataset)
        self.max_length = max_length
        self.order = list(range(self.length))
        print("initialize {}".format(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.order)
            self.data = [self.dataset[idx] for idx in self.order]
        else:
            self.data = self.dataset

        batch_num = math.ceil(self.length / self.batch_size)

        self.batches = [self.data[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                        for idx in range(0, batch_num)]
        self.batches_order = [self.order[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                              for idx in range(0, batch_num)]
        print("batch_num {} \t batches_num {}".format(batch_num, len(self.batches)))

        input_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        segment_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        mention_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        emotion_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        speaker_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        input_masks = torch.LongTensor(self.batch_size, self.max_length).cpu()
        turn_masks = torch.LongTensor(self.batch_size, self.max_length, self.max_length).cpu()
        label_ids = torch.LongTensor(self.batch_size, 1).cpu()

        for _, batch in enumerate(self.batches):
            batch_len = len(batch)

            for item in [input_ids, segment_ids, mention_ids, emotion_ids, label_ids,
                         speaker_ids, input_masks, turn_masks]:
                if item is not None:
                    item.zero_()

            graphs = []
            label_num = None
            for i, example in enumerate(batch):
                in_id, seg_id, lb_id, mt_id, em_id, sp_id, in_mask, tr_mask, graph = \
                    example['input_id'], example['segment_id'], example['label_id'], example['mention_id'], \
                    example['emotion_id'], example['speaker_id'], example['input_mask'], example['turn_mask'], example[
                        'graph']

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                graphs.append(graph.to(device))

                word_num = in_id.shape[0]
                label_num = lb_id.shape[0]

                input_ids[i, :word_num].copy_(torch.from_numpy(in_id))
                segment_ids[i, :word_num].copy_(torch.from_numpy(seg_id))
                input_masks[i, :word_num].copy_(torch.from_numpy(in_mask))
                mention_ids[i, :word_num].copy_(torch.from_numpy(mt_id))
                emotion_ids[i, :word_num].copy_(torch.from_numpy(em_id))
                speaker_ids[i, :word_num].copy_(torch.from_numpy(sp_id))
                turn_masks[i, :word_num, :word_num].copy_(torch.from_numpy(tr_mask))

                label_ids[i, :label_num].copy_(torch.from_numpy(lb_id))

            context_word_mask = input_ids > 0
            context_word_len = context_word_mask.sum()
            batch_max_len = context_word_len.max()

            yield {'input_ids': get_cuda(input_ids[:batch_len, :batch_max_len].contiguous()),
                   'segment_ids': get_cuda(segment_ids[:batch_len, :batch_max_len].contiguous()),
                   'mention_ids': get_cuda(mention_ids[:batch_len, :batch_max_len].contiguous()),
                   'emotion_ids': get_cuda(emotion_ids[:batch_len, :batch_max_len].contiguous()),
                   'speaker_ids': get_cuda(speaker_ids[:batch_len, :batch_max_len].contiguous()),
                   'input_masks': get_cuda(input_masks[:batch_len, :batch_max_len].contiguous()),
                   'turn_masks': get_cuda(turn_masks[:batch_len, :batch_max_len, :batch_max_len].contiguous()),
                   'label_ids': get_cuda(label_ids[:batch_len, :label_num].contiguous()),
                   'graphs': graphs
                   }


def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor
