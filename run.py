# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import argparse
import os
import shutil
import random
import torch
import numpy as np

from tqdm import tqdm, trange
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from dataset import MyDataset
from dataloader import MPEGDataLoader
from MPEG import MPEG_RoBERTa
from models.RoBERTa.configuration_roberta import RobertaConfig
import wandb

wandb.init(project="cause-iemocap", entity="tim4")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s[line:%(lineno)d] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default="data",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir",
                        default="models/RoBERTa",
                        type=str,
                        help="The input model dir of pre-trained model.")
    parser.add_argument("--output_dir",
                        default="output",
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--config_file",
                        default="config.json",
                        type=str,
                        help="The config json file corresponding to the pre-trained model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file",
                        default="vocab.txt",
                        type=str,
                        help="The vocabulary file that the model was trained on.")
    parser.add_argument("--merges_file",
                        default=None,
                        type=str,
                        help="The merges file that the RoBERTa model was trained on.")
    parser.add_argument("--data_name",
                        default="RECCON",
                        type=str,
                        help="The name of the dataset to train.")
    parser.add_argument("--encoder_type",
                        default="RoBERTa",
                        type=str,
                        help="The type of pre-trained model.")
    parser.add_argument("--init_checkpoint",
                        default="pytorch_model.bin",
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained model).")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--max_speaker_num",
                        default=6,
                        type=int,
                        help="The maximum total speaker number.")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--num_graph_attention",
                        default=1,
                        type=int,
                        help="The maximum total speaker number.")
    parser.add_argument("--num_train_epochs",
                        default=4.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--save_checkpoints_steps",
                        default=1000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=666,
                        help="random seed for initialization")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument("--resume",
                        default=False,
                        action='store_true',
                        help="Whether to resume the training.")
    parser.add_argument("--f1eval",
                        default=True,
                        action='store_true',
                        help="Whether to use f1 for dev evaluation during training.")

    args = parser.parse_args()
    set_random_seed_all(args.seed)

    wandb.config = {
        "encoder_type": args.encoder_type,
        "max_seq_length": args.max_seq_length,
        "num_graph_attention": args.num_graph_attention,
        "learning_rate": args.learning_rate,
        "train_epoch": args.num_train_epochs,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False

    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.encoder_type == 'RoBERTa':
        config = RobertaConfig.from_pretrained(args.model_dir)
    else:
        raise ValueError("The encoder type is invalid.")

    if args.max_seq_length > config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the pretrained model was "
            "only trained up to sequence length {}".format(args.max_seq_length, config.max_position_embeddings))

    output_dir = os.path.join(args.output_dir, args.data_name + '_' + args.encoder_type)
    if os.path.exists(output_dir) and 'model.pt' in os.listdir(output_dir):
        if args.do_train and not args.resume:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)

    if args.encoder_type == 'RoBERTa':
        tokenizer = RobertaTokenizer.from_pretrained(args.model_dir, do_lower_case=False)
        speaker2id = {}
        for i in range(1, args.max_speaker_num + 1):
            token = "S{}".format(i)
            speaker2id[token] = i
        special_tokens_dict = {'additional_special_tokens': list(speaker2id.keys())}
        tokenizer.add_special_tokens(special_tokens_dict)
    else:
        raise ValueError("The pretrained tokenizer has not been initialized.")

    num_train_steps = None

    train_set, train_loader, dev_set, dev_loader, test_set, test_loader = None, None, None, None, None, None
    saved_dir = os.path.join("processed", args.data_name + '_' + args.encoder_type)
    data_dir = os.path.join(args.data_dir, args.data_name)
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    if args.do_train:
        train_set = MyDataset(input_dir=data_dir, saved_file=os.path.join(saved_dir, "train.pkl"),
                              max_seq_length=args.max_seq_length, tokenizer=tokenizer, encoder_type=args.encoder_type,
                              max_speaker_num=args.max_speaker_num, data_name=args.data_name)
        num_train_steps = int(
            len(train_set) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        train_loader = MPEGDataLoader(dataset=train_set, batch_size=args.train_batch_size, shuffle=True,
                                      max_length=args.max_seq_length)

        dev_set = MyDataset(input_dir=data_dir, saved_file=os.path.join(saved_dir, "dev.pkl"),
                            max_seq_length=args.max_seq_length, tokenizer=tokenizer, encoder_type=args.encoder_type,
                            max_speaker_num=args.max_speaker_num, data_name=args.data_name)
        dev_loader = MPEGDataLoader(dataset=dev_set, batch_size=args.eval_batch_size, shuffle=False,
                                    max_length=args.max_seq_length)

    if args.do_eval:
        test_set = MyDataset(input_dir=data_dir, saved_file=os.path.join(saved_dir, "test.pkl"),
                             max_seq_length=args.max_seq_length, tokenizer=tokenizer, encoder_type=args.encoder_type,
                             max_speaker_num=args.max_speaker_num, data_name=args.data_name)
        test_loader = MPEGDataLoader(dataset=test_set, batch_size=args.eval_batch_size, shuffle=False,
                                     max_length=args.max_seq_length)

    if args.encoder_type == 'RoBERTa':
        model = MPEG_RoBERTa(config, num_labels=1, num_graph_attention=args.num_graph_attention,
                             data_name=args.data_name)
        model_path = os.path.join(args.model_dir, args.init_checkpoint)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        else:
            raise FileNotFoundError("The pre-trained model path does not exist: {}.".format(model_path))
        model.roberta.resize_token_embeddings(len(tokenizer))
        model.init_emotions(tokenizer, data_name=args.data_name)
    else:
        raise ValueError("The pretrained model has not been initialized.")

    if args.fp16:
        model.half()
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_())
                           for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_())
                           for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(args.warmup_proportion * num_train_steps),
                                                num_training_steps=num_train_steps)

    global_step = 0

    if args.resume:
        model.load_state_dict(torch.load(os.path.join(output_dir, "model_best.pt")))

    if args.do_train:
        best_metric = 0
        logger.info("-------- Start training --------")
        logger.info("\tExample Size: {}".format(len(train_set)))
        logger.info("\tBatch Size: {}".format(args.train_batch_size))
        logger.info("\tStep Number: {}".format(num_train_steps))

        for cur_epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
                input_ids = batch['input_ids'].to(device)
                segment_ids = batch['segment_ids'].to(device)
                mention_ids = batch['mention_ids'].to(device)
                emotion_ids = batch['emotion_ids'].to(device)
                speaker_ids = batch['speaker_ids'].to(device)
                label_ids = batch['label_ids'].to(device)
                input_masks = batch['input_masks'].to(device)
                turn_masks = batch['turn_masks'].to(device)
                graphs = batch['graphs']

                loss, _ = model(input_ids=input_ids, token_type_ids=segment_ids, attention_masks=input_masks,
                                speaker_ids=speaker_ids, graphs=graphs, mention_ids=mention_ids,
                                emotion_ids=emotion_ids,
                                labels=label_ids, turn_masks=turn_masks)
                if n_gpu > 1:
                    loss = loss.mean()
                if args.fp16 and args.loss_scale != 1.0:
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                wandb.log({"tr_loss": loss.item()})
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 128
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        scheduler.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                        scheduler.step()
                    optimizer.zero_grad()
                    model.zero_grad()
                    global_step += 1

            logger.info("-------- Start Evaluating: {}--------".format(cur_epoch))
            model.eval()
            eval_loss, eval_accuracy, eval_f1 = 0, 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            logits_all = []
            labels_all = []
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(device)
                segment_ids = batch['segment_ids'].to(device)
                mention_ids = batch['mention_ids'].to(device)
                emotion_ids = batch['emotion_ids'].to(device)
                speaker_ids = batch['speaker_ids'].to(device)
                label_ids = batch['label_ids'].to(device)
                input_masks = batch['input_masks'].to(device)
                turn_masks = batch['turn_masks'].to(device)
                graphs = batch['graphs']

                with torch.no_grad():
                    tmp_eval_loss, logits = model(input_ids=input_ids, token_type_ids=segment_ids,
                                                  attention_masks=input_masks,
                                                  speaker_ids=speaker_ids, graphs=graphs, mention_ids=mention_ids,
                                                  emotion_ids=emotion_ids, labels=label_ids, turn_masks=turn_masks)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                for i in range(len(logits)):
                    logits_all += [logits[i]]
                for i in range(len(label_ids)):
                    labels_all.append(label_ids[i])

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            result = {'eval_loss': eval_loss, 'global_step': global_step, 'loss': tr_loss / nb_tr_steps}

            if args.f1eval:
                _, _, eval_f1, _, _ = calc_test_result(logits_all, labels_all, args.data_name)
                result['f1'] = eval_f1

                wandb.log({"f1_macro": eval_f1, "eval_loss": eval_loss})
                wandb.watch(model)

            logger.info("-------- Eval Results --------")
            for key in sorted(result.keys()):
                logger.info("{}: {}".format(key, str(result[key])))

            if args.f1eval:
                if eval_f1 >= best_metric:
                    torch.save(model.state_dict(), os.path.join(output_dir, "model_best.pt"))
                    best_metric = eval_f1

        model.load_state_dict(torch.load(os.path.join(output_dir, "model_best.pt")))
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    if args.do_eval:
        model.load_state_dict(torch.load(os.path.join(output_dir, "model.pt")))

        logger.info("-------- Start Testing --------")
        logger.info("\tExample Size: {}".format(len(test_set)))
        logger.info("\tBatch Size: {}".format(args.eval_batch_size))
        get_logits4eval(model, test_loader, os.path.join(output_dir, "test_logits.txt"),
                        os.path.join(output_dir, "test_result.txt"), device, args.data_name)


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_random_seed_all(seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if test_nan and torch.isnan(param_model.grad).sum() > 0:
            is_nan = True
        if param_opti.grad is None:
            param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
        param_opti.grad.data.copy_(param_model.grad.data)
    return is_nan


def calc_test_result(logits, labels_all, data_name):
    logits = np.asarray(logits)
    logits = list(1 / (1 + np.exp(-logits)))
    true_label = np.squeeze(labels_all)
    predicted_label = np.rint(np.squeeze(logits))

    print("Confusion Matrix For {}:".format(data_name))
    print(confusion_matrix(true_label, predicted_label))
    print("Classification Report For {}:".format(data_name))
    print(classification_report(true_label, predicted_label, digits=4))
    p_weighted, r_weighted, f_weighted, support_weighted = precision_recall_fscore_support(true_label,
                                                                                           predicted_label,
                                                                                           average='weighted')
    print('Weighted FScore: \n ', p_weighted, r_weighted, f_weighted, support_weighted)
    p_macro, r_macro, f_macro, support_macro = precision_recall_fscore_support(true_label,
                                                                               predicted_label,
                                                                               average='macro')
    print('Macro FScore: \n ', p_macro, r_macro, f_macro, support_macro)
    return p_macro, r_macro, f_macro, support_macro, logits


def get_logits4eval(model, dataloader, save_file, output_file, device, data_name):
    model.eval()
    logits_all = []
    labels_all = []
    for batch in tqdm(dataloader, desc="Iteration"):
        input_ids = batch['input_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        mention_ids = batch['mention_ids'].to(device)
        emotion_ids = batch['emotion_ids'].to(device)
        speaker_ids = batch['speaker_ids'].to(device)
        label_ids = batch['label_ids'].to(device)
        input_masks = batch['input_masks'].to(device)
        turn_masks = batch['turn_masks'].to(device)
        graphs = batch['graphs']

        with torch.no_grad():
            _, logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_masks=input_masks,
                              speaker_ids=speaker_ids, graphs=graphs, mention_ids=mention_ids, emotion_ids=emotion_ids,
                              labels=label_ids, turn_masks=turn_masks)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        for i in range(len(logits)):
            logits_all += [logits[i]]
        for i in range(len(label_ids)):
            labels_all.append(label_ids[i])

    p_macro, r_macro, f_macro, support_macro, logits_all = calc_test_result(logits_all, labels_all, data_name)

    with open(output_file, "w") as f:
        f.write("p_macro: {}\n".format(str(p_macro)))
        f.write("r_macro: {}\n".format(str(r_macro)))
        f.write("f_macro: {}\n".format(str(f_macro)))
        f.write("support_macro: {}\n".format(str(support_macro)))

    with open(save_file, "w") as f:
        for i in range(len(logits_all)):
            for j in range(len(logits_all[i])):
                f.write(str(logits_all[i][j]))
                if j == len(logits_all[i]) - 1:
                    f.write("\n")
                else:
                    f.write(" ")
    wandb.log({"f1_macro_test": f_macro})


if __name__ == "__main__":
    main()
