import argparse
import os
import datetime
import pdb

os.environ["WANDB_API_KEY"] = 'b1c08cbbfa2237ed18423fee2440ca916625a460'
import numpy as np
import torch
import ujson as json
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from args import add_args
from model import DocREModel
from pruner import Pruner
from utils import set_seed, collate_fn, create_directory
from prepro import read_docred
from evaluation import to_official, official_evaluate, merge_results
import wandb
from tqdm import tqdm
import run_pruner
import time
import pandas as pd
import pickle


def load_input(batch, device, tag="dev", func="value"):

    input = {'input_ids': batch[0].to(device),
             'attention_mask': batch[1].to(device),
             'labels': batch[2].to(device),
             'entity_pos': batch[3],
             'hts': batch[4],
             'sent_pos': batch[5],
             'sent_labels': batch[6].to(device) if (not batch[6] is None) and (batch[7] is None) else None,
             'teacher_attns': batch[7].to(device) if (not batch[7] is None) else None,
             'arg_labels': batch[-1].to(device),
             'tag': tag,
             'func': func
             }

    return input

def argument_train(args, model, features, tag):
    dataloader = DataLoader(features, batch_size=1, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    # preds = []
    index = 0
    count = 0
    for batch in tqdm(dataloader, desc=f"Evaluating batches"):
        model.eval()
        inputs = load_input(batch, args.device, tag, func="argument")
        with torch.no_grad():
            outputs = model(**inputs)
            value_rel_pred = outputs["value_rel_pred"]
            policy_rel_pred = outputs["policy_rel_pred"]
            prob = outputs["prob"]
            pred = value_rel_pred[:, 1:] * policy_rel_pred[:, 1:] * torch.where(prob > args.threshold_prob, 1.0, 0.0)
            #pred = policy_rel_pred[:, 1:] * torch.where(prob > args.threshold_prob, 1.0, 0.0)
            th_labels = torch.where(torch.sum(pred, dim=-1) == 0., 1, 0).unsqueeze(-1)
            new_pred = torch.cat([th_labels, pred], dim=-1)

            # final_pred = new_pred + inputs['arg_labels']    # 累加
            final_pred = new_pred   # 非累加
            final_pred = torch.where(final_pred > 0, 1.0, 0.0)
            final_pred[:, 0] = (final_pred.sum(1) == 0.)
            count += torch.where(final_pred[:, 1:] - inputs['labels'][:, 1:] > 0, 1, 0).sum()
            features[index]['arg_labels'] = final_pred.cpu().numpy()
            index += 1
    print(count)
    return features
    

def train(args, model, tokenizer, train_features, dev_features, test_features, all_train_features, all_dev_features, pruner=None):

    def finetune(features, value_optimizer, policy_optimizer, num_epoch, num_rel_steps, num_rl_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn,
                                      drop_last=True)

        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        
        
        # linear
        value_scheduler = get_linear_schedule_with_warmup(value_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        policy_scheduler = get_linear_schedule_with_warmup(policy_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        new_policy_scheduler = get_linear_schedule_with_warmup(policy_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps * 2.0 / 3.0)

        scaler = GradScaler()
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))

        for epoch in tqdm(train_iterator, desc='Train epoch'):
            
            if epoch == 10:
                print("init policy scheduler")
                policy_scheduler = new_policy_scheduler
            
            model.train()
            model.zero_grad()
            value_optimizer.zero_grad()
            policy_optimizer.zero_grad()
            
            rl_loss_list = []
            reward_list = []
            for step, batch in enumerate(train_dataloader):
                if epoch < 10:
                    # value func learning
                    value_optimizer.zero_grad()
                    inputs = load_input(batch, args.device, tag="train", func="value")
                    outputs = model(**inputs)
                    loss = [outputs["rel_loss"]["rel_loss"]]
                    if inputs["sent_labels"] != None:
                        loss.append(outputs["rel_loss"]["evi_loss"] * args.evi_lambda)
                    loss = sum(loss) / args.gradient_accumulation_steps
                    scaler.scale(loss).backward()

                    wandb.log(outputs["rel_loss"], step=num_rel_steps)
                    if args.max_grad_norm > 0:
                        scaler.unscale_(value_optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    print('value lr: ', value_optimizer.param_groups[0]["lr"])
                    scaler.step(value_optimizer)
                    scaler.update()
                    value_scheduler.step()
                    value_optimizer.zero_grad()

                # policy func learning
                policy_optimizer.zero_grad()
                inputs = load_input(batch, args.device, tag="train", func="policy")
                outputs = model(**inputs)
                loss = [outputs["rl_loss"]["rl_loss"]]
                loss = sum(loss) / args.gradient_accumulation_steps
                scaler.scale(loss).backward()

                wandb.log(outputs["rl_loss"], step=num_rel_steps)
                wandb.log(outputs['reward'], step=num_rel_steps)
                
                if args.max_grad_norm > 0:
                    scaler.unscale_(policy_optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                print('policy lr: ', policy_optimizer.param_groups[0]["lr"])
                scaler.step(policy_optimizer)
                scaler.update()
                policy_scheduler.step()
                policy_optimizer.zero_grad()
               
                num_rel_steps += 1
            if epoch < 12:
                value_dev_scores, value_dev_output, _, _ = evaluate(args, model, dev_features, tag="dev", func="value")
                print_matrix(value_dev_output, 'dev')
                wandb.log(value_dev_scores, step=num_rel_steps)

                value_test_scores, value_test_output, _, _ = evaluate(args, model, test_features, tag="test", func="value")
                print_matrix(value_test_output, 'test')
                wandb.log(value_test_scores, step=num_rel_steps)
            
            dev_scores, dev_output, _, _ = evaluate(args, model, dev_features, tag="dev", func="policy")
            print_matrix(dev_output, 'dev')
            wandb.log(dev_scores, step=num_rel_steps)

            test_scores, test_output, _, _ = evaluate(args, model, test_features, tag="test", func="policy")
            print_matrix(test_output, 'test')
            wandb.log(test_scores, step=num_rel_steps)
            
            model.eval()
            new_train_features = argument_train(args, model, features, 'train')
            train_dataloader = DataLoader(new_train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

            if dev_scores["policy_dev_F1"] > best_score:
                best_score = dev_scores["policy_dev_F1"]
                best_output = dev_output

                ckpt_file = os.path.join(args.save_path, "best.ckpt")
                print(f"saving model checkpoint into {ckpt_file} ...")
                torch.save(model.state_dict(), ckpt_file)

            if epoch == train_iterator[-1]: # last epoch
                ckpt_file = os.path.join(args.save_path, "last.ckpt")
                print(f"saving model checkpoint into {ckpt_file} ...")
                torch.save(model.state_dict(), ckpt_file)

        return num_rel_steps, num_rl_steps


    new_layer = ["extractor", "bilinear"]
    key_list = ['value_model', 'head_extractor', 'tail_extractor', 'bilinear']
    value_optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if n.split('.')[0] in key_list and not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if n.split('.')[0] in key_list and any(nd in n for nd in new_layer)], "lr": args.lr_added},
    ]

    policy_optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if
                    n.split('.')[0] not in key_list and not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if
                    n.split('.')[0] not in key_list and any(nd in n for nd in new_layer)], "lr": args.lr_added},
    ]
    
    value_optimizer = AdamW(value_optimizer_grouped_parameters, lr=args.lr_transformer, eps=args.adam_epsilon)
    policy_optimizer = AdamW(policy_optimizer_grouped_parameters, lr=args.lr_transformer, eps=args.adam_epsilon)
    num_rel_steps = 0
    num_rl_steps = 0
    set_seed(args)
    model.zero_grad()

    paras_path = os.path.join(args.save_path, "config.json")
    device = args.device
    dump_dict = vars(args)
    dump_dict['device'] = ''
    json.dump(vars(args), open(paras_path, 'w'), indent=4)
    args.device = device
    # for iter_step in tqdm(range(int(args.num_train_epochs))):
    #     arg_train_features = read_docred(train_file, tokenizer, transformer_type=args.transformer_type,
    #                           max_seq_length=args.max_seq_length, teacher_sig_path=args.teacher_sig_path)
    finetune(train_features, value_optimizer, policy_optimizer, args.num_train_epochs, num_rel_steps, num_rl_steps)


def evaluate(args, model, features, tag="dev", func='policy', pruner=None):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    preds, evi_preds = [], []
    scores, topks = [], []
    attns = []

    for batch in tqdm(dataloader, desc=f"Evaluating batches"):
        model.eval()

        if args.save_attn:
            tag = "infer"
        if func == "policy":
            inputs = load_input(batch, args.device, tag, func="policy")
        else:
            inputs = load_input(batch, args.device, tag, func="value")
        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs["rel_pred"]

            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

            if "scores" in outputs:
                score = outputs["scores"]
                topk = outputs["topks"]

                scores.append(score.cpu().numpy())
                topks.append(topk.cpu().numpy())

            if "evi_pred" in outputs:  # relation extraction and evidence extraction
                evi_pred = outputs["evi_pred"]

                evi_pred = evi_pred.cpu().numpy()
                evi_preds.append(evi_pred)

            if "attns" in outputs:  # attention recorded
                attn = outputs["attns"]
                attns.extend([a.cpu().numpy() for a in attn])

    preds = np.concatenate(preds, axis=0)

    if scores != []:
        scores = np.concatenate(scores, axis=0)
        topks = np.concatenate(topks, axis=0)

    if evi_preds != []:
        evi_preds = np.concatenate(evi_preds, axis=0)

    official_results, results = to_official(preds, features, evi_preds=evi_preds, scores=scores, topks=topks)
    if len(official_results) > 0:
        if tag == "test":
            best_re, best_evi, best_re_ign, _ = official_evaluate(official_results, args.data_dir, args.train_file,
                                                                  args.test_file)
        elif tag == "dev":
            best_re, best_evi, best_re_ign, _ = official_evaluate(official_results, args.data_dir, args.train_file,
                                                                  args.dev_file)
        elif tag == "train":
            best_re, best_evi, best_re_ign, _ = official_evaluate(official_results, args.data_dir, args.train_file,
                                                                  args.train_file)
        elif tag == "all_train":
            best_re, best_evi, best_re_ign, _ = official_evaluate(official_results, args.data_dir, args.train_file,
                                                                  "train_revised.json")
        elif tag == "all_dev":
            best_re, best_evi, best_re_ign, _ = official_evaluate(official_results, args.data_dir, args.train_file,
                                                                  "dev_revised.json")
    else:
        best_re = best_evi = best_re_ign = [-1, -1, -1]
    output = {
        func + '_' + tag + "_rel": [i * 100 for i in best_re],
        func + '_' + tag + "_rel_ign": [i * 100 for i in best_re_ign],
        func + '_' + tag + "_evi": [i * 100 for i in best_evi],
    }
    scores = {func + '_' + tag + "_pre": best_re[0] * 100, func + '_' + tag + "_recall": best_re[1] * 100,
              func + '_' + tag + "_F1": best_re[-1] * 100}

    if args.save_attn:
        attns_path = os.path.join(args.load_path, f"{os.path.splitext(args.test_file)[0]}.attns")
        print(f"saving attentions into {attns_path} ...")
        with open(attns_path, "wb") as f:
            pickle.dump(attns, f)

    return scores, output, official_results, results


def dump_to_file(offi: list, offi_path: str, scores: list, score_path: str, results: list = [], res_path: str = "",
                 thresh: float = None):
    '''
    dump scores and (top-k) predictions to file.

    '''
    print(f"saving official predictions into {offi_path} ...")
    json.dump(offi, open(offi_path, "w"))

    print(f"saving evaluations into {score_path} ...")
    headers = ["precision", "recall", "F1"]
    scores_pd = pd.DataFrame.from_dict(scores, orient="index", columns=headers)
    print(scores_pd)
    scores_pd.to_csv(score_path, sep='\t')

    if len(results) != 0:
        assert res_path != ""
        print(f"saving topk results into {res_path} ...")
        json.dump(results, open(res_path, "w"))

    if thresh != None:
        thresh_path = os.path.join(os.path.dirname(offi_path), "thresh")
        if not os.path.exists(thresh_path):
            print(f"saving threshold into {thresh_path} ...")
            json.dump(thresh, open(thresh_path, "w"))

    return


def print_matrix(output, tag):
    print("................... " + tag + " scores of best ckpt ...................")
    headers = ["precision", "recall", "F1"]
    scores_pd = pd.DataFrame.from_dict(output, orient="index", columns=headers)
    print(scores_pd)


def main():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    wandb.init(project="DocRED", name=args.display_name)

    # create directory to save checkpoints and predicted files
    time = str(datetime.datetime.now()).replace(' ', '_')
    save_path_ = os.path.join(args.save_path, f"{time}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    value_model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    policy_model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.transformer_type = args.transformer_type

    set_seed(args)

    read = read_docred
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id

    model = DocREModel(config, value_model, policy_model, tokenizer,
                       num_labels=args.num_labels,
                       max_sent_num=args.max_sent_num,
                       evi_thresh=args.evi_thresh,
                       sample_rate=args.sample_rate,
                       add_recall=args.add_recall,
                       rl_weight=args.rl_weight,
                       add_random=args.add_random,
                       sample_times=args.sample_times,
                       ns_rate_for_value=args.ns_rate_for_value,
                       add_value=args.add_value,
                       reward_type=args.reward_type)
    model.to(args.device)

    if args.do_train:  # Training
        if args.load_value_path != "":  # load model from existing checkpoint
            value_model_path = os.path.join(args.load_value_path, "best.ckpt")
            value_model_dict = torch.load(value_model_path)
            net_dict = model.state_dict()
            to_update_dict = {}
            for k, v in value_model_dict.items():
                if k.split('.')[0] not in ['head_extractor', 'tail_extractor', 'bilinear']:
                    to_update_dict['value_' + k] = v
                else:
                    to_update_dict[k] = v
                to_update_dict['policy_' + k] = v
            net_dict.update(to_update_dict)
            model.load_state_dict(net_dict)
            print("Load pretrained model---------------------------------------")

        create_directory(save_path_)
        args.save_path = save_path_

        train_file = os.path.join(args.data_dir, args.train_file)
        dev_file = os.path.join(args.data_dir, args.dev_file)
        test_file = os.path.join(args.data_dir, args.test_file)

        all_train_file = os.path.join(args.data_dir, "train_revised.json")
        all_dev_file = os.path.join(args.data_dir, "dev_revised.json")

        train_features = read(train_file, tokenizer, transformer_type=args.transformer_type,
                              max_seq_length=args.max_seq_length, teacher_sig_path=args.teacher_sig_path)
        dev_features = read(dev_file, tokenizer, transformer_type=args.transformer_type,
                            max_seq_length=args.max_seq_length)

        test_features = read(test_file, tokenizer, transformer_type=args.transformer_type,
                             max_seq_length=args.max_seq_length)

        all_train_features = read(all_train_file, tokenizer, transformer_type=args.transformer_type,
                                  max_seq_length=args.max_seq_length, teacher_sig_path=args.teacher_sig_path)
        all_dev_features = read(all_dev_file, tokenizer, transformer_type=args.transformer_type,
                                max_seq_length=args.max_seq_length)

        # ---------------ckpt performace-------------------------------------

        '''
        all_train_scores, all_train_output, _, _ = evaluate(args, model, all_train_features, tag="all_train")
        print_matrix(all_train_output, 'all_train')
        all_dev_scores, all_dev_output, _, _ = evaluate(args, model, all_dev_features, tag="all_dev")
        print_matrix(all_dev_output, 'all_dev')

        train_scores, train_output, _, _ = evaluate(args, model, train_features, tag="train")
        print_matrix(train_output, 'train')
        '''
        dev_scores, dev_output, _, _ = evaluate(args, model, dev_features, tag="dev")
        print_matrix(dev_output, 'dev')
        test_scores, test_output, _, _ = evaluate(args, model, test_features, tag="test")
        print_matrix(test_output, 'test')

        # ---------------ckpt performace-------------------------------------

        train(args, model, tokenizer, train_features, dev_features, test_features, all_train_features, all_dev_features)

        model.load_state_dict(torch.load(os.path.join(args.save_path, "best.ckpt")))

        all_train_scores, all_train_output, _, _ = evaluate(args, model, all_train_features, tag="all_train")
        print_matrix(all_train_output, 'all_train')
        all_dev_scores, all_dev_output, _, _ = evaluate(args, model, all_dev_features, tag="all_dev")
        print_matrix(all_dev_output, 'all_dev')

        train_scores, train_output, _, _ = evaluate(args, model, train_features, tag="train")
        print_matrix(train_output, 'train')
        dev_scores, dev_output, _, _ = evaluate(args, model, dev_features, tag="dev")
        print_matrix(dev_output, 'dev')
        test_scores, test_output, _, _ = evaluate(args, model, test_features, tag="test")
        print_matrix(test_output, 'test')


    else:  # Testing
        value_model_path = os.path.join(args.load_value_path, "last.ckpt")
        value_model_dict = torch.load(value_model_path)
        model.load_state_dict(value_model_dict)

        basename = os.path.splitext(args.test_file)[0]
        test_file = os.path.join(args.data_dir, args.test_file)

        test_features = read(test_file, tokenizer, transformer_type=args.transformer_type,
                             max_seq_length=args.max_seq_length)

        if args.eval_mode != "fushion":
            if args.do_prune:
                test_scores, test_output, official_results, results = evaluate(args, model, test_features, tag="test",
                                                                               pruner=pruner)
            else:
                test_scores, test_output, official_results, results = evaluate(args, model, test_features, tag="test")

            wandb.log(test_scores)

            offi_path = os.path.join(args.load_path, args.pred_file)
            score_path = os.path.join(args.load_path, f"{basename}_scores.csv")
            res_path = os.path.join(args.load_path, f"topk_{args.pred_file}")

            dump_to_file(official_results, offi_path, test_output, score_path, results, res_path)

        else:  # inference stage fusion

            results = json.load(open(os.path.join(args.load_path, f"topk_{args.pred_file}")))

            # formulate pseudo documents from top-k (k=num_labels in arguments) predictions
            pseudo_test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length,
                                        single_results=results)

            pseudo_test_scores, pseudo_output, pseudo_official_results, pseudo_results = evaluate(args, model,
                                                                                                  pseudo_test_features,
                                                                                                  tag="test")

            if 'thresh' in os.listdir(args.load_path):
                with open(os.path.join(args.load_path, "thresh")) as f:
                    thresh = json.load(f)
                print(f"Threshold loaded from file: {thresh}")
            else:
                thresh = None

            merged_offi, thresh = merge_results(results, pseudo_results, test_features, thresh)
            merged_re, merged_evi, merged_re_ign, _ = official_evaluate(merged_offi, args.data_dir, args.train_file,
                                                                        args.test_file)

            tag = args.test_file.split('.')[0]
            merged_output = {
                tag + "_rel": [i * 100 for i in merged_re],
                tag + "_rel_ign": [i * 100 for i in merged_re_ign],
                tag + "_evi": [i * 100 for i in merged_evi],
            }

            wandb.log({"dev_F1": merged_re[-1] * 100, "dev_evi_F1": merged_evi[-1] * 100,
                       "dev_F1_ign": merged_re_ign[-1] * 100})

            offi_path = os.path.join(args.load_path, f"fused_{args.pred_file}")
            score_path = os.path.join(args.load_path, f"{basename}_fused_scores.csv")
            dump_to_file(merged_offi, offi_path, merged_output, score_path, thresh=thresh)


if __name__ == "__main__":
    main()
