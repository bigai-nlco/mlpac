import torch
import torch.nn as nn
from opt_einsum import contract
import torch.nn.functional as F
from long_seq import process_long_input
from losses import ATLoss
from torch.distributions import Categorical
import numpy as np


class DocREModel(nn.Module):

    def __init__(self, config, value_model, policy_model, tokenizer,
                emb_size=768, block_size=64, num_labels=-1,
                max_sent_num=25, evi_thresh=0.2, sample_rate=1.0, add_recall=False,
                add_random=False, rl_weight=10.0, sample_times=100, ns_rate_for_value=1.0, add_value=True, reward_type='recall'):
        '''
        Initialize the model.
        :model: Pretrained langage model encoder;
        :tokenizer: Tokenzier corresponding to the pretrained language model encoder;
        :emb_size: Dimension of embeddings for subject/object (head/tail) representations;
        :block_size: Number of blocks for grouped bilinear classification;
        :num_labels: Maximum number of relation labels for each entity pair;
        :max_sent_num: Maximum number of sentences for each document;
        :evi_thresh: Threshold for selecting evidence sentences.
        '''
        
        super().__init__()
        self.config = config
        self.value_model = value_model
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size

        self.loss_fnt = ATLoss()
        self.loss_fnt_evi = nn.KLDivLoss(reduction="batchmean")

        self.head_extractor = nn.Linear(self.hidden_size * 2, emb_size)
        self.tail_extractor = nn.Linear(self.hidden_size * 2, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.policy_head_extractor = nn.Linear(self.hidden_size * 2, emb_size)
        self.policy_tail_extractor = nn.Linear(self.hidden_size * 2, emb_size)
        self.policy_bilinear = nn.Linear(emb_size * block_size, config.num_labels)
        self.BCE_loss = nn.BCEWithLogitsLoss(reduction='none')

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.total_labels = config.num_labels
        self.max_sent_num = max_sent_num
        self.evi_thresh = evi_thresh
        self.sample_rate = sample_rate
        self.add_recall = add_recall
        self.add_random = add_random
        self.rl_weight = rl_weight
        self.sample_times = sample_times
        self.ns_rate_for_value = ns_rate_for_value
        self.add_value = add_value
        self.reward_type = reward_type

    def encode(self, input_ids, attention_mask):
        
        '''
        Get the embedding of each token. For long document that has more than 512 tokens, split it into two overlapping chunks.
        Inputs:
            :input_ids: (batch_size, doc_len)
            :attention_mask: (batch_size, doc_len)
        Outputs:
            :sequence_output: (batch_size, doc_len, hidden_dim)
            :attention: (batch_size, num_attn_heads, doc_len, doc_len)
        '''
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        # process long documents.
        value_sequence_output, value_attention = process_long_input(self.value_model, input_ids, attention_mask, start_tokens, end_tokens)
        policy_sequence_output, policy_attention = process_long_input(self.policy_model, input_ids, attention_mask, start_tokens, end_tokens)

        return value_sequence_output, value_attention, policy_sequence_output, policy_attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts, offset):

        '''
        Get head, tail, context embeddings from token embeddings.
        Inputs:
            :sequence_output: (batch_size, doc_len, hidden_dim)
            :attention: (batch_size, num_attn_heads, doc_len, doc_len)
            :entity_pos: list of list. Outer length = batch size, inner length = number of entities each batch.
            :hts: list of list. Outer length = batch size, inner length = number of combination of entity pairs each batch.
            :offset: 1 for bert and roberta. Offset caused by [CLS] token.
        Outputs:
            :hss: (num_ent_pairs_all_batches, emb_size)
            :tss: (num_ent_pairs_all_batches, emb_size)
            :rss: (num_ent_pairs_all_batches, emb_size)
            :ht_atts: (num_ent_pairs_all_batches, doc_len)
            :rels_per_batch: list of length = batch size. Each entry represents the number of entity pairs of the batch.
        '''
        
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        ht_atts = []

        for i in range(len(entity_pos)): # for each batch
            entity_embs, entity_atts = [], []
            
            # obtain entity embedding from mention embeddings.
            for eid, e in enumerate(entity_pos[i]): # for each entity
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for mid, (start, end) in enumerate(e): # for every mention
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])

                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)

                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)

            # obtain subject/object (head/tail) embeddings from entity embeddings.
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
                
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])

            ht_att = (h_att * t_att).mean(1) # average over all heads        
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30) 
            ht_atts.append(ht_att)
            
            # obtain local context embeddings.
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        
        rels_per_batch = [len(b) for b in hss]
        hss = torch.cat(hss, dim=0) # (num_ent_pairs_all_batches, emb_size)
        tss = torch.cat(tss, dim=0) # (num_ent_pairs_all_batches, emb_size)
        rss = torch.cat(rss, dim=0) # (num_ent_pairs_all_batches, emb_size)
        ht_atts = torch.cat(ht_atts, dim=0) # (num_ent_pairs_all_batches, max_doc_len)

        return hss, rss, tss, ht_atts, rels_per_batch


    def forward_rel(self, hs, ts, rs):
        '''
        Forward computation for RE.
        Inputs:
            :hs: (num_ent_pairs_all_batches, emb_size)
            :ts: (num_ent_pairs_all_batches, emb_size)
            :rs: (num_ent_pairs_all_batches, emb_size)
        Outputs:
            :logits: (num_ent_pairs_all_batches, num_rel_labels)
        '''
        
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=-1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=-1)))
        # split into several groups.
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)

        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)
        return logits

    def forward_policy(self, hs, ts, rs):
        hs = torch.tanh(self.policy_head_extractor(torch.cat([hs, rs], dim=-1)))
        ts = torch.tanh(self.policy_tail_extractor(torch.cat([ts, rs], dim=-1)))

        # split into several groups.
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)

        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.policy_bilinear(bl)
        return logits

    def forward_evi(self, doc_attn, sent_pos, batch_rel, offset):
        '''
        Forward computation for ER.
        Inputs:
            :doc_attn: (num_ent_pairs_all_batches, doc_len), attention weight of each token for computing localized context pooling.
            :sent_pos: list of list. The outer length = batch size. The inner list contains (start, end) position of each sentence in each batch.
            :batch_rel: list of length = batch size. Each entry represents the number of entity pairs of the batch.
            :offset: 1 for bert and roberta. Offset caused by [CLS] token.
        Outputs:
            :s_attn:  (num_ent_pairs_all_batches, max_sent_all_batch), sentence-level evidence distribution of each entity pair.
        '''
        
        max_sent_num = max([len(sent) for sent in sent_pos])
        rel_sent_attn = []
        for i in range(len(sent_pos)): # for each batch
            # the relation ids corresponds to document in batch i is [sum(batch_rel[:i]), sum(batch_rel[:i+1]))
            curr_attn = doc_attn[sum(batch_rel[:i]):sum(batch_rel[:i+1])]
            curr_sent_pos = [torch.arange(s[0], s[1]).to(curr_attn.device) + offset for s in sent_pos[i]] # + offset

            curr_attn_per_sent = [curr_attn.index_select(-1, sent) for sent in curr_sent_pos]
            curr_attn_per_sent += [torch.zeros_like(curr_attn_per_sent[0])] * (max_sent_num - len(curr_attn_per_sent))
            sum_attn = torch.stack([attn.sum(dim=-1) for attn in curr_attn_per_sent], dim=-1) # sum across those attentions
            rel_sent_attn.append(sum_attn)

        s_attn = torch.cat(rel_sent_attn, dim=0)
        return s_attn


    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None, # relation labels
                entity_pos=None,
                hts=None, # entity pairs
                sent_pos=None, 
                sent_labels=None, # evidence labels (0/1)
                teacher_attns=None, # evidence distribution from teacher model
                arg_labels=None,
                tag="train",
                func="value"
                ):

        ## 合并labels和arg_labels
        com_labels = labels + arg_labels
        com_labels = torch.where(com_labels > 0, 1, 0)
        com_labels[:, 0] = (com_labels[:, 1:].sum(1) == 0.)

        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        value_sequence_output, value_attention, policy_sequence_output, policy_attention = self.encode(input_ids, attention_mask)

        # value_hs, value_rs, value_ts, doc_attn, batch_rel = self.get_hrt(value_sequence_output, value_attention, entity_pos, hts, offset)

        # value_logits = self.forward_rel(value_hs, value_ts, value_rs)
        # assert not torch.any(torch.isnan(value_logits)), 'nan in value logits'
        if func == "value":
            value_hs, value_rs, value_ts, doc_attn, batch_rel = self.get_hrt(value_sequence_output, value_attention, entity_pos, hts, offset)

            value_logits = self.forward_rel(value_hs, value_ts, value_rs)
            output = {}
            output["rel_pred"] = self.loss_fnt.get_label(value_logits, num_labels=self.num_labels)

            if sent_labels != None:  # human-annotated evidence available

                s_attn = self.forward_evi(doc_attn, sent_pos, batch_rel, offset)
                output["evi_pred"] = F.pad(s_attn > self.evi_thresh, (0, self.max_sent_num - s_attn.shape[-1]))
            '''
            if tag in ["test", "dev"]:  # testing
                scores_topk = self.loss_fnt.get_score(value_logits, self.num_labels)
                output["scores"] = scores_topk[0]
                output["topks"] = scores_topk[1]
            '''
            loss = self.loss_fnt(value_logits.float(), com_labels.float(), self.ns_rate_for_value)
            output["rel_loss"] = {"rel_loss": loss.to(value_sequence_output)}
            
            if sent_labels != None:  # supervised training with human evidence
                idx_used = torch.nonzero(com_labels[:, 1:].sum(dim=-1)).view(-1)
                # evidence retrieval loss (kldiv loss)
                s_attn = s_attn[idx_used]
                sent_labels = sent_labels[idx_used]
                norm_s_labels = sent_labels / (sent_labels.sum(dim=-1, keepdim=True) + 1e-30)
                norm_s_labels[norm_s_labels == 0] = 1e-30
                s_attn[s_attn == 0] = 1e-30
                evi_loss = self.loss_fnt_evi(s_attn.log(), norm_s_labels)
                output["rel_loss"]["evi_loss"] = evi_loss.to(value_sequence_output)
            return output
        elif func == "policy":
            with torch.no_grad():
                value_hs, value_rs, value_ts, _, _ = self.get_hrt(value_sequence_output, value_attention, entity_pos, hts, offset)

                value_logits = self.forward_rel(value_hs, value_ts, value_rs)
            output = {}
            policy_hs, policy_rs, policy_ts, _, _ = self.get_hrt(policy_sequence_output, policy_attention, entity_pos, hts,
                                                                 offset)

            policy_logits = self.forward_policy(policy_hs, policy_ts, policy_rs)
            output["rel_pred"] = self.loss_fnt.get_label(policy_logits, num_labels=self.num_labels)
            assert not torch.any(torch.isnan(policy_logits)), 'nan in policy logits'
            
            '''
            if tag in ['test', 'dev']:
                scores_topk = self.loss_fnt.get_score(policy_logits, self.num_labels)
                output["scores"] = scores_topk[0]
                output["topks"] = scores_topk[1]
            '''
            # else:
            policy_prob = torch.sigmoid(policy_logits - policy_logits[:, 0].unsqueeze(-1))[:, 1:]
            to_sample_prob = torch.cat([1 - policy_prob.unsqueeze(-1), policy_prob.unsqueeze(-1)], dim=-1)
            loss_list = []
            value_reward_list = []
            recall_reward_list = []
            reward_list = []
            output['reward'] = {}
            for _ in range(self.sample_times):
                m = Categorical(to_sample_prob)  # 生成分布
                action = m.sample()  # 从分布中采样
                if self.add_random:
                    if np.random.rand() < 0.1:
                        action = torch.where(torch.rand_like(action) < 0.5, 1, 0)
                
                # negative sampling
                random_mask = torch.where(torch.rand_like(action.float()) <= self.sample_rate, 1, 0)
                ground_truth_mask = torch.where(torch.eq(labels[:, 1:], 1), labels[:, 1:], random_mask)

                reward = torch.zeros_like(torch.sum(ground_truth_mask, dim=-1)).float()
                if self.add_value:
                    value_reward = self.get_reward_base_value(value_logits, action)
                    assert not torch.any(torch.isnan(value_reward)), 'nan in value reward'

                    value_reward_list.append(torch.sum(value_reward * ground_truth_mask))
                    reward += torch.sum(value_reward * ground_truth_mask, -1)

                if self.add_recall:
                    recall_reward = self.cal_recall_reward(action, labels[:, 1:] * ground_truth_mask, self.reward_type)
                    assert not torch.any(torch.isnan(recall_reward)), 'nan in recall reward'

                    recall_reward_list.append(torch.sum(recall_reward))
                    reward += recall_reward * self.rl_weight
                reward_list.append(torch.sum(reward))

                log_prob = m.log_prob(action) * ground_truth_mask
                sum_log_prob = torch.sum(log_prob, dim=-1)
                assert not torch.any(torch.isnan(sum_log_prob)), 'nan in sum_log_prob'

                loss = torch.sum(-sum_log_prob * reward, dim=-1)
                loss_list.append(loss.to(policy_sequence_output))
            if value_reward_list:
                output['reward']['value_reward'] = sum(value_reward_list) / len(value_reward_list)
            if recall_reward_list:
                output['reward']['recall_reward'] = sum(recall_reward_list) / len(recall_reward_list)
            if reward_list:
                output['reward']['reward'] = sum(reward_list) / len(reward_list)
            if loss_list:
                output['rl_loss'] = {"rl_loss": sum(loss_list) / len(loss_list)}
            return output
        elif func == 'argument':
            output = {}
            with torch.no_grad():
                value_hs, value_rs, value_ts, _, _ = self.get_hrt(value_sequence_output, value_attention, entity_pos, hts, offset)

                value_logits = self.forward_rel(value_hs, value_ts, value_rs)

                output["value_rel_pred"] = self.loss_fnt.get_label(value_logits, num_labels=self.num_labels)

                policy_hs, policy_rs, policy_ts, _, _ = self.get_hrt(policy_sequence_output, policy_attention, entity_pos, hts, offset)

                policy_logits = self.forward_policy(policy_hs, policy_ts, policy_rs)
                output["policy_rel_pred"] = self.loss_fnt.get_label(policy_logits, num_labels=self.num_labels)
                policy_prob = torch.sigmoid(policy_logits - policy_logits[:, 0].unsqueeze(-1))[:, 1:]
                output['prob'] = policy_prob
                return output

    def cal_recall_reward(self, pred, labels, tag):
        corr_num = (pred * labels).sum(-1)
        gold_num = labels.sum(-1)
        pred_num = pred.sum(-1)
        recall = corr_num / (gold_num + 1e-9)
        precision = corr_num / (pred_num + 1e-9)
        f1 = 2*precision*recall / (precision+recall+1e-9)
        #no_recall = ((1 - pred) * labels).sum(-1) / (gold_num + 1e-9)
        #return recall - no_recall
        if tag == 'recall':
            return recall
        if tag == 'f1':
            return f1
        if tag == 'precision':
            return precision

    def get_label(self, prob):
        output = torch.zeros_like(prob).to(prob)
        mask = (prob > 0.5)
        output[mask] = 1
        return output

    def get_reward_base_value(self, logits, actions):
        pos_action_reward = torch.clamp(torch.log(
            torch.sigmoid(logits - logits[:, 0].unsqueeze(-1))[:, 1:] / (torch.sigmoid(
                logits[:, 0].unsqueeze(-1) - logits)[:, 1:])), min=-1, max=1)
        neg_action_reward = torch.clamp(torch.log(
            torch.sigmoid(logits[:, 0].unsqueeze(-1) - logits)[:, 1:] / (torch.sigmoid(
                logits - logits[:, 0].unsqueeze(-1))[:, 1:])), min=-1, max=1)

        neg_action_sum = torch.sum(1 - actions, dim=-1).unsqueeze(-1)
        reward = torch.where(torch.eq(actions, 1), pos_action_reward, neg_action_reward / (neg_action_sum + 1e-30))

        # # negative sampling
        # random_mask = torch.where(torch.rand_like(actions.float()) < self.sample_rate, 1, 0)
        # ns_actions = torch.where(torch.eq(actions, 1), actions, random_mask)

        return reward
