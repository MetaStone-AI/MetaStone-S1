# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


def get_score_mask(x, target_token_ids, think_start_token_id, think_end_token_id): # (b,L) tensor, int list, int, int
    target_tensor = torch.tensor(target_token_ids, device=x.device).view(1, 1, -1)  # (1, 1, T)
    x_expanded = x.unsqueeze(-1)  # (B, L, 1)
    mask = (x_expanded == target_tensor).any(dim=-1)  # (B, L)

    mask_shifted = torch.nn.functional.pad(mask[:, :-1], (1, 0), value=False)
    mask_first_only = mask & ~mask_shifted

    think_pos = (x == think_start_token_id)
    think_next_mask = torch.nn.functional.pad(think_pos[:, :-1], (1, 0), value=False)  
    final_mask = mask_first_only & (~think_next_mask)

    B, L = x.shape
    is_start = (x == think_start_token_id)
    is_end = (x == think_end_token_id)

    start_indices = torch.where(is_start.any(dim=1),
                                is_start.float().argmax(dim=1),
                                torch.zeros(B, dtype=torch.long, device=x.device))

    reversed_is_end = torch.flip(is_end, dims=[1])
    end_indices_from_end = reversed_is_end.float().argmax(dim=1)
    end_indices = L - 1 - end_indices_from_end

    positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)

    think_mask = (positions >= start_indices.unsqueeze(1)) & (positions <= end_indices.unsqueeze(1))
    final_mask = final_mask & think_mask

    return final_mask

def get_filter_mask(x, correct_mask, *args, **kwargs):
    mask = torch.zeros_like(x, dtype=torch.bool)  
    mask[(x>0) & correct_mask] = 1
    mask[(x<0) & (~correct_mask)] = 1
    return mask

def geommean_scores(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = (x != 0).float()
    masked_x = torch.where(mask.bool(), torch.sigmoid(x), torch.ones_like(x))

    log_x = torch.log(masked_x + eps) * mask 
    count = mask.sum(dim=1) + eps 
    geom_mean = torch.exp(log_x.sum(dim=1) / count)  

    all_zero_mask = mask.sum(dim=1) == 0
    geom_mean[all_zero_mask] = 0.0

    return geom_mean  # shape: (b,)


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        score_module = None,
        score_optimizer = None,
        tokenizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

        self.use_score = score_module is not None
        if self.use_score:
            self.score_module = score_module
            self.score_optimizer = score_optimizer
            self.tokenizer = tokenizer
            self.score_id = self.tokenizer.encode('.\n\n', add_special_tokens=False)
            self.think_start_id = self.tokenizer.encode('<think>', add_special_tokens=False)[0]
            self.think_end_id = self.tokenizer.encode('</think>', add_special_tokens=False)[0]

    def _validate_score(self, responses, reward_tensor):
        pass

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs


    def _forward_micro_batch_with_score(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            advantages = micro_batch['advantages']
            reward = micro_batch['rollout_level_scores'].max(dim=1)[0]

            score_mask = get_score_mask(input_ids, self.score_id, self.think_start_id, self.think_end_id)
            response_masks = torch.cat([torch.zeros_like(input_ids[:,:seqlen-response_length]),torch.ones_like(input_ids[:,:response_length])],dim=1)
            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                response_masks_rmpad, *_ = unpad_input(response_masks.unsqueeze(-1), attention_mask)
                response_masks_rmpad = response_masks_rmpad.transpose(0, 1)

                score_mask_rmpad, *_ = unpad_input(score_mask.unsqueeze(-1), attention_mask)
                score_mask_rmpad = score_mask_rmpad.transpose(0, 1)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                score_mask_rmpad_rolled = torch.roll(score_mask_rmpad, shifts=-1, dims=1)
                response_masks_rmpad_rolled = torch.roll(response_masks_rmpad, shifts=-1, dims=1)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False,
                                           output_hidden_states=True)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                output_features = output.hidden_states[-2][0]
                output_features = output_features*0.0001 + output_features.detach()*(1-0.0001)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                    output_features = gather_outpus_and_unpad(output_features,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                    input_ids_rmpad_rolled = gather_outpus_and_unpad(input_ids_rmpad_rolled, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                score_loss_mask = (score_mask_rmpad_rolled==1) & (response_masks_rmpad_rolled==1) #(1,L)
                assert score_loss_mask.shape[0]==1
                if score_loss_mask.any():
                    score_probs = torch.zeros_like(output_features[:,0]) #L
                    score_features = output_features[score_loss_mask[0]] #L,C
                    score_probs[score_loss_mask[0]] = self.score_module(score_features).squeeze(1) #L
                    score_probs = pad_input(hidden_states=score_probs.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                    score_probs = score_probs.squeeze(2)[:, -response_length - 1:-1]
                    correct_mask = (reward.unsqueeze(1).expand((batch_size, response_length)) >0.5)
                    filter_mask = get_filter_mask(score_probs, correct_mask)
                    criterion = nn.BCEWithLogitsLoss(reduction='none').to(score_probs.device)
                    score_loss = criterion(score_probs, correct_mask.float())
                    valid_mask = (reward.unsqueeze(1).expand((batch_size, response_length))!=0.1).float()

                    final_mask = valid_mask * filter_mask
                    pos_mask = final_mask * correct_mask.float()
                    neg_mask = final_mask * (1 - correct_mask.float())

                    pos_count = pos_mask.sum()+1
                    dist.all_reduce(pos_count, op=dist.ReduceOp.SUM)
                    neg_count = neg_mask.sum()+1
                    dist.all_reduce(neg_count, op=dist.ReduceOp.SUM)
                    # print(pos_count.detach().item(), neg_count.detach().item())
                    total = pos_count + neg_count
                    pos_weight = neg_count / total
                    neg_weight = pos_count / total
                    weights = 1.5 * pos_mask * pos_weight + neg_mask * neg_weight
                    score_loss_mean = (score_loss * weights).sum() / (weights.sum() + 1e-5)
                    
                    with torch.no_grad(): 
                        mean_prob = geommean_scores(score_probs*valid_mask)
                        # print(mean_prob.detach().tolist(), reward.detach().tolist())
                        correct_prob = mean_prob[reward>0.5].sum().detach().item()
                        correct_num = (reward>0.5).sum().detach().item()
                        false_prob = mean_prob[reward==0].sum().detach().item()
                        false_num = (reward==0).sum().detach().item()
                else:
                    score_features = output_features[:1]
                    score_probs = self.score_module(score_features)
                    score_loss_mean = score_probs.sum() * 0.0
                    correct_prob, correct_num, false_prob, false_num = 0,0,0,0
                    
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs, score_loss_mean, (correct_prob, correct_num, false_prob, false_num)

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        if self.use_score:
            if isinstance(self.score_module, FSDP):
                grad_norm_fc = self.score_module.clip_grad_norm_(max_norm=self.config.grad_clip)
            else:
                grad_norm_fc = torch.nn.utils.clip_grad_norm_(self.score_module.parameters(), max_norm=self.config.grad_clip)
            self.score_optimizer.step()
            # print([x.grad for x in self.score_optimizer.param_groups[0]['params']])
            # print(next(self.score_module.parameters()))
        else:
            grad_norm_fc = None
        return grad_norm, grad_norm_fc

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages', 'rollout_level_scores']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            self.actor_optimizer.zero_grad()
            self.score_optimizer.zero_grad()
            correct_prob, correct_num, false_prob, false_num = 0,0,0,0
            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                policy_loss = 0

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                if self.use_score:
                    entropy, log_prob, score_loss, score_metrics = self._forward_micro_batch_with_score(micro_batch=data, temperature=temperature)
                    policy_loss += score_loss
                    correct_prob += score_metrics[0]
                    correct_num += score_metrics[1]
                    false_prob += score_metrics[2]
                    false_num += score_metrics[3]
                    metrics['fc/score_loss'] = score_loss.detach().item()
                else:
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                              log_prob=log_prob,
                                                                              advantages=advantages,
                                                                              eos_mask=response_mask,
                                                                              cliprange=clip_ratio)
                if pg_loss>100:
                    print(f'!!!!!!!!!{ppo_kl}!!!!!!!!!')
                    pg_loss = pg_loss * 0
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss += pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                if self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                else:
                    loss = policy_loss / self.gradient_accumulation
                loss.backward()

                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                append_to_dict(metrics, data)

            grad_norm, grad_norm_fc = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)

            data = {
                'fc/grad_norm': grad_norm_fc.detach().item(),
                'fc/correct_prob': correct_prob/(correct_num+0.001), 'fc/false_prob': false_prob/(false_num+0.001),
                'fc/correct_num': correct_num, 'fc/false_num': false_num, 'fc/diff_prob': correct_prob/(correct_num+0.001)-false_prob/(false_num+0.001),
            }
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        if self.use_score:
            self.score_optimizer.zero_grad()
        return metrics
