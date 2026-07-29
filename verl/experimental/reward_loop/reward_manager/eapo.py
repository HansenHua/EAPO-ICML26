# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

import inspect
import re
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.eapo import (
    QPhiScore,
    build_explore_memory_target,
    compute_q_phi_reinforce_loss,
    compute_target_log_prob,
    log_prob_to_probability,
)
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.fs import copy_to_local
from verl.utils.reward_score import default_compute_score


_EXPLORE_RE = re.compile(r"<explore>(.*?)</explore>", re.IGNORECASE | re.DOTALL)
_MEMORY_RE = re.compile(r"<memory>(.*?)</memory>", re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\s*\{.*?\}", re.DOTALL)


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _format_reward(response: str, require_boxed_action: bool = True) -> float:
    has_explore = _EXPLORE_RE.search(response) is not None
    has_memory = _MEMORY_RE.search(response) is not None
    has_action = _BOXED_RE.search(response) is not None
    if require_boxed_action:
        return 1.0 if has_explore and has_memory and has_action else 0.0
    return 1.0 if has_explore and has_memory else 0.0


def _tag_text(pattern: re.Pattern, text: str) -> str:
    match = pattern.search(text)
    return match.group(1).strip() if match is not None else ""


def _get_mapping_value(mapping: Any, key: str, default: Any = None) -> Any:
    if mapping is None:
        return default
    if hasattr(mapping, "get"):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _get_non_tensor_value(data_item: DataProto, key: str, default: Any = None) -> Any:
    if key in data_item.non_tensor_batch:
        return data_item.non_tensor_batch[key]
    extra_info = data_item.non_tensor_batch.get("extra_info", {})
    return _get_mapping_value(extra_info, key, default)


def _reward_extra_output(reward_extra_infos: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], list[str]]:
    reward_extra_keys = sorted(set().union(*(info.keys() for info in reward_extra_infos))) if reward_extra_infos else []
    non_tensor_batch = {}
    for key in reward_extra_keys:
        non_tensor_batch[key] = np.array([info.get(key, None) for info in reward_extra_infos], dtype=object)
    return non_tensor_batch, reward_extra_keys


@register("eapo")
class EAPORewardManager(RewardManagerBase):
    """Reward manager for Exploration-Aware Policy Optimization.

    The custom reward function can return either a scalar task reward or a dict.
    Dict outputs may include ``eapo_explore_reward`` and ``eapo_format_reward``;
    absent format rewards are inferred from the structured response tags.
    """

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        reward_kwargs = config.reward.get("reward_kwargs", {})
        eapo_reward_cfg = reward_kwargs.get("eapo", {})
        algorithm_eapo_cfg = config.algorithm.get("eapo", {})

        self.task_reward_key = _cfg_get(eapo_reward_cfg, "task_reward_key", "score")
        self.format_reward_key = _cfg_get(eapo_reward_cfg, "format_reward_key", "eapo_format_reward")
        self.explore_reward_key = _cfg_get(eapo_reward_cfg, "explore_reward_key", "eapo_explore_reward")
        self.format_reward_weight = _cfg_get(
            eapo_reward_cfg,
            "format_reward_weight",
            _cfg_get(algorithm_eapo_cfg, "format_reward_weight", 0.5),
        )
        self.explore_reward_weight = _cfg_get(
            eapo_reward_cfg,
            "explore_reward_weight",
            _cfg_get(algorithm_eapo_cfg, "explore_reward_weight", 1.0),
        )
        self.explore_discount = _cfg_get(
            eapo_reward_cfg,
            "explore_discount",
            _cfg_get(algorithm_eapo_cfg, "explore_discount", 0.9),
        )
        self.q_phi_enable = _cfg_get(eapo_reward_cfg, "q_phi_enable", _cfg_get(algorithm_eapo_cfg, "q_phi_enable", True))
        self.q_phi_probability_mode = _cfg_get(
            eapo_reward_cfg,
            "q_phi_probability_mode",
            _cfg_get(algorithm_eapo_cfg, "q_phi_probability_mode", "sequence"),
        )
        self.q_phi_model_path = _cfg_get(
            eapo_reward_cfg,
            "q_phi_model_path",
            _cfg_get(algorithm_eapo_cfg, "q_phi_model_path", None),
        )
        # q_phi is a separate language model from the policy, but it is
        # initialized from the same checkpoint by default.
        if self.q_phi_model_path is None and self.q_phi_enable:
            self.q_phi_model_path = config.actor_rollout_ref.model.path
        self.q_phi_device = _cfg_get(eapo_reward_cfg, "q_phi_device", _cfg_get(algorithm_eapo_cfg, "q_phi_device", "auto"))
        self.q_phi_dtype = _cfg_get(eapo_reward_cfg, "q_phi_dtype", _cfg_get(algorithm_eapo_cfg, "q_phi_dtype", "auto"))
        self.q_phi_train_enable = self.q_phi_enable and _cfg_get(
            eapo_reward_cfg,
            "q_phi_train_enable",
            _cfg_get(algorithm_eapo_cfg, "q_phi_train_enable", True),
        )
        self.q_phi_lr = float(_cfg_get(eapo_reward_cfg, "q_phi_lr", _cfg_get(algorithm_eapo_cfg, "q_phi_lr", 1e-4)))
        self.q_phi_beta = float(
            _cfg_get(eapo_reward_cfg, "q_phi_beta", _cfg_get(algorithm_eapo_cfg, "q_phi_beta", 1.0))
        )
        self.q_phi_kl_coef = float(
            _cfg_get(eapo_reward_cfg, "q_phi_kl_coef", _cfg_get(algorithm_eapo_cfg, "q_phi_kl_coef", 1.0))
        )
        self.q_phi_ref_model_enable = self.q_phi_train_enable and _cfg_get(
            eapo_reward_cfg,
            "q_phi_ref_model_enable",
            _cfg_get(algorithm_eapo_cfg, "q_phi_ref_model_enable", True),
        )
        self.q_phi_ref_model_path = _cfg_get(
            eapo_reward_cfg,
            "q_phi_ref_model_path",
            _cfg_get(algorithm_eapo_cfg, "q_phi_ref_model_path", None),
        )
        if self.q_phi_ref_model_path is None:
            self.q_phi_ref_model_path = self.q_phi_model_path
        max_grad_norm = _cfg_get(
            eapo_reward_cfg,
            "q_phi_max_grad_norm",
            _cfg_get(algorithm_eapo_cfg, "q_phi_max_grad_norm", 1.0),
        )
        self.q_phi_max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.q_phi_model = None
        self.q_phi_ref_model = None
        self.q_phi_optimizer = None
        self.require_boxed_action = _cfg_get(eapo_reward_cfg, "require_boxed_action", True)

    def _resolve_q_phi_device(self) -> torch.device:
        if self.q_phi_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.q_phi_device)

    def _resolve_q_phi_dtype(self):
        if self.q_phi_dtype == "auto":
            return None
        if self.q_phi_dtype in ("bf16", "bfloat16"):
            return torch.bfloat16
        if self.q_phi_dtype in ("fp16", "float16"):
            return torch.float16
        if self.q_phi_dtype in ("fp32", "float32"):
            return torch.float32
        raise ValueError(f"Unsupported q_phi_dtype: {self.q_phi_dtype}")

    def _ensure_q_phi_model(self):
        if not self.q_phi_enable or self.q_phi_model is not None:
            return
        from transformers import AutoModelForCausalLM

        model_path = copy_to_local(self.q_phi_model_path)
        model_kwargs = {"trust_remote_code": True}
        torch_dtype = self._resolve_q_phi_dtype()
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        self.q_phi_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.q_phi_model.to(self._resolve_q_phi_device())
        self.q_phi_model.eval()

    def _ensure_q_phi_ref_model(self):
        if not self.q_phi_ref_model_enable or self.q_phi_ref_model is not None:
            return
        from transformers import AutoModelForCausalLM

        model_path = copy_to_local(self.q_phi_ref_model_path)
        model_kwargs = {"trust_remote_code": True}
        torch_dtype = self._resolve_q_phi_dtype()
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        self.q_phi_ref_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.q_phi_ref_model.to(self._resolve_q_phi_device())
        self.q_phi_ref_model.eval()
        for parameter in self.q_phi_ref_model.parameters():
            parameter.requires_grad_(False)

    def _ensure_q_phi_optimizer(self):
        if not self.q_phi_train_enable or self.q_phi_optimizer is not None:
            return
        self._ensure_q_phi_model()
        self.q_phi_optimizer = torch.optim.AdamW(self.q_phi_model.parameters(), lr=self.q_phi_lr)

    def _build_q_phi_inputs(self, state_text: str, target_text: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        prompt_ids = self.tokenizer.encode(state_text, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
        if not target_ids:
            raise ValueError("q_phi target text must contain at least one token.")
        if not prompt_ids:
            prefix_token = self.tokenizer.bos_token_id
            if prefix_token is None:
                prefix_token = self.tokenizer.eos_token_id
            if prefix_token is None:
                raise ValueError("q_phi requires a non-empty state text or a tokenizer BOS/EOS token.")
            prompt_ids = [prefix_token]

        model_device = next(self.q_phi_model.parameters()).device
        input_ids = torch.tensor([prompt_ids + target_ids], dtype=torch.long, device=model_device)
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask, len(prompt_ids)

    def _q_phi_log_prob_tensor(
        self,
        model: torch.nn.Module,
        state_text: str,
        target_text: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, attention_mask, prompt_length = self._build_q_phi_inputs(state_text, target_text)
        return compute_target_log_prob(
            model=model,
            input_ids=input_ids,
            prompt_length=prompt_length,
            attention_mask=attention_mask,
        )

    def _q_phi_score(self, state_text: str, target_text: str) -> QPhiScore:
        if not self.q_phi_enable or not target_text:
            return QPhiScore(log_prob=float("-inf"), prob=0.0, token_count=0)
        self._ensure_q_phi_model()
        self.q_phi_model.eval()
        with torch.no_grad():
            log_prob, token_count = self._q_phi_log_prob_tensor(self.q_phi_model, state_text, target_text)
            prob = log_prob_to_probability(log_prob, token_count, mode=self.q_phi_probability_mode)
        return QPhiScore(log_prob=float(log_prob.item()), prob=float(prob.item()), token_count=int(token_count.item()))

    def _q_phi_train_step(self, state_text: str, target_text: str, q_value: float) -> dict[str, float]:
        if not self.q_phi_train_enable or not target_text:
            return {}

        self._ensure_q_phi_optimizer()
        self.q_phi_model.train()
        self.q_phi_optimizer.zero_grad(set_to_none=True)

        q_log_prob, token_count = self._q_phi_log_prob_tensor(self.q_phi_model, state_text, target_text)
        ref_log_prob = None
        if self.q_phi_ref_model_enable and self.q_phi_kl_coef != 0.0:
            self._ensure_q_phi_ref_model()
            with torch.no_grad():
                ref_log_prob, _ = self._q_phi_log_prob_tensor(self.q_phi_ref_model, state_text, target_text)

        loss_pack = compute_q_phi_reinforce_loss(
            q_log_prob=q_log_prob,
            q_value=q_value,
            ref_log_prob=ref_log_prob,
            beta=self.q_phi_beta,
            kl_coef=self.q_phi_kl_coef,
        )
        loss_pack.loss.backward()
        grad_norm = None
        if self.q_phi_max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.q_phi_model.parameters(), self.q_phi_max_grad_norm)
        self.q_phi_optimizer.step()
        self.q_phi_model.eval()

        metrics = {
            "loss": float(loss_pack.loss.detach().item()),
            "pg_loss": float(loss_pack.policy_gradient_loss.detach().item()),
            "kl_loss": float(loss_pack.kl_loss.detach().item()),
            "approx_kl": float(loss_pack.approximate_kl.detach().item()),
            "return": float(q_value),
            "token_count": float(token_count.detach().item()),
        }
        if grad_norm is not None:
            metrics["grad_norm"] = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
        return metrics

    async def run_single(self, data: DataProto) -> dict:
        data = data[-1:]
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch.get("data_source", "unknown")
        reward_model = data_item.non_tensor_batch.get("reward_model", {})
        ground_truth = _get_mapping_value(reward_model, "ground_truth", None)
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info: dict[str, Any] = {}
        if isinstance(result, dict):
            reward_extra_info.update(result)
            task_reward = _as_float(result.get(self.task_reward_key, result.get("score", 0.0)))
            format_reward = _as_float(
                result.get(self.format_reward_key),
                _format_reward(response_str, require_boxed_action=self.require_boxed_action),
            )
            explore_reward = _as_float(
                result.get(self.explore_reward_key, result.get("exploration_reward", 0.0)),
                0.0,
            )
        else:
            task_reward = _as_float(result)
            format_reward = _format_reward(response_str, require_boxed_action=self.require_boxed_action)
            explore_reward = 0.0

        explore = _tag_text(_EXPLORE_RE, response_str)
        memory = _tag_text(_MEMORY_RE, response_str)
        state_id = _get_non_tensor_value(data_item, "eapo_state_id", None)
        if state_id is None:
            state_id = _get_non_tensor_value(data_item, "state_id", None)
        if state_id is None:
            raw_prompt = data_item.non_tensor_batch.get("raw_prompt", "")
            state_id = repr(raw_prompt) if raw_prompt is not None else str(data_source)
        state_text = _get_non_tensor_value(data_item, "eapo_state", None)
        if state_text is None:
            state_text = data_item.non_tensor_batch.get("raw_prompt", None)
        if isinstance(state_text, np.ndarray):
            state_text = state_text.tolist()
        state_text = repr(state_text) if state_text is not None else str(state_id)
        visitation_depth = _get_non_tensor_value(data_item, "eapo_visitation_depth", 0)
        next_state = _get_non_tensor_value(data_item, "eapo_next_state", "")

        immediate_target = build_explore_memory_target(explore=explore, memory=memory)
        q_phi_train_return = task_reward
        q_phi_train_metrics = {}
        if self.q_phi_train_enable:
            q_phi_train_metrics["immediate"] = self._q_phi_train_step(
                str(state_text),
                immediate_target,
                q_phi_train_return,
            )

        explored_memory = f"{memory}\n{next_state}".strip() if next_state else ""
        explored_target = build_explore_memory_target(explore=explore, memory=explored_memory) if explored_memory else ""
        if self.q_phi_train_enable and explored_target:
            q_phi_train_metrics["explored"] = self._q_phi_train_step(
                str(state_text),
                explored_target,
                q_phi_train_return,
            )
        immediate_q_phi = self._q_phi_score(str(state_text), immediate_target)
        explored_q_phi = self._q_phi_score(str(state_text), explored_target)

        if self.q_phi_enable:
            explore_reward = max(
                immediate_q_phi.prob,
                (float(self.explore_discount) ** 2) * explored_q_phi.prob,
            )

        reward = (
            task_reward
            + float(self.format_reward_weight) * format_reward
            + float(self.explore_reward_weight) * explore_reward
        )

        reward_extra_info["score"] = reward
        reward_extra_info["eapo_task_reward"] = task_reward
        reward_extra_info["eapo_format_reward"] = format_reward
        reward_extra_info["eapo_explore_reward"] = explore_reward
        reward_extra_info["eapo_q_phi_immediate"] = immediate_q_phi.prob
        reward_extra_info["eapo_q_phi_immediate_log_prob"] = immediate_q_phi.log_prob
        reward_extra_info["eapo_q_phi_immediate_token_count"] = immediate_q_phi.token_count
        reward_extra_info["eapo_q_phi_explored"] = explored_q_phi.prob
        reward_extra_info["eapo_q_phi_explored_log_prob"] = explored_q_phi.log_prob
        reward_extra_info["eapo_q_phi_explored_token_count"] = explored_q_phi.token_count
        reward_extra_info["eapo_q_phi_train_enable"] = bool(self.q_phi_train_enable)
        for train_target, metrics in q_phi_train_metrics.items():
            for metric_name, metric_value in metrics.items():
                reward_extra_info[f"eapo_q_phi_train_{train_target}_{metric_name}"] = metric_value
        reward_extra_info["eapo_total_reward"] = reward

        return {
            "reward_score": reward,
            "reward_extra_info": reward_extra_info,
            "eapo_q_phi_input": {
                "state_id": str(state_id),
                "visitation_depth": str(visitation_depth),
                "next_state": str(next_state),
                "explore": explore,
                "memory": memory,
                "task_reward": task_reward,
                "format_reward": format_reward,
                "external_explore_reward": explore_reward,
                "format_reward_weight": float(self.format_reward_weight),
                "explore_reward_weight": float(self.explore_reward_weight),
                "explore_discount": float(self.explore_discount),
                "q_phi_enable": bool(self.q_phi_enable),
                "q_phi_train_enable": bool(self.q_phi_train_enable),
                "q_phi_probability_mode": str(self.q_phi_probability_mode),
            },
        }

    @classmethod
    def assemble_outputs(cls, data: DataProto, outputs: list[dict[str, Any]], config: Any) -> DataProto:
        reward_kwargs = config.reward.get("reward_kwargs", {})
        eapo_reward_cfg = reward_kwargs.get("eapo", {})
        algorithm_eapo_cfg = config.algorithm.get("eapo", {})

        format_reward_weight = float(
            _cfg_get(eapo_reward_cfg, "format_reward_weight", _cfg_get(algorithm_eapo_cfg, "format_reward_weight", 0.5))
        )
        explore_reward_weight = float(
            _cfg_get(eapo_reward_cfg, "explore_reward_weight", _cfg_get(algorithm_eapo_cfg, "explore_reward_weight", 1.0))
        )
        explore_discount = float(
            _cfg_get(eapo_reward_cfg, "explore_discount", _cfg_get(algorithm_eapo_cfg, "explore_discount", 0.9))
        )
        q_phi_inputs = [output.get("eapo_q_phi_input", {}) for output in outputs]
        q_phi_inputs = [
            {
                **entry,
                "format_reward_weight": float(entry.get("format_reward_weight", format_reward_weight)),
                "explore_reward_weight": float(entry.get("explore_reward_weight", explore_reward_weight)),
                "explore_discount": float(entry.get("explore_discount", explore_discount)),
            }
            for entry in q_phi_inputs
        ]

        final_scores = []
        reward_extra_infos = []
        for sample_index, (output, entry) in enumerate(zip(outputs, q_phi_inputs)):
            del entry
            reward_extra_info = dict(output.get("reward_extra_info", {}))
            task_reward = float(reward_extra_info.get("eapo_task_reward", output.get("reward_score", 0.0)))
            format_reward = float(reward_extra_info.get("eapo_format_reward", 0.0))
            explore_reward = float(reward_extra_info.get("eapo_explore_reward", 0.0))
            score = float(
                reward_extra_info.get(
                    "eapo_total_reward",
                    task_reward + format_reward_weight * format_reward + explore_reward_weight * explore_reward,
                )
            )
            final_scores.append(score)

            reward_extra_info["score"] = score
            reward_extra_info["eapo_task_reward"] = task_reward
            reward_extra_info["eapo_format_reward"] = format_reward
            reward_extra_info["eapo_explore_reward"] = explore_reward
            reward_extra_info["eapo_total_reward"] = score
            reward_extra_infos.append(reward_extra_info)

        rm_scores = cls.assemble_rm_scores(data, final_scores)
        batch = TensorDict({"rm_scores": rm_scores}, batch_size=len(data))
        non_tensor_batch, reward_extra_keys = _reward_extra_output(reward_extra_infos)
        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={"reward_extra_keys": reward_extra_keys},
        )
