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
Rollout config
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RolloutConfig:
    name: str = "vllm"
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop: Optional[List[str]] = None
    seed: int = 1
    limit_images: int = 0
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.6
    ignore_eos: bool = False
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False  # only for v0 engine
    tensor_parallel_size: int = 2
    max_model_len: Optional[int] = None
    max_num_batched_tokens: int = 8192
    disable_log_stats: bool = True
    val_override_config: Dict[str, Any] = field(default_factory=dict)

    # Optional interleaved point-to-count trajectory rollout
    interleaved_point_to_count: bool = False
    interleaved_max_turns: int = 51
    interleaved_per_turn_max_tokens: int = 128
    interleaved_answer_turn_max_tokens: int = 1536
    interleaved_stop_tag: str = "</answer>"
    # Optional prompt templates for interleaved mode (recommended to pass by .sh)
    interleaved_first_turn_prompt_template: Optional[str] = None
    interleaved_process_prompt_template: Optional[str] = None
    interleaved_first_turn_prompt_file: Optional[str] = None
    interleaved_process_prompt_file: Optional[str] = None
    interleaved_append_process_prompt: bool = True
    # History mode aligned with eval semantics:
    #   -1: keep full text history across turns (default)
    #    0: no text history; each follow-up turn reuses base prompt + current user follow-up prompt
    interleaved_history_mode: int = -1
    interleaved_debug: bool = False
    interleaved_debug_print_chars: int = 200

    # --- Safety/auto-protect knobs (keep default fast path) ---
    # When enabled, rollout will try to avoid OOM by temporarily reducing `n`
    # and/or enabling chunked prefill after an OOM event.
    auto_protect: bool = True
    auto_protect_min_n: int = 8
    auto_protect_free_gb_threshold: float = 5.0
    auto_protect_enable_chunked_prefill_on_oom: bool = True
    auto_protect_enable_chunked_prefill_prompt_threshold: int = 8192
    auto_protect_max_retries: int = 3
    """auto keys"""
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    trust_remote_code: bool = field(default=False, init=False)

    def to_dict(self):
        return asdict(self)
