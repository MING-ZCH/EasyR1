"""
StepCount模型评测脚本
功能:
1. 加载评测数据集
2. 多轮对话调用模型
3. 解析<point>标签并在图片上标注（与convert_image_cot_points.py一致）
4. 保存多轮对话记录和标注图片
"""

from __future__ import annotations

import argparse
import fcntl
import functools
import glob
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bundle_common import (
    PROTOCOL_VERSION,
    SUITES,
    parse_image_remaps,
    protocol_manifest,
    remap_image_path,
)
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


torch = None
AutoProcessor = None
Qwen2_5_VLForConditionalGeneration = None
StoppingCriteriaList = None


def load_runtime_dependencies() -> None:
    """正式评测时才加载 torch/transformers，保证 CPU dry-run/unit tests 可导入。"""
    global torch, AutoProcessor, Qwen2_5_VLForConditionalGeneration, StoppingCriteriaList
    if torch is not None:
        return
    import torch as torch_module
    from transformers import (
        AutoProcessor as auto_processor,
    )
    from transformers import (
        Qwen2_5_VLForConditionalGeneration as model_class,
    )
    from transformers import (
        StoppingCriteriaList as stopping_list,
    )

    torch = torch_module
    AutoProcessor = auto_processor
    Qwen2_5_VLForConditionalGeneration = model_class
    StoppingCriteriaList = stopping_list


# ==================== 配置部分 ====================
class Config:
    """运行时由 CLI 填充；协议相关字段不可从 CLI 降级。"""

    MODEL_PATH = ""
    MODEL_LABEL = ""
    EVAL_DATASET_PATH = ""
    DATASET_ID = ""
    IMAGE_ROOT = ""
    IMAGE_REMAPS = ()
    OUTPUT_DIR = ""
    OUTPUT_IMAGES_DIR = ""
    OUTPUT_JSON_PATH = ""

    # 正式协议固定为无文本历史（每轮依赖当前标点图与 system prompt）。
    HISTORY_MODE = 0
    # 模型参数
    MAX_NEW_TOKENS = 8192
    MAX_ROUNDS = 51  # 最大答案+1=最大迭代轮数
    # 与 RL 一致：effective_max_turn=min(MAX_ROUNDS, GT answer + 1 answer turn + margin=2)。
    ADAPTIVE_MAX_ROUNDS = True
    ADAPTIVE_MAX_ROUNDS_EXTRA = 3
    # 保留权威 evaluator 的兼容字段，但 enforce_strict_protocol 固定禁止启用。
    ANSWER_PLUS_ONE_MAX_ROUNDS = False
    REQUIRE_EXPLICIT_ANSWER = True
    ALLOW_POINT_COUNT_FALLBACK = False
    STOP_AFTER_FIRST_COMPLETE_TAG = True
    STOP_ON_NO_PROGRESS = False
    KEEP_EVAL_IMAGES = False
    MODEL_DTYPE = "bfloat16"
    WORKERS_PER_GPU = 1
    MAX_PIXELS = 12845056  # 训练时设定的最大pixels 12845056 ；评测时可调整为1440x1440 2073600 加快评测速度
    DOT_RADIUS = 10
    FONT_SIZE = 20
    EVAL_PROTOCOL_VERSION = PROTOCOL_VERSION
    EVAL_CODE_SHA256 = ""
    ALLOW_RESUME = False
    # 🆕 新增：思考控制开关
    # 如果设为 True，将在 Assistant 回复开头强制插入空的 think 标签，跳过思考过程
    DISABLE_THINKING = False
    # 注入的内容 (模拟模型已经完成了思考)
    THINK_PLACEHOLDER = "<think>\n\n</think>\n\n"

    # 正式协议固定 greedy；这些字段只保留底层生成实现的完整性。
    DO_SAMPLE = False
    NUM_BEAMS = 1
    NUM_BEAM_GROUPS = 1
    NUM_RETURN_SEQUENCES = 1
    TEMPERATURE = 0.5
    TOP_P = 0.9
    RANDOM_SEED = -1

    # Prompt模板
    PROMPT_SYSTEM = """You are a helpful assistant. Answer the user's counting question based on the image provided. \nOutput your thinking process within the <think> and </think> tags. Whenever you need to count, you should count the target objects one-by-one from top-left to bottom-right in the given image by outputing <point>{"point_2d": [x, y], "label": "x", "count_number": "x"}</point>, where (x, y) are the normalized coordinates of the target's center, "label" refers to the target objects and "count_number" is the number of the counted target. \nOnce the final answer is confirmed, put it within <answer> and </answer>."""
    # PROMPT_SYSTEM = '''You are a helpful assistant. Answer the user's question based on the image provided. \nWhenever you need to count, you should count the target objects one-by-one from top-left to bottom-right in the given image by outputing <point>{"point_2d": [x, y], "label": "x", "count_number": "x"}</point>, where (x, y) are the normalized coordinates of the target's center, \"label\" refers to the target objects and \"count_number\" is the number of the counted target. \nOnce the final answer is confirmed, put your reasoning process inside <think> and </think> and your final answer inside <answer> and </answer>.'''
    # PROMPT_SYSTEM = '''You are a helpful assistant. Answer the user's counting question based on the image provided. \nOutput your thinking process within the <think> and </think> tags. Whenever you need to count, you should count the target objects one-by-one from top-left to bottom-right in the given image by outputing <point>{"point_2d": [x, y], "label": "object", "count_number": "n"}</point>, where (x, y) are the normalized coordinates of the target's center, "label" refers to the target objects and "count_number" is the number of the counted target. Each object you point to will be marked with a red dot on the image. In subsequent turns, use these red dots to identify already-counted objects and avoid counting them again. \nOnce all target objects have been counted and marked, put your final answer within <answer> and </answer>.'''
    PROMPT_HUMAN_QUESTION = "{question}"

    # PROMPT_HUMAN_PROCESS
    PROMPT_HUMAN_PROCESS = """{question}\nContinue your reasoning process inside <think> and </think>. \nIf needed, you can continue to count on the observation image, by outputting <point> and </point> as before. \nIf the final answer is confirmed, put your final answer inside <answer> and </answer>."""
    # PROMPT_HUMAN_PROCESS = '''{question}\nContinue your problem-solving process. \nIf needed, you can continue to count on the observation image, by outputting <point> and </point>. \nIf the final answer is confirmed, put your reasoning process inside <think> and </think> and your final answer inside <answer> and </answer>.'''
    # PROMPT_HUMAN_PROCESS = '''{question}\nContinue your reasoning process inside <think> and </think>. Carefully observe the red dots in the observation image — these mark objects you have already counted. Focus on finding any unmarked objects that have not been counted yet.\nIf there are still unmarked target objects, count the next one by outputting <point>{"point_2d": [x, y], "label": "object", "count_number": "n"}</point>, where (x, y) are the normalized coordinates of the target's center, "label" refers to the target objects and "count_number" is the number of the counted target.\nIf all target objects have been marked with red dots and counting is complete, put your final answer inside <answer> and </answer>.'''


# ==================== 时间日志函数 ====================
LOG_FILE_PATH = ""


def parse_integer_answer(value) -> Optional[int]:
    """Parse a scalar integer answer without accepting unrelated text or booleans."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    text = str(value).strip()
    if not re.fullmatch(r"[+-]?\d+", text):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def compute_effective_max_rounds(
    correct_answer,
    global_max_rounds: int,
    extra_rounds: int = 3,
    enabled: bool = True,
) -> int:
    """Return min(global cap, GT + extra rounds), falling back safely on invalid GT."""
    global_cap = max(1, int(global_max_rounds))
    if not enabled:
        return global_cap
    gt_answer = parse_integer_answer(correct_answer)
    if gt_answer is None or gt_answer < 0:
        return global_cap
    extra = max(0, int(extra_rounds))
    return max(1, min(global_cap, gt_answer + extra))


def distribute_samples_by_estimated_turn_cost(
    samples: List[Dict],
    num_workers: int,
    global_max_rounds: int,
    extra_rounds: int,
    adaptive_enabled: bool = True,
) -> Tuple[List[List[Dict]], List[int]]:
    """Deterministically balance long trajectories without changing eval semantics."""
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    chunks: List[List[Dict]] = [[] for _ in range(num_workers)]
    loads = [0 for _ in range(num_workers)]
    ranked = []
    for original_index, sample in enumerate(samples):
        cost = compute_effective_max_rounds(
            sample.get("answer"),
            global_max_rounds,
            extra_rounds,
            adaptive_enabled,
        )
        ranked.append((cost, original_index, sample))

    # Longest-processing-time scheduling reduces the tail where only one GPU remains active.
    for cost, _, sample in sorted(ranked, key=lambda item: (-item[0], item[1])):
        worker_id = min(range(num_workers), key=lambda idx: (loads[idx], idx))
        chunks[worker_id].append(sample)
        loads[worker_id] += cost
    return chunks, loads


def answers_match(predicted_answer, correct_answer) -> bool:
    predicted = parse_integer_answer(predicted_answer)
    expected = parse_integer_answer(correct_answer)
    return predicted is not None and expected is not None and predicted == expected


@functools.lru_cache(maxsize=32)
def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@functools.lru_cache(maxsize=8)
def model_identity_fingerprint(model_path: str) -> str:
    """Build a stable local model identity without hashing every weight byte."""
    root = os.path.realpath(model_path)
    digest = hashlib.sha256()
    index_path = os.path.join(root, "model.safetensors.index.json")
    candidate_names = {
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    }
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            weight_index = json.load(handle)
        candidate_names.update(weight_index.get("weight_map", {}).values())
    elif os.path.isfile(os.path.join(root, "model.safetensors")):
        candidate_names.add("model.safetensors")

    for name in sorted(candidate_names):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            digest.update(f"missing:{name}\n".encode("utf-8"))
            continue
        stat = os.stat(path)
        digest.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode("utf-8"))
        with open(path, "rb") as handle:
            if stat.st_size <= 16 * 1024 * 1024:
                digest.update(handle.read())
            else:
                digest.update(handle.read(1024 * 1024))
                handle.seek(max(0, stat.st_size - 1024 * 1024))
                digest.update(handle.read(1024 * 1024))
    return digest.hexdigest()


def evaluation_config_fingerprint(config) -> str:
    payload = {
        "protocol_version": str(config.EVAL_PROTOCOL_VERSION),
        "eval_code_sha256": str(config.EVAL_CODE_SHA256),
        "model_path": os.path.realpath(config.MODEL_PATH),
        "model_identity": model_identity_fingerprint(config.MODEL_PATH),
        "dataset_path": os.path.realpath(config.EVAL_DATASET_PATH),
        "dataset_sha256": sha256_file(os.path.realpath(config.EVAL_DATASET_PATH)),
        "dataset_id": str(config.DATASET_ID),
        "image_root": os.path.realpath(config.IMAGE_ROOT) if config.IMAGE_ROOT else "",
        "image_remaps": [
            [os.path.normpath(str(source)), os.path.normpath(str(target))] for source, target in config.IMAGE_REMAPS
        ],
        "model_label": str(config.MODEL_LABEL),
        "history_mode": int(config.HISTORY_MODE),
        "max_rounds": int(config.MAX_ROUNDS),
        "adaptive_max_rounds": bool(config.ADAPTIVE_MAX_ROUNDS),
        "adaptive_extra": int(config.ADAPTIVE_MAX_ROUNDS_EXTRA),
        "require_explicit_answer": bool(config.REQUIRE_EXPLICIT_ANSWER),
        "allow_point_count_fallback": bool(config.ALLOW_POINT_COUNT_FALLBACK),
        "stop_after_first_complete_tag": bool(config.STOP_AFTER_FIRST_COMPLETE_TAG),
        "stop_on_no_progress": bool(config.STOP_ON_NO_PROGRESS),
        "model_dtype": str(config.MODEL_DTYPE).lower(),
        "do_sample": bool(config.DO_SAMPLE),
        "num_beams": int(config.NUM_BEAMS),
        "num_beam_groups": int(config.NUM_BEAM_GROUPS),
        "num_return_sequences": int(config.NUM_RETURN_SEQUENCES),
        "temperature": float(config.TEMPERATURE),
        "top_p": float(config.TOP_P),
        "random_seed": int(config.RANDOM_SEED),
        "max_pixels": int(config.MAX_PIXELS),
        "max_new_tokens": int(config.MAX_NEW_TOKENS),
        "disable_thinking": bool(config.DISABLE_THINKING),
        "dot_radius": int(config.DOT_RADIUS),
        "font_size": int(config.FONT_SIZE),
        "keep_eval_images": bool(config.KEEP_EVAL_IMAGES),
        "prompt_system": str(config.PROMPT_SYSTEM),
        "prompt_human_question": str(config.PROMPT_HUMAN_QUESTION),
        "prompt_human_process": str(config.PROMPT_HUMAN_PROCESS),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: str, rows) -> None:
    tmp_path = f"{path}.{os.getpid()}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def output_sibling_path(output_json_path: str, suffix: str) -> str:
    """Build a sibling path without rewriting `.json` in parent directories."""
    output_path = Path(output_json_path)
    if output_path.suffix.lower() != ".json":
        raise ValueError(f"output result must end with .json: {output_json_path}")
    return str(output_path.with_name(f"{output_path.stem}{suffix}"))


def ensure_working_images_dir(output_dir: str, working_dir: str) -> Path:
    """Create the owned working directory and reject symlink/escape layouts."""
    output_root = Path(output_dir).expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    candidate = Path(working_dir).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"working image directory must not be a symlink: {candidate}")
    candidate.mkdir(parents=True, exist_ok=True)
    working_root = candidate.resolve(strict=True)
    if working_root.parent != output_root:
        raise ValueError(f"working image directory must be a direct child of output root: {working_root}")
    return working_root


def prepare_sample_workspace(output_dir: str, working_dir: str, sample_id: str) -> Path:
    working_root = ensure_working_images_dir(output_dir, working_dir)
    workspace_id = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    sample_dir = working_root / f"sample-{workspace_id}"
    if sample_dir.is_symlink():
        raise ValueError(f"sample workspace must not be a symlink: {sample_dir}")
    if sample_dir.exists():
        if not sample_dir.is_dir():
            raise ValueError(f"sample workspace is not a directory: {sample_dir}")
        shutil.rmtree(sample_dir)
    sample_dir.mkdir()
    if sample_dir.resolve(strict=True).parent != working_root:
        raise ValueError(f"sample workspace escaped working root: {sample_dir}")
    return sample_dir


def remove_sample_workspace(sample_dir: Path, working_root: Path) -> None:
    if sample_dir.is_symlink() or sample_dir.parent != working_root:
        raise ValueError(f"refusing unsafe sample workspace cleanup: {sample_dir}")
    shutil.rmtree(sample_dir)


def truncate_at_first_complete_tag(response: str) -> Tuple[str, Optional[str]]:
    """Keep exactly the text through the earliest point/answer closing tag."""
    match = re.search(r"</(point|answer)\s*>", response, re.IGNORECASE)
    if not match:
        return response, None
    return response[: match.end()], match.group(1).lower()


class StopOnAnyTokenSequence:
    """Stop a batch once every row ends in one configured closing-tag token sequence."""

    def __init__(self, token_sequences: List[List[int]]):
        self.token_sequences = [list(seq) for seq in token_sequences if seq]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.token_sequences:
            return False
        for row in input_ids:
            row_tokens = row.tolist()
            if not any(len(row_tokens) >= len(seq) and row_tokens[-len(seq) :] == seq for seq in self.token_sequences):
                return False
        return True


def log_time_info(stage: str, duration: float, extra: str = ""):
    """打印并记录耗时信息"""
    pid = os.getpid()
    timestamp = time.strftime("%H:%M:%S")
    msg = f"[{timestamp}] [PID:{pid}] [{stage}] Duration: {duration:.4f}s {extra}"
    print(msg)
    if not LOG_FILE_PATH:
        return
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
        # 使用追加模式写入日志文件
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception as e:
        print(f"Write log failed: {e}")


# ==================== 模型推理模块 (完整替换) ====================
class ModelInference:
    """模型推理类,负责加载模型和生成回复"""

    def __init__(self, model_path: str, device_id: int):
        """
        初始化模型
        Args:
            model_path: 模型路径
            device_id: 要使用的GPU设备ID
        """
        load_runtime_dependencies()
        if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {device_id} is not available")
        with torch.cuda.device(device_id):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(f"CUDA device {device_id} does not support BF16")
        device = f"cuda:{device_id}"
        print(f"正在加载模型到 {device}: {model_path}")

        # 加载processor
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"

        # 🔍 关键：检查图片处理器的配置
        if hasattr(self.processor, "image_processor"):
            img_proc = self.processor.image_processor
            print("  图片处理器配置:")

            # 获取 max_pixels 参数（对应训练时的 image_max_pixels）
            self.max_pixels = int(Config.MAX_PIXELS)
            if hasattr(img_proc, "max_pixels"):
                img_proc.max_pixels = self.max_pixels
            print(f"    max_pixels: {self.max_pixels} (eval config)")

            if hasattr(img_proc, "min_pixels"):
                print(f"    min_pixels: {img_proc.min_pixels}")

        # 加载模型
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # 🟢 修正 1: 直接修改 config 对象，而不是传参
        config.use_cache = True

        if hasattr(config, "text_config") and isinstance(config.text_config, dict):
            from transformers import PretrainedConfig

            text_config_dict = config.text_config
            # 必须先把 dict 转为 Config 对象才能修改属性
            config.text_config = PretrainedConfig(**text_config_dict)
            # 🟢 修正 2: 确保文本模型的配置也开启 cache (这是 Qwen2.5-VL 的关键)
            config.text_config.use_cache = True
        elif hasattr(config, "text_config"):
            # 如果已经是对象
            config.text_config.use_cache = True

        dtype_name = str(Config.MODEL_DTYPE).strip().lower()
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported MODEL_DTYPE={Config.MODEL_DTYPE!r}; use bf16 or fp16")
        model_dtype = dtype_map[dtype_name]
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            torch_dtype=model_dtype,
            device_map=device,
            attn_implementation="flash_attention_2",
            # use_cache=True,  <-- 🔴 删除这一行，它会导致报错
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()

        stop_sequences = [
            self.processor.tokenizer.encode(tag, add_special_tokens=False) for tag in ("</point>", "</answer>")
        ]
        self.tag_stopping_criteria = StoppingCriteriaList([StopOnAnyTokenSequence(stop_sequences)])
        print(f"  model dtype: {model_dtype}")

        # 🔥 关键修复：强制开启 generation_config.use_cache (控制推理速度的真正开关)
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.use_cache = True
            # Do not let checkpoint-local generation_config.json silently
            # switch strict evaluation to beam, contrastive, constrained, or DoLa decode.
            self.model.generation_config.do_sample = False
            self.model.generation_config.num_beams = Config.NUM_BEAMS
            self.model.generation_config.num_beam_groups = Config.NUM_BEAM_GROUPS
            self.model.generation_config.num_return_sequences = Config.NUM_RETURN_SEQUENCES
            self.model.generation_config.penalty_alpha = None
            self.model.generation_config.constraints = None
            self.model.generation_config.force_words_ids = None
            self.model.generation_config.dola_layers = None
            self.model.generation_config.prompt_lookup_num_tokens = None
            self.model.generation_config.assistant_early_exit = None
            print("  ✅ 强制开启 generation_config.use_cache = True")
        else:
            print("  ⚠️ 警告: 模型没有 generation_config 属性")

        print(f"✅ 模型已加载到 {device}\n")

    def calculate_resized_dimensions(self, width: int, height: int) -> Tuple[int, int]:
        """
        计算Qwen2.5-VL根据max_pixels调整后的图片尺寸

        参考：Qwen2.5-VL的图片处理逻辑
        https://github.com/QwenLM/Qwen2-VL/blob/main/qwen_vl_utils/vision_process.py
        """
        total_pixels = width * height

        if total_pixels <= self.max_pixels:
            # 不需要缩放,但仍需调整到28的倍数
            new_width = (width // 28) * 28
            new_height = (height // 28) * 28
            return new_width, new_height

        # 需要等比例缩小
        scale = (self.max_pixels / total_pixels) ** 0.5
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Qwen2.5-VL会调整到28的倍数（patch size）
        new_width = (new_width // 28) * 28
        new_height = (new_height // 28) * 28

        return new_width, new_height

    def generate_response(
        self, text: str, images: List[Image.Image], max_new_tokens: int = 2048
    ) -> Tuple[str, Tuple[int, int], Tuple[int, int]]:
        """
        生成模型回复

        Returns:
            (response, original_size, resized_size):
                - response: 模型输出文本
                - original_size: 原始图片尺寸 (width, height)
                - resized_size: 模型实际看到的尺寸 (width, height)
        """
        # 记录原始尺寸
        if images:
            original_width, original_height = images[0].size
            original_size = (original_width, original_height)

            # 计算调整后的尺寸
            resized_width, resized_height = self.calculate_resized_dimensions(original_width, original_height)
            resized_size = (resized_width, resized_height)

            print(f"  [推理] 原始图片: {original_width}x{original_height}")
            print(f"  [推理] 模型看到: {resized_width}x{resized_height}")

            if original_size != resized_size:
                scale = resized_width / original_width
                print(f"  [推理] 缩放比例: {scale:.4f}")
        else:
            original_size = (0, 0)
            resized_size = (0, 0)

        # 处理输入
        inputs = self.processor(text=[text], images=images if images else None, return_tensors="pt", padding=True)

        # 移动到GPU
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 生成回复
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": Config.DO_SAMPLE,
            "num_beams": Config.NUM_BEAMS,
            "num_beam_groups": Config.NUM_BEAM_GROUPS,
            "num_return_sequences": Config.NUM_RETURN_SEQUENCES,
            "penalty_alpha": None,
            "constraints": None,
            "force_words_ids": None,
            "dola_layers": None,
            "prompt_lookup_num_tokens": None,
            "assistant_early_exit": None,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }
        if Config.STOP_AFTER_FIRST_COMPLETE_TAG:
            generation_kwargs["stopping_criteria"] = self.tag_stopping_criteria
        if Config.DO_SAMPLE:
            generation_kwargs["temperature"] = Config.TEMPERATURE
            generation_kwargs["top_p"] = Config.TOP_P
            if Config.RANDOM_SEED >= 0:
                # 不同 pass 使用不同 seed；同一 pass 内保持可复现。
                random.seed(Config.RANDOM_SEED)
                torch.manual_seed(Config.RANDOM_SEED)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(Config.RANDOM_SEED)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        # 解码
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]

        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 清理内存 (已优化：移除极度耗时的 empty_cache 和 gc.collect)
        del inputs, generated_ids, generated_ids_trimmed
        # torch.cuda.empty_cache()  <-- 性能杀手，已注释
        # gc.collect()              <-- 性能杀手，已注释

        return response, original_size, resized_size


# ==================== 图像处理模块 (完整替换) ====================
class ImageAnnotator:
    """图像标注类,负责在图片上标注计数点（与convert_image_cot_points.py一致）"""

    @staticmethod
    def annotate_image(
        image_path: str,
        points: List[Dict],
        output_path: str,
        accumulated_count: int = 0,
        original_size: Tuple[int, int] = None,  # 原图尺寸 (width, height)，建议从ModelInference获取
        model_size: Tuple[int, int] = None,  # 模型实际处理尺寸 (width, height)，从ModelInference的resized_size传入
    ) -> int:
        """
        在图片上标注计数点（复用ModelInference的尺寸计算逻辑）
        """
        try:
            # 打开图片并获取原图尺寸
            img = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            img_width, img_height = img.size
            if original_size and tuple(original_size) != (img_width, img_height):
                print(
                    f"  ⚠️ 警告: 推理记录尺寸{tuple(original_size)}与标注图片尺寸"
                    f"{(img_width, img_height)}不一致，以实际图片为准"
                )
            print(f"  [标注] 原图尺寸: {img_width}x{img_height}")

            # 检查模型尺寸（必须从ModelInference获取，确保与模型输入一致）
            if not model_size:
                print("  ⚠️ 警告: 未传入model_size，使用默认512x512（可能不准确）")
                model_size = (1024, 1024)
            model_width, model_height = model_size
            print(f"  [标注] 模型实际处理尺寸: {model_width}x{model_height}")

            # 计算缩放比例（原图尺寸 / 模型处理尺寸，宽高比例一致，取其一即可）
            if model_width <= 0 or model_height <= 0:
                scale = 1.0
            else:
                scale_x = img_width / model_width
                scale_y = img_height / model_height
                scale = (scale_x + scale_y) / 2.0

            # 动态计算标注元素大小（基于模型视角的目标尺寸）
            # 1. 圆点：模型视角下直径20像素（覆盖20/28≈71%的patch，确保可见）
            base_dot_radius = int(Config.DOT_RADIUS)
            dot_radius = max(4, round((base_dot_radius / 2) * scale))  # 原图半径（至少4像素）

            # 2. 字体：模型视角下20像素（确保数字清晰，覆盖>1个patch高度）
            base_font_size = int(Config.FONT_SIZE)
            font_size = max(10, round(base_font_size * scale))  # 原图字体大小（至少10像素）

            # 设置字体
            try:
                if os.name == "nt":
                    font = ImageFont.truetype("arial.ttf", font_size)
                else:
                    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()
                font_size = max(10, round(base_font_size * scale))  #  fallback后仍保证大小

            # 绘制每个有效点；模型历史输出采用原图绝对像素，兼容少量[0,1]坐标。
            drawn_count = 0
            for idx, point in enumerate(points):
                x_coord = point["x"]
                y_coord = point["y"]

                if not (math.isfinite(x_coord) and math.isfinite(y_coord)):
                    print(f"  ⚠️ 警告: 点 {idx + 1} 坐标不是有限数: ({x_coord}, {y_coord})")
                    continue

                # 坐标转换逻辑（复用原逻辑，基于模型处理尺寸修正）
                if 0 <= x_coord <= 1 and 0 <= y_coord <= 1:
                    x = round(x_coord * max(0, img_width - 1))
                    y = round(y_coord * max(0, img_height - 1))
                    print(f"  [标注] 点 {idx + 1}: [0,1]坐标({x_coord:.4f}, {y_coord:.4f}) -> 原图像素({x}, {y})")
                else:
                    # 像素坐标：若为模型处理尺寸下的坐标，直接缩放至原图
                    # x = round(x_coord * scale)
                    # y = round(y_coord * scale)
                    # print(f"  [标注] 点 {idx+1}: 模型像素({x_coord:.1f}, {y_coord:.1f}) "
                    #     f"-> 原图像素({x}, {y})")

                    # 绝对坐标：模型输出的就是原图像素坐标，直接使用
                    x = round(x_coord)
                    y = round(y_coord)
                    print(f"  [标注] 点 {idx + 1}: 原图像素({x_coord}, {y_coord})")

                # 检查坐标是否在原图范围内
                if x < 0 or x >= img_width or y < 0 or y >= img_height:
                    print(f"  ⚠️ 警告: 点 {idx + 1} 坐标 ({x}, {y}) 超出原图范围 ({img_width}x{img_height})")
                    continue

                point_number = accumulated_count + drawn_count + 1

                # 绘制红色圆点（动态半径）
                draw.ellipse(
                    [x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius],
                    fill=(255, 0, 0),
                    outline=(255, 255, 255),
                    width=2,  # 边框宽度固定（模型视角下≈2/scale像素，足够清晰）
                )

                # 绘制数字（动态字体大小）
                text_x = x + dot_radius + 5  # 与圆点保持间距，避免重叠
                text_y = y - (font_size // 2)  # 垂直居中
                text_bbox = draw.textbbox((text_x, text_y), str(point_number), font=font)
                # 白色背景框（确保数字在任何背景下可见）
                draw.rectangle(
                    [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=(255, 255, 255)
                )
                draw.text((text_x, text_y), str(point_number), font=font, fill=(0, 0, 0))
                drawn_count += 1

            # 保存标注图
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)
            print(f"  ✅ 标注图已保存: {os.path.basename(output_path)}")
            return drawn_count

        except Exception as e:
            print(f"❌ 标注失败: {e}")
            import traceback

            traceback.print_exc()
            return 0


# ==================== 解析模块 ====================
class ResponseParser:
    """响应解析类,负责解析模型输出中的point标签和answer标签"""

    @staticmethod
    def parse_points(response: str) -> List[Dict]:
        """
        解析模型输出中的<point>标签

        Args:
            response: 模型输出文本

        Returns:
            点坐标列表 [{"x": 0.5, "y": 0.5, "label": "person", "count": 1}, ...]
        """
        points = []
        # 正则匹配 <point>{"point_2d": [x, y], "label": "x", "count_number": "x"}</point>
        pattern = r"<point>\s*(\{[^}]+\})\s*</point>"
        matches = re.findall(pattern, response)

        for match in matches:
            try:
                # 解析JSON
                raw_point_data = json.loads(match)

                # ======================================================
                # 🚀 强力修复：清洗字典的键，消除模型产生的幻觉引号
                # 将 {"\"point_2d\"": ...} 强制转换为 {"point_2d": ...}
                # ======================================================
                point_data = {}
                for k, v in raw_point_data.items():
                    # 去除键名首尾可能存在的多余单/双引号和反斜杠
                    clean_k = k.strip('"').strip("'").replace("\\", "")
                    point_data[clean_k] = v
                # ======================================================

                coords = point_data.get("point_2d")

                # 确保坐标是浮点数
                if isinstance(coords, list) and len(coords) == 2:
                    x = float(coords[0])
                    y = float(coords[1])
                else:
                    continue
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue

                label = point_data.get("label", "")
                count = int(point_data.get("count_number", 0))

                points.append({"x": x, "y": y, "label": label, "count": count})
            except Exception as e:
                print(f"解析point失败: {match}, 错误: {e}")

        return points

    @staticmethod
    def has_answer(response: str) -> bool:
        """检查模型输出是否包含<answer>标签"""
        return "<answer>" in response.lower() and "</answer>" in response.lower()

    @staticmethod
    def extract_answer(response: str, require_explicit: bool = True) -> str:
        """提取答案；严格模式只接受闭合的 <answer>...</answer>。"""
        pattern = r"<answer>\s*(.*?)\s*</answer>"
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            return answer if re.fullmatch(r"[+-]?\d+", answer) else ""

        if require_explicit:
            return ""

        # 兼容旧评测：没有标签时尝试直接提取数字。
        numbers = re.findall(r"[+-]?\d+", response)
        if numbers:
            return numbers[-1]  # 返回最后一个数字(通常是最终答案)

        return ""


# ==================== 评测主流程 ====================
class StepCountEvaluator:
    """StepCount评测器,协调各模块完成评测"""

    def __init__(self, config: Config, device_id: int = 0):
        self.config = config
        self.device_id = device_id
        self.model = ModelInference(config.MODEL_PATH, device_id)
        self.annotator = ImageAnnotator()
        self.parser = ResponseParser()

        # 创建输出目录
        ensure_working_images_dir(config.OUTPUT_DIR, config.OUTPUT_IMAGES_DIR)

    def evaluate_sample(self, sample: Dict) -> Dict:
        """
        评测单个样本（多轮对话+图片迭代）

        Args:
            sample: 样本数据 {"id": "test_0", "question": "...", "answer": 5, "image_path": "..."}

        Returns:
            评测结果字典
        """
        t_case_start = time.time()
        sample_id = str(sample["id"])
        question = sample["question"]
        correct_answer = sample["answer"]
        original_image_path = sample["image_path"]

        print(f"\n{'=' * 50}")
        print(f"评测样本: {sample_id}")
        print(f"问题: {question}")
        print(f"正确答案: {correct_answer}")
        print(f"{'=' * 50}")

        # 为每个样本创建独立的图片文件夹
        working_images_dir = ensure_working_images_dir(self.config.OUTPUT_DIR, self.config.OUTPUT_IMAGES_DIR)
        sample_image_dir = prepare_sample_workspace(self.config.OUTPUT_DIR, self.config.OUTPUT_IMAGES_DIR, sample_id)

        # 复制原图到样本文件夹
        image_basename = os.path.basename(original_image_path)
        filename, ext = os.path.splitext(image_basename)
        output_original_image = os.path.join(str(sample_image_dir), image_basename)
        shutil.copy(original_image_path, output_original_image)

        # conversation_output 用于构建最终的JSON文件，它会保留完整的历史记录
        conversation_output = []

        # 🆕 conversation_history 用于构建给模型的输入，它会根据HISTORY_MODE进行裁剪
        conversation_history = []

        # 初始System Prompt
        system_prompt = {"role": "system", "content": self.config.PROMPT_SYSTEM}
        conversation_output.append(system_prompt)
        conversation_history.append(system_prompt)

        # 保存图片路径（相对于输出目录）
        image_paths = [os.path.relpath(output_original_image, self.config.OUTPUT_DIR)]

        # 当前标注图片路径（用于累积标注）
        current_annotated_image = output_original_image
        accumulated_points_count = 0  # 累计标注的点数

        # 记录图片尺寸
        original_image_size = None
        model_image_size = None

        # 与 RL rollout 对齐：GT 个 point 轮 + 1 个 answer 轮 + margin=2，即 GT+3。
        adaptive_enabled = bool(getattr(self.config, "ADAPTIVE_MAX_ROUNDS", False))
        adaptive_extra = int(getattr(self.config, "ADAPTIVE_MAX_ROUNDS_EXTRA", 3))
        if getattr(self.config, "ANSWER_PLUS_ONE_MAX_ROUNDS", False):
            adaptive_enabled = True
            adaptive_extra = 1
        max_rounds = compute_effective_max_rounds(
            correct_answer,
            self.config.MAX_ROUNDS,
            adaptive_extra,
            adaptive_enabled,
        )
        print(
            f"  [Info] effective_max_rounds={max_rounds} "
            f"(global={self.config.MAX_ROUNDS}, adaptive={adaptive_enabled}, GT+{adaptive_extra})"
        )

        termination_reason = "max_rounds"
        last_response_event = None
        no_progress_rounds = 0

        for round_idx in range(1, max_rounds + 1):
            print(f"\n[轮次 {round_idx}/{max_rounds}]")

            # 1. 构建当前轮次的用户输入
            if round_idx == 1:
                prompt_text = self.config.PROMPT_HUMAN_QUESTION.replace("{question}", question)
                current_images = [Image.open(original_image_path).convert("RGB")]
                current_image_path_for_input = original_image_path
            else:
                prompt_text = self.config.PROMPT_HUMAN_PROCESS.replace("{question}", question)
                current_images = [Image.open(current_annotated_image).convert("RGB")]
                current_image_path_for_input = current_annotated_image

            # 2. 准备给模型的输入消息 (messages_for_model)
            #    根据 HISTORY_MODE 从 conversation_history 中提取历史记录
            messages_for_model = []

            # 总是包含 system prompt
            messages_for_model.append(conversation_history[0])

            if self.config.HISTORY_MODE == -1:  # 全部历史
                # 添加所有历史 user/model 对话
                messages_for_model.extend(conversation_history[1:])
            elif self.config.HISTORY_MODE > 0:  # 最近 N 轮历史
                n = self.config.HISTORY_MODE
                # 获取历史中的 user/model 对话部分
                history_pairs = conversation_history[1:]
                # 每轮包含 user 和 model 两条消息，所以取最后 2*n 条
                start_index = max(0, len(history_pairs) - 2 * n)
                messages_for_model.extend(history_pairs[start_index:])
            # 如果 HISTORY_MODE == 0, 则只保留 system prompt，不添加任何历史

            # 3. 添加当前轮次的用户输入
            current_user_message = {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
            }
            messages_for_model.append(current_user_message)

            # 4. 将当前轮次的用户输入完整地存入 conversation_output (用于最终报告)
            #    注意：这里保存的是相对路径
            conversation_output.append(
                {
                    "role": "human",
                    "content": prompt_text,
                    "image": os.path.relpath(current_image_path_for_input, self.config.OUTPUT_DIR),
                }
            )

            # 应用聊天模板
            text = self.model.processor.apply_chat_template(
                messages_for_model, tokenize=False, add_generation_prompt=True
            )

            # 🔥 关键修改：强制注入思考结束标记
            if self.config.DISABLE_THINKING:
                # 只有在第一轮或者每一轮(取决于您的策略)注入
                # 通常 GRPO 模型每一轮都在输出 think，所以这里每轮都加是可以的
                text += self.config.THINK_PLACEHOLDER
                if round_idx == 1:
                    print("  [Info] 已强制跳过思考过程 (注入前缀)")

            # 生成回复
            response, original_image_size, model_image_size = self.model.generate_response(
                text, current_images, self.config.MAX_NEW_TOKENS
            )
            raw_response_length = len(response)
            response, response_event = truncate_at_first_complete_tag(response)
            last_response_event = response_event
            if len(response) != raw_response_length:
                print(
                    f"  [Info] 解码后二次截断到首个 </{response_event}>，"
                    f"丢弃尾部 {raw_response_length - len(response)} 字符"
                )

            print(f"  模型回复: {response[:300]}{'...' if len(response) > 300 else ''}")

            # 5. 将模型回复保存到 conversation_output 和 conversation_history
            model_response_message = {"role": "model", "content": response, "round": round_idx}
            conversation_output.append(model_response_message)
            # 存入 history 时，为了模板处理，不需要 "round" 字段
            history_model_message = {"role": "model", "content": response}

            # 将当前轮的 user 和 model 输入都加入 history，为下一轮做准备
            # 注意：user message 需要简化，因为 apply_chat_template 不处理图片对象
            simplified_user_message = {
                "role": "user",
                "content": prompt_text,  # 只保留文本部分
            }
            conversation_history.append(simplified_user_message)
            conversation_history.append(history_model_message)

            # 检查是否有最终答案
            has_final_answer = response_event == "answer" and self.parser.has_answer(response)
            if has_final_answer:
                print("  ✓ 检测到最终答案")

            # 解析新的标注点
            new_points = self.parser.parse_points(response)[:1] if response_event == "point" else []

            # 优化后的终止逻辑
            if new_points:
                print(f"  ✓ 解析到 {len(new_points)} 个新标注点")

                # 在当前图片基础上添加新的标注点
                annotated_image_path = os.path.join(str(sample_image_dir), f"{filename}_{round_idx}{ext}")

                drawn_count = self.annotator.annotate_image(
                    current_annotated_image,
                    new_points,
                    annotated_image_path,
                    accumulated_points_count,
                    original_size=original_image_size,
                    model_size=model_image_size,
                )

                if drawn_count > 0:
                    print(f"  ✓ 已生成标注图: {os.path.basename(annotated_image_path)}")

                    # 更新状态
                    current_annotated_image = annotated_image_path
                    accumulated_points_count += drawn_count
                    image_paths.append(os.path.relpath(annotated_image_path, self.config.OUTPUT_DIR))

                    # 情况1: 有标注点 + 有最终答案
                    if has_final_answer:
                        print("  → 已获得最终答案，结束迭代")
                        termination_reason = "explicit_answer"
                        break

                    # 情况2: 有标注点 + 达到轮次上限
                    if round_idx >= max_rounds:
                        print(f"  ⚠️ 已达到最大轮次限制 ({max_rounds})")
                        if not has_final_answer:
                            print("  → 未检测到闭合<answer>标签，严格评测记为未作答")
                        termination_reason = "max_rounds"
                        break

                    # 情况3: 有标注点 + 无最终答案 + 未达上限 → 继续下一轮
                    print("  → 继续下一轮对话")

                else:
                    # 标注失败
                    print("  ✗ 图片标注失败，结束迭代")
                    termination_reason = "annotation_failed"
                    break
            else:
                # 没有新标注点
                print("  → 未检测到新的标注点")

                # 情况4: 无标注点 + 有最终答案
                if has_final_answer:
                    print("  → 已获得最终答案，结束迭代")
                    termination_reason = "explicit_answer"
                    break

                # 情况5: 无标注点 + 无最终答案
                else:
                    print("  ⚠️ 无新标注点且无最终答案")
                    no_progress_rounds += 1

                    # 如果达到上限，强制结束
                    if round_idx >= max_rounds:
                        print(f"  → 已达到最大轮次限制 ({max_rounds})")
                        termination_reason = "max_rounds"
                    else:
                        if bool(getattr(self.config, "STOP_ON_NO_PROGRESS", False)):
                            print("  → STOP_ON_NO_PROGRESS已开启，提前结束迭代")
                            termination_reason = "no_progress"
                            break
                        print("  → 与RL rollout对齐：保留当前图片并继续下一轮")
                        continue
                    break

        # 提取最终答案（优化逻辑）
        final_response = (
            conversation_output[-1]["content"]
            if conversation_output and conversation_output[-1]["role"] == "model"
            else ""
        )
        explicit_answer = self.parser.extract_answer(final_response, require_explicit=True)
        require_explicit = bool(getattr(self.config, "REQUIRE_EXPLICIT_ANSWER", True))
        predicted_answer = self.parser.extract_answer(final_response, require_explicit=require_explicit)
        fallback_predicted_answer = str(accumulated_points_count)
        used_point_count_fallback = False

        # 严格评测默认不把 point JSON 中的数字或累计点数冒充最终答案。
        if not predicted_answer or predicted_answer == "":
            if getattr(self.config, "ALLOW_POINT_COUNT_FALLBACK", False):
                predicted_answer = fallback_predicted_answer
                used_point_count_fallback = True
                print(f"未找到闭合<answer>标签，兼容模式使用累计标注点数: {predicted_answer}")
            else:
                predicted_answer = ""
                print(
                    "未找到闭合<answer>标签，严格评测记为未作答；"
                    f"累计标注点数仅保留为诊断值: {fallback_predicted_answer}"
                )
            conversation_output.append(
                {
                    "role": "system",
                    "content": (
                        "No explicit closed answer was detected. "
                        f"Cumulative annotation points (diagnostic only): {accumulated_points_count}"
                    ),
                    "round": len([c for c in conversation_output if c["role"] == "model"]),
                }
            )

        is_correct = answers_match(predicted_answer, correct_answer)
        generated_image_count = max(0, len(image_paths) - 1)

        # 构建结果
        result = {
            "id": sample_id,
            "question": question,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "explicit_answer_detected": bool(explicit_answer),
            "fallback_predicted_answer": fallback_predicted_answer,
            "used_point_count_fallback": used_point_count_fallback,
            "is_correct": is_correct,
            "effective_max_rounds": max_rounds,
            "adaptive_max_rounds_extra": adaptive_extra if adaptive_enabled else None,
            "termination_reason": termination_reason,
            "last_response_event": last_response_event,
            "no_progress_rounds": no_progress_rounds,
            "output": conversation_output,
            "image_paths": image_paths,
            "num_rounds": len([c for c in conversation_output if c["role"] == "model"]),
            "generated_image_count": generated_image_count,
            "model_dtype": str(self.config.MODEL_DTYPE),
            "eval_protocol_version": str(self.config.EVAL_PROTOCOL_VERSION),
            "dataset_id": str(self.config.DATASET_ID),
            "model_label": str(self.config.MODEL_LABEL),
            "history_mode": int(self.config.HISTORY_MODE),
            "do_sample": bool(self.config.DO_SAMPLE),
            "num_beams": int(self.config.NUM_BEAMS),
            "num_beam_groups": int(self.config.NUM_BEAM_GROUPS),
            "num_return_sequences": int(self.config.NUM_RETURN_SEQUENCES),
            "require_explicit_answer": bool(self.config.REQUIRE_EXPLICIT_ANSWER),
            "allow_point_count_fallback": bool(self.config.ALLOW_POINT_COUNT_FALLBACK),
            "stop_after_first_complete_tag": bool(self.config.STOP_AFTER_FIRST_COMPLETE_TAG),
            "stop_on_no_progress": bool(self.config.STOP_ON_NO_PROGRESS),
            "keep_eval_images": bool(self.config.KEEP_EVAL_IMAGES),
            "task_cap": int(self.config.MAX_ROUNDS),
            "eval_code_sha256": str(self.config.EVAL_CODE_SHA256),
            "model_identity": model_identity_fingerprint(self.config.MODEL_PATH),
            "dataset_sha256": sha256_file(os.path.realpath(self.config.EVAL_DATASET_PATH)),
            "eval_protocol": (
                f"oracle_min_global_gt_plus_{adaptive_extra}" if adaptive_enabled else "fixed_global_max_rounds"
            ),
            "config_fingerprint": evaluation_config_fingerprint(self.config),
            "model_path": os.path.realpath(self.config.MODEL_PATH),
        }

        if not getattr(self.config, "KEEP_EVAL_IMAGES", True):
            remove_sample_workspace(sample_image_dir, working_images_dir)
            result["image_paths"] = []

        print(f"\n{'=' * 50}")
        print(f"样本 {sample_id} 评测完成")
        print(f"  预测答案: {predicted_answer}")
        print(f"  正确答案: {correct_answer}")
        print(f"  是否正确: {'✓' if is_correct else '✗'}")
        print(f"  总轮次: {result['num_rounds']}")
        print(f"  生成标注图片数: {generated_image_count}")
        print(f"{'=' * 50}\n")

        log_time_info(f"Evaluator:Case({sample_id})", time.time() - t_case_start)

        return result

    def evaluate_all(self, samples_to_process: List[Dict], temp_json_path: str):
        """
        评测分配给当前进程的所有样本。

        Args:
            samples_to_process: 需要当前进程处理的样本列表。
            temp_json_path: 当前进程的临时结果输出文件路径。
        """
        results = []
        failures = []

        # 评测分配的样本
        worker_slot = getattr(self, "worker_slot", self.device_id)
        for sample in tqdm(
            samples_to_process,
            desc=f"GPU-{self.device_id}/W-{worker_slot} 评测进度",
            position=worker_slot,
        ):
            try:
                result = self.evaluate_sample(sample)
                results.append(result)

                # 实时保存到临时文件
                atomic_write_json(temp_json_path, results)

            except Exception as e:
                print(f"\n✗ [GPU-{self.device_id}] 评测样本 {sample.get('id', 'unknown')} 失败: {e}")
                import traceback

                traceback.print_exc()
                failures.append((str(sample.get("id", "unknown")), repr(e)))
                if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
                    raise RuntimeError(f"GPU-{self.device_id} OOM on sample {sample.get('id', 'unknown')}") from e

        print(f"\n✅ [GPU-{self.device_id}] 评测完成，结果已保存到: {temp_json_path}")
        if failures:
            raise RuntimeError(
                f"GPU-{self.device_id} had {len(failures)} failed samples; "
                f"refusing an incomplete score. examples={failures[:5]}"
            )

    def save_evaluation_report(self, results: List[Dict]):
        """生成并保存评测报告到txt文件"""
        # 生成txt文件路径（与json同名）
        txt_path = output_sibling_path(self.config.OUTPUT_JSON_PATH, ".txt")

        # 统计准确率
        correct = sum(1 for r in results if answers_match(r.get("predicted_answer"), r.get("correct_answer")))
        total = len(results)
        accuracy = correct / total if total > 0 else 0

        # 🆕 按答案范围分组统计
        range_0_5_samples = []
        range_5_plus_samples = []

        for r in results:
            try:
                correct_ans = int(r["correct_answer"])
                if 0 <= correct_ans <= 5:
                    range_0_5_samples.append(r)
                else:
                    range_5_plus_samples.append(r)
            except (ValueError, TypeError):
                # 如果答案不是整数，默认归入0-5范围
                range_0_5_samples.append(r)

        # 计算各范围的准确率
        correct_0_5 = sum(
            1 for r in range_0_5_samples if answers_match(r.get("predicted_answer"), r.get("correct_answer"))
        )
        total_0_5 = len(range_0_5_samples)
        accuracy_0_5 = correct_0_5 / total_0_5 if total_0_5 > 0 else 0

        correct_5_plus = sum(
            1 for r in range_5_plus_samples if answers_match(r.get("predicted_answer"), r.get("correct_answer"))
        )
        total_5_plus = len(range_5_plus_samples)
        accuracy_5_plus = correct_5_plus / total_5_plus if total_5_plus > 0 else 0

        # 统计轮次分布
        round_counts = {}
        for r in results:
            rounds = r.get("num_rounds", 0)
            round_counts[rounds] = round_counts.get(rounds, 0) + 1

        # 统计错误样本
        error_samples = [r for r in results if not answers_match(r.get("predicted_answer"), r.get("correct_answer"))]

        # 写入txt报告
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("StepCount模型评测报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"模型路径: {self.config.MODEL_PATH}\n")
            f.write(f"评测数据集: {self.config.EVAL_DATASET_PATH}\n")
            f.write(
                f"评测协议: {'oracle GT+' + str(self.config.ADAPTIVE_MAX_ROUNDS_EXTRA) if self.config.ADAPTIVE_MAX_ROUNDS else 'fixed rounds'}\n"
            )
            f.write(f"模型dtype: {self.config.MODEL_DTYPE}\n")
            f.write(f"协议版本: {self.config.EVAL_PROTOCOL_VERSION}\n")
            f.write(f"评测代码SHA256: {self.config.EVAL_CODE_SHA256}\n")
            f.write(f"模型身份摘要: {model_identity_fingerprint(self.config.MODEL_PATH)}\n")
            f.write(f"数据集SHA256: {sha256_file(os.path.realpath(self.config.EVAL_DATASET_PATH))}\n")
            f.write(f"配置指纹: {evaluation_config_fingerprint(self.config)}\n")
            f.write(f"评测时间: {self._get_timestamp()}\n\n")

            f.write("=" * 60 + "\n")
            f.write("总体统计\n")
            f.write("=" * 60 + "\n")
            f.write(f"总样本数: {total}\n")
            f.write(f"正确数: {correct}\n")
            f.write(f"错误数: {total - correct}\n")
            f.write(f"准确率: {accuracy:.2%}\n\n")

            # 🆕 按答案范围统计
            f.write("=" * 60 + "\n")
            f.write("答案范围统计\n")
            f.write("=" * 60 + "\n")
            f.write("答案 0-5:\n")
            f.write(f"  样本数: {total_0_5}\n")
            f.write(f"  正确数: {correct_0_5}\n")
            f.write(f"  错误数: {total_0_5 - correct_0_5}\n")
            f.write(f"  准确率: {accuracy_0_5:.2%}\n\n")

            f.write("答案 5+:\n")
            f.write(f"  样本数: {total_5_plus}\n")
            f.write(f"  正确数: {correct_5_plus}\n")
            f.write(f"  错误数: {total_5_plus - correct_5_plus}\n")
            f.write(f"  准确率: {accuracy_5_plus:.2%}\n\n")

            f.write("=" * 60 + "\n")
            f.write("轮次分布\n")
            f.write("=" * 60 + "\n")
            for rounds in sorted(round_counts.keys()):
                count = round_counts[rounds]
                percentage = count / total * 100
                f.write(f"{rounds}轮: {count}个样本 ({percentage:.1f}%)\n")
            f.write("\n")

            if error_samples:
                f.write("=" * 60 + "\n")
                f.write(f"错误样本详情 (共{len(error_samples)}个)\n")
                f.write("=" * 60 + "\n")
                for idx, err in enumerate(error_samples, 1):
                    f.write(f"{idx}. ID: {err['id']}\n")
                    f.write(f"   问题: {err['question']}\n")
                    f.write(f"   正确答案: {err['correct_answer']}\n")
                    f.write(f"   预测答案: {err['predicted_answer']}\n")
                    f.write(f"   轮次数: {err.get('num_rounds', 'N/A')}\n")
                    f.write("\n")

            f.write("=" * 60 + "\n")
            f.write("评测完成\n")
            f.write("=" * 60 + "\n")

        print(f"\n📊 评测报告已保存到: {txt_path}")

        # 同时打印到控制台
        print(f"\n{'=' * 60}")
        print("评测完成!")
        print(f"{'=' * 60}")
        print(f"总样本数: {total}")
        print(f"正确数: {correct}")
        print(f"错误数: {total - correct}")
        print(f"准确率: {accuracy:.2%}")
        print("\n答案范围统计:")
        print(f"  0-5: {total_0_5}个 (准确率: {accuracy_0_5:.2%})")
        print(f"  5+:  {total_5_plus}个 (准确率: {accuracy_5_plus:.2%})")
        print(f"{'=' * 60}\n")

    @staticmethod
    def _get_timestamp() -> str:
        """获取当前时间戳"""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==================== 多进程工作函数 ====================
def worker_part_path(
    output_json_path: str,
    worker_slot: int,
    device_id: int,
    process_id: Optional[int] = None,
) -> str:
    """Use an append-only worker filename so a resumed run never overwrites old progress."""
    pid = os.getpid() if process_id is None else int(process_id)
    return output_sibling_path(output_json_path, f"_part_pid{pid}_worker{worker_slot}_gpu{device_id}.json")


def worker(worker_slot: int, device_id: int, config: Config, samples_chunk: List[Dict]):
    """
    每个GPU进程执行的工作函数。
    """
    activate_runtime_config(config)
    # 为每个进程设置一个临时的JSON输出文件
    temp_json_path = worker_part_path(config.OUTPUT_JSON_PATH, worker_slot, device_id)

    print(f"🚀 启动进程 on GPU:{device_id}, 处理 {len(samples_chunk)} 个样本...")

    try:
        evaluator = StepCountEvaluator(config, device_id)
        evaluator.worker_slot = worker_slot
        evaluator.evaluate_all(samples_chunk, temp_json_path)
    except Exception as e:
        print(f"❌ [GPU-{device_id}] 进程发生严重错误: {e}")
        import sys
        import traceback

        traceback.print_exc()
        sys.exit(1)


# ==================== CLI 与主函数 ====================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V36 StepCount strict_oracle_gt_plus_v3 evaluator")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True, choices=("step77", "step60"))
    parser.add_argument("--task", required=True, choices=tuple(SUITES))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image-root")
    parser.add_argument(
        "--image-remap",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="可重复；仅做边界感知的绝对路径前缀映射",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--max-pixels", type=int, default=12845056)
    parser.add_argument("--resume", action="store_true")
    return parser


def snapshot_runtime_config(config_cls=Config) -> Config:
    """Copy class-level settings onto an instance so multiprocessing spawn preserves them."""
    snapshot = config_cls()
    for name in dir(config_cls):
        if name.isupper():
            setattr(snapshot, name, getattr(config_cls, name))
    return snapshot


def activate_runtime_config(config: Config) -> None:
    """Restore a serialized config inside a spawned worker process."""
    global LOG_FILE_PATH
    values = {name: value for name, value in vars(config).items() if name.isupper()}
    if not values:
        raise ValueError("spawned worker received an empty runtime config snapshot")
    for name, value in values.items():
        setattr(Config, name, value)
    enforce_strict_protocol(Config)
    LOG_FILE_PATH = os.path.join(Config.OUTPUT_DIR, "eval_time.log")


def configure_from_args(args: argparse.Namespace) -> Config:
    global LOG_FILE_PATH
    spec = SUITES[args.task]
    Config.MODEL_PATH = os.path.abspath(os.path.expanduser(args.model))
    Config.MODEL_LABEL = args.model_label
    Config.EVAL_DATASET_PATH = os.path.abspath(os.path.expanduser(args.dataset))
    Config.DATASET_ID = args.task
    Config.IMAGE_ROOT = os.path.abspath(os.path.expanduser(args.image_root)) if args.image_root else ""
    Config.IMAGE_REMAPS = parse_image_remaps(args.image_remap)
    Config.OUTPUT_DIR = os.path.abspath(os.path.expanduser(args.output_dir))
    Config.OUTPUT_IMAGES_DIR = os.path.join(Config.OUTPUT_DIR, ".working_images")
    Config.OUTPUT_JSON_PATH = os.path.join(Config.OUTPUT_DIR, "results.json")
    Config.MAX_ROUNDS = spec.task_cap
    Config.DOT_RADIUS = spec.dot_radius
    Config.WORKERS_PER_GPU = args.workers_per_gpu
    Config.MAX_PIXELS = args.max_pixels
    Config.ALLOW_RESUME = args.resume
    Config.EVAL_CODE_SHA256 = sha256_file(os.path.realpath(__file__))
    LOG_FILE_PATH = os.path.join(Config.OUTPUT_DIR, "eval_time.log")
    enforce_strict_protocol(Config)
    return snapshot_runtime_config(Config)


def enforce_strict_protocol(config) -> None:
    expected = {
        "EVAL_PROTOCOL_VERSION": PROTOCOL_VERSION,
        "MODEL_DTYPE": "bfloat16",
        "DO_SAMPLE": False,
        "NUM_BEAMS": 1,
        "NUM_BEAM_GROUPS": 1,
        "NUM_RETURN_SEQUENCES": 1,
        "HISTORY_MODE": 0,
        "ADAPTIVE_MAX_ROUNDS": True,
        "ADAPTIVE_MAX_ROUNDS_EXTRA": 3,
        "ANSWER_PLUS_ONE_MAX_ROUNDS": False,
        "REQUIRE_EXPLICIT_ANSWER": True,
        "ALLOW_POINT_COUNT_FALLBACK": False,
        "STOP_AFTER_FIRST_COMPLETE_TAG": True,
        "STOP_ON_NO_PROGRESS": False,
        "KEEP_EVAL_IMAGES": False,
    }
    mismatches = {
        key: (getattr(config, key), value) for key, value in expected.items() if getattr(config, key) != value
    }
    if mismatches:
        raise ValueError(f"strict protocol 不允许覆盖: {mismatches}")


def main(argv=None):
    """解析固定协议参数，使用全部可见 GPU 分发并汇总。"""
    args = build_arg_parser().parse_args(argv)
    config = configure_from_args(args)
    load_runtime_dependencies()
    os.makedirs(os.path.dirname(config.OUTPUT_JSON_PATH) or ".", exist_ok=True)
    run_lock = open(f"{config.OUTPUT_JSON_PATH}.lock", "a+", encoding="utf-8")
    try:
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Another evaluator is already writing {config.OUTPUT_JSON_PATH}") from exc

    # 检查GPU数量
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到CUDA；正式评测要求GPU，禁止以成功退出码静默跳过")

    num_gpus = torch.cuda.device_count()
    workers_per_gpu = int(getattr(config, "WORKERS_PER_GPU", 1))
    if workers_per_gpu not in {1, 2, 3}:
        raise ValueError(f"WORKERS_PER_GPU must be 1, 2, or 3, got {workers_per_gpu}")
    num_workers = num_gpus * workers_per_gpu
    print(f"检测到 {num_gpus} 个可用的GPU；每卡 {workers_per_gpu} 个并发worker，共 {num_workers} 个worker。")

    if not bool(getattr(config, "ALLOW_RESUME", False)):
        if os.path.exists(config.OUTPUT_JSON_PATH):
            raise FileExistsError(
                f"Fresh eval refuses existing result: {config.OUTPUT_JSON_PATH}; "
                "use an explicit resume mode only after verifying its fingerprint"
            )
        part_pattern = output_sibling_path(config.OUTPUT_JSON_PATH, "_part_*.json")
        stale_parts = sorted(glob.glob(part_pattern))
        if stale_parts:
            raise FileExistsError(
                "Fresh eval found recoverable worker part files; use --resume or remove them "
                f"explicitly: {stale_parts[:20]}"
            )

    # 加载并过滤数据集
    print(f"加载评测数据集: {config.EVAL_DATASET_PATH}")
    with open(config.EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    required_keys = {"id", "question", "answer", "image_path"}
    for row_idx, row in enumerate(eval_data):
        missing_keys = required_keys.difference(row)
        if missing_keys:
            raise ValueError(f"dataset row {row_idx} missing required keys: {sorted(missing_keys)}")
        row["image_path"] = str(
            remap_image_path(
                row["image_path"],
                dataset_path=config.EVAL_DATASET_PATH,
                image_root=config.IMAGE_ROOT or None,
                remaps=config.IMAGE_REMAPS,
            )
        )
        if not os.path.isfile(row["image_path"]):
            raise FileNotFoundError(f"dataset row {row_idx} image does not exist: {row['image_path']}")
    sample_ids = [str(row["id"]) for row in eval_data]
    if len(sample_ids) != len(set(sample_ids)):
        duplicate_ids = sorted({sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1})
        raise ValueError(f"dataset contains duplicate sample ids: {duplicate_ids[:20]}")
    expected_id_set = set(sample_ids)

    expected_fingerprint = evaluation_config_fingerprint(config)

    # Resume can recover both the final JSON and atomically-written worker part files.
    evaluated_ids = set()
    resume_results_map = {}

    def add_resume_rows(rows, source_path):
        if not isinstance(rows, list):
            raise TypeError(f"Resume source must contain a JSON list: {source_path}")
        stale_rows = [row.get("id") for row in rows if row.get("config_fingerprint") != expected_fingerprint]
        if stale_rows:
            raise RuntimeError(
                "Existing result was produced by a different model/config; "
                f"source={source_path}, stale_ids={stale_rows[:10]}"
            )
        for row in rows:
            sample_id = str(row.get("id"))
            if sample_id not in expected_id_set:
                raise RuntimeError(f"Resume source contains unexpected ID {sample_id}: {source_path}")
            previous = resume_results_map.get(sample_id)
            if previous is not None and previous != row:
                raise RuntimeError(f"Conflicting duplicate resume result for ID {sample_id}: {source_path}")
            resume_results_map[sample_id] = row

    if os.path.exists(config.OUTPUT_JSON_PATH):
        print(f"发现已存在的最终结果文件，加载进度: {config.OUTPUT_JSON_PATH}")
        try:
            with open(config.OUTPUT_JSON_PATH, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            add_resume_rows(existing_results, config.OUTPUT_JSON_PATH)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(f"Cannot safely parse existing final result: {config.OUTPUT_JSON_PATH}") from exc

    if bool(getattr(config, "ALLOW_RESUME", False)):
        part_pattern = output_sibling_path(config.OUTPUT_JSON_PATH, "_part_*.json")
        for resume_part_path in sorted(glob.glob(part_pattern)):
            try:
                with open(resume_part_path, "r", encoding="utf-8") as handle:
                    add_resume_rows(json.load(handle), resume_part_path)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Cannot safely parse resume part: {resume_part_path}") from exc

    evaluated_ids = set(resume_results_map)
    if evaluated_ids:
        print(f"  已从final/part结果中恢复 {len(evaluated_ids)} 个唯一已评测样本ID。")

    remaining_samples = [s for s in eval_data if str(s["id"]) not in evaluated_ids]

    if not remaining_samples:
        print("✅ 所有样本已评测完成!")
    else:
        print(f"开始评测剩余 {len(remaining_samples)} 个样本...\n")

        # 按 effective max turn 做 LPT 负载均衡，减少长轨迹导致的尾部 GPU 空闲。
        samples_per_worker, estimated_turn_loads = distribute_samples_by_estimated_turn_cost(
            remaining_samples,
            num_workers,
            config.MAX_ROUNDS,
            config.ADAPTIVE_MAX_ROUNDS_EXTRA,
            config.ADAPTIVE_MAX_ROUNDS,
        )
        print(f"各worker预计最大turn负载: {estimated_turn_loads}")

        # 创建并启动进程
        import multiprocessing as mp

        # 使用 'spawn' 启动方法以避免CUDA初始化问题
        ctx = mp.get_context("spawn")
        processes = []
        for worker_slot in range(num_workers):
            device_id = worker_slot % num_gpus
            if not samples_per_worker[worker_slot]:
                print(f"Worker-{worker_slot}/GPU-{device_id} 没有分配到任务，跳过。")
                continue

            p = ctx.Process(
                target=worker,
                args=(worker_slot, device_id, config, samples_per_worker[worker_slot]),
            )
            processes.append(p)
            p.start()

        # 等待所有进程完成
        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"GPU worker process failed with exit code {p.exitcode}")

    # --- 结果汇总 ---
    print("\n" + "=" * 60)
    print("所有进程已完成，开始汇总结果...")

    # 重新从所有来源加载最终结果，以保证完整性
    final_results_map = dict(resume_results_map)
    # 1. 加载主结果文件中的旧结果
    if os.path.exists(config.OUTPUT_JSON_PATH):
        try:
            with open(config.OUTPUT_JSON_PATH, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            for r in existing_results:
                final_results_map[str(r["id"])] = r
        except (json.JSONDecodeError, TypeError):
            pass

    # 2. 加载并合并所有 append-only part 文件（会覆盖同ID的完全一致结果）
    part_paths_to_cleanup = []
    part_pattern = output_sibling_path(config.OUTPUT_JSON_PATH, "_part_*.json")
    for temp_path in sorted(glob.glob(part_pattern)):
        try:
            with open(temp_path, "r", encoding="utf-8") as f:
                part_results = json.load(f)
            stale_rows = [
                row.get("id") for row in part_results if row.get("config_fingerprint") != expected_fingerprint
            ]
            if stale_rows:
                raise RuntimeError(f"Part result config mismatch in {temp_path}: stale_ids={stale_rows[:10]}")
            for r in part_results:
                sample_id = str(r["id"])
                previous = final_results_map.get(sample_id)
                if previous is not None and previous != r:
                    raise RuntimeError(f"Conflicting duplicate result for ID {sample_id}: {temp_path}")
                final_results_map[sample_id] = r
            part_paths_to_cleanup.append(temp_path)
            print(f"已合并临时文件，等待最终结果保存后清理: {temp_path}")
        except (json.JSONDecodeError, TypeError):
            print(f"⚠️ 警告: 无法解析或合并 {temp_path}，该文件将被保留。")

    # Worker scheduling may differ, but the final artifact remains in source-dataset order.
    final_results_list = [final_results_map[sample_id] for sample_id in sample_ids if sample_id in final_results_map]

    expected_ids = expected_id_set
    actual_ids = set(final_results_map)
    missing_ids = sorted(expected_ids - actual_ids)
    unexpected_ids = sorted(actual_ids - expected_ids)
    if missing_ids or unexpected_ids or len(final_results_list) != len(eval_data):
        raise RuntimeError(
            "Evaluation result is incomplete or contains unexpected samples; "
            f"expected={len(eval_data)} actual={len(final_results_list)} "
            f"missing={missing_ids[:20]} unexpected={unexpected_ids[:20]}"
        )

    print(f"汇总了 {len(final_results_list)} 条唯一的评测结果。")

    # 保存最终的JSON文件。先原子写入final，再删除part，避免中途退出造成part结果丢失。
    print(f"保存最终评测结果到: {config.OUTPUT_JSON_PATH}")
    atomic_write_json(config.OUTPUT_JSON_PATH, final_results_list)

    manifest = protocol_manifest(
        suite=config.DATASET_ID,
        dataset_path=Path(config.EVAL_DATASET_PATH),
        model_label=config.MODEL_LABEL,
        model_identity=model_identity_fingerprint(config.MODEL_PATH),
        output_path=Path(config.OUTPUT_JSON_PATH),
        workers_per_gpu=config.WORKERS_PER_GPU,
        sample_ids=sample_ids,
    )
    manifest.update(
        {
            "actual_samples": len(final_results_list),
            "eval_code_sha256": config.EVAL_CODE_SHA256,
            "config_fingerprint": expected_fingerprint,
            "image_root": config.IMAGE_ROOT,
            "image_remaps": [[str(source), str(target)] for source, target in config.IMAGE_REMAPS],
        }
    )
    atomic_write_json(os.path.join(config.OUTPUT_DIR, "run_manifest.json"), manifest)

    # 生成并保存最终的TXT报告
    # 重新创建一个Evaluator实例来调用报告生成方法（不加载模型）
    # 注意：这里需要一个虚拟的Evaluator实例，但它不应该重新加载模型
    class DummyEvaluator(StepCountEvaluator):
        def __init__(self, config: Config):
            self.config = config
            # 跳过模型加载
            # super().__init__(config)

    dummy_evaluator = DummyEvaluator(config)
    dummy_evaluator.save_evaluation_report(final_results_list)

    part_paths_to_cleanup.extend(glob.glob(output_sibling_path(config.OUTPUT_JSON_PATH, "_part_*.json")))
    for temp_path in sorted(set(part_paths_to_cleanup)):
        try:
            os.remove(temp_path)
            print(f"已删除临时文件: {temp_path}")
        except OSError as e:
            print(f"⚠️ 警告: 无法删除临时文件 {temp_path}: {e}")

    fcntl.flock(run_lock.fileno(), fcntl.LOCK_UN)
    run_lock.close()


if __name__ == "__main__":
    main()
