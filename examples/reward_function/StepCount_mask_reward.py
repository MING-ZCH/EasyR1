# =============================================================================
# StepCount Reward Function - Mask-Based 版本 v3.1
# =============================================================================
#
# 兼容 verl 训练框架的 Mask-Based Reward Function
#
# 核心特性：
#   - 与 StepCount_asymmetric.py 相同的接口签名
#   - 支持 mask-based reward（通过环境变量配置）
#   - 当没有 mask 或匹配失败时，回退到距离计算
#
# 使用方式：
#   1. 基础模式（距离计算）：直接使用，与 asymmetric 版本行为一致
#   2. Mask 模式：设置环境变量启用
#      export STEPCOUNT_MASKS_METADATA=/path/to/masks_metadata.json
#      export STEPCOUNT_MASKS_DIR=/path/to/masks/
#
# Eval/单样本评测建议：
#   - 若 eval 数据集中只保留序列中的某一张（如 xxx_5 表示第6个目标），
#     默认 used_sample_ids 为空会导致命中历史 mask 也算 +1，评测会虚高。
#   - 可设置以下环境变量让 reward 自动根据 turn 预填充“已点过”的 sample：
#       export STEPCOUNT_MASK_PREFILL_BY_TURN=1
#     且可选开启严格模式（只允许命中当前 turn 的 mask）：
#       export STEPCOUNT_MASK_ONLY_CURRENT_TURN=1
#
# 版本: 3.1 (2026.02.04) - 兼容 verl 框架
# =============================================================================

import re
import json
import math
import logging
import os
import threading
from typing import Dict, Optional, List, Any, Set, Tuple, Union
from PIL import Image as PILImage
import numpy as np
from collections import defaultdict, OrderedDict

logger = logging.getLogger(__name__)


# =============================================================================
# 全局配置（通过环境变量）
# =============================================================================

_MASK_HELPER = None  # 懒加载的 MaskRewardHelper 实例
_MASK_FALLBACK_LOGGED = set()

# debug 统计（可选，避免刷屏：按次数汇总打印）
_MASK_DEBUG_STATS = defaultdict(int)
_MASK_DEBUG_CALLS = 0
_TRAJ_REASON_DEBUG_CALLS = 0
_TRAJ_EVENT_LOG_COUNTS = defaultdict(int)

# Trajectory-path 独立 mask debug 统计（追踪 _trajectory_*_without_gt_points_details 路径）
_TRAJ_MASK_DEBUG_STATS = defaultdict(int)
_TRAJ_MASK_DEBUG_CALLS = 0

# 一次性配置打印
_MASK_CONFIG_LOGGED = False

# 默认路径（可用环境变量覆盖）
_DEFAULT_MASKS_METADATA_PATH = "/mnt/shared-storage-user/zhangchenhao/StepCount-RL_masks_output/masks_metadata.json"
_DEFAULT_MASKS_DIR = "/mnt/shared-storage-user/zhangchenhao/StepCount-RL_masks_output/masks"


class MaskMetadata:
    """Compact metadata record for mask reward indexing."""

    __slots__ = ("id", "mask_path", "image_path", "image_size", "points_pixel", "labels", "turn_number")

    def __init__(
        self,
        sample_id: str,
        mask_path: str,
        image_path: str,
        image_size: Tuple[int, int],
        points_pixel: Tuple[float, float],
        labels: Tuple[str, ...],
        turn_number: int,
    ):
        self.id = sample_id
        self.mask_path = mask_path
        self.image_path = image_path
        self.image_size = image_size
        self.points_pixel = points_pixel
        self.labels = labels
        self.turn_number = turn_number

    def get(self, key: str, default: Any = None) -> Any:
        if key == "id":
            return self.id
        if key == "mask_path":
            return self.mask_path
        if key == "image_path":
            return self.image_path
        if key == "image_size":
            return self.image_size
        if key == "points_pixel":
            return self.points_pixel
        if key == "labels":
            return self.labels
        if key == "turn_number":
            return self.turn_number
        return default


def _iter_json_array(path: str, chunk_size: int = 1 << 20):
    """Yield objects from a large top-level JSON array without loading it all."""
    decoder = json.JSONDecoder()
    with open(path, "r", encoding="utf-8") as f:
        buf = ""
        pos = 0
        eof = False
        started = False

        def _read_more() -> bool:
            nonlocal buf, pos, eof
            if eof:
                return False
            chunk = f.read(chunk_size)
            if not chunk:
                eof = True
                return False
            if pos:
                buf = buf[pos:] + chunk
                pos = 0
            else:
                buf += chunk
            return True

        _read_more()
        while True:
            while pos >= len(buf) and _read_more():
                pass
            while pos < len(buf) and buf[pos].isspace():
                pos += 1

            if not started:
                if pos >= len(buf):
                    if eof:
                        raise json.JSONDecodeError("empty JSON metadata", buf, pos)
                    continue
                if buf[pos] != "[":
                    raise json.JSONDecodeError("metadata must be a JSON array", buf, pos)
                pos += 1
                started = True
                continue

            while pos >= len(buf) and _read_more():
                pass
            while pos < len(buf) and buf[pos].isspace():
                pos += 1
            if pos >= len(buf):
                if eof:
                    raise json.JSONDecodeError("unterminated JSON array", buf, pos)
                continue

            ch = buf[pos]
            if ch == "]":
                return
            if ch == ",":
                pos += 1
                continue

            try:
                obj, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    raise
                if pos:
                    buf = buf[pos:]
                    pos = 0
                _read_more()
                continue

            yield obj
            pos = end
            if pos > (4 * chunk_size):
                buf = buf[pos:]
                pos = 0


def _mask_require_enabled() -> bool:
    """是否要求 mask reward 必须可用（否则直接报错退出）。

    用于避免训练时静默回退到距离 reward，导致长跑后才发现没用上 mask。
    """
    return os.environ.get("STEPCOUNT_MASK_REQUIRE", "0") == "1"


def _get_mask_helper():
    """获取或初始化 MaskRewardHelper（懒加载）"""
    global _MASK_HELPER
    
    if _MASK_HELPER is not None:
        return _MASK_HELPER
    
    # 检查环境变量（未设置则使用默认路径）
    masks_metadata_path = os.environ.get("STEPCOUNT_MASKS_METADATA", _DEFAULT_MASKS_METADATA_PATH)
    masks_dir = os.environ.get("STEPCOUNT_MASKS_DIR", _DEFAULT_MASKS_DIR)
    cache_masks = os.environ.get("STEPCOUNT_MASK_CACHE", "0") == "1"

    if _mask_require_enabled():
        missing = []
        if not masks_metadata_path or not os.path.exists(masks_metadata_path):
            missing.append(f"metadata not found: {masks_metadata_path}")
        if not masks_dir or not os.path.isdir(masks_dir):
            missing.append(f"masks_dir not found: {masks_dir}")
        if missing:
            raise RuntimeError(
                "[mask_reward] STEPCOUNT_MASK_REQUIRE=1 but mask resources are missing: "
                + " | ".join(missing)
                + " ; set STEPCOUNT_MASKS_METADATA/STEPCOUNT_MASKS_DIR correctly or disable STEPCOUNT_MASK_REQUIRE"
            )
    
    if masks_metadata_path and os.path.exists(masks_metadata_path):
        logger.info(f"初始化 MaskRewardHelper: {masks_metadata_path}")
        _mask_log_runtime_config_once(masks_metadata_path, masks_dir, cache_masks)
        _MASK_HELPER = MaskRewardHelper(
            masks_metadata_path=masks_metadata_path,
            masks_dir=masks_dir,
            cache_masks=cache_masks
        )
        return _MASK_HELPER
    
    return None


def _log_mask_fallback(reason: str):
    """仅打印一次的 mask 回退日志，避免刷屏"""
    if reason in _MASK_FALLBACK_LOGGED:
        return
    _MASK_FALLBACK_LOGGED.add(reason)
    msg = f"[mask_reward] fallback to point reward: {reason}"
    logger.warning(msg)
    print(msg)


def _mask_debug_enabled() -> bool:
    return os.environ.get("STEPCOUNT_MASK_DEBUG", "0") == "1"


def _mask_debug_every() -> int:
    try:
        return max(1, int(os.environ.get("STEPCOUNT_MASK_DEBUG_EVERY", "200")))
    except Exception:
        return 200


def _mask_debug_tick(**increments: int):
    """按样本计数的 debug 统计；到阈值后打印一次汇总。"""
    global _MASK_DEBUG_CALLS
    if not _mask_debug_enabled():
        return

    _MASK_DEBUG_CALLS += 1
    for k, v in increments.items():
        try:
            _MASK_DEBUG_STATS[k] += int(v)
        except Exception:
            _MASK_DEBUG_STATS[k] += 1

    every = _mask_debug_every()
    if _MASK_DEBUG_CALLS % every != 0:
        return

    # 打印聚合统计（不重置，便于长期观察）
    keys = [
        "mask_used",
        "mask_fallback_distance",
        "mask_helper_unavailable",
        "mask_gt_not_found",
        "mask_no_sequence",
        "mask_no_masks_loaded",
        "mask_hit_unused",
        "mask_hit_duplicate",
        "mask_miss",
    ]
    summary = {k: int(_MASK_DEBUG_STATS.get(k, 0)) for k in keys}
    used = max(int(summary.get("mask_used", 0)), 1)
    hit_unused = int(summary.get("mask_hit_unused", 0))
    hit_dup = int(summary.get("mask_hit_duplicate", 0))
    miss = int(summary.get("mask_miss", 0))
    rates = {
        "hit_unused_rate": round(hit_unused / used, 6),
        "miss_rate": round(miss / used, 6),
        "dup_rate": round(hit_dup / used, 6),
    }
    msg = f"[mask_reward][debug] calls={_MASK_DEBUG_CALLS} stats={summary} rates={rates}"
    logger.info(msg)
    print(msg)


def _traj_mask_debug_tick(**increments: int):
    """Trajectory 路径（无 GT 点序列）的 mask 命中统计。独立于 mask_point_reward 的统计。"""
    global _TRAJ_MASK_DEBUG_CALLS
    if not _mask_debug_enabled():
        return

    _TRAJ_MASK_DEBUG_CALLS += 1
    for k, v in increments.items():
        try:
            _TRAJ_MASK_DEBUG_STATS[k] += int(v)
        except Exception:
            _TRAJ_MASK_DEBUG_STATS[k] += 1

    every = _mask_debug_every()
    if _TRAJ_MASK_DEBUG_CALLS % every != 0:
        return

    keys = [
        "traj_steps_evaluated",
        "traj_hit_unused",
        "traj_hit_duplicate",
        "traj_miss",
        "traj_no_sequence",
        "traj_miss_decay_applied",
    ]
    summary = {k: int(_TRAJ_MASK_DEBUG_STATS.get(k, 0)) for k in keys}
    evaluated = max(int(summary.get("traj_steps_evaluated", 0)), 1)
    hit_unused = int(summary.get("traj_hit_unused", 0))
    hit_dup = int(summary.get("traj_hit_duplicate", 0))
    miss = int(summary.get("traj_miss", 0))
    rates = {
        "hit_unused_rate": round(hit_unused / evaluated, 6),
        "miss_rate": round(miss / evaluated, 6),
        "dup_rate": round(hit_dup / evaluated, 6),
    }
    msg = f"[traj_mask_reward][debug] calls={_TRAJ_MASK_DEBUG_CALLS} stats={summary} rates={rates}"
    logger.info(msg)
    print(msg)


def _mask_log_config_enabled() -> bool:
    # debug 打开时默认打印一次；也可显式开启
    return os.environ.get("STEPCOUNT_MASK_LOG_CONFIG", "0") == "1" or _mask_debug_enabled()


def _mask_log_runtime_config_once(
    masks_metadata_path: str,
    masks_dir: str,
    cache_masks: bool,
):
    """打印一次当前 mask reward 的关键开关/路径，方便在训练日志里核实是否生效。"""
    global _MASK_CONFIG_LOGGED
    if _MASK_CONFIG_LOGGED:
        return
    if not _mask_log_config_enabled():
        return

    _MASK_CONFIG_LOGGED = True
    msg = (
        "[mask_reward][config] "
        f"metadata={masks_metadata_path} exists={os.path.exists(masks_metadata_path)} | "
        f"masks_dir={masks_dir} exists={os.path.isdir(masks_dir)} | "
        f"cache_masks={cache_masks} | "
        f"require={os.environ.get('STEPCOUNT_MASK_REQUIRE','0')} | "
        f"prefill_by_turn={os.environ.get('STEPCOUNT_MASK_PREFILL_BY_TURN','0')} | "
        f"only_current_turn={os.environ.get('STEPCOUNT_MASK_ONLY_CURRENT_TURN','0')} | "
        f"debug={os.environ.get('STEPCOUNT_MASK_DEBUG','0')} every={os.environ.get('STEPCOUNT_MASK_DEBUG_EVERY','200')}"
    )
    logger.info(msg)
    print(msg)


def _traj_reason_debug_enabled() -> bool:
    return os.environ.get("STEPCOUNT_TRAJ_REASON_DEBUG", "1") == "1"


def _traj_reason_debug_every() -> int:
    try:
        return max(1, int(os.environ.get("STEPCOUNT_TRAJ_REASON_DEBUG_EVERY", "200")))
    except Exception:
        return 200


def _traj_event_log_enabled() -> bool:
    return os.environ.get("STEPCOUNT_TRAJ_EVENT_LOG", "1") == "1"


def _traj_event_log_every() -> int:
    try:
        return max(1, int(os.environ.get("STEPCOUNT_TRAJ_EVENT_LOG_EVERY", "200")))
    except Exception:
        return 200


def _traj_event_log_max() -> int:
    try:
        return max(1, int(os.environ.get("STEPCOUNT_TRAJ_EVENT_LOG_MAX", "200")))
    except Exception:
        return 200


def _traj_event_should_log(event_name: str) -> bool:
    if not _traj_event_log_enabled():
        return False
    _TRAJ_EVENT_LOG_COUNTS[event_name] += 1
    count = _TRAJ_EVENT_LOG_COUNTS[event_name]
    if count > _traj_event_log_max():
        return False
    if count == 1:
        return True
    return count % _traj_event_log_every() == 0


# =============================================================================
# 工具函数
# =============================================================================

def extract_sequence_id(image_path: str) -> str:
    """
    从图片路径提取序列ID（原始图片名，去掉 _N 后缀）
    
    示例:
        images/xxx.jpg -> xxx
        images/xxx_1.jpg -> xxx
        images/xxx_12.jpg -> xxx
    """
    basename = os.path.basename(image_path)
    # Some metadata JSONs may contain line breaks/spaces inside long filenames.
    # Normalize by removing all whitespace characters so basename matches dataset sequence_id.
    basename = re.sub(r"\s+", "", basename)
    name, ext = os.path.splitext(basename)
    
    # 去掉 _N 后缀（N是数字）
    match = re.match(r'^(.+?)_(\d+)$', name)
    if match:
        return match.group(1)
    return name


def extract_turn_number(image_path: str) -> int:
    """
    从图片路径提取轮次号
    
    示例:
        images/xxx.jpg -> 0 (第1轮)
        images/xxx_1.jpg -> 1 (第2轮)
    """
    basename = os.path.basename(image_path)
    basename = re.sub(r"\s+", "", basename)
    name, ext = os.path.splitext(basename)
    
    match = re.match(r'^(.+?)_(\d+)$', name)
    if match:
        return int(match.group(2))
    return 0


def normalize_coordinate(coord: Tuple[float, float], precision: int = 1) -> Tuple[float, float]:
    """标准化坐标（四舍五入到指定精度）"""
    return (round(coord[0], precision), round(coord[1], precision))


def _infer_image_path_from_images(images: Optional[List[Any]]) -> Optional[str]:
    """Best-effort recover image_path/sequence_id from `images`.

    In verl's SequentialFunctionRewardManager, `image_path` is extracted only when
    the first image is a str or a dict containing `path`. When images are PIL.Image
    objects, the manager often passes `images` but omits `image_path`, which makes
    trajectory dense mask scoring fail with `mask_no_sequence`.

    This function tries common representations and returns a string that can be
    either a filesystem path or a *sequence id* (both are accepted by
    `extract_sequence_id()` and the mask helper sequence index).
    """
    if not images or not isinstance(images, list):
        return None

    first = images[0]
    if isinstance(first, str) and first:
        return first

    if isinstance(first, dict):
        for key in ("path", "image_path", "url", "uri", "filename", "name", "id"):
            value = first.get(key)
            if isinstance(value, str) and value:
                return value

    # PIL.Image sometimes carries original path in `.filename`.
    for attr in ("filename", "path"):
        try:
            value = getattr(first, attr, None)
        except Exception:
            value = None
        if isinstance(value, str) and value:
            return value

    return None


# =============================================================================
# Mask Reward Helper
# =============================================================================

class MaskRewardHelper:
    """Mask Reward 辅助类 - 基于坐标匹配"""
    
    def __init__(
        self, 
        masks_metadata_path: Optional[str] = None,
        masks_dir: Optional[str] = None,
        cache_masks: bool = True
    ):
        self.metadata = {}  # sample_id -> metadata
        # Bounded LRU cache: sample_id -> masks array.
        # An OrderedDict gives O(1) move-to-end / pop-oldest so we can cap RAM
        # usage while still reusing masks across rollouts/steps that share a
        # sequence (rollout_n trajectories of the same prompt all need the same
        # masks, so a persistent cache removes ~rollout_n x redundant CephFS loads).
        self.masks_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._cache_lock = threading.Lock()  # guards masks_cache mutations (4 reward threads)
        # 0 / negative => unbounded (legacy behaviour). Default cap keeps RAM bounded.
        try:
            self._mask_cache_max = int(os.environ.get("STEPCOUNT_MASK_CACHE_MAX", "1024"))
        except Exception:
            self._mask_cache_max = 1024
        self.cache_masks = cache_masks
        self.masks_dir = masks_dir

        # 坐标索引 - (x, y) -> [metadata_list]
        self.coord_index = defaultdict(list)

        # 按序列分组的索引
        self.sequence_index = defaultdict(list)

        # 一些数据版本里：训练集使用 base sequence_id；而 masks metadata 用同前缀但追加 `_k{n}` 后缀（如 {base}_k0）。
        # 这里构建 base_id -> [sequence_id variants] 的别名索引，保证 trajectory/mask reward 能按前缀命中整组 masks。
        self.sequence_alias_index = defaultdict(list)
        
        if masks_metadata_path and os.path.exists(masks_metadata_path):
            self._load_metadata(masks_metadata_path)

    @staticmethod
    def _base_sequence_id(sequence_id: str) -> str:
        """将 sequence_id 归一到 base id。

        约定：若存在 `_k{n}` 后缀，则认为同属一个 base 序列。
        - base_k0 -> base
        - base_k12 -> base
        - base -> base
        """
        if not sequence_id:
            return ""
        m = re.match(r"^(.+?)_k\d+$", sequence_id)
        if m:
            return m.group(1)
        return sequence_id

    def _rebuild_sequence_alias_index(self):
        self.sequence_alias_index.clear()
        for seq_key in self.sequence_index.keys():
            base = self._base_sequence_id(seq_key)
            if not base:
                continue
            self.sequence_alias_index[base].append(seq_key)

        # 保持确定性顺序（尤其是合并多个 `_k` 变体时）
        for base in self.sequence_alias_index.keys():
            self.sequence_alias_index[base] = sorted(set(self.sequence_alias_index[base]))
    
    def _load_metadata(self, path: str):
        """加载元数据并构建索引"""
        logger.info(f"加载 mask 元数据: {path}")

        count = 0
        try:
            metadata_iter = _iter_json_array(path)
            for m in metadata_iter:
                if self._add_metadata_entry(m):
                    count += 1
        except json.JSONDecodeError:
            # Legacy fallback for older malformed manifests. This path is more
            # memory hungry, but keeps backward compatibility for repaired JSON.
            self.metadata.clear()
            self.coord_index.clear()
            self.sequence_index.clear()
            self.sequence_alias_index.clear()
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            content = re.sub(r'(\d)\s*\n(\s*")', r'\1,\n\2', content)
            for m in json.loads(content):
                if self._add_metadata_entry(m):
                    count += 1

        # 按轮次排序
        for seq_id in self.sequence_index:
            self.sequence_index[seq_id].sort(key=lambda x: x["turn_number"])

        # 构建 base -> variants 索引（支持 `_k` 后缀）
        self._rebuild_sequence_alias_index()
        
        # 统计
        duplicates = sum(1 for v in self.coord_index.values() if len(v) > 1)
        msg = (f"[mask_reward][metadata] loaded={len(self.metadata)} "
               f"raw_entries={count} unique_coords={len(self.coord_index)} "
               f"duplicate_coords={duplicates} sequences={len(self.sequence_index)} "
               f"aliases={len(self.sequence_alias_index)} cache_max={self._mask_cache_max}")
        logger.info(msg)
        if os.environ.get("STEPCOUNT_MASK_LOG_CONFIG", "0") == "1":
            print(msg)

    def _add_metadata_entry(self, m: Dict[str, Any]) -> bool:
        """Add one raw metadata dict to compact indexes."""
        try:
            sample_id = m.get("id")
            if not sample_id:
                return False

            image_path = m.get("image_path", "") or ""
            sequence_id = extract_sequence_id(image_path)
            turn_number = extract_turn_number(image_path)

            mask_path = m.get("mask_path") or ""
            if self.masks_dir and os.environ.get("STEPCOUNT_MASK_STORE_BASENAME", "1") == "1":
                mask_path = os.path.basename(mask_path)
            if os.environ.get("STEPCOUNT_MASK_IMAGE_BASENAME", "1") == "1":
                image_path = os.path.basename(image_path)

            size = m.get("image_size", [1024, 1024]) or [1024, 1024]
            try:
                image_size = (int(size[0]), int(size[1]))
            except Exception:
                image_size = (1024, 1024)
            
            points = m.get("points_pixel", [])
            point_xy: Tuple[float, float] = ()
            if points:
                if isinstance(points[0], list):
                    coord = (points[0][0], points[0][1])
                else:
                    coord = (points[0], points[1])
                point_xy = (float(coord[0]), float(coord[1]))

            labels = m.get("labels", ()) or ()
            if isinstance(labels, str):
                labels = (labels,)
            else:
                labels = tuple(str(x) for x in labels)

            meta = MaskMetadata(
                sample_id=str(sample_id),
                mask_path=mask_path,
                image_path=image_path,
                image_size=image_size,
                points_pixel=point_xy,
                labels=labels,
                turn_number=turn_number,
            )

            self.metadata[str(sample_id)] = meta
            if point_xy:
                self.coord_index[point_xy].append(meta)
            self.sequence_index[sequence_id].append({
                "sample_id": str(sample_id),
                "turn_number": turn_number,
                "metadata": meta
            })
            return True
        except Exception:
            return False
    
    def find_mask_by_coord(
        self, 
        point_2d: List[float], 
        label: str = None,
        problem_text: str = None
    ) -> Optional[Dict]:
        """通过坐标查找 mask"""
        coord = (point_2d[0], point_2d[1])
        candidates = self.coord_index.get(coord, [])
        
        if not candidates:
            # 尝试四舍五入匹配
            rounded = normalize_coordinate(coord)
            candidates = self.coord_index.get(rounded, [])
        
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        # 多个候选，需要消歧
        if label:
            for c in candidates:
                c_labels = c.get("labels", [])
                if label in c_labels:
                    return c
                for c_label in c_labels:
                    if label.lower() in c_label.lower() or c_label.lower() in label.lower():
                        return c
        
        if problem_text:
            for c in candidates:
                c_labels = c.get("labels", [])
                for c_label in c_labels:
                    if c_label.lower() in problem_text.lower():
                        return c
        
        return candidates[0]
    
    def get_sequence_from_mask(self, mask_metadata: Dict) -> Tuple[str, List[Dict]]:
        """从单个 mask 获取其所属的完整序列"""
        image_path = mask_metadata.get("image_path", "")
        sequence_id = extract_sequence_id(image_path)
        base_id = self._base_sequence_id(sequence_id)
        return base_id, self.get_sequence_samples(base_id)
    
    def get_sequence_samples(self, sequence_id: str) -> List[Dict]:
        """获取序列中的所有样本信息"""
        if not sequence_id:
            return []

        base_id = self._base_sequence_id(sequence_id)
        variants = self.sequence_alias_index.get(base_id)

        # 兼容：metadata 里可能没有 `_k` 后缀，或索引尚未构建
        if not variants:
            variants = [sequence_id] if sequence_id in self.sequence_index else []
            if base_id in self.sequence_index:
                variants = [base_id]

        merged: List[Dict] = []
        for v in variants:
            merged.extend(self.sequence_index.get(v, []))

        merged.sort(key=lambda x: x.get("turn_number", 0))
        return merged

    def get_sequence_from_image_path(self, image_path: Optional[str]) -> Tuple[str, List[Dict]]:
        """根据输入图片路径推断所属序列。"""
        if not image_path:
            return "", []

        sequence_id = extract_sequence_id(image_path)
        base_id = self._base_sequence_id(sequence_id)
        samples = self.get_sequence_samples(base_id)
        if samples:
            return base_id, samples

        image_name = os.path.basename(image_path)
        image_stem, _ = os.path.splitext(image_name)
        image_stem = re.sub(r"\s+", "", image_stem)
        base_stem = self._base_sequence_id(image_stem)
        samples = self.get_sequence_samples(base_stem)
        if samples:
            return base_stem, samples

        return "", []
    
    def _load_masks_from_disk(self, sample_id: str) -> Optional[np.ndarray]:
        """Load a sample's masks from disk (CephFS). No caching side-effects.

        Returns the exact same array the legacy code returned; only the
        load-vs-cache bookkeeping moved out so callers can cache results.
        """
        meta = self.metadata.get(sample_id)
        if not meta:
            return None

        mask_path = meta.get("mask_path")
        if not mask_path:
            return None

        # metadata 里可能存的是生成机的绝对路径；若当前机器路径不存在，尝试用 masks_dir + basename 回退。
        if os.path.isabs(mask_path):
            if not os.path.exists(mask_path) and self.masks_dir:
                alt_path = os.path.join(self.masks_dir, os.path.basename(mask_path))
                if os.path.exists(alt_path):
                    mask_path = alt_path
        else:
            if self.masks_dir:
                mask_path = os.path.join(self.masks_dir, os.path.basename(mask_path))

        if not os.path.exists(mask_path):
            return None

        try:
            data = np.load(mask_path)
            return data["masks"]
        except Exception as e:
            logger.error(f"加载 mask 失败 {mask_path}: {e}")
            return None

    def get_masks(self, sample_id: str) -> Optional[np.ndarray]:
        """获取单个样本的 mask（带有界 LRU 缓存）。

        缓存关闭时行为与旧版完全一致（每次从磁盘加载）。缓存开启时：
        - 命中：移动到 LRU 末尾并直接返回（避免重复 CephFS np.load）。
        - 未命中：在锁外执行磁盘加载（允许多线程并发 I/O），加载后在锁内写入，
          超出 STEPCOUNT_MASK_CACHE_MAX 时淘汰最久未用项。
        返回的数组与旧版逐 byte 一致，因此不改变任何 reward 数值。
        """
        if not self.cache_masks:
            return self._load_masks_from_disk(sample_id)

        # Fast path: cache hit.
        with self._cache_lock:
            cached = self.masks_cache.get(sample_id)
            if cached is not None:
                self.masks_cache.move_to_end(sample_id)
                return cached

        # Miss: load outside the lock so concurrent reward threads can issue
        # parallel CephFS reads instead of serialising on the cache lock.
        masks = self._load_masks_from_disk(sample_id)
        if masks is None:
            return None

        with self._cache_lock:
            existing = self.masks_cache.get(sample_id)
            if existing is not None:
                # Another thread loaded it concurrently; reuse that one.
                self.masks_cache.move_to_end(sample_id)
                return existing
            self.masks_cache[sample_id] = masks
            if self._mask_cache_max > 0:
                while len(self.masks_cache) > self._mask_cache_max:
                    self.masks_cache.popitem(last=False)  # evict least-recently-used
            return masks
    
    def get_sequence_masks(self, sequence_id: str) -> List[Tuple[str, int, np.ndarray]]:
        """获取整个序列的所有 masks"""
        result = []
        for sample_info in self.get_sequence_samples(sequence_id):
            sample_id = sample_info["sample_id"]
            turn_number = sample_info["turn_number"]
            masks = self.get_masks(sample_id)
            if masks is not None:
                result.append((sample_id, turn_number, masks))
        return result
    
    def get_image_size(self, sample_id: str) -> Tuple[int, int]:
        """获取图片尺寸 (width, height)"""
        meta = self.metadata.get(sample_id, {})
        size = meta.get("image_size", [1024, 1024])
        return size[0], size[1]
    
    def check_point_in_sequence_masks(
        self, 
        sequence_id: str,
        point: Tuple[float, float],
        used_sample_ids: Set[str],
        is_pixel_coord: bool = True,
        reference_sample_id: str = None,
        sequence_masks: Optional[List[Tuple[str, int, np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        """检查点是否落在序列的任意 mask 内。

        ``sequence_masks`` 可由调用方预加载并传入，避免对同一条 trajectory 的
        每个预测点都重新调用 ``get_sequence_masks``（后者会对整条序列逐个
        ``np.load``）。传入与否不影响判定结果，仅消除重复磁盘 I/O。
        """
        result = {
            "in_any_mask": False,
            "in_unused_mask": False,
            "matched_sample_id": None,
            "matched_turn": -1,
            "is_duplicate": False,
            "is_duplicate": False,
            "nearest_unused_distance": None,
        }
        
        if sequence_masks is None:
            sequence_masks = self.get_sequence_masks(sequence_id)
        if not sequence_masks:
            return result
        
        # 获取图片尺寸
        if reference_sample_id:
            img_w, img_h = self.get_image_size(reference_sample_id)
        else:
            img_w, img_h = self.get_image_size(sequence_masks[0][0])
        
        # 统一转换为归一化坐标，再映射到各自 mask 分辨率。
        # 说明：不同样本/turn 的 mask 可能是不同分辨率；若直接用 reference 的像素坐标再 clamp，容易导致系统性 miss。
        if is_pixel_coord:
            norm_x = float(point[0]) / max(float(img_w), 1.0)
            norm_y = float(point[1]) / max(float(img_h), 1.0)
        else:
            norm_x = float(point[0])
            norm_y = float(point[1])

        # clamp to sane range (avoid negative / >1 after numerical issues)
        norm_x = max(0.0, min(norm_x, 1.0))
        norm_y = max(0.0, min(norm_y, 1.0))
        
        # 遍历序列中的所有mask
        for sample_id, turn_number, masks in sequence_masks:
            h, w = int(masks.shape[1]), int(masks.shape[2])
            px = int(round(norm_x * float(w)))
            py = int(round(norm_y * float(h)))
            check_px = max(0, min(px, w - 1))
            check_py = max(0, min(py, h - 1))
            
            mask = masks[0]  # 每个样本只有1个mask
            if mask[check_py, check_px]:
                result["in_any_mask"] = True
                result["matched_sample_id"] = sample_id
                result["matched_turn"] = turn_number
                
                if sample_id in used_sample_ids:
                    result["is_duplicate"] = True
                else:
                    result["in_unused_mask"] = True
                
                break
        

        # Miss case: compute distance to nearest unused mask center for shaping.
        if not result["in_any_mask"]:
            best_dist = None
            for sample_id, turn_number, masks in sequence_masks:
                if sample_id in used_sample_ids:
                    continue
                meta = self.metadata.get(sample_id, {})
                points = meta.get("points_pixel", [])
                if not points:
                    continue
                if isinstance(points[0], list):
                    cx, cy = float(points[0][0]), float(points[0][1])
                else:
                    cx, cy = float(points[0]), float(points[1])
                cand_w, cand_h = self.get_image_size(sample_id)
                if not cand_w or not cand_h:
                    cand_w, cand_h = 1024, 1024
                cx_norm = cx / max(float(cand_w), 1.0)
                cy_norm = cy / max(float(cand_h), 1.0)
                # L1 distance in normalized coords
                dist = abs(norm_x - cx_norm) + abs(norm_y - cy_norm)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
            if best_dist is not None:
                result["nearest_unused_distance"] = best_dist

        return result
    
    def clear_cache(self):
        self.masks_cache.clear()


# =============================================================================
# 安全工具函数
# =============================================================================

def safe_exp(x: float, min_val: float = -50.0, max_val: float = 50.0) -> float:
    """安全的指数函数，防止溢出"""
    clamped_x = max(min_val, min(x, max_val))
    return math.exp(clamped_x)


def clamp_reward(reward: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """将 reward 限制在有效区间内"""
    if not math.isfinite(reward):
        return 0.0
    return max(min_val, min(reward, max_val))


# =============================================================================
# 解析函数
# =============================================================================

def parse_number(text: str) -> Optional[int]:
    """从文本中提取数字"""
    try:
        return int(text.strip())
    except ValueError:
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])
    except Exception:
        pass
    return None


def compute_adaptive_max_turns_from_gt(
    gt_data: Dict,
    max_turns: int,
    adaptive_max_turns: bool = False,
    adaptive_max_turns_margin: int = 2,
) -> int:
    """Align reward's max-turn rule with rollout's per-sample cap."""
    hard_cap = max(1, int(max_turns))
    if isinstance(adaptive_max_turns, str):
        adaptive_enabled = adaptive_max_turns.strip().lower() in ("1", "true", "yes", "on")
    else:
        adaptive_enabled = bool(adaptive_max_turns)
    if not adaptive_enabled:
        return hard_cap

    expected_steps = parse_number(str(gt_data.get("count_number", 0))) or 0
    if expected_steps <= 0:
        point_sequence = gt_data.get("point_sequence", [])
        if isinstance(point_sequence, list):
            expected_steps = len(point_sequence)
    if expected_steps < 0:
        return hard_cap

    margin = max(0, int(adaptive_max_turns_margin))
    return max(1, min(hard_cap, int(expected_steps) + 1 + margin))


def parse_ground_truth(ground_truth: str) -> Dict:
    """解析 GT"""
    try:
        gt_data = json.loads(ground_truth)
        if isinstance(gt_data, dict):
            count_number = None
            for k in ("count_number", "final_count", "answer", "N", "total_count"):
                if k in gt_data and gt_data.get(k) is not None:
                    count_number = gt_data.get(k)
                    break

            point_sequence_raw = (
                gt_data.get("point_gts")
                or gt_data.get("point_trajectory")
                or gt_data.get("trajectory_points")
                or gt_data.get("points")
            )

            point_sequence = []
            if isinstance(point_sequence_raw, list):
                for item in point_sequence_raw:
                    if isinstance(item, dict) and "point_2d" in item:
                        point_sequence.append(
                            {
                                "point_2d": item["point_2d"],
                                "label": item.get("label", gt_data.get("label", "object")),
                            }
                        )
                    elif isinstance(item, list) and len(item) >= 2:
                        point_sequence.append(
                            {
                                "point_2d": [item[0], item[1]],
                                "label": gt_data.get("label", "object"),
                            }
                        )

            if count_number is not None and point_sequence:
                parsed_count = parse_number(str(count_number))
                if parsed_count is not None:
                    return {
                        "type": "trajectory",
                        "count_number": parsed_count,
                        "point_sequence": point_sequence,
                        "max_turns": gt_data.get("max_turns"),
                    }

            if count_number is None and point_sequence:
                return {
                    "type": "trajectory",
                    "count_number": len(point_sequence),
                    "point_sequence": point_sequence,
                    "max_turns": gt_data.get("max_turns"),
                }

            if count_number is not None and "point_2d" not in gt_data:
                parsed_count = parse_number(str(count_number))
                if parsed_count is not None:
                    return {
                        "type": "trajectory",
                        "count_number": parsed_count,
                        "point_sequence": [],
                        "max_turns": gt_data.get("max_turns"),
                        "no_point_gt": True,
                    }

        if isinstance(gt_data, dict) and "point_2d" in gt_data:
            return {
                "type": "point",
                "point_2d": gt_data["point_2d"],
                "label": gt_data.get("label", "object"),
                "count_number": gt_data.get("count_number", 1)
            }
    except (json.JSONDecodeError, TypeError):
        pass
    
    count_num = parse_number(ground_truth)
    if count_num is not None:
        # Some training datasets only store the final answer as a plain number string.
        # Enable this switch to force those samples into trajectory reward (mask-based point scoring).
        if os.environ.get("STEPCOUNT_FORCE_TRAJECTORY_FOR_NUMERIC_GT", "0") == "1":
            return {
                "type": "trajectory",
                "count_number": int(count_num),
                "point_sequence": [],
                "max_turns": None,
                "no_point_gt": True,
            }
        return {"type": "count", "count_number": count_num}
    
    return {"type": "unknown"}


def parse_coordinates_from_text(text: str) -> Optional[List[Tuple[float, float]]]:
    """从文本中解析坐标列表"""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "point_2d" in data:
            point_2d = data["point_2d"]
            return [(float(point_2d[0]), float(point_2d[1]))]
        elif isinstance(data, list) and len(data) >= 2:
            return [(float(data[0]), float(data[1]))]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    
    try:
        coord_pattern = r'\[\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\]'
        matches = re.findall(coord_pattern, text)
        if matches:
            return [(float(x), float(y)) for x, y in matches]
    except Exception:
        pass
    
    return None


def _normalize_point_keys(raw: str) -> str:
    """Recover the canonical "point_2d" JSON key from benign RL-drift typos.

    TARGETED on purpose (only quoted JSON keys observed in rollouts): it cannot
    turn arbitrary broken JSON valid, and format scoring can still distinguish
    canonical keys from recovered aliases via TRAJ_FORMAT_STRICT_KEY.

    Observed recoverable aliases:
      - "point_22d" / "point_222d" / ... -> "point_2d"
      - "point_22" / "point_222" / ...   -> "point_2d"
      - "pointpointlabel"                 -> "point_2d"
      - "pointinglabel"                   -> "point_2d"

    This keeps answer/point reward computable while strict format credit can
    still require the original "point_2d" key.
    Gated by TRAJ_POINT_KEY_NORMALIZE (default ON)."""
    if os.environ.get("TRAJ_POINT_KEY_NORMALIZE", "1") != "1":
        return raw
    return re.sub(
        r'"(?:point_2+d|point_2{2,}|pointpointlabel|pointinglabel)"\s*:',
        '"point_2d":',
        raw,
    )


_POINT_KEY_ALIAS_RE = re.compile(
    r'"(?:point_2{2,}d|point_2{2,}|point_2{2,}_2d|pointpointlabel|pointinglabel)"\s*:'
)
_BROKEN_POINT_KEY_ALIAS_RE = re.compile(r'"point_2{2,}[A-Za-z0-9_]*(?=\s*\[)')


def _count_point_key_aliases(predict: str) -> int:
    """Count observed non-canonical point-key aliases for drift monitoring."""
    return len(_POINT_KEY_ALIAS_RE.findall(predict)) + len(_BROKEN_POINT_KEY_ALIAS_RE.findall(predict))


def _parse_pred_point(predict: str) -> Optional[Tuple[Tuple[float, float], str]]:
    """从预测中解析点坐标与标签

    严格模式 (TRAJ_POINT_STRICT_JSON=1, 默认)：JSON 解析失败或缺少
    "point_2d" 键时直接返回 None，不再回退到 regex 提取坐标。
    这是为了防止 reward hacking：模型输出破损 JSON 如 {"point_2 [x,y]}
    仄获相同 point_reward 但省更少 token，导致策略向格式漂移。
    （但良性 key typo "point_22d" 先经 _normalize_point_keys 归一恢复，
     避免漂移诱发的 format 死门吸收态。）

    宽松模式 (TRAJ_POINT_STRICT_JSON=0, 向后兼容)：JSON 失败时回退到 regex
    提取 [x,y] 坐标，恢复 v26 之前的行为。
    """
    content_match = re.search(r"<point>(.*?)</point>", predict, re.DOTALL)
    if not content_match:
        return None

    raw = _normalize_point_keys(content_match.group(1).strip())

    # 优先 JSON
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "point_2d" in data:
            point_2d = data["point_2d"]
            label = data.get("label", "object")
            return ((float(point_2d[0]), float(point_2d[1])), label)
    except Exception:
        pass

    # 严格模式：JSON 失败直接拒绝，导致 point_reward=0，强迫模型维持合法 JSON。
    if os.environ.get("TRAJ_POINT_STRICT_JSON", "1") == "1":
        return None

    # 向后兼容兔底路径 (TRAJ_POINT_STRICT_JSON=0)
    coords = parse_coordinates_from_text(raw)
    if coords and len(coords) > 0:
        return (coords[0], "object")

    return None


def _parse_pred_points(predict: str) -> List[Tuple[Tuple[float, float], str]]:
    parsed_points: List[Tuple[Tuple[float, float], str]] = []
    matches = re.findall(r"<point>(.*?)</point>", predict, re.DOTALL)
    for raw in matches:
        parsed = _parse_pred_point(f"<point>{raw}</point>")
        if parsed is not None:
            parsed_points.append(parsed)
    return parsed_points


def _extract_last_answer_number(predict: str) -> Optional[int]:
    answer_matches = re.findall(r"<answer>(.*?)</answer>", predict, re.DOTALL)
    if not answer_matches:
        return None
    return parse_number(answer_matches[-1].strip())


def _has_answer_tag(predict: str) -> bool:
    return re.search(r"<answer>.*?</answer>", predict, re.DOTALL) is not None


def _trajectory_tag_balance_violation(predict: str) -> bool:
    """Reject unbalanced or repeated trajectory control tags."""
    point_open = len(re.findall(r"<point>", predict))
    point_close = len(re.findall(r"</point>", predict))
    answer_open = len(re.findall(r"<answer>", predict))
    answer_close = len(re.findall(r"</answer>", predict))
    return (
        point_open != point_close
        or answer_open != answer_close
        or answer_close > 1
    )


def _point_count_number_integrity(
    predict: str,
    mode: Optional[str] = None,
) -> Tuple[bool, int, int]:
    """Check optional per-point count_number fields are monotonic 1..N."""
    _, _, missing, mismatched = _point_count_number_step_scores(predict, mode=mode)
    return (missing + mismatched) > 0, missing, mismatched


def _point_count_number_step_scores(
    predict: str,
    mode: Optional[str] = None,
) -> Tuple[List[float], float, int, int]:
    """Return per-point count_number correctness and aggregate score.

    The count number is only a model-output training signal. Rollout must not trust
    it to update image state, but reward can softly prefer `count_number == step`.

    Modes:
      - off/0/false: disabled, all points are neutral.
      - if_present/default: present count_number fields are checked; missing is neutral.
      - required/strict/sequence: missing count_number is a soft 0 for that step.
    """
    check_mode = (mode or os.environ.get("TRAJ_POINT_COUNT_NUMBER_CHECK", "if_present")).strip().lower()
    point_tags = re.findall(r"<point>(.*?)</point>", predict, re.DOTALL)
    if check_mode in ("0", "off", "false", "none", "no"):
        neutral_scores = [1.0 for _ in point_tags]
        return neutral_scores, 1.0, 0, 0

    require_present = check_mode in ("required", "strict", "sequence", "all")
    step_scores: List[float] = []
    missing = 0
    mismatched = 0
    for idx, raw in enumerate(point_tags, start=1):
        count_value = None
        try:
            data = json.loads(_normalize_point_keys(raw.strip()))
            if isinstance(data, dict):
                count_value = data.get("count_number")
        except Exception:
            pass

        parsed_count = None
        if count_value is not None:
            parsed_count = parse_number(str(count_value))

        if parsed_count is None:
            if require_present:
                missing += 1
                step_scores.append(0.0)
            else:
                step_scores.append(1.0)
            continue

        if int(parsed_count) != idx:
            mismatched += 1
            step_scores.append(0.0)
        else:
            step_scores.append(1.0)

    if not step_scores:
        return [], 1.0, missing, mismatched
    return step_scores, clamp_reward(sum(step_scores) / float(len(step_scores))), missing, mismatched


def _trajectory_format_reward(predict: str, expected_point_steps: int) -> float:
    # --- Hard (structural) format failures: genuine garbage MUST be rejected to 0. ---
    if not _has_answer_tag(predict):
        return 0.0

    # Answer must be parseable as an integer; otherwise treat as format failure.
    if _extract_last_answer_number(predict) is None:
        return 0.0

    point_tags = re.findall(r"<point>(.*?)</point>", predict, re.DOTALL)
    if expected_point_steps > 0 and len(point_tags) == 0:
        return 0.0

    if _trajectory_tag_balance_violation(predict):
        return 0.0

    # No points emitted (and none required) -> structurally well-formed.
    if len(point_tags) == 0:
        return 1.0

    # --- Per-point JSON validity for FORMAT credit (two-tier: exact vs benign typo). ---
    # The PARSER used for answer/point rewards (_parse_pred_point) stays typo-tolerant
    # via _normalize_point_keys, so answer/point are unaffected and never crash. FORMAT
    # credit, however, can be made to reward the *canonical* key so that benign key-drift
    # ("point_22d"/"point_222d"/... -> "point_2d") loses score and the policy gets a
    # gradient back toward the original format.
    #   - n_exact: valid JSON whose key is literally "point_2d".
    #   - n_typo : valid JSON only AFTER benign key-drift normalization.
    #
    # TRAJ_FORMAT_STRICT_KEY (default "0" -> fully backward-compatible / inert):
    #   0 : typo counts == exact (identical to the previous normalize-first logic).
    #   1 : typo points earn only TRAJ_FORMAT_TYPO_CREDIT (default 0.0) of the credit,
    #       exact points earn full credit.
    # SAFETY: when strict scoring drives format_score to 0 on a heavily-drifted batch,
    # the legacy TRAJ_FORMAT_REJECTION=1 hard-zero would collapse the whole trajectory
    # (and, on a >50% typo batch, the whole group -> dead gradient). Pair strict scoring
    # with EITHER TRAJ_FORMAT_REJECTION=structural (hard-zero only on real structural
    # garbage) OR TRAJ_FORMAT_TYPO_CREDIT > 0 (keeps format_score strictly above 0).
    n_exact = 0
    n_typo = 0
    for raw in point_tags:
        s = raw.strip()
        ok_exact = False
        try:
            data = json.loads(s)
            if isinstance(data, dict) and "point_2d" in data:
                ok_exact = True
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        if ok_exact:
            n_exact += 1
            continue
        try:
            data = json.loads(_normalize_point_keys(s))
            if isinstance(data, dict) and "point_2d" in data:
                n_typo += 1
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    strict_key = str(os.environ.get("TRAJ_FORMAT_STRICT_KEY", "0")).lower() in ("1", "true", "yes")
    if strict_key:
        try:
            typo_credit = clamp_reward(float(os.environ.get("TRAJ_FORMAT_TYPO_CREDIT", "0.0")))
        except (TypeError, ValueError):
            typo_credit = 0.0  # non-numeric env -> safe strict default (do not crash scoring)
        valid = float(n_exact) + typo_credit * float(n_typo)
        n_binary_ok = n_exact  # binary mode requires the canonical key on every point
    else:
        valid = float(n_exact + n_typo)  # legacy: benign typo counts as fully valid
        n_binary_ok = n_exact + n_typo

    # GRADED format (default): score = fraction of valid-JSON points, NOT binary
    # all-or-nothing. Binary strict-JSON over a long point chain is geometrically
    # fragile (P(format=1) ~ p^N): a single malformed point among 11-20 zeroes the
    # whole trajectory -> format_rejection death-trap -> the model abandons pointing
    # even when it pointed well and answered correctly. Grading keeps format pressure
    # (the all-invalid / structural cases above still hard-zero -> format_rejection)
    # while giving recoverable gradient on partial JSON drift.
    # Set TRAJ_FORMAT_GRADED=0 to restore legacy binary strict-JSON (e.g. for 0-10).
    graded = str(os.environ.get("TRAJ_FORMAT_GRADED", "1")).lower() in ("1", "true", "yes")
    if not graded:
        format_score = 1.0 if n_binary_ok == len(point_tags) else 0.0
    else:
        format_score = clamp_reward(valid / float(len(point_tags)))

    _, count_number_score, _, _ = _point_count_number_step_scores(predict)
    count_format_weight = clamp_reward(float(os.environ.get("TRAJ_COUNT_NUMBER_FORMAT_WEIGHT", "0.50")))
    if count_format_weight > 0:
        format_score = clamp_reward(format_score * (1.0 - count_format_weight * (1.0 - count_number_score)))
    return format_score


def _trajectory_point_dense_reward(
    predict: str,
    gt_data: Dict,
    img_width: int,
    img_height: int,
    image_path: Optional[str],
    problem: Optional[str],
) -> float:
    point_sequence = gt_data.get("point_sequence", [])
    if not point_sequence:
        return 0.0

    pred_points = _parse_pred_points(predict)
    if not pred_points:
        return 0.0

    used_sample_ids: Set[str] = set()
    total_score = 0.0
    matched_steps = min(len(pred_points), len(point_sequence))

    for step_idx in range(matched_steps):
        pred_point, pred_label = pred_points[step_idx]
        gt_step = point_sequence[step_idx]
        gt_point = gt_step.get("point_2d", [0.0, 0.0])
        gt_label = gt_step.get("label", pred_label)

        synthetic_predict = json.dumps({"point_2d": [pred_point[0], pred_point[1]], "label": pred_label})
        synthetic_predict = f"<point>{synthetic_predict}</point>"

        mask_score = mask_point_reward(
            synthetic_predict,
            gt_point,
            gt_label=gt_label,
            img_width=img_width,
            img_height=img_height,
            problem_text=problem,
            used_sample_ids=used_sample_ids,
            image_path=image_path,
        )
        if mask_score is None:
            step_score = point_reward(synthetic_predict, gt_point, img_width=img_width, img_height=img_height)
        else:
            step_score = mask_score

        total_score += clamp_reward(step_score)

    point_dense_score = clamp_reward(total_score / max(len(point_sequence), 1))
    extra_pt_lambda = float(os.environ.get("TRAJ_EXTRA_POINT_PENALTY_LAMBDA", "1.0"))
    extra_points = max(0, len(pred_points) - len(point_sequence))
    if extra_points > 0 and extra_pt_lambda > 0:
        point_dense_score = clamp_reward(
            point_dense_score - extra_pt_lambda * extra_points / float(max(len(point_sequence), 1))
        )

    return point_dense_score


def _trajectory_point_step_scores(
    predict: str,
    gt_data: Dict,
    img_width: int,
    img_height: int,
    image_path: Optional[str],
    problem: Optional[str],
) -> List[float]:
    """Return per-step point scores for trajectory tasks (only when GT has point_sequence).

    This is intended for debugging/diagnostics. Default training uses the averaged
    `point_dense_score` from `_trajectory_point_dense_reward`.
    """
    point_sequence = gt_data.get("point_sequence", [])
    if not point_sequence:
        return []

    pred_points = _parse_pred_points(predict)
    if not pred_points:
        return []

    used_sample_ids: Set[str] = set()
    matched_steps = min(len(pred_points), len(point_sequence))
    step_scores: List[float] = []

    for step_idx in range(matched_steps):
        pred_point, pred_label = pred_points[step_idx]
        gt_step = point_sequence[step_idx]
        gt_point = gt_step.get("point_2d", [0.0, 0.0])
        gt_label = gt_step.get("label", pred_label)

        synthetic_predict = json.dumps({"point_2d": [pred_point[0], pred_point[1]], "label": pred_label})
        synthetic_predict = f"<point>{synthetic_predict}</point>"

        mask_score = mask_point_reward(
            synthetic_predict,
            gt_point,
            gt_label=gt_label,
            img_width=img_width,
            img_height=img_height,
            problem_text=problem,
            used_sample_ids=used_sample_ids,
            image_path=image_path,
        )
        if mask_score is None:
            step_score = point_reward(synthetic_predict, gt_point, img_width=img_width, img_height=img_height)
        else:
            step_score = mask_score

        step_scores.append(clamp_reward(step_score))

    return step_scores


def _trajectory_point_dense_reward_without_gt_points(
    predict: str,
    gt_data: Dict,
    image_path: Optional[str],
) -> float:
    point_score, _, _, _, _, _ = _trajectory_point_dense_reward_without_gt_points_details(
        predict=predict,
        gt_data=gt_data,
        image_path=image_path,
    )
    return clamp_reward(point_score)


def _trajectory_point_dense_reward_without_gt_points_details(
    predict: str,
    gt_data: Dict,
    image_path: Optional[str],
) -> Tuple[float, List[float], List[float], List[float], float, float]:
    """Point reward for trajectory tasks when GT has no explicit point_sequence.

    Uses image_path to locate the full sequence masks, and scores each predicted point by
    whether it hits an unused mask in that sequence.

    Returns:
        point_dense_score: float
        step_contribs: per-step contribution to point_dense_score (each is 1/target_count or 0)
        step_hit_any: per-step indicator (1 if hits any mask)
        step_is_duplicate: per-step indicator (1 if hits a used mask)
        target_count: float
        eval_steps: float
    """
    helper = _get_mask_helper()
    if helper is None:
        return 0.0, [], [], [], 0.0, 0.0

    pred_points = _parse_pred_points(predict)
    if not pred_points:
        return 0.0, [], [], [], 0.0, 0.0

    sequence_id, sequence_samples = helper.get_sequence_from_image_path(image_path)
    if not sequence_samples:
        _mask_debug_tick(mask_no_sequence=1)

        # Fallback for datasets that only provide numeric GT answer and no usable
        # image_path/sequence id (common in some eval parquet exports).
        # This avoids collapsing point_reward to 0 for all samples.
        no_seq_mode = os.environ.get("TRAJ_NO_SEQUENCE_FALLBACK", "count_iou").strip().lower()
        if no_seq_mode in ("0", "off", "false", "none", "zero"):
            return 0.0, [], [], [], 0.0, 0.0

        gt_count_raw = gt_data.get("count_number")
        gt_count = parse_number(str(gt_count_raw)) if gt_count_raw is not None else None
        pred_count = len(pred_points)

        if gt_count is None or gt_count < 0:
            # If GT count is unavailable, keep conservative zero fallback.
            return 0.0, [], [], [], 0.0, 0.0

        target_count = int(gt_count)
        if target_count <= 0:
            # GT expects no points.
            score = 1.0 if pred_count == 0 else 0.0
            return clamp_reward(score), [], [], [], float(target_count), float(pred_count)

        if no_seq_mode == "count_exact":
            score = 1.0 if pred_count == target_count else 0.0
        elif no_seq_mode == "count_decay":
            alpha = float(os.environ.get("TRAJ_NO_SEQUENCE_FALLBACK_ALPHA", "6.0"))
            rel_err = abs(pred_count - target_count) / float(max(target_count, 1))
            score = safe_exp(-alpha * rel_err)
        elif no_seq_mode == "stat_mask_sim":
            # Statistical mask simulator: approximate mask-based reward using
            # empirical hit/miss rates from training data.  Matches the expected
            # value of the real mask reward much more closely than count_iou
            # (error < 0.005 vs +0.228 for count_iou when pred == GT).
            _sim_hit_rate = float(os.environ.get("TRAJ_SIM_HIT_RATE", "0.736"))
            _sim_miss_decay = float(os.environ.get("TRAJ_SIM_MISS_DECAY_AVG", "0.15"))
            eval_steps = min(pred_count, target_count)
            per_step = (_sim_hit_rate + (1.0 - _sim_hit_rate) * _sim_miss_decay) / float(target_count)
            base = eval_steps * per_step
            extra_penalty = max(0, pred_count - target_count) / float(target_count)
            score = max(0.0, min(1.0, base - extra_penalty))
        else:
            # default: count_iou (symmetric over/under-count penalty)
            score = float(min(pred_count, target_count)) / float(max(pred_count, target_count)) if max(pred_count, target_count) > 0 else 1.0

        return clamp_reward(score), [], [], [], float(target_count), float(pred_count)

    # IMPORTANT:
    # - gt_count == 0 is a valid case (no points expected). Do NOT fall back to len(sequence_samples).
    # - gt_count is missing/invalid -> treat as unknown and fall back to sequence length.
    gt_count_raw = gt_data.get("count_number")
    gt_count = None
    if gt_count_raw is not None:
        gt_count = parse_number(str(gt_count_raw))

    if gt_count == 0:
        # No points expected: reward is 1 iff the model also predicts no points.
        # Here pred_points is non-empty already, so the score is 0.
        return 0.0, [], [], [], 0.0, 0.0

    if gt_count is None or gt_count < 0:
        target_count = int(len(sequence_samples))
    else:
        target_count = int(gt_count)

    if target_count <= 0:
        return 0.0, [], [], [], float(target_count), 0.0

    used_sample_ids: Set[str] = set()
    matched_unused = 0.0
    max_eval_steps = int(min(len(pred_points), target_count))
    ref_sample_id = sequence_samples[0].get("sample_id") if sequence_samples else None

    # Load the sequence's masks ONCE for this trajectory and reuse across every
    # predicted point. Previously check_point_in_sequence_masks() reloaded the
    # whole sequence (one np.load per sample) on every step -> O(max_eval_steps x
    # seq_len) CephFS reads per trajectory. This is the dominant cost when GPUs
    # idle during reward evaluation. Result is numerically identical.
    cached_sequence_masks = helper.get_sequence_masks(sequence_id)

    denom = float(max(target_count, 1))
    step_contribs: List[float] = []
    step_hit_any: List[float] = []
    step_is_duplicate: List[float] = []

    # Distance decay parameters for miss shaping (mirrors mask_point_reward V4 defaults)
    # Controllable via env vars: TRAJ_MISS_DECAY_ENABLE, TRAJ_MISS_DECAY_ALPHA, etc.
    miss_decay_enable = os.environ.get("TRAJ_MISS_DECAY_ENABLE", "1") == "1"
    miss_decay_alpha = float(os.environ.get("TRAJ_MISS_DECAY_ALPHA", "20.0"))
    miss_decay_alpha_penalty = float(os.environ.get("TRAJ_MISS_DECAY_ALPHA_PENALTY", "50.0"))
    miss_decay_failure_threshold = float(os.environ.get("TRAJ_MISS_DECAY_FAILURE_THRESHOLD", "0.02"))

    total_miss_decay_score = 0.0

    for step_idx in range(max_eval_steps):
        pred_point, _ = pred_points[step_idx]
        is_pixel_coord = not (0.0 <= pred_point[0] <= 1.0 and 0.0 <= pred_point[1] <= 1.0)
        check = helper.check_point_in_sequence_masks(
            sequence_id=sequence_id,
            point=pred_point,
            used_sample_ids=used_sample_ids,
            is_pixel_coord=is_pixel_coord,
            reference_sample_id=ref_sample_id,
            sequence_masks=cached_sequence_masks,
        )

        hit_any = 1.0 if check.get("in_any_mask") else 0.0
        hit_unused = 1.0 if check.get("in_unused_mask") else 0.0
        is_dup = 1.0 if check.get("is_duplicate") else 0.0

        step_hit_any.append(hit_any)
        step_is_duplicate.append(is_dup)

        if hit_unused > 0.0:
            step_contribs.append(clamp_reward(1.0 / denom))
            sample_id = check.get("matched_sample_id")
            if sample_id:
                used_sample_ids.add(sample_id)
            matched_unused += 1.0
            _traj_mask_debug_tick(traj_steps_evaluated=1, traj_hit_unused=1)
        elif is_dup > 0.0:
            step_contribs.append(0.0)
            _traj_mask_debug_tick(traj_steps_evaluated=1, traj_hit_duplicate=1)
        else:
            # Miss: apply distance decay shaping if enabled
            miss_score = 0.0
            if miss_decay_enable:
                nearest_dist = check.get("nearest_unused_distance")
                if nearest_dist is not None and nearest_dist >= 0:
                    # nearest_dist is normalized [0, ~1] from helper
                    nd = float(nearest_dist)
                    if nd <= 0.0:
                        miss_score = 1.0  # extremely close to mask edge
                    elif nd > miss_decay_failure_threshold:
                        score_at_th = safe_exp(-miss_decay_alpha * miss_decay_failure_threshold)
                        miss_score = score_at_th * safe_exp(-miss_decay_alpha_penalty * (nd - miss_decay_failure_threshold))
                    else:
                        miss_score = safe_exp(-miss_decay_alpha * nd)
                    miss_score = clamp_reward(miss_score)
                    total_miss_decay_score += miss_score / denom
                    _traj_mask_debug_tick(traj_steps_evaluated=1, traj_miss=1, traj_miss_decay_applied=1)
                else:
                    _traj_mask_debug_tick(traj_steps_evaluated=1, traj_miss=1)
            else:
                _traj_mask_debug_tick(traj_steps_evaluated=1, traj_miss=1)
            step_contribs.append(clamp_reward(miss_score / denom))

    point_dense_score = clamp_reward(float(matched_unused) / denom + total_miss_decay_score)
    # --- Extra point penalty: eliminate structural overcounting advantage ---
    _extra_pt_lambda = float(os.environ.get("TRAJ_EXTRA_POINT_PENALTY_LAMBDA", "1.0"))
    _extra_points = max(0, len(pred_points) - target_count)
    if _extra_points > 0 and _extra_pt_lambda > 0:
        _extra_penalty = _extra_pt_lambda * _extra_points / denom
        point_dense_score = clamp_reward(point_dense_score - _extra_penalty)
    return point_dense_score, step_contribs, step_hit_any, step_is_duplicate, float(target_count), float(max_eval_steps)


def _trajectory_coverage_stats(
    predict: str,
    gt_data: Dict,
    image_path: Optional[str],
) -> Tuple[int, int, int]:
    """Return (gt_count, matched_unused, uncovered) for coverage penalty.

    Uses mask-based matching (gt_count - matched_unused) rather than
    expected_steps - num_predicted. Falls back to step-count proxy when masks
    are unavailable.
    """
    pred_points = _parse_pred_points(predict)
    gt_count_raw = gt_data.get("count_number")
    gt_count = parse_number(str(gt_count_raw)) if gt_count_raw is not None else None
    expected_steps = len(gt_data.get("point_sequence", []))
    if expected_steps <= 0 and gt_count is not None:
        expected_steps = int(gt_count)
    if gt_count is None or gt_count < 0:
        gt_count = int(expected_steps)
    gt_count = int(max(gt_count, 0))

    helper = _get_mask_helper()
    if helper is None or not image_path or gt_count <= 0:
        matched_proxy = min(len(pred_points), gt_count)
        uncovered = max(0, gt_count - matched_proxy)
        return gt_count, matched_proxy, uncovered

    sequence_id, sequence_samples = helper.get_sequence_from_image_path(image_path)
    if not sequence_samples:
        matched_proxy = min(len(pred_points), gt_count)
        uncovered = max(0, gt_count - matched_proxy)
        return gt_count, matched_proxy, uncovered

    used_sample_ids: Set[str] = set()
    matched_unused = 0
    max_eval_steps = int(min(len(pred_points), gt_count))
    ref_sample_id = sequence_samples[0].get("sample_id") if sequence_samples else None
    cached_sequence_masks = helper.get_sequence_masks(sequence_id)

    for step_idx in range(max_eval_steps):
        pred_point, _ = pred_points[step_idx]
        is_pixel_coord = not (0.0 <= pred_point[0] <= 1.0 and 0.0 <= pred_point[1] <= 1.0)
        check = helper.check_point_in_sequence_masks(
            sequence_id=sequence_id,
            point=pred_point,
            used_sample_ids=used_sample_ids,
            is_pixel_coord=is_pixel_coord,
            reference_sample_id=ref_sample_id,
            sequence_masks=cached_sequence_masks,
        )
        if check.get("in_unused_mask"):
            sample_id = check.get("matched_sample_id")
            if sample_id:
                used_sample_ids.add(sample_id)
            matched_unused += 1

    uncovered = max(0, gt_count - matched_unused)
    return gt_count, matched_unused, uncovered


def _dense_continuous_answer_reward(pred_answer: int, gt_answer: int) -> Optional[float]:
    """Capped Gaussian partial credit for dense counting (Phase2, training only)."""
    enable = str(os.environ.get("TRAJ_DENSE_CONTINUOUS_REWARD", "0")).lower() in ("1", "true", "yes")
    if not enable:
        return None
    min_gt = int(os.environ.get("TRAJ_DENSE_CONTINUOUS_MIN_GT", "11"))
    if gt_answer < min_gt:
        return None
    if pred_answer == gt_answer:
        return 1.0
    err = abs(int(pred_answer) - int(gt_answer))
    rel_denom = max(0.06 * float(gt_answer), 1.0)
    norm_err = err / rel_denom
    raw = math.exp(-0.5 * norm_err * norm_err)
    wrong_cap = float(os.environ.get("TRAJ_DENSE_CONTINUOUS_WRONG_CAP", "0.35"))
    within1_cap = float(os.environ.get("TRAJ_DENSE_CONTINUOUS_WITHIN1_CAP", "0.45"))
    score = min(wrong_cap, raw)
    if err <= 1:
        score = min(within1_cap, score + 0.1)
    return clamp_reward(score)


def _has_explicit_stop_none(predict: str) -> bool:
    """Detect explicit no-more-points stop token."""
    if re.search(r"<none\s*/?>", predict, re.IGNORECASE):
        return True
    if re.search(r"no[-\s]?more[-\s]?points", predict, re.IGNORECASE):
        return True
    return False


def is_pixel_coordinate(coord: List[float], img_width: int = 1024, img_height: int = 1024) -> bool:
    """判断是否为像素坐标"""
    return coord[0] > 1.0 or coord[1] > 1.0


# =============================================================================
# Format Reward
# =============================================================================

def format_reward(predict: str, ground_truth: str) -> float:
    """格式正确奖励"""
    try:
        gt_data = parse_ground_truth(ground_truth)
        task_type = gt_data.get("type", "unknown")

        pattern_answer = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
        pattern_point = re.compile(r"<point>.*?</point>", re.DOTALL)
        
        if task_type == "count" and pattern_answer.search(predict):
            return 1.0
        elif task_type == "point" and pattern_point.search(predict):
            return 1.0
            
        return 0.0
    except Exception:
        return 0.0


# =============================================================================
# Accuracy Reward - 非对称版本
# =============================================================================

def accuracy_reward(predict: str, ground_truth: str) -> float:
    """
    计数准确度奖励 - 非对称版本
    
    设计理念：
        - 多数 (pred > gt)：模型产生幻觉 → 重罚 (alpha * 2)
        - 漏数 (pred < gt)：模型遗漏 → 保持原 alpha
    """
    try:
        content_match = re.search(r"<answer>(.*?)</answer>", predict, re.DOTALL)
        if not content_match:
            return 0.0
        
        pred_text = content_match.group(1).strip()
        pred_num = parse_number(pred_text)
        gt_num = parse_number(ground_truth)
        
        if pred_num is None or gt_num is None:
            return 0.0
        
        if pred_num == gt_num:
            return 1.0
        
        error = pred_num - gt_num
        abs_error = abs(error)
        
        denominator = max(gt_num, 1)
        normalized_error = abs_error / denominator
        normalized_error = min(normalized_error, 5.0)
        
        # 非对称 alpha
        alpha_base = 5.0 if gt_num <= 10 else 2.5
        alpha = alpha_base * 2.0 if error > 0 else alpha_base
        
        reward = safe_exp(-alpha * normalized_error)
        return clamp_reward(reward)
    
    except Exception as e:
        logger.warning(f"accuracy_reward exception: {e}")
        return 0.0


# =============================================================================
# Point Reward - 距离计算（回退方案）
# =============================================================================

def compute_l1_distance(pred_point: tuple, gt_point: list, img_width: int, img_height: int) -> float:
    """计算归一化 L1 距离（与 StepCount_asymmetric 保持一致）"""
    px, py = pred_point
    gx, gy = gt_point[0], gt_point[1]
    l1_distance = abs(px - gx) + abs(py - gy)
    max_distance = max(img_width + img_height, 1)
    return min(l1_distance / max_distance, 1.0)


def check_coordinates_in_bounds(coords: list, img_width: int, img_height: int) -> float:
    """检查坐标越界惩罚"""
    penalty = 0.0
    for x, y in coords:
        if x < 0 or x > img_width or y < 0 or y > img_height:
            penalty += 0.7
    return min(penalty, 0.7)


def point_reward(predict: str, gt_point: list, 
                 img_width: int = None, img_height: int = None) -> float:
    """坐标点奖励（距离计算，与 StepCount_asymmetric 保持一致）"""
    try:
        content_match = re.search(r"<point>(.*?)</point>", predict, re.DOTALL)
        if not content_match:
            return 0.0
        
        pred_text = content_match.group(1).strip()
        pred_coords = parse_coordinates_from_text(pred_text)
        
        if pred_coords is None or len(pred_coords) == 0:
            return 0.0
        
        pred_point = pred_coords[0]
        
        if img_width is None or img_height is None:
            img_width, img_height = 1024, 1024

        # 若坐标为归一化 (0-1)，则转换为像素坐标
        if 0.0 <= pred_point[0] <= 1.0 and 0.0 <= pred_point[1] <= 1.0:
            pred_point = (pred_point[0] * img_width, pred_point[1] * img_height)
        
        normalized_distance = compute_l1_distance(pred_point, gt_point, img_width, img_height)
        normalized_distance = min(max(normalized_distance, 0.0), 1.0)
        
        out_of_bounds_penalty = check_coordinates_in_bounds([pred_point], img_width, img_height)
        
        # 参数设置（与 StepCount_asymmetric 一致）
        tolerance = 0.05      # 5% 容忍区 V3 0.05 - V4 0.05 - v5 0.02
        failure_threshold = 0.02  # V3 0.10 - V4 0.02 - v5 0.002
        alpha = 20.0
        alpha_penalty = 50.0  # 降低惩罚强度，防止梯度消失
        
        if normalized_distance <= tolerance:
            main_reward = 1.0
        elif normalized_distance > failure_threshold:
            score_at_threshold = safe_exp(-alpha * (failure_threshold - tolerance))
            main_reward = score_at_threshold * safe_exp(-alpha_penalty * (normalized_distance - failure_threshold))
        else:
            main_reward = safe_exp(-alpha * (normalized_distance - tolerance))

        reward = main_reward - out_of_bounds_penalty
        return clamp_reward(reward)
    
    except Exception as e:
        logger.warning(f"point_reward exception: {e}")
        return 0.0


def mask_point_reward(predict: str, gt_point: list, gt_label: str = None,
                      img_width: int = None, img_height: int = None,
                      problem_text: str = None, used_sample_ids: set = None,
                      image_path: str = None) -> Optional[float]:
    """优先使用 mask 判断命中，失败则返回 None 触发回退"""
    helper = _get_mask_helper()
    if helper is None:
        _log_mask_fallback("mask helper unavailable or metadata missing")
        _mask_debug_tick(mask_helper_unavailable=1)
        return None

    # 允许调用方不传 used_sample_ids（例如 eval 单样本场景）；此时本地追踪仅用于本次调用。
    if used_sample_ids is None:
        used_sample_ids = set()

    parsed = _parse_pred_point(predict)
    if parsed is None:
        _mask_debug_tick(mask_used=1, mask_miss=1)
        return 0.0

    pred_point, pred_label = parsed
    label = gt_label or pred_label

    if img_width is None or img_height is None:
        img_width, img_height = 1024, 1024

    # GT 点转为像素坐标用于匹配 metadata
    gt_point_px = [gt_point[0], gt_point[1]]
    if 0.0 <= gt_point_px[0] <= 1.0 and 0.0 <= gt_point_px[1] <= 1.0:
        gt_point_px = [gt_point_px[0] * img_width, gt_point_px[1] * img_height]

    # 1️⃣ 从 GT 坐标查找对应的 mask（支持label消歧）
    mask_meta = helper.find_mask_by_coord(gt_point_px, label=label, problem_text=problem_text)
    if not mask_meta:
        _log_mask_fallback("gt point not found in masks metadata")
        _mask_debug_tick(mask_gt_not_found=1)
        return None

    sample_id = mask_meta.get("id")
    if not sample_id:
        _log_mask_fallback("mask metadata missing id")
        return None

    # 2️⃣ 从 GT mask 提取 sequence_id 和序列的所有样本
    sequence_id, sequence_samples = helper.get_sequence_from_mask(mask_meta)
    if not sequence_samples:
        _log_mask_fallback("no sequence samples found")
        _mask_debug_tick(mask_no_sequence=1)
        return None

    # Eval 场景：根据 turn 号预填充已使用的 sample_ids，避免单样本评测把历史 mask 也算命中。
    if os.environ.get("STEPCOUNT_MASK_PREFILL_BY_TURN", "0") == "1":
        try:
            ref_path = image_path or mask_meta.get("image_path", "")
            current_turn = extract_turn_number(ref_path) if ref_path else None
            # Fallback: 如果 image_path 是 base 图（无 turn 标记 -> None/0），
            # 尝试从 GT mask 的 metadata 中获取 turn_number 作为 fallback。
            if current_turn is None or current_turn == 0:
                gt_turn = mask_meta.get("turn_number")
                if gt_turn is not None and gt_turn > 0:
                    current_turn = gt_turn
            if current_turn is not None and current_turn > 0:
                for info in sequence_samples:
                    info_turn = info.get("turn_number", -1)
                    if info_turn >= 0 and info_turn < current_turn:
                        sid = info.get("sample_id")
                        if sid:
                            used_sample_ids.add(sid)
        except Exception as e:
            logger.warning(f"mask prefill by turn failed: {e}")
    only_current_turn = os.environ.get("STEPCOUNT_MASK_ONLY_CURRENT_TURN", "0") == "1"
    current_turn_for_filter = None
    if only_current_turn:
        try:
            ref_path = image_path or mask_meta.get("image_path", "")
            current_turn_for_filter = extract_turn_number(ref_path) if ref_path else None
        except Exception:
            current_turn_for_filter = None

    # 3️⃣ 预测点转换为像素坐标
    img_w, img_h = helper.get_image_size(sample_id)
    if not img_w or not img_h:
        img_w, img_h = img_width, img_height

    pred_px = list(pred_point)
    if 0.0 <= pred_px[0] <= 1.0 and 0.0 <= pred_px[1] <= 1.0:
        pred_px = [pred_px[0] * img_w, pred_px[1] * img_h]

    # 供后续距离 shaping 使用的归一化坐标（尽量与不同 mask 分辨率无关）
    try:
        if 0.0 <= pred_point[0] <= 1.0 and 0.0 <= pred_point[1] <= 1.0:
            pred_norm = (float(pred_point[0]), float(pred_point[1]))
        else:
            pred_norm = (
                float(pred_px[0]) / max(float(img_w), 1.0),
                float(pred_px[1]) / max(float(img_h), 1.0),
            )
    except Exception:
        pred_norm = (
            float(pred_px[0]) / max(float(img_w), 1.0),
            float(pred_px[1]) / max(float(img_h), 1.0),
        )

    # 4️⃣ 在序列的所有 mask 中检查预测点
    loaded_any_mask = False
    # miss 时用于“最近未使用 mask 距离衰减”的候选（只收集未使用的 mask）
    eligible_masks: List[Tuple[Dict, np.ndarray]] = []
    for sample_info in sequence_samples:
        seq_sample_id = sample_info["sample_id"]
        if only_current_turn and current_turn_for_filter is not None:
            if sample_info.get("turn_number") != current_turn_for_filter:
                continue
        seq_masks = helper.get_masks(seq_sample_id)
        
        if seq_masks is None:
            continue

        loaded_any_mask = True

        if seq_sample_id not in used_sample_ids:
            eligible_masks.append((sample_info, seq_masks))

        # 获取mask尺寸并调整预测点坐标
        h, w = seq_masks.shape[1], seq_masks.shape[2]
        if img_w != w or img_h != h:
            adjusted_px = [pred_px[0] * w / img_w, pred_px[1] * h / img_h]
        else:
            adjusted_px = pred_px

        px = int(max(0, min(adjusted_px[0], w - 1)))
        py = int(max(0, min(adjusted_px[1], h - 1)))

        # 检查是否在这个mask内
        if bool(seq_masks[0][py, px]):
            # 5️⃣ 判断是否重复
            if seq_sample_id in used_sample_ids:
                # ⚠️ 重复点击已使用的mask
                _mask_debug_tick(mask_used=1, mask_hit_duplicate=1)
                return 0.0  # duplicate_penalty
            else:
                # ✅ 命中未使用的mask
                used_sample_ids.add(seq_sample_id)
                _mask_debug_tick(mask_used=1, mask_hit_unused=1)
                return 1.0

    # ❌ 不在序列的任何mask内（幻觉/miss）
    if not loaded_any_mask:
        _log_mask_fallback("no masks could be loaded for this sequence")
        _mask_debug_tick(mask_no_masks_loaded=1)
        return None
    # ❌ 不在任何可用 mask 内：用“最近未使用 mask 的中心距离”做 shaping（类似 StepCount_asymmetric 的指数衰减）
    # 注意：未使用 mask 的定义与命中逻辑一致（当前 mask + 未被 used_sample_ids 标记的 trajectory masks）。
    if not eligible_masks:
        _mask_debug_tick(mask_used=1, mask_miss=1)
        return 0.0

    best_outside_px = None
    best_w_h = None
    for sample_info, seq_masks in eligible_masks:
        try:
            h, w = int(seq_masks.shape[1]), int(seq_masks.shape[2])
            if h <= 0 or w <= 0:
                continue

            meta = sample_info.get("metadata") or {}
            points = meta.get("points_pixel", [])
            if not points:
                continue
            if isinstance(points[0], list):
                cx, cy = float(points[0][0]), float(points[0][1])
            else:
                cx, cy = float(points[0]), float(points[1])

            cand_img_w, cand_img_h = helper.get_image_size(sample_info.get("sample_id"))
            if not cand_img_w or not cand_img_h:
                cand_img_w, cand_img_h = w, h

            cx_norm = cx / max(float(cand_img_w), 1.0)
            cy_norm = cy / max(float(cand_img_h), 1.0)
            center_x = cx_norm * float(w)
            center_y = cy_norm * float(h)

            pred_x = float(pred_norm[0]) * float(w)
            pred_y = float(pred_norm[1]) * float(h)

            dist_l1_px = abs(pred_x - center_x) + abs(pred_y - center_y)

            # 等效半径：按 mask 面积估计（等面积圆），代表“mask 范围”
            area = float(np.count_nonzero(seq_masks[0]))
            radius_px = math.sqrt(max(area, 0.0) / math.pi) if area > 0 else 0.0

            outside_px = max(0.0, dist_l1_px - radius_px)
            if best_outside_px is None or outside_px < best_outside_px:
                best_outside_px = outside_px
                best_w_h = (w, h, pred_x, pred_y)
        except Exception:
            continue

    if best_outside_px is None or best_w_h is None:
        _mask_debug_tick(mask_used=1, mask_miss=1)
        return 0.0

    w, h, pred_x, pred_y = best_w_h
    normalized_outside = float(best_outside_px) / max(float(w + h), 1.0)
    out_of_bounds_penalty = check_coordinates_in_bounds([(pred_x, pred_y)], w, h)

    tolerance = 0.0
    failure_threshold = 0.02 # V3 0.10 - V4 0.02 - v5 0.002; 危险范围让模型学会尽量靠近 mask 边界，而不是远远落在外面；同时也避免过度惩罚轻微偏离的预测。
    alpha = 20.0 # V4: 10.0 - 20.0; 距离衰减的陡峭度，让模型更倾向于预测在 mask 边界附近，而不是远远落在外面。
    alpha_penalty = 50.0

    if normalized_outside <= tolerance:
        main_reward = 1.0
    elif normalized_outside > failure_threshold:
        score_at_threshold = safe_exp(-alpha * (failure_threshold - tolerance))
        main_reward = score_at_threshold * safe_exp(-alpha_penalty * (normalized_outside - failure_threshold))
    else:
        main_reward = safe_exp(-alpha * (normalized_outside - tolerance))

    reward = main_reward - out_of_bounds_penalty
    _mask_debug_tick(mask_used=1, mask_miss=1)
    return clamp_reward(reward)



# =============================================================================
# Main Score Function - 兼容 verl 框架
# =============================================================================

def compute_score(
    predict: str, 
    ground_truth: str,
    image_path: str = None,
    images: List[Any] = None,
    format_weight: float = 0.2,
    content_weight: float = 0.8,
    answer_weight: float = 0.6,
    point_weight: float = 0.2,
    trajectory_format_weight: float = 0.2,
    max_turns: int = 11,
    adaptive_max_turns: bool = False,
    adaptive_max_turns_margin: int = 2,
    problem: str = None,
    used_sample_ids: set = None,
    sample_id: Any = None,
    sample_index: int = -1,
    is_eval: bool = False,
) -> Dict[str, float]:
    """
    计算综合得分 - 兼容 verl 训练框架
    
    Args:
        predict: 模型预测文本
        ground_truth: GT 答案（JSON 或纯数字）
        image_path: 图片路径（可选）
        images: PIL Image 列表（可选，verl 框架传入）
        format_weight: 格式分数权重
        content_weight: 内容分数权重
    
    Returns:
        {
            "overall": float,  # 总分（必需）
            "format": float,   # 格式分数
            "content": float,  # 内容分数
            "point": float,    # 点任务分数（如适用）
            "accuracy": float, # 计数任务分数（如适用）
        }
    """
    default_score = {
        "overall": 0.0,
        "format": 0.0,
        "content": 0.0,
        "answer": 0.0,
        "point": 0.0,
        "is_point_task": 0.0,
        "is_count_task": 0.0,
        "is_trajectory_task": 0.0,
        "consistency_violation": 0.0,
        "answer_gated": 0.0,
        "format_fail": 0.0,
        "no_point_pred": 0.0,
        "turns_exceeded": 0.0,
        "stopped_by_answer": 0.0,
        "stop_violation": 1.0,
        "tag_balance_violation": 0.0,
        "point_count_number_score": 1.0,
        "point_count_number_violation": 0.0,
        "point_count_number_missing": 0.0,
        "point_count_number_mismatch": 0.0,
        "integrity_rejected": 0.0,
    }
    
    try:
        # IMPORTANT: trajectory dense mask reward requires a sequence identifier.
        # `sample_id` in verl is often a numeric row id, which is NOT usable as an image path.
        # Prefer recovering from `images` payload (may be a sequence id string) when `image_path` is missing.
        if not image_path:
            inferred = _infer_image_path_from_images(images)
            if inferred:
                image_path = inferred
            # `sample_id` in verl is often a numeric row id. A pure-digit string is almost
            # certainly NOT a usable image path / sequence id.
            elif isinstance(sample_id, str) and sample_id and (not sample_id.strip().isdigit()):
                image_path = sample_id

        # 获取图片尺寸
        img_width, img_height = 1024, 1024
        if images and len(images) > 0:
            try:
                if hasattr(images[0], 'size'):
                    img_width, img_height = images[0].size
            except Exception:
                pass
        elif image_path:
            try:
                with PILImage.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception:
                pass

        # 解析 GT
        gt_data = parse_ground_truth(ground_truth)
        task_type = gt_data.get("type", "unknown")

        # Task-type indicators (helps debug validation composition).
        is_point_task = 1.0 if task_type == "point" else 0.0
        is_count_task = 1.0 if task_type == "count" else 0.0
        is_trajectory_task = 1.0 if task_type == "trajectory" else 0.0

        content_score = 0.0
        score = {}

        if task_type == "trajectory":
            global _TRAJ_REASON_DEBUG_CALLS
            answer_gate_point_threshold = float(os.environ.get("TRAJ_ANSWER_GATE_THRESHOLD", "0.4"))
            expected_steps = len(gt_data.get("point_sequence", []))
            if expected_steps <= 0:
                expected_steps = parse_number(str(gt_data.get("count_number", 0))) or 0
            pred_points = _parse_pred_points(predict)
            stopped_by_answer = _has_answer_tag(predict)
            effective_max_turns = compute_adaptive_max_turns_from_gt(
                gt_data,
                max_turns=max_turns,
                adaptive_max_turns=adaptive_max_turns,
                adaptive_max_turns_margin=adaptive_max_turns_margin,
            )
            # In interleaved rollout, max_turns is the hard cap of turns. If we see too many point tags,
            # treat it as exceeding and zero out reward.
            turns_exceeded = len(pred_points) >= effective_max_turns
            pred_answer = _extract_last_answer_number(predict) if stopped_by_answer else None
            answer_gated = False
            tag_balance_violation = _trajectory_tag_balance_violation(predict)
            (
                point_count_number_step_scores,
                point_count_number_score,
                point_count_number_missing,
                point_count_number_mismatch,
            ) = _point_count_number_step_scores(predict)
            point_count_number_violation = (point_count_number_missing + point_count_number_mismatch) > 0

            format_score = clamp_reward(_trajectory_format_reward(predict, expected_steps))

            # Hoist return_step_scores check to avoid double-calling _details for
            # trajectory tasks without GT point_sequence.
            return_step_scores = str(os.environ.get("TRAJ_RETURN_POINT_STEP_SCORES", "0")).lower() in (
                "1",
                "true",
                "yes",
            )

            # For W&B/verl compatibility we only emit scalar floats.
            # `step_scores` are per-step contributions that (approximately) sum to `point_dense_score`.
            step_scores: List[float] = []
            step_hit_any: List[float] = []
            step_is_duplicate: List[float] = []
            point_target_count: float = 0.0
            point_eval_steps: float = 0.0

            point_dense_score = 0.0
            if not turns_exceeded:
                # Special-case: GT expects zero points (answer==0). We should not force <point>.
                # Give full point score iff the model also predicts zero points; otherwise 0.
                if expected_steps == 0:
                    point_dense_score = 1.0 if len(pred_points) == 0 else 0.0
                else:
                    if gt_data.get("point_sequence"):
                        point_dense_score = _trajectory_point_dense_reward(
                            predict=predict,
                            gt_data=gt_data,
                            img_width=img_width,
                            img_height=img_height,
                            image_path=image_path,
                            problem=problem,
                        )
                    elif return_step_scores:
                        # Compute point_dense_score AND step details in a single _details call,
                        # avoiding the redundant sequence lookup + per-step mask evaluation.
                        try:
                            (
                                point_dense_score,
                                step_scores,
                                step_hit_any,
                                step_is_duplicate,
                                point_target_count,
                                point_eval_steps,
                            ) = _trajectory_point_dense_reward_without_gt_points_details(
                                predict=predict,
                                gt_data=gt_data,
                                image_path=image_path,
                            )
                        except Exception:
                            pass  # point_dense_score stays 0.0, step_scores stays []
                    else:
                        point_dense_score = _trajectory_point_dense_reward_without_gt_points(
                            predict=predict,
                            gt_data=gt_data,
                            image_path=image_path,
                        )

            if return_step_scores and (not turns_exceeded) and expected_steps > 0:
                if gt_data.get("point_sequence"):
                    try:
                        raw_step_scores = _trajectory_point_step_scores(
                            predict=predict,
                            gt_data=gt_data,
                            img_width=img_width,
                            img_height=img_height,
                            image_path=image_path,
                            problem=problem,
                        )
                        denom = float(max(len(gt_data.get("point_sequence", [])), 1))
                        step_scores = [clamp_reward(float(s) / denom) for s in raw_step_scores]
                        point_target_count = float(len(gt_data.get("point_sequence", [])))
                        point_eval_steps = float(len(step_scores))
                    except Exception:
                        step_scores = []
                # else: already computed above in the unified _details call

            count_number_point_penalty_weight = clamp_reward(
                float(os.environ.get("TRAJ_COUNT_NUMBER_POINT_PENALTY_WEIGHT", "0.20"))
            )
            if expected_steps > 0 and count_number_point_penalty_weight > 0 and point_count_number_step_scores:
                count_factor = 1.0 - count_number_point_penalty_weight * (1.0 - point_count_number_score)
                point_dense_score = clamp_reward(point_dense_score * count_factor)
                if step_scores:
                    adjusted_step_scores: List[float] = []
                    for step_idx, step_score in enumerate(step_scores):
                        count_step_score = (
                            point_count_number_step_scores[step_idx]
                            if step_idx < len(point_count_number_step_scores)
                            else 0.0
                        )
                        step_factor = 1.0 - count_number_point_penalty_weight * (1.0 - count_step_score)
                        adjusted_step_scores.append(clamp_reward(float(step_score) * step_factor))
                    step_scores = adjusted_step_scores

            answer_point_count_consistent = True
            if stopped_by_answer and pred_answer is not None:
                answer_point_count_consistent = int(pred_answer) == int(len(pred_points))
            consistency_violation = 0.0 if answer_point_count_consistent else 1.0

            if not answer_point_count_consistent:
                if _traj_event_should_log("consistency_violation"):
                    logger.warning(
                        "[trajectory_reward] sample_index=%s consistency_violation: "
                        "pred_answer=%s point_count=%s -> consistency penalty",
                        sample_index,
                        pred_answer,
                        len(pred_points),
                    )

            strict_trajectory_integrity = str(
                os.environ.get("TRAJ_STRICT_TRAJECTORY_INTEGRITY", "0")
            ).lower() in ("1", "true", "yes")
            strict_answer_point_consistency = str(
                os.environ.get(
                    "TRAJ_STRICT_ANSWER_POINT_CONSISTENCY",
                    "0",
                )
            ).lower() in ("1", "true", "yes")
            hard_reject_tag_balance = str(
                os.environ.get(
                    "TRAJ_HARD_REJECT_TAG_BALANCE",
                    "1" if strict_trajectory_integrity else "0",
                )
            ).lower() in ("1", "true", "yes")
            hard_reject_count_number = str(
                os.environ.get("TRAJ_HARD_REJECT_COUNT_NUMBER", "0")
            ).lower() in ("1", "true", "yes")
            trajectory_integrity_violation = (
                tag_balance_violation
                or point_count_number_violation
                or (not answer_point_count_consistent)
            )
            trajectory_hard_reject = (
                (hard_reject_tag_balance and tag_balance_violation)
                or (hard_reject_count_number and point_count_number_violation)
                or (strict_answer_point_consistency and (not answer_point_count_consistent))
            )

            # ============================================================
            # Two-Level Reward Architecture (v5)
            # Level 1: answer_score  -> drives advantage ranking across K trajectories
            # Level 2: point_dense_score -> dense shaping signal within trajectory
            # ============================================================

            # ==============================================================
            # PATH A: Validation / Evaluation  (is_eval = True)
            # Pure binary answer correctness – NO gating, NO penalty, NO weights
            # This guarantees val metric == independent eval metric.
            # ==============================================================
            if is_eval:
                eval_require_integrity = str(
                    os.environ.get(
                        "TRAJ_EVAL_REQUIRE_INTEGRITY",
                        "0",
                    )
                ).lower() in ("1", "true", "yes")
                answer_score = 0.0
                if (
                    stopped_by_answer
                    and pred_answer is not None
                    and not turns_exceeded
                    and (not eval_require_integrity or not trajectory_integrity_violation)
                    and not trajectory_hard_reject
                ):
                    gt_answer = gt_data.get("count_number")
                    if gt_answer is not None and int(pred_answer) == int(gt_answer):
                        answer_score = 1.0
                overall_score = clamp_reward(answer_score)
                score["overall"] = overall_score
                score["format"] = format_score
                score["content"] = answer_score
                score["answer"] = answer_score
                score["point"] = point_dense_score  # for diagnostics only
                score["answer_for_ranking"] = answer_score

            # ==============================================================
            # PATH B: Training  (is_eval = False)
            # Soft gating + weighted combination + format rejection
            # ==============================================================
            else:
                # Determine if this sample lacks mask data → fallback to answer-only
                _eval_answer_only_env = str(os.environ.get("TRAJ_EVAL_ANSWER_ONLY_ON_NO_MASK", "1")).lower() in ("1", "true")
                eval_answer_only_on_no_mask = _eval_answer_only_env and (expected_steps == 0) and (not gt_data.get("point_sequence"))

                answer_score = 0.0
                soft_gate_mode = str(os.environ.get("TRAJ_ANSWER_GATE_MODE", "soft")).lower()

                if eval_answer_only_on_no_mask:
                    # No mask sequence available → answer-only (binary)
                    if stopped_by_answer and pred_answer is not None:
                        gt_answer = gt_data.get("count_number")
                        if gt_answer is not None and int(pred_answer) == int(gt_answer):
                            answer_score = 1.0

                elif stopped_by_answer and not turns_exceeded:
                    gt_answer = gt_data.get("count_number")
                    if pred_answer is not None and gt_answer is not None and int(pred_answer) == int(gt_answer):
                        if soft_gate_mode == "off":
                            # No gating: pure binary answer correctness
                            answer_score = 1.0
                        elif soft_gate_mode == "soft":
                            # Soft gating: answer = base + bonus * point_quality
                            # Correct answer always gets ≥ base reward; better pointing adds bonus
                            soft_base = float(os.environ.get("TRAJ_SOFT_GATE_BASE", "0.8"))
                            soft_bonus = 1.0 - soft_base  # 0.3 by default
                            answer_score = soft_base + soft_bonus * clamp_reward(point_dense_score)
                        else:
                            # Legacy hard gating (soft_gate_mode == "hard")
                            gate_required = expected_steps > 0
                            gate_ok = (not gate_required) or (point_dense_score >= answer_gate_point_threshold)
                            if gate_ok:
                                answer_score = 1.0
                            else:
                                answer_gated = True
                                if _traj_event_should_log("answer_gated"):
                                    logger.info(
                                        "[trajectory_reward] sample_index=%s answer_gated: point_score=%.4f < %.2f",
                                        sample_index, point_dense_score, answer_gate_point_threshold,
                                    )
                    else:
                        # Wrong answer — asymmetric exponential decay partial reward (training only)
                        # Mirrors accuracy_reward(): exp(-alpha * |pred-gt|/max(gt,1))
                        # 多数 (pred > gt): hallucination → heavier penalty (alpha * 2)
                        # 漏数 (pred < gt): miss → base alpha
                        _dense_partial = None
                        if pred_answer is not None and gt_answer is not None:
                            _dense_partial = _dense_continuous_answer_reward(int(pred_answer), int(gt_answer))
                        if _dense_partial is not None:
                            answer_score = _dense_partial
                        else:
                            _soft_answer_decay = str(os.environ.get("TRAJ_SOFT_ANSWER_DECAY", "0")).lower() in ("1", "true", "yes")
                            if _soft_answer_decay and pred_answer is not None and gt_answer is not None:
                                _pred_int = int(pred_answer)
                                _gt_int = int(gt_answer)
                                _error = _pred_int - _gt_int
                                _abs_error = abs(_error)
                                _denom = max(_gt_int, 1)
                                _norm_err = min(_abs_error / _denom, 5.0)
                                # Asymmetric alpha: overcount penalized 2x harder (hallucination is worse)
                                _alpha_base = float(os.environ.get("TRAJ_ANSWER_DECAY_ALPHA", "8.0"))
                                # Asymmetric alpha + GT-scaled undercounting penalty
                                if _error > 0:
                                    _alpha = _alpha_base * 2.0  # overcount: 2x harder
                                else:
                                    _under_gt_scale = float(os.environ.get("TRAJ_UNDER_ALPHA_GT_SCALE", "0.5"))
                                    _under_gt_th = int(float(os.environ.get("TRAJ_UNDER_ALPHA_GT_THRESHOLD", "5")))
                                    if _gt_int > _under_gt_th and _under_gt_scale > 0:
                                        _alpha = _alpha_base * (1.0 + _under_gt_scale * (_gt_int - _under_gt_th) / max(_under_gt_th, 1))
                                    else:
                                        _alpha = _alpha_base
                                _raw_decay = safe_exp(-_alpha * _norm_err)
                                # Cap: wrong answer reward never exceeds this ceiling (prevents close-enough local optimum)
                                _decay_cap = float(os.environ.get("TRAJ_ANSWER_DECAY_CAP", "0.4"))
                                answer_score = clamp_reward(min(_raw_decay, _decay_cap))
                                if _traj_event_should_log("answer_decay"):
                                    logger.info(
                                        "[trajectory_reward] sample_index=%s answer_decay: pred=%d gt=%d error=%+d alpha=%.1f raw=%.4f cap=%.2f score=%.4f",
                                        sample_index, _pred_int, _gt_int, _error, _alpha, _raw_decay, _decay_cap, answer_score,
                                    )


                    # Consistency penalty: pred_answer != len(pred_points)
                    if not answer_point_count_consistent:
                        if strict_answer_point_consistency:
                            answer_score = 0.0
                        else:
                            cv_penalty = float(os.environ.get("TRAJ_CONSISTENCY_PENALTY", "0.5"))
                            answer_score = answer_score * cv_penalty

                    # Explicit <none> / no-more-points stop bonus (Phase3 stop mechanism)
                    _stop_none_enable = str(os.environ.get("TRAJ_STOP_NONE_ENABLE", "0")).lower() in ("1", "true", "yes")
                    if _stop_none_enable and _has_explicit_stop_none(predict):
                        _stop_bonus = float(os.environ.get("TRAJ_STOP_NONE_BONUS", "0.05"))
                        answer_score = clamp_reward(answer_score + _stop_bonus)
                        score["stop_none_bonus"] = _stop_bonus
                    elif _stop_none_enable and stopped_by_answer and pred_answer is not None and gt_answer is not None:
                        if int(pred_answer) < int(gt_answer) and len(pred_points) < int(gt_answer):
                            _early_stop_pen = float(os.environ.get("TRAJ_EARLY_STOP_PENALTY", "0.15"))
                            answer_score = max(0.0, answer_score - _early_stop_pen)
                            score["early_stop_penalty"] = _early_stop_pen

                # --- Coverage Penalty (Phase 1B, V34: mask-based uncovered count) ---
                # Penalises early stopping when the model gives <answer> but GT still
                # has uncovered masks (gt_count - matched_unused), not merely fewer
                # predicted steps than expected_steps.
                _coverage_penalty_enable = str(
                    os.environ.get("TRAJ_COVERAGE_PENALTY_ENABLE", "0")
                ).lower() in ("1", "true")
                if _coverage_penalty_enable and stopped_by_answer and not turns_exceeded:
                    _cov_mode = str(os.environ.get("TRAJ_COVERAGE_PENALTY_MODE", "mask")).lower()
                    if _cov_mode == "mask":
                        _gt_cov, _matched_cov, _num_uncovered = _trajectory_coverage_stats(
                            predict=predict,
                            gt_data=gt_data,
                            image_path=image_path,
                        )
                        score["coverage_gt_count"] = float(_gt_cov)
                        score["coverage_matched"] = float(_matched_cov)
                    else:
                        _num_uncovered = max(0, expected_steps - len(pred_points))
                    if _num_uncovered > 0:
                        _penalty_per_miss = float(
                            os.environ.get("TRAJ_COVERAGE_PENALTY_PER_MISS", "0.3")
                        )
                        _penalty_cap = float(
                            os.environ.get("TRAJ_COVERAGE_PENALTY_CAP", "0.6")
                        )
                        _coverage_penalty = min(
                            _num_uncovered * _penalty_per_miss, _penalty_cap
                        )
                        answer_score = max(0.0, answer_score - _coverage_penalty)
                        score["coverage_penalty"] = _coverage_penalty
                        score["coverage_uncovered"] = float(_num_uncovered)

                if trajectory_hard_reject:
                    answer_score = 0.0
                    point_dense_score = 0.0
                    format_score = 0.0
                    score["integrity_rejected"] = 1.0

                # --- Weighted overall score ---
                if eval_answer_only_on_no_mask:
                    overall_score = clamp_reward(answer_score)
                else:
                    overall_score = clamp_reward(
                        answer_weight * answer_score
                        + point_weight * point_dense_score
                        + trajectory_format_weight * format_score
                    )

                # --- Format Rejection: zero reward for format-broken trajectories ---
                # Modes: "1"/"true"/"yes" = legacy (hard-zero whenever format_score<=0);
                #        "structural"      = hard-zero ONLY on genuine structural garbage
                #                            (no <answer>, unparseable answer, or zero <point>
                #                            tags when points expected). JSON-key drift that
                #                            still has <point> tags is left to graded format,
                #                            avoiding the format=0 -> hard-zero -> dead-gradient
                #                            absorbing collapse on long dense chains.
                _fmt_rej_raw = str(os.environ.get("TRAJ_FORMAT_REJECTION", "0")).lower()
                format_rejection = _fmt_rej_raw in ("1", "true", "yes", "structural")
                if format_rejection and format_score <= 0.0:
                    if _fmt_rej_raw == "structural":
                        _has_point_tag = re.search(r"<point>", predict) is not None
                        _structural_fail = (
                            (not _has_answer_tag(predict))
                            or (_extract_last_answer_number(predict) is None)
                            or (expected_steps > 0 and not _has_point_tag)
                        )
                        if _structural_fail:
                            overall_score = 0.0
                            score["format_rejected"] = 1.0
                    else:
                        overall_score = 0.0
                        score["format_rejected"] = 1.0

                # --- Invalid trajectories: no answer or max-turns exceeded ---
                if (not stopped_by_answer) or (pred_answer is None):
                    overall_score = 0.0
                    answer_score = 0.0
                    point_dense_score = 0.0
                if turns_exceeded:
                    overall_score = 0.0
                    answer_score = 0.0
                    point_dense_score = 0.0

                score["overall"] = overall_score
                score["format"] = format_score
                score["content"] = answer_score
                score["answer"] = answer_score
                score["point"] = point_dense_score
                score["answer_for_ranking"] = answer_score


            score["is_point_task"] = is_point_task
            score["is_count_task"] = is_count_task
            score["is_trajectory_task"] = is_trajectory_task

            if return_step_scores:
                # Keep as scalar floats for verl/W&B compatibility.
                score["point_steps"] = float(len(step_scores))
                if point_target_count > 0:
                    score["point_target_count"] = float(point_target_count)
                if point_eval_steps > 0:
                    score["point_eval_steps"] = float(point_eval_steps)

                for i, s in enumerate(step_scores[:effective_max_turns]):
                    score[f"point_step_{i+1}"] = float(s)

                # Extra diagnostics (mostly meaningful for mask-derived scoring).
                for i, v in enumerate(step_hit_any[:effective_max_turns]):
                    score[f"point_step_{i+1}_hit_any"] = float(v)
                for i, v in enumerate(step_is_duplicate[:effective_max_turns]):
                    score[f"point_step_{i+1}_duplicate"] = float(v)
                for i, v in enumerate(point_count_number_step_scores[:effective_max_turns]):
                    score[f"point_step_{i+1}_count_number"] = float(v)
            score["effective_max_turns"] = float(effective_max_turns)
            score["consistency_violation"] = consistency_violation
            score["answer_gated"] = 1.0 if answer_gated else 0.0
            score["integrity_rejected"] = 1.0 if trajectory_hard_reject else 0.0
            score["format_fail"] = 1.0 if format_score <= 0.0 else 0.0
            # Observability: count all known point-key drift aliases, including recoverable
            # aliases normalized by _normalize_point_keys and broken aliases seen before OOM.
            # `point_key_typo` is the per-sample alias count; `point_key_typo_rate`
            # is a batch-meanable sample-level drift rate.
            point_key_alias_count = _count_point_key_aliases(predict)
            score["point_key_typo"] = float(point_key_alias_count)
            score["point_key_typo_rate"] = 1.0 if point_key_alias_count > 0 else 0.0
            score["no_point_pred"] = 1.0 if len(pred_points) == 0 else 0.0
            score["turns_exceeded"] = 1.0 if turns_exceeded else 0.0
            score["stopped_by_answer"] = 1.0 if stopped_by_answer else 0.0
            score["tag_balance_violation"] = 1.0 if tag_balance_violation else 0.0
            score["point_count_number_score"] = float(point_count_number_score)
            score["point_count_number_violation"] = 1.0 if point_count_number_violation else 0.0
            score["point_count_number_missing"] = float(point_count_number_missing)
            score["point_count_number_mismatch"] = float(point_count_number_mismatch)
            score["stop_violation"] = (
                1.0
                if (
                    turns_exceeded
                    or (not stopped_by_answer)
                    or (pred_answer is None)
                    or trajectory_hard_reject
                )
                else 0.0
            )

            # --- Per-turn process rewards for step-level GRPO ---
            # _step_rewards: list of per-turn reward scalars to be placed at
            # </point> token positions.  The reward worker distributes these.
            # Enabled by PROCESS_REWARD_ENABLE=1.
            if step_scores and str(os.environ.get("PROCESS_REWARD_ENABLE", "0")).lower() in ("1", "true", "yes"):
                # Each step gets: point_weight * step_score_k
                _per_turn_rewards = [float(point_weight) * float(s) for s in step_scores]
                # Answer portion (placed at last token by reward worker):
                # = overall - sum(step_rewards)
                # This naturally includes answer_weight*answer + format_weight*format
                # + any rounding difference
                score["_step_rewards"] = _per_turn_rewards

                # --- Unified per-turn step VALUE for advantage credit (B/C arms) ---
                # BOK_STEP_SIGNAL: pointhit (default; no override -> estimator uses
                # token_level_rewards = legacy) | progress | stoptiming.
                # We emit a parallel `_point_step_value` list aligned 1:1 with
                # `_step_rewards` (i.e. with the </point> turn boundaries). The reward
                # worker records these at the same token positions; the step estimator
                # consumes them instead of the placed token reward when present.
                _bok_step_signal = os.environ.get("BOK_STEP_SIGNAL", "pointhit").strip().lower()
                if _bok_step_signal in ("progress", "stoptiming"):
                    _n_steps = len(_per_turn_rewards)
                    _step_value: List[float] = []
                    if _bok_step_signal == "progress":
                        # Dedup-aware coverage delta: +1 new GT, -dup repeat, 0 miss.
                        # FAIL-FAST OBSERVABILITY (reviewer P1): if step_hit_any/duplicate
                        # are empty (e.g. data carries point_sequence -> the _details path
                        # is skipped), every progress value collapses to 0.0 and Arm C
                        # SILENTLY degrades to outcome-only (== Arm A). Surface it as a
                        # metric so smoke/train logs catch it instead of a misleading null.
                        if _n_steps > 0 and len(step_hit_any) == 0:
                            score["progress_degraded"] = 1.0
                            if _traj_event_should_log("progress_degraded"):
                                logger.warning(
                                    "[trajectory_reward] sample_index=%s progress_degraded: "
                                    "step_hit_any empty (point_sequence path?) -> Arm C aux=0",
                                    sample_index,
                                )
                        _dup_pen = float(os.environ.get("BOK_PROGRESS_DUP_PENALTY", "1.0"))
                        for _k in range(_n_steps):
                            _hit = step_hit_any[_k] if _k < len(step_hit_any) else 0.0
                            _dup = step_is_duplicate[_k] if _k < len(step_is_duplicate) else 0.0
                            if _dup > 0.5:
                                _step_value.append(-_dup_pen)
                            elif _hit > 0.5:
                                _step_value.append(1.0)
                            else:
                                _step_value.append(0.0)
                    else:  # stoptiming: sign(N - running_count) on the continue (point) token
                        _N = None
                        try:
                            _N = int(gt_answer) if gt_answer is not None else None
                        except Exception:
                            _N = None
                        if _N is not None and _N > 0:
                            for _k in range(_n_steps):
                                _cnt = _k + 1  # running count at this point-turn (1-indexed)
                                if _cnt < _N:
                                    _step_value.append(1.0)   # keep going (under target)
                                elif _cnt == _N:
                                    _step_value.append(0.0)   # at target (neutral)
                                else:
                                    _step_value.append(-1.0)  # overshoot (suppress extra points)
                    if _step_value:
                        score["_point_step_value"] = _step_value

            if _traj_reason_debug_enabled():
                _TRAJ_REASON_DEBUG_CALLS += 1
                if _TRAJ_REASON_DEBUG_CALLS % _traj_reason_debug_every() == 0:
                    reasons: List[str] = []
                    if not stopped_by_answer:
                        reasons.append("missing_answer_tag")
                    if pred_answer is None:
                        reasons.append("answer_not_parseable")
                    if turns_exceeded:
                        reasons.append("turns_exceeded")
                    if format_score <= 0.0:
                        reasons.append("format_fail")
                    if answer_gated:
                        reasons.append("answer_gated_by_point")
                    if not answer_point_count_consistent:
                        reasons.append("answer_point_count_inconsistent")
                    if tag_balance_violation:
                        reasons.append("tag_balance_violation")
                    if point_count_number_violation:
                        reasons.append("point_count_number_violation")
                    if not reasons:
                        reasons.append("ok")

                    try:
                        predict_tail = str(predict)[-260:].replace("\n", " ")
                    except Exception:
                        predict_tail = ""

                    msg = (
                        "[trajectory_reward][detail] "
                        f"sample_index={sample_index} "
                        f"overall={score['overall']:.4f} format={score['format']:.4f} "
                        f"point={score['point']:.4f} answer={score['answer']:.4f} "
                        f"expected_steps={expected_steps} pred_points={len(pred_points)} "
                        f"pred_answer={pred_answer} gt_answer={gt_data.get('count_number')} "
                        f"reasons={reasons} tail='{predict_tail}'"
                    )
                    logger.info(msg)
                    print(msg)
            return score

        # 非 trajectory 任务保持旧逻辑兼容
        format_score = clamp_reward(format_reward(predict, ground_truth))

        if task_type == "point":
            # Point 任务：优先使用 mask reward，失败则回退距离计算
            mask_score = mask_point_reward(
                predict,
                gt_data["point_2d"],
                gt_label=gt_data.get("label", "object"),
                img_width=img_width,
                img_height=img_height,
                problem_text=problem,
                used_sample_ids=used_sample_ids,
                image_path=image_path
            )
            if mask_score is None:
                _mask_debug_tick(mask_fallback_distance=1)
                pt_score = point_reward(
                    predict,
                    gt_data["point_2d"],
                    img_width=img_width,
                    img_height=img_height
                )
                content_score = clamp_reward(pt_score)
            else:
                content_score = clamp_reward(mask_score)
            score["point"] = content_score
            score["is_point_task"] = is_point_task
            score["is_count_task"] = is_count_task
            score["is_trajectory_task"] = is_trajectory_task
            
        elif task_type == "count":
            # Count 任务：使用非对称 accuracy reward
            acc_score = accuracy_reward(predict, str(gt_data["count_number"]))
            content_score = clamp_reward(acc_score)
            score["accuracy"] = content_score
            score["is_point_task"] = is_point_task
            score["is_count_task"] = is_count_task
            score["is_trajectory_task"] = is_trajectory_task
        
        # 计算总分
        overall_score = clamp_reward(format_weight * format_score + content_weight * content_score)
        
        score["overall"] = overall_score
        score["format"] = format_score
        score["content"] = content_score
        # Ensure consistent key schema across all task types (point/count/unknown)
        # so downstream metric aggregation never encounters KeyError.
        score.setdefault("answer", content_score)
        score.setdefault("point", 0.0)
        score.setdefault("is_point_task", is_point_task)
        score.setdefault("is_count_task", is_count_task)
        score.setdefault("is_trajectory_task", is_trajectory_task)
        score.setdefault("consistency_violation", 0.0)
        score.setdefault("answer_gated", 0.0)
        score.setdefault("format_fail", 1.0 if format_score <= 0.0 else 0.0)
        score.setdefault("no_point_pred", 0.0)
        score.setdefault("turns_exceeded", 0.0)
        score.setdefault("stopped_by_answer", 0.0)
        score.setdefault("stop_violation", 1.0)
        
        return score
        
    except Exception as e:
        logger.error(f"compute_score exception: {e}")
        return default_score


# =============================================================================
# 测试代码
# =============================================================================

# if __name__ == "__main__":
#     print("=" * 70)
#     print("StepCount Mask-Based Reward v3.1 测试")
#     print("=" * 70)
    
#     # 测试基础功能
#     print("\n【基础功能测试】")
#     pred1 = "<think>I see 5</think><answer>5</answer>"
#     pred2 = '<point>{"point_2d": [512, 384], "label": "people"}</point>'
    
#     print(f"format_reward (answer): {format_reward(pred1, '5')}")
#     print(f"format_reward (point): {format_reward(pred2, '{\"point_2d\": [512, 384]}')}")
#     print(f"accuracy_reward (correct): {accuracy_reward(pred1, '5')}")
    
#     # 测试 compute_score
#     print("\n【compute_score 测试】")
    
#     # 测试 count 任务
#     result = compute_score(
#         predict="<think>I counted carefully</think><answer>5</answer>",
#         ground_truth="5"
#     )
#     print(f"Count 任务 (正确): {result}")
    
#     result = compute_score(
#         predict="<think>I counted</think><answer>10</answer>",
#         ground_truth="5"
#     )
#     print(f"Count 任务 (多数): {result}")
    
#     result = compute_score(
#         predict="<think>I counted</think><answer>3</answer>",
#         ground_truth="5"
#     )
#     print(f"Count 任务 (漏数): {result}")
    
#     # 测试 point 任务
#     gt_point = '{"point_2d": [512, 384], "label": "people", "count_number": 1}'
    
#     result = compute_score(
#         predict='<point>{"point_2d": [512, 384], "label": "people"}</point>',
#         ground_truth=gt_point
#     )
#     print(f"Point 任务 (精确): {result}")
    
#     result = compute_score(
#         predict='<point>{"point_2d": [520, 390], "label": "people"}</point>',
#         ground_truth=gt_point
#     )
#     print(f"Point 任务 (接近): {result}")
    
#     result = compute_score(
#         predict='<point>{"point_2d": [100, 100], "label": "people"}</point>',
#         ground_truth=gt_point
#     )
#     print(f"Point 任务 (远离): {result}")
    
#     print("\n【非对称惩罚测试】GT=25")
#     for pred in [15, 20, 25, 30, 35]:
#         result = compute_score(
#             predict=f"<think>test</think><answer>{pred}</answer>",
#             ground_truth="25"
#         )
#         diff = pred - 25
#         label = "多数" if diff > 0 else ("漏数" if diff < 0 else "正确")
#         print(f"  Pred={pred} ({label}): accuracy={result.get('accuracy', 0):.3f}")
