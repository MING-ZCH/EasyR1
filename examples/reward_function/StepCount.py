# reward设计总体思路：
#     - format (权重 0.3)：
#         - Thinking Format: 1.<think></think><point></point>  2.<think></think><answer></answer>
#         - 满足格式得 1.0，否则得 0.0。
#
#     - content (权重 0.7)：
#         - 核心理念：放弃线性的 "Threshold + Clip" 逻辑，改用 **指数衰减 (Exponential Decay)**。
#         - 优势：
#             1. **消除断崖 (No Cliff)**：避免了“差一点点就从满分掉到0分”的梯度截断，提供平滑的优化信号。
#             2. **持续梯度 (Continuous Gradient)**：即使模型预测非常接近真值（如误差 1%），指数函数的尖峰特性仍能提供强烈的梯度，引导模型追求完美。
#             3. **难度对齐**：将 Accuracy 和 Point 任务的难度统一归一化到 [0, 1] 区间。
#
#         - accuracy reward (计数任务)：
#             - 公式：$R_{acc} = 1.0 \times \exp(-\alpha \times \text{error\_rate})$，其中 $\alpha=5.0$。
#             - 即使 Accuracy 数据占比少 (15%)，强梯度也能保证模型不忽略计数准确性。
#             - **自适应 Alpha**：
#                 - 小数(<=10): $\alpha=5.0$ (严格要求，误差1即大错)
#                 - 大数(>10):  $\alpha=2.5$ (放宽限制，解决大数预测的梯度消失问题，鼓励模型敢于预测大数)。
#
#         - point reward (定位任务)：
#             - 公式：$R_{point} = 1.0 \times \exp(-\alpha \times \max(0, \text{dist} - \text{tolerance})) - \text{penalty}$。
#             - **容忍区间 (Tolerance)**：允许 2% (约20px) 的相对误差，在此范围内直接得满分 1.0，避免过拟合标注噪声。
#             - **连续衰减**：超出容忍区后才开始指数衰减 ($\alpha=5.0$)，保持函数连续性。
#             - 减去越界惩罚，确保预测点在图像范围内。  

import re
import json
import math
from typing import Dict, Optional, List, Any
from PIL import Image as PILImage

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

def format_reward(predict: str, ground_truth: str) -> float:
    """
    格式正确奖励 (Context-Aware)：
    检查是否符合 <think>...</think><answer>...</answer> 或 <think>...</think><point>...</point>
    1. 解析 ground_truth 判断任务类型 (count vs point)。
    2. 只有输出格式与任务类型匹配，才给分。
    """
    # 解析任务类型
    gt_data = parse_ground_truth(ground_truth)
    task_type = gt_data.get("type", "unknown")

    # 定义正则
    pattern_answer = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    # pattern_point = re.compile(r"<think>.*?</think>\s*<point>.*?</point>", re.DOTALL)
    pattern_point = re.compile(r"<point>.*?</point>", re.DOTALL)
    
    # 逻辑判断
    if task_type == "count":
        # 计数任务：必须包含 <answer>
        if pattern_answer.search(predict):
            return 1.0
    elif task_type == "point":
        # 定位任务：必须包含 <point>
        if pattern_point.search(predict):
            return 1.0
            
    return 0.0


def parse_number(text: str) -> Optional[int]:
    """从文本中提取数字"""
    try:
        # 尝试直接转换为整数
        return int(text.strip())
    except ValueError:
        # 尝试从文本中提取数字
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])
    return None


def parse_ground_truth(ground_truth: str) -> Dict:
    """
    解析ground_truth，判断类型并提取内容
    返回: {"type": "count" | "point", "value": ...}
    """
    try:
        # 尝试解析为JSON（point类型）
        gt_data = json.loads(ground_truth)
        if "point_2d" in gt_data:
            return {
                "type": "point",
                "point_2d": gt_data["point_2d"],  # [x, y]
                "count_number": gt_data.get("count_number", 1)
            }
    except (json.JSONDecodeError, TypeError):
        pass
    
    # 否则尝试解析为数字（count类型）
    count_num = parse_number(ground_truth)
    if count_num is not None:
        return {
            "type": "count",
            "count_number": count_num
        }
    
    return {"type": "unknown"}


def parse_coordinates_from_text(text: str) -> Optional[list]:
    """从文本中提取坐标列表，格式如：[x1,y1]"""
    try:
        # 匹配所有 [x,y] 格式的坐标
        coord_pattern = r'\[\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\s*\]'
        matches = re.findall(coord_pattern, text)
        if matches:
            coords = [(float(x), float(y)) for x, y in matches]
            return coords
    except Exception:
        pass
    return None


def get_image_size(image_path: str) -> tuple:
    """获取图片的实际尺寸"""
    try:
        with PILImage.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
        # 如果无法读取图片，返回默认值
        return (1024, 1024)


def compute_l1_distance(pred_point: tuple, gt_point: list, img_width: int, img_height: int) -> float:
    """
    计算预测坐标和GT坐标之间的L1距离（归一化）
    pred_point: (x, y)
    gt_point: [x, y]
    """
    px, py = pred_point
    gx, gy = gt_point[0], gt_point[1]
    
    # L1距离：|px - gx| + |py - gy|
    l1_distance = abs(px - gx) + abs(py - gy)
    
    # 归一化：使用图片对角线长度作为最大距离
    max_distance = max(img_width + img_height, 1)
    normalized_distance = min(l1_distance / max_distance, 1.0)
    
    return normalized_distance


def check_coordinates_in_bounds(coords: list, img_width: int, img_height: int) -> float:
    """检查坐标是否在图片范围内，超出范围给予惩罚"""
    penalty = 0.0
    for x, y in coords:
        if x < 0 or x > img_width or y < 0 or y > img_height:
            penalty += 0.2  # 每个超出范围的点扣0.2分
    return min(penalty, 0.7)  # 最多扣0.7分


def accuracy_reward(predict: str, ground_truth: str) -> float:
    """
    计数准确度奖励：采用 Exponential Decay
    - 新逻辑采用 reward = 1.0* exp(-5.0 * error_rate)
    - 即使数据只占 15%，这种强梯度信号也能保证模型学会计数。
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
        
        # 完全正确直接返回 1.0 (包括 0 == 0)
        if pred_num == gt_num:
            return 1.0
        
        # 计算归一化误差
        # 为避免除以零，当 gt_num 为 0 时，使用 max(gt_num, 1) 作为分母
        denominator = max(gt_num, 1)
        normalized_error = abs(pred_num - gt_num) / denominator
        
        # 使用指数衰减对齐难度 (Alpha 调整)
        # alpha=5.0 对于大数值(11-50)过于严苛，中间态(如预测25/50)得分仅0.08，导致梯度消失。
        # 【优化】采用自适应 Alpha：
        # - 小数值(<=10): Alpha=5.0 (保持严格，误差1就是大错)
        # - 大数值(>10):  Alpha=2.5 (放宽限制，鼓励模型先"敢于"数大数)
        # 举例 GT=50, Pred=25 (Err=0.5):
        # - Alpha=5.0 -> Score 0.08 (模型认为全错)
        # - Alpha=2.5 -> Score 0.28 (模型收到正反馈，知道比预测5要好)
        if gt_num > 10:
            alpha = 2.5
        else:
            alpha = 5.0
            
        reward = 1.0 * safe_exp(-alpha * normalized_error)
        
        return clamp_reward(reward)
    
    except Exception:
        return 0.0


def point_reward(predict: str, gt_point: list, image_path: str = None, 
                 img_width: int = None, img_height: int = None) -> float:
    """
    坐标点奖励：基于归一化L1距离的指数衰减 (Exponential Decay)
    - 采用 Exponential Decay: reward = 1.0 * exp(-alpha * normalized_dist)
    - 即使误差很小，梯度依然存在，鼓励像素级对齐
    - 减去越界惩罚
    
    Args:
        predict: 模型预测输出
        gt_point: GT坐标 [x, y]
        image_path: 图片路径（用于获取实际尺寸）
        img_width: 图片宽度（可选，如果提供则不读取图片）
        img_height: 图片高度（可选，如果提供则不读取图片）
    """
    try:
        # 从预测中提取坐标
        content_match = re.search(r"<point>(.*?)</point>", predict, re.DOTALL)
        if not content_match:
            return 0.0
        
        pred_text = content_match.group(1).strip()
        
        # 解析预测的坐标（可能有多个点，取第一个）
        pred_coords = parse_coordinates_from_text(pred_text)
        
        if pred_coords is None or len(pred_coords) == 0:
            return 0.0
        
        pred_point = pred_coords[0]  # 取第一个预测点
        
        # 获取图片尺寸
        if img_width is None or img_height is None:
            if image_path:
                img_width, img_height = get_image_size(image_path)
            else:
                img_width, img_height = 1024, 1024  # 默认值
        
        # 计算L1距离
        normalized_distance = compute_l1_distance(pred_point, gt_point, img_width, img_height)
        
        # 检查坐标是否在边界内 
        out_of_bounds_penalty = check_coordinates_in_bounds([pred_point], img_width, img_height)
        
        # 使用指数衰减 (Exponential Decay) 
        # alpha 控制衰减的敏锐度（Temperature）。
        # 设计考量：
        # - GT 是中心点，但物体有体积。模型打中物体边缘也应给予正向反馈（Recall）。
        # - 过于严格的 alpha (如 20) 会导致打中物体边缘得分过低，模型难以收敛。
        # - Alpha=20: 误差 0.04 -> 得分 0.45 (不及格，打击模型积极性)
        # - Alpha=10: 误差 0.04 -> 得分 0.67 (良好，鼓励模型先"打中"物体)
        # - Alpha=5: 误差 0.04 -> 得分 0.82 (优秀，鼓励模型进一步精确定位)
        # - Alpha=3: 误差 0.04 -> 得分 0.90 (极好，鼓励追求完美)
        # - Alpha=10: 误差 0.01 (中心) -> 得分 0.90 (优秀，引导精确定位)
        # - Alpha=5: 误差 0.01 (中心) -> 得分 0.95 (极好，鼓励追求完美)
        # - Alpha=3: 误差 0.01 (中心) -> 得分 0.98 (极好，鼓励追求完美)
        # 这样 Point 和 Accuracy 的分数分布更加一致，防止模型倾向于刷 Point 分。
        # 相比线性截断，这种设计在顶点处尖锐 (Sharp Peak)，能持续提供优化梯度。        
        # 【优化1】容忍半径 (Tolerance)
        # 假设物体有大小，允许 2% 的相对误差 (在1024图上约为20像素)
        # 只要在这个范围内，就给满分，避免过拟合标注噪声
        tolerance = 0.02
        
        # 【优化2】防污染截断 (Anti-Pollution Cutoff)
        # alpha=5时，0.1(10%)的误差仍有 0.67 分。这个分数对于一个"画歪了100像素"的严重错误来说太高了。
        # 在多轮任务中，这意味着模型可能觉得"随便画个大概"也是一种不错的策略，
        # 从而导致后续轮次的图片被污染。
        # 因此设置硬截断是必要的。
        # 【调整】采用 Alpha=10.0 (更严格的衰减)
        # 并在 > 0.1 处使用强力惩罚，形成断崖式下跌，防止污染数据。
        failure_threshold = 0.1
        
        # 0.1(10%)处得分约为 0.45 (不及格)，迫使模型尽可能向中心靠拢
        alpha = 10.0
        
        if normalized_distance <= tolerance:
            main_reward = 1.0
        elif normalized_distance > failure_threshold:
            # 【调整】不直接给0，而是给一个极低分(如 < 0.1)，保持微弱梯度
            # 增加惩罚 alpha_penalty = 50.0 (从20.0提升到50.0)
            # 在 0.1 处接续 alpha=10 的得分 0.45 进行超断崖式下跌
            # dist=0.12 (偏差12%) -> Score ≈ 0.16 (几乎归零)
            score_at_threshold = 1.0 * safe_exp(-alpha * (failure_threshold - tolerance))
            main_reward = score_at_threshold * safe_exp(-50.0 * (normalized_distance - failure_threshold))
        else:
            # 在容忍区和失败区之间，进行指数衰减
            # 保持连续性，从 tolerance 处开始衰减
            main_reward = 1.0 * safe_exp(-alpha * (normalized_distance - tolerance))
            
        # 减去越界惩罚 
        reward = max(0.0, main_reward - out_of_bounds_penalty)
        
        return clamp_reward(reward)
    
    except Exception:
        return 0.0


def content_reward(predict: str, ground_truth: str, img_width: int, img_height: int) -> float:
    """
    内容奖励：根据GT类型自动选择accuracy或point reward
    """
    # 解析ground_truth
    gt_data = parse_ground_truth(ground_truth)
    
    if gt_data["type"] == "point":
        # GT包含坐标，使用point reward
        return point_reward(predict, gt_data["point_2d"], img_width=img_width, img_height=img_height)
    elif gt_data["type"] == "count":
        # GT是数字，使用accuracy reward
        return accuracy_reward(predict, str(gt_data["count_number"]))
    else:
        return 0.0


def compute_score(
    predict: str, 
    ground_truth: str,
    image_path: str = None,
    images: List[Any] = None,
    format_weight: float = 0.3,
    content_weight: float = 0.7
) -> Dict[str, float]:
    """
    计算综合得分
    
    Args:
        predict: 模型预测输出
        ground_truth: 标准答案（可能是数字或JSON格式的point数据）
        image_path: 图片路径（用于获取实际尺寸，备用）
        images: 图片对象列表（优先使用）
        format_weight: 格式得分权重
        content_weight: 内容得分权重（accuracy或point）
    
    Returns:
        包含各项得分的字典
    """
    # 1. 获取图片尺寸
    img_width, img_height = 1024, 1024 # 默认值
    
    # 优先从 images 对象获取 (Verl 传递的 PIL Image)
    if images and len(images) > 0:
        try:
            # images[0] 应该是 PIL Image 对象
            if hasattr(images[0], 'size'):
                img_width, img_height = images[0].size
        except Exception:
            pass
    # 其次尝试从 image_path 获取 (如果传递了路径)
    elif image_path:
        try:
            with PILImage.open(image_path) as img:
                img_width, img_height = img.size
        except Exception:
            pass

    format_score = clamp_reward(format_reward(predict, ground_truth))

    # content_score = content_reward(predict, ground_truth, img_width, img_height)

    # NEW: 解析任务类型，分别计算并只返回对应的 reward key
    # 这样在 wandb 聚合时，point 和 accuracy 会分别计算各自存在的样本的平均值，
    # 避免因为样本类型不同导致的 0 值拉低曲线。
    gt_data = parse_ground_truth(ground_truth)
    task_type = gt_data.get("type", "unknown")
    
    content_score = 0.0
    score = {}

    if task_type == "point":
        # GT包含坐标，使用point reward
        pt_score = point_reward(predict, gt_data["point_2d"], img_width=img_width, img_height=img_height)
        pt_score = clamp_reward(pt_score)
        content_score = pt_score
        score["point"] = pt_score
    elif task_type == "count":
        # GT是数字，使用accuracy reward
        acc_score = accuracy_reward(predict, str(gt_data["count_number"]))
        acc_score = clamp_reward(acc_score)
        content_score = acc_score
        score["accuracy"] = acc_score
    else:
        # 未知类型，content_score 为 0
        content_score = 0.0
    
    # 计算overall得分
    overall_score = clamp_reward(format_weight * format_score + content_weight * content_score)
    
    # return {
    #     "overall": overall_score,
    #     "format": format_score,
    #     "content": content_score,
    # }

    # NEW: 返回更细化的 score 字典
    score["overall"] = overall_score
    score["format"] = format_score
    score["content"] = content_score
    return score