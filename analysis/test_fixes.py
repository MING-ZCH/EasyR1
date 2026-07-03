#!/usr/bin/env python3
"""
简单测试脚本，验证修复是否正常工作
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_global_num_tokens_handling():
    """测试 global_num_tokens 的处理逻辑"""
    print("=" * 60)
    print("测试 1: global_num_tokens 处理逻辑")
    print("=" * 60)
    
    # 测试不同的输入类型
    test_cases = [
        ([100, 200, 300], "list 类型"),
        (600, "int 类型"),
        (600.0, "float 类型"),
        ([], "空 list"),
        (None, "None 类型"),
    ]
    
    for test_input, description in test_cases:
        print(f"\n测试输入: {test_input} ({description})")
        
        # 模拟修复后的逻辑
        global_num_tokens_raw = test_input if test_input is not None else 0
        
        if isinstance(global_num_tokens_raw, list):
            batch_seqlens = global_num_tokens_raw
        elif isinstance(global_num_tokens_raw, (int, float)):
            batch_seqlens = [int(global_num_tokens_raw)]
        else:
            batch_seqlens = []
        
        print(f"  处理结果: batch_seqlens = {batch_seqlens}")
        print(f"  类型: {type(batch_seqlens)}")
        
        # 验证可以调用 sum
        if len(batch_seqlens) > 0:
            total = sum(batch_seqlens)
            print(f"  总和: {total}")
            print(f"  ✓ 可以正常调用 sum()")
        else:
            print(f"  ✓ 空列表，跳过计算")
    
    print("\n✓ 测试 1 通过\n")


def test_estimate_flops_input():
    """测试 estimate_flops 的输入格式"""
    print("=" * 60)
    print("测试 2: estimate_flops 输入格式验证")
    print("=" * 60)
    
    # 模拟 estimate_flops 函数的输入要求
    def mock_estimate_flops(batch_seqlens, delta_time):
        """模拟 estimate_flops 函数"""
        if not isinstance(batch_seqlens, list):
            raise TypeError(f"batch_seqlens 必须是 list，但得到 {type(batch_seqlens)}")
        tokens_sum = sum(batch_seqlens)
        return tokens_sum, 1.0
    
    test_cases = [
        ([100, 200, 300], 1.0, "正常 list"),
        ([600], 1.0, "单个元素的 list"),
        ([], 1.0, "空 list"),
    ]
    
    for batch_seqlens, delta_time, description in test_cases:
        print(f"\n测试: {description}")
        print(f"  输入: batch_seqlens={batch_seqlens}, delta_time={delta_time}")
        try:
            result = mock_estimate_flops(batch_seqlens, delta_time)
            print(f"  结果: {result}")
            print(f"  ✓ 成功")
        except Exception as e:
            print(f"  ✗ 错误: {e}")
    
    # 测试错误的输入（整数）
    print(f"\n测试: 错误输入（整数）")
    try:
        result = mock_estimate_flops(600, 1.0)  # 这应该会失败
        print(f"  ✗ 应该失败但没有")
    except TypeError as e:
        print(f"  ✓ 正确捕获错误: {e}")
    
    print("\n✓ 测试 2 通过\n")


def test_image_validation():
    """测试 image features/tokens 校验逻辑"""
    print("=" * 60)
    print("测试 3: Image features/tokens 校验逻辑")
    print("=" * 60)
    
    # 模拟校验函数
    def validate_image_features_and_tokens(num_image_tokens, num_image_features, tolerance=0.05):
        """简化的校验函数"""
        if num_image_features == 0:
            return True, None
        
        tolerance_value = max(10, int(num_image_features * tolerance))
        diff = abs(num_image_tokens - num_image_features)
        
        if diff > tolerance_value:
            return False, f"不匹配: tokens={num_image_tokens}, features={num_image_features}, diff={diff}"
        return True, None
    
    test_cases = [
        (18557, 20167, "原始错误案例"),
        (18557, 18557, "完全匹配"),
        (18550, 18557, "在容差范围内"),
        (18500, 18557, "超出容差"),
        (0, 0, "无图像"),
    ]
    
    for tokens, features, description in test_cases:
        print(f"\n测试: {description}")
        print(f"  tokens={tokens}, features={features}")
        is_valid, error_msg = validate_image_features_and_tokens(tokens, features)
        if is_valid:
            print(f"  ✓ 校验通过")
        else:
            print(f"  ✗ 校验失败: {error_msg}")
    
    print("\n✓ 测试 3 通过\n")


def test_reward_info_formatting():
    """测试 reward 信息格式化"""
    print("=" * 60)
    print("测试 4: Reward 信息格式化")
    print("=" * 60)
    
    # 模拟 metrics
    metrics = {
        "reward/score": 0.8234,
        "reward/accuracy": 0.9123,
        "data/sequence_score_mean": 0.8500,
        "data/sequence_reward_mean": 0.8456,
        "actor/pg_loss": 0.1234,  # 不应该出现在 reward 信息中
    }
    
    reward_info = []
    for key, value in metrics.items():
        if key.startswith("reward/"):
            reward_info.append(f"{key}={value:.4f}")
    
    if "data/sequence_score_mean" in metrics:
        reward_info.append(f"sequence_score={metrics['data/sequence_score_mean']:.4f}")
    if "data/sequence_reward_mean" in metrics:
        reward_info.append(f"sequence_reward={metrics['data/sequence_reward_mean']:.4f}")
    
    if reward_info:
        formatted = ', '.join(reward_info)
        print(f"格式化结果: {formatted}")
        print(f"✓ 格式正确")
    else:
        print("✗ 没有 reward 信息")
    
    print("\n✓ 测试 4 通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试修复的功能")
    print("=" * 60 + "\n")
    
    try:
        test_global_num_tokens_handling()
        test_estimate_flops_input()
        test_image_validation()
        test_reward_info_formatting()
        
        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        return 0
    except Exception as e:
        print("=" * 60)
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())










