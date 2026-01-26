# analyze_selected_patterns.py

import numpy as np
import matplotlib.pyplot as plt
import os
import json


def load_and_analyze_results(result_path):
    """
    从result.json或原始输出文件中加载 `selected_sequence`，分析通过LocalBranch/TASE使用的最终列。
    :param result_path: 包含 selected_sequence 和列池信息的 JSON 文件路径
    """
    # Step 1: 读取生成的结果文件
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"指定的结果文件 {result_path} 不存在！")

    print(f"正在读取结果文件：{result_path}...")
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    selected_sequence = data.get('selected_sequence', [])
    P_matrix = np.array(data.get('optimized_patterns', []))
    if not selected_sequence:
        raise ValueError(
            "结果文件中没有找到 selected_sequence！请确认优化结果是否正确生成。")

    print(f"成功加载 {len(selected_sequence)} 个选用模式，开始分析...")

    # Step 2: 分析模式的具体信息（选用次数、修整损失等）
    analyze_selected_patterns(selected_sequence, P_matrix)


def analyze_selected_patterns(selected_sequence, P_matrix):
    """
    对选用列进行分析：修整损失、模式复杂性、使用频次等
    :param selected_sequence: 每一步选用的列（包含修整损失、使用次数等信息）
    :param P_matrix: 如果提供了最终列池矩阵，用于参考完整列池内容
    """
    trim_losses = [step['trim'] for step in selected_sequence]
    usage_counts = [step['k'] for step in selected_sequence]
    used_lengths = [step['used_len'] for step in selected_sequence]
    patterns = [step['a'] for step in selected_sequence]

    # 统计修整损失
    print("\n** 模式修整损失 (Trim Loss) 分布总结：")
    print(f"最大修整损失: {max(trim_losses)}")
    print(f"最小修整损失: {min(trim_losses)}")
    print(f"修整损失的均值: {np.mean(trim_losses):.2f}")

    # 最大频次模式
    print("\n** 使用频次 k 总结 (Usage Counts):")
    max_usage_count_idx = np.argmax(usage_counts)
    print(
        f"模式 {max_usage_count_idx} 被使用次数最多，k = {usage_counts[max_usage_count_idx]}")
    print(f"最大模式内容 (Pattern): {patterns[max_usage_count_idx]}")

    # 可视化修整损失 (Trim Loss Histogram)
    plt.hist(trim_losses, bins=10)
    plt.title("修整损失分布 (Trim Loss Distribution)")
    plt.xlabel("Trim Loss")
    plt.ylabel("Frequency")
    plt.show()

    # 绘制修整损失与列模式使用长度之间的关系 (Scatter)
    plt.scatter(used_lengths, trim_losses, c='blue', alpha=0.7)
    plt.title("Used Length vs Trim Loss of Patterns")
    plt.xlabel("Used Length")
    plt.ylabel("Trim Loss")
    plt.grid()
    plt.show()

    if P_matrix.size > 0:
        print("\n** 全局模式池 (P_matrix) 分析：")
        print(f"模式池总列数: {P_matrix.shape[1]}")
        print(f"模式池行列子矩阵形状: {P_matrix.shape}")
        print(f"模式池示例：\n{P_matrix[:, :5]}")  # 打印前 5 个模式


# Example Usage
if __name__ == "__main__":
    # 使用的结果路径（根据脚本运行路径调整）
    example_result_path = './result.json'

    # 加载并分析结果
    load_and_analyze_results(result_path=example_result_path)
