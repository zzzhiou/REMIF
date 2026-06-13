#!/usr/bin/env python3
"""
从原始数据中提取多目标样本

将目标数量 >= 2 的样本提取出来，保存为新表格
"""

import pandas as pd
import re
import os


def parse_targets(target_str):
    """解析目标字符串 - 使用中文分号分割"""
    if pd.isna(target_str):
        return []
    # 中文全角分号 U+FF1B
    targets = str(target_str).split('\uff1b')
    return [t.strip() for t in targets if t.strip()]


def parse_stances(stance_str):
    """解析立场字符串 - 使用中文分号分割"""
    if pd.isna(stance_str):
        return []
    stances = str(stance_str).split('\uff1b')
    return [s.strip() for s in stances if s.strip()]


def extract_multi_target_samples(input_path, output_path, min_targets=2):
    """
    提取多目标样本

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        min_targets: 最少目标数量（默认 2）
    """
    print(f"加载数据：{input_path}")
    df = pd.read_excel(input_path)

    print(f"总样本数：{len(df)}")

    # 解析目标并计数
    df['targets_list'] = df['target'].apply(parse_targets)
    df['target_count'] = df['targets_list'].apply(len)
    df['stances_list'] = df['stance'].apply(parse_stances)
    df['stance_count'] = df['stances_list'].apply(len)

    # 统计信息
    print(f"\n目标数量分布:")
    target_count_dist = df['target_count'].value_counts().sort_index()
    for count, freq in target_count_dist.items():
        print(f"  {count} 个目标：{freq} 个样本 ({freq/len(df)*100:.1f}%)")

    # 提取多目标样本
    multi_target_df = df[df['target_count'] >= min_targets].copy()

    print(f"\n多目标样本（>= {min_targets} 个目标）: {len(multi_target_df)} 个")
    print(f"占比：{len(multi_target_df)/len(df)*100:.1f}%")

    if len(multi_target_df) == 0:
        print("没有找到多目标样本!")
        return

    # 创建输出 DataFrame
    output_df = multi_target_df[['blog_text', 'target', 'stance']].copy()

    # 添加额外信息列
    output_df['目标数量'] = multi_target_df['target_count']
    output_df['立场数量'] = multi_target_df['stance_count']

    # 保存
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    output_df.to_excel(output_path, index=False)

    print(f"\n已保存到：{output_path}")

    # 显示前几个样本
    print(f"\n前 5 个多目标样本预览:")
    print("="*80)
    for idx, row in output_df.head(5).iterrows():
        targets = parse_targets(row['target'])
        print(f"\n样本 {idx}:")
        print(f"  目标数量：{row['目标数量']}")
        print(f"  目标：{'; '.join(targets)}")
        print(f"  立场：{row['stance']}")
        print(f"  文本：{str(row['blog_text'])[:80]}...")


def create_detailed_dataset(input_path, output_path):
    """
    创建详细的数据集（每个目标一行）

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
    """
    print(f"\n{'='*80}")
    print("创建详细数据集（每个目标一行）...")
    print(f"{'='*80}")

    df = pd.read_excel(input_path)

    # 展开数据
    detailed_rows = []

    for idx, row in df.iterrows():
        text = row['blog_text']
        targets = parse_targets(row['target'])
        stances = parse_stances(row['stance'])

        for i, target in enumerate(targets):
            stance = stances[i] if i < len(stances) else '未知'
            detailed_rows.append({
                'sample_id': idx,
                'text': text,
                'target': target,
                'stance': stance,
                'target_index': i + 1
            })

    detailed_df = pd.DataFrame(detailed_rows)

    print(f"原始样本数：{len(df)}")
    print(f"展开后（目标数）: {len(detailed_df)}")

    # 保存
    detailed_df.to_excel(output_path, index=False)
    print(f"已保存到：{output_path}")

    return detailed_df


def main():
    """主函数"""
    input_path = '../测试-data-最终版.xlsx'

    # 输出路径
    multi_target_path = '../多目标样本.xlsx'
    detailed_path = '../详细数据集（按目标展开）.xlsx'

    print("="*80)
    print("多目标样本提取工具")
    print("="*80)

    # 1. 提取多目标样本
    extract_multi_target_samples(input_path, multi_target_path, min_targets=2)

    # 2. 创建详细数据集
    create_detailed_dataset(input_path, detailed_path)

    print("\n" + "="*80)
    print("处理完成!")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  1. {multi_target_path} - 多目标样本（每个样本一行）")
    print(f"  2. {detailed_path} - 详细数据集（每个目标一行）")


if __name__ == "__main__":
    main()
