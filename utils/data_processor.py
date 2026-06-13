"""
数据处理工具

功能:
1. 加载 Excel 数据
2. 解析目标和立场标签
3. 数据预处理
"""

import pandas as pd
from typing import List, Dict, Any, Tuple


class DataProcessor:
    """数据处理器"""

    def __init__(self, data_path: str):
        """
        Args:
            data_path: Excel 文件路径
        """
        self.data_path = data_path
        self.df = None

    def load(self) -> pd.DataFrame:
        """加载数据"""
        self.df = pd.read_excel(self.data_path)
        return self.df

    def parse_targets(self, target_str: str) -> List[str]:
        """
        解析目标字符串

        Args:
            target_str: "目标 1；目标 2；目标 3"

        Returns:
            目标列表
        """
        if pd.isna(target_str):
            return []

        # 按分号分割
        targets = [t.strip() for t in str(target_str).split(';')]
        targets = [t for t in targets if t]  # 去除空字符串

        return targets

    def parse_stances(self, stance_str: str, targets: List[str]) -> List[Dict[str, str]]:
        """
        解析立场字符串

        Args:
            stance_str: "支持；中立；反对"
            targets: 目标列表

        Returns:
            [{target, stance}, ...]
        """
        if pd.isna(stance_str):
            return []

        stances = [s.strip() for s in str(stance_str).split(';')]
        stances = [s for s in stances if s]

        # 对齐目标和立场
        result = []
        for i, target in enumerate(targets):
            if i < len(stances):
                result.append({
                    'target': target,
                    'stance': stances[i]
                })
            else:
                # 目标多于立场，默认中立
                result.append({
                    'target': target,
                    'stance': '中立'
                })

        return result

    def get_samples(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        获取样本列表

        Args:
            limit: 限制数量

        Returns:
            [{text, targets, stances}, ...]
        """
        if self.df is None:
            self.load()

        samples = []
        df = self.df if limit is None else self.df.head(limit)

        for idx, row in df.iterrows():
            text = row['blog_text']
            targets = self.parse_targets(row['target'])
            stance_info = self.parse_stances(row['stance'], targets)

            samples.append({
                'id': idx,
                'text': text,
                'targets': targets,
                'stances': stance_info
            })

        return samples

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        if self.df is None:
            self.load()

        # 目标数量分布
        target_counts = self.df['target'].apply(
            lambda x: len(str(x).split(';')) if pd.notna(x) else 0
        )

        # 立场分布
        stance_dist = {}
        for stance_str in self.df['stance']:
            if pd.notna(stance_str):
                for s in str(stance_str).split(';'):
                    s = s.strip()
                    stance_dist[s] = stance_dist.get(s, 0) + 1

        return {
            'total_samples': len(self.df),
            'avg_targets': target_counts.mean(),
            'max_targets': target_counts.max(),
            'min_targets': target_counts.min(),
            'stance_distribution': stance_dist
        }


if __name__ == "__main__":
    # 测试
    processor = DataProcessor('../测试-data-最终版.xlsx')
    processor.load()

    stats = processor.get_statistics()
    print("数据统计:")
    print(f"  总样本数：{stats['total_samples']}")
    print(f"  平均目标数：{stats['avg_targets']:.2f}")
    print(f"  最多目标数：{stats['max_targets']}")
    print(f"  最少目标数：{stats['min_targets']}")
    print(f"  立场分布：{stats['stance_distribution']}")