"""
VA Word Mapper - 基于Valence-Arousal坐标映射词汇

该模块提供根据VA坐标在ANEW词表中查找最近词汇的功能。
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class VAWordMapper:
    """基于Valence-Arousal坐标的词汇映射器"""
    def __init__(self, csv_path: Optional[str] = None):
        """
        初始化VA词汇映射器
        
        Args:
            csv_path: ANEW数据集CSV文件路径，如果为None则使用默认路径
        """
        if csv_path is None:
            # 默认路径：相对于当前文件的knowledge_base目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(
                os.path.dirname(current_dir), 'knowledge_base',
                'anew_overall_ratings.csv'
            )

        # 加载数据
        self.df = pd.read_csv(csv_path)
        # print(f"已加载 {len(self.df)} 个词汇")

        # 预处理：确保Frequency列存在且为数值
        if 'Frequency' not in self.df.columns:
            print("警告: 数据集中没有Frequency列，将使用默认频率1")
            self.df['Frequency'] = 1
        else:
            # 转换为数值类型
            self.df['Frequency'] = pd.to_numeric(
                self.df['Frequency'], errors='coerce'
            )
            # 将频率为0或NaN的词设置为1（确保所有词都有被选中的概率）
            self.df['Frequency'] = self.df['Frequency'].fillna(1)
            self.df.loc[self.df['Frequency'] == 0, 'Frequency'] = 1

    def get_nearest_words(
        self,
        valence: float,
        arousal: float,
        k: int = 10,
        return_word: bool = True,
        random_seed: Optional[int] = None
    ) -> str | Tuple[str, pd.DataFrame]:
        """
        根据VA坐标查找最近的K个词，基于频率加权随机选择一个
        
        Args:
            valence: 效价值 (1-9)
            arousal: 唤醒度值 (1-9)
            k: 选择最近的k个词汇
            return_word: 如果为True，只返回选中的词；如果为False，返回词和候选词信息
            random_seed: 随机种子，用于复现结果
        
        Returns:
            如果return_word=True: 返回选中的词汇
            如果return_word=False: 返回 (选中的词汇, 候选词DataFrame)
        """
        # 参数验证
        if not (1 <= valence <= 9):
            raise ValueError(f"Valence必须在[1, 9]范围内，当前值: {valence}")
        if not (1 <= arousal <= 9):
            raise ValueError(f"Arousal必须在[1, 9]范围内，当前值: {arousal}")
        if k <= 0:
            raise ValueError(f"K必须大于0，当前值: {k}")
        if k > len(self.df):
            print(f"警告: k={k} 大于词汇总数 {len(self.df)}，将使用所有词汇")
            k = len(self.df)

        # 计算欧氏距离
        distances = np.sqrt((self.df['Valence Mean'] - valence)**2 +
                            (self.df['Arousal Mean'] - arousal)**2)

        # 获取最近的k个词
        nearest_indices = distances.nsmallest(k).index
        nearest_words = self.df.loc[nearest_indices].copy()
        nearest_words['Distance'] = distances[nearest_indices]

        # 基于频率计算概率（频率越高，被选中概率越大）
        frequencies = nearest_words['Frequency'].values
        probabilities = frequencies / frequencies.sum()

        # 基于概率随机选择一个词
        if random_seed is not None:
            np.random.seed(random_seed)

        selected_idx = np.random.choice(len(nearest_words), p=probabilities)
        selected_word = nearest_words.iloc[selected_idx]['Word']

        # 添加概率列到结果中
        nearest_words['Probability'] = probabilities

        if return_word:
            return selected_word
        else:
            return selected_word, nearest_words

    def get_word_info(self, word: str) -> Optional[pd.Series]:
        """
        获取指定词汇的完整信息
        
        Args:
            word: 词汇
        
        Returns:
            词汇信息Series，如果词汇不存在则返回None
        """
        result = self.df[self.df['Word'].str.lower() == word.lower()]
        if len(result) == 0:
            return None
        return result.iloc[0]


def map_va_to_word(
    valence: float,
    arousal: float,
    k: int = 10,
    csv_path: Optional[str] = None,
    random_seed: Optional[int] = None
) -> str:
    """
    便捷函数：根据VA坐标映射到一个词汇
    
    Args:
        valence: 效价值 (1-9)
        arousal: 唤醒度值 (1-9)
        k: 选择最近的k个词汇
        csv_path: ANEW数据集路径
        random_seed: 随机种子
    
    Returns:
        选中的词汇
    """
    mapper = VAWordMapper(csv_path)
    return mapper.get_nearest_words(
        valence, arousal, k, return_word=True, random_seed=random_seed
    )


if __name__ == "__main__":
    print("=" * 70)
    print("VA Word Mapper 测试")
    print("=" * 70)

    # 初始化映射器
    mapper = VAWordMapper()

    # 测试用例
    test_cases = [
        (8.0, 7.0, 5, "高效价、高唤醒（兴奋/激动）"),
        (2.0, 7.0, 5, "低效价、高唤醒（焦虑/恐惧）"),
        (2.0, 2.0, 5, "低效价、低唤醒（沮丧/悲伤）"),
        (7.0, 2.0, 5, "高效价、低唤醒（平静/满足）"),
        (5.0, 5.0, 10, "中性情感"),
    ]

    print("\n【测试1: 基本功能测试】")
    for valence, arousal, k, description in test_cases:
        print(f"\n测试: {description}")
        print(f"输入坐标: Valence={valence}, Arousal={arousal}, K={k}")

        # 获取详细信息
        selected_word, candidates = mapper.get_nearest_words(
            valence,
            arousal,
            k,
            return_word=False,
            random_seed=42  # 固定种子以便复现
        )

        print(f"\n选中的词汇: {selected_word}")
        print(f"\n候选词汇（前{k}个最近的词）:")
        print("-" * 70)

        # 显示候选词信息
        display_cols = [
            'Word', 'Valence Mean', 'Arousal Mean', 'Frequency', 'Distance',
            'Probability'
        ]
        display_df = candidates[display_cols].round(4)

        for idx, row in display_df.iterrows():
            marker = "★" if row['Word'] == selected_word else " "
            print(
                f"{marker} {row['Word']:15s} | V:{row['Valence Mean']:5.2f} A:{row['Arousal Mean']:5.2f} | "
                f"Freq:{row['Frequency']:4.0f} | Dist:{row['Distance']:5.3f} | Prob:{row['Probability']:6.2%}"
            )

    print("\n" + "=" * 70)
    print("【测试2: 多次采样测试 - 验证概率分布】")
    print("=" * 70)

    # 测试同一个坐标多次采样，验证频率高的词被选中次数更多
    valence, arousal, k = 7.5, 7.0, 5
    n_samples = 1000

    print(f"\n坐标: Valence={valence}, Arousal={arousal}, K={k}")
    print(f"采样次数: {n_samples}")

    # 获取候选词信息
    _, candidates = mapper.get_nearest_words(
        valence, arousal, k, return_word=False, random_seed=42
    )

    # 多次采样
    sampled_words = []
    for i in range(n_samples):
        word = mapper.get_nearest_words(valence, arousal, k, return_word=True)
        sampled_words.append(word)

    # 统计结果
    from collections import Counter
    word_counts = Counter(sampled_words)

    print("\n采样结果统计:")
    print("-" * 70)
    print(f"{'词汇':<15} | {'理论概率':>10} | {'实际频次':>10} | {'实际概率':>10}")
    print("-" * 70)

    for word in candidates['Word']:
        theoretical_prob = candidates[candidates['Word'] == word
                                     ]['Probability'].values[0]
        actual_count = word_counts.get(word, 0)
        actual_prob = actual_count / n_samples
        print(
            f"{word:<15} | {theoretical_prob:>9.2%} | {actual_count:>10} | {actual_prob:>9.2%}"
        )

    print("\n" + "=" * 70)
    print("【测试3: 边界值测试】")
    print("=" * 70)

    boundary_cases = [
        (1.0, 1.0, 3, "最小边界"),
        (9.0, 9.0, 3, "最大边界"),
        (1.0, 9.0, 3, "左上角"),
        (9.0, 1.0, 3, "右下角"),
    ]

    for valence, arousal, k, description in boundary_cases:
        word = mapper.get_nearest_words(valence, arousal, k, random_seed=42)
        print(f"{description} (V={valence}, A={arousal}): {word}")

    print("\n" + "=" * 70)
    print("【测试4: 词汇信息查询】")
    print("=" * 70)

    test_words = ['happy', 'sad', 'angry', 'calm', 'excited']
    for word in test_words:
        info = mapper.get_word_info(word)
        if info is not None:
            print(f"\n{word.capitalize()}:")
            print(
                f"  Valence: {info['Valence Mean']:.2f} (±{info['Valence SD']:.2f})"
            )
            print(
                f"  Arousal: {info['Arousal Mean']:.2f} (±{info['Arousal SD']:.2f})"
            )
            print(f"  Frequency: {info['Frequency']:.0f}")
        else:
            print(f"\n{word.capitalize()}: 未在词表中找到")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
