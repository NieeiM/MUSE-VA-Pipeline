"""
VA坐标到情绪标签的映射工具
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Dict
import random

# 情绪字母到完整名称的映射
EMOTION_LABELS = {
    'A': 'Amusing',
    'B': 'Annoying',
    'C': 'Anxious, tense',
    'D': 'Beautiful',
    'E': 'Calm, relaxing, serene',
    'F': 'Dreamy',
    'G': 'Energizing, pump-up',
    'H': 'Erotic, desirous',
    'I': 'Indignant, defiant',
    'J': 'Joyful, cheerful',
    'K': 'Sad, depressing',
    'L': 'Scary, fearful',
    'M': 'Triumphant, heroic'
}

# 所有情绪字母（按顺序）
ALL_EMOTION_LETTERS = list(EMOTION_LABELS.keys())


class VAToEmotionMapper:
    """
    Valence-Arousal坐标到情绪标签的映射器
    """
    def __init__(self, csv_path: str = None):
        """
        初始化映射器
        
        Args:
            csv_path: CSV文件路径，如果为None则使用默认路径
        """
        if csv_path is None:
            # 默认路径
            csv_path = Path(
                __file__
            ).parent.parent / 'knowledge_base' / 'cowen_emotion_data.csv'

        self.csv_path = Path(csv_path)
        self.df = None
        self._load_data()

    def _load_data(self):
        """加载CSV数据"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV文件不存在: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        # 验证必要的列是否存在
        required_cols = ['Valence', 'Arousal'] + ALL_EMOTION_LETTERS
        missing_cols = [
            col for col in required_cols if col not in self.df.columns
        ]

        if missing_cols:
            raise ValueError(f"CSV文件缺少必要的列: {missing_cols}")

        # print(f"✓ 成功加载数据: {len(self.df)} 条记录")

    def _calculate_distance(
        self, valence: float, arousal: float
    ) -> np.ndarray:
        """
        计算输入坐标到所有数据点的欧式距离
        
        Args:
            valence: Valence值 [1-9]
            arousal: Arousal值 [1-9]
        
        Returns:
            距离数组
        """
        # 提取所有数据点的VA坐标
        va_coords = self.df[['Valence', 'Arousal']].values

        # 计算欧式距离
        distances = np.sqrt((va_coords[:, 0] - valence)**2 +
                            (va_coords[:, 1] - arousal)**2)

        return distances

    def find_nearest_samples(
        self,
        valence: float,
        arousal: float,
        k: int = 10,
        random_seed: int = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        找到距离最近的K条数据
        
        Args:
            valence: Valence值 [1-9]
            arousal: Arousal值 [1-9]
            k: 返回的样本数量
            random_seed: 随机种子（用于相同距离时的随机选择）
        
        Returns:
            (最近的K条数据DataFrame, 对应的距离数组)
        """
        # 验证输入范围
        if not (1 <= valence <= 9):
            raise ValueError(f"Valence必须在[1,9]范围内，当前值: {valence}")
        if not (1 <= arousal <= 9):
            raise ValueError(f"Arousal必须在[1,9]范围内，当前值: {arousal}")

        # 计算距离
        distances = self._calculate_distance(valence, arousal)

        # 添加距离列到DataFrame（临时）
        df_with_dist = self.df.copy()
        df_with_dist['_distance'] = distances

        # 按距离排序
        df_sorted = df_with_dist.sort_values('_distance')

        # 获取第k小的距离值
        if k > len(df_sorted):
            k = len(df_sorted)
            print(f"Warning: K值({k})大于数据总量，使用全部数据")

        kth_distance = df_sorted.iloc[k - 1]['_distance']

        # 找出所有距离小于等于第k小距离的数据
        candidates = df_sorted[df_sorted['_distance'] <= kth_distance]

        # 如果候选数量恰好等于k，直接返回
        if len(candidates) == k:
            nearest_samples = candidates
        # 如果候选数量大于k，需要随机选择
        elif len(candidates) > k:
            if random_seed is not None:
                random.seed(random_seed)
                np.random.seed(random_seed)

            # 随机选择k条
            indices = np.random.choice(candidates.index, size=k, replace=False)
            nearest_samples = df_sorted.loc[indices]
        else:
            # 候选数量小于k（理论上不应该发生）
            nearest_samples = candidates

        # 移除临时距离列
        nearest_distances = nearest_samples['_distance'].values
        nearest_samples = nearest_samples.drop('_distance', axis=1)

        return nearest_samples, nearest_distances

    def get_top_emotions(
        self,
        valence: float,
        arousal: float,
        k: int = 10,
        n: int = 3,
        return_full_labels: bool = False,
        random_seed: int = None
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """
        根据VA坐标获取最相关的N个情绪标签
        
        Args:
            valence: Valence值 [1-9]
            arousal: Arousal值 [1-9]
            k: 选择最近的K条数据
            n: 返回得分最高的N个情绪
            return_full_labels: 是否返回完整的情绪名称（False返回字母）
            random_seed: 随机种子
        
        Returns:
            如果return_full_labels=False: ['A', 'B', 'C', ...]
            如果return_full_labels=True: [('Amusing', 0.85), ('Annoying', 0.72), ...]
        """
        # 找到最近的K条数据
        nearest_samples, distances = self.find_nearest_samples(
            valence, arousal, k, random_seed
        )

        # 提取13维情绪标签数据
        emotion_data = nearest_samples[ALL_EMOTION_LETTERS].values

        # 计算平均值
        emotion_avg = emotion_data.mean(axis=0)

        # 创建(字母, 分数)的列表
        emotion_scores = [
            (letter, score)
            for letter, score in zip(ALL_EMOTION_LETTERS, emotion_avg)
        ]

        # 按分数降序排序
        emotion_scores_sorted = sorted(
            emotion_scores, key=lambda x: x[1], reverse=True
        )

        # 取前N个
        top_n = emotion_scores_sorted[:n]

        if return_full_labels:
            # 返回完整标签和分数
            return [(EMOTION_LABELS[letter], score) for letter, score in top_n]
        else:
            # 只返回字母
            return [letter for letter, score in top_n]

    def get_emotion_distribution(
        self,
        valence: float,
        arousal: float,
        k: int = 10,
        random_seed: int = None
    ) -> Dict[str, float]:
        """
        获取完整的情绪分布（13维）
        
        Args:
            valence: Valence值 [1-9]
            arousal: Arousal值 [1-9]
            k: 选择最近的K条数据
            random_seed: 随机种子
        
        Returns:
            字典，格式: {'A': 0.15, 'B': 0.08, ...}
        """
        # 找到最近的K条数据
        nearest_samples, distances = self.find_nearest_samples(
            valence, arousal, k, random_seed
        )

        # 提取13维情绪标签数据
        emotion_data = nearest_samples[ALL_EMOTION_LETTERS].values

        # 计算平均值
        emotion_avg = emotion_data.mean(axis=0)

        # 返回字典
        return {
            letter: float(score)
            for letter, score in zip(ALL_EMOTION_LETTERS, emotion_avg)
        }

    def query(
        self,
        valence: float,
        arousal: float,
        k: int = 10,
        n: int = 3,
        return_format: str = 'letters',
        random_seed: int = None,
        verbose: bool = False
    ) -> Union[List[str], List[Tuple[str, float]], Dict[str, float], str]:
        """
        查询函数（综合接口）
        
        Args:
            valence: Valence值 [1-9]
            arousal: Arousal值 [1-9]
            k: 选择最近的K条数据
            n: 返回得分最高的N个情绪
            return_format: 返回格式
                - 'letters': 只返回字母列表 ['A', 'B', 'C']
                - 'full': 返回完整标签和分数 [('Amusing', 0.85), ...]
                - 'distribution': 返回完整的13维分布字典
                - 'sample': 基于概率采样返回单个完整情绪标签
            random_seed: 随机种子
            verbose: 是否打印详细信息
        
        Returns:
            根据return_format返回不同格式的结果
        """
        if verbose:
            print(
                f"查询参数: Valence={valence:.2f}, Arousal={arousal:.2f}, K={k}, N={n}"
            )

        if return_format == 'distribution':
            result = self.get_emotion_distribution(
                valence, arousal, k, random_seed
            )

            if verbose:
                print("\n情绪分布:")
                for letter in ALL_EMOTION_LETTERS:
                    score = result[letter]
                    emotion_name = EMOTION_LABELS[letter]
                    print(f"  {letter} ({emotion_name:30s}): {score:.3f}")

            return result

        elif return_format == 'full':
            result = self.get_top_emotions(
                valence, arousal, k, n, True, random_seed
            )

            if verbose:
                print(f"\n前{n}个情绪:")
                for i, (emotion_name, score) in enumerate(result, 1):
                    letter = [
                        k
                        for k, v in EMOTION_LABELS.items() if v == emotion_name
                    ][0]
                    print(f"  {i}. {letter} - {emotion_name:30s}: {score:.3f}")

            return result

        elif return_format == 'sample':
            # 获取得分最高的N个情绪
            top_emotions = self.get_top_emotions(
                valence, arousal, k, n, True, random_seed
            )

            # 提取分数
            scores = np.array([score for _, score in top_emotions])

            # 归一化为概率
            probabilities = scores / scores.sum()

            if verbose:
                print(f"\n前{n}个情绪及其概率:")
                for i, ((emotion_name, score), prob
                       ) in enumerate(zip(top_emotions, probabilities), 1):
                    letter = [
                        k
                        for k, v in EMOTION_LABELS.items() if v == emotion_name
                    ][0]
                    print(
                        f"  {i}. {letter} - {emotion_name:30s}: 分数={score:.3f}, 概率={prob:.3f}"
                    )

            # 设置随机种子
            if random_seed is not None:
                np.random.seed(random_seed)

            # 基于概率采样
            selected_idx = np.random.choice(len(top_emotions), p=probabilities)
            selected_emotion = top_emotions[selected_idx][0]

            if verbose:
                letter = [
                    k
                    for k, v in EMOTION_LABELS.items() if v == selected_emotion
                ][0]
                print(f"\n采样结果: {letter} - {selected_emotion}")

            return selected_emotion

        else:  # 'letters'
            result = self.get_top_emotions(
                valence, arousal, k, n, False, random_seed
            )

            if verbose:
                print(f"\n前{n}个情绪标签: {', '.join(result)}")
                # 同时显示完整名称
                for letter in result:
                    print(f"  {letter}: {EMOTION_LABELS[letter]}")

            return result


# 便捷函数
def va_to_emotions(
    valence: float,
    arousal: float,
    k: int = 10,
    n: int = 3,
    return_format: str = 'letters',
    csv_path: str = None,
    random_seed: int = None,
    verbose: bool = False
) -> Union[List[str], List[Tuple[str, float]], Dict[str, float], str]:
    """
    便捷函数：直接从VA坐标获取情绪标签
    
    Args:
        valence: Valence值 [1-9]
        arousal: Arousal值 [1-9]
        k: 选择最近的K条数据
        n: 返回得分最高的N个情绪
        return_format: 'letters', 'full', 'distribution', 或 'sample'
        csv_path: CSV文件路径（可选）
        random_seed: 随机种子
        verbose: 是否打印详细信息
    
    Returns:
        根据return_format返回不同格式的结果
    
    Example:
        >>> # 返回字母
        >>> va_to_emotions(7.5, 6.2, k=10, n=3)
        ['G', 'J', 'E']
        
        >>> # 返回完整标签和分数
        >>> va_to_emotions(7.5, 6.2, k=10, n=3, return_format='full')
        [('Energizing, pump-up', 0.85), ('Joyful, cheerful', 0.72), ...]
        
        >>> # 返回完整分布
        >>> va_to_emotions(7.5, 6.2, return_format='distribution')
        {'A': 0.15, 'B': 0.08, 'C': 0.12, ...}
        
        >>> # 基于概率采样返回单个情绪
        >>> va_to_emotions(7.5, 6.2, k=10, n=3, return_format='sample')
        'Joyful, cheerful'
    """
    mapper = VAToEmotionMapper(csv_path)
    return mapper.query(
        valence, arousal, k, n, return_format, random_seed, verbose
    )


def test_emotion_distribution(
    valence: float,
    arousal: float,
    k: int = 10,
    csv_path: str = None,
    random_seed: int = None,
    threshold_k: float = 0.0
):
    """
    测试函数：打印某个VA坐标对应的情绪标签分布
    
    Args:
        valence: Valence值 [1-9]
        arousal: Arousal值 [1-9]
        k: 选择最近的K条数据
        csv_path: CSV文件路径（可选）
        random_seed: 随机种子
        threshold_k: 阈值参数，用于计算 T = μ + k·σ
                     k=0: 阈值为平均分
                     k=0.5或1.0: 更严格的阈值
    """
    # 初始化映射器
    mapper = VAToEmotionMapper(csv_path)

    # 打印输入信息
    print(f"输入: Valence={valence}, Arousal={arousal}")
    print(f"✓ 成功加载数据: {len(mapper.df)} 条记录")

    # 获取情绪分布
    dist = mapper.get_emotion_distribution(valence, arousal, k, random_seed)

    # 按分数降序排序
    sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)

    # 提取所有分数用于统计
    scores = np.array([score for _, score in sorted_dist])

    # 计算平均值和标准差
    mean_score = scores.mean()
    std_score = scores.std()

    # 计算阈值 T = μ + k·σ
    threshold = mean_score + threshold_k * std_score

    # 打印统计信息
    print(f"平均分 (μ): {mean_score:.3f}")
    print(f"标准差 (σ): {std_score:.3f}")
    print(f"阈值 (T = μ + {threshold_k}·σ): {threshold:.3f}")
    print()

    # 找出最大分数用于归一化条形图
    max_score = sorted_dist[0][1] if sorted_dist else 1.0

    # 打印分布
    print("所有情绪的分数 (降序):")
    for letter, score in sorted_dist:
        emotion_name = EMOTION_LABELS[letter]

        # 计算条形图长度（最大15个字符）
        bar_length = int((score / max_score) * 15) if max_score > 0 else 0
        bar = '█' * bar_length

        # 判断是否低于阈值
        below_threshold = score < threshold
        marker = " [低于阈值]" if below_threshold else ""

        # 打印格式：字母 (情绪名称): 分数 条形图 [标记]
        print(f"{letter} ({emotion_name:30s}): {score:.3f} {bar}{marker}")


if __name__ == "__main__":
    # # 测试代码
    # print("=" * 80)
    # print("VA坐标到情绪标签映射器 - 测试")
    # print("=" * 80)

    # # 初始化映射器
    # mapper = VAToEmotionMapper()

    # # 测试案例
    # test_cases = [
    #     (7.5, 6.2, "高Valence高Arousal - 期待激动、快乐的情绪"),
    #     (3.0, 7.0, "低Valence高Arousal - 期待紧张、愤怒的情绪"),
    #     (3.0, 3.0, "低Valence低Arousal - 期待悲伤、平静的情绪"),
    #     (7.5, 3.0, "高Valence低Arousal - 期待平静、美好的情绪"),
    #     (5.0, 5.0, "中等VA - 混合情绪"),
    # ]

    # for valence, arousal, description in test_cases:
    #     print(f"\n{'=' * 80}")
    #     print(f"测试: {description}")
    #     print(f"输入: Valence={valence}, Arousal={arousal}")
    #     print("-" * 80)

    #     # 测试不同的返回格式
    #     print("\n1. 返回字母标签 (top 3):")
    #     letters = mapper.query(
    #         valence, arousal, k=10, n=3, return_format='letters'
    #     )
    #     print(f"   结果: {letters}")
    #     for letter in letters:
    #         print(f"   {letter}: {EMOTION_LABELS[letter]}")

    #     print("\n2. 返回完整标签和分数 (top 5):")
    #     full = mapper.query(valence, arousal, k=10, n=5, return_format='full')
    #     for i, (emotion, score) in enumerate(full, 1):
    #         print(f"   {i}. {emotion:30s} - {score:.3f}")

    #     print("\n3. 完整情绪分布:")
    #     dist = mapper.query(
    #         valence, arousal, k=10, return_format='distribution'
    #     )
    #     sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    #     for letter, score in sorted_dist:
    #         if score > 0.05:  # 只显示分数大于0.05的
    #             print(
    #                 f"   {letter} ({EMOTION_LABELS[letter]:30s}): {score:.3f}"
    #             )

    # 测试新增的情绪分布可视化函数
    print("\n" + "=" * 80)
    print("测试情绪分布可视化函数 - 不同的阈值参数")
    print("=" * 80 + "\n")

    # # 测试一个示例VA坐标，使用不同的阈值
    # test_va = (2.0, 2.0)
    # threshold_values = [0.0, 0.5, 1.0]

    # for k_val in threshold_values:
    #     print(f"{'=' * 80}")
    #     print(f"阈值参数 k = {k_val}")
    #     print(f"{'=' * 80}")
    #     test_emotion_distribution(
    #         test_va[0], test_va[1], k=10, threshold_k=k_val
    #     )
    #     print()

    # 再测试几个不同的VA坐标（使用默认阈值k=0）
    threshold_k = 1
    print("\n" + "=" * 80)
    print(f"不同VA坐标的情绪分布 (k={threshold_k})")
    print("=" * 80 + "\n")

    test_distribution_cases = [(i, j) for i in [2.0, 5.0, 8.0]
                               for j in [2.0, 5.0, 8.0]]

    for valence, arousal in test_distribution_cases:
        test_emotion_distribution(
            valence, arousal, k=10, threshold_k=threshold_k
        )
        print()
