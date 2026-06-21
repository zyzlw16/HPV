"""
km_utils.py
通用Kaplan-Meier生存曲线绘制与log-rank检验工具
"""

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import pandas as pd
import numpy as np


def plot_km_curves(
    data,
    time_col="time",
    event_col="event",
    group_col="risk_group",
    title="KM curves by group",
    xlabel="Follow-up time (years)",
    ylabel="Overall survival probability",
    ci_show=True,
    figsize=(6, 5),
    return_stats=False,
    pairwise=False,
):
    """
    绘制Kaplan-Meier生存曲线，并根据组数自动进行log-rank检验。

    参数
    ----------
    data : pandas.DataFrame
        包含生存时间、事件指示符和分组列的数据框。
    time_col : str, default='time'
        生存时间列名。
    event_col : str, default='event'
        事件列名（1表示事件发生，0表示删失）。
    group_col : str, default='risk_group'
        分组列名（可以是数值或类别）。
    title : str, default='KM curves by group'
        图表标题。
    xlabel : str, default='Follow-up time (years)'
        x轴标签。
    ylabel : str, default='Overall survival probability'
        y轴标签。
    ci_show : bool, default=True
        是否显示置信区间。
    figsize : tuple, default=(6,5)
        图表尺寸。
    return_stats : bool, default=False
        是否返回统计检验结果字典。
    pairwise : bool, default=False
        当组数>2时，是否计算两两组间的log-rank检验（仅在return_stats=True时有效）。

    返回
    -------
    dict (可选)
        当return_stats=True时，返回包含以下键的字典：
        - 'n_groups': 组数
        - 'group_names': 组名列表
        - 'overall_p': 多组整体log-rank检验的p值（组数>2时存在）
        - 'overall_statistic': 整体检验统计量
        - 'overall_dof': 整体检验自由度
        - 'pairwise_p': 两两比较的p值矩阵（当pairwise=True且组数>2时）
        - 'pairwise_stats': 两两比较的统计量矩阵
        - 'pairwise_dof': 两两比较的自由度矩阵
        若组数==2，则返回：
        - 'p_value': 两组log-rank检验p值
        - 'test_statistic': 检验统计量
        - 'degrees_of_freedom': 自由度
    """
    # 确保分组列存在
    if group_col not in data.columns:
        raise ValueError(f"分组列 '{group_col}' 不存在于数据中。")

    # 检查必要列
    for col in [time_col, event_col]:
        if col not in data.columns:
            raise ValueError(f"列 '{col}' 不存在于数据中。")

    # 获取分组信息
    groups = data[group_col].unique()
    n_groups = len(groups)
    if n_groups < 2:
        raise ValueError("分组数必须至少为2。")

    # 绘制KM曲线
    plt.figure(figsize=figsize)
    kmf = KaplanMeierFitter()

    for name in sorted(groups):  # 排序使图例有序
        subset = data[data[group_col] == name]
        kmf.fit(
            durations=subset[time_col],
            event_observed=subset[event_col],
            label=str(name),
        )
        kmf.plot_survival_function(ci_show=ci_show)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 统计检验
    if return_stats:
        stats = {}
        if n_groups == 2:
            group_a = data[data[group_col] == groups[0]]
            group_b = data[data[group_col] == groups[1]]
            lr = logrank_test(
                group_a[time_col],
                group_b[time_col],
                event_observed_A=group_a[event_col],
                event_observed_B=group_b[event_col],
            )
            stats["p_value"] = lr.p_value
            stats["test_statistic"] = lr.test_statistic
            stats["degrees_of_freedom"] = lr.degrees_of_freedom
            # 打印结果
            print(f"\nLog-rank test (2 groups): p = {lr.p_value:.4e}")
        else:
            # 多组整体检验
            mlr = multivariate_logrank_test(
                data[time_col], data[group_col], data[event_col]
            )
            stats["overall_p"] = mlr.p_value
            stats["overall_statistic"] = mlr.test_statistic
            stats["overall_dof"] = mlr.degrees_of_freedom
            print(f"\nMultivariate log-rank test (overall): p = {mlr.p_value:.4e}")
            print(mlr.summary)

            # 两两比较（可选）
            if pairwise:
                group_names = sorted(groups)
                n = len(group_names)
                p_mat = np.zeros((n, n))
                stat_mat = np.zeros((n, n))
                dof_mat = np.zeros((n, n))
                for i in range(n):
                    for j in range(i + 1, n):
                        g1 = data[data[group_col] == group_names[i]]
                        g2 = data[data[group_col] == group_names[j]]
                        lr = logrank_test(
                            g1[time_col],
                            g2[time_col],
                            event_observed_A=g1[event_col],
                            event_observed_B=g2[event_col],
                        )
                        p_mat[i, j] = p_mat[j, i] = lr.p_value
                        stat_mat[i, j] = stat_mat[j, i] = lr.test_statistic
                        dof_mat[i, j] = dof_mat[j, i] = lr.degrees_of_freedom
                        print(
                            f"  {group_names[i]} vs {group_names[j]}: p = {lr.p_value:.4e}"
                        )
                stats["pairwise_p"] = p_mat
                stats["pairwise_stats"] = stat_mat
                stats["pairwise_dof"] = dof_mat

        return stats


def create_groups_by_quantiles(data, score_col, q=[0, 1/3, 2/3, 1], labels=None):
    """
    根据连续变量的分位数创建分组列（辅助函数）。

    参数
    ----------
    data : DataFrame
        包含连续变量的数据框。
    score_col : str
        连续变量列名（如预测得分）。
    q : list, default=[0, 1/3, 2/3, 1]
        分位数列表，长度至少为2。例如 [0, 0.5, 1] 对应中位数二分。
    labels : list, optional
        分组标签，长度必须等于 len(q)-1。若为None，则自动生成如 'Q1', 'Q2', ... 的标签。

    返回
    -------
    pandas.Series
        分组标签序列，可直接赋值给数据框的新列。
    """
    if len(q) < 3:
        raise ValueError("q 必须至少包含两个分位数。")

    if labels is None:
        labels = [f"Group {i+1}" for i in range(len(q) - 1)]

    if len(q) == 3:
        groups = np.where(data[score_col] < q[1], labels[0], labels[1])
    else:
        # 计算实际分位数值
        quantiles = data[score_col].quantile(q).values
        # 使用 pd.cut 进行分组
        groups = pd.cut(data[score_col], bins=quantiles, labels=labels, include_lowest=True)

    return groups


def print_survival_rates(
    data,
    time_col="time",
    event_col="event",
    group_col="risk_group",
    time_point=5,
    ci=True,
    as_percent=True,
    digits=0,
):
    """
    计算并打印指定时间点各组的生存率及其置信区间。

    参数
    ----------
    data : pandas.DataFrame
        包含生存时间、事件指示符和分组列的数据框。
    time_col : str, default='time'
        生存时间列名。
    event_col : str, default='event'
        事件列名（1表示事件发生，0表示删失）。
    group_col : str, default='risk_group'
        分组列名。
    time_point : float, default=5
        生存率的时间点（如5年）。
    ci : bool, default=True
        是否打印置信区间。
    as_percent : bool, default=True
        是否将生存率格式化为百分比。
    digits : int, default=0
        小数点后保留的位数（百分比时为整数位数）。

    返回
    -------
    pandas.DataFrame
        包含每组生存率和置信区间的数据框，便于后续使用。
    """
    groups = data[group_col].unique()
    results = []
    print(f"\n{time_point}-year survival rates by {group_col}:")
    for name in groups:
        subset = data[data[group_col] == name]
        kmf = KaplanMeierFitter()
        kmf.fit(subset[time_col], event_observed=subset[event_col])

        # 找到最接近 time_point 的时间点的生存率
        times = kmf.survival_function_.index.values
        if time_point > times.max():
            print(f"  警告：分组 '{name}' 的最大随访时间 {times.max():.2f} < {time_point}，使用最后一个时间点。")
            idx = -1
        else:
            idx = np.searchsorted(times, time_point, side='right') - 1
            if idx < 0:
                idx = 0

        surv = kmf.survival_function_.iloc[idx, 0]
        if ci:
            ci_df = kmf.confidence_interval_
            ci_lower = ci_df.iloc[idx, 0]
            ci_upper = ci_df.iloc[idx, 1]
        else:
            ci_lower = ci_upper = None

        if as_percent:
            surv_str = f"{surv*100:.{digits}f}%"
            ci_lower_str = f"{ci_lower*100:.{digits}f}%" if ci else ""
            ci_upper_str = f"{ci_upper*100:.{digits}f}%" if ci else ""
        else:
            surv_str = f"{surv:.{digits}f}"
            ci_lower_str = f"{ci_lower:.{digits}f}" if ci else ""
            ci_upper_str = f"{ci_upper:.{digits}f}" if ci else ""

        if ci:
            print(f"  {name}: {surv_str} [{ci_lower_str}–{ci_upper_str}]")
        else:
            print(f"  {name}: {surv_str}")

        results.append({
            'Group': name,
            f'{time_point}-year survival': surv_str,
            'CI lower': ci_lower_str,
            'CI upper': ci_upper_str
        })

    return pd.DataFrame(results)

# 使用示例（可注释掉，仅供演示）
if __name__ == "__main__":
    # 假设已有数据框 df_all 包含 time, event, pred_score
    # 示例：创建二分和三分组，并调用绘图函数
    pass