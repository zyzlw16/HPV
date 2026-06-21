import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
from scipy import stats


def prepare_categorical_variables(df, categorical_cols, reference_levels=None):
    """
    将分类变量转换为虚拟变量，并删除参照水平。
    （保留该函数，以便其他用途）
    """
    df = df.copy()
    for col in categorical_cols:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' 不存在于数据中。")
        df[col] = df[col].astype('category')
        levels = df[col].cat.categories.tolist()
        if reference_levels and col in reference_levels:
            ref = reference_levels[col]
            if ref not in levels:
                raise ValueError(f"参照水平 '{ref}' 不在 '{col}' 的取值中。")
        else:
            ref = levels[0]
        dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_')
        dummies = dummies.drop(columns=f"{col}_{ref}")
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df


def cox_analysis(
    df,
    time_col,
    event_col,
    group_col,
    ref_group,
    covariates=None,
    reference_levels=None,
    plot=True,
    figsize=(6, 4),
    title=None,
):
    """
    通用Cox回归分析（单因素或多因素），返回各预测变量的风险比、置信区间和p值。

    参数
    ----------
    df : pandas.DataFrame
        包含生存数据的数据框。
    time_col : str
        生存时间列名。
    event_col : str
        事件列名（1=事件，0=删失）。
    group_col : str
        分组列名（如风险组），应为类别型。
    ref_group : str
        参照组的标签。
    covariates : list, optional
        协变量列名列表。若为None，则仅进行单因素分析（只包含分组变量）。
    reference_levels : dict, optional
        指定每个分类变量的参照水平，如 {'sex': 'Male', 'stage': 'I'}。
        若未指定，则默认使用该变量第一个出现的水平作为参照。
    plot : bool, default=True
        是否绘制森林图。
    figsize : tuple, default=(6,4)
        森林图尺寸（仅当plot=True时有效）。
    title : str, optional
        森林图标题（仅当plot=True时有效）。

    返回
    -------
    pandas.DataFrame
        包含所有预测变量的风险比、95%置信区间下限/上限、p值。
    """
    df = df.copy()

    # ---------- 1. 处理分组变量 ----------
    df[group_col] = df[group_col].astype('category')
    if ref_group not in df[group_col].cat.categories:
        raise ValueError(f"参照组 '{ref_group}' 不在分组列的取值中。")

    # 生成分组虚拟变量，并删除参照组
    dummies_group = pd.get_dummies(df[group_col], prefix=group_col, prefix_sep='_', drop_first=False)
    ref_col = f"{group_col}_{ref_group}"
    if ref_col in dummies_group.columns:
        dummies_group = dummies_group.drop(columns=[ref_col])
    # 合并时间、事件和分组虚拟变量
    data_cox = pd.concat([df[[time_col, event_col]], dummies_group], axis=1)

    # ---------- 2. 处理协变量（如果有） ----------
    if covariates is not None and len(covariates) > 0:
        # 提取协变量数据
        cov_data = df[covariates].copy()

        # 识别分类变量（object 或 category 类型）
        categorical_covs = [col for col in covariates if df[col].dtype.name in ['object', 'category']]

        # 对每个分类变量生成虚拟变量，删除参照水平
        for col in categorical_covs:
            # 确定参照水平
            if reference_levels and col in reference_levels:
                ref = reference_levels[col]
            else:
                # 默认使用该变量的第一个水平
                ref = df[col].astype('category').cat.categories[0]

            # 生成虚拟变量，并删除参照列
            dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_', drop_first=False)
            ref_col_cov = f"{col}_{ref}"
            if ref_col_cov in dummies.columns:
                dummies = dummies.drop(columns=[ref_col_cov])

            # 将虚拟变量加入 cov_data，并删除原列
            cov_data = pd.concat([cov_data.drop(columns=[col]), dummies], axis=1)

        # 将处理后的协变量合并到 data_cox
        data_cox = pd.concat([data_cox, cov_data], axis=1)

    # ---------- 3. 拟合 Cox 模型 ----------
    cph = CoxPHFitter()
    cph.fit(data_cox, duration_col=time_col, event_col=event_col)

    # ---------- 4. 提取结果 ----------
    summary = cph.summary.copy()
    # 计算 HR 和 95% CI
    summary['HR'] = np.exp(summary['coef'])
    summary['CI_lower'] = np.exp(summary['coef'] - 1.96 * summary['se(coef)'])
    summary['CI_upper'] = np.exp(summary['coef'] + 1.96 * summary['se(coef)'])
    result = summary[['HR', 'CI_lower', 'CI_upper', 'p']].round(4)
    result.columns = ['Hazard Ratio', '95% CI Lower', '95% CI Upper', 'p-value']

    # 如果有多因素，打印模型整体信息
    if covariates is not None and len(covariates) > 0:
        print("\n多因素Cox模型概况:")
        print(f"Concordance: {cph.concordance_index_:.3f}")
        print(f"Partial AIC: {cph.AIC_partial_:.2f}")
        print(f"Log-likelihood: {cph.log_likelihood_:.2f}")

    # ---------- 5. 绘制森林图（可选） ----------
    if plot:
        plot_cox_forest_lifelines(cph, figsize=figsize, title=title)

    return result


def plot_hr_by_score(df, time_col, event_col, score_col, n_groups=10, reference='first'):
    """
    将得分分箱，计算每组相对于参考组的HR，绘制HR随得分变化的曲线，
    并在横坐标上标记Q1、中位数、Q3的位置。

    参数：
    df : pandas.DataFrame
        包含生存时间、事件和得分的数据框。
    time_col : str
        生存时间列名。
    event_col : str
        事件列名（1=事件，0=删失）。
    score_col : str
        得分列名（连续变量）。
    n_groups : int, default=10
        分组数量（基于分位数等分）。
    reference : {'first', 'last'}, default='first'
        指定参考组为最低得分组('first')或最高得分组('last')。

    返回：
    cph : lifelines.CoxPHFitter
        拟合的Cox模型对象。
    summary : pandas.DataFrame
        各组HR及置信区间。
    group_medians : pandas.Series
        各组的得分中位数（横坐标值）。
    """
    data = df.copy()

    # 按分位数分组（避免重复边界）
    data['score_group'] = pd.qcut(data[score_col], q=n_groups, duplicates='drop')
    # 生成有序标签
    group_labels = [f'G{i}' for i in range(len(data['score_group'].cat.categories))]
    data['score_group'] = pd.qcut(data[score_col], q=n_groups, labels=group_labels, duplicates='drop')

    # 确定参考组
    if reference == 'first':
        ref_group = group_labels[0]
    else:
        ref_group = group_labels[-1]

    # 生成虚拟变量，删除参考组
    dummies = pd.get_dummies(data['score_group'], prefix='group', drop_first=False)
    ref_col = f'group_{ref_group}'
    if ref_col in dummies.columns:
        dummies = dummies.drop(columns=[ref_col])

    # 准备Cox数据
    cox_data = pd.concat([data[[time_col, event_col]], dummies], axis=1)
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col=time_col, event_col=event_col)

    # 提取结果
    summary = cph.summary.copy()
    summary['HR'] = np.exp(summary['coef'])
    summary['CI_lower'] = np.exp(summary['coef'] - 1.96 * summary['se(coef)'])
    summary['CI_upper'] = np.exp(summary['coef'] + 1.96 * summary['se(coef)'])

    # 计算各组的得分中位数作为横坐标
    group_medians = data.groupby('score_group')[score_col].median()
    # 确保顺序与summary一致（summary的索引是 'group_G1' 等）
    group_medians = group_medians.loc[[idx.replace('group_', '') for idx in summary.index]]

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(group_medians, summary['HR'],
                yerr=[summary['HR'] - summary['CI_lower'],
                      summary['CI_upper'] - summary['HR']],
                fmt='o-', capsize=4, color='blue', ecolor='blue', markersize=6)
    ax.axhline(1, linestyle='--', color='gray', alpha=0.7, label='HR=1')

    # 计算整个数据集中得分的Q1、中位数、Q3
    q1 = data[score_col].quantile(0.25)
    median = data[score_col].quantile(0.5)
    q3 = data[score_col].quantile(0.75)
    ax.axvline(q1, linestyle=':', color='red', alpha=0.7, label=f'Q1 = {q1:.2f}')
    ax.axvline(median, linestyle=':', color='green', alpha=0.7, label=f'Median = {median:.2f}')
    ax.axvline(q3, linestyle=':', color='orange', alpha=0.7, label=f'Q3 = {q3:.2f}')

    ax.set_xlabel(score_col)
    ax.set_ylabel('Hazard Ratio')
    ax.set_title(f'Hazard Ratio by {score_col} (grouped into {n_groups} groups)')
    ax.legend()
    plt.tight_layout()
    plt.show()

    return cph, summary, group_medians

def plot_cox_forest_lifelines(cph_model, figsize=(6, 4), title=None):
    """
    使用 lifelines 内置的 plot 方法绘制 Cox 模型的风险比森林图。
    """
    fig, ax = plt.subplots(figsize=figsize)
    cph_model.plot(hazard_ratios=True, ax=ax)
    ax.axvline(1, linestyle='--', color='gray', alpha=0.7)
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Hazard Ratios from Cox Model')
    plt.tight_layout()
    plt.show()
    return ax

def likelihood_ratio_test(model_full, model_reduced):
    """
    计算两个Cox模型的似然比检验。
    参数
    ----------
    model_full : CoxPHFitter 拟合对象
        包含所有预测变量的模型。
    model_reduced : CoxPHFitter 拟合对象
        嵌套于全模型的简化模型。
    返回
    -------
    lrt_stat : float
        似然比统计量。
    p_value : float
        对应的p值（自由度 = 参数个数差）。
    """
    ll_full = model_full.log_likelihood_
    ll_reduced = model_reduced.log_likelihood_
    df_diff = model_full._n_features - model_reduced._n_features
    lrt_stat = 2 * (ll_full - ll_reduced)
    p = 1 - stats.chi2.cdf(lrt_stat, df_diff)
    return lrt_stat, p


# ===================== 示例用法 =====================
if __name__ == "__main__":
    # 创建模拟数据（同前）
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'time': np.random.exponential(5, n),
        'event': np.random.binomial(1, 0.6, n),
        'risk_group': np.random.choice(['High', 'Mid', 'Low'], n, p=[0.2,0.3,0.5]),
        'age': np.random.normal(60, 10, n),
        'sex': np.random.choice(['Male', 'Female'], n),
        'stage': np.random.choice(['I', 'II', 'III', 'IV'], n, p=[0.2,0.3,0.3,0.2]),
        'HPV_status': np.random.choice(['Positive', 'Negative'], n, p=[0.4,0.6])
    })

    # 1. 单因素分析（仅风险组）
    print("="*50)
    print("单因素Cox (以Low组为参照)")
    uni_res = cox_analysis(df, time_col='time', event_col='event',
                           group_col='risk_group', ref_group='Low',
                           covariates=None, plot=True)
    print(uni_res)

    # 2. 多因素分析（风险组 + 年龄 + 性别 + 分期）
    print("\n" + "="*50)
    print("多因素Cox (调整年龄、性别、分期)")
    multi_res = cox_analysis(df, time_col='time', event_col='event',
                             group_col='risk_group', ref_group='Low',
                             covariates=['age', 'sex', 'stage'],
                             reference_levels={'sex': 'Male', 'stage': 'I'},
                             plot=True, title="Multivariate Cox Analysis")
    print(multi_res)

    # 3. HPV 状态单因素
    print("\n" + "="*50)
    print("HPV 状态单因素")
    df['HPV_status'] = df['HPV_status'].astype('category')
    hpv_uni = cox_analysis(df, time_col='time', event_col='event',
                           group_col='HPV_status', ref_group='Negative',
                           covariates=None, plot=False)
    print(hpv_uni)

    # 4. HPV 状态多因素
    print("\n" + "="*50)
    print("HPV 状态多因素 (调整年龄、性别、分期)")
    hpv_multi = cox_analysis(df, time_col='time', event_col='event',
                             group_col='HPV_status', ref_group='Negative',
                             covariates=['age', 'sex', 'stage'],
                             reference_levels={'sex': 'Male', 'stage': 'I'},
                             plot=True, title="HPV Multivariate Cox")
    print(hpv_multi)

    # 可选：演示 plot_hr_by_score 的使用（注释掉以免干扰原示例）
    # print("\n" + "="*50)
    # print("演示 plot_hr_by_score (使用 pred_score 列，此处用 risk_group 的编码代替)")
    # # 为演示创建一个得分列（模拟）
    # df['pred_score'] = np.random.uniform(0, 1, n)
    # cph, hr_summary, medians = plot_hr_by_score(
    #     df, time_col='time', event_col='event', score_col='pred_score',
    #     n_groups=8, reference='first'
    # )
    # print(hr_summary)



