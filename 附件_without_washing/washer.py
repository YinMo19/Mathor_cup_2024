import pandas as pd

# 读取CSV数据
data = pd.read_csv('附件2.csv')

# 分组计算每个SC和小时的上下限
def compute_limits(group):
    q1 = group['value'].quantile(0.25)
    q3 = group['value'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound

# 应用上下限替换异常值
def replace_outliers(row, limits):
    key = (row['center'], row['hour'])
    lower_bound, upper_bound = limits[key]
    if row['value'] > upper_bound:
        row['value'] = upper_bound
    elif row['value'] < lower_bound:
        row['value'] = lower_bound
    return row

# 计算每个SC和小时组合的限制
limits = data.groupby(['center', 'hour']).apply(compute_limits).to_dict()

# 替换数据中的异常值
cleaned_data = data.apply(replace_outliers, args=(limits,), axis=1)

# 将清理后的数据保存到新的CSV文件
cleaned_data.to_csv('附件-2.csv', index=False)
