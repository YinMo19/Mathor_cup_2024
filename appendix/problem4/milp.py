import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpInteger, LpBinary, PULP_CBC_CMD

# 读取数据
df = pd.read_csv("path_to_read.csv")
df['date'] = pd.to_datetime(df['date']).dt.date

# 设定班次及每个班次的时间段
shifts = [(0, 8), (5, 13), (8, 16), (12, 20), (14, 22), (16, 24)]
shift_labels = ['Shift1', 'Shift2', 'Shift3', 'Shift4', 'Shift5', 'Shift6']
dates = sorted(df['date'].unique())

# 计算每个班次的需求
demand_per_shift = {date: {} for date in dates}
for date in dates:
    daily_data = df[df['date'] == date]
    for label, (start, end) in zip(shift_labels, shifts):
        demand_per_shift[date][label] = daily_data[(daily_data['hour'] >= start) & (daily_data['hour'] < end)]['value'].sum()

# 建立问题
model = LpProblem("Personnel_Scheduling", LpMaximize)

# 定义变量
x = LpVariable.dicts("FullTime", [(date, shift) for date in dates for shift in shift_labels], lowBound=0, upBound=1, cat=LpBinary)
y = LpVariable.dicts("Temp", [(date, shift) for date in dates for shift in shift_labels], lowBound=0, cat=LpInteger)

# 满足每个班次需求的约束
for date in dates:
    for shift in shift_labels:
        model += 25 * lpSum(x[(date, shift)] * i for i in range(200)) + 20 * y[(date, shift)] >= demand_per_shift[date][shift]

# 正式工的出勤率不超过85%
for i in range(200):
    model += lpSum(x[(date, shift)] for date in dates for shift in shift_labels) <= 25

# 解决问题
model.solve(PULP_CBC_CMD(msg=0))

# 收集结果并输出为CSV
results = []
for date in dates:
    for shift in shift_labels:
        for i in range(200):
            if x[(date, shift)].varValue * i > 0:
                results.append({"Sorting_Center": "SC60", "Date": date, "Shift": shift, "Employee": f"FullTime({i})"})
        temp_workers = int(y[(date, shift)].varValue)
        for j in range(temp_workers):
            results.append({"Sorting_Center": "SC60", "Date": date, "Shift": shift, "Employee": f"Temp({j})"})

# 创建DataFrame并保存到CSV
results_df = pd.DataFrame(results)
results_df.to_csv("scheduling_results.csv", index=False)
print("CSV file has been saved with the scheduling results.")
