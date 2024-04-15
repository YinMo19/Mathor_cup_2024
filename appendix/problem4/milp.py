import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

# 读取数据
df = pd.read_csv("read.csv")
df["date"] = pd.to_datetime(df["date"]).dt.date

# 班次及时间段
shifts = [(0, 8), (5, 13), (8, 16), (12, 20), (14, 22), (16, 24)]
shift_labels = ["Shift1", "Shift2", "Shift3", "Shift4", "Shift5", "Shift6"]
dates = sorted(df["date"].unique())

# 计算每个班次的需求
demand_per_shift = {date: {} for date in dates}
for date in dates:
    daily_data = df[df["date"] == date]
    for label, (start, end) in zip(shift_labels, shifts):
        demand_per_shift[date][label] = daily_data[
            (daily_data["hour"] >= start) & (daily_data["hour"] < end)
        ]["value"].sum()

# 建立优化模型
model = LpProblem("Personnel_Scheduling", LpMinimize)

# 定义变量
full_time = LpVariable.dicts(
    "FullTime", (dates, shift_labels, range(200)), 0, 1, LpBinary
)
temp_workers = LpVariable.dicts(
    "TempWorkers", (dates, shift_labels), 0, None, cat="Integer"
)

# 目标函数：最小化总人天数
model += lpSum(
    full_time[date][shift][i]
    for date in dates
    for shift in shift_labels
    for i in range(200)
) + lpSum(temp_workers[date][shift] for date in dates for shift in shift_labels)

# 每个班次的需求必须被满足的约束
for date in dates:
    for shift in shift_labels:
        model += (
            25 * lpSum(full_time[date][shift][i] for i in range(200))
            + 20 * temp_workers[date][shift]
            >= demand_per_shift[date][shift]
        )

# 正式工的出勤率不超过85%
for i in range(200):
    model += (
        lpSum(full_time[date][shift][i] for date in dates for shift in shift_labels)
        <= 30 * 0.85
    )

# 正式工连续出勤天数不超过7天
for i in range(200):
    for d in range(len(dates) - 6):
        model += (
            lpSum(
                full_time[dates[d + k]][shift][i]
                for k in range(7)
                for shift in shift_labels
            )
            <= 7
        )
# 求解问题
# model.solve(PULP_CBC_CMD(msg=1))
model.solve(PULP_CBC_CMD(msg=1, threads=8,gapRel=0.001))
print("soolved")
# 收集结果并输出为CSV
results = []
for date in dates:
    for shift in shift_labels:
        for i in range(200):
            if full_time[date][shift][i].varValue > 0:
                results.append(
                    {
                        "Sorting_Center": "SC60",
                        "Date": date,
                        "Shift": shift,
                        "Employee": f"FullTime({i})",
                    }
                )
        temp_workers_count = int(temp_workers[date][shift].varValue)
        for j in range(temp_workers_count):
            results.append(
                {
                    "Sorting_Center": "SC60",
                    "Date": date,
                    "Shift": shift,
                    "Employee": f"Temp({j})",
                }
            )

# 创建DataFrame并保存到CSV
results_df = pd.DataFrame(results)
results_df.to_csv("scheduling_results.csv", index=False)
print("CSV file has been saved with the scheduling results.")
