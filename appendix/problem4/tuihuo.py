import pandas as pd
import random
import math

# 读取数据并预处理
df = pd.read_csv("for_read.csv")
df["date"] = pd.to_datetime(df["date"]).dt.date
shifts = [(0, 8), (5, 13), (8, 16), (12, 20), (14, 22), (16, 24)]
shift_labels = ["Shift1", "Shift2", "Shift3", "Shift4", "Shift5", "Shift6"]
dates = sorted(df["date"].unique())

# 计算每个班次的需求量
demand_per_shift = {date: {} for date in dates}
for date in dates:
    daily_data = df[df["date"] == date]
    for label, (start, end) in zip(shift_labels, shifts):
        demand_per_shift[date][label] = daily_data[
            (daily_data["hour"] >= start) & (daily_data["hour"] < end)
        ]["value"].sum()

# 生成初始解
def initial_solution(num_employees=200, dates=dates, shift_labels=shift_labels):
    solution = {}
    for date in dates:
        solution[date] = {}
        for shift in shift_labels:
            solution[date][shift] = {'full_time': [], 'temp': 0}
        employees = list(range(num_employees))
        random.shuffle(employees)
        for employee in employees:
            chosen_shift = random.choice(shift_labels)
            solution[date][chosen_shift]['full_time'].append(employee)
    return solution

# 目标函数
def objective_function(solution, demand_per_shift, num_employees=200):
    total_cost = 0
    employee_days = {i: 0 for i in range(num_employees)}
    max_days_allowed = 30 * 0.85
    for date, shifts in solution.items():
        for shift_label, staff in shifts.items():
            num_full_time = len(staff['full_time'])
            num_temp = staff['temp']
            total_cost += num_full_time + num_temp
            for emp in staff['full_time']:
                employee_days[emp] += 1
            demand = demand_per_shift[date][shift_label]
            if 25 * num_full_time + 20 * num_temp < demand:
                total_cost += (demand - (25 * num_full_time + 20 * num_temp)) * 1000
    for days in employee_days.values():
        if days > max_days_allowed:
            total_cost += (days - max_days_allowed) * 50
    average_days = sum(employee_days.values()) / num_employees
    variance_penalty = sum((days - average_days)**2 for days in employee_days.values()) / num_employees
    total_cost += variance_penalty * 5
    return total_cost

# 生成邻域解
def get_neighbors(solution, dates, shift_labels):
    new_solution = solution.copy()
    date = random.choice(dates)
    shift1, shift2 = random.sample(shift_labels, 2)
    if new_solution[date][shift1]['full_time'] and new_solution[date][shift2]['full_time']:
        emp1 = new_solution[date][shift1]['full_time'].pop()
        emp2 = new_solution[date][shift2]['full_time'].pop()
        new_solution[date][shift1]['full_time'].append(emp2)
        new_solution[date][shift2]['full_time'].append(emp1)
    return new_solution

# 模拟退火算法
def simulated_annealing(initial_solution, dates, shift_labels, demand_per_shift):
    current_solution = initial_solution
    current_cost = objective_function(current_solution, demand_per_shift)
    temperature = 100.0
    temperature_min = 0.001
    alpha = 0.99
    while temperature > temperature_min:
        i = 1
        while i <= 100:
            new_solution = get_neighbors(current_solution, dates, shift_labels)
            new_cost = objective_function(new_solution, demand_per_shift)
            cost_diff = new_cost - current_cost
            if cost_diff < 0 or random.uniform(0, 1) < math.exp(-cost_diff / temperature):
                current_solution = new_solution
                current_cost = new_cost
            i += 1
        temperature *= alpha
    return current_solution

# 收集解决方案数据以便输出
def collect_results(solution, dates, shift_labels):
    results = []
    for date in dates:
        for shift_label in shift_labels:
            staff = solution[date][shift_label]
            for emp in staff['full_time']:
                results.append({
                    "Date": date,
                    "Shift": shift_label,
                    "Employee Type": "Full Time",
                    "Employee ID": emp
                })
            if staff['temp'] > 0:
                results.append({
                    "Date": date,
                    "Shift": shift_label,
                    "Employee Type": "Temporary",
                    "Employee ID": f"{staff['temp']} temps"  # 显示临时工的数量
                })
    return results

# 运行模拟退火
initial = initial_solution()
final_solution = simulated_annealing(initial, dates, shift_labels, demand_per_shift)

# 收集结果
results = collect_results(final_solution, dates, shift_labels)

# 创建DataFrame并保存到CSV
results_df = pd.DataFrame(results)
results_df.to_csv("tuihuoans.csv", index=False)
print("CSV file has been saved with the scheduling results.")