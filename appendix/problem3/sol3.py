import pandas as pd

# Load the provided CSV file to examine its structure
file_path = "../problem2/predicted_future_volumes_hours.csv"
predicted_volumes = pd.read_csv(file_path)
# Convert the date-time string to a datetime object
predicted_volumes["日期时间"] = pd.to_datetime(predicted_volumes["日期时间"])

# Define the shift times
shift_times = {
    "00:00-08:00": ("00:00", "08:00"),
    "05:00-13:00": ("05:00", "13:00"),
    "08:00-16:00": ("08:00", "16:00"),
    "12:00-20:00": ("12:00", "20:00"),
    "14:00-22:00": ("14:00", "22:00"),
    "16:00-24:00": ("16:00", "24:00"),
}


# Create a function to assign each hour to a shift
def assign_shift(hour):
    for shift, (start, end) in shift_times.items():
        if start <= hour.strftime("%H:%M") < end:
            return shift
    return None


# Apply the function to assign shifts
predicted_volumes["班次"] = predicted_volumes["日期时间"].apply(
    lambda x: assign_shift(x)
)

# Group by center, date, and shift to sum up volumes
grouped_volumes = (
    predicted_volumes.groupby(
        ["分拣中心", predicted_volumes["日期时间"].dt.date, "班次"]
    )["货量"]
    .sum()
    .reset_index()
)
grouped_volumes.rename(columns={"日期时间": "日期"}, inplace=True)

for special_SC in grouped_volumes["分拣中心"].unique():
    pre_grouped_volumes = grouped_volumes[grouped_volumes["分拣中心"] == special_SC]

    import pulp
    from pulp import PULP_CBC_CMD

    # Set up the linear programming problem to minimize the total number of person-days
    model = pulp.LpProblem("Staff_Scheduling", pulp.LpMinimize)

    # Decision variables
    # Number of regular and temporary workers per shift, per day, per sorting center
    regular_workers = pulp.LpVariable.dicts(
        "Regular",
        [
            (center, date, shift)
            for center, date, shift in zip(
                pre_grouped_volumes["分拣中心"],
                pre_grouped_volumes["日期"],
                pre_grouped_volumes["班次"],
            )
        ],
        lowBound=0,
        cat="Integer",
    )
    temporary_workers = pulp.LpVariable.dicts(
        "Temporary",
        [
            (center, date, shift)
            for center, date, shift in zip(
                pre_grouped_volumes["分拣中心"],
                pre_grouped_volumes["日期"],
                pre_grouped_volumes["班次"],
            )
        ],
        lowBound=0,
        cat="Integer",
    )

    # Objective: Minimize the total person-days
    model += pulp.lpSum(
        [
            regular_workers[center, date, shift]
            + temporary_workers[center, date, shift]
            for center, date, shift in zip(
                pre_grouped_volumes["分拣中心"],
                pre_grouped_volumes["日期"],
                pre_grouped_volumes["班次"],
            )
        ]
    )

    # Constraints
    # Each shift's staffing needs must meet the volume requirements
    for i, row in pre_grouped_volumes.iterrows():
        center, date, shift, volume = (
            row["分拣中心"],
            row["日期"],
            row["班次"],
            row["货量"],
        )
        model += (
            25 * regular_workers[center, date, shift]
            + 20 * temporary_workers[center, date, shift]
            >= volume
        )

    # Maximum of 60 regular workers per sorting center per day
    for (center, date), group in pre_grouped_volumes.groupby(["分拣中心", "日期"]):
        model += (
            pulp.lpSum(regular_workers[center, date, shift] for shift in group["班次"])
            <= 60
        )

    # Solve the model
    model.solve(PULP_CBC_CMD(msg=1, threads=8, gapRel=0.01))
    # model.solve()

    # Output results
    results = []
    for center, date, shift in zip(
        pre_grouped_volumes["分拣中心"], pre_grouped_volumes["日期"], pre_grouped_volumes["班次"]
    ):
        result = {
            "分拣中心": center,
            "日期": date,
            "班次": shift,
            "正式工": regular_workers[center, date, shift].varValue,
            "临时工": temporary_workers[center, date, shift].varValue,
        }
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"sol3_/{special_SC}scheduling_results.csv", index=False)
