"""Upload Transcom-Group.csv to Colab's base directory (/content) first.
Pandas, NumPy, Matplotlib, Seaborn, and IPython are included in Colab.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display


def section(number, title):
    """Separate each question's output in a single notebook cell."""
    print("\n" + "-" * 80)
    print(f"{number}. {title}")
    print("-" * 80)


sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({"figure.figsize": (8, 5), "figure.dpi": 110})


# 1. Load the dataset. The reference to 'chillox' in the instructions
# is treated as a typo; this project uses Transcom-Group.csv.
section(1, "Load the Transcom Group dataset")
dataset_path = Path("/content/Transcom-Group.csv")
if not dataset_path.is_file():
    dataset_path = Path("Transcom-Group.csv")
if not dataset_path.is_file():
    raise FileNotFoundError(
        "Upload Transcom-Group.csv to Colab's Files panel (/content), "
        "then run this cell again."
    )

df = pd.read_csv(dataset_path)
print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
print("Analysis unit: one restaurant listing (one CSV row).")
print("Repeated restaurant names are retained as separate listings.")


# 2. Show the original data before cleaning.
section(2, "First 5 rows")
display(df.head())


# 3. Count missing values, including cells containing only whitespace.
section(3, "Check missing values")
df = df.replace(r"^\s*$", np.nan, regex=True)
missing_before = df.isna().sum()
display(missing_before.rename("Missing values").to_frame())
print(f"Total missing cells before cleaning: {missing_before.sum():,}")


# 4. Convert ratings such as '4.1/5' to 4.1.
# Non-numeric entries such as 'NEW' or '-' become NaN (missing), not zero.
section(4, "Clean ratings and prepare analysis columns")
original_rate = df["rate"].copy()
df["rate"] = pd.to_numeric(
    df["rate"].astype("string").str.split("/").str[0].str.strip(),
    errors="coerce",
)
df["rate"] = df["rate"].where(df["rate"].between(0, 5))

# Remove thousands separators before converting costs and votes.
cost_column = "approx_cost(for two people)"
type_column = "listed_in(type)"
for column in [cost_column, "votes"]:
    df[column] = pd.to_numeric(
        df[column].astype("string").str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )
    df[column] = df[column].where(df[column] >= 0)

df[type_column] = df[type_column].astype("string").str.strip()
df["online_order"] = (
    df["online_order"].astype("string").str.strip().str.lower()
    .map({"yes": "Yes", "no": "No"})
)

display(pd.DataFrame({"Original rating": original_rate, "Cleaned rating": df["rate"]}).head())
print(f"Cleaned rating data type: {df['rate'].dtype}")
print("Missing or invalid values after cleaning:")
display(df.isna().sum().rename("Missing values").to_frame())
print("Missing values are excluded only from calculations that need that column.")


# 5. Count listings in each restaurant type, retaining ties for first place.
section(5, "Most common restaurant type")
type_counts = df[type_column].value_counts()
display(type_counts.rename("Number of listings").to_frame())
common_types = type_counts[type_counts == type_counts.max()].index.tolist()
print(f"Most common type(s): {', '.join(common_types)} ({type_counts.max():,} listings each).")


# 6. Sum votes across all listings belonging to each restaurant type.
section(6, "Restaurant type with the highest total votes")
votes_by_type = df.groupby(type_column)["votes"].sum(min_count=1).sort_values(ascending=False)
display(votes_by_type.rename("Total votes").to_frame())
top_vote_types = votes_by_type[votes_by_type == votes_by_type.max()].index.tolist()
print(f"Highest total: {', '.join(top_vote_types)} ({votes_by_type.max():,.0f} votes each).")


# 7. Identify the highest-voted individual listing; show every tie.
section(7, "Restaurant with the highest number of votes")
highest_votes = df["votes"].max()
top_restaurants = df.loc[
    df["votes"] == highest_votes, ["name", type_column, "votes", "rate"]
]
display(top_restaurants.reset_index(drop=True))
print(f"Highest votes for a single listing: {highest_votes:,.0f}")


# 8. Show how many listings offer online ordering.
section(8, "Online ordering availability")
online_counts = df["online_order"].value_counts().reindex(["Yes", "No"], fill_value=0)
display(online_counts.rename("Number of listings").to_frame())
fig, ax = plt.subplots()
bars = ax.bar(["Yes", "No"], online_counts.values, color=["#319c91", "#ed936d"])
ax.bar_label(bars, padding=3)
ax.set(title="Online ordering availability", xlabel="Offers online ordering", ylabel="Number of listings")
ax.margins(y=0.15)
fig.tight_layout()
plt.show()
plt.close(fig)


# 9. Use a histogram to show the distribution of valid ratings.
section(9, "Distribution of restaurant ratings")
display(df["rate"].describe().round(2).rename("Rating summary").to_frame())
fig, ax = plt.subplots()
sns.histplot(df["rate"].dropna(), bins=np.linspace(0, 5, 26), color="#319c91", ax=ax)
ax.set(title="Distribution of restaurant ratings", xlabel="Rating (out of 5)", ylabel="Number of listings", xlim=(0, 5))
fig.tight_layout()
plt.show()
plt.close(fig)


# 10. The mode is the most frequently occurring cost; there may be ties.
section(10, "Most common cost for two people")
cost_counts = df[cost_column].value_counts()
common_costs = df[cost_column].mode()
display(pd.DataFrame({
    "Cost for two people": common_costs,
    "Number of listings": [int(cost_counts.loc[cost]) for cost in common_costs],
}))
print("Costs use the dataset's units; the CSV does not specify a currency.")


# 11. Compare rating distributions and print each group's mean and median.
# 'No' means online ordering is unavailable, not that a rating came offline.
section(11, "Compare ratings by online ordering availability")
rating_summary = (
    df.groupby("online_order")["rate"].agg(["count", "mean", "median"])
    .reindex(["Yes", "No"])
)
display(rating_summary.rename(columns={
    "count": "Rated listings", "mean": "Mean rating", "median": "Median rating"
}).round(2))
fig, ax = plt.subplots()
sns.boxplot(
    data=df.dropna(subset=["online_order", "rate"]),
    x="online_order", y="rate", order=["Yes", "No"], color="#77bfb5", ax=ax,
)
ax.set(title="Ratings by online ordering availability", xlabel="Offers online ordering", ylabel="Rating (out of 5)", ylim=(0, 5))
fig.tight_layout()
plt.show()
plt.close(fig)
print("The box shows the middle 50% of ratings; its center line marks the median.")


# 12. Cross-tabulate counts of restaurant types and online ordering status.
section(12, "Restaurant type and online ordering heatmap")
type_online_counts = pd.crosstab(df[type_column], df["online_order"]).reindex(
    columns=["Yes", "No"], fill_value=0
)
display(type_online_counts)
fig, ax = plt.subplots()
sns.heatmap(type_online_counts, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={"label": "Number of listings"}, ax=ax)
ax.set(title="Restaurant type vs. online ordering", xlabel="Offers online ordering", ylabel="Restaurant type")
ax.tick_params(axis="y", rotation=0)
fig.tight_layout()
plt.show()
plt.close(fig)
print("Each cell counts listings. Larger restaurant categories can have larger counts.")


# 13. Generate five conclusions from the actual results above.
section(13, "Five simple conclusions")
known_online = online_counts.sum()
online_percentage = 100 * online_counts["Yes"] / known_online
cost_text = ", ".join(f"{cost:,.0f}" for cost in common_costs)
top_names = ", ".join(top_restaurants["name"].drop_duplicates())
online_mean = rating_summary.loc["Yes", "mean"]
offline_mean = rating_summary.loc["No", "mean"]

conclusions = [
    f"{', '.join(common_types)} is the most common restaurant type, with "
    f"{type_counts.max():,} of the {len(df):,} listings.",
    f"{', '.join(top_vote_types)} has the highest combined vote total "
    f"({votes_by_type.max():,.0f}); the highest-voted individual listing "
    f"is {top_names} ({highest_votes:,.0f} votes).",
    f"{online_counts['Yes']:,} listings offer online ordering "
    f"({online_percentage:.1f}% of listings with known availability), "
    f"while {online_counts['No']:,} do not.",
    f"The most common cost(s) for two people is/are {cost_text}, "
    f"appearing in {cost_counts.max():,} listings each (dataset units).",
    f"Listings with online ordering average {online_mean:.2f}/5, "
    f"compared with {offline_mean:.2f}/5 for those without it. "
    "This comparison alone does not show that online ordering causes rating differences.",
]
for number, conclusion in enumerate(conclusions, start=1):
    print(f"{number}. {conclusion}")

print("\n" + "-" * 80)
print("Analysis complete.")
print("-" * 80)
