#%%
import pandas as pd
import calendar

df = pd.read_parquet("data.parquet")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)
df.insert(0, "timestamp", df.index)
df["count"] = 1

df['words'] = df['message'].apply(lambda x:len(x.split(" ")))
df['weekday'] = df['timestamp'].dt.day_name()

df_weekdays = df.groupby(["author", "weekday"]).sum(numeric_only=True).reset_index()
cats = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]
df_weekdays["weekday"] = pd.Categorical(df_weekdays["weekday"], cats)
df_weekdays = df_weekdays.sort_values("weekday")

# Counts
df_counts = df.drop(columns="timestamp").reset_index()
df_counts["month_date"] = pd.to_datetime(
    df_counts["timestamp"].dt.strftime("%Y-%m")
)

at_df = df_counts[df_counts.author == "Antonio"]
ph_df = df_counts[df_counts.author == "Perlei"]

# Monthly stats by month name
df_months = (
    df_counts.groupby(["author", df_counts.timestamp.dt.month])["words"]
    .sum()
    .reset_index()
)
df_months["timestamp"] = df_months["timestamp"].apply(
    lambda x: calendar.month_name[x]
)

# Monthly dates stats
love_words_per_month_date = (
    df_counts.groupby(["author", df_counts.month_date.dt.date])["words"]
    .sum()
    .reset_index()
)

# Monthly stats
love_words_per_month = df_counts.groupby(df_counts.timestamp.dt.month)[
    "words"
].sum()
at_words_per_month = at_df.groupby(at_df.timestamp.dt.month)["words"].sum()
ph_words_per_month = ph_df.groupby(ph_df.timestamp.dt.month)["words"].sum()

# Daily dates stats
love_words_per_day_date = (
    df_counts.groupby(["author", df_counts.timestamp.dt.date])["words"]
    .sum()
    .reset_index()
)

# Daily stats
love_words_per_day = df_counts.groupby(df_counts.timestamp.dt.date)["words"].sum()
at_words_per_day = at_df.groupby(at_df.timestamp.dt.date)["words"].sum()
ph_words_per_day = ph_df.groupby(ph_df.timestamp.dt.date)["words"].sum()

love_max_words_per_day = love_words_per_day.max()
love_date_max = love_words_per_day.idxmax().strftime("%d/%m/%Y")
at_max_words_per_day = at_words_per_day.max()
at_date_max = at_words_per_day.idxmax().strftime("%d/%m/%Y")
ph_max_words_per_day = ph_words_per_day.max()
ph_date_max = ph_words_per_day.idxmax().strftime("%d/%m/%Y")

love_min_words_per_day = love_words_per_day.min()
love_date_min = love_words_per_day.idxmin().strftime("%d/%m/%Y")
at_min_words_per_day = at_words_per_day.min()
at_date_min = at_words_per_day.idxmin().strftime("%d/%m/%Y")
ph_min_words_per_day = ph_words_per_day.min()
ph_date_min = ph_words_per_day.idxmin().strftime("%d/%m/%Y")

love_mean_words_per_day = round(love_words_per_day.mean(), 2)
at_mean_words_per_day = round(at_words_per_day.mean(), 2)
ph_mean_words_per_day = round(ph_words_per_day.mean(), 2)

metrics_dict = {
    "max": {
        "value": {
            "love": love_max_words_per_day,
            "AT": at_max_words_per_day,
            "PH": ph_max_words_per_day,
        },
        "date": {"love": love_date_max, "AT": at_date_max, "PH": ph_date_max},
    },
    "min": {
        "value": {
            "love": love_min_words_per_day,
            "AT": at_min_words_per_day,
            "PH": ph_min_words_per_day,
        },
        "date": {"love": love_date_min, "AT": at_date_min, "PH": ph_date_min},
    },
    "mean": {
        "value": {
            "love": love_mean_words_per_day,
            "AT": at_mean_words_per_day,
            "PH": ph_mean_words_per_day,
        },
        "date": {"love": "", "AT": "", "PH": ""},
    },
}
result = {
        "df": df,
        "df_weekdays": df_weekdays,
        "df_counts": df_counts,
        "df_months": df_months,
        "love_words_per_month_date": love_words_per_month_date,
        "love_words_per_month": love_words_per_month,
        "at_words_per_month": at_words_per_month,
        "ph_words_per_month": ph_words_per_month,
        "love_words_per_day_date": love_words_per_day_date,
        "love_words_per_day": love_words_per_day,
        "at_words_per_day": at_words_per_day,
        "ph_words_per_day": ph_words_per_day,
        "love_max_words_per_day": love_max_words_per_day,
        "love_date_max": love_date_max,
        "at_max_words_per_day": at_max_words_per_day,
        "at_date_max": at_date_max,
        "ph_max_words_per_day": ph_max_words_per_day,
        "ph_date_max": ph_date_max,
        "love_min_words_per_day": love_min_words_per_day,
        "love_date_min": love_date_min,
        "at_min_words_per_day": at_min_words_per_day,
        "at_date_min": at_date_min,
        "ph_min_words_per_day": ph_min_words_per_day,
        "ph_date_min": ph_date_min,
        "love_mean_words_per_day": love_mean_words_per_day,
        "at_mean_words_per_day": at_mean_words_per_day,
        "ph_mean_words_per_day": ph_mean_words_per_day,
        "metrics_dict": metrics_dict,
    }

# %%
