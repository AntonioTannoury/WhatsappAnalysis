#%%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import calplot
from cv2 import COLOR_BGR2RGB, cvtColor, imread
import calendar
from st_aggrid import GridOptionsBuilder, AgGrid
import streamlit as st
from os import listdir
from os.path import isfile, join
import random
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from PIL import Image
import numpy as np

def load_data():
    df = pd.read_csv("data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.insert(0, "timestamp", df.index)
    df["count"] = 1
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

    return {
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


data_params = load_data()
df = data_params["df"]
df_weekdays = data_params["df_weekdays"]
df_counts = data_params["df_counts"]
df_months = data_params["df_months"]
love_words_per_month_date = data_params["love_words_per_month_date"]
love_words_per_month = data_params["love_words_per_month"]
at_words_per_month = data_params["at_words_per_month"]
ph_words_per_month = data_params["ph_words_per_month"]
love_words_per_day_date = data_params["love_words_per_day_date"]
love_words_per_day = data_params["love_words_per_day"]
at_words_per_day = data_params["at_words_per_day"]
ph_words_per_day = data_params["ph_words_per_day"]
love_max_words_per_day = data_params["love_max_words_per_day"]
love_date_max = data_params["love_date_max"]
at_max_words_per_day = data_params["at_max_words_per_day"]
at_date_max = data_params["at_date_max"]
ph_max_words_per_day = data_params["ph_max_words_per_day"]
ph_date_max = data_params["ph_date_max"]
love_min_words_per_day = data_params["love_min_words_per_day"]
love_date_min = data_params["love_date_min"]
at_min_words_per_day = data_params["at_min_words_per_day"]
at_date_min = data_params["at_date_min"]
ph_min_words_per_day = data_params["ph_min_words_per_day"]
ph_date_min = data_params["ph_date_min"]
love_mean_words_per_day = data_params["love_mean_words_per_day"]
at_mean_words_per_day = data_params["at_mean_words_per_day"]
ph_mean_words_per_day = data_params["ph_mean_words_per_day"]
metrics_dict = data_params["metrics_dict"]

#%%
def text_format(color, name, header):
    title = f"<{header} style='text-align: center; color: {color};'>{name}</{header}>"
    return title

@st.cache_data
def filter_df(
    name=["Antonio", "Perlei"],
    date_min=df.timestamp.min().strftime("%Y/%m/%d"),
    date_max=df.timestamp.max().strftime("%Y/%m/%d"),
):
    df_filtered = df[
        (df.author.isin(name)) & (df.index.to_series().between(date_min, date_max))
    ].sort_index()

    df_filtered.columns = [i.title() for i in df_filtered.columns]
    df_filtered = df_filtered.drop(columns=['Unnamed: 0','Count',"Hour"])
    return df_filtered

@st.cache_data
def ad_grid(data, height = 400):
    gb = GridOptionsBuilder.from_dataframe(data)
    gb.configure_pagination(paginationAutoPageSize=True, enabled=True)  # Add pagination
    gb.configure_side_bar()  # Add a sidebar
    gridOptions = gb.build()
    grid_response = AgGrid(
        data,
        gridOptions=gridOptions,
        data_return_mode="AS_INPUT",
        update_mode="MODEL_CHANGED",
        fit_columns_on_grid_load=True,
        theme="streamlit", # Add theme color to the table
        enable_enterprise_modules=True,
        height=height,
        width="100%",
        reload_data=True,
    )
    return grid_response


def metrics(max=True, min=True, mean=True):

    fig = go.Figure()

    fig.update_layout(
        height=600, grid={"rows": int(max) + int(min) + int(mean), "columns": 3}
    )

    # Perlei stats
    # max
    c = 0
    if max:
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=ph_max_words_per_day,
                domain={"row": c, "column": 0},
                title=f"Max<br>{ph_date_max}",
            )
        )
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=love_max_words_per_day,
                domain={"row": c, "column": 1},
                title=f"Max<br>{love_date_max}",
            )
        )
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=at_max_words_per_day,
                domain={"row": c, "column": 2},
                title=f"Max<br>{at_date_max}",
            )
        )
        c += 1

    if min:
        # min
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=ph_min_words_per_day,
                domain={"row": c, "column": 0},
                title=f"Min<br>{ph_date_min}",
            )
        )
        # min
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=love_min_words_per_day,
                domain={"row": c, "column": 1},
                title=f"Min<br>{love_date_min}",
            )
        )
        # min
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=at_min_words_per_day,
                domain={"row": c, "column": 2},
                title=f"Min<br>{at_date_min}",
            )
        )
        c += 1
    if mean:
        # mean
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=ph_mean_words_per_day,
                domain={"row": c, "column": 0},
                title="Mean",
            )
        )
        # mean
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=love_mean_words_per_day,
                domain={"row": c, "column": 1},
                title="Mean",
            )
        )
        # mean
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=at_mean_words_per_day,
                domain={"row": c, "column": 2},
                title="Mean",
            )
        )

    return fig


def daily_scatter():
    fig = px.scatter(
        love_words_per_day_date,
        x="timestamp",
        y="words",
        color="author",
        title="Daily Word count per entity",
        color_discrete_map={"Antonio": "blue", "Perlei": "yellow"},
    )
    return fig


def monthly_bar():
    fig = px.bar(
        love_words_per_month_date,
        x="month_date",
        y="words",
        color="author",
        title="Monthly Word count per entity",
        color_discrete_map={"Antonio": "blue", "Perlei": "yellow"},
    )
    return fig


def month_bar():
    fig = px.bar(
        df_months,
        x="timestamp",
        y="words",
        color="author",
        title="Monthly Chat per entity",
        color_discrete_map={"Antonio": "blue", "Perlei": "yellow"},
    )
    return fig


def weekday_bar():
    fig = px.bar(
        df_weekdays,
        x="weekday",
        y="words",
        color="author",
        title="Week days Chat per entity",
        color_discrete_map={"Antonio": "blue", "Perlei": "yellow"},
    )
    return fig


def daily_calender(name):
    if name == "Antonio":
        data_cal = df[df.author == "Antonio"]
        color = "PuBu"
    elif name == "Perlei":
        data_cal = df[df.author == "Antonio"]
        color = "Wistia"
    else:
        data_cal = df.copy()
        color = "magma_r"

    cal = calplot.calplot(
        data=data_cal["words"],
        how="sum",
        cmap=color,
        figsize=(16, 8),
        suptitle="Word count Calender",
    )
    return cal[0]


def image_show(height=500):
    mypath = "pics"
    onlyfiles = [mypath + "/" + f for f in listdir(mypath) if isfile(join(mypath, f))]
    random_path = random.choice(onlyfiles)
    img = imread(random_path)
    img = cvtColor(img, COLOR_BGR2RGB)
    fig = px.imshow(img, height=height)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def metrics_df_style():
    metrics_dict = {
        'ㅤ': ["Perlei", "Love", "Antonio"], 
        'Max': [ph_max_words_per_day, love_max_words_per_day, at_max_words_per_day], 
        'Max Date': [ph_date_max, love_date_max, at_date_max], 
        'Min': [ph_min_words_per_day, love_min_words_per_day, at_min_words_per_day], 
        'Min Date': [ph_date_min, love_date_min, at_date_min], 
        'Mean': [int(ph_mean_words_per_day), int(love_mean_words_per_day), int(at_mean_words_per_day)]
    }
    metrics_df = pd.DataFrame(metrics_dict).set_index('ㅤ').T
    header_styles = [
        {'header': 'Perlei', 'props': [('font-size', '50px'), ('color', 'yellow')]},
        {'header': 'Love', 'props': [('font-size', '50px'), ('color', 'red')]},
        {'header': 'Antonio', 'props': [('font-size', '50px'), ('color', 'blue')]}
    ]
    # Apply styling to the metrics dataframe
    df = metrics_df.style
    
    for style in header_styles:
        header = style['header']
        props = style['props']
        df = df.set_properties(subset=[header], **dict(props))
    condition = [False,True,False,True,False]
    df = df.apply(lambda x: ['font-size: 50px' if c else 'font-size: 50px' for c in condition])
    

    # Intialize a list of tuples containing the CSS styles for table headers
    th_props = [('font-size', '15px'), ('text-align', 'left'),
                ('font-weight', 'bold'),('color', ''),
                ('background-color', ''), ('border','0px #eeeeef'),
        ]

    # Intialize a list of tuples containing the CSS styles for table data
    td_props = [('font-size', '25px'), ('text-align', 'center')]

    # Define hover props for table data and headers
    # cell_hover_props = [('background-color', '#eeeeef')]
    headers_props = [('text-align','center'), ('font-size','40px')]

    # Aggregate styles in a list
    styles = [
        dict(selector="th", props=th_props),
        dict(selector="td", props=td_props),
        # dict(selector="td:hover",props=cell_hover_props),
        dict(selector='th.col_heading',props=headers_props),
        dict(selector='th.col_heading.level0',props=headers_props),
        dict(selector='th.col_heading.level1',props=td_props)
    ]

    df = df.set_table_styles(styles).to_html()

    return df


stop_words = {
        "media omited",
        "omited",
        "media",
        "ba",
        # "ana",
        # "eh",
        # "la",
        # "ma",
        # "bl",
        # "ba",
        # "enno",
        # "bass",
        # "bas",
        # 'li',
        # "aam",
        # "aw",
        # "chi",
        # "bi",
        # "fi",
        # "ss",
        # "s ",
        # "h",
        # "eno",
        # "eza",
        # "3am",
        # "chou",
        # "shi",
        # "mesh",
    }

def reduce_consecutive_letters(text):
    # Use regex to replace consecutive letters with a single letter
    reduced_text = re.sub(r'(.)\1+', r'\1', text)
    return reduced_text

def remove_emoji(text):

    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
    return reduce_consecutive_letters(emoji_pattern.sub(r'', re.sub(r'[^\w]', ' ', text)))

def remove_single_character_words(text):
    # Define the pattern to match single character words
    pattern = r'\b\w{1}\b'

    # Remove single character words from the text
    text_without_single_char_words = re.sub(pattern, ' ', text)

    return text_without_single_char_words


def remove_words_from_string(text, words_to_remove):
    # Construct the regex pattern
    pattern = r'\b(?:{})\b'.format('|'.join(map(re.escape, words_to_remove)))

    # Remove the words from the string
    modified_text = re.sub(pattern, ' ', text)

    return modified_text

@st.cache_data
def generate_word_cloud_2020( df, words_to_remove = {""}, author=['Antonio','Perlei']):
    words_to_remove = words_to_remove.union(stop_words)
    df_filter = df.loc[(df.timestamp.dt.year.isin([2020])) & (df.author.isin(author))]
    text = " ".join(df_filter.message.dropna().to_list()).lower()
    text = remove_single_character_words(text)
    text = remove_emoji(text)
    text = remove_words_from_string(text, words_to_remove)
    counts = pd.Series(text.split()).value_counts()
    counts = pd.DataFrame(counts, columns=['Count'])
# Create WordCloud object
    mypath = f"masks/2020.jpg"
    mask = np.array(Image.open(mypath))
    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color='white',
        contour_width=0.0,
        mask=mask).generate(text)

      # Convert word cloud to an image using PIL
    image = Image.fromarray(np.array(wordcloud))
    return {"image":image, "counts":counts}

@st.cache_data
def generate_word_cloud_2021( df, words_to_remove = [""], author=['Antonio','Perlei']):
    words_to_remove = words_to_remove.union(stop_words)
    df_filter = df.loc[(df.timestamp.dt.year.isin([2021])) & (df.author.isin(author))]
    text = " ".join(df_filter.message.dropna().to_list()).lower()
    text = remove_single_character_words(text)
    text = remove_emoji(text)
    text = remove_words_from_string(text, words_to_remove)
    counts = pd.Series(text.split()).value_counts()
    counts = pd.DataFrame(counts, columns=['Count'])

    mypath = f"masks/2021.jpg"
    mask = np.array(Image.open(mypath))
    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color='white',
        contour_width=0.0,
        mask=mask).generate(text)

      # Convert word cloud to an image using PIL
    image = Image.fromarray(np.array(wordcloud))
    return {"image":image, "counts":counts}

@st.cache_data
def generate_word_cloud_2022( df, words_to_remove = [""], author=['Antonio','Perlei']):
    words_to_remove = words_to_remove.union(stop_words)
    df_filter = df.loc[(df.timestamp.dt.year.isin([2022])) & (df.author.isin(author))]
    text = " ".join(df_filter.message.dropna().to_list()).lower()
    text = remove_single_character_words(text)
    text = remove_emoji(text)
    text = remove_words_from_string(text, words_to_remove)
    counts = pd.Series(text.split()).value_counts()
    counts = pd.DataFrame(counts, columns=['Count'])

    mypath = f"masks/2022.jpg"
    mask = np.array(Image.open(mypath))
    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color='white',
        contour_width=0.0,
        mask=mask).generate(text)

      # Convert word cloud to an image using PIL
    image = Image.fromarray(np.array(wordcloud))
    return {"image":image, "counts":counts}

@st.cache_data
def generate_word_cloud_2023( df, words_to_remove = [""], author=['Antonio','Perlei']):
    words_to_remove = words_to_remove.union(stop_words)
    df_filter = df.loc[(df.timestamp.dt.year.isin([2023])) & (df.author.isin(author))]
    text = " ".join(df_filter.message.dropna().to_list()).lower()
    text = remove_single_character_words(text)
    text = remove_emoji(text)
    text = remove_words_from_string(text, words_to_remove)
    counts = pd.Series(text.split()).value_counts()
    counts = pd.DataFrame(counts, columns=['Count'])

    mypath = "masks/2023.jpg"
    mask = np.array(Image.open(mypath))
    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color='white',
        contour_width=0.0,
        mask=mask).generate(text)

      # Convert word cloud to an image using PIL
    image = Image.fromarray(np.array(wordcloud))
    return {"image":image, "counts":counts}

@st.cache_data
def generate_word_cloud( df, words_to_remove = [""], author=['Antonio','Perlei'], year=[2020,2021,2022,2023,2024]):
    words_to_remove = words_to_remove.union(stop_words)
    df_filter = df.loc[(df.timestamp.dt.year.isin(year)) & (df.author.isin(author))]
    text = " ".join(df_filter.message.dropna().to_list()).lower()
    text = remove_single_character_words(text)
    text = remove_emoji(text)
    text = remove_words_from_string(text, words_to_remove)
    counts = pd.Series(text.split()).value_counts()
    counts = pd.DataFrame(counts, columns=['Count']).reset_index()
    counts.columns = ['Word','Count']

    onlyfiles = ["masks/chaos.jpg","masks/love.jpg","masks/cloud9.jpg","masks/heart.jpg"]
    random_path = random.choice(onlyfiles)
    mask = np.array(Image.open(random_path))

    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color='white',
        contour_width=0.0,
        mask=mask).generate(text)

      # Convert word cloud to an image using PIL
    image = Image.fromarray(np.array(wordcloud))

    return {"image":image, "counts":counts}

@st.cache_data
def unique_words(df):
    text = " ".join(df.message.dropna().to_list()).lower()
    text = remove_emoji(text)
    words = set(text.split())
    return words
