import streamlit as st
from viz import metrics, image_show, filter_df, ad_grid

import random
import datetime as dt


def write():
    """Used to write the page in the app.py file"""

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            # When Love Hits Randomly

            ## It hits hard
            ### And basic logic becomes illogical 
            .
            ## It breaks rules
            ### So an Ostriche can become a Flamingo 🦩

            """
        )
        st.markdown("#")
        st.markdown("#")
        st.markdown("#")
        st.button("Don't push!")
    with col2:
        image = image_show()
        st.plotly_chart(image, theme="streamlit", use_container_width=True)

    st.sidebar.header("Select the Metrics you want to see")
    max = st.sidebar.checkbox("Maximum", value=True)
    min = st.sidebar.checkbox("Minimum", value=True)
    mean = st.sidebar.checkbox("Average", value=True)

    names = ["Perlei", "Love", "Antonio"]
    colors = ["yellow", "red", "blue"]

    col11, col12, col13 = st.columns(3)

    with col11:
        title = "<h1 style='text-align: center; color: yellow;'>Perlei</h1>"
        st.markdown(title, unsafe_allow_html=True)
    with col12:
        title = "<h1 style='text-align: center; color: red;'>Love</h1>"
        st.markdown(title, unsafe_allow_html=True)
    with col13:
        title = "<h1 style='text-align: center; color: blue;'>Antonio</h1>"
        st.markdown(title, unsafe_allow_html=True)

    metrics_figs = metrics(max=max, min=min, mean=mean)
    st.plotly_chart(metrics_figs, theme="streamlit", use_container_width=True)

    st.sidebar.header("More Details?")
    detailed_results = st.sidebar.checkbox("Show Chat History", value=True)

    if detailed_results:
        st.markdown(
            """
            ### Chat History
            """
        )
        filter_data = filter_df()
        ad_grid(filter_data)


#%%
