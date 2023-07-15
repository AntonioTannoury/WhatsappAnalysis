import streamlit as st
from viz import (
    metrics_df_style,
    metrics, 
    image_show, 
    filter_df, 
    ad_grid)

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
            ## It breaks rules
            ### So an Ostriche can become a Flamingo 🦩

            """
        )
        st.markdown("#")
        st.button("Don't push!")
    with col2:
        image = image_show()
        st.plotly_chart(image, theme="streamlit", use_container_width=True)

    col21, col22 = st.columns([1,2])
    with col21:
        st.markdown(
            """
            ### Summary Metrics
            """
        )
        st.markdown(metrics_df_style(),unsafe_allow_html=True)
    with col22:
        st.markdown(
            """
            ### Chat History
            """
        )
        filter_data = filter_df()
        ad_grid(filter_data)





#%%
