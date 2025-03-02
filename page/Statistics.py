import streamlit as st

from viz import (
    image_show,
    show_image,
    daily_scatter,
    monthly_bar,
    weekday_bar,
    month_bar,
    daily_calender,
)


def write():

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

        img = show_image()
        st.image(img,use_column_width=False)

    st.sidebar.header("Select the Metrics you want to see")
    historical_trend = st.sidebar.checkbox("Historical Trend", value=True)
    date_category = st.sidebar.checkbox("By Time Category", value=True)
    calendar = st.sidebar.checkbox("Calendar", value=True)

    col11, col12 = st.columns(2)

    if historical_trend:
        col11, col12 = st.columns(2)
        with col11:
            st.plotly_chart(
                daily_scatter(), theme="streamlit", use_container_width=True
            )

        with col12:
            st.plotly_chart(monthly_bar(), theme="streamlit", use_container_width=True)

    if date_category:
        col11, col12 = st.columns(2)
        with col11:
            st.plotly_chart(weekday_bar(), theme="streamlit", use_container_width=True)

        with col12:
            st.plotly_chart(month_bar(), theme="streamlit", use_container_width=True)

    if calendar:
        option = st.selectbox("Select Author:", ("Love", "Perlei", "Antonio"))
        st.pyplot(daily_calender(option))
