import streamlit as st
from viz import image_show


def write():

    st.markdown(
        """
        # When Random Love Hits
        """
    )
    image = image_show(height=800)
    st.plotly_chart(image, theme="streamlit", use_container_width=True)
    (
        _,
        _,
        col3,
        col4,
        col5,
        col6,
        _,
        _,
        _,
    ) = st.columns(9)
    with col5:
        st.button("Don't push!")
