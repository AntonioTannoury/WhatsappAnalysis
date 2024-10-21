import streamlit as st
from viz import image_show, show_image


def write():

    st.markdown(
        """
        # When Random Love Hits
        """
    )
    img = show_image(height=800)
    st.image(img,use_column_width=False)
    st.button("Don't push!")
