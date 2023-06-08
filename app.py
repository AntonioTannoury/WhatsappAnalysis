import awesome_streamlit as ast
import streamlit as st

import page.Statistics
import page.home
import page.Pics
import page.NLP

st.set_page_config(page_title="Random Love Demo", layout="wide")


def main():

    PAGES = {
        "Home Sweet Home": page.home,
        "Random Statistics": page.Statistics,
        "Random Visualization": page.Pics,
        "Random NLP": page.NLP,
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Love Pages", list(PAGES.keys()))

    _page = PAGES[selection]
    st.spinner(f"Loading {selection} ...")
    ast.shared.components.write_page(_page)


if __name__ == "__main__":
    main()
#%%
