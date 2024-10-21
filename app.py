import awesome_streamlit as ast
import streamlit as st

import page.Statistics
import page.home
import page.Pics
import page.NLP
import yaml
import streamlit_authenticator as st_auth

# create a session id for the user to keep track of the conversation
# st.session_state.session_id = "streamlit_interface"
def load_config(file_path):
    """Load configuration from a YAML file."""
    with open(file_path) as file:
        return yaml.load(file, Loader=SafeLoader)
    
def authenticate_user(config):
    """Authenticate the user."""
    authenticator = st_auth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )
    return authenticator.login()

def setup_page():
    """Set up the Streamlit page configuration."""
    st.set_page_config(page_title="Random Love Demo", layout="wide", page_icon="💎")

def main_app():
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

def main():        
    config = load_config("users.yaml")
    setup_page()

    name, authentication_status, username = authenticate_user(config)
    if authentication_status:    
        main_app()
    elif authentication_status is False:
        st.error("Username/password is incorrect")
    else:
        st.warning("Please enter your username and password")

if __name__ == "__main__":
    main()
