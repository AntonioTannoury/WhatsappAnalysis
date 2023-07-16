import streamlit as st
from viz import (
    ad_grid,
    unique_words,
    generate_word_cloud,
    generate_word_cloud_2020,
    generate_word_cloud_2021,
    generate_word_cloud_2022,
    generate_word_cloud_2023,
    df)


def write():

    st.markdown(
        """
        # Cloud 9 ☁️☁️☁️
        """
    )
        
    uniques = unique_words(df)
    names = df.author.unique()
    years = df.timestamp.dt.year.unique()
    if "start" not in st.session_state:
        st.session_state['years_option'] = 1
    if 'names_option' not in st.session_state:
        st.session_state['names_option'] = list(names)
    if 'years_option' not in st.session_state:
        st.session_state['years_option'] = years
    if 'words' not in st.session_state:
        st.session_state['words'] = set()

    
    col11, col12 = st.columns(2)
    with col11:
        words_to_remove = st.multiselect("Select words to remove:", uniques, key=0)
        submit_button = st.button("Submit Words")
        st.session_state['words'] = st.session_state['words'].union(set(words_to_remove))

    col21, col22, col23, col24 = st.columns(4)
    if submit_button or st.session_state['years_option']==1:
        st.session_state['years_option'] = 2
        with col21:
            image21 = generate_word_cloud_2020(df=df, words_to_remove=st.session_state['words'])['image']
            st.image(image21, use_column_width=True)
        with col22:
            image22 = generate_word_cloud_2021(df=df, words_to_remove=st.session_state['words'])['image']
            st.image(image22, use_column_width=True)
        with col23:
            image23 = generate_word_cloud_2022(df=df, words_to_remove=st.session_state['words'])['image']
            st.image(image23, use_column_width=True)
        with col24:
            image24 = generate_word_cloud_2023(df=df, words_to_remove=st.session_state['words'])['image']
            st.image(image24, use_column_width=True)



        
    
    col31, col32 = st.columns(2)

    with col31:
        names_option = st.multiselect("Select Author:", names, default=names, key=2)
        
    with col32:
        years_option = st.multiselect("Select Year:", years, default=years, key=3)

    if (st.session_state['years_option'] == 2) or (submit_button) or (set(st.session_state['years_option']) != set(years_option)) or (set(st.session_state['names_option']) != set(names_option)):
    
        col41, col42 = st.columns([3,1])
        if len(years_option)==0 or len(names_option)==0:
            st.write("To filter data, please choose at least one author and at least one year!")
        else:
            with col41:
                image = generate_word_cloud(df=df, author=names_option, year=years_option, words_to_remove=st.session_state['words'])
                st.session_state['years_option'] = years_option
                st.session_state['names_option'] = names_option
                st.session_state['years_option'] = 3
                st.image(image['image'], use_column_width=True)            
            with col42:
                data = image['counts']
                ad_grid(data,height=555)


