import pickle
import joblib
import streamlit as st
import pandas as pd

from utils import perform_match_wrapper, visualize_venue_match_results_wrapper, generate_ui_df, \
    LIST_CITY_DATA_FILE_NAME, read_data_file, get_common_feature_list, LIST_CITY, col_grain, colList_meta

WELCOME_MSG = """We understand! Moving to a new city is stressful. It disturbs your life. Choosing the right neighborhood is crucial. A neighborhood that can give you a similar lifestyle, and a similar cost of living. If you ask people, you would get biased opinions. No worries! Using the "Machine Learning" algorithms, we solve the problem for you. We scan your current neighborhood for its many attributes and recommend the most suitable neighborhood in the new city."""


@st.cache()
def prediction():
    # Pre-processing user input
    return None


# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:yellow;padding:5px"> 
    <h1 style ="color:black;text-align:center;">Neighborhood Recommender</h1> 
    </div> 
    """

    st.set_page_config(layout="wide", initial_sidebar_state='expanded')

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown(WELCOME_MSG)
    st.markdown("Isn't it cool? Try it!")
    st.text('Note: We will be adding more cities soon. :)')

    st.sidebar.title('User Inputs')
    # L1-inputs
    #st.markdown('Moving from:')
    SOURCE_CITY = st.sidebar.selectbox('From City', LIST_CITY)

    # L3 input
    # st.text("")
    # st.markdown('Moving to:')
    DEST_CITY = st.sidebar.selectbox('To City', LIST_CITY)

    st.text("")
    st.text("")
    if SOURCE_CITY == DEST_CITY:
        st.markdown('Moving to a new Neighborhood in {}?'.format(SOURCE_CITY))
    else:
        st.markdown('Moving from {} to {}?'.format(SOURCE_CITY, DEST_CITY))
    # L2 input
    file_name = [i for i in LIST_CITY_DATA_FILE_NAME if SOURCE_CITY in i][0]
    X_source = read_data_file(file_name=file_name, data_type='artifact_app')
    list_sources_venues = list(X_source[col_grain].values)
    SOURCE_VENUE = st.selectbox('From Neighborhood', list_sources_venues)

    # L2-inputs
    st.text("")
    st.text("")
    st.sidebar.markdown('Below inputs help in better visualization of recommendations')
    #NUM_MATCH = st.selectbox("num matching neighborhood to be displayed", [i for i in range(5, 11)])
    #NUM_VENUES = st.selectbox("num venues to be displayed", [i for i in range(6, 11)])
    NUM_MATCH = st.sidebar.slider("Choose num matching neighborhood to be displayed: ", min_value=3, max_value=8, value=4, step=1)
    NUM_VENUES = st.sidebar.slider("Choose num venues to be displayed: ", min_value=4, max_value=15, value=8, step=1)

    file_name = [i for i in LIST_CITY_DATA_FILE_NAME if DEST_CITY in i][0]
    X_dest = read_data_file(file_name=file_name, data_type='artifact_app')

    source_name = SOURCE_VENUE
    colList_features = get_common_feature_list(X_source=X_source, X_dest=X_dest)
    X_match, X_meta_mapper = perform_match_wrapper(X_source=X_source, X_dest=X_dest, source_name=source_name,
                                                   num_match=None, precise_match=True,
                                                   colList_features=colList_features, colList_meta=colList_meta)

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Search"):
        # user_input : num_match
        # user_input : num_venues
        num_match = NUM_MATCH
        num_venues = NUM_VENUES
        source_name = SOURCE_VENUE
        with st.spinner('Finding the right neighborhood for you ....'):
            X_match_sorted_named, graph = visualize_venue_match_results_wrapper(X_source=X_source, X_match=X_match,
                                                                                X_meta_mapper=X_meta_mapper,
                                                                                source_name=source_name,
                                                                                colList_features=colList_features,
                                                                                num_match=num_match, num_venues=num_venues)

            st.success('Here are a few neighborhood/s suggestion for you. Good luck!')

            df = X_match_sorted_named.copy()
            df = df.drop(columns=['index'])
            #st.table(df)
            st.dataframe(df)

        with st.spinner('Generating the visualization....'):
            st.pyplot(graph)


if __name__ == '__main__':
    main()
