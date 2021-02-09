import pickle
import joblib
import streamlit as st
import pandas as pd

from utils import perform_match_wrapper, visualize_venue_match_results_wrapper, generate_ui_df, \
    LIST_CITY_DATA_FILE_NAME, read_data_file, get_common_feature_list, LIST_CITY, col_grain, colList_meta

# loading the trained model
project_path = r'C:\Users\GS-1931\Desktop\GIT_DESKTOP\0_invisibly\04_poc_code_files\16_LoanPredict'
artifacts_path = project_path + '\\migration_notebooks\\artifacts\\'

#X_source = joblib.load(artifacts_path + 'X_source')
#X_dest = joblib.load(artifacts_path + 'X_dest')
#list_sources = joblib.load(artifacts_path + 'list_sources')
#colList_features = joblib.load(artifacts_path + 'colList_features')
#colList_meta = joblib.load(artifacts_path + 'colList_meta')


@st.cache()
# defining the function which will make the prediction using the data which the user inputs
def prediction():
    # Pre-processing user input
    return 'abc'


# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Migration ML App</h1> 
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    # L1-inputs
    SOURCE_CITY = st.selectbox('Source City', LIST_CITY)

    file_name = [i for i in LIST_CITY_DATA_FILE_NAME if SOURCE_CITY in i][0]
    X_source = read_data_file(file_name=file_name, data_type='artifact_app')
    list_sources_venues = list(X_source[col_grain].values)

    # L1-p2 input
    SOURCE_VENUE = st.selectbox('Source Neighborhood', list_sources_venues)
    DEST_CITY = st.selectbox('Destination City', LIST_CITY)

    # L2-inputs
    NUM_VENUES = st.selectbox("select num venues to be displayed", [i for i in range(1, 11)])
    NUM_MATCH = st.selectbox("num matching locations to be displayed", [i for i in range(1, 11)])
    result = ""

    file_name = [i for i in LIST_CITY_DATA_FILE_NAME if DEST_CITY in i][0]
    X_dest = read_data_file(file_name=file_name, data_type='artifact_app')

    source_name = SOURCE_VENUE
    colList_features = get_common_feature_list(X_source=X_source, X_dest=X_dest)
    X_match, X_meta_mapper = perform_match_wrapper(X_source=X_source, X_dest=X_dest, source_name=source_name,
                                                   num_match=None, precise_match=True,
                                                   colList_features=colList_features, colList_meta=colList_meta)

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Process Request"):
        # user_input : num_match
        # user_input : num_venues
        num_match = NUM_MATCH
        num_venues = NUM_VENUES
        source_name = SOURCE_VENUE
        X_match_sorted_named, graph = visualize_venue_match_results_wrapper(X_source=X_source, X_match=X_match,
                                                                            X_meta_mapper=X_meta_mapper,
                                                                            source_name=source_name,
                                                                            colList_features=colList_features,
                                                                            num_match=num_match, num_venues=num_venues)

        st.success('processing completed')
        st.dataframe(X_match_sorted_named)
        st.pyplot(graph)
        print('done')


if __name__ == '__main__':
    main()
