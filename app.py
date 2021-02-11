import os
import re
import streamlit as st

code = """<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-611FPFGKT5"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'UA-611FPFGKT5');
</script>"""


a = os.path.dirname(st.__file__)+'/static/index.html'
with open(a, 'r') as f:
    data = f.read()
    if len(re.findall('UA-', data)) == 0:
        with open(a, 'w') as ff:
            newdata = re.sub('<head>', '<head>'+code, data)
            ff.write(newdata)


from utils import perform_match_wrapper, visualize_venue_match_results_wrapper, generate_ui_df, \
    LIST_CITY_DATA_FILE_NAME, read_data_file, get_common_feature_list, LIST_CITY, col_grain, colList_meta

# WELCOME_MSG = """We understand! Moving to a new city is stressful. It disturbs your life. Choosing the right neighborhood is crucial. A neighborhood that can give you a similar lifestyle, and a similar cost of living. If you ask people, you would get biased opinions. No worries! Using the "Machine Learning" algorithms, we solve the problem for you. We scan your current neighborhood for its many attributes and recommend the most suitable neighborhood in the new city."""
WELCOME_MSG = """<p class="big-font">We understand! Moving to a new city is stressful. A neighborhood that can give you a similar lifestyle. If you ask people, you would get biased opinions. No worries! Using the "Machine Learning" algorithms, we solve the problem for you. We scan your current neighborhood and recommend the most suitable neighborhood in the new city.</p>"""


@st.cache()
def prediction():
    # Pre-processing user input
    return None


# this is the main function in which we define our webpage
def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:yellow;padding:1px"> 
    <h1 style ="color:black;text-align:center;">Neighborhood Recommender</h1> 
    </div> 
    """

    st.set_page_config(layout="wide", initial_sidebar_state='expanded')
    st.markdown(html_temp, unsafe_allow_html=True)

    st.sidebar.title('About')
    st.markdown("""
                    <style>
                    .big-font {
                        font-size:12px !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
    st.sidebar.markdown(WELCOME_MSG, unsafe_allow_html=True)
    st.sidebar.markdown("""<p class="big-font">Isn't it cool? Try it!</p>""", unsafe_allow_html=True)

    st.markdown("")
    #st.markdown('Select Cities')
    col1, col2 = st.beta_columns(2)

    # L1-inputs
    SOURCE_CITY = col1.selectbox('Moving from city:', LIST_CITY)

    # L3 input
    same_city = st.checkbox("moving within same city")
    if same_city:
        DEST_CITY = SOURCE_CITY
        st.markdown("#### Awesome! Let's find a new neighborhood for you in {}.".format(DEST_CITY))
    else:
        DEST_CITY = col2.selectbox('Moving to city:', LIST_CITY)
        st.markdown("#### Great! Let's explore {}.".format(DEST_CITY))

    # L2 input
    file_name = [i for i in LIST_CITY_DATA_FILE_NAME if SOURCE_CITY in i][0]
    X_source = read_data_file(file_name=file_name, data_type='artifact_app')
    list_sources_venues = list(X_source[col_grain].values)
    SOURCE_VENUE = st.selectbox('To get suitable locations, please select the neighborhood you are moving from:',
                                    list_sources_venues)
    # L2-inputs
    # st.sidebar.markdown('Below inputs help in better visualization of recommendations')
    NUM_MATCH = st.sidebar.slider("Choose number of matching neighborhood to be displayed: ", min_value=3, max_value=8,
                                  value=4, step=1)
    NUM_VENUES = st.sidebar.slider("Choose number of venues to be displayed: ", min_value=4, max_value=15, value=10,
                                   step=1)

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
                                                                                num_match=num_match,
                                                                                num_venues=num_venues)

            #st.balloons()
            st.success('Here are a few neighborhood/s suggestion for you. Good luck!')

            df = X_match_sorted_named.copy()
            df = df.drop(columns=['index'])
            # st.table(df)
            st.dataframe(df)

        #show_chart = st.checkbox("display chart", False)
        #if show_chart:
        with st.spinner('Generating the visualization....'):
            st.pyplot(graph)
    # st.subheader("A Chart you can show or hide")
    #   with st.section(label="the_chart"):
    #   st.write(df)


if __name__ == '__main__':
    main()
