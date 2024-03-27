import streamlit as st
import pandas as pd
import time
import tempfile
import os
from backend import process_teams  # Replace 'your_backend_module' with the actual name of your Python file without '.py'

# Page title
st.set_page_config(page_title='Team Formation App', page_icon='👥')
st.title('👥 Team Formation App')

with st.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app allows users to upload a CSV file with user skills and project descriptions to form teams based on their skills and interests.')

    st.markdown('**How to use the app?**')
    st.warning('Simply upload your CSV file and the app will process it to form teams, which you can then download.')

# Sidebar for file upload
with st.sidebar:
    st.header('Upload your CSV')
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success('File successfully uploaded!')

# Process uploaded file and return results
if uploaded_file:
    with st.spinner('Processing...'):
        # Temporary files setup
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file_path = os.path.join(temp_dir, 'input.csv')
            output_file_path = os.path.join(temp_dir, 'output.csv')

            # Save uploaded file to temp directory
            df.to_csv(input_file_path, index=False)

            # Process teams based on the backend logic
            process_teams(input_file_path, output_file_path)  # This should match the function from your backend logic

            # Read the results
            result_df = pd.read_csv(output_file_path)

            # Show results and allow download
            st.header('Formed Teams')
            st.dataframe(result_df)
            st.download_button(
                label="Download formed teams CSV",
                data=result_df.to_csv(index=False).encode('utf-8'),
                file_name='formed_teams.csv',
                mime='text/csv'
            )
else:
    st.warning('Please upload a CSV file to get started!')
