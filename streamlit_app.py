import streamlit as st
import pandas as pd
import tempfile
import os
from backend import process_teams

# Apply custom CSS for button styling
custom_css = """
<style>
/* Apply grey background to the entire sidebar */
.sidebar .sidebar-content {
    background-color: #f1f1f1;
}
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Page title

st.markdown("""
    <h1>
        <span style='color: #E84A27;'>👥 Match</span>-<span style='color: #000079;'>Illinois</span>
    </h1>
    """, unsafe_allow_html=True)

# Sidebar for file upload and sample CSV download
with st.sidebar:
    st.markdown("""
     <h2 style="color:#000000; margin:0;">Upload your CSV</h2>
     """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    # Use markdown to create a visual separation and apply custom styles
    st.markdown("""
        <h2 style="color:#000000; margin:0;">Need a CSV to try?</h2>
        <div style="background-color:#ffffff; padding:10px; border-radius:5px; margin-bottom:10px;">
            <p style="color:#000000; margin:0;">Download a sample below:</p>
        </div>
    """, unsafe_allow_html=True)
    
    with open("example_data.csv", "rb") as file:
        st.download_button(
            label="Download Sample CSV",
            data=file,
            file_name="example_data.csv",
            mime="text/csv"
        )

# Process uploaded file and return results
if uploaded_file:
    with st.spinner('Processing...it takes a minute!'):
        # Temporary files setup
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file_path = os.path.join(temp_dir, 'input.csv')
            output_file_path = os.path.join(temp_dir, 'output.csv')

            # Save uploaded file to temp directory
            df = pd.read_csv(uploaded_file)
            df.to_csv(input_file_path, index=False)

            # Process teams based on the backend logic
            process_teams(input_file_path, output_file_path)

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
    st.success('File successfully uploaded and processed!')

# Information about the app
st.markdown('**What can this app do?**')
st.info('This app allows users to upload a CSV file with user skills and project descriptions to form teams based on their skills and interests.')

st.markdown('**How to use the app?**')
st.info('Simply upload your CSV file and the app will process it to form teams, which you can then download.')

st.markdown('**About the algorithm:**')
st.info("""
**Team Formation Algorithm 🔍 :**

- **Weighted Preferences:**
  - Users are assigned to project areas (e.g., Web Dev, Data Analytics) based on their preferences.
  - Preferences are weighted to give higher priority to stronger interests.

- **BERTopic:**
  - Utilizes advanced NLP to model topics from project descriptions.
  - Groups project descriptions into topics for better team alignment based on content.

- **Latent Dirichlet Allocation (LDA):**
  - Applied to descriptions where BERTopic is unable to define clear topics.
  - Further categorizes projects to ensure all descriptions are grouped into coherent topics.

The algorithm aims to create balanced teams where each member’s preferences and project topics are well represented.
""")
