import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
import os


code = st.secrets["secrets"]
API = code["api_key"]


# Initialize the language model
llm = ChatGroq(model_name="llama3-70b-8192", api_key=API)

# Load Data
df = pd.read_excel("Superstore.xls")
smart_df = SmartDataframe(df, config={"llm": llm})

# Streamlit Interface
st.title("Exploring Conversations with Data using LLM")

# Initialize session state for question history
if 'question_history' not in st.session_state:
    st.session_state.question_history = []

# Display data
st.sidebar.header("Options")
show_data = st.sidebar.checkbox("Show raw data")
if show_data:
    st.subheader("Superstore Data")
    st.dataframe(df.head(20))

user_question = st.text_input("Ask a question about the data")

# Display the response to the user's question
if st.button("Get Answer"):
    try:
        answer = smart_df.chat(user_question)
        st.write(answer)
        st.session_state.question_history.append(user_question)

        # Display the image if it exists
        image_path = "exports/charts/temp_chart.png"
        if os.path.exists(image_path):
            st.image(image_path, caption="Sample Chart", use_column_width=True)
        else:
            st.write(f"No image found at path: {image_path}")

        # Delete the image file after displaying it
        if os.path.exists(image_path):
            os.remove(image_path)
    except Exception as e:
        st.error(f"Error occurred: {e}")

# Display the history of questions
st.subheader("Question History")
for i, question in enumerate(st.session_state.question_history, 1):
    st.write(f"{i}. {question}")
