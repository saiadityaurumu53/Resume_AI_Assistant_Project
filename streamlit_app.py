import streamlit as st

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage




load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Streaming Nvidia NIM")

st.title("Streaming Nvidia NIM")

#Showing the previous conversation
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)



#Hitting the user query
user_query = st.chat_input("Your message")
if (user_query is not None) and (user_query != ""):
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)
        #appends to the start after we hit send
    
    with st.chat_message("AI"):
        ai_response = "i don't know"
        st.markdown(ai_response)

    st.session_state.chat_history.append(AIMessage(ai_response))