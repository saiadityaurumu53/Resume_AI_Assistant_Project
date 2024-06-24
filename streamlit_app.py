import streamlit as st

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

#importing the PDF loader to train our Vector store 
import PyPDF2 


#Importing the FIASS vector store and the Langchain's text splitter documents
from langchain_community.vectorstores import FAISS
#from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter


#LLM Initialization 
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser

from constants import main_resume_template, hyde_template, template


#importing the chain from the runnables (If we want to have a logic in between the |'s of Langchain then we can use this)
from langchain_core.runnables import chain

#For the importing of the required 
from langchain_core.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain
from constants import job_desc_template, resume_prompt_template, compare_prompt_template

import getpass

import os

st.set_page_config(page_title="Chat With Your Resume: Powered by Nvidia NIM and LangChain")

st.title("Resume AIAssistant: Powered by NVIDIA and LangChain")


#FUNCTIONS FOR THE IMPLEMENTATION OF THE CHAT WITH YOUR RESUME USING "RAG"

def get_text_chunks(text):
    #we will use the LangChain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,
        chunk_overlap=50,
        length_function=len 
    )
    #text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    #For the Ingestion we are using the FAISS vector store to ingest my data
    #Note: Here, we are using FAISS-CPU
    nvidia_embeddings = NVIDIAEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=nvidia_embeddings)
    return vectorstore

@st.cache_resource
def get_hyde_query_chain():
    nvidia_model = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")
    hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
    hyde_query_transformer = hyde_prompt | nvidia_model | StrOutputParser()
    print("===============Hyde query transformer is loaded successfully=================")
    return hyde_query_transformer

    # @chain
    # def hyde_retriever(question):
    #     hypothetical_document = hyde_query_transformer.invoke({"question": question})
    #     return retriever.invoke(hypothetical_document)

@st.cache_resource
def get_conversation_chain():
    nvidia_model = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")
    prompt = ChatPromptTemplate.from_template(template)
    answer_chain = prompt | nvidia_model | StrOutputParser()
    print("=================Conversation Answer chain is created======================================")
    return answer_chain


def generate_response(user_query):
    #first step is to invoke the hyde using retriever
    retriever_var = st.session_state.retriever_resume
    hypothetical_document = st.session_state.hyde_chain.invoke({"question": user_query})
    documents = retriever_var.invoke(hypothetical_document)
    #now we got the similar documents
    generated_response_str = st.session_state.conversation_chain.invoke({"question": user_query, "context": documents})
    return generated_response_str

#FUNCTIONS FOR THE IMPLEMENTATION OF THE SEQUENTIAL CHAINS

@st.cache_resource
def create_resume_chain():
    nvidia_model = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")
    prompt = PromptTemplate.from_template(resume_prompt_template)
    resume_chain = prompt | nvidia_model | StrOutputParser()
    return resume_chain

@st.cache_resource
def create_job_desc_chain():
    nvidia_model = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")
    prompt = PromptTemplate.from_template(job_desc_template)
    job_desc_chain = prompt | nvidia_model | StrOutputParser()
    return job_desc_chain

@st.cache_resource
def create_compare_chain():
    nvidia_model = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")
    prompt = PromptTemplate.from_template(compare_prompt_template)
    compare_chain = prompt | nvidia_model | StrOutputParser()
    return compare_chain


def main():
    load_dotenv()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    
    if "hyde_chain" not in st.session_state:
        st.session_state.hyde_chain = None

    if "retriever_resume" not in st.session_state:
        st.session_state.retriever_resume = None 

    #Resume and the Job description storage
    if "RESUME_TEXT" not in st.session_state:
        st.session_state.RESUME_TEXT = None
    
    if "JOB_DESC_TEXT" not in st.session_state:
        st.session_state.JOB_DESC_TEXT = None

    if "job_desc_chain" not in st.session_state:
        st.session_state.job_desc_chain = None
    
    if "resume_chain" not in st.session_state:
        st.session_state.resume_chain = None

    if "compare_chain" not in st.session_state:
        st.session_state.compare_chain = None

    #Intermediate steps answers 
    if "resume_skills" not in st.session_state:
        st.session_state.resume_skills = None

    if "job_skills" not in st.session_state:
        st.session_state.job_skills = None
    

 
    

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
            #code to generate the response
            #ai_response = get_conversation_chain(retriever)
            
            #[answer_chain_var, hyde_retriever_var] = st.session_state.conversation_chain
            ai_response = generate_response(user_query)

            st.markdown(ai_response)

        st.session_state.chat_history.append(AIMessage(ai_response))

    #now we also want a side bar where the user is going to upload the documents
    #Also if we want to keep or put things inside it then we are gonna use the with keyword
    with st.sidebar:
        st.subheader("Upload your Resume")
        pdf_doc = st.file_uploader("Upload your Resume PDF here and click on 'Train' to train Vector Store with your Resume", accept_multiple_files=False, type="pdf")

        #now we will create a button with the name and whose functionality is to process
        #Here, the multiple files are stored here in the pdf_docs variable
        content = ""
        if pdf_doc is not None:
            #Reading the pdf file
            pdf_reader = PyPDF2.PdfReader(pdf_doc)
            #Now extreacting the content
            content = ""
            for page in range(len(pdf_reader.pages)):
                present_content = pdf_reader.pages[page].extract_text()
                #st.write(present_content)
                content += present_content
            #st.write(content)

        st.session_state.RESUME_TEXT = content
        chunks = get_text_chunks(content)
        st.write(chunks)

        if st.button("Train"):
            with st.spinner("Training Using Chunks and Creating VectorStore and Chains!!!"):
                
                nvidia_embeddings = NVIDIAEmbeddings()

                def create_vector_store(chunks,nvidia_embeddings):
                    #For the Ingestion we are using the FAISS vector store to ingest my data
                    #Note: Here, we are using FAISS-CPU
                    vector = FAISS.from_texts(texts=chunks, embedding=nvidia_embeddings)
                    retriever = vector.as_retriever()
                    return retriever
                st.session_state.retriever_resume = create_vector_store(chunks, nvidia_embeddings)

                #Step 3: Create the vector store with the Embeddings
                #vectorstore = get_vectorstore(text_chunks)

                #Step 4: Create Conversational Chain
                st.session_state.conversation_chain = get_conversation_chain()
                st.session_state.hyde_chain = get_hyde_query_chain()
                #print("In Main")
                #print("Length of the conversation_chain is", end="")
                #print(len(st.session_state.conversation_chain))
        
        #Here, we need to Give the Job Description
        st.subheader("Paste your Job Description Below:")
        st.session_state.JOB_DESC_TEXT = st.text_input("Job Description", "Enter Here ")
        st.write("The current Job Description is: ", st.session_state.JOB_DESC_TEXT)

        if st.button("Extract skills, requirements and experiance summary of Resume and Job Description"):
            with st.spinner("Creating Resume chain, Job desc chain and extracting"):
                st.session_state.resume_chain = create_resume_chain()
                st.session_state.job_desc_chain = create_job_desc_chain()
                st.session_state.job_skills = st.session_state.job_desc_chain.invoke(st.session_state.JOB_DESC_TEXT)
                st.write(st.session_state.job_skills)
                st.session_state.resume_skills = st.session_state.resume_chain.invoke(st.session_state.RESUME_TEXT)
                st.write(st.session_state.resume_skills)

                #HERE, WE HAVE STORED THE RESPONSE OF THE LLM MODEL

        if st.button("Analyze and compare the Resume and Job description"):
            with st.spinner("Creating the Analyzation report by comparing the Resume and Job description"):
                st.session_state.compare_chain = create_compare_chain()
                comparison_output =  st.session_state.compare_chain.invoke({"resume_skills": st.session_state.resume_skills, "job_skills":  st.session_state.job_skills})
                st.write(comparison_output)
                st.session_state.chat_history.append(AIMessage(comparison_output))


        
        

                


if __name__ == "__main__":
    main()