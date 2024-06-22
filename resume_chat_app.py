from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

#importing the web based loader to train our Vector store 
from langchain_community.document_loaders import WebBaseLoader

#Importing the FIASS vector store and the Langchain's text splitter documents
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


#LLM Initialization 
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser

from constants import main_resume_template, hyde_template, template


#importing the chain from the runnables (If we want to have a logic in between the |'s of Langchain then we can use this)
from langchain_core.runnables import chain

import getpass

import os



from dotenv import load_dotenv


"""
Aim: 
The main aim is to use the web based loader to load the documents and get a
Creating an Enterprise ready RAG Resume chatter. 

Status: Successfully Created the streaming RAG chain using LangChain and NVIDIA NIM
"""




def load_documents(URL_var):
    #loading document from the website
    loader = WebBaseLoader(URL_var)
    docs = loader.load()
    print(len(docs))
    print("Loaded the documents in the menthod")
    return docs

def create_vector_store(docs,nvidia_embeddings):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(docs)
    #For the Ingestion we are using the FAISS vector store to ingest my data
    #Note: Here, we are using FAISS-CPU
    vector = FAISS.from_documents(documents, nvidia_embeddings)
    retriever = vector.as_retriever()
    return retriever

"""
Note: the retriever.invoke("testing llm systems") => this will give the documents which are the most similar to the given user question
        Returns 3 documents by default
"""

def create_hyde_query_transformer(model, hyde_template ):
    #here, we will create the hyde query transformer
    hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
    hyde_query_transformer = hyde_prompt | model | StrOutputParser()
    return hyde_query_transformer
"""
Why? 
This is because the question is transformed and the transformed question is closer to the embedding space of documents than the real question.
hyde_query_transformer.invole({"question": "testing llm systems"})
"""


# @chain
# def hyde_retriever(question):
#     hypothetical_document = hyde_query_transformer.invoke({"question": question})
#     # print("==========================Hyde Retriever question=======================")
#     # print(hypothetical_document)
#     # print("========================================================================")
#     return retriever.invoke(hypothetical_document)


# def create_answerchain_withprompt( model, main_resume_template ):
#     prompt = ChatPromptTemplate.from_template(main_resume_template)
#     answer_chain = prompt | model | StrOutputParser()
#     return answer_chain




def main():
 
    #importing the Nvidia embeddings 
    #We are initilizing the Nvidia embeddings

    nvidia_embeddings = NVIDIAEmbeddings()

    #Step 1: Loading documents: passing the web based document into the loader docs

    docs = load_documents("https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/")
    print("Loaded the documents and stored it in the docs variable")

    #Step 2: creating the vector store from the retrieveri.e., now we will create vector store from the documents which we have
    retriever = create_vector_store(docs, nvidia_embeddings)
    print("Retriever is set up")

    #Step 3: Initilizing the model
    nvidia_model = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")
    print("Loaded the Mistral Model 8*22b model successfully")
    #result = nvidia_model.invoke("Write a ballad about LangChain.")
    #print(result.content)
    print("-----------------------------------------------------------------------------")

    #Step 4: Initializing Hyde retriever
    hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
    hyde_query_transformer = hyde_prompt | nvidia_model | StrOutputParser()
    print("Hyde query transformer is loaded successfully")

    @chain
    def hyde_retriever(question):
        hypothetical_document = hyde_query_transformer.invoke({"question": question})
        # print("==========================Hyde Retriever question=======================")
        # print(hypothetical_document)
        # print("========================================================================")
        return retriever.invoke(hypothetical_document)

    #Step 5: Creating Answer chain
    prompt = ChatPromptTemplate.from_template(template)
    answer_chain = prompt | nvidia_model | StrOutputParser()
    print("=================Answer chain is created======================================")

    @chain
    def final_chain(question):
        documents = hyde_retriever.invoke(question)
        for s in answer_chain.stream({"question": question, "context": documents}):
            yield s


    # #Step 6: Getting the answer
    print("final_chain")
    for s in final_chain.stream("how can langsmith help with testing"):
        print(s, end="")

    






if __name__ == "__main__":

    load_dotenv()

    if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
        nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
        assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
        os.environ["NVIDIA_API_KEY"] = nvidia_api_key
    
    main()