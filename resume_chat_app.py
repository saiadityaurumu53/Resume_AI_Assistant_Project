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

from constants import main_resume_template, hyde_template


#importing the chain from the runnables and "Wrapping the original retriever and the new chain"
from langchain_core.runnables import chain

import getpass

import os



from dotenv import load_dotenv







def load_documents(URL_var):
    #loading document from the website
    loader = WebBaseLoader(URL_var)
    docs = loader.load()
    return docs

def create_vector_store(docs,nvidia_embeddings):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, nvidia_embeddings)
    retriever = vector.as_retriever()
    return retriever

def create_hyde_query_transformer(model, hyde_template ):
    #here, we will create the hyde query transformer
    hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
    hyde_query_transformer = hyde_prompt | model | StrOutputParser()
    return hyde_query_transformer

def create_answerchain_withprompt( model, main_resume_template ):
    prompt = ChatPromptTemplate.from_template(main_resume_template)
    answer_chain = prompt | model | StrOutputParser()
    return answer_chain


@chain
def hyde_retriever(question, hyde_query_transformer, retriever):
    hypothetical_document = hyde_query_transformer.invoke({"question": question})
    return retriever.invoke(hypothetical_document)

@chain
def final_chain(question, answer_chain):
    documents = hyde_retriever.invoke(question)
    for s in answer_chain.stream({"question": question, "context": documents}):
        yield s



def main():
    load_dotenv()
    #importing the Nvidia embeddings
    nvidia_embeddings = NVIDIAEmbeddings()

    #Step 1: Loading documents: passing the web based document into the loader docs

    docs = load_documents("https://docs.smith.langchain.com/user_guide")
    print("Loaded the documents")

    #Step 2: creating the vector store from the retrieveri.e., now we will create vector store from the documents which we have
    retriever = create_vector_store(docs, nvidia_embeddings)
    print("Retriever is set up")

    #Step 3: Initilizing the model
    nvidia_model = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")
    print("Loaded the model successfully")
    result = nvidia_model.invoke("Write a ballad about LangChain.")
    print(result.content)

    #Step 4: Initializing Hyde retriever
    hyde_query_transformer = create_hyde_query_transformer(nvidia_model, hyde_template)
    print("Hyde query transformer is loaded successfully")

    #Step 5: Creating Answer chain
    answer_chain = create_answerchain_withprompt(nvidia_model, main_resume_template)
    print("Answer chain is loaded")


    #Step 6: Getting the answer
    for s in final_chain.stream("how can langsmith help with testing", answer_chain):
        print(s, end="")

    






if __name__ == "__main__":

    if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
        nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
        assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
        os.environ["NVIDIA_API_KEY"] = nvidia_api_key
    
    main()