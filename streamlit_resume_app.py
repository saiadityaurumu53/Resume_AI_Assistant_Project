import streamlit as st
from dotenv import load_dotenv
#note: this is the function which we will load in the main in order for our application to use the secret variables inside the main
#Now as we have initialized the function in the first line of the main, Lang Chain will be able to use it 

from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader


from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA


from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain


#Importing the FIASS vector store and the Langchain's text splitter documents
from langchain_community.vectorstores import FAISS



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        #now initializing the PDF reader object
        pdf_reader = PdfReader(pdf)
        #here, the pdf object that has pages is given
        #Now, we will read each page and add it to the text
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_document_chunks(docs):
    #we will use the LangChain
    # Initialize a text splitter to divide documents into smaller chunks with specified size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    # Split the loaded documents into chunks
    documents = text_splitter.split_documents(docs)
    return documents


def get_vectorstore(text_chunks):
    # Initialize the NVIDIA Embeddings module
    embeddings = NVIDIAEmbeddings()
    #embeddings = HugginFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # Initialize a FAISS vector store from the document chunks and embeddings
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    return vectorstore.as_retriever()

def create_chain(retriver, llm):
    retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=retriver)

@st.cache_resource
def load_chain(retriever):
    # Initialize the ChatNVIDIA model with a specified model version
    model = ChatNVIDIA(model="mistral_7b")

    # Define a template for generating hypothetical answers to questions
    hyde_template = """Generate a one-paragraph hypothetical answer to the below question:
    {question}"""

    # Initialize a prompt template from the hypothetical answer template
    hyde_prompt = ChatPromptTemplate.from_template(hyde_template)

    # Chain the prompt template with the model and a string output parser to form a query transformer
    hyde_query_transformer = hyde_prompt | model | StrOutputParser()

    # Define a chainable function to generate hypothetical documents based on a question!
    @chain
    def hyde_retriever(question):
        hypothetical_document = hyde_query_transformer.invoke({"question": question})
        return retriever.invoke(hypothetical_document)

    # Define a template for answering questions based on provided context
    template = """Answer the question strictly based on the following context:
    {context}

    Question: {question}
    """

    # Initialize a prompt template from the answer template
    prompt = ChatPromptTemplate.from_template(template)

    # Chain the prompt template with the model and a string output parser to form an answer chain
    answer_chain = prompt | model | StrOutputParser()

    # Define a chainable function that generates answers based on a question and context
    @chain
    def final_chain(question):
        # Retrieve documents using the hyde_retriever
        documents = hyde_retriever.invoke(question)

        # Stream answers using the answer_chain and yield them one by one
        for s in answer_chain.stream({"question": question, "context": documents}):
            yield s

    # Execute the final chain with a specific question and print the results
    for s in final_chain.stream("What is Nvidia NIM and what are some of its benefits?"):
        print(s, end="")
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    #st.write(css, unsafe_allow_html=True)






def handle_user_question(user_question):
    response = st.session_state.conversation({'question': user_question})
    #st.write(response)
    #we are going to take this object right here and we are going to format it for the template which we will have 
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if (i%2 == 0):
            #here we will write the user input template
            message1 = st.chat_message("user")
            message1.write(message.content)
        else:
            #here we will display the bot template
            message2 = st.chat_message("assistant")
            message2.write(message.content)


def main():
    load_dotenv()

    load_chain()

    

    
    #this variable of conversation is created in the session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None   

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None 

    st.header("Chat with multiple PDFs :books:")

    #below the header we want the user to have the user to type inputs so we will 
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_user_question(user_question)

    


    #now we also want a side bar where the user is going to upload the documents
    #Also if we want to keep or put things inside it then we are gonna use the with keyword
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        #now we will create a button with the name and whose functionality is to process
        #Here, the multiple files are stored here in the pdf_docs variable

        if st.button("Process"):
            with st.spinner("Processing"):
                #Now if the button is pressed then this if block is then executed

                #Step1:  we are going to get the pdf text : Here, we are gonna take the raw text from our pdfs
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text)

                #Step2: get the text chunks
                text_chunks = get_document_chunks(raw_text)
                st.write(text_chunks)

                #Step 3: Create the vector store with the Embeddings
                vectorstore = get_vectorstore(text_chunks)

                #Step 4: Create Conversational Chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()