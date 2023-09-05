import streamlit as st

import langchain, pinecone
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
#import environment variables
import PyPDF2
import uuid


# Streamlit Setup
st.title('Search Demo')
openai_api_key = st.secrets['OPENAI_API_KEY']
pinecone_api_key = st.secrets['PINECONE_API_KEY']

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp-free")

# Create a variable to hold the current index name
current_index_name = None

uploaded_file = st.file_uploader("Choose a text or PDF file", type=['txt', 'pdf'])

if uploaded_file is not None:
    unique_id = uuid.uuid4().hex  # Generate a unique id
    new_index_name = f'testsearchbook-{unique_id}'  # Create a unique index name

    # Delete existing index if there is one
    if current_index_name is not None and current_index_name in pinecone.list_indexes():
        pinecone.delete_index(current_index_name)

    # Update the current index name
    current_index_name = new_index_name

    # Create a new index
    pinecone.create_index(name=current_index_name, metric='cosine', shards=1)
    
    if uploaded_file.type == "application/pdf":
        file_content = read_pdf(uploaded_file)
    else:
        file_content = uploaded_file.read().decode('utf-8')

    # ... (rest of your code)


    # Split the file content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0,
        length_function=len,
    )
    book_texts = text_splitter.create_documents([file_content])

    # Create Pinecone index
    with get_openai_callback() as cb:
        book_docsearch = Pinecone.from_texts([t.page_content for t in book_texts], embeddings, index_name=index_name)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def generate_response(input_text):
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=300)
        with get_openai_callback() as cb:
            query = input_text
            docs = book_docsearch.similarity_search(query,k=2)
            chain = load_qa_chain(llm, chain_type="stuff")
            answers = chain.run(input_documents=docs, question=query)
            
            # Append user's query and bot's answer to the session state messages
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "assistant", "content": answers})  # Make sure 'answers' is in a displayable format
    
    
    
    # Form for question input
    with st.form('my_form'):
        text = st.text_area('Enter your query:', 'Type your query here...')
        submitted = st.form_submit_button('Submit')
        if submitted:
            generate_response(text)

    # Display existing chat messages AFTER form submission
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
