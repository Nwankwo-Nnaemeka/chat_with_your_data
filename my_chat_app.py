import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
#from HtmlFile import css, user_template, bot_template
from PyPDF2 import PdfReader
from dotenv import load_dotenv, find_dotenv
import os
import openai

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.pinimg.com/564x/09/b8/8d/09b88d21f070c54635a2bc0c3c079ddd.jpg" >
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''


def get_pdf_document(files):
  #loader = PyPDFLoader(files)
  #pages = loader.load()
  text = ""
  for filex in files:
    pdf_reader = PdfReader(filex)
    for page in pdf_reader.pages:
      text += page.extract_text()
  return text

def split_and_get_chunks(documents,chunk_size=150, chunk_overlap=20):
  splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
  # splitter.split_documents
  # splitter.create_documents
  splitted_documents = splitter.split_text(documents)
  return splitted_documents


def create_vectore_stores(splitted_documents, directory="embeddings_vec/chroma/"):
  embeddings = OpenAIEmbeddings()
  PERSIST_DIRECTORY = directory
  # Chroma.from_documents
  vector_db = Chroma.from_texts(texts = splitted_documents,
                                    embedding=embeddings,
                                    persist_directory=PERSIST_DIRECTORY
                                    )
  return vector_db

def conversationQa(vector_store, search_type="mmr", k=4, fetch_k=20, chain_type = "stuff"):
  memory = ConversationBufferMemory(memory_key = "chat_history",
                                      return_messages = True)

  llm = ChatOpenAI(temperature=0)
  conversation_chain = ConversationalRetrievalChain.from_llm(
      llm,
      memory = memory,
      retriever = vector_store.as_retriever(search_type= search_type, search_kwargs = {"k":k, "fetch_k":fetch_k}),
      chain_type = chain_type
  )
  return conversation_chain

def question_and_answering(user_input):
  answer = st.session_state.conversations({"question": user_input})
  st.session_state.chat_history = answer["chat_history"]
  for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
  _ = load_dotenv(find_dotenv())

  openai.api_key  = os.environ['OPENAI_API_KEY']
  st.set_page_config(page_title = "Chat WIth Your Data", page_icon=":books:")
  st.write(css, unsafe_allow_html = True)

  if "conversations" not in st.session_state:
    st.session_state.conversations= None
  if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

  st.header("CHAT WITH YOUR DATA :books:")
  user_question = st.text_input("Ask a question about your documents")
  if user_question:
    question_and_answering(user_question)

  with st.sidebar:
    st.subheader("Your Documents")
    pdf_documents = st.file_uploader("upload your pdf files and click on 'Load' ", accept_multiple_files=True)
    if st.button("Load"):
      with st.spinner("loading"):
        # get pdf
        docs = get_pdf_document(pdf_documents)
        # createchunks
        chunks = split_and_get_chunks(docs)
        # vector store
        vectors = create_vectore_stores(chunks)
        # retrieve and chat
        st.session_state.conversations = conversationQa(vectors)


if __name__ == "__main__":
  main()
