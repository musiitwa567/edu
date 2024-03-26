from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

class ChatBot():
    DB_FAISS_PATH = r'db_faiss'
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50}, huggingfacehub_api_token=os.getenv('"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"')
    )

    template = """
    You are a teacher. The students will ask you a questions about their life. Use following piece of context to answer the question.
    If you don't know the answer, just say you don't know.
    You answer with short and concise answer.

    Context: {context}
    Question: {question}
    Answer:

    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    rag_chain = (
        {"context": db.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# Outside ChatBot() class
bot = ChatBot()

import streamlit as st

st.set_page_config(page_title="Student Companion Asistant Chat Bot")
with st.sidebar:
    st.title('Student Companion Asistant Chat Bot')

# Function for generating LLM response
def generate_response(input):
    result = bot.rag_chain.invoke(input)
    return result

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, let's answer your question"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input_text := st.text_input("Ask me anything:"):
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.chat_message("user"):
        st.write(input_text)

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.spinner("Thinking about Your Question.."):
        response = generate_response(input_text)
        # Extracting only the answer from the response
        answer_index = response.find("Answer:")
        if answer_index != -1:
            answer = response[answer_index + len("Answer:"):].strip()
            message = {"role": "assistant", "content": answer}
            st.session_state.messages.append(message)
            with st.chat_message("assistant"):
                st.write(answer)
        else:
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
            with st.chat_message("assistant"):
                st.write(response)
