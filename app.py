import streamlit as st
import os
from main import VectorStore, QAChain

def main():
    st.set_page_config(page_title="RAG Q&A System", layout="wide")
    st.title("RAG Q&A System")

    # Initialize session state for chat history and prompt template
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'prompt_template' not in st.session_state:
        st.session_state.prompt_template = None

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Prompt template input
    default_prompt = """System message here, e.g. "You are an AI assistant that helps users to find information from the provided documents. 
General instructions: 
-	Your tone is polite.
-	Your audience are data privacy specialists.
-	In your answer always include the exact document, chapter, article and page where you found relevant data.
-	If you don't know the answer, just say that you don't know, don't try to make up an answer. 
-	Consider information from multiple documents if relevant. If the question involves a comparison, make sure to address both aspects.
 

{context}

Chat History:
{chat_history}

Question: {question}
Answer:"""
    
    prompt_template = st.sidebar.text_area("Prompt Template", value=default_prompt, height=300)

    # LLM model selection
    llm_model = st.sidebar.selectbox(
        "Select OpenAI Model",
        ["gpt-4o", "gpt-4o-mini", "gpt-4-32k", "gpt-4", "gpt-3.5-turbo-0125", "gpt-3.5-turbo"]
    )

    # Temperature slider
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

    # Button to start a new conversation (moved to main area)
    if st.button("Start New Conversation", key="new_conversation"):
        st.session_state.chat_history = []
        st.session_state.prompt_template = prompt_template
        if 'qa_chain' in st.session_state:
            del st.session_state.qa_chain
        st.rerun()

    # Initialize or update VectorStore and QAChain
    if 'qa_chain' not in st.session_state or st.session_state.prompt_template != prompt_template or st.session_state.qa_chain.llm_model != llm_model or st.session_state.qa_chain.temperature != temperature:
        vector_store = VectorStore("./chroma_db")
        vector_store.initialize_embeddings()
        retriever = vector_store.get_retriever()
        st.session_state.qa_chain = QAChain(retriever, prompt_template, llm_model, temperature)
        st.session_state.prompt_template = prompt_template

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if question := st.chat_input("Ask a question"):
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            response = st.session_state.qa_chain.run(question)
            st.markdown(response['answer'])
            st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})

        # Display references
        with st.expander("References"):
            for ref in response['references']:
                st.write(f"Source: {ref['source']}")
                st.write(f"Content: {ref['content']}")
                st.write("---")

if __name__ == "__main__":
    main()
