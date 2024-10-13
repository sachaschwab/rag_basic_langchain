import os
from typing import Dict, Type
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredHTMLLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
import json
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

class DocumentProcessor:
    def __init__(self, directory: str):
        self.directory = directory
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: '{directory}'")
        if not os.path.isdir(directory):
            raise ValueError(f"Expected directory, got file: '{directory}'")
        self.loaders: Dict[str, Type] = {
            '.pdf': PyPDFLoader,
            '.html': UnstructuredHTMLLoader,
            '.docx': Docx2txtLoader
        }

    def load_documents(self):
        loader = DirectoryLoader(self.directory, glob="**/*", loader_cls=lambda file_path: self.loaders[os.path.splitext(file_path)[1]](file_path))
        return loader.load()

    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)

class VectorStore:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.embeddings = None
        self.vectorstore = None

    def initialize_embeddings(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            api_key = input("Please enter your OpenAI API key: ").strip()
            os.environ["OPENAI_API_KEY"] = api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    def create_vectorstore(self, texts):
        self.vectorstore = Chroma.from_documents(texts, self.embeddings, persist_directory=self.persist_directory)
        return self.vectorstore

    def get_retriever(self, search_kwargs={"k": 3}):
        if self.vectorstore is None:
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def persist(self):
        if self.vectorstore:
            self.vectorstore.persist()

class QAChain:
    def __init__(self, retriever):
        prompt_template = """You are a legal assistant. Use the following pieces 
        of context to answer the question at the end. If you don't know the answer, 
        just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Specify the output key here
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0),
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True
        )

    def run(self, query: str):
        result = self.qa_chain({"question": query})
        answer = result['answer']
        source_documents = result['source_documents']
        references = [{"source": doc.metadata.get("source", "Unknown"), "content": doc.page_content[:100]} for doc in source_documents]
        return json.dumps({"answer": answer, "references": references})

def main():
    # Ask user whether to load documents newly
    load_new = input("Do you want to load the documents newly? (yes/no): ").lower().strip() == 'yes'

    vector_store = VectorStore("./chroma_db")
    vector_store.initialize_embeddings()

    if load_new:
        # Initialize and use the classes
        documents_dir = 'documents'
        if not os.path.exists(documents_dir):
            print(f"Error: Directory not found: '{documents_dir}'")
            print("Please make sure the 'documents' directory exists in the current working directory.")
            return

        doc_processor = DocumentProcessor(documents_dir)
        documents = doc_processor.load_documents()
        texts = doc_processor.split_documents(documents)

        vectorstore = vector_store.create_vectorstore(texts)
        retriever = vector_store.get_retriever()

        # Persist the vector store
        vector_store.persist()
        print("Vector store created and persisted.")
    else:
        # Load existing vector store
        retriever = vector_store.get_retriever()
        print("Using existing vector store.")

    qa_chain = QAChain(retriever)

    # Ask the user for their question
    query = input("Please enter your question: ")
    response = qa_chain.run(query)
    print("Query:", query)
    print("Response:", response)

if __name__ == "__main__":
    main()
