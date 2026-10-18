import os
import shutil
from dotenv import load_dotenv

# Load environment variables (including LangSmith keys)
load_dotenv()

# Verify LangSmith tracing status
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print(f"LangSmith Tracing enabled for project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.output_parsers import StrOutputParser

api_key = os.getenv("OPENAI_API_KEY")
DATA_FILE = "schemes_data.txt"
UPLOAD_FOLDER = "./uploaded_docs"
CHROMA_DIR = "./chroma_db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

rag_chain = None
vectorstore = None

def format_docs_with_sources(docs):
    formatted = []
    for doc in docs:
        source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
        page_num = doc.metadata.get("page", None)
        header = f"[{source_name} - Page {page_num + 1}]" if page_num is not None else f"[{source_name}]"
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(formatted)

def build_knowledge_base():
    global rag_chain, vectorstore

    all_docs = []

    # 1. Load Scraped Web Data
    if os.path.exists(DATA_FILE):
        try:
            loader = TextLoader(DATA_FILE, encoding="utf-8")
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {DATA_FILE}: {e}")

    # 2. Load all Uploaded PDFs
    indexed_pdfs = []
    if os.path.exists(UPLOAD_FOLDER):
        for fname in os.listdir(UPLOAD_FOLDER):
            if fname.lower().endswith(".pdf"):
                pdf_path = os.path.join(UPLOAD_FOLDER, fname)
                try:
                    pdf_loader = PyPDFLoader(pdf_path)
                    all_docs.extend(pdf_loader.load())
                    indexed_pdfs.append(fname)
                except Exception as e:
                    print(f"Error loading {fname}: {e}")

    if not all_docs:
        return {"status": "empty", "total_chunks": 0, "pdfs": []}

    # 3. Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_docs)

    # 4. Embeddings & Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    
    if os.path.exists(CHROMA_DIR):
        try:
            shutil.rmtree(CHROMA_DIR)
        except Exception:
            pass

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)

    system_prompt = (
        "You are TN AgriBOT, an AI agricultural assistant specializing in Tamil Nadu government farming schemes, subsidies, and official guidelines.\n\n"
        "Follow these strict operating instructions:\n"
        "1. Grounding: Answer strictly using ONLY the retrieved context below (portal web data and uploaded PDFs). Do not invent or assume facts.\n"
        "2. Missing Info: If the answer cannot be found in the context, respond clearly: "
        "'The requested details are not available in the uploaded schemes and documents.'\n"
        "3. Specific Figures: State exact financial values, subsidy rates, and machinery charges directly as reported in the context.\n"
        "4. Citations: When referencing data from files or PDFs, cite the source name or page number shown in the context header (e.g., [filename.pdf - Page X])."
    )

    human_prompt = (
        "Context from Agriculture Documents & Scraped Portal:\n"
        "---------------------\n"
        "{context}\n"
        "---------------------\n\n"
        "Question: {question}\n\n"
        "Helpful & Grounded Answer:"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    # Assign runnable names for clear trace visualizations in LangSmith
    rag_chain = (
        {
            "context": (retriever | format_docs_with_sources).with_config({"run_name": "RetrieveAgricultureDocs"}),
            "question": RunnablePassthrough()
        }
        | prompt.with_config({"run_name": "AgriBOTPromptBuilder"})
        | llm.with_config({"run_name": "OpenAIGeneration"})
        | StrOutputParser().with_config({"run_name": "ParseResponse"})
    ).with_config({"run_name": "TN-AgriBOT-MainPipeline"})

    return {
        "status": "success",
        "total_chunks": len(split_docs),
        "pdfs": indexed_pdfs
    }

def reset_knowledge_base():
    global rag_chain, vectorstore
    rag_chain = None
    vectorstore = None
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    if os.path.exists(UPLOAD_FOLDER):
        for f in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    return {"status": "cleared"}

def ask_question(query: str):
    global rag_chain
    if rag_chain is None:
        build_knowledge_base()
    if rag_chain is None:
        return {"answer": "Knowledge base is empty. Please upload PDFs or run the scraper first."}

    # Pass tracing tags and metadata into LangSmith for tracking
    config: RunnableConfig = {
        "tags": ["tamil-nadu-agriculture", "rag-bot"],
        "metadata": {
            "source_type": "hybrid_web_and_pdf",
            "model": "gpt-4o-mini"
        }
    }
    
    answer = rag_chain.invoke(query, config=config)
    return {"answer": answer}

if os.path.exists(DATA_FILE) or (os.path.exists(UPLOAD_FOLDER) and os.listdir(UPLOAD_FOLDER)):
    try:
        build_knowledge_base()
    except Exception as e:
        print(f"Initial index build postponed: {e}")