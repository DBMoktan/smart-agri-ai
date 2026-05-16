import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
DATA_PATH = "data/raw/"
DB_PATH = "data/vector_db/"

def create_vector_db():
    print(f"--- Starting Data Ingestion from {DATA_PATH} ---")
    
    # 1. Load PDFs manually to show progress
    print("Step 1: Loading PDF documents...")
    all_documents = []
    
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in data/raw!")
        return

    for pdf_file in pdf_files:
        file_path = os.path.join(DATA_PATH, pdf_file)
        print(f"  -> Processing: {pdf_file}...")
        try:
            loader = PyPDFLoader(file_path)
            all_documents.extend(loader.load())
        except Exception as e:
            print(f"  [!] Error loading {pdf_file}: {e}")

    print(f"Total pages extracted: {len(all_documents)}")

    # 2. Split Text into Chunks
    print("Step 2: Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_documents)
    print(f"Created {len(chunks)} text chunks.")

    # 3. Create Embeddings
    print("Step 3: Generating embeddings (this may take a minute)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # 4. Store in ChromaDB
    print(f"Step 4: Creating Vector Store at {DB_PATH}...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print("--- Vector Database Created Successfully! ---")
    return vector_db

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created {DATA_PATH}. Please put your PDFs there.")
    else:
        create_vector_db()
