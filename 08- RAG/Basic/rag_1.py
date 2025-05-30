#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 🧠 Step 0: Import Required Libraries
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader  # For loading PDF content
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text into chunks
from langchain_openai import OpenAIEmbeddings  # For creating text embeddings using OpenAI
from langchain_qdrant import QdrantVectorStore  # For storing and retrieving vectors using Qdrant DB

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 🗂️ Step 1: Load the PDF File
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Set the path to your PDF file
pdf_path = Path(__file__).parent / "nodejs.pdf"

# Initialize the PDF loader
loader = PyPDFLoader(file_path=pdf_path)

# Load and split the PDF into pages (each page = one document initially)
docs = loader.load()

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ✂️ Step 2: Split the Documents into Uniform Chunks
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

#now it's not good to break the pdf on basis of pages -> some pages have less data, while some have huge data, so some page content may go out of context range
#so we will use a textSplitter here, here we are uniformly defining a chunk size of 1,000 words
# Reason: Pages can vary in length, which might break the context.
# So we split them based on characters rather than page boundaries.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200, #chunks disconnect na ho jaye -> unke pass context rhe lsat wale chunk ka bhi , overlapped chunks
)
#here we are doing the dumbest thing ki character se split kr doh -> 1000 mei bhi context break ho jayega -> this is the naive solution  


#we will get the splitted documents,  Split documents into smaller, manageable chunks
split_docs = text_splitter.split_documents(documents=docs)

#we can do this for finding the length of the doc and the chunk
print("DOCS", len(docs));
print("SPLIT", len(split_docs));

#NOtes ke mind map mei -> yaha tk we did data source, and chunking,
#now ww have to do embeddings ->

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 🧠 Step 3: Generate Embeddings for Each Chunk
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

# We’ll convert text chunks into embeddings (numeric vector representations)

# Initialize OpenAI Embeddings (replace with your actual API key)
#Embedder -> iske through we will do embeddings like a loder 
embedder = OpenAIEmbeddings(
    model="text-embedding-3-large", #OpenAI embedding models are diff thna usual 
    api_key=""
)
#now ihave to embedding -> and store in the qdrant db

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 💾 Step 4: Store Embeddings in Qdrant Vector Database
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Vector DB lets us do similarity search efficiently.
vector_store = QdrantVectorStore.from_documents(
    documents=[],  # Initially empty (you can load directly or add later)
    url="http://localhost:6333",  # Qdrant server address
    collection_name="learning_langchain",  # Like a table name in DB
    embedding=embedder
)

# Add the split documents (with embeddings) to the vector store
vector_store.add_documents(documents=split_docs)
print("Injection Part Done")
#ingestion tk ho gya kaam

#Now we are going in th retrieval part-> 

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 🔍 Step 5: Retrieve Relevant Chunks Using Similarity Search
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Now we query the vector store using a natural language question.

# Load the same Qdrant collection for retrieval
retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)

# Perform similarity search: this query gets embedded and compared with stored vectors
search_result = retriver.similarity_search(
    query="What is FS Module?"  #yeh internally iski bhi vector embeddingbnayaega , so yeh embed ko vector store se similarity search kr layega and give us the relevant chunks
)

print("Relevant Chunks", search_result)

#AB yeh relevant chunks ko system prompt mei dalenge and we will get the output 
