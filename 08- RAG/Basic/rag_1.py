from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_qdrant import QdrantVectorStore

pdf_path = Path(__file__).parent / "nodejs.pdf"

#Loader created
loader = PyPDFLoader(file_path=pdf_path)
#This will split the pdf -> pages (it's an array)
#it will load the pdf
docs = loader.load()


#now it's not good to break the pdf on basis of pages -> some pages have less data, while some have huge data, so some page content may go out of context range
#so we will use a textSplitter here, here we are uniformly defining a chunk size of 1,000 words
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200, #chunks disconnect na ho jaye -> unke pass context rhe lsat wale chunk ka bhi , overlapped chunks
)

#we will get the splitted documents
split_docs = text_splitter.split_documents(documents=docs)

#we can do this for finding the length of the doc and the chunk
print("DOCS", len(docs));
print("SPLIT", len(split_docs));

#NOtes ke mind map mei -> yaha tk we did data source, and chunking,
#now ww have to do embeddings ->

#Embedder -> iske through we will do embeddings like a loder 
embedder = OpenAIEmbeddings(
    model="text-embedding-3-large", #OpenAI embedding models are diff thna usual 
    api_key=""
)
#now ihave to embedding -> and store in the qdrant db


vector_store = QdrantVectorStore.from_documents(
    documents=[],
    url="http://localhost:6333",
    collection_name="learning_langchain",  #like db has tables, we can give any name 
    embedding=embedder
)

vector_store.add_documents(documents=split_docs)
print("Injection Part Done")
#ingestion tk ho gya kaam

#Now we are going in th retrieval part-> 

retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)

search_result = retriver.similarity_search(
    query="What is FS Module?"  #yeh internally iski bhi vector embeddingbnayaega , so yeh embed ko vector store se similarity search kr layega and give us the relevant chunks
)

print("Relevant Chunks", search_result)
