from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_qdrant import QdrantVectorStore

pdf_path = Path(__file__).parent / "nodejs.pdf"

#Loader created
loader = PyPDFLoader(file_path=pdf_path)
#This will split the pdf -> pages (it's an array)
docs = loader.load()


#now it's not good to break the pdf on basis of pages -> some pages have less data, while some have huge data, so some page content may go out of context range
#so we will use a textSplitter here, here we are uniformly defining a chunk size of 1,000 words
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200, #chunks disconnect na ho jaye -> unke pass context rhe lsat wale chunk ka bhi , overlapped chunks
)

split_docs = text_splitter.split_documents(documents=docs)

embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=""
)

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="learning_langchain",
#     embedding=embedder
# )

# vector_store.add_documents(documents=split_docs)
print("Injection Done")

retriver = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)

search_result = retriver.similarity_search(
    query="What is FS Module?"
)

print("Relevant Chunks", search_result)
