import chromadb
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import uuid
from typing import List

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. Please check your .env file."
    )


def load_headphone_buying_guide():
    with open("data/headphone_buying_guide.txt", "r", encoding="utf-8") as file:
        content = file.read()
    print(f"Loaded headphone guide with {len(content)} characters")
    doc = Document(
        page_content=content, metadata={"source": "headphone_buying_guide.txt"}
    )
    return doc


def chunk_content(doc: Document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
    )

    chunks = text_splitter.split_documents([doc])
    return chunks


def store_in_chroma(chunks: List[Document]):
    client = chromadb.PersistentClient(path="./vector_storage")

    embedding_function = OpenAIEmbeddingFunction(
        api_key=openai_api_key, model_name="text-embedding-3-small"
    )

    try:
        # Try to get existing collection
        collection = client.get_collection(
            name="headphone_buying_guide",
            embedding_function=embedding_function,  # type: ignore
        )
        print("Using existing collection")
    except ValueError:
        # Collection doesn't exist, create it
        collection = client.create_collection(
            name="headphone_buying_guide",
            embedding_function=embedding_function,  # type: ignore
        )
        print("Created new collection")

    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        documents.append(chunk.page_content)
        metadatas.append(
            {
                "source": chunk.metadata.get("source", "unknown"),
                "chunk_index": i,
                "chunk_size": len(chunk.page_content),
            }
        )
        ids.append(str(uuid.uuid4()))

    # Add documents to collection
    try:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"Successfully stored {len(documents)} chunks in ChromaDB")

        # Print collection info
        collection_count = collection.count()
        print(f"Total documents in collection: {collection_count}")

    except Exception as e:
        print(f"Error storing chunks: {e}")


def main():
    headphone_doc = load_headphone_buying_guide()
    chunks = chunk_content(headphone_doc)
    print(f"Created {len(chunks)} chunks")
    store_in_chroma(chunks)
    print("Process completed!")


if __name__ == "__main__":
    main()
