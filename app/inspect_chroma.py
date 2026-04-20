import chromadb

client = chromadb.PersistentClient(path="./rag_store")
collection = client.get_collection("langchain")  # change if your collection name is different

results = collection.get(
    limit=3,
    include=["embeddings", "documents", "metadatas"]
)

print("IDs:", results["ids"])
print("First vector length:", len(results["embeddings"][0]))
print("First 10 values of first vector:", results["embeddings"][0][:10])
print("First metadata:", results["metadatas"][0])
print("First document:", results["documents"][0])