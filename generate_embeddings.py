from sentence_transformers import SentenceTransformer



model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")

document_embeddings = model.encode(a)
print(document_embeddings.shape)
