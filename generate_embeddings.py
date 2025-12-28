import os

for key in ['REQUESTS_CA_BUNDLE', 'SSL_CERT_FILE', 'CURL_CA_BUNDLE']:
    if key in os.environ:
        del os.environ[key]

from sentence_transformers import SentenceTransformer

def main():
    print("✅ SSL Environment cleaned. Initializing model download...")
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
    
    print("✅ Model loaded successfully.")
    sentences = ["This is an example sentence", "Each sentence is converted"]
    embeddings = model.encode(sentences)
    print(f"Generated embeddings shape: {embeddings.shape}")

if __name__ == "__main__":
    main()