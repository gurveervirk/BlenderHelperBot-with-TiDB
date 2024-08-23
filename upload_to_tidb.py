from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.tidbvector import TiDBVectorStore
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext
import json
import sys

def load_settings():
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("The settings file doesn't exist. Creating a new one...")
        settings = {"connectionString": "connString/from/tidb/cloud?ssl_ca=complete/path/to/isrgrootx1.pem"}
        with open('settings.json', 'w+') as f:
            json.dump(settings, f)
        return settings

if __name__ == "__main__":
    folder = sys.argv[1]

    settings = load_settings()
    # Load the embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="mixedbread-ai/mxbai-embed-large-v1", device="cuda")
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    VECTOR_TABLE_NAME = "default"
    tidbvec = TiDBVectorStore(
        connection_string=settings["connectionString"],
        table_name=VECTOR_TABLE_NAME,
        distance_strategy="cosine",
        vector_dimension=1024,
        drop_existing_table=False
    )
    vector_index = VectorStoreIndex.from_vector_store(vector_store=tidbvec)
    storage_context = StorageContext.from_defaults(vector_store=tidbvec)

    documents = SimpleDirectoryReader(folder).load_data()
    vector_index = vector_index.from_documents(documents, storage_context=storage_context, show_progress=True)