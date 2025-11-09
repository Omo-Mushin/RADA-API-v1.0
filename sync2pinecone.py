import os
import re
import firebase_admin
from firebase_admin import credentials, firestore
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm.auto import tqdm
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
# from API_STORE import PINECONE_API_KEY

# -------------------------------
# Config
# -------------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
FIREBASE_CRED_PATH = "cred_2.json"
COLLECTION_NAME = "rada_chatbot_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Pinecone config
PINECONE_API_KEY = PINECONE_API_KEY
PINECONE_ENV = "us-east1-gcp"
INDEX_NAME = COLLECTION_NAME.lower().replace("_", "-")

# -------------------------------
# Firebase Init
# -------------------------------
def init_firebase():
    cred = credentials.Certificate(FIREBASE_CRED_PATH)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    print("✅ Firebase initialized.")
    return firestore.client()

# -------------------------------
# Model Init
# -------------------------------
def init_model():
    print(f"✅ Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)

# -------------------------------
# Pinecone Init
# -------------------------------
def init_pinecone(dimension):
    """Initialize Pinecone index (serverless)"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if it doesn't exist
    existing = [i['name'] for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"🆕 Creating Pinecone index '{INDEX_NAME}' (dim={dimension})")
        pc.create_index(
            name=INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    else:
        print(f"✅ Pinecone index '{INDEX_NAME}' already exists")

    index = pc.Index(INDEX_NAME)
    return index

# -------------------------------
# Clean Text Utility
# -------------------------------
def clean_text(value):
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(round(value, 4))
    return str(value).replace("\n", " ").strip()

# -------------------------------
# Extract All Fields Recursively
# -------------------------------
def extract_all_fields(data, prefix=""):
    """Recursively extract all fields from nested dictionaries"""
    result = {}
    for key, value in data.items():
        full_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            result.update(extract_all_fields(value, full_key))
        elif isinstance(value, (list, tuple)):
            result[full_key] = ", ".join([str(v) for v in value])
        else:
            result[full_key] = value
    return result

# -------------------------------
# Flatten Record
# -------------------------------
def flatten_record(collection_name, doc_id, data):
    """Converts a Firestore document into text for embedding and structured metadata"""
    flat_data = extract_all_fields(data)

    metadata = {"collection": collection_name, "doc_id": doc_id}
    for key, value in flat_data.items():
        if value is not None:
            metadata[key] = str(value)[:500]

    # Build text representation for embedding
    text_parts = [f"Collection: {collection_name}"]
    
    priority_fields = [
        'asset', 'assetName', 'field', 'flowStation', 'flowstation',
        'well', 'reservoir', 'drainagePoint', 'date', 'productionDate',
        'listId', 'spudDate', 'onStreamDate'
    ]
    for field in priority_fields:
        for key, value in flat_data.items():
            if field.lower() in key.lower() and value is not None:
                text_parts.append(f"{key}: {clean_text(value)}")

    production_keywords = ['production', 'oil', 'gas', 'water', 'bsw', 'gor', 'pressure', 'volume', 'gross', 'net']
    for key, value in flat_data.items():
        if any(keyword in key.lower() for keyword in production_keywords):
            if value is not None and key not in [p for p in priority_fields]:
                text_parts.append(f"{key}: {clean_text(value)}")
    
    for key, value in flat_data.items():
        formatted_line = f"{key}: {clean_text(value)}"
        if formatted_line not in text_parts and value is not None:
            text_parts.append(formatted_line)
    
    text = "\n".join(text_parts)
    return text, metadata

# -------------------------------
# Fetch All Firestore Data
# -------------------------------
def fetch_all_firestore_data(db):
    print("📦 Fetching all collections from Firestore...")
    all_records = []

    collections = list(db.collections())
    print(f"🔍 Found {len(collections)} collections.\n")

    for col in collections:
        docs = list(col.stream())
        print(f"🔹 {col.id}: {len(docs)} documents")

        for doc in tqdm(docs, desc=f"Processing {col.id}"):
            data = doc.to_dict()
            if not data:
                continue
            
            text, metadata = flatten_record(col.id, doc.id, data)
            all_records.append({
                "id": f"{col.id}_{doc.id}",
                "text": text,
                "metadata": metadata,
            })

    print(f"\n✅ Prepared {len(all_records)} total records for embedding.")
    return all_records

# -------------------------------
# Upsert to Pinecone
# -------------------------------
def upsert_to_pinecone(index, model, records, batch_size=64):
    print("🚀 Embedding and uploading records to Pinecone...")
    
    for i in tqdm(range(0, len(records), batch_size), desc="Embedding batches"):
        batch = records[i:i + batch_size]
        texts = [r["text"] for r in batch]
        embeddings = model.encode(texts).tolist()
        
        clean_metadatas = []
        for r in batch:
            clean_meta = {}
            for k, v in r["metadata"].items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            clean_metadatas.append(clean_meta)
        
        vectors = [
            {"id": r["id"], "values": emb, "metadata": meta}
            for r, emb, meta in zip(batch, embeddings, clean_metadatas)
        ]
        index.upsert(vectors=vectors)
    
    print("✅ All data successfully embedded in Pinecone")

# -------------------------------
# Debug Preview
# -------------------------------
def debug_preview(records, n=3):
    print("\n🧠 Sample Records for Preview:")
    sample = records[:n]
    for i, r in enumerate(sample):
        print(f"\n{'='*80}\n--- Record #{i+1} ---\n{'='*80}")
        print(r["text"][:500] + "..." if len(r["text"]) > 500 else r["text"])
        print("\nMetadata (first 10 fields):")
        meta_items = list(r["metadata"].items())[:10]
        for k, v in meta_items:
            print(f"  {k}: {v}")

# -------------------------------
# Main
# -------------------------------
def main():
    print("="*80)
    print("🚀 STARTING FIRESTORE → PINECONE INGESTION")
    print(f"📁 Target Index: {INDEX_NAME}")
    print("="*80 + "\n")
    
    db = init_firebase()
    model = init_model()
    dimension = model.get_sentence_embedding_dimension()
    index = init_pinecone(dimension)
    
    records = fetch_all_firestore_data(db)
    if not records:
        print("⚠️ No data found in Firestore.")
        return

    upsert_to_pinecone(index, model, records)
    debug_preview(records)
    
    print("\n" + "="*80)
    print("✅ INGESTION COMPLETE")
    print(f"📊 Total records embedded: {len(records)}")
    print(f"📁 Pinecone Index: {INDEX_NAME}")
    print("="*80)

if __name__ == "__main__":
    main()
