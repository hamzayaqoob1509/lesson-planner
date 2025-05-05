from openai import OpenAI
import tiktoken
import os
from pinecone import Pinecone, ServerlessSpec
from django.conf import settings
from decouple import config

# Load API keys from environment or settings
OPENAI_API_KEY = config("OPENAI_API_KEY")
PINECONE_API_KEY = config("PINECONE_API_KEY")
PINECONE_ENV = config("PINECONE_ENV", "us-east-1")
PINECONE_INDEX = "lesson-index"

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if PINECONE_INDEX not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=3072,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=PINECONE_ENV)
    )
index = pc.Index(PINECONE_INDEX)

# Get tokenizer for the model
tokenizer = tiktoken.encoding_for_model("text-embedding-3-large")

def chunk_text(text, max_tokens=1000):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = tokens[start:end]
        chunks.append(tokenizer.decode(chunk))
        start = end
    return chunks

def embed_text_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        response = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=chunk
        )
        embedding = response.data[0].embedding
        embeddings.append((chunk, embedding))
    return embeddings

def store_document_in_pinecone(document):
    text = document.content
    chunks = chunk_text(text)
    embedded_chunks = embed_text_chunks(chunks)

    # Store each chunk as a vector in Pinecone with metadata
    for i, (chunk, embedding) in enumerate(embedded_chunks):
        index.upsert(vectors=[{
            "id": f"{document.id}-{i}",
            "values": embedding,
            "metadata": {
                "document_id": str(document.id),
                "chunk_index": i,
                "text": chunk  # Optional: store only first 1000 characters
            }
        }])
def search_similar_chunks(query, top_k=5, use_gpt=False):
    # Step 1: Embed the query
    response = openai.Embedding.create(
        input=query,
        model="text-embedding-3-large"
    )
    query_vector = response["data"][0]["embedding"]

    # Step 2: Search Pinecone
    search_results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    chunks = [
        {
            "text": match["metadata"]["text"],
            "score": match["score"]
        }
        for match in search_results["matches"]
    ]

    if not use_gpt:
        return chunks

    # Step 3: Use GPT-4o-mini to synthesize answer or rerank
    context = "\n\n".join([f"- {chunk['text']}" for chunk in chunks])

    prompt = f"""You are a helpful assistant. Based on the following context, answer the user's question:

User question: {query}

Context:
{context}

Answer:"""

    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based on context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    answer = completion.choices[0].message.content.strip()

    return {
        "answer": answer,
        "chunks": chunks
    }


def extract_text_from_file(uploaded_file):
    name = uploaded_file.name
    ext = os.path.splitext(name)[-1].lower()

    if ext == '.txt':
        return uploaded_file.read().decode('utf-8', errors='ignore')
    
    elif ext == '.docx':
        import docx
        from io import BytesIO
        doc = docx.Document(BytesIO(uploaded_file.read()))
        return '\n'.join([p.text for p in doc.paragraphs])
    
    elif ext == '.pdf':
        import PyPDF2
        from io import BytesIO
        reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
        return "\n".join(page.extract_text() or '' for page in reader.pages)

    else:
        raise ValueError("Unsupported file type.")
