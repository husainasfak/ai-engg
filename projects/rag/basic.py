import chromadb
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set")


client = OpenAI(api_key=openai_api_key)

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# Initialize ChromaDB (in-memory client; data is not persisted to disk)
chroma = chromadb.Client()
collection = chroma.create_collection(name="my_docs")

# Sample documents about a fictional company
documents = [
    "Acme Corp offers a 30-day return policy for all physical products.",
    "Digital products can be refunded within 7 days of purchase.",
    "Premium subscribers get priority support with a 4-hour response time.",
    "Free-tier users have access to email support with 48-hour response times.",
    "Acme Corp was founded in 2019 and is headquartered in Austin, Texas.",
    "The enterprise plan includes dedicated account management and SLA guarantees.",
    "Student discounts of 20% are available with a valid .edu email address.",
    "Acme Corp processes refunds within 5-7 business days to the original payment method.",
    "The API rate limit for free-tier users is 100 requests per minute.",
    "Enterprise customers can request custom rate limits through their account manager.",
]

# Step 1: Embed and index all documents
response = client.embeddings.create(input=documents, model=EMBEDDING_MODEL)
embeddings = [d.embedding for d in response.data]

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)



# Step 2: Define the query function
def ask(question:str, n_results:int = 3)->str:
    q_embeddings = client.embeddings.create(
        input=question, model=EMBEDDING_MODEL
    ).data[0].embedding
    
    results = collection.query(query_embeddings=[q_embeddings], n_results=n_results)

    context = "\n".join(results["documents"][0])

    # Generate answer using retrieved context
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question based only on the provided context. "
                    "If the context does not contain the answer, say "
                    "'I don't have that information.'"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content



# Step 3: Ask questions
print(ask("What is the refund policy for digital products?"))
# "Digital products can be refunded within 7 days of purchase.
#  Refunds are processed within 5-7 business days to the original
#  payment method."

print(ask("Do you offer student discounts?"))
# "Yes, Acme Corp offers a 20% student discount with a valid .edu
#  email address."

print(ask("What programming languages does Acme support?"))