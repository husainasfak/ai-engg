# Approach 1 - Json Mode
#  response_format={"type": "json_object"}

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from typing import Optional
import instructor

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Set OPENAI_API_KEY before running this script.")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": "Extract product information from the review. "
                       "Return JSON with fields: product_name, price, rating, pros, cons."
        },
        {
            "role": "user",
            "content": "I bought the Sony WH-1000XM5 for $348. Amazing noise cancellation, "
                       "super comfortable for long flights. Battery lasts forever. "
                       "Only downside is the carrying case feels cheap. I'd give it 4.5/5."
        }
    ],
    max_completion_tokens=200,
)

data = json.loads(response.choices[0].message.content)
print(json.dumps(data, indent=2))


# Structured Outputs with JSON Schema

# JSON Schema mode goes further. You give the API a schema that describes the exact object you want. OpenAI supports this on compatible models through response_format with type: "json_schema".



response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "product_review",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string"},
                    "price": {"type": "number"},
                    "rating": {"type": "number"},
                    "pros": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "cons": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "recommendation": {"type": "boolean"}
                },
                "required": ["product_name", "price", "rating", "pros", "cons", "recommendation"],
                "additionalProperties": False
            }
        }
    },
    messages=[
        {
            "role": "system",
            "content": "Extract product information from the review."
        },
        {
            "role": "user",
            "content": "I bought the Sony WH-1000XM5 for $348. Amazing noise cancellation, "
                       "super comfortable for long flights. Battery lasts forever. "
                       "Only downside is the carrying case feels cheap. I'd give it 4.5/5."
        }
    ],
    max_completion_tokens=200,
)

data = json.loads(response.choices[0].message.content)
print(json.dumps(data, indent=2))




# Approach 3: Tool-Based Extraction (OpenAI Tools Format via OpenRouter)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "extract_product_review",
                "description": "Extract structured product review information from text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_name": {
                            "type": "string",
                            "description": "The name of the product"
                        },
                        "price": {
                            "type": "number",
                            "description": "Price in USD"
                        },
                        "rating": {
                            "type": "number",
                            "description": "Rating out of 5"
                        },
                        "pros": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of positive points"
                        },
                        "cons": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of negative points"
                        }
                    },
                    "required": ["product_name", "price", "rating", "pros", "cons"]
                }
            }
        }
    ],
    tool_choice={"type": "function", "function": {"name": "extract_product_review"}},
    messages=[
        {
            "role": "user",
            "content": "Extract the product info: I bought the Sony WH-1000XM5 for $348. "
                       "Amazing noise cancellation, super comfortable for long flights. "
                       "Battery lasts forever. Only downside is the carrying case feels cheap. "
                       "I'd give it 4.5/5."
        }
    ],
    max_completion_tokens=500,
)

tool_call = response.choices[0].message.tool_calls[0]
data = json.loads(tool_call.function.arguments)
print(json.dumps(data, indent=2))


# Pydantic for Validation


# class ProductReview(BaseModel):
#     product_name: str = Field(description="Name of the product")
#     price: float = Field(ge=0, description="Price in USD")
#     rating: float = Field(ge=0, le=5, description="Rating out of 5")
#     pros: list[str] = Field(description="Positive aspects")
#     cons: list[str] = Field(description="Negative aspects")
#     recommendation: bool = Field(description="Whether the reviewer recommends the product")
#     reviewer_name: Optional[str] = Field(default=None, description="Name of the reviewer if mentioned")

# valid_data = {
#     "product_name": "Sony WH-1000XM5",
#     "price": 348.0,
#     "rating": 4.5,
#     "pros": ["Great noise cancellation", "Comfortable"],
#     "cons": ["Expensive case"],
#     "recommendation": True,
# }
# review = ProductReview(**valid_data)
# print(review.model_dump_json(indent=2))

# try:
#     bad_data = {
#         "product_name": "Sony WH-1000XM5",
#         "price": -50,
#         "rating": 11,
#         "pros": "good",
#         "cons": [],
#         "recommendation": True,
#     }
#     ProductReview(**bad_data)
# except Exception as error:
#     print(f"Validation error: {error}")
    
    


# Writing the same parsing and validation code repeatedly gets old. The instructor library wraps an OpenAI-compatible client so you can pass a Pydantic model and receive a validated object.


class ProductReview(BaseModel):
    product_name: str = Field(description="Name of the product")
    price: float = Field(ge=0, description="Price in USD")
    rating: float = Field(ge=0, le=5, description="Rating out of 5")
    pros: list[str] = Field(description="Positive aspects")
    cons: list[str] = Field(description="Negative aspects")
    recommendation: bool

review = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=ProductReview,
    messages=[
        {
            "role": "user",
            "content": "I bought the Sony WH-1000XM5 for $348. Amazing noise cancellation, "
                       "super comfortable for long flights. Battery lasts forever. "
                       "Only downside is the carrying case feels cheap. I'd give it 4.5/5."
        }
    ],
    max_completion_tokens=200,
)

print(type(review))
print(review.product_name)
print(review.price)
print(review.rating)





# Schema-Driven Extraction


class Person(BaseModel):
    name: str
    role: str = Field(description="Their role or title")
    organization: str = Field(description="Organization they belong to")

class Organization(BaseModel):
    name: str
    industry: str
    headquarters: str = Field(description="City or country of headquarters, if mentioned")

class ArticleEntities(BaseModel):
    people: list[Person]
    organizations: list[Organization]
    key_events: list[str] = Field(description="Main events described in the article")

article = """
Sundar Pichai, CEO of Google, announced a new partnership with Samsung at
the Consumer Electronics Show in Las Vegas. The deal will integrate Google's
Gemini AI into Samsung's Galaxy S25 smartphone lineup. Samsung's head of
mobile division, TM Roh, called it "a milestone for on-device AI."
Apple, which was not involved in the deal, declined to comment.
"""

result = client.chat.completions.create(
    model="openai/gpt-5.4-mini",
    response_model=ArticleEntities,
    messages=[
        {"role": "user", "content": f"Extract all entities from this article:\n\n{article}"}
    ],
    max_completion_tokens=300,
)

for person in result.people:
    print(f"Person: {person.name} ({person.role} at {person.organization})")

for org in result.organizations:
    print(f"Org: {org.name} ({org.industry}, {org.headquarters})")

for event in result.key_events:
    print(f"Event: {event}")
    
    
    
    
    
    
# Extracting  relationship
class Entity(BaseModel):
    name: str
    entity_type: str = Field(description="person, organization, product, or event")

class Relationship(BaseModel):
    subject: str = Field(description="Name of the first entity")
    predicate: str = Field(description="Nature of the relationship (e.g., 'CEO of', 'partnered with')")
    object: str = Field(description="Name of the second entity")

class KnowledgeGraph(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]

article = """
Sundar Pichai, CEO of Google, announced a new partnership with Samsung at
the Consumer Electronics Show in Las Vegas. The deal will integrate Google's
Gemini AI into Samsung's Galaxy S25 smartphone lineup. Samsung's head of
mobile division, TM Roh, called it "a milestone for on-device AI."
Apple, which was not involved in the deal, declined to comment.
"""

graph = client.chat.completions.create(
    model="openai/gpt-5.4-mini",
    response_model=KnowledgeGraph,
    messages=[
        {"role": "user", "content": f"Extract entities and relationships:\n\n{article}"}
    ],
    max_completion_tokens=500,
)

for rel in graph.relationships:
    print(f"{rel.subject} --[{rel.predicate}]--> {rel.object}")
    
    
    
    
# instrcutor built in retry
client = instructor.from_openai(
    OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
)

class MovieReview(BaseModel):
    title: str
    year: int = Field(ge=1888, le=2030)
    genre: list[str]
    rating: float = Field(ge=0, le=10)
    summary: str = Field(max_length=200)

review = client.chat.completions.create(
    model="openai/gpt-5.4-mini",
    response_model=MovieReview,
    max_retries=3,
    messages=[
        {
            "role": "user",
            "content": "Just watched Inception (2010). Nolan's best work. "
                       "A mind-bending sci-fi thriller. 8.5/10."
        }
    ],
    max_completion_tokens=200,
)