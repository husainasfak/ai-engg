"""
Some questions need more than one search.

Question: "What caused the March 5 outage, and which enterprise customers were affected?"



"""

class RetrievalStep(BaseModel):
    query: str
    purpose: str

class RetrievalPlan(BaseModel):
    steps: list[RetrievalStep]

def plan_retrieval(question: str) -> RetrievalPlan:
    response = client.chat.completions.parse(
        model="gpt-5.4-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Plan 1-4 retrieval steps needed to answer the question. "
                    "Each step should be a concrete search query with a purpose. "
                    "Do not answer the question."
                ),
            },
            {"role": "user", "content": question},
        ],
        response_format=RetrievalPlan,
        temperature=0,
    )
    return response.choices[0].message.parsed

def planned_retrieval(question: str, search, top_k_per_step: int = 3) -> list[dict]:
    plan = plan_retrieval(question)
    evidence = []
    seen_ids = set()

    for step in plan.steps:
        hits = search(step.query, top_k=top_k_per_step)
        for hit in hits:
            if hit["id"] not in seen_ids:
                seen_ids.add(hit["id"])
                evidence.append({**hit, "retrieval_purpose": step.purpose})

    return evidence