the key idea is: if the right information is not retrieved, even a powerful LLM cannot give a reliable answer.

Basic vector search converts the user's question and document chunks into embeddings and retrieves chunks whose meanings are mathematically similar.
The problem is that similar meaning does not always mean the right evidence.

Vector search is good at meaning similarity, but real questions often require exact words, aliases, context, multiple pieces of evidence, or clarification of intent.


### Query Expansion
Query expansion solves one specific problem: the user and the documents may use different words for the same thing.

Rewrite question in several useful ways, search all of them, and combine the results.

Instead of searching with one query, we create several related queries with different vocabulary.

```Example
User asks: “How do I set up SSO?”

Instead of searching only: How do I set up SSO?

the system expands it into:
single sign-on setup
SAML identity provider configuration
configure Okta or Azure AD login
enterprise authentication setup

Now the system searches using all four queries.
```

Query expansion isn't automatically better.
It can also retrieve more irrelevant documents, creating noise. The text therefore recommends using reranking after expansion when necessary.


### Techniques

| Technique | What it does | Simple example |
|---|---|---|
| **Query Expansion** | Creates multiple versions of the user's query | “How do I set up SSO?” → “SAML configuration”, “Okta login”, “enterprise authentication” |
| **HyDE** | Generates a hypothetical answer and uses it to search | “How do refunds work?” → generates a sample refund-policy paragraph → searches for similar real documents |
| **Hybrid Search** | Combines semantic/vector + keyword search | “Error 4042” → vector search finds meaning, keyword search finds the exact `4042` |
| **Reranking** | Reorders retrieved results based on relevance | 20 results are retrieved → reranker puts the actual refund policy at #1 |
| **Parent-Child Retrieval** | Searches small chunks but returns their larger surrounding section | Finds “vacation accrues monthly” → returns the entire HR policy section |
| **Contextual Retrieval** | Adds context to ambiguous chunks | “This limit increases to 10GB” → adds context saying it refers to the **Pro plan storage limit** |
| **Multi-Step Retrieval** | Searches again after learning something from the first search | Find outage → learn it involved billing → search for customers affected by billing outage |
| **Structured Retrieval** | Gets live facts directly from databases/APIs | “How many refunds happened in Germany last week?” → query the database, not the vector store |







