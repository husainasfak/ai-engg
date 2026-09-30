# list  - Ordered, mutable, allows duplicate elements

token_ids = [101, 7592, 1010, 2129, 2024, 2017, 102]
print(token_ids)

# A batch of embedding vectors
embeddings = [
    [0.12, -0.34, 0.56, 0.78],
    [0.91, -0.23, 0.45, 0.67],
    [0.33, -0.11, 0.89, 0.22],
]
print(embeddings)


results = []
results.append({"prompt": "What is RAG?", "response": "...", "score": 0.85})
results.append({"prompt": "Explain embeddings", "response": "...", "score": 0.92})
print(results)



# DIictionary - Unordered, mutable, no duplicate keys. kind of object in JS
model_config = {
    "model": "large-model",
    "temperature": 0.7,
    "max_output_tokens": 1024,
    "top_p": 0.9,
}
print(model_config)

vocab = {"hello": 7592, "world": 2088, "[CLS]": 101, "[SEP]": 102}
print(vocab)
# dictionary.get(key, default_value)
token_id = vocab.get("unknown_word",0) 
print(token_id)


# Sets: Deduplication and Fast Lookup
stopwords = {"the", "a", "an", "is", "at", "in", "on", "and", "or"}

tokens = ["the", "cat", "is", "on", "the", "mat"]
filtered = [t for t in tokens if t not in stopwords]
print(filtered)  # ['cat', 'mat']

unique_words = set(["apple", "banana", "apple", "cherry", "banana"])
print(unique_words)  # Print order is not guaranteed


# Tuples: Immutable and Hashable
image_shape = (3, 224, 224)  # channels, height, width

embedding_cache = {}
embedding_cache[(0.1, 0.2, 0.3)] = "document_42"

def evaluate_model(predictions, labels):
    precision = 0.91
    recall = 0.87
    f1 = 0.89
    return precision, recall, f1

predictions = ["positive", "negative", "positive"]
labels = ["positive", "negative", "negative"]
p, r, f1 = evaluate_model(predictions, labels)
print(p, r, f1)




# List Comprehensions
raw_scores = [0.2, 0.8, 0.5, 0.1, 0.9]
min_val, max_val = min(raw_scores), max(raw_scores)
normalized = [(s - min_val) / (max_val - min_val) for s in raw_scores]
print("Normalized",normalized)

predictions = [
    {"label": "positive", "score": 0.95},
    {"label": "negative", "score": 0.42},
    {"label": "positive", "score": 0.88},
    {"label": "neutral", "score": 0.31},
]
confident = [p for p in predictions if p["score"] > 0.5]
print(confident)

labels = [p["label"] for p in confident]
print(labels)  # ['positive', 'positive']


# Dict Comprehensions

words = ["hello", "world", "embedding", "vector", "token"]
word_to_idx = {word: idx for idx, word in enumerate(words)}
print(word_to_idx)

idx_to_word = {idx: word for word, idx in word_to_idx.items()}
print(idx_to_word)

config = {
    "model": "large-model",
    "temperature": 0.7,
    "max_output_tokens": 1024,
    "stream": True,
}
numeric_params = {
    k: v
    for k, v in config.items()
    if isinstance(v, (int, float)) and not isinstance(v, bool)
}
print(numeric_params)


# Set Comprehensions
dataset = [
    {"text": "Great product!", "label": "positive"},
    {"text": "Terrible service", "label": "negative"},
    {"text": "It was okay", "label": "neutral"},
    {"text": "Love it!", "label": "positive"},
]
unique_labels = {item["label"] for item in dataset}
print(unique_labels)  # Print order varies\
    
    
# Tuple Unpacking and Multiple Returns
# Basic Unpacking
point = (3, 7)
x, y = point

pairs = [("cat", 0.95), ("dog", 0.82), ("bird", 0.71)]
for animal, score in pairs:
    print(f"{animal}: {score}")

model_name, _, version = ("model-family", "unused", "v2")
print(model_name, version)

# Star Unpacking The * operator captures "the rest" into a list.
scores = [0.95, 0.88, 0.76, 0.71, 0.65]
best, *middle, worst = scores
print(best, middle, worst)

lines = ["name,score,label", "doc1,0.95,positive", "doc2,0.42,negative"]
header, *data_rows = lines
print(header)
print(data_rows)


# Functions Returning Multiple Values
def train_epoch():
    loss = 0.342
    accuracy = 0.891
    num_samples = 1024
    return loss, accuracy, num_samples
    return loss, accuracy, num_samples

loss, acc, n = train_epoch()
print(f"Loss: {loss:.3f}, Accuracy: {acc:.1%}, Samples: {n}")


# F-Strings: Clean Formatting

model = "large-model"
tokens = 1523
cost = 0.00457

print(f"Model: {model}, Tokens: {tokens}, Cost: ${cost}")

print(f"Cost: ${cost:.4f}")          # 4 decimal places: $0.0046
print(f"Accuracy: {0.8912:.1%}")     # Percentage: 89.1%
print(f"Tokens: {tokens:,}")         # Thousands separator: 1,523
print(f"{'Model':<20} {'Score':>10}")  # Alignment example

scores = [0.9, 0.85, 0.78]
print(f"Average: {sum(scores)/len(scores):.2f}")  # Average: 0.84


# Slicing: Working with Sequences
# Slicing extracts portions of lists, strings, arrays, and tensors. The syntax is sequence[start:stop:step], where start is inclusive and stop is exclusive.

# tokens[start:stop:step]
tokens = ["[CLS]", "how", "does", "RAG", "work", "?", "[SEP]"]

content = tokens[1:-1] # Remove the first and last elements
print("1",content)

first_three = tokens[:3] # Get the first three elements
print("2",first_three)

last_two = tokens[-2:] # Get the last two elements
print("3",last_two)

every_other = tokens[::2] 
# start is omitted, so it starts at index 0.

# stop is omitted, so it continues to the end.

# step is 2, so it takes every second element.
print("4",every_other)

reversed_tokens = tokens[::-1]
print("5",reversed_tokens)


# String Slicing
prompt = "Explain how transformers work in simple terms"

truncated = prompt[:20]
print(truncated)

filename = "model_weights.safetensors"
extension = filename[filename.rfind("."):]
print(extension)



# The Walrus Operator :=
# The walrus operator (:=), introduced in Python 3.8, assigns a value as part of an expression.

# Use it sparingly. It is helpful when it removes duplicated work without making the condition harder to read.
#  assign a value to a variable and use that value in the same expression.
while (chunk := stream.next_chunk()) is not None:
    print(chunk.text, end="", flush=True)

# Without walrus
while True:
    chunk = stream.next_chunk()

    if chunk is None:
        break

    print(chunk.text, end="", flush=True)

while (line := file.readline()):
    tokens = line.split()
    process(tokens)

results = []
for text in documents:
    if (score := compute_similarity(text, query)) > 0.8:
        results.append({"text": text, "score": score})
        
# Without :=, the last example would need to compute the score before the if and then use it again inside the block. That is fine too. Use the walrus operator only when it improves readability.


# Essential Python Idioms
# enumerate: Loop with Index
documents = ["intro to RAG", "embedding models", "vector databases"]

for i in range(len(documents)):
    print(f"Doc {i}: {documents[i]}")

for i, doc in enumerate(documents):
    print(f"Doc {i}: {doc}")

for i, doc in enumerate(documents, start=1):
    print(f"Doc {i}: {doc}")

# zip: Iterate in Parallel
models = ["large-model", "balanced-model", "small-local-model"]
scores = [0.92, 0.89, 0.85]
latencies = [1.2, 1.8, 0.6]

for model, score, latency in zip(models, scores, latencies):
    print(f"{model}: score={score}, latency={latency}s")

model_scores = dict(zip(models, scores))
print(model_scores)


# any and all: Bulk Boolean Checks
# These short-circuit through an iterable and return a single boolean. Think of any as "does at least one item satisfy this?" and all as "do all items satisfy this?"

scores = [0.95, 0.88, 0.42, 0.76]

has_high_score = any(s > 0.9 for s in scores)  # True
print(has_high_score)

all_passing = all(s > 0.5 for s in scores)  # False (0.42 fails)
print(all_passing)

required = ["model", "temperature", "max_output_tokens"]
config = {"model": "large-model", "temperature": 0.7}
missing = [f for f in required if f not in config]
has_missing = any(f not in config for f in required)  # True
print(missing)
print(has_missing)


# sorted with key: Custom Sorting

results = [
    {"doc": "RAG tutorial", "score": 0.72},
    {"doc": "Vector DB guide", "score": 0.95},
    {"doc": "Embedding basics", "score": 0.88},
]
ranked = sorted(results, key=lambda r: r["score"], reverse=True)
print(ranked)

words = ["transformer", "RAG", "embedding", "LLM"]
by_length = sorted(words, key=len)
print(by_length)


# dict.get with Defaults
# Using .get() with a default value is the standard pattern when a missing key has a legitimate fallback.

config = {"model": "large-model", "temperature": 0.7}

# This would raise KeyError if the key is missing:
# max_output_tokens = config["max_output_tokens"]

max_output_tokens = config.get("max_output_tokens", 1024)
stream = config.get("stream", False)          # False
print(max_output_tokens, stream)



# String Methods for NLP and Text Processing
# When you work with text data, you will lean on a small set of string methods. They are useful for preprocessing inputs, parsing outputs, and preparing prompts.


# split and join: Tokenization Basics - split breaks a string into a list. join does the reverse. They are not a replacement for a model tokenizer, but they are useful for basic text processing.

text = "How does retrieval augmented generation work?"
tokens = text.split()
print(tokens)

csv_line = "positive,0.95,This product is great"
label, score, text = csv_line.split(",", maxsplit=2)
print(label, score, text)

cleaned_tokens = ["how", "does", "rag", "work"]
sentence = " ".join(cleaned_tokens)
print(sentence)

items = ["retrieval", "augmented", "generation"]
formatted = ", ".join(items)
print(formatted)


# strip, replace, lower: Cleaning Text
raw_response = "\n  The answer is 42.  \n\n"
clean = raw_response.strip()
print(clean)

query = "What is RAG?"
normalized = query.lower()
print(normalized)

text = "Hello\nWorld\n\nHow are you?"
single_line = text.replace("\n", " ")
print(single_line)

dirty = "  \n HELLO, WORLD! \t "
clean = dirty.strip().lower().replace("!", "")
print(clean)


# startswith and endswith: Pattern Matching

filename = "model_weights.safetensors"

if filename.endswith(".safetensors"):
    print("SafeTensors format")

if filename.endswith((".pt", ".pth", ".safetensors")):
    print("Model weights file")

model_files = ["encoder.onnx", "ranker.pt", "tokenizer.json", "weights.safetensors"]
weight_files = [
    name
    for name in model_files
    if name.endswith((".pt", ".safetensors"))
]
print(weight_files)


# Ternary Expressions

score = 0.85

# Ternary expression
label = "high confidence" if score > 0.8 else "low confidence"
print(label)

use_large_model = True
model = "large-model" if use_large_model else "small-model"
print(model)

user_temperature = None
temperature = user_temperature if user_temperature is not None else 0.7
print(temperature)

for name, val in [("accuracy", 0.92), ("loss", 0.45)]:
    status = "good" if val > 0.8 else "needs work"
    print(f"{name}: {val} ({status})")
    
    
# None Checks and Truthiness

# None	NoneType	The explicit "nothing" value
# False	bool	    The boolean false value
# 0	    int	        Zero is falsy
# 0.0	float	    Zero float is falsy
# ""	str	        Empty string is falsy
# []	list	    Empty list is falsy
# {}	dict	    Empty dict is falsy
# set()	set	        Empty set is falsy


# The Common Pattern: Default Mutable Arguments - This is a common Python mistake. Do not use a mutable default argument:
def add_document(doc, collection=[]):
    collection.append(doc)
    return collection

def add_document(doc, collection=None):
    if collection is None:
        collection = []
    collection.append(doc)
    return collection

# The problem with the first version is that the default [] is created once when the function is defined, not each time it is called. Every call without an explicit collection argument shares and mutates the same list. This comes up often in AI code when you are accumulating documents, messages, examples, or results.