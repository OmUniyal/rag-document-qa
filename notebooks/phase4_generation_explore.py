# =============================================================
# PHASE 4: Generation — connecting retrieval to an LLM
# Run with: python notebooks\phase4_generation_explore.py
# =============================================================

# ------------------------------------------------------------
# BLOCK 1: The prompt is the most important engineering decision
# Most RAG failures are prompt failures, not retrieval failures
# ------------------------------------------------------------

# Simulated retrieved chunks — as if retriever.py returned these
retrieved_chunks = [
    {
        "text": "Backpropagation computes gradients by applying the chain rule recursively through each layer. The gradient of the loss with respect to each weight is used to update that weight via gradient descent.",
        "source": "dl_textbook.pdf",
        "page": 4,
        "score": 0.82,
    },
    {
        "text": "The learning rate controls how large each weight update step is. Too high and training diverges. Too low and training is slow or gets stuck in local minima.",
        "source": "dl_textbook.pdf",
        "page": 5,
        "score": 0.71,
    },
    {
        "text": "Gradient descent has several variants: batch, stochastic (SGD), and mini-batch. Mini-batch is the most commonly used in practice as it balances speed and stability.",
        "source": "dl_textbook.pdf",
        "page": 6,
        "score": 0.64,
    },
]

query = "How does backpropagation work and what role does learning rate play?"


# ------------------------------------------------------------
# The naive prompt — what beginners write
# ------------------------------------------------------------
naive_prompt = f"Answer this question: {query}"

print("NAIVE PROMPT:")
print("-" * 40)
print(naive_prompt)
print("""
Problem: LLM answers from training data, not your documents.
No grounding. No source attribution. Hallucination risk is high.
This is just a chatbot, not a RAG system.
""")


# ------------------------------------------------------------
# The grounded prompt — what RAG actually is
# ------------------------------------------------------------
def build_grounded_prompt(query, chunks):
    system = """You are a precise document assistant. Answer ONLY using the context below.
If the answer is not in the context, say: "I could not find this in the provided documents."
Always cite which source and page your answer comes from."""

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""{system}

=== CONTEXT ===
{context}

=== QUESTION ===
{query}

=== ANSWER ==="""
    return prompt


grounded_prompt = build_grounded_prompt(query, retrieved_chunks)

print("GROUNDED PROMPT:")
print("-" * 40)
print(grounded_prompt)

print("\n" + "=" * 60)
print("PROMPT ANALYSIS")
print("=" * 60)

token_estimate = len(grounded_prompt.split()) * 1.3  # rough tokens from words
print(f"Prompt word count : {len(grounded_prompt.split())}")
print(f"Estimated tokens  : {int(token_estimate)}")
print(f"MiniLM token limit: 512 tokens")
print(f"Mistral 7B context: ~8192 tokens")
print(f"Fits in context?  : {'YES' if token_estimate < 8192 else 'NO — needs truncation'}")

print("""
Key design decisions in this prompt:
1. System instruction — tells model to ONLY use context
2. Source labels — [Source 1: file.pdf, Page 4] enable citation
3. Explicit fallback — "I could not find this" prevents hallucination
4. Separator markers — === CONTEXT === makes structure unambiguous
""")

# ------------------------------------------------------------
# BLOCK 2: Sending the prompt to HuggingFace Inference API
# ------------------------------------------------------------

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate(prompt, temperature=0.1):
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=temperature,
    )
    return completion.choices[0].message.content.strip()


# Test 1: Grounded prompt
print("=" * 60)
print("TEST 1: Grounded prompt")
print("=" * 60)
print(generate(grounded_prompt))

# Test 2: No relevant context — hallucination test
print("\n" + "=" * 60)
print("TEST 2: No relevant context — hallucination test")
print("=" * 60)

irrelevant_chunks = [
    {
        "text": "The Eiffel Tower was constructed between 1887 and 1889.",
        "source": "paris_guide.pdf",
        "page": 1,
        "score": 0.12,
    }
]

off_topic_prompt = build_grounded_prompt(
    "What is the capital of Australia?",
    irrelevant_chunks
)
print(generate(off_topic_prompt))

# Test 3: Temperature effect
print("\n" + "=" * 60)
print("TEST 3: Temperature 0.1 vs 0.9 — same question, different outputs")
print("=" * 60)

short_prompt = build_grounded_prompt(
    "What is backpropagation?",
    [retrieved_chunks[0]]
)

print("Temperature 0.1 (factual):")
print(generate(short_prompt, temperature=0.1))

print("\nTemperature 0.9 (creative):")
print(generate(short_prompt, temperature=0.9))