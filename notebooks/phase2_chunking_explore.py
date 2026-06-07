# =============================================================
# PHASE 2: Chunking — the design decision that breaks most RAG systems
# Run with: python notebooks/phase2_chunking_explore.py
# =============================================================

# ------------------------------------------------------------
# BLOCK 1: What does raw extracted text actually look like?
# ------------------------------------------------------------

# We don't need a real PDF yet — let's simulate what pdf_loader.py returns.
# This is realistic text with the messiness you'd see in a real document.

raw_page_text = """
Introduction to Neural Networks

Neural networks are computational models inspired by the human brain.
They consist of layers of interconnected nodes, or neurons, that process
information using connectionist approaches to computation.

The first layer is called the input layer. It receives raw data such as
pixel values, text tokens, or numerical features. The last layer is the
output layer, which produces the final prediction or classification.

Hidden Layers and Depth

Between input and output are hidden layers. The depth of a network refers
to the number of hidden layers. Deep networks can learn hierarchical
representations — low-level features in early layers, high-level concepts
in later layers.

Activation Functions

Each neuron applies an activation function to introduce non-linearity.
Common choices include ReLU, sigmoid, and tanh. Without activation
functions, a neural network collapses into a linear model regardless
of its depth.

Training

Networks are trained using backpropagation and gradient descent.
The loss function measures prediction error. Gradients flow backwards
through the network, updating weights to minimise loss.
"""

print("Total characters:", len(raw_page_text))
print("Total words:", len(raw_page_text.split()))
print("Total lines:", len(raw_page_text.strip().splitlines()))
print("\nFirst 200 characters:")
print(repr(raw_page_text[:200]))
print("\nNotice the \\n characters — these are what chunking has to deal with.")

# ------------------------------------------------------------
# BLOCK 2: Fixed-size chunking — watching where cuts happen
# ------------------------------------------------------------

def chunk_fixed(text, chunk_size, overlap):
    """
    Pure Python fixed-size chunker — no libraries.
    Same logic as src/ingestion/chunker.py
    """
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# Experiment 1: small chunk, no overlap
print("=" * 60)
print("EXPERIMENT 1: chunk_size=200, overlap=0")
print("=" * 60)
chunks_no_overlap = chunk_fixed(raw_page_text, chunk_size=200, overlap=0)
for i, chunk in enumerate(chunks_no_overlap):
    print(f"\n--- Chunk {i+1} ---")
    print(repr(chunk))

# Experiment 2: same size, with overlap
print("\n" + "=" * 60)
print("EXPERIMENT 2: chunk_size=200, overlap=50")
print("=" * 60)
chunks_with_overlap = chunk_fixed(raw_page_text, chunk_size=200, overlap=50)
for i, chunk in enumerate(chunks_with_overlap):
    print(f"\n--- Chunk {i+1} ---")
    print(repr(chunk))

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"No overlap:   {len(chunks_no_overlap)} chunks")
print(f"With overlap: {len(chunks_with_overlap)} chunks")
print(f"\nExtra chunks from overlap: {len(chunks_with_overlap) - len(chunks_no_overlap)}")
print("(These extra chunks are the cost of overlap — more storage, better boundary coverage)")

# ------------------------------------------------------------
# BLOCK 3: The section boundary problem
# Fixed-size chunking doesn't respect document structure
# ------------------------------------------------------------

print("=" * 60)
print("BLOCK 3: Where do section boundaries fall?")
print("=" * 60)

# Look at chunks 4 and 5 from experiment 1 (no overlap)
print("\nChunk 4 ends with:")
print(repr(chunks_no_overlap[3][-80:]))

print("\nChunk 5 starts with:")
print(repr(chunks_no_overlap[4][:80]))

print("""
Notice: "Activation Functions" section header got split from its content.
Chunk 4 ends mid-section. Chunk 5 starts with just "y." — the end of "non-linearity"
— before the actual Activation Functions explanation begins.

If someone queries "what activation functions are used in neural networks",
Chunk 5 contains the answer (ReLU, sigmoid, tanh) but starts with "y." 
which is confusing context for the embedder.
""")

# ------------------------------------------------------------
# BLOCK 4: Sentence-aware chunking — a smarter approach
# Split at sentence boundaries, not arbitrary character counts
# ------------------------------------------------------------

print("=" * 60)
print("BLOCK 4: Sentence-aware chunking")
print("=" * 60)

def chunk_by_sentences(text, max_chars, overlap_sentences=1):
    """
    Split at sentence boundaries instead of arbitrary character positions.
    
    Strategy:
    - Split text into sentences first
    - Accumulate sentences until chunk would exceed max_chars
    - Start new chunk, carrying over last `overlap_sentences` sentences
    """
    import re
    # Split on period/exclamation/question followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_sentences = []
    current_len = 0
    
    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current_sentences:
            # Save current chunk
            chunks.append(" ".join(current_sentences))
            # Carry over last N sentences as overlap
            current_sentences = current_sentences[-overlap_sentences:]
            current_len = sum(len(s) for s in current_sentences)
        
        current_sentences.append(sentence)
        current_len += len(sentence)
    
    if current_sentences:
        chunks.append(" ".join(current_sentences))
    
    return chunks


chunks_sentence = chunk_by_sentences(raw_page_text, max_chars=300, overlap_sentences=1)

for i, chunk in enumerate(chunks_sentence):
    print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk)

print("\n" + "=" * 60)
print("COMPARISON: fixed-size vs sentence-aware")
print("=" * 60)
print(f"Fixed-size (200, no overlap) : {len(chunks_no_overlap)} chunks — cuts mid-word/mid-sentence")
print(f"Fixed-size (200, overlap=50) : {len(chunks_with_overlap)} chunks — better boundaries, more storage")
print(f"Sentence-aware (max 300)     : {len(chunks_sentence)} chunks — always ends at sentence boundary")

# ------------------------------------------------------------
# BLOCK 5: The production insight — what chunk strategy to use when
# ------------------------------------------------------------

print("=" * 60)
print("BLOCK 5: Choosing a chunking strategy")
print("=" * 60)

strategies = {
    "Fixed-size (no overlap)": {
        "when": "Never in production — only for quick prototyping",
        "problem": "Cuts mid-word, mid-sentence. No boundary awareness.",
        "example": "connec | tionist"
    },
    "Fixed-size (with overlap)": {
        "when": "Simple docs, uniform text, tight storage budget",
        "problem": "Still cuts mid-sentence. Overlap helps but doesn't fix structure.",
        "example": "Good for: plain prose documents"
    },
    "Sentence-aware": {
        "when": "Most general-purpose RAG systems — good default",
        "problem": "Section headers bleed into wrong chunks. Pronoun context lost.",
        "example": "Good for: articles, reports, textbooks"
    },
    "Semantic chunking": {
        "when": "High-quality production RAG, when compute allows",
        "problem": "Slower — embeds every sentence to detect topic shifts.",
        "example": "Splits when embedding similarity between consecutive sentences drops"
    },
    "Structure-aware (headers)": {
        "when": "Docs with clear structure: PDFs with headings, markdown, HTML",
        "problem": "Requires parsing structure — not all PDFs have clean headers.",
        "example": "Each section = one chunk. Header always stays with its content."
    },
}

for name, info in strategies.items():
    print(f"\n{name}")
    print(f"  Use when : {info['when']}")
    print(f"  Problem  : {info['problem']}")
    print(f"  Example  : {info['example']}")
