# evaluation/generate_testset.py
# Generates a synthetic golden dataset from our legal documents
# Uses GPT-4o to create realistic Q&A pairs for RAGAS evaluation
# Run this ONCE to generate testset.json, then run run_evaluation.py

import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ── Configuration ─────────────────────────────────────────────────────────────
DOCS_DIR = Path(__file__).parent.parent / "data"
OUTPUT_PATH = Path(__file__).parent / "testset.json"

DOCUMENTS = [
    "Raw Test Data.pdf",
    "complaint_for_a_civil_case.docx",
    "settlement_agreement.txt",
    "deposition_transcript.txt",
    "legal_memo.txt",
]

QUESTIONS_PER_DOC = 3  # 3 per doc × 5 docs = 15 total questions

# ── Load Documents ────────────────────────────────────────────────────────────
def load_document_text(filename: str) -> str:
    """Extract clean text from a document using our ingestion pipeline."""
    from modules.ingestion import extract_text
    path = DOCS_DIR / filename
    if not path.exists():
        print(f"  ⚠️  Skipping {filename} — file not found")
        return ""
    pages = extract_text(str(path))
    return "\n\n".join([p["text"] for p in pages])


# ── Generate Q&A Pairs ────────────────────────────────────────────────────────
def generate_qa_pairs(doc_text: str, doc_name: str,
                      n: int = 3) -> list[dict]:
    """
    Use GPT-4o to generate realistic legal Q&A pairs from a document.
    Each pair includes question, ground_truth answer, and source document.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    prompt = f"""You are a legal AI evaluation expert creating test questions 
for a RAG system used by litigation attorneys.

Given the following legal document excerpt, generate exactly {n} question-answer 
pairs that a litigation attorney might ask. 

Requirements:
- Questions must be answerable from the document text provided
- Questions should be diverse — factual, analytical, and cross-referencing
- Answers must be specific, accurate, and cite the relevant information
- Include at least one question that requires understanding legal terminology
- Make questions realistic for a practicing attorney

Format your response as a JSON array with this exact structure:
[
  {{
    "question": "the question",
    "ground_truth": "the correct answer based on the document",
    "question_type": "factual|analytical|legal_terminology"
  }}
]

Return ONLY the JSON array, no preamble or explanation.

DOCUMENT: {doc_name}
---
{doc_text[:4000]}
---"""

    messages = [
        SystemMessage(content="You are a legal AI evaluation expert. Return only valid JSON."),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    
    # Parse response
    content = response.content.strip()
    # Strip markdown code blocks if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()
    
    pairs = json.loads(content)
    
    # Add source document to each pair
    for pair in pairs:
        pair["source_doc"] = doc_name
    
    return pairs


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n🔬 Generating RAGAS Golden Test Dataset")
    print("=" * 60)
    print(f"Model: GPT-4o")
    print(f"Documents: {len(DOCUMENTS)}")
    print(f"Questions per document: {QUESTIONS_PER_DOC}")
    print(f"Target total: {len(DOCUMENTS) * QUESTIONS_PER_DOC} Q&A pairs")
    print("=" * 60)

    all_pairs = []

    for doc_name in DOCUMENTS:
        print(f"\n📄 Processing: {doc_name}")
        
        doc_text = load_document_text(doc_name)
        if not doc_text:
            continue
        
        print(f"  Extracted {len(doc_text)} characters")
        print(f"  Generating {QUESTIONS_PER_DOC} Q&A pairs...")
        
        try:
            pairs = generate_qa_pairs(doc_text, doc_name, n=QUESTIONS_PER_DOC)
            all_pairs.extend(pairs)
            print(f"  ✅ Generated {len(pairs)} pairs")
            
            # Preview first question
            print(f"  Preview: {pairs[0]['question'][:80]}...")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue

    # Save testset
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    
    testset = {
        "metadata": {
            "total_questions": len(all_pairs),
            "documents": DOCUMENTS,
            "model_used": "gpt-4o",
            "questions_per_doc": QUESTIONS_PER_DOC
        },
        "questions": all_pairs
    }
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(testset, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"✅ Testset saved to: {OUTPUT_PATH}")
    print(f"   Total Q&A pairs: {len(all_pairs)}")
    print(f"\nQuestion type breakdown:")
    
    from collections import Counter
    types = Counter(p.get("question_type", "unknown") for p in all_pairs)
    for qtype, count in types.items():
        print(f"  {qtype}: {count}")
    
    print(f"\nNext step: Run evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()