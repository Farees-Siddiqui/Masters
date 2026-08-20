# **Pipeline Status & Development Roadmap**

## **Executive Summary**

The **Structured Document & Information Extraction Pipeline** (Structured\_Data/layout\_pipeline/) is a high-precision, locally-executed document reconstruction engine. Operating entirely on local GPU hardware (4x NVIDIA V100), the system processes raw PDF/image documents into two distinct outputs:

> 1. **Structural XML (.reconstructed.xml)**: Physical geometry, reading order, bounding boxes, and extracted elements (tables, formulas, figures, text).  
> 2. **Semantic XML (.semantic.xml)**: Schema-agnostic, LLM-driven entity and attribute extraction mapping unstructured narrative into domain-specific hierarchical trees.

## **1\. Completed Milestones**

### **Structural Layout Engine & Dual Extraction**

* **OCR & Reading Order**: Integrated PaddleOCR and reading-order reconstruction via XY-Cut++.  
* **Crop Router & Crop Handlers**: Created specialized handlers for visual regions (Vision, Formula, Table).  
* **Table Extraction Logic**: Implemented MIN\_CELL\_FILL density thresholds to prevent silent failures on sparse or unbordered table layouts.

### **Structural XML Projection Engine**

* **Metadata & Bounding Boxes**: Implemented complete round-trip XML generation carrying block IDs, bbox, and extraction status (extracted, completed, fallback).  
* **XML Escaping & Sanitation**: Handled control character stripping and strict raw HTML escaping for tables to ensure byte-identical unescaped parsing without breaking XML 1.0 specifications.  
* **Corpus Verification**: Verified structural fidelity against densenet.pdf (9 pages, 150 blocks) and single-page evaluation corpora.

### **Dynamic Information Extraction (IE) Engine**

* **Schema-Agnostic Extraction**: Replaced static Pydantic schemas with an LLM-driven dynamic extraction architecture (src/ie\_engine/).  
* **Data Model (node\_schema.py)**: Defined stdlib DynamicElement dataclasses. Converts scalars to XML attributes, nested objects to child nodes, and normalizes strings to valid XML tag names (e.g., 2024 grades $\\rightarrow$ n2024-grades).  
* **Zero-Dependency LLM Client (llm\_client.py)**: Built on stdlib urllib targeting Ollama (/api/chat) or OpenAI-compatible vLLM endpoints (/v1/chat/completions). System prompt is de-primed using neutral shipping manifest examples to prevent hallucinated entity echos.  
* **Pipeline Integration (main.py)**: Integrated CLI \--mode {structural, semantic, both} with both set as the default behavior.

### **Testing & Quality Assurance**

| Metric | Status |
| :---- | :---- |
| **Total Test Suite** | 278+ passing unit and integration tests |
| **IE Engine Coverage** | 49 dedicated tests covering dynamic element trees, JSON parsing, and mock backends |
| **Verified Models** | Tested against llama3.1:8b and validated for deployment on llama3.3:70b |

## **2\. Technical Architecture Overview**

                          \[ Input PDF / Image \]  
                                    │  
                         ┌──────────┴──────────┐  
                         ▼                     ▼  
                  \[ OCR Engine \]       \[ Layout Router \]  
                         │                     │  
                         └──────────┬──────────┘  
                                    ▼  
                         \[ Dual Extraction Router \]  
                         ┌──────────┼──────────┐  
                         ▼          ▼          ▼  
                      (Vision)  (Formula)   (Table)  
                                    │  
                                    ▼  
                          \[ Semantic Blocks \]  
                                    │  
                 ┌──────────────────┴──────────────────┐  
                 ▼                                     ▼  
     \[ Structural XML Projector \]            \[ Dynamic IE Engine \]  
                 │                             (Local Llama 3\)  
                 ▼                                     │  
    {doc}.reconstructed.xml                            ▼  
                                            \[ Dynamic XML Projector \]  
                                                       │  
                                                       ▼  
                                              {doc}.semantic.xml

## **3\. Immediate Next Steps & Backlog**

### **Phase 1: Batch Corpus Handling & Page Collapse Resolution**

* **Source Metadata Integration**: Update block metadata to track source document IDs when running directory-wide batches. This resolves the $p\_0$ multi-file collapse issue where multi-document directories fall back into single \<page number="0"\> wrappers.  
* **Multi-Page Semantic Aggregation**: Update DynamicInformationExtractor to intelligently merge semantic contexts across multi-page document spans without exceeding model context windows.

### **Phase 2: Local GPU Optimization & Inference Bottlenecks**

* **Multi-GPU vLLM Setup**: Configure tensor parallelism across the 4x V100 (32GB) GPUs to run larger models (llama3.3:70b) efficiently.  
* **Stream & Batch Inference**: Implement batching logic in LocalLLMClient to process multi-paragraph text blocks concurrently rather than strictly sequentially.

### **Phase 3: Post-Processing & Semantic Validation**

* **Empty Node Pruning**: Add dynamic XML cleaner passes to remove redundant empty tags or low-confidence model artifacts prior to serialization.  
* **Schema Normalization Options**: Allow optional downstream mapping rules for standardizing key names (e.g., snake\_case to kebab-case) across diverse extraction runs.