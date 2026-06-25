# Local AI-Powered Hybrid ATS Engine 🚀

A fully local, offline Applicant Tracking System (ATS) backend API designed to securely parse, analyze, and rank resumes against job descriptions without relying on external web APIs. 

Built with **FastAPI**, **ChromaDB**, and **Sentence Transformers**, this engine combines deterministic keyword matching with semantic vector similarity to provide accurate, un-gameable candidate rankings.

## 🌟 Key Features
* **100% Offline & Private:** No data is sent to OpenAI or third-party cloud providers. All parsing and vectorization happen locally.
* **Dynamic Skill Extraction:** Uses **GLiNER** (Named Entity Recognition) to dynamically extract hard technical requirements directly from raw job descriptions.
* **Hybrid Scoring Algorithm:** Computes ATS scores by balancing exact keyword overlap (60% weight) with deep semantic context matching (40% weight).
* **Vector Database Integration:** Leverages **ChromaDB** for efficient resume embedding storage and retrieval.
* **REST API:** Clean, documented **FastAPI** endpoints (e.g., `/analyze-local-resume`) for easy frontend integration.

## 🛠️ Technology Stack
* **Backend:** Python, FastAPI, Uvicorn
* **Machine Learning:** PyTorch, GLiNER, Sentence Transformers (`all-MiniLM-L6-v2`)
* **Vector Database:** ChromaDB
* **Data Processing:** PyPDF2, Scikit-Learn, Pandas

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone [https://github.com/harsh3549w/local-ai-ats-engine.git](https://github.com/harsh3549w/local-ai-ats-engine.git)
cd local-ai-ats-engine
