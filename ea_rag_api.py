import os
# Disable Hugging Face symlink warnings on Windows just like your notebook
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import PyPDF2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gliner import GLiNER

app = FastAPI(title="Harsh's Dynamic Hybrid ATS Engine API")

# 1. Load the AI Models exactly like your notebook's final cell
print("⏳ Loading GLiNER (Skill Extractor) & Sentence Transformer (Semantic Scorer)...")
gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define what the API expects you to send (Defaults are pulled directly from your notebook data)
class ATSAnalysisRequest(BaseModel):
    target_role: str = "Machine Learning Engineer"
    raw_job_description: str = """
    We are looking for a Machine Learning Engineer to join our AI Research team. 
    The ideal candidate will have strong programming experience using Python and PyTorch. 
    You should be familiar with Deep Learning principles, Computer Vision, and training CNNs. 
    Hands-on experience with Generative AI workflows and optimizing ResNet architectures is a huge plus.
    """

def extract_text_from_pdf(pdf_path):
    """Reads the local PDF file from your directory."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + " "
        return text.strip()
    except Exception as e:
        print(f"❌ PDF Error: {e}")
        return ""

# ==================================================
# 🏆 THE FUSED HYBRID ATS ENDPOINT
# ==================================================
@app.post("/analyze-local-resume")
def analyze_local_resume(req: ATSAnalysisRequest):
    # Automatically tracks your file in the same folder
    resume_file = "Harsh_resume.pdf" 
    resume_text = extract_text_from_pdf(resume_file)
    
    if not resume_text:
        raise HTTPException(
            status_code=404, 
            detail=f"Could not read '{resume_file}'. Make sure it is saved in your C:\\Users\\HARSH RAJPUT folder."
        )
        
    resume_text_lower = resume_text.lower()
    
    # 1. Dynamically Extract Required Skills from Job Description using GLiNER
    labels = ["programming language", "software framework", "machine learning model", "neural network architecture"]
    entities = gliner_model.predict_entities(req.raw_job_description, labels, threshold=0.7)
    
    required_skills = []
    for entity in entities:
        skill = entity["text"]
        if skill not in required_skills:
            required_skills.append(skill)
            
    # 2. METRIC 1: KEYWORD OVERLAP (60% Weight)
    found_skills = []
    for skill in required_skills:
        if skill.lower() in resume_text_lower:
            found_skills.append(skill)
            
    keyword_score = (len(found_skills) / len(required_skills)) * 100 if required_skills else 0
    
    # 3. METRIC 2: AI SEMANTIC SCORE (40% Weight)
    combined_jd_context = req.target_role + " " + req.raw_job_description
    jd_embedding = embedding_model.encode([combined_jd_context])
    resume_embedding = embedding_model.encode([resume_text])
    
    similarity = cosine_similarity(jd_embedding, resume_embedding)[0][0]
    ai_score = min(float(similarity) * 200, 100.0) 
    
    # 4. FINAL HYBRID CALCULATION (Matching your notebook weights)
    final_score = (keyword_score * 0.60) + (ai_score * 0.40)
    
    # Return the clean structured engine metrics directly to localhost
    return {
        "candidate_name": "Harsh Rajput",
        "target_role": req.target_role,
        "dynamic_skills_detected": required_skills,
        "skills_matched_on_resume": found_skills,
        "scoring_metrics": {
            "keyword_match_score": f"{round(keyword_score, 1)}%",
            "ai_context_score": f"{round(ai_score, 1)}%",
            "final_ats_ranking": f"{round(final_score, 1)}%"
        }
    }