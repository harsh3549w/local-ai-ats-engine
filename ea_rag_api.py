import os
# Disable Hugging Face symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import pypdf  # Standard modern library matching your requirements.txt
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gliner import GLiNER

app = FastAPI(title="Harsh's Dynamic Hybrid ATS Engine API")

# 1. Load the AI Models locally
print("⏳ Loading GLiNER (Skill Extractor) & Sentence Transformer (Semantic Scorer)...")
gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ==================================================
# 🏆 THE DYNAMIC HYBRID ATS ENDPOINT
# ==================================================
@app.post("/analyze-local-resume")
async def analyze_local_resume(
    target_role: str = Form("Machine Learning Engineer"),
    raw_job_description: str = Form(
        "We are looking for a Machine Learning Engineer to join our AI Research team. "
        "The ideal candidate will have strong programming experience using Python and PyTorch. "
        "You should be familiar with Deep Learning principles, Computer Vision, and training CNNs. "
        "Hands-on experience with Generative AI workflows and optimizing ResNet architectures is a huge plus."
    ),
    file: UploadFile = File(...)
):
    # 1. Dynamically extract text straight from the uploaded PDF object in memory
    resume_text = ""
    try:
        reader = pypdf.PdfReader(file.file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                resume_text += extracted + " "
        resume_text = resume_text.strip()
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to read file '{file.filename}'. Technical Error: {e}"
        )
        
    if not resume_text:
        raise HTTPException(
            status_code=422, 
            detail="The uploaded PDF file does not contain any machine-readable text."
        )
        
    resume_text_lower = resume_text.lower()
    
    # 2. Dynamically Extract Required Skills from Job Description using GLiNER
    labels = ["programming language", "software framework", "machine learning model", "neural network architecture"]
    entities = gliner_model.predict_entities(raw_job_description, labels, threshold=0.7)
    
    required_skills = []
    for entity in entities:
        skill = entity["text"]
        if skill not in required_skills:
            required_skills.append(skill)
            
    # 3. METRIC 1: KEYWORD OVERLAP (60% Weight)
    found_skills = []
    for skill in required_skills:
        if skill.lower() in resume_text_lower:
            found_skills.append(skill)
            
    keyword_score = (len(found_skills) / len(required_skills)) * 100 if required_skills else 0
    
    # 4. METRIC 2: AI SEMANTIC SCORE (40% Weight)
    combined_jd_context = target_role + " " + raw_job_description
    jd_embedding = embedding_model.encode([combined_jd_context])
    resume_embedding = embedding_model.encode([resume_text])
    
    similarity = cosine_similarity(jd_embedding, resume_embedding)[0][0]
    ai_score = min(float(similarity) * 200, 100.0) 
    
    # 5. FINAL HYBRID CALCULATION 
    final_score = (keyword_score * 0.60) + (ai_score * 0.40)
    
    # Generate a dynamic display name based on the uploaded file name
    inferred_name = file.filename.split('.')[0].replace('_', ' ').replace('-', ' ').title()
    
    # Return clean, structured engine metrics
    return {
        "candidate_name": inferred_name,
        "target_role": target_role,
        "dynamic_skills_detected": required_skills,
        "skills_matched_on_resume": found_skills,
        "scoring_metrics": {
            "keyword_match_score": f"{round(keyword_score, 1)}%",
            "ai_context_score": f"{round(ai_score, 1)}%",
            "final_ats_ranking": f"{round(final_score, 1)}%"
        }
    }
