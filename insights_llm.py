import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

# Use external providers (not hf-inference) for better LLM support
# These providers offer free tier through HF routing
# Order = preference (verified working models first)
PREFERRED_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",       # ✓ Verified working - Meta Llama
    "google/gemma-2-2b-it",                    # ✓ Verified working - Google Gemma
    "Qwen/Qwen2.5-7B-Instruct",               # Qwen 2.5 - large model
    "meta-llama/Llama-3.2-1B-Instruct",       # Smaller Llama variant
]


def build_context(summary, subject_stats, top_students, weak_students):
    """
    Build plain text context from numeric analysis.
    Signature must match what app.py calls.
    """
    lines = []
    lines.append(f"Class Average: {summary.get('class_avg')}")
    lines.append(f"Total Students: {summary.get('total_students')}")
    lines.append(f"Weak Students: {summary.get('weak_students')}")
    lines.append("")

    lines.append("Subject Averages:")
    try:
        for _, row in subject_stats["subject_avg"].iterrows():
            lines.append(f"- {row['subject']}: {round(row['average'], 2)}")
    except Exception:
        lines.append(str(subject_stats.get("subject_avg", "")))

    lines.append("")
    lines.append("Top 3 performers:")
    try:
        for _, row in top_students.head(3).iterrows():
            id_or_roll = row.iloc[0] if len(row) > 0 else ""
            name = row.iloc[1] if len(row) > 1 else ""
            avg = row["avg_marks"] if "avg_marks" in row.index else (row.iloc[-1] if len(row) > 0 else "")
            try:
                lines.append(f"- {id_or_roll} ({name}): {float(avg):.2f}")
            except Exception:
                lines.append(f"- {id_or_roll} ({name}): {avg}")
    except Exception:
        lines.append(str(top_students))

    lines.append("")
    lines.append("Some weak students:")
    try:
        for _, row in weak_students.head(5).iterrows():
            id_or_roll = row.iloc[0] if len(row) > 0 else ""
            name = row.iloc[1] if len(row) > 1 else ""
            avg = row["avg_marks"] if "avg_marks" in row.index else (row.iloc[-1] if len(row) > 0 else "")
            try:
                lines.append(f"- {id_or_roll} ({name}): {float(avg):.2f}")
            except Exception:
                lines.append(f"- {id_or_roll} ({name}): {avg}")
    except Exception:
        lines.append(str(weak_students))

    return "\n".join(lines)


def _dummy_insights(context_text: str) -> str:
    """
    Static fallback report used when HF key not present or API fails.
    """
    return f"""
AI INSIGHTS REPORT (Fallback – no live API response)

Overall Performance:
- The class performance is moderate based on the calculated averages.
- Students with low scores need focused academic support.

Subject-wise Observations:
- Subjects with lower average marks should receive extra revision and practice.
- High-performing subjects reflect effective teaching methods.

Actionable Recommendations:
- Conduct remedial sessions for weak students.
- Provide additional worksheets and periodic tests.
- Monitor progress through regular assessment.
- Encourage top performers with advanced exercises.

--------------------------------------------------
RAW ANALYSIS DATA USED:
{context_text}
--------------------------------------------------
"""


def _call_hf_chat_completion(model_id: str, prompt: str, max_tokens: int = 500):
    """
    Call HF Inference using chat completion API (the correct way for LLMs).
    Routes through HF to external providers.
    """
    if not HF_API_KEY:
        raise RuntimeError("HF_API_KEY not set in environment")
    
    try:
        # Initialize client (no provider needed - HF auto-routes)
        client = InferenceClient(api_key=HF_API_KEY)
        
        # Use chat completion API (OpenAI-compatible)
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert academic performance analyst."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        # Extract the response text
        return response.choices[0].message.content
        
    except Exception as e:
        raise RuntimeError(f"Model {model_id} failed: {str(e)}")


def generate_ai_insights(llm_client, context_text: str) -> str:
    """
    Generate AI insights from academic data.
    Signature matches app.py: generate_ai_insights(llm_client, context_text)
    """
    # Build a concise prompt
    prompt = f"""Analyze this class performance data and provide comprehensive educational insights.

DATA:
{context_text}

Please provide:
1. Overall Performance Summary (2-3 sentences about class performance)
2. Subject-wise Analysis:
   - Identify weak subjects and suggest improvement strategies
   - Highlight strong subjects and ways to maintain them
3. Student Performance Insights:
   - Observations about top performers
   - Concerns about struggling students
4. Actionable Recommendations (3-5 specific steps for teachers/administrators)

Write in a professional, educational tone suitable for school administrators."""

    # 1) If a custom LLM client was passed, try to use it
    if llm_client:
        try:
            if hasattr(llm_client, "chat_completion") or hasattr(llm_client, "chat"):
                try:
                    if hasattr(llm_client, "chat_completion") and hasattr(llm_client.chat_completion, "create"):
                        resp = llm_client.chat_completion.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You are an expert academic performance analyst."},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.3,
                            max_tokens=600,
                        )
                        if isinstance(resp, dict):
                            try:
                                return resp["choices"][0]["message"]["content"]
                            except Exception:
                                return str(resp)
                        return str(resp)

                    if hasattr(llm_client, "chat") and hasattr(llm_client.chat, "create"):
                        resp = llm_client.chat.create(
                            model="gpt-4o-mini", 
                            messages=[{"role": "user", "content": prompt}]
                        )
                        if hasattr(resp, "choices"):
                            return str(resp.choices[0].message.content)
                        if isinstance(resp, dict) and "output" in resp:
                            return str(resp["output"])
                        return str(resp)
                except Exception:
                    pass

            if hasattr(llm_client, "completion") and hasattr(llm_client.completion, "create"):
                try:
                    resp = llm_client.completion.create(
                        model="gpt-4o-mini", 
                        prompt=prompt, 
                        max_tokens=600
                    )
                    if isinstance(resp, dict) and "choices" in resp:
                        return resp["choices"][0].get("text") or str(resp)
                    return str(resp)
                except Exception:
                    pass
        except Exception:
            pass

    # 2) Fall back to HF Inference with auto-routing
    if not HF_API_KEY:
        return _dummy_insights(context_text) + "\n\n[Note: No HF_API_KEY set in environment]"

    last_exc = None
    errors_log = []
    
    for model_id in PREFERRED_MODELS:
        try:
            print(f"Trying model: {model_id}")
            result_text = _call_hf_chat_completion(model_id, prompt, max_tokens=500)
            
            # Check if we got meaningful output
            if result_text and len(result_text.strip()) > 50:
                print(f"✓ Success with model: {model_id}")
                return result_text
            else:
                error_msg = f"Model {model_id} returned insufficient output (length: {len(result_text.strip())})"
                print(f"✗ {error_msg}")
                errors_log.append(error_msg)
                
        except Exception as e:
            last_exc = e
            error_msg = f"Model {model_id}: {str(e)}"
            print(f"✗ {error_msg}")
            errors_log.append(error_msg)
            continue

    # 3) Nothing worked — return dummy insights with detailed error info
    error_msg = f"\n\n[HF API Error: All models failed]\n"
    error_msg += f"Last error: {last_exc}\n\n"
    error_msg += "Detailed errors:\n"
    for err in errors_log:
        error_msg += f"  - {err}\n"
    error_msg += f"\nTried models: {', '.join(PREFERRED_MODELS)}"
    
    return _dummy_insights(context_text) + error_msg