import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import OPENAI_API_KEY, CONFIDENCE_THRESHOLD, PERSONA_PROMPTS
from src.detector import PersonaAnalysis

def process_customer_ticket(query: str, analysis: PersonaAnalysis, vector_db) -> tuple[str, bool]:
    """
    Evaluates inputs against retrieval layers, monitors triggers, and 
    either shapes a tailored response or formats a human handoff summary.
    """
    # 1. Similarity Check with Relevance Scores
    matched_chunks = vector_db.similarity_search_with_relevance_scores(query, k=2)
    
    # Evaluate score confidence
    insufficient_context = False
    if not matched_chunks or matched_chunks[0][1] < CONFIDENCE_THRESHOLD:
        insufficient_context = True
        
    # 2. Check for Immediate Escalation
    if analysis.escalation_flag or insufficient_context:
        handoff_packet = {
            "escalation_event": "HUMAN_HANDOFF_TRIGGERED",
            "detected_persona": analysis.persona,
            "user_issue_summary": query,
            "trigger_reason": "Sensitive account request / Direct command" if analysis.escalation_flag else "Context not found / Low retrieval confidence",
            "documents_referenced": [doc.metadata.get("source", "Unknown") for doc, _ in matched_chunks] if matched_chunks else [],
            "keywords_extracted": analysis.extracted_keywords,
            "recommended_next_action": "Route immediately to premium account engineering queue for explicit resolution."
        }
        return json.dumps(handoff_packet, indent=2), True

    # 3. Compile Context and Generate Response
    context_str = "\n\n".join([chunk.page_content for chunk, _ in matched_chunks])
    
    system_prompt_template = (
        "You are a highly capable AI Support Agent. You must answer the customer's question using ONLY "
        "the verified context details provided below. Do not assume or extrapolate facts outside this scope.\n\n"
        "=== VERIFIED DATA CONTEXT ===\n{context}\n\n"
        "=== COMMUNICATION STYLE MANDATE ===\n{persona_style_guide}\n\n"
        "If the answer cannot be confidently deduced directly from the context block, state clearly that "
        "you do not possess the required data."
    )
    
    formatted_system = system_prompt_template.format(
        context=context_str,
        persona_style_guide=PERSONA_PROMPTS[analysis.persona]
    )
    
    prompt_blueprint = ChatPromptTemplate.from_messages([
        ("system", formatted_system),
        ("user", "{user_query}")
    ])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=OPENAI_API_KEY)
    execution_chain = prompt_blueprint | llm
    
    ai_response = execution_chain.invoke({"user_query": query})
    return ai_response.content, False
