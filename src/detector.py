from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import OPENAI_API_KEY

class PersonaAnalysis(BaseModel):
    persona: Literal["Technical Expert", "Frustrated User", "Business Executive"] = Field(
        description="Categorize the customer based on terminology, tone, urgency, and sentence structure."
    )
    sentiment: Literal["Positive", "Neutral", "Anxious", "Angry"] = Field(
        description="The current underlying sentiment of the input message."
    )
    escalation_flag: bool = Field(
        description="Set to True ONLY if the message deals with billing discrepancies, legal threats, account compromise, data leaks, or an explicit demand for a human."
    )
    extracted_keywords: List[str] = Field(
        description="Extract 2-4 specific product entities or actions mentioned (e.g., ['API key', 'billing error'])."
    )

def analyze_incoming_query(query: str) -> PersonaAnalysis:
    """Uses structural output matching to triage incoming messages."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
    structured_llm = llm.with_structured_output(PersonaAnalysis)
    
    system_instruction = (
        "You are an expert behavior analyst and triage agent for a premium enterprise customer desk. "
        "Examine the message and extract the exact persona classification along with key situational parameters."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("user", "Analyze the following incoming customer query: {query}")
    ])
    
    chain = prompt | structured_llm
    return chain.invoke({"query": query})
