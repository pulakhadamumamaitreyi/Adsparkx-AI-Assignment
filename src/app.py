import streamlit as st
import os

from src.detector import analyze_incoming_query
from src.database import build_or_load_vector_db
from src.agent import process_customer_ticket

st.set_page_config(page_title="Persona Support Agent Desk", layout="wide", page_icon="🤖")

st.title("⚡ Persona-Adaptive Intelligent Support Dashboard")
st.caption("AI Engineering Intern Assessment Framework — Multi-Persona Triage System")
st.write("---")

# Setup safe state caching for the vector DB index
@st.cache_resource
def load_shared_knowledge_base():
    try:
        return build_or_load_vector_db()
    except Exception as e:
        st.error(f"Initialization Failed: {str(e)}")
        return None

# Verify local structural rules
if not os.path.exists(".env"):
    st.warning("⚠️ Critical Configuration Missing: Please construct a `.env` file containing your valid `OPENAI_API_KEY` token.")
    st.stop()

db = load_shared_knowledge_base()

if db is not None:
    user_message = st.text_input("Simulate Customer Inquiry Input:", placeholder="Paste user ticket contents here...")
    
    if user_message:
        with st.spinner("Executing Pipeline Sequence (Triage -> Retrieval -> Routing)..."):
            
            # Step 1: Analyze Intent and Style Matrix
            insights = analyze_incoming_query(user_message)
            
            # Draw Analytics Dashboard Columns
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(label="Classified Persona Profile", value=insights.persona)
            with m_col2:
                st.metric(label="Assessed Sentiment", value=insights.sentiment)
            with m_col3:
                # Highlight active handoffs with distinct styling
                status_label = "🚨 ESCALATION CRITERIA MET" if insights.escalation_flag else "✅ AI AUTONOMOUS RUN"
                st.metric(label="Routing Path Evaluation", value=status_label)
                
            st.info(f"**Extracted Diagnostic Keywords:** {', '.join(insights.extracted_keywords)}")
            st.write("---")
            
            # Step 2: Route, Process, and Build context response outputs
            final_output, is_escalated = process_customer_ticket(user_message, insights, db)
            
            if is_escalated:
                st.error("### 🛑 Automation Suspended: Handing Off to Human Support Representative")
                st.markdown("The pipeline automatically generated a structured hand-off ticket to preserve context for the human agent:")
                st.code(final_output, language="json")
            else:
                st.success(f"### 🤖 Adaptive Response Generated ({insights.persona} Mode)")
                st.write(final_output)
