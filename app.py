import streamlit as st
import json
from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(page_title="AutoStream Lead Agent", layout="centered")
st.title("AutoStream AI Agent")

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


class Intent(BaseModel):
    intent: Literal["greeting", "inquiry", "high_intent"] = Field(description="The classification of the user's intent.")

class LeadInfo(BaseModel):
    name: Optional[str] = Field(None)
    email: Optional[str] = Field(None)
    platform: Optional[str] = Field(None)




@st.cache_resource
def get_vectorstore():
    try:
        loader = JSONLoader(
        file_path="data.json",
        jq_schema=".",
        text_content=False
        )

        raw_data = loader.load()
        data = json.loads(raw_data[0].page_content)
    except Exception as e:
        print(f"Error loading data: {e}")
    
    chunks = []

    # pricing plans
    for plan in data["AutoStream"]["pricing_plans"]:
        content = f"""
        Plan: {plan['plan_name']}
        Price: ${plan['price_per_month']} per month
        Features: {plan['features']}
        """
        chunks.append(Document(page_content=content.strip()))

    # policies
    policies = data["AutoStream"]["company_policies"]

    chunks.append(Document(
        page_content=f"Refund Policy: {policies['refund_policy']}"
    ))

    chunks.append(Document(
        page_content=f"Support: {policies['support']}"
    ))
    
    return FAISS.from_documents(chunks, embeddings)

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k":1, "lambda_mult": 0.5}
)

parser = StrOutputParser()

# Chains

# A. Intent Chain
intent_parser = PydanticOutputParser(pydantic_object=Intent)
intent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an intent classifier for AutoStream, a SaaS for video editing.
    Classify the user's latest message into:
    1. 'greeting': hi, hello, etc.
    2. 'inquiry': questions about pricing, features, or policies.
    3. 'high_intent': ready to sign up, buy, or try the pro plan.
    
    Answer instructions: {format_instructions}"""),
    MessagesPlaceholder(variable_name="history"), 
    ("human", "{question}")
]).partial(format_instructions=intent_parser.get_format_instructions())
intent_chain = intent_prompt | llm | intent_parser



# B. RAG Chain
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the AutoStream assistant. Use the context to answer product questions.\nContext: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
rag_chain = (
    {
        "context": lambda x: retriever.invoke(x["input"]),
        "history": lambda x: x["history"], 
        "input": lambda x: x["input"]
    } 
    | rag_prompt | llm | parser
)



# C. Lead Extraction Chain
lead_parser = PydanticOutputParser(pydantic_object=LeadInfo)
lead_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract name, email, and platform from history. Use 'null' for missing.\n{format_instructions}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
]).partial(format_instructions=lead_parser.get_format_instructions())
lead_chain = lead_prompt | llm | lead_parser



# 5. SESSION STATE (Memory)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lead_data" not in st.session_state:
    st.session_state.lead_data = {"name": None, "email": None, "platform": None}




def handle_agent_response(user_input):
    history = st.session_state.messages
    

    intent_obj = intent_chain.invoke({"question": user_input, "history": history})
    intent = intent_obj.intent

    if intent == "greeting":
        return "Hello! I'm the AutoStream agent. How can I help you today?"
    
    elif intent == "inquiry":
        return rag_chain.invoke({"input": user_input, "history": history})
    
    elif intent == "high_intent":

        extracted = lead_chain.invoke({"input": user_input, "history": history})
        

        for key in ["name", "email", "platform"]:
            if getattr(extracted, key):
                st.session_state.lead_data[key] = getattr(extracted, key)
        

        data = st.session_state.lead_data
        if not data["name"]:
            return "I'd love to get you started! What is your full name?"
        elif not data["email"]:
            return f"Thanks {data['name']}! What is your email address?"
        elif not data["platform"]:
            return "Last step: Which platform do you create for (YouTube, Instagram, etc.)?"
        else:
            return f"Perfect! Lead captured for {data['name']} ({data['platform']}). Welcome to the Pro plan!"

# STREAMLIT UI

for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

if prompt := st.chat_input("Ask about AutoStream..."):
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate Response
    with st.spinner("Thinking..."):
        response_text = handle_agent_response(prompt)
        
    # Update Session State History
    from langchain_core.messages import HumanMessage, AIMessage
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.messages.append(AIMessage(content=response_text))
    
    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(response_text)