import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List
import numpy as np

# LangChain & AI Imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

load_dotenv()

# --- Custom Embedding Class (Notebook Style) --- 
class SimpleEmbeddings(Embeddings):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

# --- Advanced RAG Pipeline Class Krish Naik jesa ---
class AdvancedRAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []

    def query(self, question: str, top_k: int = 3, min_score: float = 0.1, summarize: bool = False):
        # 1. Database se chunks mangwao


        # HYDE Question Expansion
        expansion_prompt = f"""
        You are an AI assistant that re-writes user queries to find better matches in academic PDFs.
        The user asked: "{question}"
        Generate 3 different versions of this question that focus on academic terminology, 
        key concepts, and strategic management language.
        Output only the queries, one per line.
        """
        expanded_queries_response = self.llm.invoke(expansion_prompt)
        queries = [question] + expanded_queries_response.content.strip().split("\n")
        
        # Ab in sab queries ke liye chunks dhoondo
        all_docs = []
        for q in queries:
            all_docs.extend(self.retriever.invoke(q))
        
        # Duplicate chunks khatam karne ke liye (Simple set logic)
        unique_docs = {doc.page_content: doc for doc in all_docs}.values()
        relevant_chunks = list(unique_docs)[:top_k]

        retrieved_docs = self.retriever.invoke(question)
        
        # Sirf top_k = 3 chunks uthao aur unka text milao
        relevant_chunks = retrieved_docs[:top_k]
        context = "\n".join([doc.page_content for doc in relevant_chunks])
        
        # LLM ko prompt bhejo
        system_prompt = f"""
        ### ROLE:
        You are "Alex", a professional "Strict Evidence-Based Research Assistant". 
        Your expertise is limited EXCLUSIVELY to the provided Context below. 
        You are designed to assist C-suite executives in strategic decision-making based on rigorous academic literature.

        ###TASK:
        Answer the user's question using ONLY the provided context. If the exact answer is not there, but related concepts are discussed (e.g., 'Structured Data' for 'Knowledge Graphs'), use that information to explain.

        ###STRICT RULES:
        - Do not say "I am sorry" if there is related info.
        - If the user greets you, reply as "Hello, I am Alex, your Strict Evidence-Based Research Assistant."
        - Always cite the Source (Book Name) if you find the answer.

        ### CORE OPERATING PRINCIPLE:
        - "Grounding": If the answer is not in the Context, you MUST admit it. 
        - "Zero-External Knowledge": Do not use your own training data or general knowledge. 
        - "Source Integrity": Only provide information that can be traced back to the retrieved snippets.

        ### INSTRUCTIONS:
        1. **Analyze with Type 2 Thinking**: Carefully evaluate the Context to find direct or semantic answers to the Question.
        2. **Strict Boundary**: If the Context does not contain the answer, say: "I am sorry, but the provided documentation does not contain information to answer this question."
        3. **Professional Tone**: Maintain a scholarly, objective, and concise tone (referencing Rousseau, 2006 style).
        4. **Citation (If available)**: If the context mentions authors (e.g., Pan et al., Davis, or Manning), include them in your response.
        5. **Summarization**: If the user requested a summary, distill the retrieved evidence without adding external interpretation.

        ### WHAT TO AVOID:
        - NO Hallucinations.
        - NO Phrases like "Based on my knowledge" or "Commonly known as". 
        - NO Creative writing or making up facts to be "helpful".

        ----------------
        CONTEXT:
        {context}
        ----------------
        QUESTION: 
        {question}

        FINAL ANSWER (Strictly based on Context):
        """
        response = self.llm.invoke(system_prompt)
        answer = response.content
        
        # 4. Snippets taiyar karna (Yahan se Ctrl+F wala kaam shuru hota hai)
        sources_info = []
        for doc in relevant_chunks:
            full_path = doc.metadata.get('source', 'Unknown Book')
            book_name = os.path.basename(full_path)

            # doc.page_content wahi text hai jo book mein likha hai
            full_text = doc.page_content.strip() # Pura chunk snippet ban gaya
            page_num = doc.metadata.get('page', 'Unknown')

            sentences = full_text.split('.')
            short_snippet = ". ".join(sentences[:3]) + "."
            
            # Sources info mein add karo Book name, Page number, aur Snippet
            sources_info.append({
                'book': book_name,
                'page': page_num + 1,
                'snippet': short_snippet
            })
        
        # Summary (agar chahiye)
        summary = ""
        if summarize:
            summary_prompt = f"Summarize this: {answer}"
            summary = self.llm.invoke(summary_prompt).content
                
        # Sab kuch wapas bhej do
        return {
            'answer': answer, 
            'summary': summary, 
            'sources': sources_info
        }
# --- Setup Functions with Caching ---
import os
from pathlib import Path

# ... (Baqi class aur imports)

@st.cache_resource # Streamlit caching for resource-intensive functions
def get_vector_db():
    # Folder ka path jahan FAISS save hoga
    BASE_DIR = Path(__file__).resolve().parent.parent
    PDF_DIR = BASE_DIR / "data" / "pdf"
    PERSIST_DIR = BASE_DIR / "faiss_db"
    persist_dir = str(PERSIST_DIR)
    pdf_path = str(PDF_DIR)

    embeddings = SimpleEmbeddings()
    
    # Agar Database pehle se bana hua hai toh use load karo
    if os.path.exists(persist_dir):
        print("--- FAISS Folder found, Loading Database ---")
        # allow_dangerous_deserialization isliye zaroori hai kyunke hum local file load kar rahe hain
        return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    
    # 2. Agar nahi hai toh Naya Database banao
    print("--- Creating NEW FAISS Database ---")
    st.warning("Database nahi mila. Naya knowledge base ban raha hai, thora intezar karein...")
    
    # pdf_path = r"..\data\pdf"

    
    
    if not os.path.exists(pdf_path):
        st.error(f"PDF folder nahi mila: {pdf_path}")
        st.stop()

    # Documents load karna
    loader = DirectoryLoader(
        pdf_path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader
    )
    documents = loader.load()
    
    if not documents:
        st.error("PDF folder khali hai!")
        st.stop()

    # Text splitting (TPM limit ke liye 500 best hai)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    
    # FAISS Database banana
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # --- YAHAN SAVE HOGA ---
    # FAISS local folder mein 2 files save karta hai: index.faiss aur index.pkl
    vectorstore.save_local(persist_dir)
    
    st.success("FAISS Database kamyabi se ban gaya aur save ho gaya!")
    print("--- FAISS Saved Successfully ---")
    
    return vectorstore

# --- Streamlit UI Setup ---
st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("🤖 RAG ChatBot")

# Sidebar settings
st.sidebar.header("Pipeline Settings")
show_summary = st.sidebar.checkbox("Show Summary", value=True)
num_chunks = st.sidebar.slider("Chunks to retrieve", 1, 5, 3)

# Initialize Components
vector_db = get_vector_db()
llm = ChatGroq(
    groq_api_key=os.getenv("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"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)
adv_rag = AdvancedRAGPipeline(vector_db.as_retriever(), llm)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("🔍 Sources"):
                st.write(msg["sources"])


if prompt := st.chat_input("Ask your question about Machine Learning..."):
    # 1. User ka sawal display krna
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        result = adv_rag.query(prompt, top_k=num_chunks, summarize=show_summary)
        answer = result['answer']
        st.markdown(answer)

        # --- LOGIC CHECKS ---
        greetings = ["hi", "hello", "hey", "greetings", "good morning", "how are you"]
        is_greeting = any(greet in prompt.lower() for greet in greetings) and len(prompt.split()) < 4
        

        # agr answer mein koi no info wala phrase hai toh uske liay view sources na dikhayen
        no_info_keywords = ["I am sorry", "does not contain", "no information"]
        has_no_answer = any(key in answer for key in no_info_keywords)

        # --- DISPLAY LOGIC (Sirf aik baar summary dikhane ke liye) ---
        if not is_greeting and not has_no_answer:
            # A. Summary check (Sirf yahan dikhayega)
            if show_summary and result['summary']:
                with st.expander("✨ Summary"):
                    st.write(result['summary'])
            
            # B. Sources check
            with st.expander("🔍 Verify from Book (Ctrl + F Snippets)"):
                st.write("Find below the relevant snippets from the book:")
                for i, source in enumerate(result['sources']):
                    st.success(f"📖 **Book:** {source['book']} | 📄 **Page:** {source['page']}")
                    st.code(source['snippet'], language=None) 
                    st.divider()
        
        elif has_no_answer:
            st.warning("No relevant sources found in the document for this specific question.")

    # Assistant ka jawab history mein save hota hai
    st.session_state.messages.append({"role": "assistant", "content": answer})
