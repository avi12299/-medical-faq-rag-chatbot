import os
import json
import csv
import streamlit as st
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from typing import List, Dict, Any

# Configuration
MEDICAL_FAQ_CSV_PATH = "data\medquad.csv"  # Path for medical FAQ CSV
DB_PATH = "./medical_chatbot/db/medical_faqs"
BATCH_SIZE = 20  # Process documents in batches
MAX_CHUNK_SIZE = 1000  # Maximum chunk size for text splitting
CHUNK_OVERLAP = 200    # Overlap between chunks

# Sample Medical FAQs (for demonstration if no file is provided)
SAMPLE_MEDICAL_FAQS = [
    {
        "question": "What are the early symptoms of diabetes?",
        "answer": "Early symptoms of diabetes include increased thirst, frequent urination, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections. Type 1 diabetes symptoms can develop quickly, while Type 2 diabetes symptoms develop more gradually and may be subtle initially."
    },
    {
        "question": "Can children take paracetamol?",
        "answer": "Yes, children can take paracetamol, but the dosage must be appropriate for their age and weight. For children 2 months to 18 years: use pediatric formulations and follow dosing guidelines. Never exceed the maximum daily dose. Always consult a healthcare provider for children under 2 months or if unsure about dosing."
    },
    {
        "question": "What foods are good for heart health?",
        "answer": "Heart-healthy foods include: fatty fish (salmon, mackerel), leafy green vegetables, whole grains, berries, avocados, nuts and seeds, olive oil, tomatoes, and legumes. These foods are rich in omega-3 fatty acids, antioxidants, fiber, and healthy fats that support cardiovascular health."
    },
    {
        "question": "How much water should I drink daily?",
        "answer": "The general recommendation is about 8 glasses (64 ounces) of water per day, but individual needs vary based on activity level, climate, overall health, and pregnancy/breastfeeding status. A good indicator is pale yellow urine. Increase intake during exercise, hot weather, or illness."
    },
    {
        "question": "What are the side effects of high blood pressure?",
        "answer": "High blood pressure often has no symptoms (silent killer) but can cause headaches, shortness of breath, dizziness, chest pain, and nosebleeds in severe cases. Long-term effects include increased risk of heart disease, stroke, kidney damage, and vision problems."
    },
    {
        "question": "How can I improve my sleep quality?",
        "answer": "To improve sleep quality: maintain a consistent sleep schedule, create a comfortable sleep environment (cool, dark, quiet), avoid screens before bedtime, limit caffeine and alcohol, exercise regularly but not close to bedtime, and establish a relaxing bedtime routine."
    },
    {
        "question": "What are the symptoms of anxiety?",
        "answer": "Anxiety symptoms include excessive worry, restlessness, fatigue, difficulty concentrating, irritability, muscle tension, and sleep disturbances. Physical symptoms may include rapid heartbeat, sweating, trembling, shortness of breath, and digestive issues."
    },
    {
        "question": "When should I see a doctor for a fever?",
        "answer": "See a doctor for fever if: temperature exceeds 103°F (39.4°C), fever lasts more than 3 days, accompanied by severe symptoms like difficulty breathing, persistent vomiting, severe headache, stiff neck, or unusual rash. For infants under 3 months, any fever requires immediate medical attention."
    },
    {
        "question": "What are the benefits of regular exercise?",
        "answer": "Regular exercise benefits include improved cardiovascular health, stronger bones and muscles, better mental health, weight management, reduced risk of chronic diseases, improved sleep, increased energy levels, and enhanced immune function. Aim for at least 150 minutes of moderate activity weekly."
    },
    {
        "question": "How can I boost my immune system naturally?",
        "answer": "Natural ways to boost immunity include: eating a balanced diet rich in fruits and vegetables, getting adequate sleep (7-9 hours), managing stress, exercising regularly, staying hydrated, avoiding smoking, limiting alcohol, washing hands frequently, and maintaining a healthy weight."
    }
]

class MedicalFAQChatbot:
    def __init__(self):
        self.embedding_model = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def initialize_embedding_model(self):
        """Initialize the embedding model using Ollama"""
        try:
            if self.embedding_model is None:
                self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")
            return self.embedding_model
        except Exception as e:
            st.error(f"Failed to initialize embedding model: {str(e)}")
            st.error("Please ensure Ollama is running with nomic-embed-text model installed.")
            return None
    
    def load_llm(self):
        """Initialize the LLaMA model using Ollama"""
        try:
            if self.llm is None:
                self.llm = ChatOllama(model="llama3.2", temperature=0.1)
            return self.llm
        except Exception as e:
            st.error(f"Failed to initialize LLM: {str(e)}")
            st.error("Please ensure Ollama is running with llama3.2 model installed.")
            return None
    
    def load_medical_faqs_from_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """Load medical FAQs from CSV file"""
        faqs = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                # Try to detect if file has headers
                sample = file.read(1024)
                file.seek(0)
                sniffer = csv.Sniffer()
                has_header = sniffer.has_header(sample)
                
                reader = csv.DictReader(file) if has_header else csv.reader(file)
                
                if has_header:
                    # Automatically detect question and answer columns
                    fieldnames = reader.fieldnames
                    question_col = None
                    answer_col = None
                    
                    # Look for common question column names
                    for field in fieldnames:
                        if any(q_word in field.lower() for q_word in ['question', 'query', 'q', 'ask']):
                            question_col = field
                        elif any(a_word in field.lower() for a_word in ['answer', 'response', 'a', 'reply']):
                            answer_col = field
                    
                    # If not found, use first two columns
                    if not question_col or not answer_col:
                        question_col = fieldnames[0]
                        answer_col = fieldnames[1] if len(fieldnames) > 1 else fieldnames[0]
                    
                    for row in reader:
                        if row[question_col] and row[answer_col]:
                            faqs.append({
                                'question': row[question_col].strip(),
                                'answer': row[answer_col].strip()
                            })
                else:
                    # No headers, assume first column is question, second is answer
                    for row in reader:
                        if len(row) >= 2 and row[0] and row[1]:
                            faqs.append({
                                'question': row[0].strip(),
                                'answer': row[1].strip()
                            })
                        
        except FileNotFoundError:
            st.warning(f"CSV file not found at {csv_path}. Using sample data.")
            faqs = SAMPLE_MEDICAL_FAQS
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}. Using sample data.")
            faqs = SAMPLE_MEDICAL_FAQS
            
        return faqs
    
    def load_medical_faqs_from_json(self, json_path: str) -> List[Dict[str, Any]]:
        """Load medical FAQs from JSON file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    # If it's a dict, try to extract FAQ list
                    for key in ['faqs', 'questions', 'data', 'medical_faqs']:
                        if key in data and isinstance(data[key], list):
                            return data[key]
                    # If no standard key found, convert dict to list
                    return [data]
                else:
                    st.error("Invalid JSON structure")
                    return SAMPLE_MEDICAL_FAQS
                    
        except FileNotFoundError:
            st.warning(f"JSON file not found at {json_path}. Using sample data.")
            return SAMPLE_MEDICAL_FAQS
        except Exception as e:
            st.error(f"Error loading JSON file: {str(e)}. Using sample data.")
            return SAMPLE_MEDICAL_FAQS
    
    def prepare_documents(self, faqs: List[Dict[str, Any]]) -> List[Document]:
        """Convert FAQs to LangChain documents and split them"""
        documents = []
        
        for i, faq in enumerate(faqs):
            # Create comprehensive content for better retrieval
            content = f"Question: {faq['question']}\n\nAnswer: {faq['answer']}"
            
            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    'faq_id': i,
                    'question': faq['question'],
                    'answer': faq['answer'],
                    'source': 'medical_faq_database'
                }
            )
            documents.append(doc)
        
        # Split documents if they're too large
        split_documents = self.text_splitter.split_documents(documents)
        
        return split_documents
    
    def create_vector_database(self, documents: List[Document], progress_bar, status_text):
        """Create vector database using ChromaDB"""
        embedding_model = self.initialize_embedding_model()
        if not embedding_model:
            return None
        
        # Create directory if it doesn't exist
        os.makedirs(DB_PATH, exist_ok=True)
        
        # Process documents in batches
        total_docs = len(documents)
        vectorstore = None
        
        try:
            for i in range(0, total_docs, BATCH_SIZE):
                batch = documents[i:i + BATCH_SIZE]
                batch_num = (i // BATCH_SIZE) + 1
                total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
                
                status_text.text(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                
                if vectorstore is None:
                    # Create new vectorstore for first batch
                    vectorstore = Chroma.from_documents(
                        batch, 
                        embedding_model, 
                        persist_directory=DB_PATH
                    )
                else:
                    # Add to existing vectorstore
                    vectorstore.add_documents(batch)
                
                # Update progress
                progress = min((i + BATCH_SIZE) / total_docs, 1.0)
                progress_bar.progress(progress)
                
                # Small delay to show progress
                time.sleep(0.1)
            
            # Persist the vectorstore
            if vectorstore:
                vectorstore.persist()
                
        except Exception as e:
            st.error(f"Error creating vector database: {str(e)}")
            return None
        
        return vectorstore
    
    def load_or_create_vectorstore(self, faqs: List[Dict[str, Any]], progress_bar, status_text):
        """Load existing vectorstore or create new one"""
        if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
            status_text.text("Loading existing vector database...")
            try:
                embedding_model = self.initialize_embedding_model()
                if embedding_model:
                    vectorstore = Chroma(
                        persist_directory=DB_PATH, 
                        embedding_function=embedding_model
                    )
                    progress_bar.progress(1.0)
                    return vectorstore
                else:
                    return None
            except Exception as e:
                st.error(f"Error loading existing database: {str(e)}")
                st.info("Creating new database...")
        
        # Create new vectorstore
        status_text.text("Creating new vector database...")
        documents = self.prepare_documents(faqs)
        return self.create_vector_database(documents, progress_bar, status_text)
    
    def build_qa_chain(self):
        """Build QA chain for medical FAQ chatbot"""
        if not self.vectorstore:
            return None
            
        llm = self.load_llm()
        if not llm:
            return None
        
        try:
            # Create retriever with optimized settings for medical queries
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",  # Maximal Marginal Relevance for diverse results
                search_kwargs={
                    "k": 5,  # Retrieve top 5 relevant documents
                    "lambda_mult": 0.7  # Balance between relevance and diversity
                }
            )
            
            # Build QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": self.get_medical_prompt_template()
                }
            )
            
            return self.qa_chain
            
        except Exception as e:
            st.error(f"Error building QA chain: {str(e)}")
            return None
    
    def get_medical_prompt_template(self):
        """Create a specialized prompt template for medical queries"""
        from langchain.prompts import PromptTemplate
        
        template = """You are a helpful medical information assistant. Use the following context from medical FAQs to answer the user's question. 

IMPORTANT GUIDELINES:
1. Only provide information based on the context provided
2. If the context doesn't contain relevant information, clearly state that
3. Always recommend consulting healthcare professionals for medical advice
4. Be clear, concise, and accurate in your responses
5. If discussing symptoms or treatments, emphasize the importance of professional medical consultation

Context from Medical FAQs:
{context}

Human Question: {question}

Medical Assistant Response:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Process a medical question and return response with sources"""
        if not self.qa_chain:
            return {
                "answer": "Sorry, the medical knowledge base is not ready. Please initialize the system first.",
                "sources": [],
                "confidence": 0.0
            }
        
        try:
            # Process the question
            result = self.qa_chain.invoke({"query": question})
            
            answer = result.get('result', 'No answer found.')
            source_docs = result.get('source_documents', [])
            
            # Extract source information
            sources = []
            for doc in source_docs[:3]:  # Limit to top 3 sources
                if hasattr(doc, 'metadata'):
                    sources.append({
                        'question': doc.metadata.get('question', 'N/A'),
                        'faq_id': doc.metadata.get('faq_id', 'N/A')
                    })
            
            # Calculate simple confidence score
            confidence = min(len(source_docs) * 0.2, 1.0)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def clear_database(self):
        """Clear the vector database"""
        try:
            import shutil
            if os.path.exists(DB_PATH):
                shutil.rmtree(DB_PATH)
                return True
        except Exception as e:
            st.error(f"Error clearing database: {str(e)}")
        return False

def main():
    st.set_page_config(
        page_title="Medical FAQ RAG Chatbot",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏥 Medical FAQ RAG Chatbot")
    st.markdown("Ask medical questions and get answers from our curated medical knowledge base")
    st.markdown("⚠️ **Disclaimer**: This is for informational purposes only. Always consult healthcare professionals for medical advice.")
    st.markdown("---")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = MedicalFAQChatbot()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for setup and configuration
    with st.sidebar:
        st.header("🔧 Setup & Configuration")
        
        # Data source selection
        st.subheader("📊 Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Use Sample Data", "Upload CSV File", "Upload JSON File"]
        )
        
        faqs_data = None
        
        if data_source == "Use Sample Data":
            faqs_data = SAMPLE_MEDICAL_FAQS
            st.success(f"✅ Using {len(SAMPLE_MEDICAL_FAQS)} sample medical FAQs")
        
        elif data_source == "Upload CSV File":
            uploaded_csv = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_csv:
                # Save uploaded file temporarily
                with open("temp_medical_faqs.csv", "wb") as f:
                    f.write(uploaded_csv.getvalue())
                faqs_data = st.session_state.chatbot.load_medical_faqs_from_csv("temp_medical_faqs.csv")
                st.success(f"✅ Loaded {len(faqs_data)} FAQs from CSV")
        
        elif data_source == "Upload JSON File":
            uploaded_json = st.file_uploader("Upload JSON file", type=['json'])
            if uploaded_json:
                # Save uploaded file temporarily
                with open("temp_medical_faqs.json", "wb") as f:
                    f.write(uploaded_json.getvalue())
                faqs_data = st.session_state.chatbot.load_medical_faqs_from_json("temp_medical_faqs.json")
                st.success(f"✅ Loaded {len(faqs_data)} FAQs from JSON")
        
        # Database status
        st.subheader("💾 Database Status")
        db_exists = os.path.exists(DB_PATH) and os.listdir(DB_PATH)
        
        if db_exists:
            st.success("✅ Vector database exists")
        else:
            st.info("ℹ️ Vector database not found")
        
        # Initialize/Update database button
        if st.button("🚀 Initialize/Update Knowledge Base", type="primary"):
            if faqs_data:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                vectorstore = st.session_state.chatbot.load_or_create_vectorstore(
                    faqs_data, progress_bar, status_text
                )
                
                if vectorstore:
                    st.session_state.chatbot.vectorstore = vectorstore
                    qa_chain = st.session_state.chatbot.build_qa_chain()
                    
                    if qa_chain:
                        st.success("✅ Knowledge base initialized successfully!")
                        status_text.text("✅ Ready to answer medical questions!")
                    else:
                        st.error("❌ Failed to build QA chain")
                else:
                    st.error("❌ Failed to create knowledge base")
            else:
                st.error("Please select and load data first")
        
        # Clear database button
        if st.button("🗑️ Clear Knowledge Base"):
            if st.session_state.chatbot.clear_database():
                st.success("Knowledge base cleared!")
                st.rerun()
        
        # System information
        st.subheader("ℹ️ System Info")
        st.info(f"""
        **Embedding Model**: nomic-embed-text
        **LLM Model**: llama3.2
        **Vector DB**: ChromaDB
        **Max Chunk Size**: {MAX_CHUNK_SIZE}
        **Batch Size**: {BATCH_SIZE}
        """)
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Check if system is ready
    system_ready = (st.session_state.chatbot.vectorstore is not None and 
                   st.session_state.chatbot.qa_chain is not None)
    
    if not system_ready:
        st.warning("⚠️ Please initialize the knowledge base using the sidebar before asking questions.")
        # Still show sample questions for demonstration
        st.subheader("🎯 Sample Questions You Can Ask:")
        sample_questions = [
            "What are the early symptoms of diabetes?",
            "Can children take paracetamol?",
            "What foods are good for heart health?",
            "How much water should I drink daily?",
            "When should I see a doctor for a fever?",
            "What are the benefits of regular exercise?"
        ]
        
        for i, question in enumerate(sample_questions, 1):
            st.write(f"{i}. {question}")
    
    # Chat input
    user_question = st.text_input(
        "Ask a medical question:",
        placeholder="e.g., What are the symptoms of high blood pressure?",
        disabled=not system_ready
    )
    
    # Process question
    if st.button("🔍 Ask Question", disabled=not system_ready or not user_question):
        if user_question.strip():
            with st.spinner("Searching medical knowledge base..."):
                response = st.session_state.chatbot.ask_question(user_question)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": response["answer"],
                    "sources": response["sources"],
                    "confidence": response["confidence"]
                })
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("📋 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['question'][:100]}..." if len(chat['question']) > 100 else f"Q: {chat['question']}", expanded=(i==0)):
                st.write("**Question:**")
                st.write(chat['question'])
                
                st.write("**Answer:**")
                st.write(chat['answer'])
                
                # Show confidence and sources
                col1, col2 = st.columns(2)
                with col1:
                    confidence_color = "🟢" if chat['confidence'] > 0.7 else "🟡" if chat['confidence'] > 0.4 else "🔴"
                    st.write(f"**Confidence:** {confidence_color} {chat['confidence']:.1%}")
                
                with col2:
                    st.write(f"**Sources:** {len(chat['sources'])} relevant FAQs")
                
                # Show source details
                if chat['sources']:
                    st.write("**Related FAQ Questions:**")
                    for j, source in enumerate(chat['sources'], 1):
                        st.write(f"{j}. {source['question']}")
        
        # Clear chat history button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        **Getting Started:**
        1. **Choose Data Source**: Select from sample data or upload your own CSV/JSON file
        2. **Initialize Knowledge Base**: Click the initialize button in the sidebar
        3. **Ask Questions**: Type your medical questions in the chat interface
        
        **File Formats:**
        - **CSV**: Should have columns for questions and answers (auto-detected)
        - **JSON**: Array of objects with 'question' and 'answer' fields
        
        **Features:**
        - 🔍 **Smart Retrieval**: Uses semantic search to find relevant medical information
        - 🧠 **AI-Powered**: Generates natural responses using LLaMA 3.2
        - 📊 **Confidence Scoring**: Shows how confident the system is in its answers
        - 📝 **Source Tracking**: Shows which FAQs were used to generate answers
        - 💾 **Persistent Storage**: Knowledge base is saved and can be reused
        
        **Medical Disclaimer:**
        This chatbot provides general medical information only and should not replace professional medical advice, diagnosis, or treatment.
        """)
    
    # Requirements check
    with st.expander("🔧 System Requirements"):
        st.markdown("""
        **Required Software:**
        - Ollama installed and running
        - Models: `nomic-embed-text` and `llama3.2`
        
        **Installation Commands:**
        ```bash
        # Install Ollama (if not installed)
        curl -fsSL https://ollama.ai/install.sh | sh
        
        # Pull required models
        ollama pull nomic-embed-text
        ollama pull llama3.2
        ```
        
        **Python Dependencies:**
        ```bash
        pip install streamlit langchain-community langchain-ollama chromadb pandas
        ```
        """)

if __name__ == "__main__":
    main()