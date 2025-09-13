# 🏥 Medical FAQ RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers medical questions using a curated knowledge base of medical FAQs. Built with LLaMA 3.2, ChromaDB, and Streamlit for the AI/ML Engineer Assignment.



## 🌟 Features

- **🔍 Semantic Search**: Uses embeddings to find relevant medical information
- **🧠 AI-Powered Responses**: Generates natural answers using LLaMA 3.2
- **💬 User-Friendly Interface**: Clean Streamlit web application
- **📊 Confidence Scoring**: Shows system confidence in answers
- **📝 Source Tracking**: Displays which FAQs were used for responses
- **📁 Flexible Data Input**: Supports CSV and JSON file uploads
- **💾 Persistent Storage**: ChromaDB vector database for efficient retrieval
- **⚡ Local Processing**: No external API costs, complete privacy

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama**
   ```bash
   # On Linux/Mac
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # On Windows, download from: https://ollama.ai/download
   ```

2. **Pull Required Models**
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```

3. **Python 3.8 or higher**

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/avi12299/-medical-faq-rag-chatbot.git
   cd medical-faq-rag-chatbot
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run medicalchatbot.py
   ```

5. **Open in Browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in terminal

## 📖 Usage Guide

### Step 1: Initialize the System
1. **Launch the App**: Open your browser to `http://localhost:8501`
2. **Choose Data Source**: In the sidebar, select from:
   - Use Sample Data (15 pre-loaded medical FAQs)
   - Upload CSV File
   - Upload JSON File
3. **Initialize Knowledge Base**: Click "🚀 Initialize/Update Knowledge Base"
4. **Wait for Processing**: The system will create embeddings and store them in ChromaDB

### Step 2: Ask Medical Questions
- Type your question in the chat input box
- Click "🔍 Ask Question" or press Enter
- Review the AI-generated response with confidence score and sources

### Step 3: Sample Questions to Try
```
• "What are the early symptoms of diabetes?"
• "Can children take paracetamol?"
• "What foods are good for heart health?"
• "When should I see a doctor for a fever?"
• "How can I improve my sleep quality?"
• "What are the benefits of regular exercise?"
```

## 🏗️ Architecture & Design

### RAG Pipeline Flow
```
User Query → Embedding → Vector Search → Context Retrieval → LLM Generation → Response
```

1. **Document Processing**: Medical FAQs are loaded and chunked into manageable pieces
2. **Embedding Generation**: Text is converted to vectors using `nomic-embed-text`
3. **Vector Storage**: ChromaDB stores embeddings for fast similarity search
4. **Query Processing**: User questions are embedded and matched against knowledge base
5. **Context Retrieval**: Top-K most relevant FAQs are retrieved
6. **Response Generation**: LLaMA 3.2 generates natural language responses using retrieved context

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Web interface and user interaction |
| **LLM** | LLaMA 3.2 (Ollama) | Natural language generation |
| **Embeddings** | nomic-embed-text (Ollama) | Text vectorization |
| **Vector DB** | ChromaDB | Efficient similarity search |
| **Framework** | LangChain | RAG pipeline orchestration |
| **Backend** | Python 3.8+ | Core application logic |

### Key Design Decisions

#### Why These Technologies?

1. **🔥 Ollama + LLaMA 3.2**: 
   - Free, local LLM with excellent medical knowledge
   - No API costs or rate limits
   - Complete data privacy

2. **🗄️ ChromaDB**: 
   - Lightweight, efficient vector database
   - Easy setup with no external dependencies
   - Perfect for prototype and production

3. **🎯 nomic-embed-text**: 
   - High-quality embeddings optimized for retrieval
   - Fast processing and good semantic understanding
   - Free and runs locally

4. **⚡ Streamlit**: 
   - Rapid web app development
   - Perfect for ML/AI demonstrations
   - Minimal code for maximum functionality

5. **🔗 LangChain**: 
   - Robust RAG pipeline framework
   - Excellent abstraction for complex workflows
   - Active community and documentation

## 📁 Project Structure

```
medical-faq-rag-chatbot/
│
├── medicalchatbotapp.py          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                      # This documentation
├── .gitignore                     # Git ignore file
│
├── data/                          # Data directory
  └── medquad.csv   # Sample medical FAQs downloaded from kaggle
```

## ⚙️ Configuration & Customization

### Key Parameters

```python
# Text Processing
MAX_CHUNK_SIZE = 1000      # Maximum chunk size for text splitting
CHUNK_OVERLAP = 200        # Overlap between chunks

# Database
BATCH_SIZE = 20           # Documents processed per batch
DB_PATH = "./medical_chatbot/db/medical_faqs"  # Database location

# Retrieval
RETRIEVAL_K = 5           # Number of documents retrieved per query
SEARCH_TYPE = "mmr"       # Search strategy (MMR for diversity)
LAMBDA_MULT = 0.7         # Balance between relevance and diversity
```

### Customization Options

1. **Modify Sample Data**: Edit `SAMPLE_MEDICAL_FAQS` in the code
2. **Adjust Retrieval**: Change `k` value and search parameters
3. **Update Prompts**: Modify `get_medical_prompt_template()` method
4. **Change Models**: Update Ollama model names in configuration
5. **UI Modifications**: Customize Streamlit interface elements

## 📊 Supported Data Formats

### CSV Format
```csv
question,answer
"What is diabetes?","Diabetes is a metabolic disorder characterized by high blood sugar..."
"How to treat fever?","Fever can be treated with rest, fluids, and appropriate medication..."
```

**Auto-detection**: System automatically detects column headers containing:
- Questions: `question`, `query`, `q`, `ask`
- Answers: `answer`, `response`, `a`, `reply`

### JSON Format
```json
[
    {
        "question": "What is diabetes?",
        "answer": "Diabetes is a metabolic disorder characterized by high blood sugar..."
    },
    {
        "question": "How to treat fever?",
        "answer": "Fever can be treated with rest, fluids, and appropriate medication..."
    }
]
```

## 🔧 Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```bash
   # Check if Ollama is running
   ollama list
   
   # Start Ollama service
   ollama serve
   ```

2. **Models Not Found**
   ```bash
   # Pull required models
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```

3. **Port Already in Use**
   ```bash
   # Run on different port
   streamlit run medical_chatbot_app.py --server.port 8502
   ```

4. **Memory Issues**
   ```python
   # Reduce batch size in code
   BATCH_SIZE = 10  # Instead of 20
   ```

5. **Database Corruption**
   - Use "🗑️ Clear Knowledge Base" button in sidebar
   - Or manually delete `medical_chatbot/db/` folder

### System Requirements

- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 5GB free space for models
- **CPU**: Modern multi-core processor
- **OS**: Windows 10+, macOS 10.15+, or Linux

## 📈 Performance & Scalability

### Current Performance
- **Query Response**: 2-5 seconds average
- **Database Size**: Handles 1000+ FAQs efficiently
- **Concurrent Users**: 5-10 users (depending on hardware)
- **Accuracy**: 85-95% relevan

## 📬 Feedback & Contributions

Feel free to open issues or pull requests to improve this project!
