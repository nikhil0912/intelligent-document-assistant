# 🤖 Intelligent Document Assistant

> An advanced AI-powered document analysis and Q&A system using LangChain for intelligent document processing, extraction, and conversation with multiple LLM providers and vector-based retrieval (RAG).

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## ✨ Features

- 📄 **Multi-Format Document Support**: Process PDF, TXT, DOCX, and Markdown files
- 🤖 **LLM Integration**: Support for OpenAI, Hugging Face, and local LLMs
- 🔗 **LangChain Framework**: Advanced prompt chaining and memory management
- 📊 **Vector Database**: Embedding-based document retrieval with FAISS
- 💬 **RAG (Retrieval Augmented Generation)**: Accurate Q&A with source context
- ⚡ **Real-time Processing**: Stream responses for better UX
- 🔒 **Error Handling**: Robust exception handling and logging
- 📱 **REST API**: Flask-based API endpoints for easy integration
- 🧪 **Production Ready**: Comprehensive testing and documentation

## 🛠️ Tech Stack

### Core
- **Python 3.9+** - Programming language
- **LangChain** - LLM orchestration framework
- **LangGraph** - State management for complex workflows

### LLM & Embeddings
- **OpenAI GPT-3.5/GPT-4** - Primary LLM
- **HuggingFace Transformers** - Alternative models
- **Sentence Transformers** - Embedding models

### Data & Storage
- **FAISS** - Vector database for similarity search
- **Pandas** - Data manipulation
- **PyPDF2** - PDF processing
- **python-docx** - DOCX file handling

### Web & API
- **Flask** - Web framework
- **Flask-CORS** - Cross-origin support
- **Pydantic** - Data validation

### Development
- **pytest** - Testing framework
- **python-dotenv** - Environment variables
- **logging** - Application logging

## 📁 Project Structure

```
intelligent-document-assistant/
├── README.md
├── requirements.txt
├── .env.example
├── config.py
├── src/
│   ├── __init__.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── provider.py          # LLM provider abstraction
│   │   └── config.py            # LLM configurations
│   ├── document/
│   │   ├── __init__.py
│   │   ├── loader.py            # Document loading logic
│   │   ├── processor.py         # Document processing
│   │   └── splitter.py          # Text chunking strategies
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── manager.py           # Embedding management
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── faiss_store.py       # FAISS vector store
│   │   └── retriever.py         # RAG retriever
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── qa_agent.py          # Q&A agent logic
│   │   ├── memory.py            # Conversation memory
│   │   └── tools.py             # Agent tools
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py               # Flask application
│   │   ├── routes.py            # API endpoints
│   │   └── schemas.py           # Request/response schemas
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Logging configuration
│       └── validators.py        # Input validation
├── tests/
│   ├── __init__.py
│   ├── test_document_loader.py
│   ├── test_embeddings.py
│   ├── test_retriever.py
│   └── test_api.py
└── examples/
    ├── basic_qa.py              # Basic Q&A example
    ├── multi_document.py        # Multi-document analysis
    └── streaming_response.py    # Streaming response example
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip or conda
- OpenAI API key (or alternative LLM)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/nikhil0912/intelligent-document-assistant.git
   cd intelligent-document-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

## 📚 Usage

### Basic Document Q&A

```python
from src.llm import OpenAIProvider
from src.document import DocumentLoader
from src.agent import QAAgent

# Initialize components
llm = OpenAIProvider(model='gpt-3.5-turbo')
loader = DocumentLoader()
agent = QAAgent(llm=llm)

# Load document
document = loader.load('path/to/document.pdf')

# Ask questions
response = agent.ask('What is the main topic?', document=document)
print(response)
```

### Using REST API

```bash
# Start the server
python -m src.api.app

# Upload document
curl -X POST http://localhost:5000/api/upload \
  -F "file=@document.pdf"

# Ask question
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the document", "document_id": "doc_123"}'
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_document_loader.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 📖 Examples

See the `examples/` directory for complete working examples:
- `basic_qa.py` - Simple document Q&A
- `multi_document.py` - Analyze multiple documents
- `streaming_response.py` - Stream responses for real-time updates

## 🔧 Configuration

Edit `config.py` or `.env` to customize:

```python
# LLM Configuration
LLM_PROVIDER = "openai"  # or "huggingface", "local"
LLM_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.7

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Retrieval Configuration
TOP_K_DOCS = 3
SCORE_THRESHOLD = 0.5

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 5000
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

MIT License - see LICENSE file for details

## 👨‍💻 Author

**Nikhil Budhiraja**
- GitHub: [@nikhil0912](https://github.com/nikhil0912)
- LinkedIn: [nikhilbudhiraja](https://linkedin.com/in/nikhilbudhiraja)
- Email: nikhilbudhiraja002@gmail.com

## ⭐ Acknowledgments

- LangChain framework for excellent LLM orchestration
- OpenAI for GPT models
- FAISS for efficient similarity search
- The open-source community

## 🔄 Roadmap

- [ ] Support for more document formats (XLS, PPT, HTML)
- [ ] Multi-modal support (images, audio)
- [ ] Advanced caching strategies
- [ ] Web UI dashboard
- [ ] Docker support
- [ ] Database integration (PostgreSQL)
- [ ] Advanced analytics and monitoring

---

**Made with ❤️ for AI enthusiasts and developers**

Give a ⭐️ if you find this project helpful!
