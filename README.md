# Enhanced Document Assistant with Advanced Retrieval

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1.x-orange.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.x-green.svg)

A sophisticated document question-answering system with hybrid retrieval capabilities, combining vector search and keyword search with conversational AI.

## Features

- **Hybrid Retrieval**: Combines vector embeddings (Google Generative AI) with BM25 keyword search
- **Conversational AI**: Maintains context across multiple turns in conversation
- **Advanced Query Understanding**: Analyzes and expands queries for better retrieval
- **Performance Tracking**: Built-in evaluation and debugging tools
- **Persistent Storage**: ChromaDB database for document embeddings

## Prerequisites

- Python 3.8+
- Google API key (for Gemini models)
- Poetry (recommended for dependency management)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/enhanced-document-assistant.git
cd enhanced-document-assistant
```

2. Install dependencies:
```bash
poetry install
# or
pip install -r requirements.txt
```

3. Create a `.env` file with your API key:
```env
GOOGLE_API_KEY=your_api_key_here
```

## Usage

1. Place your text documents in the `data/` directory
2. Run the assistant:
```bash
python basic_rag.py
```

### Commands
- Ask natural language questions about your documents
- Type `debug` to see retrieval details
- Type `quit` to exit and view session statistics

## Configuration

Customize these parameters in the code:
- `chunk_size` and `chunk_overlap` in text splitting
- Retrieval parameters (k, score thresholds)
- LLM model selection (Gemini version)

## Project Structure

```
.
├── data/                   # Directory for text documents
├── db/                     # ChromaDB storage
├── .env                    # Environment variables
├── basic_rag.py   # Main application
├── pyproject.toml          # Poetry config
└── README.md               # This file
```

## Performance Optimization

The system implements several advanced techniques:
- Reciprocal Rank Fusion for result blending
- Query expansion and reformulation
- Context-aware retrieval
- Continuous performance evaluation

## Troubleshooting

**Issue**: Poor retrieval results
- **Solution**: Adjust chunking parameters or retrieval thresholds

**Issue**: API errors
- **Solution**: Verify your Google API key is valid

**Issue**: Missing dependencies
- **Solution**: Reinstall requirements with `poetry install` or `pip install -r requirements.txt`

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- LangChain team for the framework
- Google for the Gemini models
- ChromaDB for the vector database
