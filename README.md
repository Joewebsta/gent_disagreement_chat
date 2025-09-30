# A Gentleman's Disagreement Chat

An intelligent chat application powered by Retrieval-Augmented Generation (RAG) for analyzing and querying podcast transcripts from **A Gentleman's Disagreement Podcast**. The system uses advanced query enhancement, vector search, and adaptive thresholding to provide accurate, context-aware responses.

## Features

- **RAG-Powered Chat Interface**: Natural language querying of podcast transcripts with contextual responses
- **Advanced Query Enhancement**: Multi-query search with HyDE (Hypothetical Document Embeddings) and query expansion
- **Episode Grouping**: Intelligent organization of search results by episode with relevance scoring
- **Adaptive Thresholding**: Dynamic similarity threshold adjustment for optimal retrieval quality
- **Vector Search**: Semantic search using sentence transformers and PostgreSQL with pgvector
- **Real-time Streaming**: AI SDK-compatible text streaming for responsive user experience
- **Modern UI**: React-based chat interface with markdown support, syntax highlighting, and responsive design

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│                 │         │                  │         │                 │
│  React Client   │────────▶│  FastAPI Server  │────────▶│   PostgreSQL    │
│   (Vite/TS)     │         │   (Python 3.11)  │         │   + pgvector    │
│                 │         │                  │         │                 │
└─────────────────┘         └──────────────────┘         └─────────────────┘
                                     │
                                     │
                                     ▼
                            ┌──────────────────┐
                            │                  │
                            │   OpenAI API     │
                            │  (GPT-4o-mini)   │
                            │                  │
                            └──────────────────┘
```

## Tech Stack

### Client
- **Framework**: React 19 with TypeScript
- **Build Tool**: Vite
- **UI Components**: Radix UI, Tailwind CSS
- **AI Integration**: Vercel AI SDK
- **Markdown**: react-markdown with KaTeX and syntax highlighting
- **State Management**: React hooks

### Server
- **Framework**: FastAPI
- **Language**: Python 3.11+
- **Database**: PostgreSQL with psycopg2
- **Vector Search**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM Integration**: OpenAI API
- **Package Manager**: Poetry
- **Dependencies**: numpy, python-dotenv

## Prerequisites

- **Node.js**: v18+ and npm
- **Python**: 3.11 or higher
- **Poetry**: Python package manager
- **PostgreSQL**: v12+ with pgvector extension
- **OpenAI API Key**: For LLM integration

## Setup Instructions

### 1. Database Setup

Ensure PostgreSQL is installed and running:

```bash
# Create database
createdb gent_disagreement

# Install pgvector extension (if needed)
psql gent_disagreement -c 'CREATE EXTENSION vector;'
```

### 2. Server Setup

```bash
cd server

# Install dependencies using Poetry
poetry install

# Create .env file
cat > .env << EOF
# Database connection settings
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=gent_disagreement

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key
EOF

# Initialize database and load transcripts (if needed)
# poetry run python -m gent_disagreement_chat.data.setup_database
```

### 3. Client Setup

```bash
cd client

# Install dependencies
npm install

# Create .env.local file
cat > .env.local << EOF
# API Configuration
VITE_API_URL=http://localhost:8000/api/chat
EOF
```

## Running the Application

### Start the Server

```bash
cd server
poetry run uvicorn gent_disagreement_chat.main:app --host 0.0.0.0 --port 8000

# Or use the Poetry script
poetry run start
```

The server will be available at `http://localhost:8000`

### Start the Client

```bash
cd client
npm run dev
```

The client will be available at `http://localhost:5173`

### Health Check

Verify the server is running:
```bash
curl http://localhost:8000/health
# Response: {"status":"healthy"}
```

## Project Structure

```
gent_disagreement_chat/
├── client/                          # React frontend application
│   ├── src/
│   │   ├── components/              # UI components (AI elements, UI primitives)
│   │   ├── assets/                  # Images and static assets
│   │   ├── lib/                     # Utility functions
│   │   ├── App.tsx                  # Main application component
│   │   └── main.tsx                 # Application entry point
│   ├── package.json                 # Client dependencies
│   └── vite.config.ts               # Vite configuration
│
├── server/                          # FastAPI backend application
│   ├── src/gent_disagreement_chat/
│   │   ├── core/                    # Core RAG functionality
│   │   │   ├── rag_service.py       # Main RAG service with query enhancement
│   │   │   ├── query_enhancer.py    # Query preprocessing and enhancement
│   │   │   ├── vector_search.py     # Vector similarity search
│   │   │   ├── embedding_service.py # Embedding generation
│   │   │   └── database_manager.py  # Database connection management
│   │   ├── evaluation/              # Evaluation and metrics tracking
│   │   ├── data/                    # Data loading and management
│   │   └── main.py                  # FastAPI application entry point
│   ├── pyproject.toml               # Python dependencies (Poetry)
│   └── evaluation_metrics.db        # SQLite database for evaluation metrics
│
├── docs/                            # Documentation
│   └── PRECISION_TRACKING_GUIDE.md  # Evaluation framework documentation
│
├── .env                             # Environment variables (not in git)
└── README.md                        # This file
```

## API Endpoints

### Core Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check endpoint
- `POST /api/chat` - AI SDK-compatible chat endpoint

### Chat Endpoint

**Request:**
```json
{
  "messages": [
    {
      "role": "user",
      "text": "What are the hosts' views on technology?"
    }
  ]
}
```

**Response:** Text stream compatible with Vercel AI SDK's `TextStreamChatTransport`

## RAG System Overview

### Query Enhancement

The system uses a sophisticated `QueryEnhancer` class that implements:

1. **Query Preprocessing**: Cleans and normalizes input queries
2. **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical answers to improve retrieval
3. **Query Expansion**: Expands queries with podcast-specific vocabulary and context
4. **Multi-Query Search**: Combines results from original, expanded, and HyDE queries

### Vector Search

Vector search is powered by:
- **Embedding Model**: `all-MiniLM-L6-v2` from sentence-transformers
- **Storage**: PostgreSQL with pgvector extension
- **Adaptive Thresholding**: Dynamically adjusts similarity thresholds based on score distribution
- **Episode Grouping**: Organizes results by episode with relevance scoring

### Adaptive Thresholding

The system automatically adjusts similarity thresholds to:
- Ensure minimum document count (default: 3)
- Limit maximum document count (default: 10)
- Filter low-quality results (default threshold: 0.6)
- Adapt based on score distribution and gaps

## Environment Variables

### Server (.env)

| Variable | Description | Required |
|----------|-------------|----------|
| `DB_HOST` | PostgreSQL host | Yes |
| `DB_PORT` | PostgreSQL port | Yes |
| `DB_USER` | Database user | Yes |
| `DB_PASSWORD` | Database password | Yes |
| `DB_NAME` | Database name | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Yes |

### Client (.env.local)

| Variable | Description | Required |
|----------|-------------|----------|
| `VITE_API_URL` | Backend API URL | Yes |

## License

MIT License

Copyright (c) 2025 Joe Webster

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author

**Joe Webster**
Email: joseph.scott.webster@gmail.com

## Acknowledgments

- **A Gentleman's Disagreement Podcast** for the transcript content
- **Vercel AI SDK** for seamless AI integration
- **OpenAI** for GPT models
- **Sentence Transformers** for embedding models