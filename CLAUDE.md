# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG-powered chat application for querying podcast transcripts from "A Gentleman's Disagreement Podcast". The system uses adaptive thresholding and vector search to provide accurate, context-aware responses.

**Architecture**: React/TypeScript frontend (Vite) + FastAPI Python backend + PostgreSQL with pgvector extension

## Development Commands

### Client (React/TypeScript)
```bash
cd client
npm install           # Install dependencies
npm run dev           # Start dev server (http://localhost:5173)
npm run build         # Build for production
npm run lint          # Run ESLint
npm run preview       # Preview production build
```

### Server (Python/FastAPI)
```bash
cd server
poetry install        # Install dependencies

# Start dev server with auto-reload (recommended for development):
poetry run uvicorn gent_disagreement_chat.main:app --host 0.0.0.0 --port 8000 --reload

# Health check:
curl http://localhost:8000/health
```

### Database Setup
```bash
# Create database
createdb gent_disagreement

# Install pgvector extension
psql gent_disagreement -c 'CREATE EXTENSION vector;'
```

## RAG System Architecture

The RAG system is the core of this application and consists of several interconnected components:

### Core Components (server/src/gent_disagreement_chat/core/)

1. **RAGService** (`rag_service.py`)
   - Main orchestrator for RAG operations
   - Manages vector search and response generation
   - Default parameters: `DEFAULT_THRESHOLD=0.6`, `DEFAULT_MIN_DOCS=3`, `DEFAULT_MAX_DOCS=10`
   - Implements `ask_question_text_stream()` for AI SDK-compatible text streaming

2. **VectorSearch** (`vector_search.py`)
   - Handles semantic search using pgvector
   - Key method: `find_relevant_above_adaptive_threshold()` - dynamically adjusts similarity thresholds based on score distribution
   - **Adaptive Thresholding**: Ensures min/max document counts while filtering low-quality results
   - **Episode Grouping**: Organizes results by episode with relevance scoring
   - Uses `all-MiniLM-L6-v2` embedding model (384 dimensions)

3. **EmbeddingService** (`embedding_service.py`)
   - Generates embeddings using sentence-transformers
   - Model: `all-MiniLM-L6-v2`

4. **DatabaseManager** (`database_manager.py`)
   - Manages PostgreSQL connections
   - Requires `DB_PASSWORD` environment variable (security-enforced)
   - Uses RealDictCursor for convenient result handling

### Query Flow
1. User query → `RAGService.ask_question_text_stream()`
2. Vector search → `VectorSearch.find_relevant_above_adaptive_threshold()` finds relevant transcript segments
3. Results grouped by episode and formatted as context
4. Context passed to GPT-4o-mini via streaming API
5. Response streamed back to client via AI SDK TextStreamChatTransport

**API Calls Per Query:**
- 1x OpenAI embedding generation (for user query)
- 2x PostgreSQL queries (adaptive threshold search + fallback)
- 1x OpenAI streaming call (for final answer generation)

## Evaluation System

The evaluation framework (server/src/gent_disagreement_chat/evaluation/) provides three levels of precision tracking:

### Components
- **evaluation_runner.py**: CLI for running baselines, evaluations, and generating reports
- **automated_metrics.py**: Calculates similarity, diversity, coverage, and coherence metrics
- **precision_tracker.py**: Tracks and compares evaluation results over time
- **ground_truth_generator.py**: LLM-assisted ground truth generation

### Usage
```bash
cd server/src/gent_disagreement_chat

# Create baseline
python -m evaluation.evaluation_runner baseline --questions 20

# Run evaluation against baseline
python -m evaluation.evaluation_runner evaluate --baseline-id <baseline_id>

# Generate comparison report
python -m evaluation.evaluation_runner report --baseline-id <baseline_id> --improvement-id <improvement_id>
```

Metrics are stored in SQLite database: `server/evaluation_metrics.db`

## Frontend Architecture

### Key Components (client/src/components/)
- **App.tsx**: Main app component using `useChat()` from AI SDK
- **ChatConversation.tsx**: Displays message history
- **ChatInput.tsx**: User input with suggestions
- **ChatMessage.tsx**: Individual message rendering with markdown support
- **ai-elements/**: AI-specific UI components from AI SDK
- **ui/**: Reusable Radix UI components with Tailwind CSS

### State Management
Uses `useChat()` hook from `@ai-sdk/react` with `TextStreamChatTransport`:
- Connects to backend via `VITE_API_URL` environment variable
- Handles message streaming, status tracking, and error states

### Markdown Rendering
Messages support:
- GitHub Flavored Markdown (remark-gfm)
- Math equations (KaTeX via rehype-katex, remark-math)
- Code syntax highlighting (react-syntax-highlighter)

## Environment Configuration

### Server (.env in server/)
```
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=gent_disagreement
OPENAI_API_KEY=your_openai_api_key
ALLOWED_ORIGINS=http://localhost:5173  # Optional, comma-separated
```

### Client (.env.local in client/)
```
VITE_API_URL=http://localhost:8000/api/chat
```

## API Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check
- `POST /api/chat` - AI SDK-compatible chat endpoint
  - Accepts: `{"messages": [{"role": "user", "text": "question"}]}`
  - Returns: Text stream compatible with AI SDK TextStreamChatTransport
  - Supports both "text" field and "parts" array format for message content

## Database Schema

The application uses PostgreSQL with pgvector extension. Key tables:
- Transcript segments with vector embeddings (384-dimensional)
- Episode metadata (title, date, speakers)
- Speaker information

Database operations use `psycopg2` with RealDictCursor for query results.

## Code Style & Patterns

### Python
- Type hints encouraged but not universally applied
- Classes use docstrings for documentation
- Error handling with try/except blocks and console logging
- Environment variables loaded via `python-dotenv`

### TypeScript/React
- Functional components with hooks
- TypeScript strict mode enabled
- Path alias `@/` maps to `client/src/`
- Tailwind CSS for styling
- Component composition with Radix UI primitives

## Important Notes

- The embedding model (`all-MiniLM-L6-v2`) generates 384-dimensional vectors - ensure pgvector column dimensions match
- Adaptive thresholding parameters (min_docs, max_docs, similarity_threshold) significantly impact retrieval quality
- The system is optimized for podcast transcript queries
- Speaker names are referenced as 'name' field in database (recent refactor from 'speaker' field)
