# Exab-AI

FastAPI service for OCR (legal documents), Chat Completion API, and AI-powered Legal Tools for Saudi Arabian Law.

## Features

- **OCR Service**: Extract text from Saudi legal documents (images/PDFs) using Google Gemini AI
- **Chat API**: Unified chat API with models from OpenAI, Google Gemini, and Groq with streaming support
  - **Three AI Legal Assistants**:
    - **Sanad (سند)**: 40+ years expertise - Fast & efficient (Groq/Llama)
    - **Rashid (راشد)**: 20+ years expertise - Balanced performance (OpenAI GPT-4o)
    - **Nora (نورة)**: 5+ years expertise - Long context specialist (Gemini 2.0)
  - **Thinking Modes** for complex legal analysis:
    - **Rashid-Thinking**: Advanced reasoning with OpenAI o1
    - **Nora-Thinking**: Enhanced reasoning with Gemini Thinking
  - Specialized in Saudi Arabian law, Sharia, commercial law, labor law, and civil law
  - Bilingual support (Arabic & English) with professional legal eloquence
  - **Multi-Provider Support**: 
    - OpenAI models (GPT-4o, o1 reasoning models)
    - Google Gemini models (2.0 Flash, Thinking experimental)
    - Groq/Llama models (Llama 3.1 8B Instant - fast inference)
  - **Web Search**: Real-time web search integration using Perplexity API
    - Three modes: `disabled`, `fast`, `deep`

- **Legal Tools (10 AI-Powered Services)** - All specialized for KSA laws only:
  1. **Legal Memo Generator (صائغ المذكرات)**: Generate professional legal memos (12 types including defense, prosecution, appeal, objection, etc.)
  2. **Judgment Analysis (محلل الأحكام)**: Analyze court rulings with detailed/summary analysis
  3. **Lawsuit Petition Draft (صائغ اللوائح)**: Draft lawsuit petitions for various case types
  4. **Legal Article Explanation (شارح المواد)**: Explain legal articles at different complexity levels
  5. **Judgment Comparison (مقارنة الأحكام)**: Compare 2+ court judgments systematically (facts, reasoning, verdict, legal principles)
  6. **Legal Summary (لخص قانوني)**: Summarize legal documents (executive summary, key points, memo format)
  7. **Precedent Search (باحث السوابق)**: Search for relevant judicial precedents with similarity scoring
  8. **Case Evaluation (تقييم القضايا)**: Evaluate case strength with success probability analysis
  9. **Regulatory Updates (مستجدات الأنظمة)**: Track Saudi regulatory and legislative updates
  10. **Legal Grounds (الإسناد القانوني)**: Link facts to relevant legal provisions (Sharia, regulations, bylaws, precedents)

## Quick Start

### Prerequisites
- Python 3.11+
- Google Gemini API Key (for OCR and Gemini chat models)
- OpenAI API Key (for OpenAI chat models)
- Groq API Key (for fast Llama inference)
- Perplexity API Key (optional, for web search)
- Pinecone API Key (for RAG vector database)

### Installation

```bash
# Clone repository
git clone https://Exab-Azure@dev.azure.com/Exab-Azure/Exab-Ai/_git/Exab-Ai
cd Exab-Ai

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env
echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
echo "GROQ_API_KEY=your_groq_api_key_here" >> .env
echo "PERPLEXITY_API_KEY=your_perplexity_api_key_here" >> .env
echo "PINECONE_API_KEY=your_pinecone_api_key_here" >> .env
echo "PINECONE_ENVIRONMENT=us-east-1-aws" >> .env

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t exab-ai .
docker run -d -p 8000:8000 \
  -e GOOGLE_API_KEY="your_key" \
  -e OPENAI_API_KEY="your_key" \
  -e GROQ_API_KEY="your_key" \
  -e PERPLEXITY_API_KEY="your_key" \
  exab-ai
```

## API Endpoints

### OCR
- `POST /api/v1/ocr/image` - Extract text from image
- `POST /api/v1/ocr/pdf` - Extract text from PDF
- `GET /api/v1/ocr/health` - Health check

### Chat
- `GET /api/v1/chat/list_models` - List available models
- `POST /api/v1/chat/model_sse` - Chat completion (streaming)
- `GET /api/v1/chat/health` - Health check

### Legal Tools
- `GET /api/v1/legal-tools/types` - Get all legal tools types and options
- `POST /api/v1/legal-tools/legal-memo-generator` - Generate legal memos (SSE streaming)
- `POST /api/v1/legal-tools/judgment-analysis` - Analyze court judgments (SSE streaming)
- `POST /api/v1/legal-tools/lawsuit-petition-draft` - Draft lawsuit petitions (SSE streaming)
- `POST /api/v1/legal-tools/legal-article-explanation` - Explain legal articles (SSE streaming)
- `POST /api/v1/legal-tools/judgment-comparison` - Compare judgments (JSON response)
- `POST /api/v1/legal-tools/legal-summary` - Summarize legal documents (JSON response)
- `POST /api/v1/legal-tools/precedent-search` - Search judicial precedents (JSON response)
- `POST /api/v1/legal-tools/case-evaluation` - Evaluate case strength (JSON response)
- `POST /api/v1/legal-tools/regulatory-updates` - Get regulatory updates (JSON response)
- `POST /api/v1/legal-tools/legal-grounds` - Find legal grounds for facts (JSON response)

## Documentation

Interactive docs available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Gemini API key | For OCR and Gemini chat models |
| `OPENAI_API_KEY` | OpenAI API key | For OpenAI chat models |
| `GROQ_API_KEY` | Groq API key | For fast Llama inference |
| `PERPLEXITY_API_KEY` | Perplexity API key | No (for Web Search) |
| `HOST` | Server host | No (default: 0.0.0.0) |
| `PORT` | Server port | No (default: 8000) |
| `DEBUG` | Debug mode | No (default: True) |

**Note**: You need at least one of `GOOGLE_API_KEY`, `OPENAI_API_KEY`, or `GROQ_API_KEY` for the chat service to work.

## Web Search Modes

The chat API supports web search integration via Perplexity API with three modes:

- **`off`**: Web search is disabled (default)
- **`on`**: Always perform web search for every query
- **`if_needed`**: Automatically detect if web search is needed based on query content
  - Triggers on keywords like: current, latest, recent, today, news, etc.
  - Works with both Arabic and English queries

**Example Request (OpenAI):**
```json
{
  "model": "gpt-5",
  "messages": [{"role": "user", "content": "What are the latest news in Saudi Arabia?"}],
  "web_search_mode": "on",
  "stream": true,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0
}
```

**Example Request (Gemini):**
```json
{
  "model": "gemini-pro-latest",
  "messages": [{"role": "user", "content": "ما هي آخر الأخبار في المملكة العربية السعودية؟"}],
  "web_search_mode": "if_needed",
  "stream": true,
  "top_k": 40
}
```

## Available Models

### OpenAI Models (31 models)
- GPT-3.5 series: `gpt-3.5-turbo`, `gpt-3.5-turbo-1106`, `gpt-3.5-turbo-0125`
- GPT-4 series: `gpt-4`, `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`
- GPT-4.1 series: `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`
- GPT-5 series: `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-pro`, `gpt-5-codex`

### Google Gemini Models (25 models)
**Gemini 2.5 Series:**
- `gemini-2.5-pro`, `gemini-2.5-pro-preview-03-25`, `gemini-2.5-pro-preview-05-06`, `gemini-2.5-pro-preview-06-05`
- `gemini-2.5-flash`, `gemini-2.5-flash-preview-05-20`, `gemini-2.5-flash-preview-09-2025`
- `gemini-2.5-flash-lite`, `gemini-2.5-flash-lite-preview-06-17`, `gemini-2.5-flash-lite-preview-09-2025`

**Gemini 2.0 Series:**
- `gemini-2.0-flash`, `gemini-2.0-flash-001`, `gemini-2.0-flash-exp`
- `gemini-2.0-flash-lite`, `gemini-2.0-flash-lite-001`, `gemini-2.0-flash-lite-preview`, `gemini-2.0-flash-lite-preview-02-05`
- `gemini-2.0-flash-thinking-exp`, `gemini-2.0-flash-thinking-exp-01-21`, `gemini-2.0-flash-thinking-exp-1219`

**Other Models:**
- `gemini-flash-latest`, `gemini-flash-lite-latest`, `gemini-pro-latest` - Latest stable releases
- `gemini-robotics-er-1.5-preview` - Specialized robotics model
- `learnlm-2.0-flash-experimental` - Educational AI model

### Groq/Llama Models
- `llama-3.1-8b-instant` - Fast inference Llama 3.1 8B model (~560 tokens/second)

## Model-Specific Parameters

### OpenAI Models
- `frequency_penalty`: Reduce repetition (-2.0 to 2.0)
- `presence_penalty`: Encourage new topics (-2.0 to 2.0)

### Gemini Models
- `top_k`: Limit token sampling to top K options (1 to 100)
- All Gemini models support up to 1M input tokens

### Groq/Llama Models
- Uses standard `temperature` and `top_p` parameters
- Context length: 131K tokens
- Optimized for fast inference with Groq's infrastructure

## Legal Tools Details

All legal tools are specialized for **Saudi Arabian (KSA) law only**. If a user requests information about laws from other countries, the system will politely decline and indicate it only supports KSA laws.

### Available Tools

| # | Tool | Arabic Name | Endpoint | Response Type |
|---|------|-------------|----------|---------------|
| 1 | Legal Memo Generator | صائغ المذكرات | `/legal-memo-generator` | SSE Stream |
| 2 | Judgment Analysis | محلل الأحكام | `/judgment-analysis` | SSE Stream |
| 3 | Lawsuit Petition Draft | صائغ اللوائح | `/lawsuit-petition-draft` | SSE Stream |
| 4 | Legal Article Explanation | شارح المواد | `/legal-article-explanation` | SSE Stream |
| 5 | Judgment Comparison | مقارنة الأحكام | `/judgment-comparison` | JSON |
| 6 | Legal Summary | لخص قانوني | `/legal-summary` | JSON |
| 7 | Precedent Search | باحث السوابق | `/precedent-search` | JSON |
| 8 | Case Evaluation | تقييم القضايا | `/case-evaluation` | JSON |
| 9 | Regulatory Updates | مستجدات الأنظمة | `/regulatory-updates` | JSON |
| 10 | Legal Grounds | الإسناد القانوني | `/legal-grounds` | JSON |

### Memo Types (صائغ المذكرات)
- Defense Memo (مذكرة دفاع)
- Prosecution Memo (مذكرة ادعاء)
- Reply Memo (مذكرة جوابية)
- Appeal Memo (مذكرة استئنافية)
- Objection Memo (مذكرة اعتراضية)
- Cassation Memo (مذكرة نقض)
- Execution Memo (مذكرة تنفيذية)
- Precautionary Memo (مذكرة احترازية)
- Settlement Memo (مذكرة تسوية)
- Legal Opinion Memo (مذكرة رأي قانوني)
- Contract Commentary (مذكرة تعليق على عقد)
- Regulation Commentary (مذكرة تعليق على نظام)

### Case Types
- Commercial (تجارية)
- Labor (عمالية)
- Administrative (إدارية)
- General (عامة)

### Courts
- Ministry of Justice (وزارة العدل)
- Board of Grievances (ديوان المظالم)

### Legal Grounds Sources (Hierarchical Order)
1. Islamic Sharia (الشريعة الإسلامية) - Quran, Sunnah, Ijma, Qiyas
2. Regulations (الأنظمة)
3. Bylaws (اللوائح التنفيذية)
4. Regulatory Decisions (القرارات التنظيمية)
5. Judicial Precedents (السوابق القضائية)
6. Jurisprudential Principles (المبادئ الفقهية)
7. Ratified Agreements (الاتفاقيات المصادق عليها)
