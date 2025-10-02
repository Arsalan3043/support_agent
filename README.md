# 🤖 Intelligent Support Agent with Agentic RAG

A production-ready, scalable customer support agent built with **LangGraph**, **ChromaDB**, and modern LLMs. Features modular architecture ready for multi-channel integration (Instagram, WhatsApp, etc.) and multi-agent expansion.

## 🌟 Features

- **🧠 Agentic RAG**: Intelligent agent that decides when to search knowledge base
- **🔄 ReAct Pattern**: Reasoning + Acting pattern for transparent decision-making
- **🎯 Anti-Hallucination**: Confidence scoring, source citation, and validation
- **💬 Human-like Conversation**: Intent classification, clarification handling, empathy
- **📚 ChromaDB Integration**: Free, efficient vector storage
- **🔌 Multi-Channel Ready**: Modular adapter architecture for easy integration
- **📊 Streamlit UI**: Beautiful chat interface for testing
- **🚀 REST API**: FastAPI backend ready for webhooks
- **📈 Scalable Architecture**: Easy to add more agents and channels

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   CHANNEL LAYER                          │
│  Web Adapter | Instagram (future) | WhatsApp (future)   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│              MESSAGE ROUTER / ORCHESTRATOR               │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                   AGENT LAYER (LangGraph)                │
│  Support Agent | Sales Agent (future) | Custom Agents   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│           TOOLS: RAG (ChromaDB) | CRM | Analytics       │
└─────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
support-agent-system/
├── agents/
│   └── support_agent.py      # Main LangGraph agent with ReAct
├── channels/
│   ├── base.py               # Base adapter interface
│   └── web_adapter.py        # Web/API adapter
├── tools/
│   └── rag_tool.py           # ChromaDB RAG tool
├── storage/
│   └── vector_store.py       # ChromaDB wrapper
├── core/
│   ├── config.py             # Configuration management
│   └── models.py             # Pydantic models
├── api/
│   └── main.py               # FastAPI application
├── ui/
│   └── streamlit_app.py      # Streamlit chat interface
├── utils/
│   └── load_knowledge.py     # Knowledge base loader
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo>
cd support-agent-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys
# For Anthropic (recommended):
ANTHROPIC_API_KEY=your_key_here
LLM_PROVIDER=anthropic

# For OpenAI:
OPENAI_API_KEY=your_key_here
LLM_PROVIDER=openai
```

### 3. Load Knowledge Base

```bash
# Load sample documents
python utils/load_knowledge.py --sample

# Or load from JSON file
python utils/load_knowledge.py --json path/to/documents.json

# Or load from directory of text files
python utils/load_knowledge.py --directory path/to/docs/
```

### 4. Run the Application

**Option A: Streamlit UI (Recommended for Testing)**

```bash
streamlit run ui/streamlit_app.py
```

Open http://localhost:8501 in your browser

**Option B: FastAPI Backend**

```bash
python api/main.py
```

API docs available at http://localhost:8000/docs

## 💡 Usage Examples

### Chat with the Agent

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "What's your return policy?",
        "user_id": "user123"
    }
)

print(response.json())
# {
#   "response": "Our return policy allows...",
#   "confidence": 0.95,
#   "sources": ["return_policy.pdf"],
#   "needs_clarification": false,
#   "session_id": "...",
#   "metadata": {...}
# }
```

### Add Documents to Knowledge Base

```python
import requests

requests.post(
    "http://localhost:8000/documents/add",
    json={
        "content": "Your documentation content here",
        "metadata": {
            "source": "filename.pdf",
            "category": "policy"
        }
    }
)
```

### Using the Agent Directly

```python
from agents.support_agent import support_agent

response = support_agent.process_message(
    user_message="How do I track my order?",
    session_id="session123"
)

print(response.content)
print(f"Confidence: {response.confidence}")
print(f"Sources: {response.sources}")
```

## 🔧 Configuration Options

Edit `.env` file:

```bash
# LLM Settings
LLM_PROVIDER=anthropic  # or "openai"
LLM_MODEL=claude-sonnet-4-5-20250929
LLM_TEMPERATURE=0.1

# RAG Settings
TOP_K_RESULTS=5
MIN_SIMILARITY_SCORE=0.5
CONFIDENCE_THRESHOLD=0.7

# Agent Settings
MAX_ITERATIONS=10
```

## 📊 Agent Workflow

The agent uses a LangGraph workflow with these nodes:

1. **Classify Intent**: Understand user's intent (question, complaint, greeting, etc.)
2. **Decide RAG**: Determine if knowledge base search is needed
3. **Retrieve Knowledge**: Search ChromaDB for relevant documents
4. **Generate Answer**: Create response using LLM + retrieved context
5. **Validate Answer**: Check quality and confidence
6. **Ask Clarification**: Request more info if needed

## 🔌 Adding New Channels

To add Instagram/WhatsApp/other channels:

1. Create adapter in `channels/`:

```python
# channels/instagram_adapter.py
from channels.base import ChannelAdapter

class InstagramAdapter(ChannelAdapter):
    def receive_message(self) -> StandardMessage:
        # Parse Instagram webhook
        pass
    
    def send_message(self, message: AgentResponse, recipient_id: str):
        # Send via Instagram API
        pass
```

2. Register webhook in `api/main.py`:

```python
@app.post("/webhook/instagram")
async def instagram_webhook(payload: Dict[str, Any]):
    adapter = InstagramAdapter()
    std_message = adapter.receive_message()
    response = support_agent.process_message(std_message.content)
    adapter.send_message(response, std_message.user_id)
```

3. Done! Agent logic remains unchanged ✅

## 🎯 Adding New Agents

To add specialized agents (sales, billing, etc.):

1. Create new agent file:

```python
# agents/sales_agent.py
class SalesAgent(SupportAgent):
    def __init__(self):
        super().__init__()
        # Custom tools, prompts, etc.
```

2. Register in agent registry
3. Route messages based on intent/keywords

## 📈 Production Considerations

### Scaling

- Use **Redis** for session management (set `REDIS_URL` in .env)
- Use **PostgreSQL** for conversation history (set `DATABASE_URL`)
- Deploy ChromaDB separately or use managed service
- Use message queue (RabbitMQ/Redis) for high-volume webhooks

### Monitoring

- Logs are saved to `logs/` directory
- Use LangSmith for tracing (built-in support)
- Monitor confidence scores and RAG hit rates
- Track escalation to human agents

### Security

- Add authentication to API endpoints
- Rate limiting on webhooks
- Validate webhook signatures
- Encrypt sensitive data in database

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Test individual components
python -c "from agents.support_agent import support_agent; \
           print(support_agent.process_message('Hello!'))"
```

## 🐛 Troubleshooting

**Issue: "No module named 'langchain'"**
- Solution: `pip install -r requirements.txt`

**Issue: ChromaDB connection error**
- Solution: Delete `data/chroma` directory and restart

**Issue: Low quality responses**
- Solution: Add more documents to knowledge base
- Adjust `CONFIDENCE_THRESHOLD` and `MIN_SIMILARITY_SCORE`

**Issue: Agent takes too long**
- Solution: Reduce `TOP_K_RESULTS`, use faster embedding model

## 📚 Documentation

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

MIT License - feel free to use for commercial projects!

## 🎉 Next Steps

1. ✅ Test the agent with your own knowledge base
2. ✅ Customize prompts in `agents/support_agent.py`
3. ✅ Add your specific tools (CRM, order lookup, etc.)
4. ✅ Deploy to production
5. ✅ Add Instagram/WhatsApp when APIs are available
6. ✅ Add specialized agents (sales, billing, technical)

## 💬 Support

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Built with ❤️ using LangGraph, ChromaDB, and Claude**