# Resume Positioning

## One-Line Pitch

Built a multi-agent SEC EDGAR workflow that ingests 10-K/10-Q filings, retrieves comparable-company evidence, and generates schema-valid investment memos through FastAPI, LangGraph-ready orchestration, and OpenAI/Anthropic tool schemas.

## Resume Bullet Options

- Built a multi-agent finance workflow over SEC EDGAR using Python, FastAPI, LangGraph-ready orchestration, FAISS-compatible retrieval, and schema-enforced LLM outputs for investment memo generation.
- Engineered parser, retrieval, and synthesis agents to ingest 10-K/10-Q filings, extract XBRL financial facts, retrieve peer evidence, and emit validated JSON investment memos.
- Added reusable workflow templates for deal screening, credit memos, due diligence, earnings quality, covenant risk, and MD&A analysis with an evaluation harness for completion rate, latency, and schema validity.

## Interview Talking Points

- SEC API usage and fair-access constraints.
- Why parser, retrieval, and synthesis are separate agent roles.
- How schema validation prevents malformed LLM output.
- How offline deterministic synthesis makes demos reliable.
- How FAISS-compatible retrieval supports comparable-company analysis.
- How to extend the benchmark to 200+ filings and 81%+ completion.

## Future Upgrade Path

- Add real embedding models through OpenAI or local sentence transformers.
- Persist indexes in FAISS binary format for faster reloads.
- Add industry classification from SIC/NAICS metadata.
- Compare current filing language against prior-period filings.
- Add a Streamlit analyst cockpit for memo review and source inspection.
