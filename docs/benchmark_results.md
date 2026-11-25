| Configuration | Model | Exec. accuracy | Easy | Medium | Hard | Extra | Multi-join (3+ tables) | Wrong answers returned | Refused unanswerable | Drafts with hallucinated identifiers |
|---|---|---|---|---|---|---|---|---|---|---|
| Semantic layer only | none | **32.9%** | 28.6% | 26.3% | 60.9% | 0.0% | 48.3% | 0.0% | 100.0% | n/a |
| LLM, no guardrails | qwen2.5-coder:3b | **64.5%** | 95.2% | 79.0% | 52.2% | 15.4% | 44.8% | 25.0% | 100.0% | 12.8% |
| LLM + guardrails | qwen2.5-coder:3b | **65.8%** | 95.2% | 79.0% | 56.5% | 15.4% | 48.3% | 23.7% | 100.0% | 12.8% |
| Router + LLM + guardrails | qwen2.5-coder:3b | **77.6%** | 95.2% | 94.7% | 82.6% | 15.4% | 69.0% | 18.4% | 100.0% | 9.4% |
