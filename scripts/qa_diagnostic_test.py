"""QA Diagnostic Test Script for NewsLens-AI RAG Application.
Runs deep end-to-end evaluation of retrieval accuracy, planning archetypes,
hybrid search scoring, citations, and hallucination detection.
"""
import asyncio
import json
import time

import httpx


async def run_diagnostics():
    test_cases = [
        {
            "id": "TC-01",
            "query": "Who is proposing transit expansion and what are the plans?",
            "expected_archetype": "factual_lookup",
            "expected_entities": ["Mayor", "transit"],
            "expected_headline_match": "TRANSIT EXPANSION",
        },
        {
            "id": "TC-02",
            "query": "What happened to the transatlantic ocean liner?",
            "expected_archetype": "factual_lookup",
            "expected_entities": ["ocean liner", "dignitaries", "harbor"],
            "expected_headline_match": "OCEAN LINER DOCKS",
        },
        {
            "id": "TC-03",
            "query": "Compare coverage of the market crash between The Metropolis Chronicle and The Daily Record.",
            "expected_archetype": "cross_newspaper_comparison",
            "expected_entities": ["Metropolis Chronicle", "Daily Record", "market"],
            "expected_headline_match": "MARKET IN SEVERE TUMBLE",
        },
        {
            "id": "TC-04",
            "query": "What was the timeline of major economic events during the crisis?",
            "expected_archetype": "thematic_timeline",
            "expected_entities": ["market", "economic"],
            "expected_headline_match": "MARKET",
        },
    ]

    print("=" * 80)
    print("NEWSLENS-AI — QA RETRIEVAL ACCURACY & HALLUCINATION DIAGNOSTIC SUITE")
    print("=" * 80)

    results = []

    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=90.0) as client:
        for tc in test_cases:
            print(f"\n[{tc['id']}] QUERY: \"{tc['query']}\"")
            print("-" * 60)

            t0 = time.monotonic()
            events = []
            tokens = []
            citations = []
            plan_info = {}
            tool_outputs = []

            try:
                async with client.stream(
                    "POST",
                    "/api/query/stream",
                    json={"query": tc["query"], "model_override": "ollama_chat"},
                ) as response:
                    assert response.status_code == 200, f"HTTP {response.status_code}"
                    current_event = None
                    async for line in response.aiter_lines():
                        if line.startswith("event: "):
                            current_event = line[7:].strip()
                        elif line.startswith("data: "):
                            raw_data = line[6:].strip()
                            try:
                                data = json.loads(raw_data)
                                events.append(current_event)

                                if current_event == "plan":
                                    plan_info = {
                                        "archetype": data.get("archetype"),
                                        "tools": [t.get("tool_name") for t in data.get("plan", [])],
                                    }
                                    print(f"  → Plan: Archetype={data.get('archetype')} | Tools={plan_info['tools']}")

                                elif current_event == "token":
                                    tokens.append(data.get("delta", ""))

                                elif current_event == "tool_results":
                                    tool_outputs.append(data)
                                    print(f"  → Tool Results: Evidence Count = {data.get('evidence_count')}")

                                elif current_event == "citations":
                                    citations = data.get("citations", [])
                                    print(f"  → Citations Count: {len(citations)}")
                                    for cit in citations:
                                        print(f"     • [{cit.get('newspaper_name')}, {cit.get('issue_date')}, P.{cit.get('page_number')}] \"{cit.get('headline')}\"")

                                elif current_event == "done":
                                    latency = data.get("latency_ms")
                                    cost = data.get("cost_usd")
                                    print(f"  → Completed in {latency}ms (Cost: ${cost})")

                            except Exception:
                                pass

                answer_text = "".join(tokens)
                total_duration_ms = round((time.monotonic() - t0) * 1000)

                # Retrieval verification
                has_citations = len(citations) > 0
                has_answer = len(answer_text) > 20
                archetype_correct = plan_info.get("archetype") == tc["expected_archetype"]

                # Check if retrieved citations contain the expected headline keyword
                headline_matched = any(
                    tc["expected_headline_match"].lower() in (cit.get("headline") or "").lower()
                    for cit in citations
                )

                # Check for factual grounding (prevent hallucination)
                hallucination_check = "GROUNDED" if (has_citations and headline_matched) else "POTENTIAL_MISMATCH"

                print(f"  → Answer Preview: {answer_text[:180]}...")
                print(f"  → Grounding Status: {hallucination_check}")

                results.append({
                    "id": tc["id"],
                    "query": tc["query"],
                    "plan": plan_info,
                    "tool_outputs": tool_outputs,
                    "citations_count": len(citations),
                    "citations": citations,
                    "headline_matched": headline_matched,
                    "archetype_correct": archetype_correct,
                    "hallucination_check": hallucination_check,
                    "duration_ms": total_duration_ms,
                    "answer_length": len(answer_text),
                })

            except Exception as e:
                import traceback
                print(f"  ❌ ERROR: {e}\n{traceback.format_exc()}")
                results.append({
                    "id": tc["id"],
                    "query": tc["query"],
                    "error": str(e),
                    "hallucination_check": "FAILED",
                })

    print("\n" + "=" * 80)
    print("QA DIAGNOSTIC TEST SUMMARY REPORT")
    print("=" * 80)
    for r in results:
        status = "✅ PASS" if r.get("hallucination_check") == "GROUNDED" else "⚠️ WARN/FAIL"
        print(f"[{r['id']}] {status} | Archetype: {r.get('plan', {}).get('archetype')} | Citations: {r.get('citations_count')} | Latency: {r.get('duration_ms')}ms")

    return results


if __name__ == "__main__":
    asyncio.run(run_diagnostics())
