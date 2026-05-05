import os
import json
from openai import OpenAI
from query import search_images

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def fallback_expand_query(user_query):
    user_query = user_query.strip()
    return [
        user_query,
        f"photo of {user_query}",
        f"image of {user_query}",
        f"scene containing {user_query}",
        f"object or animal matching {user_query}",
    ]


def llm_expand_query(user_query):
    if not os.getenv("OPENAI_API_KEY"):
        return fallback_expand_query(user_query)

    prompt = f"""
Return ONLY valid JSON.

User image search query:
"{user_query}"

Generate 5 short visual search queries for a CLIP/FAISS image search system.
They should be concrete, visual, and useful for finding matching images.

Format:
{{
  "queries": ["...", "...", "...", "...", "..."]
}}
"""

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )

        text = response.output_text.strip()
        data = json.loads(text)

        queries = data.get("queries", [])
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]

        if not queries:
            return fallback_expand_query(user_query)

        return queries[:5]

    except Exception as e:
        print("[AGENT] LLM expansion failed, using fallback:", e)
        return fallback_expand_query(user_query)


def agent_search(user_query, limit=9):
    expanded_queries = llm_expand_query(user_query)

    print("\n[AGENT] Expanded queries:")
    for q in expanded_queries:
        print(" -", q)

    all_results = []
    seen = set()

    for q in expanded_queries:
        results = search_images(q)

        for item in results:
            key = str(item)

            if key not in seen:
                seen.add(key)
                all_results.append(key)

            if len(all_results) >= limit:
                return all_results

    return all_results[:limit]