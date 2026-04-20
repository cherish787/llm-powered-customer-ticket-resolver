ZERO_SHOT_PROMPT = """
You are a customer support agent. Analyze the following query and provide:
1. Intent
2. Summary
3. Suggested Response

Query: {query}

Format your response as structured JSON with keys: "intent", "summary", "response".
"""

FEW_SHOT_PROMPT = """
You are a customer support agent. Analyze the query using the following examples as a guide.

Example 1:
Query: How do I reset my account password?
Response: {{"intent": "Password Reset", "summary": "User needs help resetting password.", "response": "Go to Settings > Security > Reset Password and follow the email instructions."}}

Example 2:
Query: I haven't received my order yet. It's been 5 days.
Response: {{"intent": "Order Tracking", "summary": "User is inquiring about a late delivery.", "response": "Check your email for the tracking number or visit the 'My Orders' section in your profile."}}

Current Query: {query}
Similar Past Tickets:
{retrieved_context}

Format your response as structured JSON with keys: "intent", "summary", "response".
"""

def get_zero_shot_prompt(query: str) -> str:
    return ZERO_SHOT_PROMPT.format(query=query)

def get_few_shot_prompt(query: str, context: str) -> str:
    return FEW_SHOT_PROMPT.format(query=query, retrieved_context=context)
