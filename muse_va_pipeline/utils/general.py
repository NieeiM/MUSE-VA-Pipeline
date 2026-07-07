"""
General utility functions for the VA to Caption pipeline.
"""

import re


def extract_json_from_response(response_str: str) -> str:
    """Extract JSON string from LLM response.
    
    Handles cases where the response contains JSON wrapped in markdown code blocks
    or other text.
    
    Args:
        response_str: Raw response string from LLM
        
    Returns:
        Extracted JSON string
    """
    # Try to find JSON in markdown code blocks
    json_match = re.search(
        r'```(?:json)?\s*(\{.*?\})\s*```', response_str, re.DOTALL
    )
    if json_match:
        return json_match.group(1)

    # Try to find JSON object directly
    json_match = re.search(r'\{.*\}', response_str, re.DOTALL)
    if json_match:
        return json_match.group(0)

    return response_str.strip()
