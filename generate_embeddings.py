import json
import time

import requests

from config import OPENROUTER_API_KEY


API_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL_CONFIG = {
    "qwen": "qwen/qwen3-embedding-8b"
}
BATCH_SIZE = 100


def generate_embeddings_with_model(model_name: str, texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for texts using the specified model.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Set OPENROUTER_API_KEY before calling this script.")

    if not texts:
        return []

    print(f"Generating embeddings for {len(texts)} texts in batches of {BATCH_SIZE}...")
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    all_embeddings = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)...")
        
        payload = {
            "model": MODEL_CONFIG[model_name],
            "input": batch_texts,
            "encoding_format": "float",
        }
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                
                # Check status before parsing JSON
                if resp.status_code != 200:
                    error_msg = f"API returned status {resp.status_code}"
                    # Try to get error message from response (skip HTML responses)
                    try:
                        if 'application/json' in resp.headers.get('Content-Type', ''):
                            error_body = resp.json()
                            error_msg += f": {error_body}"
                        else:
                            # For HTML responses (like 503 from Cloudflare), just note it
                            error_msg += " (Service temporarily unavailable)"
                    except:
                        pass
                    
                    # Retry for rate limits and service unavailable errors
                    if resp.status_code in [429, 503]:  # Rate limit or Service Unavailable
                        wait_time = retry_delay * (2 ** attempt)
                        error_type = "Rate limited" if resp.status_code == 429 else "Service unavailable"
                        print(f"{error_type} (status {resp.status_code}). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    
                    raise RuntimeError(error_msg)
                
                # Parse JSON response
                try:
                    body = resp.json()
                except json.JSONDecodeError as e:
                    # Log the actual response for debugging
                    response_preview = resp.text[:500]
                    raise RuntimeError(
                        f"Failed to parse JSON response. "
                        f"Response preview (first 500 chars): {response_preview}. "
                        f"JSON error: {str(e)}"
                    )
                
                # Extract embeddings
                batch_embeddings = [item["embedding"] for item in body.get("data", [])]
                
                if len(batch_embeddings) != len(batch_texts):
                    raise RuntimeError(
                        f"Mismatch: requested {len(batch_texts)} embeddings, "
                        f"but got {len(batch_embeddings)}"
                    )
                
                all_embeddings.extend(batch_embeddings)
                
                # Small delay between batches to avoid rate limiting
                if i + BATCH_SIZE < len(texts):
                    time.sleep(0.5)
                
                break  # Success, exit retry loop
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Request failed: {str(e)}. Retrying in {wait_time}s ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed after {max_retries} attempts: {str(e)}")
    
    print(f"Successfully generated {len(all_embeddings)} embeddings")
    return all_embeddings