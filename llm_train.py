import json
import os
import time
import random
import pandas as pd
from typing import Dict, List, Tuple
import requests
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import re
# import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


class OpinionPredictor:
    """LLM-based opinion predictor with conversation-aware analysis."""
    
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3.1-8B-Instruct-Turbo", api_provider: str = "together"):
        self.api_key = api_key
        self.model = model
        self.api_provider = api_provider.lower()
        # Set base URL and delay based on provider
        if self.api_provider == "together":
            self.base_url = "https://api.together.xyz/v1"
            self.delay = 8  # Together AI has good rate limits
        elif self.api_provider == "gemini":
            genai.configure(api_key=self.api_key)
            self.gemini_model = genai.GenerativeModel(self.model)
            self.delay = 1.0
        elif self.api_provider == "openai":
            self.base_url = "https://api.openai.com/v1"
            self.delay = 0.5
        elif self.api_provider == "openrouter":
            self.base_url = "https://openrouter.ai/api/v1"
            self.delay = 0.5
        else:  # groq
            self.base_url = "https://api.groq.com/openai/v1"
            self.delay = 2.0 if "70b" in model else 1.0
        
        self.total_tokens = 0
        
        # Create organized folder structure
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.experiment_folder = f"llm_experiment_{timestamp}"
        os.makedirs(self.experiment_folder, exist_ok=True)
        
        # Create separate files
        self.results_file = os.path.join(self.experiment_folder, "results.txt")
        self.responses_file = os.path.join(self.experiment_folder, "llm_responses.txt")
        self.summary_file = os.path.join(self.experiment_folder, "experiment_summary.txt")
        
        # Initialize files with headers
        self._initialize_files()
        
        print(f"🚀 Initialized {model}")
        print(f"📁 Experiment folder: {self.experiment_folder}")

    def _initialize_files(self):
        """Initialize all log files with proper headers."""
        
        # Initialize results file
        with open(self.results_file, "w", encoding='utf-8') as f:
            f.write("EXPERIMENT RESULTS LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
        
        # Initialize LLM responses file
        with open(self.responses_file, "w", encoding='utf-8') as f:
            f.write("LLM RESPONSES LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("This file contains all prompts sent to LLM and their responses.\n")
            f.write("=" * 80 + "\n\n")
        
        # Initialize summary file
        with open(self.summary_file, "w", encoding='utf-8') as f:
            f.write("EXPERIMENT SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def log_result(self, message: str):
        """Log to results file."""
        with open(self.results_file, "a", encoding='utf-8') as f:
            f.write(message)
        print(message.strip())

    def log_response(self, message: str):
        """Log to LLM responses file."""
        with open(self.responses_file, "a", encoding='utf-8') as f:
            f.write(message)

    def log_summary(self, message: str):
        """Log to summary file."""
        with open(self.summary_file, "a", encoding='utf-8') as f:
            f.write(message)

    def create_prompt(self, dataset: str, posts: List[str], comments: List[str], 
                     conversations: List[Dict], categories: List[str]) -> str:
        """Create analysis prompt with conversation-aware content."""
        
        # Format posts
        posts_text = "No posts available"
        if posts:
            posts_text = ""
            for i, post in enumerate(posts[:4]):
                short_post = post[:2000] + "..." if len(post) > 2000 else post
                posts_text += f"\nPost {i+1}: {short_post}\n" + "-"*40
        
        # Format comments on posts
        comments_text = "No comments available"
        if comments:
            comments_text = ""
            for i, comment in enumerate(comments[:6]):
                short_comment = comment[:3000] + "..." if len(comment) > 3000 else comment
                comments_text += f"\nComment {i+1}: {short_comment}\n" + "-"*30
        
        # Format conversation exchanges (NEW: with parent content)
        conversations_text = ""
        if conversations:
            conversations_text = "\nCONVERSATION EXCHANGES:\n"
            
            for i, conv in enumerate(conversations[:8]):  # Show more conversations
                target_opinion = conv.get('target_opinion', 'Unknown')
                user_reply = conv.get('user_reply', '')
                parent_content = conv.get('parent_content', '')
                interaction_type = conv.get('interaction_type', 'unknown')
                
                # Create conversation thread visualization
                conversations_text += f"\nConversation {i+1}:\n"
                
                if parent_content:
                    # Show the original comment/context they're replying to
                    short_parent = parent_content[:800] + "..." if len(parent_content) > 800 else parent_content
                    conversations_text += f"  Original comment by {target_opinion} user:\n"
                    conversations_text += f"  → \"{short_parent}\"\n"
                    conversations_text += f"  \n"
                
                if user_reply:
                    # Show their reply
                    short_reply = user_reply[:1000] + "..." if len(user_reply) > 1000 else user_reply
                    conversations_text += f"  User's reply:\n"
                    conversations_text += f"  → \"{short_reply}\"\n"
                    conversations_text += f"  (Replying to: {target_opinion} user, Type: {interaction_type})\n"
                
                conversations_text += "-" * 50 + "\n"
            
            # Add interaction summary
            opinion_counts = {}
            interaction_types = {}
            for conv in conversations:
                opinion = conv.get('target_opinion', 'Unknown')
                interaction = conv.get('interaction_type', 'unknown')
                opinion_counts[opinion] = opinion_counts.get(opinion, 0) + 1
                interaction_types[interaction] = interaction_types.get(interaction, 0) + 1
            
            conversations_text += "\nINTERACTION SUMMARY:\n"
            for opinion, count in sorted(opinion_counts.items(), key=lambda x: x[1], reverse=True):
                conversations_text += f"• {count} conversations with {opinion} users\n"
            
            for interaction, count in sorted(interaction_types.items(), key=lambda x: x[1], reverse=True):
                conversations_text += f"• {count} interactions\n"
        
        # Create categories with hints
        categories_text = ""
        for category in categories:
            hint = ""
            if dataset == "religion":
                if category == "Christian":
                    hint = " (includes Catholic, Protestant, Orthodox, etc.)"
                elif category == "Islamic":
                    hint = " (includes Muslim, Sunni, Shia, etc.)"
                elif category == "philosophical/other":
                    hint = " (includes spiritual, deist, humanist, etc.)"
            elif dataset == "capitalism":
                if category == "communism/socialism":
                    hint = " (includes Marxist, leftist, socialist)"
                elif category == "libertarianism":
                    hint = " (includes minimal government, free market)"
            elif dataset == "abortion":
                if category == "Pro-Life":
                    hint = " (opposes abortion, supports fetal rights)"
                elif category == "Pro-Choice":
                    hint = " (supports reproductive choice, women's rights)"
                elif category == "Mixed View":
                    hint = " (nuanced position, contextual support)"
            
            categories_text += f"• {category}{hint}\n"
        
        # Build enhanced prompt
        prompt = f"""ACADEMIC RESEARCH ANALYSIS: This is an objective analysis of social media content for academic research on opinion classification. The goal is to categorize user perspectives based on their communication patterns, not to promote any particular viewpoint.
You are an expert at analyzing social media conversations to understand people's beliefs and opinions.
TASK: Analyze this user's posts, comments, and conversation exchanges to determine their stance on {dataset}.

USER'S POSTS:
{posts_text}

USER'S COMMENTS ON POSTS:
{comments_text}

{conversations_text}

ANALYSIS STEPS:
1. What beliefs do they express in their own posts?
2. What do their comments on posts reveal about their views?
3. How do they engage in conversations - what do they reply to and how?
4. Do they engage more with people who agree or disagree with them?
5. What patterns emerge from the full conversation context?
6. Which category best matches their overall worldview?

AVAILABLE CATEGORIES:
{categories_text}

Pay special attention to:
- The content of what they're replying to (shows what triggers their responses)
- How they respond to different viewpoints 
- Whether they challenge, support, or provide nuanced takes
- The tone and approach in their conversation exchanges

Think step by step about the evidence, then select the single best category.

REASONING:
1. Key beliefs from posts: 
2. Patterns from post comments: 
3. Conversation engagement patterns: 
4. Response style and tone: 
5. Best category match: 

IMPORTANT: End your response with exactly one line containing only your final answer.
FINAL ANSWER: [Select exactly one category from the available categories' list above]"""

        return prompt

    def call_api(self, prompt: str) -> Tuple[str, int]:
        """Call Gemini, OpenAI, or Groq API with error handling."""
        for attempt in range(3):
            try:
                if self.api_provider == "gemini":
                    # Gemini API call
                    generation_config = genai.types.GenerationConfig(
                        max_output_tokens=500,
                        temperature=0.0,
                        top_p=1.0
                    )
                    
                    # Create the full prompt with system instruction
                    full_prompt = """You are a precise analyst who analyzes conversation patterns and content. Follow the analysis steps and select exactly one category.""" + prompt
                    
                    response = self.gemini_model.generate_content(
                        full_prompt,
                        generation_config=generation_config
                    )
                    
                    prediction = response.text.strip()
                    
                    # Estimate tokens (Gemini doesn't always return token count)
                    tokens = len(prompt.split()) + len(prediction.split())
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        tokens = getattr(response.usage_metadata, 'total_token_count', tokens)
                    
                    self.total_tokens += tokens
                    time.sleep(self.delay)
                    
                    return prediction, tokens
                    
                else:
                    # OpenAI/Groq API call (existing code)
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    
                    if self.api_provider == "together":
                        max_tokens = 400  # Conservative for cost control
                    elif self.api_provider == "openai":
                        max_tokens = 500
                    else:
                        max_tokens = 400
                    
                    data = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a neutral and precise analyst who analyzes conversation patterns and content. Follow the analysis steps and select exactly one category."},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                        "top_p": 1.0
                    }

                    response = requests.post(f"{self.base_url}/chat/completions", 
                                           headers=headers, json=data, timeout=60)
                    response.raise_for_status()
                    
                    result = response.json()
                    prediction = result["choices"][0]["message"]["content"].strip()
                    tokens = result.get("usage", {}).get("total_tokens", 100)
                    
                    self.total_tokens += tokens
                    time.sleep(self.delay)
                    
                    return prediction, tokens
                
            except Exception as e:
                error_msg = f"API attempt {attempt + 1} failed: {str(e)}\n"
                self.log_response(error_msg)
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return "ERROR", 0

    def extract_answer(self, response: str, categories: List[str], user_id: str) -> Tuple[str, str]:
        """Extract and log the LLM response with organized logging."""
        
        # Log full LLM response to responses file
        response_entry = f"\n{'='*80}\n"
        response_entry += f"🤖 LLM RESPONSE for {user_id}:\n"
        response_entry += f"{'='*80}\n"
        response_entry += f"{response}\n"
        response_entry += f"{'='*80}\n\n"
        
        self.log_response(response_entry)
        
        if response == "ERROR":
            error_msg = f"❌ API Error for {user_id}\n"
            self.log_result(error_msg)
            return categories[0], "api_error"
        
        extraction_log = f"🔍 EXTRACTION for {user_id}:\n"
        
        # Step 1: Get all non-empty lines
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        if not lines:
            extraction_log += f"❌ No lines found, using fallback\n"
            self.log_response(extraction_log)
            return categories[0], "no_match"
        
        # Step 2: Try to find explicit "FINAL ANSWER:" pattern first
        for line in reversed(lines):
            final_answer_match = re.search(r"FINAL ANSWER:\s*(.+)", line, re.IGNORECASE)
            if final_answer_match:
                answer = final_answer_match.group(1).strip()
                extraction_log += f"🎯 Found FINAL ANSWER: '{answer}'\n"
                break
        else:
            # Step 3: Work backwards through lines to find the first one that matches a category
            answer = None
            for i, line in enumerate(reversed(lines)):
                # Clean the line
                clean_line = line.strip('"\'•-*[]()').strip()
                clean_line = re.sub(r'^(The user is|Answer:|Category:|Based on|This suggests|While|However)\s*', '', clean_line, flags=re.IGNORECASE)
                
                # Check if this line contains a category
                for category in categories:
                    if (category.lower() == clean_line.lower() or 
                        category.lower() in clean_line.lower() or 
                        clean_line.lower() in category.lower()):
                        answer = clean_line
                        extraction_log += f"🎯 Found category in line {i+1} from end: '{answer}'\n"
                        break
                
                if answer:
                    break
                    
                # If we've checked the last 5 lines and found nothing, just use the last line
                if i >= 4:
                    answer = lines[-1].strip('"\'•-*[]()').strip()
                    extraction_log += f"🔄 Using last line after checking 5 lines: '{answer}'\n"
                    break
            
            # Final fallback: use the very last line
            if not answer:
                answer = lines[-1].strip('"\'•-*[]()').strip()
                extraction_log += f"🔄 Final fallback - last line: '{answer}'\n"
        
        # Clean the answer further
        answer = re.sub(r'^(The user is|Answer:|Category:|Based on|This suggests|While|However)\s*', '', answer, flags=re.IGNORECASE)
        answer = answer.strip('"\'•-*[]()').strip()
        
        # Match to categories
        match_log = ""
        
        # 1. Exact match
        for category in categories:
            if category.lower() == answer.lower():
                match_log += f"✅ Exact match: '{answer}' → '{category}'\n"
                self.log_response(extraction_log + match_log + "\n")
                return category, "exact"
        
        # 2. Partial match
        for category in categories:
            if answer.lower() in category.lower() or category.lower() in answer.lower():
                match_log += f"🔧 Partial match: '{answer}' → '{category}'\n"
                self.log_response(extraction_log + match_log + "\n")
                return category, "partial"
        
        # 3. Enhanced semantic mapping
        mappings = {
            # Political
            "trump supporter": ["trump supporter", "pro trump", "maga", "republican", "conservative", 
                              "support trump", "sympathetic to trump", "trump"],
            "non-trump supporter": ["non-trump supporter", "anti trump", "democrat", "liberal", 
                                  "against trump", "oppose trump"],
            
            # Religious
            "christian": ["christian", "catholic", "protestant", "orthodox", "evangelical", "baptist"],
            "islamic": ["islamic", "muslim", "sunni", "shia", "islam"],
            "jewish": ["jewish", "judaism", "jew"],
            "buddhist": ["buddhist", "buddhism", "buddha"],
            "hindu": ["hindu", "hinduism"],
            "non-theistic": ["non-theistic", "atheist", "agnostic", "secular", "non-religious"],
            "philosophical/other": ["philosophical", "spiritual", "deist", "humanist", "other"],
            
            # Abortion
            "pro-life": ["pro-life", "pro life", "right to life", "anti abortion", "prolife"],
            "pro-choice": ["pro-choice", "pro choice", "reproductive rights", "women's choice", "prochoice"],
            "mixed view": ["mixed view", "mixed", "moderate", "both sides", "nuanced"],
            
            # Economic
            "communism/socialism": ["communism", "socialism", "communist", "socialist", "marxist", "leftist"],
            "capitalism": ["capitalism", "capitalist", "free market", "business"],
            "libertarianism": ["libertarianism", "libertarian", "minimal government"],
            
            # LGBTQ
            "transgender": ["transgender", "trans", "ftm", "mtf"],
            "gay": ["gay", "homosexual", "gay man"],
            "lesbian": ["lesbian", "gay woman"],
            "bisexual": ["bisexual", "bi", "pansexual"],
            "asexual": ["asexual", "ace", "asexuality"],
            "nonbinary": ["nonbinary", "non-binary", "nb", "genderqueer"]
        }
        
        answer_lower = answer.lower()
        for canonical, variants in mappings.items():
            # Check if this canonical category exists in our categories
            matching_category = None
            for cat in categories:
                if cat.lower() == canonical:
                    matching_category = cat
                    break
            
            if matching_category:
                for variant in variants:
                    if variant in answer_lower:
                        match_log += f"🎯 Semantic match: '{answer}' → '{matching_category}' (via '{variant}')\n"
                        self.log_response(extraction_log + match_log + "\n")
                        return matching_category, "semantic"
        
        # 4. No match found
        match_log += f"❌ NO MATCH: '{answer}' not found in {categories}\n"
        match_log += f"Using fallback: {categories[0]}\n"
        self.log_response(extraction_log + match_log + "\n")
        return categories[0], "no_match"


def load_data(path: str, dataset: str) -> Tuple[Dict, List[str]]:
    """Load dataset with enhanced conversation context support from ORIGINAL JSON files."""
    
    # Use original JSON file with correct naming pattern
    original_file = os.path.join(path, f"graph_data_{dataset}.json")
    
    print(f"📖 Loading {dataset} from original file: {original_file}")
    
    if not os.path.exists(original_file):
        raise FileNotFoundError(f"Original dataset file not found: {original_file}")
    
    with open(original_file, "r", encoding='utf-8') as f:
        original_data = json.load(f)
    
    def process_data(data, split):
        users = {}
        posts = {}
        
        # Process nodes
        for node in data["nodes"]:
            if node["type"] == "user":
                users[node["id"]] = {
                    "label": node["label"],
                    "posts": [],
                    "comments": [],
                    "conversations": []  # New: conversation exchanges
                }
            elif node["type"] == "post":
                posts[node["id"]] = {
                    "title": str(node.get("title", "")).strip(),
                    "content": str(node.get("content", "")).strip(),
                    "author": None
                }
        
        # Process edges
        for edge in data["edges"]:
            try:
                if edge["type"] == "user_publish_post":
                    user_id = edge["source"]
                    post_id = edge["target"]
                    if user_id in users and post_id in posts:
                        posts[post_id]["author"] = user_id
                        
                        title = posts[post_id]["title"]
                        content = posts[post_id]["content"]
                        
                        if title and content:
                            if title.lower() not in content.lower():
                                post_text = f"Title: {title}\nContent: {content}"
                            else:
                                post_text = content
                        elif title:
                            post_text = f"Title: {title}"
                        elif content:
                            post_text = content
                        else:
                            continue
                        
                        users[user_id]["posts"].append(post_text)
                
                elif edge["type"] == "user_comment_post":
                    user_id = edge["source"]
                    post_id = edge["target"]
                    
                    if user_id in users and post_id in posts and "content" in edge:
                        comment_text = str(edge["content"]).strip()
                        if len(comment_text) > 5:
                            # Get post context
                            post_title = posts[post_id]["title"]
                            post_content = posts[post_id]["content"]
                            
                            # Create post context
                            if post_title and post_content:
                                if post_title.lower() not in post_content.lower():
                                    post_context = f"Title: {post_title}\nContent: {post_content[:200]}..."
                                else:
                                    post_context = post_content[:200] + "..."
                            elif post_title:
                                post_context = f"Title: {post_title}"
                            elif post_content:
                                post_context = post_content[:200] + "..."
                            else:
                                post_context = "[Empty post]"
                            
                            # Combine post context with user's comment
                            full_comment = f"Post context: ({post_context}) → User's comment: {comment_text}"
                            users[user_id]["comments"].append(full_comment)
                
                elif edge["type"] == "user_comment_user":
                    user_a_id = edge["source"]  # User A (who made original comment)
                    user_b_id = edge["target"]  # User B (who replied) - THIS IS WHO WE'RE PREDICTING
                    if user_a_id in users and user_b_id in users:
        
                        # Extract conversation components with corrected field mapping
                        # - "content" is User A's (source) original comment on post
                        # - "reply_content" is User B's (target) reply to User A
                        user_a_original = str(edge.get("content", "")).strip()      # User A's original comment
                        user_b_reply = str(edge.get("reply_content", "")).strip()   # User B's reply to User A
                        interaction_type = edge.get("interaction_type", "unknown")
        
                        # Create conversation exchange record for User B (who we're predicting)
                        # This shows what User B replied to and how they replied
                        conversation = {
                            "target_user": user_a_id,  # User A (who User B replied to)
                            "target_opinion": users[user_a_id]["label"],  # User A's opinion
                            "user_reply": user_b_reply,      # What User B replied
                            "parent_content": user_a_original  # What User A originally said (context)
                        }
        
                        # Add this conversation to User B's profile (who we want to predict)
                        users[user_b_id]["conversations"].append(conversation)
                        
            except Exception as e:
                print(f"Warning: Error processing edge: {e}")
                continue
        
        return users, posts
    
    # Process the complete original data
    all_users, all_posts = process_data(original_data, "complete")
    
    # ADDED: Filter users based on label frequency thresholds
    from collections import Counter
    
    # Set threshold based on dataset
    if dataset.lower() == "trump":
        min_threshold = 0.05  # 5% for trump dataset
    else:
        min_threshold = 0.01  # 1% for all other datasets
    
    # Count label frequencies
    label_counts = Counter()
    for user in all_users.values():
        if user["label"]:
            label_counts[user["label"]] += 1
    
    total_users = len(all_users)
    
    # Determine valid labels based on threshold
    valid_labels = []
    ignored_labels = []
    
    for label, count in label_counts.items():
        percentage = count / total_users
        if percentage >= min_threshold:
            valid_labels.append(label)
        else:
            ignored_labels.append((label, count, percentage * 100))
    
    # Report filtering results
    print(f"\n📊 Label frequency filtering (threshold: {min_threshold*100}%):")
    print(f"Total users before filtering: {total_users}")
    
    if ignored_labels:
        print(f"\n❌ Ignoring {len(ignored_labels)} rare labels:")
        for label, count, percentage in ignored_labels:
            print(f"  - '{label}': {count} users ({percentage:.2f}%)")
    
    print(f"\n✅ Valid labels ({len(valid_labels)}):")
    for label in sorted(valid_labels):
        count = label_counts[label]
        percentage = count / total_users * 100
        print(f"  - '{label}': {count} users ({percentage:.2f}%)")
    
    # Filter users to only include those with valid labels
    filtered_users = {}
    for user_id, user_data in all_users.items():
        if user_data["label"] in valid_labels:
            filtered_users[user_id] = user_data
    
    print(f"\n🎯 Users for LLM prediction: {len(filtered_users)}/{total_users} ({len(filtered_users)/total_users*100:.1f}%)")
    
    # Get categories from filtered data
    categories = sorted(list(valid_labels))
    
    print(f"📊 Final categories: {categories}")
    
    return {
        "train": {},  # Empty - LLM doesn't need training data
        "test": filtered_users  # Only users with valid labels for testing
    }, categories


def run_experiment(dataset: str, api_key: str, model: str, api_provider: str = "gemini", max_samples: int = 64):
    """Run experiment with conversation-aware analysis using original complete data."""
    
    print(f"\n🚀 Running {model} on {dataset} using {api_provider.upper()}")
    
    # Load data from original complete dataset with correct path
    data, categories = load_data("data", dataset)
    
    # Initialize predictor
    predictor = OpinionPredictor(api_key, model, api_provider)
    
    # Log experiment start
    experiment_header = f"\n{'#'*80}\n"
    experiment_header += f"CONVERSATION-AWARE EXPERIMENT: {model} on {dataset.upper()}\n"
    experiment_header += f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    experiment_header += f"Categories: {categories}\n"
    experiment_header += f"{'#'*80}\n\n"
    
    predictor.log_result(experiment_header)
    predictor.log_response(experiment_header)
    predictor.log_summary(experiment_header)
    
    # Sample test users
    test_users = data["test"]
    if len(test_users) > max_samples:
        user_ids = random.sample(list(test_users.keys()), max_samples)
        test_users = {uid: test_users[uid] for uid in user_ids}
    
    print(f"🎯 Testing {len(test_users)} users")
    
    # Track results
    predictions = []
    true_labels = []
    match_types = {"exact": 0, "partial": 0, "semantic": 0, "no_match": 0, "api_error": 0}
    correct = 0
    detailed_results = []
    
    # Run predictions
    for user_id, user_data in tqdm(test_users.items(), desc="Predicting"):
        try:
            # Log user info
            user_header = f"\n{'='*60}\n"
            user_header += f"USER: {user_id}\n"
            user_header += f"TRUE LABEL: {user_data['label']}\n"
            user_header += f"POSTS: {len(user_data['posts'])}\n"
            user_header += f"COMMENTS: {len(user_data['comments'])}\n"
            user_header += f"CONVERSATIONS: {len(user_data['conversations'])}\n"
            
            # Add conversation breakdown
            if user_data['conversations']:
                conv_stats = {}
                for conv in user_data['conversations']:
                    opinion = conv.get('target_opinion', 'Unknown')
                    conv_stats[opinion] = conv_stats.get(opinion, 0) + 1
                
                user_header += f"CONVERSATION BREAKDOWN: "
                for opinion, count in conv_stats.items():
                    user_header += f"{opinion}({count}) "
                user_header += "\n"
            
            user_header += f"{'='*60}\n"
            predictor.log_result(user_header)
            
            # Create conversation-aware prompt
            prompt = predictor.create_prompt(
                dataset=dataset,
                posts=user_data["posts"],
                comments=user_data["comments"],
                conversations=user_data["conversations"],
                categories=categories
            )
            
            # Log prompt
            prompt_log = f"\nPROMPT SENT TO LLM for {user_id}:\n"
            prompt_log += f"{'-'*40}\n"
            prompt_log += f"{prompt}\n"
            prompt_log += f"{'-'*40}\n\n"
            predictor.log_response(prompt_log)
            
            # Get LLM response
            response, tokens = predictor.call_api(prompt)
            
            # Extract answer
            prediction, match_type = predictor.extract_answer(response, categories, user_id)
            
            # Record results
            predictions.append(prediction)
            true_labels.append(user_data["label"])
            match_types[match_type] += 1
            
            is_correct = prediction == user_data["label"]
            if is_correct:
                correct += 1
            
            # Store detailed result
            detailed_results.append({
                "user_id": user_id,
                "true_label": user_data["label"],
                "prediction": prediction,
                "correct": is_correct,
                "match_type": match_type,
                "tokens": tokens,
                "num_posts": len(user_data["posts"]),
                "num_comments": len(user_data["comments"]),
                "num_conversations": len(user_data["conversations"])
            })
            
            # Log result
            result_emoji = "✅ CORRECT" if is_correct else "❌ WRONG"
            result_log = f"PREDICTION: {prediction} | TRUE: {user_data['label']} | {result_emoji}\n"
            result_log += f"MATCH TYPE: {match_type} | TOKENS: {tokens}\n"
            result_log += f"{'='*60}\n\n"
            predictor.log_result(result_log)
            
        except Exception as e:
            error_msg = f"❌ Error with user {user_id}: {e}\n\n"
            predictor.log_result(error_msg)
            
            predictions.append(categories[0])
            true_labels.append(user_data["label"])
            match_types["api_error"] += 1
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
    micro_f1 = f1_score(true_labels, predictions, average='micro', zero_division=0)
    
    # Create summary
    summary = f"""
{'#'*80}
CONVERSATION-AWARE RESULTS: {model} on {dataset.upper()}
{'#'*80}
Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}

PERFORMANCE METRICS:
• Accuracy: {accuracy:.4f}
• Macro F1: {macro_f1:.4f}
• Micro F1: {micro_f1:.4f}
• Correct: {correct}/{len(test_users)} ({correct/len(test_users)*100:.1f}%)
• Total Tokens: {predictor.total_tokens:,}

MATCH TYPE BREAKDOWN:
"""
    
    for match_type, count in match_types.items():
        pct = count/len(test_users)*100 if len(test_users) > 0 else 0
        summary += f"• {match_type}: {count} ({pct:.1f}%)\n"
    
    predictor.log_result(summary)
    predictor.log_summary(summary)
    
    return {
        "dataset": dataset,
        "model": model,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "predictions": predictions,
        "true_labels": true_labels,
        "match_types": match_types,
        "total_tokens": predictor.total_tokens,
        "detailed_results": detailed_results,
        "experiment_folder": predictor.experiment_folder,
        "results_file": predictor.results_file,
        "responses_file": predictor.responses_file,
        "summary_file": predictor.summary_file
    }


def run_all_experiments(datasets: List[str], api_key: str, models: List[str], api_provider: str = "gemini", max_samples: int = 64):
    """Run conversation-aware experiments across all datasets and models."""
    
    # Create main experiment folder
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    main_folder = f"llm_conversation_experiments_{timestamp}"
    os.makedirs(main_folder, exist_ok=True)
    
    # Create overall summary file
    overall_summary_file = os.path.join(main_folder, "overall_summary.txt")
    
    all_results = {}
    experiment_folders = []
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"🆓 DATASET: {dataset.upper()}")
        print(f"{'='*60}")
        
        dataset_results = {}
        
        for model in models:
            results = run_experiment(dataset, api_key, model, api_provider, max_samples)
            dataset_results[model] = results
            experiment_folders.append(results["experiment_folder"])
        
        all_results[dataset] = dataset_results
    
    # Create overall summary
    with open(overall_summary_file, "w", encoding='utf-8') as f:
        f.write(f"CONVERSATION-AWARE EXPERIMENT SUMMARY\n")
        f.write(f"{'='*80}\n")
        f.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Datasets: {len(datasets)}\n")
        f.write(f"Total Models: {len(models)}\n")
        f.write(f"Enhanced with parent content and conversation context\n")
        f.write(f"{'='*80}\n\n")
        
        # Summary table
        f.write(f"{'Dataset':<12} {'Model':<20} {'Accuracy':<10} {'Macro F1':<10} {'Micro F1':<10} {'Tokens':<10}\n")
        f.write(f"{'-'*80}\n")
        
        for dataset, dataset_results in all_results.items():
            for model, model_results in dataset_results.items():
                f.write(f"{dataset:<12} {model:<20} {model_results['accuracy']:<10.4f} ")
                f.write(f"{model_results['macro_f1']:<10.4f} {model_results['micro_f1']:<10.4f} ")
                f.write(f"{model_results['total_tokens']:<10,}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"EXPERIMENT FOLDERS:\n")
        for i, folder in enumerate(experiment_folders, 1):
            f.write(f"{i}. {folder}\n")
        f.write(f"{'='*80}\n")
    
    print(f"\n📁 Main experiment folder: {main_folder}")
    print(f"📋 Overall summary: {overall_summary_file}")
    
    return all_results, main_folder


def save_and_analyze(results: Dict, main_folder: str):
    """Create final analysis summary with conversation awareness metrics."""
    
    # Create summary table for terminal
    summary = []
    for dataset, dataset_results in results.items():
        for model, model_results in dataset_results.items():
            summary.append({
                "Dataset": dataset,
                "Model": model,
                "Accuracy": f"{model_results['accuracy']:.4f}",
                "Macro_F1": f"{model_results['macro_f1']:.4f}",
                "Micro_F1": f"{model_results['micro_f1']:.4f}",
                "Tokens": f"{model_results['total_tokens']:,}",
                "Folder": model_results['experiment_folder']
            })
    
    if summary:
        df = pd.DataFrame(summary)
        print(f"\n{'='*60}")
        print("🏆 CONVERSATION-AWARE RESULTS SUMMARY")
        print(f"{'='*60}")
        print(df.to_string(index=False))
        
        # Save CSV summary
        csv_file = os.path.join(main_folder, "conversation_results_summary.csv")
        df.to_csv(csv_file, index=False)
        
        print(f"\n📁 Main experiment folder: {main_folder}")
        print(f"📊 Results CSV: {csv_file}")
        print(f"\n🔬 CONVERSATION-AWARE FEATURES:")
        print(f"  ✅ Parent content analysis")
        print(f"  ✅ Interaction type tracking")
        print(f"  ✅ Conversation thread context")
        print(f"  ✅ Enhanced prompt structure")
        
        print(f"\nIndividual experiment folders:")
        for i, row in df.iterrows():
            print(f"  • {row['Dataset']} - {row['Model']}: {row['Folder']}")
            print(f"    └── results.txt (performance metrics & user results)")
            print(f"    └── llm_responses.txt (all LLM prompts & responses)")
            print(f"    └── experiment_summary.txt (experiment overview)")
        
        return main_folder
    return None


if __name__ == "__main__":
    print("🚀 CONVERSATION-AWARE LLM OPINION PREDICTOR")
    print("Enhanced with parent content and conversation context analysis!")
    print("="*70)
    
    # Choose API provider and models
    API_PROVIDER = "openrouter"  # Options: "together", "gemini", "openai", "groq", "openrouter"

    match API_PROVIDER:
        case "together":
            MODELS = [
                # "meta-llama/Llama-3-8b-chat-hf",
                # "deepseek-ai/DeepSeek-R1-0528-tput",
                "deepseek-ai/DeepSeek-V3",
                "moonshotai/Kimi-K2-Instruct"
            ]
            API_KEY = ""  # Replace with your actual key
            print("💰 Using Together AI with budget-friendly models!")
            print("🎯 Estimated cost for 4000 requests:")
            print("   • Llama 3.2 3B Turbo: ~$0.12")
            print("   • Llama 3.1 8B Turbo: ~$0.36") 
            print("   • DeepSeek R1 Distill: ~$0.36")
            print("   • Kimi K2 Instruct: ~$0.36")
        case "gemini":
            MODELS = ["gemini-1.5-flash"]
            API_KEY = "your_gemini_api_key_here"
        case "openai":
            MODELS = ["gpt-4o-mini"]
            API_KEY = ""
        case "groq":
            MODELS = ["llama3-70b-8192", "gemma2-9b-it"]
            API_KEY = "your_groq_api_key_here"
        case "openrouter":
            MODELS = ["qwen/qwen3-235b-a22b-2507"]
            API_KEY = OPENROUTER_API_KEY
        case _:
            raise ValueError(f"Invalid API provider: {API_PROVIDER}")
    
    # Datasets to test
    DATASETS = ["lgbtq", "religion", "abortion", "trump"]
    # DATASETS = ["capitalism"]
    
    if API_KEY == "your_groq_api_key_here":
        print("❌ Please set your Groq API key!")
        print("Get it from: https://console.groq.com")
        exit(1)
    
    print(f"🔬 Testing {len(MODELS)} models on {len(DATASETS)} datasets")
    print(f"💰 Cost: $0 (FREE with Groq)")
    print(f"📁 Each experiment gets its own folder with organized files")
    print(f"🔍 NEW: Enhanced with conversation context and parent content!")
    
    # Run all experiments
    results, main_folder = run_all_experiments(DATASETS, API_KEY, MODELS, API_PROVIDER, max_samples=200)
    
    # Save and analyze
    final_folder = save_and_analyze(results, main_folder)
    
    print(f"\n🎉 CONVERSATION-AWARE EXPERIMENTS COMPLETE!")
    if final_folder:
        print(f"📁 All results organized in: {final_folder}")
        print(f"\nFolder structure:")
        print(f"  {final_folder}/")
        print(f"  ├── overall_summary.txt")
        print(f"  ├── conversation_results_summary.csv")
        print(f"  └── Individual experiment folders:")
        print(f"      ├── llm_experiment_[timestamp]/")
        print(f"      │   ├── results.txt")
        print(f"      │   ├── llm_responses.txt")
        print(f"      │   └── experiment_summary.txt")
        print(f"      └── [more experiment folders...]")
        print(f"\n🆕 KEY IMPROVEMENTS:")
        print(f"  • Parent content included in conversation analysis")
        print(f"  • Interaction type tracking (same_view, different_view, etc.)")
        print(f"  • Enhanced conversation thread visualization")
        print(f"  • More detailed conversation engagement patterns")
        print(f"  • Updated data path (split_datasets_text)")
    else:
        print(f"📋 Check the experiment folders for organized results")
