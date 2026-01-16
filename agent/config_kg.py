OPENAI_API_KEY = "your_openai_api_key_here"

DATA_CONFIG = {
    "movie_info_file": "movie_info.csv",
    "kg_triplets_file": "processed_kg_text.tsv",
    "pretrain_kg_text_file": "pretrain-output_kg_text_top10.tsv",
    "ratings_file": "ratings.csv"
}

AGENT2_CONFIG = {
    "max_kg_triplets_per_item": 20,
    "gpt_model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 300,
    "embedding_model": "text-embedding-ada-002",
    "use_pretrain_kg": True
}

OUTPUT_CONFIG = {
    "agent2_results_file": "agent2_results.json",
    "agent2_embeddings_file": "agent2_embeddings.pkl"
}

PROMPT_CONFIG = {
    "item_profile_system": """You are a professional movie analyst. Your task is to analyze and describe movie characteristics based on their metadata and user ratings.

Please provide a comprehensive analysis of each movie including:
1. Movie genre characteristics
2. Target audience
3. Movie style and themes
4. Quality indicators based on ratings
5. Unique selling points

Please use natural, fluent English descriptions."""
}

EVALUATION_CONFIG = {
    "high_rating_threshold": 4,
    "coverage_weight": 0.3,
    "diversity_weight": 0.3,
    "precision_weight": 0.4
} 
