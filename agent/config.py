OPENAI_API_KEY = "your_openai_api_key_here"

DATA_CONFIG = {
    "train_file": "ml-100k/train_set.txt",
    "movie_info_file": "ml-100k/movie_info.csv",
    "dislike_file": "ml-100k/dislike.txt",
    "test_file": "ml-100k/test_set.txt",
    "valid_file": "ml-100k/valid_set.txt"
}

AGENT1_CONFIG = {
    "sample_size": 1,
    "max_items_per_user": 50,
    "gpt_model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 200,
    "random_seed": 42
}

OUTPUT_CONFIG = {
    "sampled_data_file": "agent1_sampled_data.txt",
    "detailed_results_file": "agent1_detailed_results.json",
    "memory_file": "agent1_memory.pkl"
}

PROMPT_CONFIG = {
    "user_profile_system": """You are a professional movie recommendation system analyst. Your task is to analyze and describe users' unique tastes and preferences based on their viewing history and ratings.

Please describe this user's unique taste characteristics in 2-3 sentences based on the provided viewing history, including:
1. User's preferred movie genres
2. User's favorite movie styles or themes
3. User's viewing preference characteristics

Please use natural, fluent English descriptions and avoid overly technical terms.""",

    "action_system": """You are an intelligent movie recommendation system. Your task is to select the most representative movies for a user's taste based on their viewing history and taste description.

Please carefully analyze the user's viewing history and taste characteristics, and select movies that best represent the user's unique taste. Selection criteria:
1. Movies with high user ratings (4-5 stars)
2. Movies that match the user's taste description
3. Movies that can represent the user's viewing preferences

Please select up to 50 movies from the user's interacted movies, sorted by importance."""
}

RATING_CONFIG = {
    "high_rating_threshold": 4,
    "medium_rating_threshold": 3,
    "max_rating": 5
} 
