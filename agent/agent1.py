import os
import json
import random
import pandas as pd
import pickle
from typing import List, Dict
from openai import OpenAI

class UserSamplingAgent:
    
    def __init__(self, openai_api_key: str, sample_size: int = 1):
        self.openai_api_key = openai_api_key
        self.sample_size = sample_size
        
        self.client = OpenAI(api_key=openai_api_key)
        
        self.train_file = "ml-100k/train_set.txt"
        self.movie_info_file = "ml-100k/movie_info.csv"
        
        self.user_profile_module = UserProfileModule(self.client)
        self.memory_module = MemoryModule()
        self.action_module = ActionModule(self.client)
        
        self.load_data()
    
    def load_data(self):
        print("Loading data...")
        
        self.train_data = pd.read_csv(self.train_file, sep=' ', names=['user_id', 'item_id', 'rating', 'timestamp'])
        
        try:
            self.movie_info = pd.read_csv(self.movie_info_file, sep='|', header=None, encoding='utf-8',
                                         names=['movie_id', 'movie_name', 'release_date', 'video_release_date', 'IMDb_URL',
                                               'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                                               'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                                               'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
        except UnicodeDecodeError:
            try:
                self.movie_info = pd.read_csv(self.movie_info_file, sep='|', header=None, encoding='latin-1',
                                             names=['movie_id', 'movie_name', 'release_date', 'video_release_date', 'IMDb_URL',
                                                   'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                                                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                                                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
            except UnicodeDecodeError:
                self.movie_info = pd.read_csv(self.movie_info_file, sep='|', header=None, encoding='cp1252',
                                             names=['movie_id', 'movie_name', 'release_date', 'video_release_date', 'IMDb_URL',
                                                   'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                                                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                                                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
        
        self.movie_id_to_name = dict(zip(self.movie_info['movie_id'], self.movie_info['movie_name']))
        
        print(f"Data loading completed: {len(self.train_data)} training records, {len(self.movie_info)} movies")
    
    def run(self):
        print("Starting User Sampling Agent...")
        
        sampled_users = self.sample_users()
        print(f"Sampled {len(sampled_users)} users")
        
        print("Generating user profiles...")
        user_profiles = self.user_profile_module.generate_profiles(sampled_users, self.train_data, self.movie_id_to_name)
        
        print("Generating unique user taste descriptions...")
        user_tastes = self.user_profile_module.generate_user_tastes(user_profiles)
        
        print("Storing user information to memory module...")
        self.memory_module.store_user_profiles(user_profiles, user_tastes)
        
        print("Selecting sampling items...")
        sampled_items = self.action_module.select_items(user_profiles, user_tastes, self.train_data, self.movie_id_to_name)
        
        self.output_results(sampled_users, sampled_items)
        
        print("User Sampling Agent completed!")
        
    def sample_users(self) -> List[int]:
        all_users = self.train_data['user_id'].unique()
        sampled_users = random.sample(list(all_users), min(self.sample_size, len(all_users)))
        return sampled_users
    
    def output_results(self, sampled_users: List[int], sampled_items: Dict[int, List[int]]):
        output_file = "agent1_sampled_data.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for user_id in sampled_users:
                items = sampled_items.get(user_id, [])
                items_str = ' '.join(map(str, items))
                f.write(f"{user_id} {items_str}\n")
        
        print(f"Sampling results saved to {output_file}")
        
        detailed_results = {
            'sampled_users': [int(uid) for uid in sampled_users],
            'user_items': {int(uid): [int(item) for item in items] for uid, items in sampled_items.items()},
            'user_profiles': self._convert_profiles_for_json(self.memory_module.get_all_profiles()),
            'user_tastes': self.memory_module.get_all_tastes()
        }
        
        detailed_results = self._ensure_json_serializable(detailed_results)
        
        with open("agent1_detailed_results.json", 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        print("Detailed results saved to agent1_detailed_results.json")
    
    def _ensure_json_serializable(self, obj):
        if isinstance(obj, dict):
            return {str(k): self._ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        elif hasattr(obj, 'dtype'):
            return obj.tolist()
        else:
            return obj
    
    def _convert_profiles_for_json(self, profiles: Dict[int, Dict]) -> Dict[int, Dict]:
        converted = {}
        for user_id, profile in profiles.items():
            converted[int(user_id)] = {
                'user_id': int(profile['user_id']),
                'interactions': [
                    {
                        'movie_id': int(item['movie_id']),
                        'movie_name': str(item['movie_name']),
                        'rating': int(item['rating'])
                    }
                    for item in profile['interactions']
                ]
            }
        return converted


class UserProfileModule:
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.system_prompt = """You are a professional movie recommendation system analyst. Your task is to analyze and describe users' unique tastes and preferences based on their viewing history and ratings.

Please describe this user's unique taste characteristics in 2-3 sentences based on the provided viewing history, including:
1. User's preferred movie genres
2. User's favorite movie styles or themes
3. User's viewing preference characteristics

Please use natural, fluent English descriptions and avoid overly technical terms."""
    
    def generate_profiles(self, sampled_users: List[int], train_data: pd.DataFrame, 
                         movie_id_to_name: Dict[int, str]) -> Dict[int, Dict]:
        user_profiles = {}
        
        for user_id in sampled_users:
            user_interactions = train_data[train_data['user_id'] == user_id]
            
            profile = {
                'user_id': user_id,
                'interactions': []
            }
            
            for _, row in user_interactions.iterrows():
                movie_id = row['item_id']
                rating = row['rating']
                movie_name = movie_id_to_name.get(movie_id, f"Unknown Movie {movie_id}")
                
                profile['interactions'].append({
                    'movie_id': movie_id,
                    'movie_name': movie_name,
                    'rating': rating
                })
            
            user_profiles[user_id] = profile
        
        return user_profiles
    
    def generate_user_tastes(self, user_profiles: Dict[int, Dict]) -> Dict[int, str]:
        user_tastes = {}
        
        for user_id, profile in user_profiles.items():
            try:
                user_prompt = self._build_user_prompt(profile)
                
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=200,
                    temperature=0.7
                )
                
                taste_description = response.choices[0].message.content.strip()
                user_tastes[user_id] = taste_description
                
                print(f"Taste description for user {user_id} generated")
                
            except Exception as e:
                print(f"Error generating taste description for user {user_id}: {e}")
                user_tastes[user_id] = "This user's taste characteristics need further analysis"
        
        return user_tastes
    
    def _build_user_prompt(self, profile: Dict) -> str:
        user_id = profile['user_id']
        interactions = profile['interactions']
        
        high_rated = [item for item in interactions if item['rating'] >= 4]
        medium_rated = [item for item in interactions if 3 <= item['rating'] < 4]
        
        prompt = f"User ID: {user_id}\n\n"
        prompt += "User viewing history (sorted by rating):\n\n"
        
        if high_rated:
            prompt += "High-rated movies (4-5 stars):\n"
            for item in high_rated[:10]:
                prompt += f"- {item['movie_name']} (Rating: {item['rating']})\n"
            prompt += "\n"
        
        if medium_rated:
            prompt += "Medium-rated movies (3-4 stars):\n"
            for item in medium_rated[:5]:
                prompt += f"- {item['movie_name']} (Rating: {item['rating']})\n"
            prompt += "\n"
        
        prompt += f"Total movies watched: {len(interactions)} movies\n"
        prompt += f"Average rating: {sum(item['rating'] for item in interactions) / len(interactions):.2f}\n\n"
        
        prompt += "Please analyze and describe this user's unique taste characteristics based on the above viewing history."
        
        return prompt


class MemoryModule:
    
    def __init__(self):
        self.user_profiles = {}
        self.user_tastes = {}
    
    def store_user_profiles(self, user_profiles: Dict[int, Dict], user_tastes: Dict[int, str]):
        self.user_profiles = user_profiles
        self.user_tastes = user_tastes
        
        self.save_to_file()
    
    def get_user_profile(self, user_id: int) -> Dict:
        return self.user_profiles.get(user_id, {})
    
    def get_user_taste(self, user_id: int) -> str:
        return self.user_tastes.get(user_id, "")
    
    def get_all_profiles(self) -> Dict[int, Dict]:
        return self.user_profiles
    
    def get_all_tastes(self) -> Dict[int, str]:
        return self.user_tastes
    
    def save_to_file(self):
        data = {
            'user_profiles': self.user_profiles,
            'user_tastes': self.user_tastes
        }
        with open("agent1_memory.pkl", 'wb') as f:
            pickle.dump(data, f)
    
    def load_from_file(self):
        try:
            with open("agent1_memory.pkl", 'rb') as f:
                data = pickle.load(f)
                self.user_profiles = data['user_profiles']
                self.user_tastes = data['user_tastes']
        except FileNotFoundError:
            print("Memory file does not exist, will create new memory")


class ActionModule:
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.system_prompt = """You are an intelligent movie recommendation system. Your task is to select the most representative movies for a user's taste based on their viewing history and taste description.

Please carefully analyze the user's viewing history and taste characteristics, and select movies that best represent the user's unique taste. Selection criteria:
1. Movies with high user ratings (4-5 stars)
2. Movies that match the user's taste description
3. Movies that can represent the user's viewing preferences

Please select up to 50 movies from the user's interacted movies, sorted by importance."""
    
    def select_items(self, user_profiles: Dict[int, Dict], user_tastes: Dict[int, str],
                    train_data: pd.DataFrame, movie_id_to_name: Dict[int, str]) -> Dict[int, List[int]]:
        sampled_items = {}
        
        for user_id in user_profiles.keys():
            profile = user_profiles[user_id]
            taste = user_tastes.get(user_id, "")
            
            user_interactions = train_data[train_data['user_id'] == user_id]
            
            selected_items = self._smart_select_items(user_id, profile, taste, user_interactions, movie_id_to_name)
            sampled_items[user_id] = selected_items
        
        return sampled_items
    
    def _smart_select_items(self, user_id: int, profile: Dict, taste: str, 
                           interactions: pd.DataFrame, movie_id_to_name: Dict[int, str]) -> List[int]:
        try:
            selection_prompt = self._build_selection_prompt(user_id, profile, taste, interactions, movie_id_to_name)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": selection_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content.strip()
            selected_items = self._parse_gpt_response(response_text, interactions)
            
            return selected_items
            
        except Exception as e:
            print(f"Intelligent selection failed for user {user_id}, using fallback method: {e}")
            return self._fallback_selection(interactions)
    
    def _build_selection_prompt(self, user_id: int, profile: Dict, taste: str, 
                               interactions: pd.DataFrame, movie_id_to_name: Dict[int, str]) -> str:
        prompt = f"User ID: {user_id}\n\n"
        prompt += f"User taste description: {taste}\n\n"
        
        high_rated = interactions[interactions['rating'] >= 4].sort_values('rating', ascending=False)
        
        prompt += "User's high-rated movies (4-5 stars):\n"
        for _, row in high_rated.head(10).iterrows():
            movie_name = movie_id_to_name.get(row['item_id'], f"Unknown Movie {row['item_id']}")
            prompt += f"- {movie_name} (Rating: {row['rating']})\n"
        
        prompt += f"\nTotal movies user has interacted with: {len(interactions)}\n"
        prompt += "Please select the movie IDs that best represent this user's taste based on their taste description and viewing history."
        
        return prompt
    
    def _parse_gpt_response(self, response_text: str, interactions: pd.DataFrame) -> List[int]:
        try:
            import re
            numbers = re.findall(r'\d+', response_text)
            
            user_movie_ids = set(interactions['item_id'].tolist())
            selected_items = []
            
            for num in numbers:
                movie_id = int(num)
                if movie_id in user_movie_ids and movie_id not in selected_items:
                    selected_items.append(movie_id)
                    if len(selected_items) >= 50:
                        break
            
            return selected_items[:50]
            
        except Exception as e:
            print(f"Failed to parse GPT response: {e}")
            return self._fallback_selection(interactions)
    
    def _fallback_selection(self, interactions: pd.DataFrame) -> List[int]:
        high_rated = interactions[interactions['rating'] >= 4].sort_values('rating', ascending=False)
        selected_items = high_rated['item_id'].head(50).tolist()
        return selected_items


def main():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        try:
            from config import OPENAI_API_KEY as config_api_key
            if config_api_key and config_api_key != "your_openai_api_key_here":
                openai_api_key = config_api_key
            else:
                print("Please set your OpenAI API key in config.py file")
                print("Or set environment variable OPENAI_API_KEY")
                return
        except ImportError:
            print("Cannot import config.py file")
            print("Please set environment variable OPENAI_API_KEY")
            return
    
    try:
        from config import AGENT1_CONFIG
        sample_size = AGENT1_CONFIG.get("sample_size", 1)
    except ImportError:
        sample_size = 1
    
    random.seed(42)
    
    agent = UserSamplingAgent(openai_api_key, sample_size=sample_size)
    agent.run()


if __name__ == "__main__":
    main() 
