

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from openai import OpenAI
import pickle

class ItemProfileAgent:
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)
        

        try:
            from config_kg import DATA_CONFIG, AGENT2_CONFIG
            self.data_config = DATA_CONFIG
            self.agent_config = AGENT2_CONFIG
        except ImportError:

            self.data_config = {
                "movie_info_file": "movie_info.csv",
                "kg_triplets_file": "processed_kg_text.tsv",
                "pretrain_kg_text_file": "pretrain-output_kg_text.tsv",
                "ratings_file": "ratings.csv"
            }
            self.agent_config = {
                "max_kg_triplets_per_item": 20,
                "gpt_model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 300,
                "embedding_model": "text-embedding-ada-002",
                "use_pretrain_kg": True
            }
        
        self.movie_info_file = self.data_config["movie_info_file"]
        self.kg_triplets_file = self.data_config["kg_triplets_file"]
        self.pretrain_kg_text_file = self.data_config["pretrain_kg_text_file"]
        self.ratings_file = self.data_config["ratings_file"]
        
        self.item_profile_module = ItemProfileModule(self.client)
        self.kg_enhancement_module = KGEnhancementModule()
        self.embedding_module = EmbeddingModule(self.client)
        
        self.load_data()
    
    def load_data(self):
        print("Loading data for Item Profile Agent...")
        
        self.movie_info = pd.read_csv(self.movie_info_file, sep='|', header=None, encoding='utf-8',
                                     names=['movie_id', 'movie_name', 'release_date', 'video_release_date', 'IMDb_URL',
                                           'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                                           'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                                           'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
        

        if self.agent_config.get("use_pretrain_kg", False) and os.path.exists(self.pretrain_kg_text_file):
            print(f"Using pretrain KG file: {self.pretrain_kg_text_file}")
            kg_file = self.pretrain_kg_text_file
        else:
            print(f"Using original KG file: {self.kg_triplets_file}")
            kg_file = self.kg_triplets_file
        
        self.kg_triplets = pd.read_csv(kg_file, sep='\t', header=None, encoding='utf-8',
                                      names=['head', 'relation', 'tail'])
        
        self.ratings = pd.read_csv(self.ratings_file, sep=' ', names=['user_id', 'item_id', 'rating', 'timestamp'])
        
        print(f"Data loading completed: {len(self.movie_info)} movies, {len(self.kg_triplets)} KG triplets, {len(self.ratings)} ratings")
    
    def run(self):
        print("Starting Item Profile Agent...")
        
        print("Generating item profiles...")
        item_profiles = self.item_profile_module.generate_item_profiles(self.movie_info, self.ratings)
        
        print("Enhancing profiles with knowledge graph information...")
        enhanced_profiles = self.kg_enhancement_module.enhance_profiles(item_profiles, self.kg_triplets)
        
        print("Generating item embeddings...")
        item_embeddings = self.embedding_module.generate_embeddings(enhanced_profiles)
        
        print("Saving results...")
        self.save_results(item_profiles, enhanced_profiles, item_embeddings)
        
        print("Item Profile Agent completed!")
    
    def save_results(self, item_profiles: Dict, enhanced_profiles: Dict, item_embeddings: Dict):
        results = {
            'item_profiles': item_profiles,
            'enhanced_profiles': enhanced_profiles,
            'item_embeddings': item_embeddings
        }
        
        with open("agent2_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        with open("agent2_embeddings.pkl", 'wb') as f:
            pickle.dump(item_embeddings, f)
        
        print("Results saved to agent2_results.json and agent2_embeddings.pkl")


class ItemProfileModule:
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.system_prompt = """You are a professional movie analyst. Your task is to analyze and describe movie characteristics based on their metadata and user ratings.

Please provide a comprehensive analysis of each movie including:
1. Movie genre characteristics
2. Target audience
3. Movie style and themes
4. Quality indicators based on ratings
5. Unique selling points

Please use natural, fluent English descriptions."""
    
    def generate_item_profiles(self, movie_info: pd.DataFrame, ratings: pd.DataFrame) -> Dict[int, Dict]:
        item_profiles = {}
        
        for _, movie in movie_info.iterrows():
            movie_id = movie['movie_id']
            
            profile = {
                'movie_id': movie_id,
                'movie_name': movie['movie_name'],
                'release_date': movie['release_date'],
                'genres': self._extract_genres(movie),
                'rating_stats': self._calculate_rating_stats(movie_id, ratings),
                'description': self._generate_description(movie, ratings)
            }
            
            item_profiles[movie_id] = profile
        
        return item_profiles
    
    def _extract_genres(self, movie: pd.Series) -> List[str]:
        genre_columns = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
                        'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 
                        'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        genres = []
        for genre in genre_columns:
            if movie[genre] == 1:
                genres.append(genre)
        
        return genres
    
    def _calculate_rating_stats(self, movie_id: int, ratings: pd.DataFrame) -> Dict:
        movie_ratings = ratings[ratings['item_id'] == movie_id]['rating']
        
        if len(movie_ratings) == 0:
            return {
                'avg_rating': 0,
                'num_ratings': 0,
                'rating_distribution': {}
            }
        
        rating_dist = movie_ratings.value_counts().to_dict()
        
        return {
            'avg_rating': float(movie_ratings.mean()),
            'num_ratings': int(len(movie_ratings)),
            'rating_distribution': {int(k): int(v) for k, v in rating_dist.items()}
        }
    
    def _generate_description(self, movie: pd.Series, ratings: pd.DataFrame) -> str:
        try:
            prompt = f"Movie: {movie['movie_name']}\n"
            prompt += f"Release Date: {movie['release_date']}\n"
            prompt += f"Genres: {', '.join(self._extract_genres(movie))}\n"
            
            rating_stats = self._calculate_rating_stats(movie['movie_id'], ratings)
            prompt += f"Average Rating: {rating_stats['avg_rating']:.2f}\n"
            prompt += f"Number of Ratings: {rating_stats['num_ratings']}\n\n"
            
            prompt += "Please provide a comprehensive analysis of this movie's characteristics."
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating description for movie {movie['movie_id']}: {e}")
            return f"A {', '.join(self._extract_genres(movie))} movie released in {movie['release_date']}"


class KGEnhancementModule:
    
    def __init__(self):
        pass
    
    def enhance_profiles(self, item_profiles: Dict[int, Dict], kg_triplets: pd.DataFrame) -> Dict[int, Dict]:
        enhanced_profiles = {}
        
        for movie_id, profile in item_profiles.items():
            movie_name = profile['movie_name']
            
            kg_info = self._extract_kg_info(movie_name, kg_triplets)
            
            enhanced_profile = profile.copy()
            enhanced_profile['kg_relations'] = kg_info
            enhanced_profile['enhanced_description'] = self._enhance_description(profile, kg_info)
            
            enhanced_profiles[movie_id] = enhanced_profile
        
        return enhanced_profiles
    
    def _extract_kg_info(self, movie_name: str, kg_triplets: pd.DataFrame) -> List[Dict]:
        kg_info = []
        
        movie_triplets = kg_triplets[
            (kg_triplets['head'].str.contains(movie_name, case=False, na=False)) |
            (kg_triplets['tail'].str.contains(movie_name, case=False, na=False))
        ]
        
        for _, triplet in movie_triplets.iterrows():
            kg_info.append({
                'head': triplet['head'],
                'relation': triplet['relation'],
                'tail': triplet['tail']
            })
        
        return kg_info[:20]  # Limit to 20 most relevant triplets
    
    def _enhance_description(self, profile: Dict, kg_info: List[Dict]) -> str:
        if not kg_info:
            return profile['description']
        
        kg_context = "Knowledge Graph Information:\n"
        for info in kg_info[:5]:  # Use top 5 triplets
            kg_context += f"- {info['head']} {info['relation']} {info['tail']}\n"
        
        enhanced_desc = profile['description'] + "\n\n" + kg_context
        return enhanced_desc


class EmbeddingModule:
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def generate_embeddings(self, enhanced_profiles: Dict[int, Dict]) -> Dict[int, List[float]]:
        embeddings = {}
        
        for movie_id, profile in enhanced_profiles.items():
            try:
                text = self._prepare_embedding_text(profile)
                
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                
                embedding = response.data[0].embedding
                embeddings[movie_id] = embedding
                
                print(f"Generated embedding for movie {movie_id}")
                
            except Exception as e:
                print(f"Error generating embedding for movie {movie_id}: {e}")
                embeddings[movie_id] = [0.0] * 1536  # Default embedding size
        
        return embeddings
    
    def _prepare_embedding_text(self, profile: Dict) -> str:
        text_parts = [
            profile['movie_name'],
            ', '.join(profile['genres']),
            profile['enhanced_description']
        ]
        
        return ' '.join(text_parts)


def main():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        try:
            import sys
            sys.path.append('..')
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
    
    agent = ItemProfileAgent(openai_api_key)
    agent.run()


if __name__ == "__main__":
    main() 
