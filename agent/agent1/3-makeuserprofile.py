#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A1-1: Make User Profile Module
Load dataset and generate user profile files
"""

import os
import json
import random
import pandas as pd
import pickle
from typing import List, Dict, Tuple
import argparse

class UserProfileGenerator:
    """User Profile Generator Module - Load dataset and generate user profiles"""
    
    def __init__(self, train_file: str = "../dataset/ml1m/train_set.txt", 
                 movie_info_file: str = "../dataset/ml1m/movie_info.csv",
                 users_file: str = "../dataset/ml1m/users.dat"):
        self.train_file = train_file
        self.movie_info_file = movie_info_file
        self.users_file = users_file
        
        # Data storage
        self.train_data = None
        self.movie_info = None
        self.movie_id_to_name = {}
        self.users_info = {}
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load training data and movie information"""
        print("Loading data...")
        
        # Load training data
        try:
            self.train_data = pd.read_csv(self.train_file, sep=' ', names=['user_id', 'item_id', 'rating', 'timestamp'])
            print(f"✅ Training data loaded: {len(self.train_data)} records")
            print(f"   Unique users: {self.train_data['user_id'].nunique()}")
            print(f"   Unique items: {self.train_data['item_id'].nunique()}")
            print(f"   Rating range: {self.train_data['rating'].min()} - {self.train_data['rating'].max()}")
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            return False
        
        # Load movie information - try different encodings
        try:
            self.movie_info = pd.read_csv(self.movie_info_file, sep='|', header=None, encoding='utf-8',
                                         names=['movie_id', 'movie_name', 'release_date', 'video_release_date', 'IMDb_URL',
                                               'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                                               'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                                               'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
            print(f"✅ Movie info loaded: {len(self.movie_info)} movies")
        except UnicodeDecodeError:
            try:
                self.movie_info = pd.read_csv(self.movie_info_file, sep='|', header=None, encoding='latin-1',
                                             names=['movie_id', 'movie_name', 'release_date', 'video_release_date', 'IMDb_URL',
                                                   'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                                                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                                                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
                print(f"✅ Movie info loaded with latin-1 encoding: {len(self.movie_info)} movies")
            except UnicodeDecodeError:
                try:
                    self.movie_info = pd.read_csv(self.movie_info_file, sep='|', header=None, encoding='cp1252',
                                                 names=['movie_id', 'movie_name', 'release_date', 'video_release_date', 'IMDb_URL',
                                                       'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                                                       'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                                                       'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
                    print(f"✅ Movie info loaded with cp1252 encoding: {len(self.movie_info)} movies")
                except Exception as e:
                    print(f"❌ Error loading movie info: {e}")
                    return False
        
        # Create movie ID to movie name mapping
        self.movie_id_to_name = dict(zip(self.movie_info['movie_id'], self.movie_info['movie_name']))
        print(f"✅ Movie ID mapping created: {len(self.movie_id_to_name)} mappings")
        
        # Load users information
        try:
            self.users_info = self._load_users_info()
            print(f"✅ Users info loaded: {len(self.users_info)} users")
        except Exception as e:
            print(f"⚠️ Warning: Could not load users info: {e}")
            print("   User profiles will be generated without demographic information")
            self.users_info = {}
        
        return True
    
    def _load_users_info(self) -> Dict[int, Dict]:
        """Load users demographic information from users.dat"""
        users_info = {}
        
        # Age mapping
        age_mapping = {
            1: "Under 18",
            18: "18-24", 
            25: "25-34",
            35: "35-44",
            45: "45-49",
            50: "50-55",
            56: "56+"
        }
        
        # Occupation mapping
        occupation_mapping = {
            0: "other",
            1: "academic/educator",
            2: "artist",
            3: "clerical/admin",
            4: "college/grad student",
            5: "customer service",
            6: "doctor/health care",
            7: "executive/managerial",
            8: "farmer",
            9: "homemaker",
            10: "K-12 student",
            11: "lawyer",
            12: "programmer",
            13: "retired",
            14: "sales/marketing",
            15: "scientist",
            16: "self-employed",
            17: "technician/engineer",
            18: "tradesman/craftsman",
            19: "unemployed",
            20: "writer"
        }
        
        try:
            with open(self.users_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('::')
                        if len(parts) == 5:
                            user_id = int(parts[0])
                            gender = parts[1]
                            age_code = int(parts[2])
                            occupation_code = int(parts[3])
                            zip_code = parts[4]
                            
                            users_info[user_id] = {
                                'gender': gender,
                                'age': age_mapping.get(age_code, f"Unknown Age ({age_code})"),
                                'occupation': occupation_mapping.get(occupation_code, f"Unknown Occupation ({occupation_code})")
                            }
        except Exception as e:
            print(f"Error loading users info: {e}")
            return {}
        
        return users_info
    
    def generate_user_profiles(self, user_ids: List[int] = None) -> Dict[int, Dict]:
        """Generate user profiles for specified users or all users"""
        if user_ids is None:
            user_ids = list(self.train_data['user_id'].unique())
        
        print(f"Generating user profiles for {len(user_ids)} users...")
        
        user_profiles = {}
        
        for i, user_id in enumerate(user_ids):
            if i % 100 == 0:
                print(f"Processing user {i+1}/{len(user_ids)}...")
            
            user_interactions = self.train_data[self.train_data['user_id'] == user_id]
            
            if len(user_interactions) == 0:
                print(f"Warning: No interactions found for user {user_id}")
                continue
            
            profile = {
                'user_id': int(user_id),
                'interactions': [],
                'statistics': self._calculate_user_statistics(user_interactions)
            }
            
            # Add demographic information if available
            if user_id in self.users_info:
                profile['demographics'] = self.users_info[user_id]
            
            # Add interaction details
            for _, row in user_interactions.iterrows():
                movie_id = int(row['item_id'])
                rating = int(row['rating'])
                movie_name = self.movie_id_to_name.get(movie_id, f"Unknown Movie {movie_id}")
                
                profile['interactions'].append({
                    'movie_id': movie_id,
                    'movie_name': movie_name,
                    'rating': rating
                })
            
            user_profiles[user_id] = profile
        
        print(f"✅ User profiles generated: {len(user_profiles)} profiles")
        return user_profiles
    
    def _calculate_user_statistics(self, interactions: pd.DataFrame) -> Dict:
        """Calculate simplified user statistics"""
        if len(interactions) == 0:
            return {}
        
        ratings = interactions['rating']
        
        # Simplified statistics - only essential information
        stats = {
            'total_movies': int(len(interactions)),
            'rating_distribution': ratings.value_counts().to_dict()
        }
        
        # Convert rating distribution keys to int
        stats['rating_distribution'] = {int(k): int(v) for k, v in stats['rating_distribution'].items()}
        
        return stats
    
    def save_user_profiles(self, user_profiles: Dict[int, Dict], output_file: str = "user_profiles.json"):
        """Save user profiles to JSON file"""
        print(f"Saving user profiles to {output_file}...")
        
        try:
            # Ensure all data is JSON serializable
            serializable_profiles = self._ensure_json_serializable(user_profiles)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_profiles, f, ensure_ascii=False, indent=2)
            
            print(f"✅ User profiles saved: {output_file}")
            print(f"   Total profiles: {len(user_profiles)}")
            
            # Print file size
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"   File size: {file_size:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving user profiles: {e}")
            return False
    
    def save_user_profiles_pickle(self, user_profiles: Dict[int, Dict], output_file: str = "user_profiles.pkl"):
        """Save user profiles to pickle file (faster, smaller)"""
        print(f"Saving user profiles to {output_file}...")
        
        try:
            with open(output_file, 'wb') as f:
                pickle.dump(user_profiles, f)
            
            print(f"✅ User profiles saved (pickle): {output_file}")
            print(f"   Total profiles: {len(user_profiles)}")
            
            # Print file size
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"   File size: {file_size:.2f} MB")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving user profiles (pickle): {e}")
            return False
    
    def _ensure_json_serializable(self, obj):
        """Ensure object is JSON serializable"""
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
    
    def get_profile_summary(self, user_profiles: Dict[int, Dict]) -> Dict:
        """Get summary statistics of all user profiles"""
        if not user_profiles:
            return {}
        
        total_users = len(user_profiles)
        total_interactions = sum(len(profile['interactions']) for profile in user_profiles.values())
        
        # Rating statistics
        all_ratings = []
        for profile in user_profiles.values():
            for interaction in profile['interactions']:
                all_ratings.append(interaction['rating'])
        
        all_ratings = pd.Series(all_ratings)
        
        summary = {
            'total_users': total_users,
            'total_interactions': total_interactions,
            'average_interactions_per_user': total_interactions / total_users if total_users > 0 else 0,
            'rating_statistics': {
                'rating_distribution': all_ratings.value_counts().to_dict()
            },
            'demographics_info': {
                'users_with_demographics': len([p for p in user_profiles.values() if 'demographics' in p]),
                'users_without_demographics': len([p for p in user_profiles.values() if 'demographics' not in p])
            }
        }
        
        # Convert rating distribution keys to int
        summary['rating_statistics']['rating_distribution'] = {
            int(k): int(v) for k, v in summary['rating_statistics']['rating_distribution'].items()
        }
        
        return summary
    
    def print_profile_summary(self, user_profiles: Dict[int, Dict]):
        """Print summary of user profiles"""
        summary = self.get_profile_summary(user_profiles)
        
        print("\n" + "=" * 60)
        print("USER PROFILES SUMMARY")
        print("=" * 60)
        print(f"Total users: {summary['total_users']}")
        print(f"Total interactions: {summary['total_interactions']}")
        print(f"Average interactions per user: {summary['average_interactions_per_user']:.2f}")
        print(f"\nRating Statistics:")
        print(f"  Rating distribution: {summary['rating_statistics']['rating_distribution']}")
        print(f"\nDemographics Information:")
        print(f"  Users with demographics: {summary['demographics_info']['users_with_demographics']}")
        print(f"  Users without demographics: {summary['demographics_info']['users_without_demographics']}")
        print("=" * 60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate user profiles from training data')
    parser.add_argument('--train-file', type=str, default='../dataset/ml1m/train_set.txt',
                       help='Training data file path')
    parser.add_argument('--movie-info-file', type=str, default='../dataset/ml1m/movie_info.csv',
                       help='Movie information file path')
    parser.add_argument('--users-file', type=str, default='../dataset/ml1m/users.dat',
                       help='Users demographic information file path')
    parser.add_argument('--output-json', type=str, default='user_profiles.json',
                       help='Output JSON file path')
    parser.add_argument('--output-pickle', type=str, default='user_profiles.pkl',
                       help='Output pickle file path')
    parser.add_argument('--sample-users', type=int, default=None,
                       help='Number of users to sample (None for all users) - DEPRECATED: Always processes all users')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for sampling')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("A1-1: Make User Profile Module")
    print("=" * 60)
    print(f"Training file: {args.train_file}")
    print(f"Movie info file: {args.movie_info_file}")
    print(f"Users file: {args.users_file}")
    print(f"Output JSON: {args.output_json}")
    print(f"Output Pickle: {args.output_pickle}")
    print(f"Sample users: All users (sampling disabled)")
    print(f"Random seed: {args.random_seed}")
    print()
    
    # Set random seed
    random.seed(args.random_seed)
    
    # Create user profile generator
    generator = UserProfileGenerator(args.train_file, args.movie_info_file, args.users_file)
    
    # Check if data loaded successfully
    if generator.train_data is None or generator.movie_info is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Generate user profiles for ALL users
    print("Generating user profiles for ALL users in the dataset...")
    user_profiles = generator.generate_user_profiles()
    
    # Print summary
    generator.print_profile_summary(user_profiles)
    
    # Save profiles
    success_json = generator.save_user_profiles(user_profiles, args.output_json)
    success_pickle = generator.save_user_profiles_pickle(user_profiles, args.output_pickle)
    
    if success_json and success_pickle:
        print("\n" + "=" * 60)
        print("✅ User Profile Generation Completed Successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️ User Profile Generation Completed with Warnings!")
        print("=" * 60)


if __name__ == "__main__":
    main()
