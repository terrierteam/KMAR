import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json
import os

class DatasetAnalyzer:
    """Dataset feature analyzer"""
    
    def __init__(self):
        """Initialize dataset analyzer"""
        self.features = {}
        print(" Dataset Analyzer initialized")
    
    def analyze_dataset(self, ratings_file: str, kg_file: str) -> Dict:
        """Analyze dataset features
        
        Args:
            ratings_file: User interaction data file path
            kg_file: Knowledge graph triplets file path
            
        Returns:
            Dataset features dictionary
        """
        print(f" Analyzing dataset features...")
        print(f"   - Ratings file: {ratings_file}")
        print(f"   - KG file: {kg_file}")
        
        features = {}
        
        # Analyze user interaction data
        features.update(self._analyze_ratings(ratings_file))
        
        # Analyze knowledge graph data
        features.update(self._analyze_knowledge_graph(kg_file))
        
        # Comprehensive feature analysis
        features.update(self._analyze_comprehensive_features(features))
        
        print(f"✅ Dataset analysis completed")
        return features
    
    def _analyze_ratings(self, ratings_file: str) -> Dict:
        
        try:
            
            if ratings_file.endswith('.csv'):
                
                try:
                    df = pd.read_csv(ratings_file)
                    
                    if len(df.columns) == 1:
                        df = pd.read_csv(ratings_file, sep='\t', header=None,
                                       names=['user_id', 'item_id', 'rating', 'timestamp'])
                except:
                    df = pd.read_csv(ratings_file, sep='\t', header=None,
                                   names=['user_id', 'item_id', 'rating', 'timestamp'])
            elif ratings_file.endswith('.txt'):
                
                df = pd.read_csv(ratings_file, sep=',', header=None,
                               names=['user_id', 'item_id', 'rating', 'timestamp'])
            elif ratings_file.endswith('.dat'):
                df = pd.read_csv(ratings_file, sep='::', header=None, 
                               names=['user_id', 'item_id', 'rating', 'timestamp'])
            else:
                df = pd.read_csv(ratings_file, sep='\t', header=None,
                               names=['user_id', 'item_id', 'rating', 'timestamp'])
            
            
            user_col = df.columns[0]
            item_col = df.columns[1]
            
            
            total_interactions = len(df)
            unique_users = df[user_col].nunique()
            unique_items = df[item_col].nunique()
            
           
            if unique_users == 0 or unique_items == 0:
                print(f"   Error: No valid users or items found")
                return {"error": "No valid users or items found"}
            
            features = {
                
                "total_interactions": total_interactions,
                "unique_users": unique_users,
                "unique_items": unique_items,
                "avg_ratings_per_user": total_interactions / unique_users,
                "avg_ratings_per_item": total_interactions / unique_items,
                
                
                "sparsity": 1 - (total_interactions / (unique_users * unique_items)),
                
                
                "user_interaction_std": df[user_col].value_counts().std(),
                "item_interaction_std": df[item_col].value_counts().std(),
                "user_interaction_mean": df[user_col].value_counts().mean(),
                "item_interaction_mean": df[item_col].value_counts().mean(),
                
                
                "dataset_scale": self._classify_dataset_scale(len(df), len(df[user_col].unique()), len(df[item_col].unique())),
                
                
                "user_activity_distribution": self._analyze_user_activity(df, user_col),
                
                
                "item_popularity_distribution": self._analyze_item_popularity(df, item_col),
                
                
                "has_timestamp": len(df.columns) > 2,
                "time_span_days": self._analyze_time_span(df) if len(df.columns) > 2 else 0
            }
            
            
            if len(df.columns) > 2:
                rating_col = df.columns[2]
                if df[rating_col].dtype in ['int64', 'float64']:
                    features.update(self._analyze_rating_distribution(df, rating_col))
            
            print(f"    Ratings analysis: {features['unique_users']:,} users, {features['unique_items']:,} items")
            return features
            
        except Exception as e:
            print(f"    Error analyzing ratings: {e}")
            return {"error": str(e)}
    
    def _analyze_knowledge_graph(self, kg_file: str) -> Dict:
        
        try:
            
            df = pd.read_csv(kg_file, sep='\t', header=None, 
                           names=['head', 'relation', 'tail'])
            
            features = {
                
                "total_triplets": len(df),
                "unique_entities": df['head'].nunique() + df['tail'].nunique(),
                "unique_relations": df['relation'].nunique(),
                "unique_items_in_kg": df['head'].nunique(),
                
                
                "kg_density": len(df) / (df['head'].nunique() * df['relation'].nunique()) if df['head'].nunique() > 0 else 0,
                "avg_triplets_per_item": len(df) / df['head'].nunique() if df['head'].nunique() > 0 else 0,
                "avg_triplets_per_relation": len(df) / df['relation'].nunique() if df['relation'].nunique() > 0 else 0,
                
               
                "relation_diversity": df['relation'].value_counts().std(),
                "entity_connectivity": self._analyze_entity_connectivity(df),
                
               
                "relation_distribution": self._analyze_relation_distribution(df)
            }
            
            print(f"    KG analysis: {features['total_triplets']:,} triplets, {features['unique_relations']} relations")
            return features
            
        except Exception as e:
            print(f"    Error analyzing KG: {e}")
            return {"kg_error": str(e)}
    
    def _analyze_comprehensive_features(self, features: Dict) -> Dict:
        
        return {
            
            "complexity_score": self._calculate_complexity_score(features),
            
            
            "kg_richness": self._calculate_kg_richness(features),
            
            
            "recommendation_difficulty": self._calculate_recommendation_difficulty(features),
            
            
            "computational_demand": self._estimate_computational_demand(features),
            
            
            "data_quality_score": self._calculate_data_quality_score(features)
        }
    
    def _classify_dataset_scale(self, total_interactions: int, unique_users: int = 0, unique_items: int = 0) -> str:

        if unique_users >= 50000:
            return "large"  
        elif unique_items >= 20000 and unique_users < 50000:
            return "medium" 
        elif total_interactions < 1000000 and unique_users < 10000 and unique_items < 10000:
            return "small"  
        elif total_interactions < 100000:
            return "small"
        elif total_interactions < 1000000:
            return "medium"
        elif total_interactions < 10000000:
            return "large"
        else:
            return "very_large"
    
    def _analyze_user_activity(self, df: pd.DataFrame, user_col: str) -> Dict:
        
        user_counts = df[user_col].value_counts()
        return {
            "highly_active_users": len(user_counts[user_counts >= 100]),
            "moderately_active_users": len(user_counts[(user_counts >= 20) & (user_counts < 100)]),
            "low_active_users": len(user_counts[user_counts < 20]),
            "activity_gini": self._calculate_gini_coefficient(user_counts.values)
        }
    
    def _analyze_item_popularity(self, df: pd.DataFrame, item_col: str) -> Dict:
        
        item_counts = df[item_col].value_counts()
        return {
            "popular_items": len(item_counts[item_counts >= 100]),
            "moderate_items": len(item_counts[(item_counts >= 10) & (item_counts < 100)]),
            "long_tail_items": len(item_counts[item_counts < 10]),
            "popularity_gini": self._calculate_gini_coefficient(item_counts.values)
        }
    
    def _analyze_relation_distribution(self, df: pd.DataFrame) -> Dict:
        
        relation_counts = df['relation'].value_counts()
        return {
            "most_common_relation": relation_counts.index[0] if len(relation_counts) > 0 else None,
            "relation_entropy": self._calculate_entropy(relation_counts.values),
            "relation_balance": len(relation_counts[relation_counts >= relation_counts.mean()])
        }
    
    def _analyze_time_span(self, df: pd.DataFrame) -> int:
        
        try:
            timestamp_col = df.columns[3]  
            timestamps = pd.to_numeric(df[timestamp_col], errors='coerce')
            if timestamps.notna().any():
                time_span = (timestamps.max() - timestamps.min()) / (24 * 3600)  
                return int(time_span)
        except:
            pass
        return 0
    
    def _analyze_rating_distribution(self, df: pd.DataFrame, rating_col: str) -> Dict:
        
        ratings = df[rating_col]
        return {
            "rating_mean": ratings.mean(),
            "rating_std": ratings.std(),
            "rating_min": ratings.min(),
            "rating_max": ratings.max(),
            "rating_distribution": ratings.value_counts().to_dict()
        }
    
    def _analyze_entity_connectivity(self, df: pd.DataFrame) -> Dict:
        
        try:
            
            entity_connections = pd.concat([df['head'], df['tail']]).value_counts()
            return {
                "avg_connections_per_entity": entity_connections.mean(),
                "max_connections": entity_connections.max(),
                "min_connections": entity_connections.min(),
                "connection_std": entity_connections.std()
            }
        except:
            return {"avg_connections_per_entity": 0}
    
    def _calculate_complexity_score(self, features: Dict) -> float:
        
        try:
            
            user_complexity = np.log10(features.get("unique_users", 1) + 1)
            item_complexity = np.log10(features.get("unique_items", 1) + 1)
            sparsity_complexity = features.get("sparsity", 0.5) * 10
            
            return (user_complexity + item_complexity + sparsity_complexity) / 3
        except:
            return 1.0
    
    def _calculate_kg_richness(self, features: Dict) -> float:
        
        try:
            
            triplet_density = features.get("total_triplets", 0) / max(features.get("unique_items", 1), 1)
            relation_diversity = features.get("unique_relations", 0) / 100  
            
            return min(1.0, (triplet_density + relation_diversity) / 2)
        except:
            return 0.0
    
    def _calculate_recommendation_difficulty(self, features: Dict) -> float:
        
        try:
            
            sparsity_factor = features.get("sparsity", 0.5)
            activity_factor = features.get("user_activity_distribution", {}).get("activity_gini", 0.5)
            
            return (sparsity_factor + activity_factor) / 2
        except:
            return 0.5
    
    def _estimate_computational_demand(self, features: Dict) -> str:
       
        try:
            complexity = features.get("complexity_score", 1.0)
            scale = features.get("dataset_scale", "medium")
            total_interactions = features.get("total_interactions", 0)
            
            if scale == "very_large" or total_interactions > 10000000 or complexity > 5:
                return "high"
            elif scale == "large" or total_interactions > 1000000 or complexity > 3:
                return "medium"
            else:
                return "low"
        except:
            return "medium"
    
    def _calculate_data_quality_score(self, features: Dict) -> float:
        
        try:
           
            completeness = 1.0 - features.get("sparsity", 0.5)
            balance = 1.0 - features.get("user_activity_distribution", {}).get("activity_gini", 0.5)
            
            return (completeness + balance) / 2
        except:
            return 0.5
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        
        try:
            values = np.sort(values)
            n = len(values)
            cumsum = np.cumsum(values)
            return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        except:
            return 0.5
    
    def _calculate_entropy(self, values: np.ndarray) -> float:
        
        try:
            probabilities = values / np.sum(values)
            probabilities = probabilities[probabilities > 0]
            return -np.sum(probabilities * np.log2(probabilities))
        except:
            return 0.0
    
    def save_features(self, features: Dict, output_file: str = "dataset_features.json"):
        
        try:
            
            serializable_features = self._make_json_serializable(features)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_features, f, ensure_ascii=False, indent=2)
            print(f"Features saved to {output_file}")
        except Exception as e:
            print(f"Error saving features: {e}")
    
    def _make_json_serializable(self, obj):
        
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def load_features(self, input_file: str) -> Dict:
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                features = json.load(f)
            print(f" Features loaded from {input_file}")
            return features
        except Exception as e:
            print(f" Error loading features: {e}")
            return {}


def main():
    
    analyzer = DatasetAnalyzer()
    
    
    features = analyzer.analyze_dataset(
        ratings_file="../ml-1m/ratings.csv",
        kg_file="../ml-1m/processed_kg_id.tsv"
    )
    
    
    analyzer.save_features(features, "movie_dataset_features.json")
    
    
    print("\nKey Features:")
    print(f"   - Dataset Scale: {features.get('dataset_scale', 'unknown')}")
    print(f"   - Users: {features.get('unique_users', 0):,}")
    print(f"   - Items: {features.get('unique_items', 0):,}")
    print(f"   - Interactions: {features.get('total_interactions', 0):,}")
    print(f"   - KG Triplets: {features.get('total_triplets', 0):,}")
    print(f"   - Complexity Score: {features.get('complexity_score', 0):.2f}")
    print(f"   - KG Richness: {features.get('kg_richness', 0):.2f}")
    print(f"   - Computational Demand: {features.get('computational_demand', 'unknown')}")


if __name__ == "__main__":
    main()
