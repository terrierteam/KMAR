import sys
import random
import pandas as pd
import numpy as np
import pickle
import argparse
from sklearn.cluster import KMeans, DBSCAN
from collections import Counter
import math

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Select users for training using different sampling methods')
    parser.add_argument('--sample-method', type=str, default='random',
                      choices=['random', 'kmeans', 'dbscan'],
                      help='Sampling method: random, kmeans, or dbscan')
    parser.add_argument('--sample-size', type=int, default=1000,
                      help='Number of users to select')
    parser.add_argument('--output-file', type=str, default='1-selected_user.txt',
                      help='Output file path')
    return parser.parse_args()

def load_data():
    """Load training data and movie information"""
    print("Loading data...")
    
    # Load training data
    try:
        df_like = pd.read_csv('../dataset/ml1m/train_set.txt', names=['u', 'i', 'r', 't'], sep=' ')
        print(f"Loaded training data: {len(df_like)} records")
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None, None
    
    # Load movie information
    try:
        movie_info = pd.read_csv('../dataset/ml1m/movie_info.csv', header=None, sep='|',
                                names=['movie_id', 'movie_name', 'release_date', 'genre'],
                                encoding='utf-8')
        print(f"Loaded movie info: {len(movie_info)} movies")
    except Exception as e:
        print(f"Error loading movie info: {e}")
        movie_info = None
    
    return df_like, movie_info

def load_embeddings():
    """Load user and item embeddings if available"""
    try:
        # Try to load embeddings from initial-ranking folder
        with open('user.pkl', "rb") as file:
            cm_user_emb = pickle.load(file)
        with open('item.pkl', "rb") as file:
            cm_item = pickle.load(file)
        with open('user_id_mapping.pkl', "rb") as file:
            mf_user = pickle.load(file)
        with open('item_id_mapping.pkl', "rb") as file:
            mf_item = pickle.load(file)
        
        print(f"Loaded embeddings: {len(cm_user_emb)} users, {len(cm_item)} items")
        return cm_user_emb, cm_item, mf_user, mf_item
    except Exception as e:
        print(f"Warning: Could not load embeddings: {e}")
        return None, None, None, None

def random_sampling(user_list, sample_size):
    """Random sampling method - EXACTLY SAME AS make-train-random.py"""
    print("Using random sampling method...")
    sample_list = []
    for i in range(sample_size):
        sample_ = random.sample(user_list, 1)[0]
        sample_list.append(sample_)
    return sample_list

def kmeans_sampling(user_list, cm_user_emb, mf_user, sample_size, df_like):
    """KMeans clustering sampling method - EXACTLY SAME AS make-train-random.py"""
    print("Using KMeans clustering sampling method...")
    
    if cm_user_emb is None:
        print("Warning: No embeddings available, falling back to random sampling")
        return random_sampling(user_list, sample_size)
    
    # EXACTLY THE SAME LOGIC AS make-train-random.py
    sample_list1 = []
    sample_list2 = []
    
    sample_imp = int(sample_size * 0.6)
    
    # Step 1: Prepare embedding data (EXACTLY SAME)
    user_ids = sorted(cm_user_emb.keys(), key=int)
    cm_user_emb_matrix = np.array([cm_user_emb[user] for user in user_ids])
    
    # FIXED: Ensure consistent data types for weight calculation
    weights = []
    for user in user_ids:
        try:
            # Convert user to int for comparison with df_like
            user_int = int(user)
            user_count = len(df_like[df_like['u'] == user_int])
            weights.append(math.log(user_count) if user_count > 0 else 0)
        except (ValueError, TypeError):
            weights.append(0)
    
    print(f"Step 1: Weighted random sampling {sample_imp} users...")
    # Step 2: Weighted random sampling (EXACTLY SAME)
    for i in range(sample_imp):
        sample_ = random.choices(user_list, weights, k=1)[0]
        sample_list1.append(sample_)
    
    print(f"Step 2: KMeans clustering for remaining users...")
    # Step 3: KMeans clustering (EXACTLY SAME)
    kmeans = KMeans(n_clusters=10, random_state=0).fit(cm_user_emb_matrix)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    samples_per_cluster = np.round(counts / counts.sum() * sample_imp).astype(int)
    
    sampled_ids = []
    for cluster_id, samples in enumerate(samples_per_cluster):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_users = [user_ids[i] for i in cluster_indices]
        sampled_ids.extend(np.random.choice(cluster_users, samples, replace=True))
    
    print(f"Step 3: Mapping user IDs...")
    # Step 4: ID mapping (EXACTLY SAME)
    mf_user_i = {v: k for k, v in mf_user.items()}
    
    # FIXED: Handle ID mapping with proper type conversion
    mapped_ids = []
    for sampled_id in sampled_ids:
        try:
            if isinstance(sampled_id, int) and isinstance(user_ids[0], str):
                # Find the original user ID from embedding ID
                if sampled_id in mf_user_i:
                    mapped_ids.append(mf_user_i[sampled_id])
            else:
                # Direct mapping
                if sampled_id in mf_user:
                    mapped_ids.append(sampled_id)
        except Exception as e:
            print(f"Warning: Could not map user ID {sampled_id}: {e}")
            continue
    
    sample_list1.extend(mapped_ids)
    
    print(f"Step 4: Frequency-based weighting...")
    # Step 5: Frequency-based weighting (EXACTLY SAME)
    occurrences = Counter(sample_list1)
    t_occurrences = {element: 0.95 ** (count - 1) for element, count in occurrences.items()}
    sample_list2 = [t_occurrences[_] for _ in sample_list1]
    
    sample_list = random.choices(sample_list1, weights=sample_list2, k=sample_size)
    print(f"KMeans sampling completed: {len(sample_list)} users selected")
    return sample_list

def dbscan_sampling(user_list, cm_user_emb, mf_user, sample_size, df_like):
    """DBSCAN clustering sampling method - EXACTLY SAME AS make-train-random.py"""
    print("Using DBSCAN clustering sampling method...")
    
    if cm_user_emb is None:
        print("Warning: No embeddings available, falling back to random sampling")
        return random_sampling(user_list, sample_size)
    
    # EXACTLY THE SAME LOGIC AS make-train-random.py
    sample_list1 = []
    sample_list2 = []
    
    sample_imp = int(sample_size * 0.6)
    
    # Step 1: Prepare embedding data (EXACTLY SAME)
    user_ids = sorted(cm_user_emb.keys(), key=int)
    cm_user_emb_matrix = np.array([cm_user_emb[user] for user in user_ids])
    
    # FIXED: Ensure consistent data types for weight calculation
    weights = []
    for user in user_ids:
        try:
            user_int = int(user)
            user_count = len(df_like[df_like['u'] == user_int])
            weights.append(math.log(user_count) if user_count > 0 else 0)
        except (ValueError, TypeError):
            weights.append(0)
    
    print(f"Step 1: Weighted random sampling {sample_imp} users...")
    # Step 2: Weighted random sampling (EXACTLY SAME)
    for i in range(sample_imp):
        sample_ = random.choices(user_list, weights, k=1)[0]
        sample_list1.append(sample_)
    
    print(f"Step 2: DBSCAN clustering for remaining users...")
    # Step 3: DBSCAN clustering (EXACTLY SAME)
    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean', n_jobs=-1)
    labels = dbscan.fit_predict(cm_user_emb_matrix)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_clusters = unique_labels[unique_labels != -1]
    valid_counts = counts[unique_labels != -1]
    
    if len(valid_clusters) > 0:
        samples_per_cluster = np.round(valid_counts / valid_counts.sum() * sample_imp).astype(int)
        
        sampled_ids = []
        for cluster_id, samples in zip(valid_clusters, samples_per_cluster):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_users = [user_ids[i] for i in cluster_indices]
            sampled_ids.extend(np.random.choice(cluster_users, samples, replace=True))
    else:
        print("Warning: No valid clusters found by DBSCAN, using only weighted random sampling")
        sampled_ids = []
    
    print(f"Step 3: Mapping user IDs...")
    # Step 4: ID mapping (EXACTLY SAME)
    mf_user_i = {v: k for k, v in mf_user.items()}
    
    # FIXED: Handle ID mapping with proper type conversion (EXACTLY SAME AS make-train-random.py)
    mapped_ids = []
    for sampled_id in sampled_ids:
        try:
            if isinstance(sampled_id, int) and isinstance(user_ids[0], str):
                # Find the original user ID from embedding ID
                if sampled_id in mf_user_i:
                    mapped_ids.append(mf_user_i[sampled_id])
            else:
                # Direct mapping
                if sampled_id in mf_user:
                    mapped_ids.append(sampled_id)
        except Exception as e:
            print(f"Warning: Could not map user ID {sampled_id}: {e}")
            continue
    
    sample_list1.extend(mapped_ids)
    
    print(f"Step 4: Frequency-based weighting...")
    # Step 5: Frequency-based weighting (EXACTLY SAME)
    occurrences = Counter(sample_list1)
    t_occurrences = {element: 0.95 ** (count - 1) for element, count in occurrences.items()}
    sample_list2 = [t_occurrences[_] for _ in sample_list1]
    
    sample_list = random.choices(sample_list1, weights=sample_list2, k=sample_size)
    print(f"DBSCAN sampling completed: {len(sample_list)} users selected")
    return sample_list

def main():
    args = parse_args()
    
    print(f"User Selection Script (ORIGINAL KMEANS METHOD - FIXED)")
    print(f"Method: {args.sample_method}")
    print(f"Sample size: {args.sample_size}")
    print(f"Output file: {args.output_file}")
    
    # Load data
    df_like, movie_info = load_data()
    if df_like is None:
        print("Failed to load data. Exiting.")
        return
    
    # Load embeddings
    cm_user_emb, cm_item, mf_user, mf_item = load_embeddings()
    
    # Get unique users
    user_list = list(df_like['u'].unique())
    print(f"Total unique users: {len(user_list)}")
    
    # Select users based on method
    if args.sample_method == 'random':
        selected_users = random_sampling(user_list, args.sample_size)
    elif args.sample_method == 'kmeans':
        selected_users = kmeans_sampling(user_list, cm_user_emb, mf_user, args.sample_size, df_like)
    elif args.sample_method == 'dbscan':
        selected_users = dbscan_sampling(user_list, cm_user_emb, mf_user, args.sample_size, df_like)
    else:
        print(f"Unknown sampling method: {args.sample_method}")
        return
    
    # FIXED: Convert all user IDs to int to ensure consistent data types
    try:
        selected_users = [int(user_id) for user_id in selected_users]
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not convert some user IDs to int: {e}")
        # Filter out non-convertible IDs
        selected_users = []
        for user_id in selected_users:
            try:
                selected_users.append(int(user_id))
            except (ValueError, TypeError):
                continue
    
    # Remove duplicates and sort
    selected_users = sorted(list(set(selected_users)))
    
    # Ensure we have enough users
    if len(selected_users) < args.sample_size:
        print(f"Warning: Only {len(selected_users)} unique users selected, need {args.sample_size}")
        # Add more users if needed
        remaining_users = [u for u in user_list if u not in selected_users]
        additional_users = random.sample(remaining_users, min(args.sample_size - len(selected_users), len(remaining_users)))
        selected_users.extend(additional_users)
        selected_users = sorted(selected_users)
    
    # Take only the required number of users
    selected_users = selected_users[:args.sample_size]
    
    print(f"Selected {len(selected_users)} users")
    
    # Save to file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for user_id in selected_users:
            f.write(f"{user_id}\n")
    
    print(f"Selected users saved to {args.output_file}")
    
    # Print some statistics
    print("\nSelection Statistics:")
    print(f"Total users selected: {len(selected_users)}")
    print(f"Unique users: {len(set(selected_users))}")
    print(f"User ID range: {min(selected_users)} - {max(selected_users)}")
    
    # Check user interaction counts
    interaction_counts = []
    for user_id in selected_users:
        count = len(df_like[df_like['u'] == user_id])
        interaction_counts.append(count)
    
    print(f"Average interactions per user: {np.mean(interaction_counts):.2f}")
    print(f"Min interactions: {min(interaction_counts)}")
    print(f"Max interactions: {max(interaction_counts)}")

if __name__ == "__main__":
    main()
