#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User Data Filter Script - Filter train_set.txt based on selected user IDs
"""

import pandas as pd
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Filter train_set.txt based on selected user IDs')
    parser.add_argument('--user-file', type=str, required=True,
                       help='File containing selected user IDs (one per line)')
    parser.add_argument('--train-file', type=str, default='../dataset/ml1m/train_set.txt',
                       help='Training data file path')
    parser.add_argument('--output-file', type=str, default='filtered_train_set.txt',
                       help='Output filtered training data file path')
    return parser.parse_args()

def load_user_ids(user_file):
    """Load user IDs from the specified file"""
    print(f"Loading user IDs from {user_file}...")
    
    try:
        with open(user_file, 'r', encoding='utf-8') as f:
            user_ids = [int(line.strip()) for line in f if line.strip()]
        
        print(f"Loaded {len(user_ids)} user IDs")
        print(f"User ID range: {min(user_ids)} - {max(user_ids)}")
        return set(user_ids)
        
    except FileNotFoundError:
        print(f"Error: User file '{user_file}' not found")
        return None
    except ValueError as e:
        print(f"Error: Invalid user ID format in {user_file}: {e}")
        return None
    except Exception as e:
        print(f"Error loading user file: {e}")
        return None

def load_train_data(train_file):
    """Load training data from the specified file"""
    print(f"Loading training data from {train_file}...")
    
    try:
        # Try different encodings for the train file
        encodings = ['utf-8', 'latin-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(train_file, names=['u', 'i', 'r', 't'], sep=' ', encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print(f"Error: Could not read {train_file} with any encoding")
            return None
        
        print(f"Loaded {len(df)} training records")
        print(f"Unique users in training data: {df['u'].nunique()}")
        print(f"User ID range in training data: {df['u'].min()} - {df['u'].max()}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Training file '{train_file}' not found")
        return None
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None

def filter_train_data(df, user_ids):
    """Filter training data to include only selected users"""
    print("Filtering training data...")
    
    # Filter the dataframe
    filtered_df = df[df['u'].isin(user_ids)]
    
    print(f"Filtered to {len(filtered_df)} records")
    print(f"Unique users in filtered data: {filtered_df['u'].nunique()}")
    
    # Check if all selected users are present
    missing_users = user_ids - set(filtered_df['u'].unique())
    if missing_users:
        print(f"Warning: {len(missing_users)} selected users not found in training data")
        print(f"Missing user IDs: {sorted(list(missing_users))[:10]}{'...' if len(missing_users) > 10 else ''}")
    else:
        print("All selected users found in training data")
    
    return filtered_df

def save_filtered_data(df, output_file):
    """Save filtered data to output file"""
    print(f"Saving filtered data to {output_file}...")
    
    try:
        # Save in the same format as the original train_set.txt
        df.to_csv(output_file, sep=' ', index=False, header=False)
        
        print(f"Successfully saved {len(df)} records to {output_file}")
        
        # Print some statistics
        print("\nOutput Statistics:")
        print(f"Total records: {len(df)}")
        print(f"Unique users: {df['u'].nunique()}")
        print(f"Unique items: {df['i'].nunique()}")
        print(f"Average rating: {df['r'].mean():.2f}")
        print(f"Rating range: {df['r'].min()} - {df['r'].max()}")
        
        # Show user interaction distribution
        user_interactions = df.groupby('u').size()
        print(f"Average interactions per user: {user_interactions.mean():.2f}")
        print(f"Min interactions per user: {user_interactions.min()}")
        print(f"Max interactions per user: {user_interactions.max()}")
        
    except Exception as e:
        print(f"Error saving filtered data: {e}")
        return False
    
    return True

def main():
    args = parse_args()
    
    print("=" * 60)
    print("User Data Filter Script")
    print("=" * 60)
    print(f"User file: {args.user_file}")
    print(f"Training file: {args.train_file}")
    print(f"Output file: {args.output_file}")
    print()
    
    # Check if input files exist
    if not os.path.exists(args.user_file):
        print(f"Error: User file '{args.user_file}' does not exist")
        return
    
    if not os.path.exists(args.train_file):
        print(f"Error: Training file '{args.train_file}' does not exist")
        return
    
    # Load user IDs
    user_ids = load_user_ids(args.user_file)
    if user_ids is None:
        return
    
    # Load training data
    train_df = load_train_data(args.train_file)
    if train_df is None:
        return
    
    # Filter training data
    filtered_df = filter_train_data(train_df, user_ids)
    
    # Save filtered data
    success = save_filtered_data(filtered_df, args.output_file)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Filtering completed successfully!")
        print(f"📁 Output file: {args.output_file}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Filtering failed!")
        print("=" * 60)

if __name__ == "__main__":
    main()
