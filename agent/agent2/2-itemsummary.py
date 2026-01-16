import os
import csv
import json
import argparse
import time
import random
import pandas as pd
from typing import List, Dict, Optional
from openai import OpenAI

class ItemSummaryGenerator:
    """Generate intelligent movie summaries using GPT-3.5"""
    
    def __init__(self, openai_api_key: str = None):
        """Initialize the Item Summary Generator
        
        Args:
            openai_api_key: OpenAI API key for GPT analysis
        """
        # Initialize OpenAI client
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            # Try to get from environment variable
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                print("⚠️  Warning: No OpenAI API key provided. GPT analysis will be disabled.")
                self.client = None
        
        # Rate limiting configuration
        self.base_delay = 0.5  # Base delay between requests (seconds)
        self.max_delay = 5.0   # Maximum delay between requests (seconds)
        self.requests_per_minute = 60  # Target requests per minute
        self.retry_attempts = 3  # Number of retry attempts for failed requests
        self.consecutive_failures = 0  # Track consecutive failures
        
        if self.client:
            print("✅ OpenAI client initialized for GPT analysis")
            print(f"📊 Rate limiting: {self.requests_per_minute} requests/minute")
            print(f"⏱️  Base delay: {self.base_delay}s, Max delay: {self.max_delay}s")
        else:
            print("⚠️  OpenAI client not available - using fallback summary method")
    
    def load_movie_info(self, movie_info_file: str) -> List[Dict]:
        """Load movie information from CSV file
        
        Args:
            movie_info_file: Path to movie_info.csv file
            
        Returns:
            List of movie dictionaries with id, name, year, genres
        """
        print(f"Loading movie information from {movie_info_file}...")
        
        movies = []
        try:
            # Try different encodings for movie_info.csv
            encodings = ['utf-8', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(movie_info_file, sep='|', header=None, 
                                   names=['movie_id', 'title', 'year', 'genres'],
                                   encoding=encoding)
                    print(f"✅ Successfully loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read movie_info.csv with any encoding")
            
            # Convert to list of dictionaries
            for _, row in df.iterrows():
                movie = {
                    'movie_id': int(row['movie_id']),
                    'title': str(row['title']).strip(),
                    'year': str(row['year']).strip() if pd.notna(row['year']) else 'Unknown',
                    'genres': str(row['genres']).strip() if pd.notna(row['genres']) else 'Unknown'
                }
                movies.append(movie)
            
            print(f"✅ Loaded {len(movies)} movies")
            return movies
            
        except Exception as e:
            print(f"❌ Error loading movie information: {e}")
            return []
    
    def _smart_delay(self):
        """Implement smart delay with exponential backoff"""
        if self.consecutive_failures == 0:
            # Normal operation - use base delay with small random variation
            delay = self.base_delay + random.uniform(0, 0.2)
        else:
            # Exponential backoff with jitter
            delay = min(self.base_delay * (2 ** self.consecutive_failures), self.max_delay)
            delay += random.uniform(0, delay * 0.1)  # Add jitter
        
        time.sleep(delay)
    
    def _handle_api_error(self, error: Exception) -> bool:
        """Handle API errors and determine if retry is needed
        
        Args:
            error: The exception that occurred
            
        Returns:
            True if retry is recommended, False otherwise
        """
        error_str = str(error).lower()
        
        # Rate limit errors - definitely retry
        if any(phrase in error_str for phrase in ['rate limit', 'too many requests', 'quota exceeded']):
            print(f"⚠️  Rate limit hit, increasing delay...")
            self.consecutive_failures += 1
            return True
        
        # Server errors - retry with backoff
        elif any(phrase in error_str for phrase in ['server error', 'internal error', 'service unavailable']):
            print(f"⚠️  Server error, will retry...")
            self.consecutive_failures += 1
            return True
        
        # Authentication errors - don't retry
        elif any(phrase in error_str for phrase in ['authentication', 'unauthorized', 'invalid api key']):
            print(f"❌ Authentication error, skipping...")
            return False
        
        # Other errors - retry once
        else:
            print(f"⚠️  Unexpected error, will retry once...")
            self.consecutive_failures += 1
            return True
    
    def generate_movie_summary(self, movie: Dict) -> str:
        """Generate a summary for a single movie using GPT-3.5
        
        Args:
            movie: Movie dictionary with id, title, year, genres
            
        Returns:
            Generated summary string
        """
        if not self.client:
            return self._generate_fallback_summary(movie)
        
        for attempt in range(self.retry_attempts):
            try:
                # Build prompt for GPT
                prompt = self._build_summary_prompt(movie)
                
                # Call GPT
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a movie expert. Generate concise, informative summaries that capture the essence and appeal of movies."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.3
                )
                
                summary = response.choices[0].message.content.strip()
                
                # Success - reset failure counter and apply smart delay
                if self.consecutive_failures > 0:
                    print(f"✅ Recovered from {self.consecutive_failures} consecutive failures")
                    self.consecutive_failures = 0
                
                self._smart_delay()
                return summary
                
            except Exception as e:
                print(f"GPT analysis attempt {attempt + 1} failed for movie {movie['movie_id']}: {e}")
                
                if attempt < self.retry_attempts - 1 and self._handle_api_error(e):
                    print(f"Retrying in {self.base_delay * (2 ** self.consecutive_failures):.1f}s...")
                    self._smart_delay()
                    continue
                else:
                    print(f"All retry attempts failed, using fallback method")
                    break
        
        # All attempts failed, use fallback
        return self._generate_fallback_summary(movie)
    
    def _build_summary_prompt(self, movie: Dict) -> str:
        """Build prompt for GPT to generate movie summary"""
        prompt = f"""Generate a concise and informative summary for the following movie:

Movie ID: {movie['movie_id']}
Title: {movie['title']}
Year: {movie['year']}
Genres: {movie['genres']}

Please provide a 2-3 sentence summary that:
1. Describes what the movie is about
2. Highlights its main appeal or unique characteristics
3. Mentions key genre elements or themes
4. Is engaging and informative for recommendation purposes

Summary:"""
        return prompt
    
    def _generate_fallback_summary(self, movie: Dict) -> str:
        """Generate a fallback summary when GPT is not available"""
        year = movie['year'] if movie['year'] != 'Unknown' else 'unknown year'
        genres = movie['genres'] if movie['genres'] != 'Unknown' else 'various genres'
        
        return f"A {year} film in the {genres} genre. {movie['title']} offers an engaging cinematic experience with its unique storytelling and thematic elements."
    
    def generate_all_summaries(self, movies: List[Dict], max_movies: Optional[int] = None) -> List[Dict]:
        """Generate summaries for all movies
        
        Args:
            movies: List of movie dictionaries
            max_movies: Maximum number of movies to process (None for all)
            
        Returns:
            List of movie dictionaries with added summary field
        """
        if max_movies:
            movies = movies[:max_movies]
        
        total_movies = len(movies)
        print(f"Generating summaries for {total_movies} movies...")
        
        # Estimate processing time
        estimated_time_per_movie = (self.base_delay + 0.1)  # Base delay + API call time
        estimated_total_time = total_movies * estimated_time_per_movie / 60  # Convert to minutes
        print(f"⏱️  Estimated time: {estimated_total_time:.1f} minutes")
        print(f"📊 Target rate: {self.requests_per_minute} requests/minute")
        
        movies_with_summaries = []
        start_time = time.time()
        
        for i, movie in enumerate(movies):
            # Progress update with time estimation
            if i % 10 == 0 or i == total_movies - 1:
                elapsed_time = time.time() - start_time
                if i > 0:
                    avg_time_per_movie = elapsed_time / i
                    remaining_movies = total_movies - i
                    estimated_remaining = remaining_movies * avg_time_per_movie / 60
                    print(f"Processing movie {i+1}/{total_movies}: {movie['title']}")
                    print(f"   ⏱️  Elapsed: {elapsed_time/60:.1f}m, Remaining: {estimated_remaining:.1f}m")
                else:
                    print(f"Processing movie {i+1}/{total_movies}: {movie['title']}")
            
            try:
                summary = self.generate_movie_summary(movie)
                
                movie_with_summary = {
                    'item_id': movie['movie_id'],
                    'item_name': movie['title'],
                    'item_summary': summary,
                    'year': movie['year'],
                    'genres': movie['genres']
                }
                
                movies_with_summaries.append(movie_with_summary)
                
            except Exception as e:
                print(f"Error generating summary for movie {movie['movie_id']}: {e}")
                # Add movie with fallback summary
                movie_with_summary = {
                    'item_id': movie['movie_id'],
                    'item_name': movie['title'],
                    'item_summary': self._generate_fallback_summary(movie),
                    'year': movie['year'],
                    'genres': movie['genres']
                }
                movies_with_summaries.append(movie_with_summary)
        
        total_time = time.time() - start_time
        print(f"✅ Generated summaries for {len(movies_with_summaries)} movies")
        print(f"⏱️  Total processing time: {total_time/60:.1f} minutes")
        print(f"📊 Average time per movie: {total_time/total_movies:.2f} seconds")
        
        return movies_with_summaries
    
    def save_summaries(self, movies_with_summaries: List[Dict], output_file: str):
        """Save movie summaries to output file
        
        Args:
            movies_with_summaries: List of movies with summaries
            output_file: Output file path
        """
        print(f"Saving summaries to {output_file}...")
        
        try:
            # Save as JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(movies_with_summaries, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Saved {len(movies_with_summaries)} movie summaries to {output_file}")
            
            # Also save as CSV for easy viewing
            csv_file = output_file.replace('.json', '.csv')
            df = pd.DataFrame(movies_with_summaries)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"✅ Also saved as CSV: {csv_file}")
            
        except Exception as e:
            print(f"❌ Error saving summaries: {e}")
    
    def run_full_workflow(self, movie_info_file: str, output_file: str, 
                         max_movies: Optional[int] = None):
        """Run the complete workflow
        
        Args:
            movie_info_file: Path to movie_info.csv
            output_file: Output file path
            max_movies: Maximum number of movies to process
        """
        print("=" * 60)
        print("A2-1: Item Summary Generation Module")
        print("=" * 60)
        print(f"Movie info file: {movie_info_file}")
        print(f"Output file: {output_file}")
        print(f"Max movies: {max_movies if max_movies else 'All movies'}")
        print(f"OpenAI API: {'Available' if self.client else 'Not available'}")
        print()
        
        # Load movie information
        movies = self.load_movie_info(movie_info_file)
        if not movies:
            print("❌ No movies loaded. Exiting.")
            return
        
        # Generate summaries
        movies_with_summaries = self.generate_all_summaries(movies, max_movies)
        
        # Save results
        self.save_summaries(movies_with_summaries, output_file)
        
        print("\n" + "=" * 60)
        print("✅ Item Summary Generation Complete!")
        print("=" * 60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate movie summaries using GPT-3.5')
    parser.add_argument('--movie-info', type=str, default='../电影数据集/movie_info.csv',
                       help='Path to movie_info.csv file')
    parser.add_argument('--output', type=str, default='item_summaries.json',
                       help='Output file path for summaries')
    parser.add_argument('--max-movies', type=int, default=None,
                       help='Maximum number of movies to process (None for all)')
    parser.add_argument('--openai-api-key', type=str, default=None,
                       help='OpenAI API key for GPT analysis')
    parser.add_argument('--base-delay', type=float, default=0.5,
                       help='Base delay between requests in seconds (default: 0.5)')
    parser.add_argument('--max-delay', type=float, default=5.0,
                       help='Maximum delay between requests in seconds (default: 5.0)')
    parser.add_argument('--requests-per-minute', type=int, default=60,
                       help='Target requests per minute (default: 60)')
    
    args = parser.parse_args()
    
    # Create summary generator
    generator = ItemSummaryGenerator(args.openai_api_key)
    
    # Override default rate limiting if specified
    if args.base_delay != 0.5:
        generator.base_delay = args.base_delay
        print(f"📊 Custom base delay: {generator.base_delay}s")
    if args.max_delay != 5.0:
        generator.max_delay = args.max_delay
        print(f"📊 Custom max delay: {generator.max_delay}s")
    if args.requests_per_minute != 60:
        generator.requests_per_minute = args.requests_per_minute
        print(f"📊 Custom rate limit: {generator.requests_per_minute} requests/minute")
    
    # Run workflow
    generator.run_full_workflow(args.movie_info, args.output, args.max_movies)

if __name__ == "__main__":
    main()
