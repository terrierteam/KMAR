import os
import json
import argparse
import pandas as pd
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

class PromptPreparator:
    """Prepare prompts for item triplet analysis"""
    
    def __init__(self):
        """Initialize the Prompt Preparator"""
        # Load mapping files
        self.item_mapping = {}
        self.relation_mapping = {}
        self.entity_mapping = {}
        
        print("✅ Prompt Preparator initialized")
    
    def load_mapping_files(self, movie_info_file: str, relations_file: str, entities_file: str):
        """Load mapping files for ID to text conversion
        
        Args:
            movie_info_file: Path to movie_info.csv
            relations_file: Path to all_relations_id.tsv
            entities_file: Path to only_entity-id.tsv
        """
        print("Loading mapping files...")
        
        try:
            # Load movie info mapping (item_id -> movie_title)
            movie_df = pd.read_csv(movie_info_file, sep='|', header=None, 
                                 names=['movie_id', 'title', 'year', 'genres'],
                                 encoding='utf-8')
            self.item_mapping = dict(zip(movie_df['movie_id'], movie_df['title']))
            print(f"✅ Loaded {len(self.item_mapping)} item mappings")
            
            # Load relations mapping (relation_id -> relation_text)
            relations_df = pd.read_csv(relations_file, sep='\t', header=None,
                                    names=['relation_id', 'relation_text'])
            self.relation_mapping = dict(zip(relations_df['relation_id'], relations_df['relation_text']))
            print(f"✅ Loaded {len(self.relation_mapping)} relation mappings")
            
            # Load entities mapping (entity_id -> entity_text)
            entities_df = pd.read_csv(entities_file, sep='\t', header=None,
                                   names=['entity_id', 'entity_text1', 'entity_text2', 'entity_text3', 'entity_id_num'])
            self.entity_mapping = dict(zip(entities_df['entity_id_num'], entities_df['entity_text2']))  # Use English text
            print(f"✅ Loaded {len(self.entity_mapping)} entity mappings")
            
        except Exception as e:
            print(f"❌ Error loading mapping files: {e}")
            raise
    
    def load_kg_data(self, kg_file: str) -> pd.DataFrame:
        """Load KG data from TSV file
        
        Args:
            kg_file: Path to pretrain-output_kg_id.tsv
            
        Returns:
            DataFrame with columns: item, relation, entity
        """
        print(f"Loading KG data from {kg_file}...")
        
        try:
            df = pd.read_csv(kg_file, sep='\t', header=None, 
                           names=['item', 'relation', 'entity'])
            print(f"✅ Loaded {len(df)} triplets")
            return df
        except Exception as e:
            print(f"❌ Error loading KG data: {e}")
            raise
    
    def load_item_summaries(self, summaries_file: str) -> Dict[int, str]:
        """Load item summaries from JSON file
        
        Args:
            summaries_file: Path to item_summaries.json
            
        Returns:
            Dictionary mapping item_id to summary
        """
        print(f"Loading item summaries from {summaries_file}...")
        
        try:
            with open(summaries_file, 'r', encoding='utf-8') as f:
                summaries_data = json.load(f)
            
            summaries_dict = {}
            for item in summaries_data:
                summaries_dict[item['item_id']] = item['item_summary']
            
            print(f"✅ Loaded {len(summaries_dict)} item summaries")
            return summaries_dict
            
        except Exception as e:
            print(f"❌ Error loading item summaries: {e}")
            raise
    
    def convert_triplet_to_text(self, item_id: int, relation_id: int, entity_id: int) -> Tuple[str, str, str]:
        """Convert triplet IDs to text representations
        
        Args:
            item_id: Item ID
            relation_id: Relation ID
            entity_id: Entity ID
            
        Returns:
            Tuple of (item_text, relation_text, entity_text)
        """
        item_text = self.item_mapping.get(item_id, f"Unknown_Item_{item_id}")
        relation_text = self.relation_mapping.get(relation_id, f"Unknown_Relation_{relation_id}")
        entity_text = self.entity_mapping.get(entity_id, f"Unknown_Entity_{entity_id}")
        
        return item_text, relation_text, entity_text
    
    def prepare_item_data(self, item_id: int, triplets: List[Dict], 
                          item_summary: str) -> Dict:
        """Prepare basic item data for triplet analysis
        
        Args:
            item_id: Item ID
            triplets: List of available triplets
            item_summary: Item summary for context
            
        Returns:
            Dictionary containing basic item information and triplet data
        """
        # Create triplet representations
        triplet_texts = []
        for triplet in triplets:
            item_text, relation_text, entity_text = self.convert_triplet_to_text(
                triplet['item'], triplet['relation'], triplet['entity']
            )
            triplet_texts.append({
                'id_format': f"({triplet['item']}, {triplet['relation']}, {triplet['entity']})",
                'text_format': f"({item_text}, {relation_text}, {entity_text})",
                'original': triplet
            })
        
        return {
            'item_id': item_id,
            'item_text': self.item_mapping.get(item_id, f"Item_{item_id}"),
            'item_summary': item_summary,
            'triplet_texts': triplet_texts
        }
    

    
    def process_all_items(self, kg_data: pd.DataFrame, item_summaries: Dict[int, str],
                         max_items: Optional[int] = None, only_with_summaries: bool = True) -> List[Dict]:
        """Process all items to prepare basic data
        
        Args:
            kg_data: KG DataFrame
            item_summaries: Dictionary of item summaries
            max_items: Maximum number of items to process
            only_with_summaries: If True, only process items that have summaries
            
        Returns:
            List of prepared item data for each item
        """
        # Group triplets by item
        item_triplets = defaultdict(list)
        for _, row in kg_data.iterrows():
            item_triplets[row['item']].append({
                'item': int(row['item']),
                'relation': int(row['relation']),
                'entity': int(row['entity'])
            })
        
        # Get all items from summaries (not just from KG data)
        all_items = set(item_summaries.keys())
        items_with_triplets = set(item_triplets.keys())
        
        # Items without triplets
        items_without_triplets = all_items - items_with_triplets
        
        # Items with triplets but no summaries
        items_with_triplets_no_summaries = items_with_triplets - all_items
        
        print(f"📊 Data Analysis:")
        print(f"   Items in summaries: {len(all_items)}")
        print(f"   Items with triplets: {len(items_with_triplets)}")
        print(f"   Items with triplets but no summaries: {len(items_with_triplets_no_summaries)}")
        print(f"   Items with summaries but no triplets: {len(items_without_triplets)}")
        
        if only_with_summaries:
            print(f"🔍 Processing mode: Only items with summaries")
            # Only process items that have both triplets and summaries
            items_to_process = list(all_items & items_with_triplets)
            print(f"   Items to process: {len(items_to_process)}")
        else:
            print(f"🔍 Processing mode: All items with triplets")
            # Process all items with triplets
            items_to_process = list(items_with_triplets)
            print(f"   Items to process: {len(items_to_process)}")
        
        if max_items:
            items_to_process = items_to_process[:max_items]
            print(f"   Limited to: {len(items_to_process)} items")
        
        print(f"Preparing data for {len(items_to_process)} items...")
        
        results = []
        
        for i, item_id in enumerate(items_to_process):
            if i % 100 == 0:
                print(f"Processing item {i+1}/{len(items_to_process)}: {item_id}")
            
            try:
                triplets = item_triplets[item_id]
                item_summary = item_summaries.get(item_id, "No summary available")
                
                # Prepare basic data for this item
                result = self.prepare_item_data(item_id, triplets, item_summary)
                results.append(result)
                
            except Exception as e:
                print(f"Error processing item {item_id}: {e}")
                # Add error result
                results.append({
                    'item_id': int(item_id),
                    'error': str(e)
                })
        
        # Add items without triplets (for reference, but marked as no_triplets)
        if only_with_summaries and (not max_items or len(results) < max_items):
            remaining_slots = max_items - len(results) if max_items else len(items_without_triplets)
            items_to_add = list(items_without_triplets)[:remaining_slots]
            
            for item_id in items_to_add:
                item_summary = item_summaries.get(item_id, "No summary available")
                results.append({
                    'item_id': int(item_id),
                    'item_text': self.item_mapping.get(item_id, f"Item_{item_id}"),
                    'item_summary': item_summary,
                    'triplet_texts': [],
                    'no_triplets': True
                })
        
        print(f"✅ Prepared data for {len(results)} items")
        return results
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        else:
            return obj
    
    def save_data(self, results: List[Dict], output_file: str):
        """Save prepared data to output file
        
        Args:
            results: List of data preparation results
            output_file: Output file path
        """
        print(f"Saving data to {output_file}...")
        
        try:
            # Convert numpy types to Python native types
            converted_results = self._convert_numpy_types(results)
            
            # Save as JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(converted_results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Saved {len(results)} item data sets to {output_file}")
            
            # Also save as CSV for easy viewing
            csv_file = output_file.replace('.json', '.csv')
            csv_data = []
            for result in results:
                if 'error' not in result:
                    csv_data.append({
                        'item_id': int(result.get('item_id', '')),
                        'item_text': str(result.get('item_text', '')),
                        'triplet_count': int(len(result.get('triplet_texts', []))),
                        'has_summary': bool(result.get('item_summary')),
                        'no_triplets': bool(result.get('no_triplets', False))
                    })
                else:
                    csv_data.append({
                        'item_id': int(result.get('item_id', '')),
                        'item_text': 'ERROR',
                        'triplet_count': 0,
                        'has_summary': False,
                        'no_triplets': False
                    })
            
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"✅ Also saved as CSV: {csv_file}")
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            import traceback
            traceback.print_exc()
    
    def run_full_workflow(self, kg_file: str, summaries_file: str, output_file: str,
                          max_items: Optional[int] = None, only_with_summaries: bool = True):
        """Run the complete data preparation workflow
        
        Args:
            kg_file: Path to pretrain-output_kg_id.tsv
            summaries_file: Path to item_summaries.json
            output_file: Output file path
            max_items: Maximum number of items to process
            only_with_summaries: If True, only process items that have summaries
        """
        print("=" * 60)
        print("A2-2-1: Data Preparation Module")
        print("=" * 60)
        print(f"KG file: {kg_file}")
        print(f"Summaries file: {summaries_file}")
        print(f"Output file: {output_file}")
        print(f"Max items: {max_items if max_items else 'All items'}")
        print(f"Only with summaries: {only_with_summaries}")
        print()
        
        # Load mapping files
        self.load_mapping_files(
            '../ml1m/movie_info.csv',
            '../ml1m/all_relations_id.tsv',
            '../ml1m/only_entity-id.tsv'
        )
        
        # Load KG data
        kg_data = self.load_kg_data(kg_file)
        
        # Load item summaries
        item_summaries = self.load_item_summaries(summaries_file)
        
        # Process all items
        results = self.process_all_items(kg_data, item_summaries, max_items, only_with_summaries)
        
        # Save results
        self.save_data(results, output_file)
        
        print("\n" + "=" * 60)
        print("✅ Data Preparation Complete!")
        print("=" * 60)
        
        # Print summary statistics
        successful_items = [r for r in results if 'error' not in r]
        error_items = [r for r in results if 'error' in r]
        items_with_triplets = [r for r in successful_items if not r.get('no_triplets', False)]
        items_without_triplets = [r for r in successful_items if r.get('no_triplets', False)]
        
        print(f"\n📊 Summary:")
        print(f"   Total items processed: {len(results)}")
        print(f"   Successful: {len(successful_items)}")
        print(f"   Items with triplets: {len(items_with_triplets)}")
        print(f"   Items without triplets: {len(items_without_triplets)}")
        print(f"   Errors: {len(error_items)}")
        
        if items_with_triplets:
            avg_triplets = sum(len(r.get('triplet_texts', [])) for r in items_with_triplets) / len(items_with_triplets)
            print(f"   Average triplets per item (with triplets): {avg_triplets:.1f}")
        
        if error_items:
            print(f"\n❌ Items with errors:")
            for item in error_items[:5]:  # Show first 5 errors
                print(f"   Item {item['item_id']}: {item['error']}")
        
        if items_without_triplets:
            print(f"\n⚠️  Items without triplets (will be skipped in GPT analysis):")
            for item in items_without_triplets[:5]:  # Show first 5
                print(f"   Item {item['item_id']}: {item['item_text']}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Prepare basic data for item triplet analysis')
    parser.add_argument('--kg-file', type=str, default='pretrain-output_kg_id.tsv',
                       help='Path to pretrain-output_kg_id.tsv file')
    parser.add_argument('--summaries-file', type=str, default='item_summaries.json',
                       help='Path to item_summaries.json file')
    parser.add_argument('--output', type=str, default='prepared_data.json',
                       help='Output file path for prepared data')
    parser.add_argument('--max-items', type=int, default=None,
                       help='Maximum number of items to process (None for all)')
    parser.add_argument('--only-with-summaries', action='store_true', default=True,
                       help='Only process items that have summaries (default: True)')
    parser.add_argument('--all-items', action='store_true', default=False,
                       help='Process all items with triplets, even without summaries')
    
    args = parser.parse_args()
    
    # Determine processing mode
    only_with_summaries = args.only_with_summaries and not args.all_items
    
    # Create data preparator
    preparator = PromptPreparator()
    
    # Run workflow
    preparator.run_full_workflow(args.kg_file, args.summaries_file, args.output, 
                                args.max_items, only_with_summaries)

if __name__ == "__main__":
    main()
