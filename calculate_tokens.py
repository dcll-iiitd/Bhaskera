import os
import glob
import pyarrow.parquet as pq
import argparse

def count_tokens_in_cache(cache_dir_path):
    """
    Calculates the total number of tokens by reading the parquet files in the Bhaskera cache.
    """
    if not os.path.isdir(cache_dir_path):
        print(f"Error: Directory '{cache_dir_path}' does not exist.")
        return

    # Find all parquet files in the given directory and its subdirectories
    parquet_files = glob.glob(os.path.join(cache_dir_path, "**/*.parquet"), recursive=True)
    
    if not parquet_files:
        print(f"No parquet files found in {cache_dir_path}")
        print("Make sure you point directly to the tokenized cache folder (e.g., /mnt/disk1/slakshna/training/Bhaskera/cache/local_train_<hash>)")
        return
    
    total_tokens = 0
    total_rows = 0
    
    print(f"Found {len(parquet_files)} parquet files. Calculating tokens...")
    
    for file_path in parquet_files:
        try:
            # We only need 'input_ids' to count tokens, skipping other columns saves memory/time
            table = pq.read_table(file_path, columns=['input_ids'])
            
            # Using PyArrow's ListArray flattening is extremely fast
            for chunk in table.column('input_ids').iterchunks():
                # chunk.values gives the flattened 1D array of all tokens
                total_tokens += len(chunk.values)
                total_rows += len(chunk)
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    print("-" * 40)
    print(f"Total Sequences (Rows): {total_rows:,}")
    print(f"Total Tokens:           {total_tokens:,}")
    if total_rows > 0:
        print(f"Average tokens/row:     {total_tokens / total_rows:.1f}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate total tokens in a Bhaskera parquet cache")
    parser.add_argument(
        "cache_path", 
        type=str, 
        help="Path to the specific cache directory (e.g., /mnt/disk1/slakshna/training/Bhaskera/cache/local_train_abcd123)"
    )
    
    args = parser.parse_args()
    count_tokens_in_cache(args.cache_path)
