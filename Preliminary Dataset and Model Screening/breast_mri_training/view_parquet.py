"""
Script to view and inspect Parquet files
Displays data preview, schema, statistics, and allows interactive exploration
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def display_file_info(file_path, show_schema=True, show_stats=True):
    """
    Display basic information about a Parquet file
    
    Args:
        file_path: Path to Parquet file
        show_schema: Whether to show the schema
        show_stats: Whether to show basic statistics
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"\n{'='*80}")
    print(f"Parquet File: {file_path.name}")
    print(f"Path: {file_path.absolute()}")
    print(f"Size: {file_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"{'='*80}\n")
    
    # Read parquet file
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        raise ValueError(f"Error reading Parquet file: {e}")
    
    # Basic info
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB\n")
    
    # Schema
    if show_schema:
        print("Schema:")
        print("-" * 80)
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isna().sum()
            null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
            print(f"  {col:30s} {str(dtype):15s} (nulls: {null_count:,} ({null_pct:.1f}%))")
        print()
    
    # Data preview
    print("First 10 rows:")
    print("-" * 80)
    print(df.head(10).to_string())
    print()
    
    print("Last 5 rows:")
    print("-" * 80)
    print(df.tail(5).to_string())
    print()
    
    # Statistics
    if show_stats:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("Numeric columns statistics:")
            print("-" * 80)
            print(df[numeric_cols].describe().to_string())
            print()
        
        # Categorical columns info
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print("Categorical columns summary:")
            print("-" * 80)
            for col in categorical_cols[:5]:  # Limit to first 5 to avoid too much output
                unique_count = df[col].nunique()
                if unique_count <= 20:
                    print(f"\n{col} ({unique_count} unique values):")
                    print(df[col].value_counts().head(10).to_string())
                else:
                    print(f"{col}: {unique_count} unique values (showing top 10)")
                    print(df[col].value_counts().head(10).to_string())
            print()
    
    return df


def interactive_explore(df):
    """
    Interactive exploration mode
    
    Args:
        df: DataFrame to explore
    """
    print("\n" + "="*80)
    print("Interactive Mode")
    print("="*80)
    print("Commands:")
    print("  head [N]        - Show first N rows (default: 10)")
    print("  tail [N]        - Show last N rows (default: 10)")
    print("  cols            - List all columns")
    print("  info            - Show DataFrame info")
    print("  describe        - Show statistics")
    print("  col <name>      - Show details for a specific column")
    print("  filter <expr>   - Filter DataFrame (e.g., 'age > 30')")
    print("  save <path>     - Save filtered DataFrame to CSV")
    print("  q/quit          - Exit")
    print("="*80 + "\n")
    
    current_df = df.copy()
    
    while True:
        try:
            command = input("parquet> ").strip().split()
            if not command:
                continue
            
            cmd = command[0].lower()
            
            if cmd in ['q', 'quit', 'exit']:
                print("Exiting...")
                break
            
            elif cmd == 'head':
                n = int(command[1]) if len(command) > 1 else 10
                print(current_df.head(n).to_string())
            
            elif cmd == 'tail':
                n = int(command[1]) if len(command) > 1 else 10
                print(current_df.tail(n).to_string())
            
            elif cmd == 'cols':
                print("\nColumns:")
                for i, col in enumerate(current_df.columns, 1):
                    dtype = current_df[col].dtype
                    print(f"  {i:3d}. {col:30s} {str(dtype)}")
            
            elif cmd == 'info':
                print(current_df.info())
            
            elif cmd == 'describe':
                print(current_df.describe().to_string())
            
            elif cmd == 'col':
                if len(command) < 2:
                    print("Usage: col <column_name>")
                    continue
                col_name = command[1]
                if col_name not in current_df.columns:
                    print(f"Column '{col_name}' not found.")
                    print(f"Available columns: {list(current_df.columns)}")
                    continue
                
                print(f"\nColumn: {col_name}")
                print("-" * 80)
                print(f"Type: {current_df[col_name].dtype}")
                print(f"Nulls: {current_df[col_name].isna().sum()} ({(current_df[col_name].isna().sum()/len(current_df)*100):.1f}%)")
                
                if current_df[col_name].dtype in ['object', 'category']:
                    print(f"\nUnique values: {current_df[col_name].nunique()}")
                    print("\nValue counts:")
                    print(current_df[col_name].value_counts().head(20).to_string())
                else:
                    print(f"\nStatistics:")
                    print(current_df[col_name].describe().to_string())
            
            elif cmd == 'filter':
                if len(command) < 2:
                    print("Usage: filter <expression>")
                    print("Example: filter age > 30")
                    continue
                
                filter_expr = ' '.join(command[1:])
                try:
                    # Use eval with controlled namespace
                    filtered = current_df.query(filter_expr)
                    print(f"Filtered to {len(filtered)} rows (from {len(current_df)} rows)")
                    current_df = filtered
                except Exception as e:
                    print(f"Error in filter expression: {e}")
            
            elif cmd == 'save':
                if len(command) < 2:
                    print("Usage: save <output_path.csv>")
                    continue
                
                output_path = command[1]
                try:
                    current_df.to_csv(output_path, index=False)
                    print(f"Saved to {output_path}")
                except Exception as e:
                    print(f"Error saving: {e}")
            
            elif cmd == 'reset':
                current_df = df.copy()
                print("Reset to original DataFrame")
            
            else:
                print(f"Unknown command: {cmd}")
                print("Type a command or 'q' to quit")
            
            print()
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='View and inspect Parquet files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View parquet file
  python view_parquet.py data.parquet
  
  # View without statistics
  python view_parquet.py data.parquet --no-stats
  
  # Interactive mode
  python view_parquet.py data.parquet --interactive
  
  # Save to CSV
  python view_parquet.py data.parquet --to-csv output.csv
  
  # View only first N rows
  python view_parquet.py data.parquet --rows 50
        """
    )
    
    parser.add_argument('parquet_file', type=str,
                        help='Path to Parquet file')
    parser.add_argument('--no-schema', action='store_true',
                        help='Skip schema display')
    parser.add_argument('--no-stats', action='store_true',
                        help='Skip statistics display')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Enter interactive exploration mode')
    parser.add_argument('--to-csv', type=str, default=None,
                        help='Convert and save to CSV file')
    parser.add_argument('--rows', type=int, default=None,
                        help='Limit number of rows to display (0 = all)')
    parser.add_argument('--columns', type=str, nargs='+', default=None,
                        help='Show only specified columns')
    
    args = parser.parse_args()
    
    if not Path(args.parquet_file).exists():
        raise FileNotFoundError(f"Parquet file not found: {args.parquet_file}")
    
    # Load the file
    try:
        df = pd.read_parquet(args.parquet_file)
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        sys.exit(1)
    
    # Filter columns if specified
    if args.columns:
        missing_cols = [col for col in args.columns if col not in df.columns]
        if missing_cols:
            print(f"Warning: Columns not found: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
        df = df[args.columns]
    
    # Limit rows if specified
    if args.rows is not None and args.rows > 0:
        df = df.head(args.rows)
    
    # Save to CSV if requested
    if args.to_csv:
        df.to_csv(args.to_csv, index=False)
        print(f"Saved {len(df)} rows to {args.to_csv}")
        return
    
    # Display info
    display_file_info(
        args.parquet_file,
        show_schema=not args.no_schema,
        show_stats=not args.no_stats
    )
    
    # Interactive mode
    if args.interactive:
        interactive_explore(df)


if __name__ == '__main__':
    main()

