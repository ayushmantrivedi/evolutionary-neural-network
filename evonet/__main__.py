"""
EvoNet CLI Entry Point

Allows running the package as: python -m evonet
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evonet.utils.logging import setup_logging


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='EvoNet - Evolutionary Neural Network',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evonet --dataset iris
  python -m evonet --dataset cancer --output ./results
  python -m evonet --interactive

Supported datasets:
  - housing: California Housing (regression)
  - cancer: Breast Cancer (binary classification)
  - iris: Iris (multi-class)
  - wine: Wine (multi-class)
  - digits: Digits (multi-class)
  - Or path to .csv / .json file
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        help='Dataset name or path to file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for plots'
    )
    parser.add_argument(
        '--training-method', '-t',
        type=int,
        choices=[1, 2, 3],
        default=2,
        help='Training method: 1=Standard, 2=MiniBatch, 3=EarlyStopping'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    import logging
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    
    # Set output directory if provided
    if args.output:
        os.environ['EVO_OUTPUT_DIR'] = args.output
    
    # Interactive mode or dataset specified
    if args.interactive or not args.dataset:
        # Import and run the original hope.py interactive mode
        print("Running in interactive mode...")
        print("Use 'python hope.py' for the full interactive experience.")
        print("Or specify --dataset to train directly from CLI.")
    else:
        print(f"Dataset: {args.dataset}")
        print(f"Training method: {args.training_method}")
        print("Training functionality will be fully integrated in next update.")


if __name__ == '__main__':
    main()
