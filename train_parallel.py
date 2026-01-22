

import multiprocessing
import time
import argparse
import numpy as np
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

# Try importing torch at top level to verify main process
try:
    import torch
except ImportError:
    print("="*60)
    print("CRITICAL ERROR: Main process cannot import 'torch'.")
    print(f"Current Python: {sys.executable}")
    print(f"Python Version: {sys.version.split()[0]}")
    print("\nDIAGNOSIS: You might be running the script with a Python version")
    print("           that doesn't have PyTorch installed.")
    print("           (Check if you are using Python 3.13 instead of 3.12?)")
    print("="*60)
    sys.exit(1)

from train_memory_autopilot import train_memory_autopilot

def worker_task(seed):
    """Worker function to run a single training instance."""
    try:
        # Debugging: Ensure torch is importable here
        import torch
        # Each worker runs the training script
        # We assume GPU is shared. PyTorch handles this relatively well.
        result = train_memory_autopilot(seed=seed, quiet=True)
        return result
    except ImportError as e:
        return {
            'seed': seed, 
            'error': f"ImportError: {e}. Python: {sys.executable}. Path: {sys.path[:3]}...", 
            'success': False
        }
    except Exception as e:
        return {'seed': seed, 'error': str(e), 'success': False}

def main():
    # CRITICAL FIX: Ensure workers use the EXACT same python interpreter as the main process
    try:
        multiprocessing.set_executable(sys.executable)
    except AttributeError:
        pass # Not available on all platforms/versions, safe to ignore if failed

    parser = argparse.ArgumentParser(description='Parallel Verification for EvoNet Memory')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds to test')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: CPU count - 2)')
    args = parser.parse_args()

    # Hardware Info
    num_cpus = multiprocessing.cpu_count()
    if args.workers is None:
        num_workers = max(1, num_cpus - 2)
    else:
        num_workers = args.workers

    print("="*60)
    print(f"🚀 EVO-NET PARALLEL VERIFICATION SYSTEM (DIAGNOSTIC MODE)")
    print(f"   Interpreter: {sys.executable}")
    print(f"   CPU Cores:   {num_cpus}")
    print(f"   Workers:     {num_workers}")
    print(f"   Seeds:       {args.seeds}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU:         {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"   Note:        All workers will share this GPU.")
    else:
        print("   GPU:         Not detected (Running on CPU)")
    
    print("="*60)
    print("\nStarting parallel execution... (This may take a few minutes)\n")

    # Define seeds
    seeds = [42 + i for i in range(args.seeds)]

    start_time = time.time()
    
    # Use spawn context for compatibility with PyTorch/CUDA
    ctx = multiprocessing.get_context('spawn')
    
    with ctx.Pool(processes=num_workers) as pool:
        # Map tasks
        results = pool.map(worker_task, seeds)
    
    total_time = time.time() - start_time
    
    # Report Results
    print("\n" + "="*80)
    print(f"{'SEED':<6} | {'STATUS':<10} | {'PRE-CRASH':<10} | {'RECOVERED':<10} | {'PEAK':<10} | {'EPOCHS':<6}")
    print("-" * 80)
    
    success_count = 0
    pre_crash_scores = []
    post_recovery_scores = []
    
    for res in results:
        if 'error' in res:
            print(f"{res['seed']:<6} | {'ERROR':<10} | {res.get('error', 'Unknown')}")
            continue
            
        status = "SUCCESS" if res['success'] else "FAILED"
        if res['success']:
            success_count += 1
            
        pre_crash = res['pre_crash_best']
        post_rec = res['post_recovery_best']
        peak = res['peak_score']
        
        pre_crash_scores.append(pre_crash)
        post_recovery_scores.append(post_rec)
        
        print(f"{res['seed']:<6} | {status:<10} | {pre_crash:10.1f} | {post_rec:10.1f} | {peak:10.1f} | {res['epochs']:<6}")

    print("-" * 80)
    
    # Summary Utils
    def safe_mean(lst): return np.mean(lst) if lst else 0.0
    def safe_std(lst): return np.std(lst) if lst else 0.0
    
    print("\n📊 AGGREGATE STATISTICS")
    print(f"   Total Time:      {total_time:.1f}s")
    print(f"   Success Rate:    {success_count}/{len(seeds)} ({success_count/len(seeds)*100:.1f}%)")
    print(f"   Mean Pre-Crash:  {safe_mean(pre_crash_scores):.1f} ± {safe_std(pre_crash_scores):.1f}")
    print(f"   Mean Recovered:  {safe_mean(post_recovery_scores):.1f} ± {safe_std(post_recovery_scores):.1f}")
    
    print("="*80)

if __name__ == "__main__":
    # Windows support for multiprocessing
    multiprocessing.freeze_support()
    multiprocessing.set_executable(sys.executable)
    main()
