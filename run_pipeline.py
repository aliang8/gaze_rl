import os
import subprocess
import time

def run_pipeline():
    """Run the entire gaze-informed RL pipeline."""
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/hierarchical', exist_ok=True)
    os.makedirs('results/hierarchical/bayesian', exist_ok=True)
    
    # Start timer
    total_start_time = time.time()
    
    # Step 1: Collect multimodal trajectories
    print("\n" + "="*80)
    print("STEP 1: Collecting multimodal trajectories")
    print("="*80)
    start_time = time.time()
    subprocess.run(['python', 'data_collector.py'])
    print(f"Data collection completed in {time.time() - start_time:.2f} seconds")
    
    # Step 2: Generate gaze data
    print("\n" + "="*80)
    print("STEP 2: Generating simulated gaze data")
    print("="*80)
    start_time = time.time()
    subprocess.run(['python', 'gaze_simulator.py'])
    print(f"Gaze simulation completed in {time.time() - start_time:.2f} seconds")
    
    # Step 3: Train BC policy without gaze
    print("\n" + "="*80)
    print("STEP 3: Training BC policy WITHOUT gaze information")
    print("="*80)
    start_time = time.time()
    subprocess.run(['python', 'train_bc.py'])
    print(f"BC training completed in {time.time() - start_time:.2f} seconds")
    
    # Step 4: Train BC policy with gaze
    print("\n" + "="*80)
    print("STEP 4: Training BC policy WITH gaze information")
    print("="*80)
    start_time = time.time()
    subprocess.run(['python', 'train_bc_with_gaze.py'])
    print(f"BC with gaze training completed in {time.time() - start_time:.2f} seconds")
    
    # Step 5: Compare metrics
    print("\n" + "="*80)
    print("STEP 5: Comparing performance metrics")
    print("="*80)
    start_time = time.time()
    subprocess.run(['python', 'compare_results.py'])
    print(f"Results comparison completed in {time.time() - start_time:.2f} seconds")
    
    # Step 6: Visualize rollouts
    print("\n" + "="*80)
    print("STEP 6: Visualizing rollout behavior")
    print("="*80)
    start_time = time.time()
    subprocess.run(['python', 'visualize_rollouts.py'])
    print(f"Rollout visualization completed in {time.time() - start_time:.2f} seconds")
    
    # Step 7: Prepare hierarchical data
    print("\n" + "="*80)
    print("STEP 7: Preparing data for hierarchical BC models")
    print("="*80)
    start_time = time.time()
    subprocess.run(['python', 'prepare_hierarchical_data.py'])
    print(f"Hierarchical data preparation completed in {time.time() - start_time:.2f} seconds")
    
    # Step 8: Train hierarchical BC models
    print("\n" + "="*80)
    print("STEP 8: Training hierarchical BC models")
    print("="*80)
    start_time = time.time()
    subprocess.run(['python', 'train_hierarchical_bc.py'])
    print(f"Hierarchical BC training completed in {time.time() - start_time:.2f} seconds")
    
    # Step 9: Train Bayesian gaze model
    print("\n" + "="*80)
    print("STEP 9: Training Bayesian gaze BC model")
    print("="*80)
    start_time = time.time()
    subprocess.run(['python', 'train_bayesian_gaze_bc.py'])
    print(f"Bayesian gaze BC training completed in {time.time() - start_time:.2f} seconds")
    
    # Step 10: Evaluate hierarchical BC models
    print("\n" + "="*80)
    print("STEP 10: Evaluating hierarchical BC models")
    print("="*80)
    start_time = time.time()
    subprocess.run(['python', 'evaluate_hierarchical_bc.py', '--create-animations', '--evaluate-bayesian'])
    print(f"Hierarchical BC evaluation completed in {time.time() - start_time:.2f} seconds")
    
    # End timer
    total_time = time.time() - total_start_time
    print("\n" + "="*80)
    print(f"COMPLETE: Pipeline finished in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("Results and visualizations can be found in the 'results' directory")
    print("Hierarchical BC model results can be found in the 'results/hierarchical' directory")
    print("Bayesian gaze model results can be found in the 'results/hierarchical/bayesian' directory")
    print("="*80)

if __name__ == "__main__":
    run_pipeline() 