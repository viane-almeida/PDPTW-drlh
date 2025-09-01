#!/usr/bin/env python3
"""
Simple example script for the DRLH PDPTW solver.

This demonstrates how to use the PDPTW solver with actual instance files.

Usage:
    python example_pdptw.py
"""

import sys
import os
sys.path.insert(0, '.')

from agents import DRL_Agent
from problems.pdptw import PDPTW
import glob


def find_sample_instances():
    """Find sample PDPTW instance files in the data directory."""
    sample_dirs = [
        "data/pdptw/pdptw-test",
        "data/pdptw/later", 
        "data/pdptw/pdptw-training-192"
    ]
    
    for sample_dir in sample_dirs:
        if os.path.exists(sample_dir):
            instance_files = glob.glob(os.path.join(sample_dir, "*.txt"))
            if instance_files:
                return sorted(instance_files)[:3]  # Return first 3 files
    
    print("❌ No PDPTW instance files found in data/pdptw/")
    print("   Expected directories: data/pdptw/pdptw-test, data/pdptw/pdptw-training-192")
    return []


def create_pdptw_from_file(instance_file):
    """Create a PDPTW instance from a file."""
    print(f"📁 Loading instance: {os.path.basename(instance_file)}")
    
    # Create placeholder PDPTW to use load_instance method
    pdptw = PDPTW(
        n_nodes=1, n_calls=1, n_vehicles=1,
        calls=[], vehicles=[], vehicles_compatibility=[], calls_compatibility=[],
        dist_matrix=[], cost_matrix=[], wait_times=[], toll_costs=[],
        T_f=0.05
    )
    
    # Load the actual instance
    instance_data = pdptw.load_instance(instance_file)
    
    # Create proper PDPTW with loaded data
    pdptw = PDPTW(
        n_nodes=instance_data.n_nodes,
        n_calls=instance_data.n_calls,
        n_vehicles=instance_data.n_vehicles,
        calls=instance_data.calls,
        vehicles=instance_data.vehicles,
        vehicles_compatibility=instance_data.vehicles_compatibility,
        calls_compatibility=instance_data.calls_compatibility,
        dist_matrix=instance_data.dist_matrix,
        cost_matrix=instance_data.cost_matrix,
        wait_times=instance_data.wait_times,
        toll_costs=instance_data.toll_costs,
        T_f=0.05,
        state_rep="reduced_dist___dist_from_min___no_improvement___index_step___was_changed___unseen",
        reward_func="5310"
    )
    
    # Calculate call difficulty after the instance is properly initialized
    from utils.utils.utils import pdptw_calculate_call_difficulty
    pdptw.call_difficulty = pdptw_calculate_call_difficulty(pdptw)
    
    return pdptw


def example_solve_instance():
    """Example: Solve a single PDPTW instance."""
    print("🚛 DRLH PDPTW Solver - Single Instance Example")
    print("=" * 55)
    
    # Find sample instances
    instance_files = find_sample_instances()
    if not instance_files:
        return
    
    # Use first available instance
    instance_file = instance_files[0]
    
    try:
        # Load PDPTW instance
        pdptw = create_pdptw_from_file(instance_file)
        
        print(f"✅ PDPTW instance loaded successfully")
        print(f"   - Nodes: {pdptw.n_nodes}")
        print(f"   - Calls: {pdptw.n_calls}")
        print(f"   - Vehicles: {pdptw.n_vehicles}")
        print(f"   - Call difficulties: {len(pdptw.call_difficulty)} calculated")
        
        # Create initial solution
        initial_solution = pdptw.generate_initial_solution()
        print(f"   - Initial solution: {len(initial_solution)} vehicle routes")
        
        # Calculate initial cost
        initial_cost = pdptw.objective_function(initial_solution)
        print(f"   - Initial cost: {initial_cost:.2f}")
        
        # Create DRL agent
        print(f"\n🤖 Creating DRL agent...")
        agent = DRL_Agent(
            problem=pdptw,
            alpha=0.001,  # Learning rate
            gamma=0.95,   # Discount factor
            batch_size=32,
            n_epochs=5,
            logdir="logs/pdptw_example",
            use_cuda=False  # Use CPU for compatibility
        )
        
        print(f"✅ DRL agent created")
        print(f"   - Action space: {agent.problem.action_space.n} heuristic operations")
        print(f"   - State space: {agent.problem.observation_space.shape}")
        
        # Quick training
        print(f"\n🎯 Quick training (5 episodes)...")
        agent.train(
            max_samples_train=5,
            learning_rate=0.001,
            logging=False,
            save_model=False,
            verbose=True,
            print_every_n_step=50
        )
        
        # Solve the instance
        print(f"\n🔧 Solving instance...")
        final_solution, final_cost = agent.solve(
            problem=pdptw,
            logging=False,
            verbose=True
        )
        
        # Results
        improvement = initial_cost - final_cost
        improvement_pct = (improvement / initial_cost) * 100 if initial_cost > 0 else 0
        
        print(f"\n📊 Results Summary")
        print("-" * 30)
        print(f"Initial cost:  {initial_cost:.2f}")
        print(f"Final cost:    {final_cost:.2f}")
        print(f"Improvement:   {improvement:.2f} ({improvement_pct:.1f}%)")
        print(f"Instance:      {os.path.basename(instance_file)}")
        
        print(f"\n✅ PDPTW example completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in PDPTW example: {e}")
        import traceback
        traceback.print_exc()


def example_compare_instances():
    """Example: Compare solving multiple PDPTW instances."""
    print("\n" + "=" * 55)
    print("🔍 PDPTW Multi-Instance Comparison")
    print("=" * 55)
    
    # Find sample instances
    instance_files = find_sample_instances()
    if len(instance_files) < 2:
        print("Need at least 2 instances for comparison")
        return
    
    results = []
    
    for i, instance_file in enumerate(instance_files):
        print(f"\n🎯 Instance {i+1}/{len(instance_files)}: {os.path.basename(instance_file)}")
        
        try:
            # Load instance
            pdptw = create_pdptw_from_file(instance_file)
            
            # Create agent
            agent = DRL_Agent(
                problem=pdptw,
                alpha=0.002,
                batch_size=16,
                n_epochs=3,
                use_cuda=False
            )
            
            # Quick training
            print("   Training...")
            agent.train(
                max_samples_train=3,
                learning_rate=0.002,
                logging=False,
                save_model=False,
                verbose=False
            )
            
            # Solve
            print("   Solving...")
            initial_solution = pdptw.generate_initial_solution()
            initial_cost = pdptw.objective_function(initial_solution)
            
            final_solution, final_cost = agent.solve(
                problem=pdptw,
                logging=False,
                verbose=False
            )
            
            improvement = initial_cost - final_cost
            improvement_pct = (improvement / initial_cost) * 100 if initial_cost > 0 else 0
            
            results.append({
                'instance': os.path.basename(instance_file),
                'nodes': pdptw.n_nodes,
                'calls': pdptw.n_calls,
                'vehicles': pdptw.n_vehicles,
                'initial_cost': initial_cost,
                'final_cost': final_cost,
                'improvement': improvement,
                'improvement_pct': improvement_pct
            })
            
            print(f"   ✅ Cost: {initial_cost:.1f} → {final_cost:.1f} ({improvement_pct:.1f}% improvement)")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Summary table
    if results:
        print(f"\n📊 Comparison Summary")
        print("-" * 80)
        print(f"{'Instance':<25} {'Nodes':<6} {'Calls':<6} {'Vehicles':<8} {'Initial':<8} {'Final':<8} {'Improv%':<8}")
        print("-" * 80)
        
        for r in results:
            print(f"{r['instance']:<25} {r['nodes']:<6} {r['calls']:<6} {r['vehicles']:<8} "
                  f"{r['initial_cost']:<8.1f} {r['final_cost']:<8.1f} {r['improvement_pct']:<8.1f}")
        
        avg_improvement = sum(r['improvement_pct'] for r in results) / len(results)
        print("-" * 80)
        print(f"Average improvement: {avg_improvement:.1f}%")


def main():
    """Run PDPTW examples."""
    print("🚛 DRLH PDPTW Solver - Examples")
    print("=" * 40)
    
    # Check if instance files exist
    instance_files = find_sample_instances()
    if not instance_files:
        print("\n💡 To run this example, you need PDPTW instance files.")
        print("   Place .txt instance files in one of these directories:")
        print("   - data/pdptw/pdptw-test/")
        print("   - data/pdptw/pdptw-training-192/") 
        print("   - data/pdptw/later/")
        return
    
    print(f"Found {len(instance_files)} PDPTW instance files")
    
    # Run examples
    example_solve_instance()
    
    if len(instance_files) > 1:
        example_compare_instances()
    
    print(f"\n🎉 All PDPTW examples completed!")


if __name__ == "__main__":
    main()
