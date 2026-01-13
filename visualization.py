import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

def load_results(filename='results.json'):
    """Load results from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def create_performance_charts(results):
    """Create clean, impactful performance comparison charts"""
    
    # Modern style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Extract data
    scenarios = []
    dijkstra_times = []
    astar_times = []
    dijkstra_nodes = []
    astar_nodes = []
    dijkstra_distance = []
    astar_distance = []
    
    for i in range(0, len(results), 2):
        dijkstra_result = results[i] if results[i]['algorithm'] == 'dijkstra' else results[i+1]
        astar_result = results[i+1] if results[i+1]['algorithm'] == 'astar' else results[i]
        
        scenarios.append(dijkstra_result['scenario_name'])
        dijkstra_times.append(dijkstra_result['runtime'] * 1000)  # to ms
        astar_times.append(astar_result['runtime'] * 1000)
        dijkstra_nodes.append(dijkstra_result['nodes_explored'])
        astar_nodes.append(astar_result['nodes_explored'])
        dijkstra_distance.append(dijkstra_result['distance'])
        astar_distance.append(astar_result['distance'])
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Emergency Pathfinding: Algorithm Performance Comparison', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Colors
    color_dijk = '#3498db'  # Blue
    color_astar = '#e74c3c'  # Red
    
    # 1. Runtime Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width/2, dijkstra_times, width, label='Dijkstra', 
                    color=color_dijk, alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, astar_times, width, label='A*', 
                    color=color_astar, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Runtime (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('⏱️  Runtime Performance', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=9)
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)
    
    # 2. Nodes Explored
    ax2 = axes[0, 1]
    bars3 = ax2.bar(x - width/2, dijkstra_nodes, width, label='Dijkstra', 
                    color=color_dijk, alpha=0.8, edgecolor='black', linewidth=1.2)
    bars4 = ax2.bar(x + width/2, astar_nodes, width, label='A*', 
                    color=color_astar, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Nodes Explored', fontsize=12, fontweight='bold')
    ax2.set_title('🔍  Search Space Efficiency', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=9)
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8)
    
    # 3. Path Distance (Accuracy Check)
    ax3 = axes[1, 0]
    bars5 = ax3.bar(x - width/2, dijkstra_distance, width, label='Dijkstra', 
                    color=color_dijk, alpha=0.8, edgecolor='black', linewidth=1.2)
    bars6 = ax3.bar(x + width/2, astar_distance, width, label='A*', 
                    color=color_astar, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax3.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Distance (meters)', fontsize=12, fontweight='bold')
    ax3.set_title('✓  Path Optimality Verification', fontsize=14, fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=9)
    ax3.legend(fontsize=11, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # 4. Efficiency Improvement
    ax4 = axes[1, 1]
    time_improvement = [(d - a) / d * 100 for d, a in zip(dijkstra_times, astar_times)]
    nodes_improvement = [(d - a) / d * 100 for d, a in zip(dijkstra_nodes, astar_nodes)]
    
    bars7 = ax4.bar(x - width/2, time_improvement, width, label='Runtime Reduction', 
                    color='#27ae60', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars8 = ax4.bar(x + width/2, nodes_improvement, width, label='Node Exploration Reduction', 
                    color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax4.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax4.set_title('📈  A* Efficiency Gains', fontsize=14, fontweight='bold', pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(scenarios, rotation=30, ha='right', fontsize=9)
    ax4.legend(fontsize=11, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add value labels
    for bars in [bars7, bars8]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Chart saved: performance_comparison.png")
    plt.close()
    
    # Print statistics
    print("\n" + "=" * 70)
    print("PERFORMANCE STATISTICS")
    print("=" * 70)
    
    avg_dijk_time = np.mean(dijkstra_times)
    avg_astar_time = np.mean(astar_times)
    avg_dijk_nodes = np.mean(dijkstra_nodes)
    avg_astar_nodes = np.mean(astar_nodes)
    
    print(f"\n📊 Average Runtime:")
    print(f"   Dijkstra: {avg_dijk_time:.2f} ms")
    print(f"   A*:       {avg_astar_time:.2f} ms")
    print(f"   ➜ A* is {(avg_dijk_time - avg_astar_time) / avg_dijk_time * 100:.1f}% faster")
    
    print(f"\n🔍 Average Nodes Explored:")
    print(f"   Dijkstra: {avg_dijk_nodes:.0f}")
    print(f"   A*:       {avg_astar_nodes:.0f}")
    print(f"   ➜ A* explores {(avg_dijk_nodes - avg_astar_nodes) / avg_dijk_nodes * 100:.1f}% fewer nodes")
    
    print(f"\n✓ Path Accuracy:")
    distance_diffs = [abs(d - a) for d, a in zip(dijkstra_distance, astar_distance)]
    print(f"   Max distance difference: {max(distance_diffs):.2f} meters")
    print(f"   Both algorithms find optimal paths: {max(distance_diffs) < 1.0}")
    
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    print("\nGenerating performance visualizations...\n")
    results = load_results()
    create_performance_charts(results)
    print("✓ Analysis complete!\n")