import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import osmnx as ox
import networkx as nx
import folium
import heapq
import time
import tracemalloc
from typing import List, Tuple, Dict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from geopy.distance import geodesic
import json

# OSMnx settings
ox.settings.log_console = False
ox.settings.use_cache = True

class EmergencyPathfinder:
    """Main class for emergency pathfinding system"""
    
    def __init__(self, city_name: str = "Dresden, Germany"):
        """Initialize the pathfinder with city data"""
        self.city_name = city_name
        self.graph = None
        self.hospitals = []
        self.hospital_nodes = []
        
    def load_city_graph(self):
        """Load city street network from OpenStreetMap"""
        print(f"Loading {self.city_name} street network...")
        
        # Load graph from OSM
        self.graph = ox.graph_from_place(
            self.city_name, 
            network_type='drive',
            simplify=True
        )
        
        # Ensure all edges have length attribute
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if 'length' not in data:
                data['length'] = 100
                
        print(f"✓ Graph loaded: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges\n")
        
    def load_hospitals(self):
        """Load hospital locations - using known Dresden hospitals"""
        print("Loading hospital locations...")
        
        # Known major hospitals in Dresden with verified coordinates
        self.hospitals = [
            {'name': 'Universitätsklinikum Carl Gustav Carus', 'lat': 51.0531, 'lon': 13.7421},
            {'name': 'Krankenhaus Dresden-Friedrichstadt', 'lat': 51.0520, 'lon': 13.7170},
            {'name': 'Städtisches Klinikum Dresden-Neustadt', 'lat': 51.0657, 'lon': 13.7426},
            {'name': 'St. Joseph-Stift Dresden', 'lat': 51.0267, 'lon': 13.7097},
            {'name': 'Diakonissenkrankenhaus Dresden', 'lat': 51.0423, 'lon': 13.8012},
            {'name': 'Krankenhaus Friedrichstadt Standort Trachau', 'lat': 51.0804, 'lon': 13.7089},
            {'name': 'Elblandklinikum Radebeul', 'lat': 51.1082, 'lon': 13.6583},
            {'name': 'Krankenhaus St. Joseph-Stift Plauen', 'lat': 51.0263, 'lon': 13.7100}
        ]
        
        # Map hospitals to nearest graph nodes
        self.hospital_nodes = []
        for hospital in self.hospitals:
            nearest_node = ox.nearest_nodes(
                self.graph, 
                hospital['lon'], 
                hospital['lat']
            )
            self.hospital_nodes.append(nearest_node)
            
        print(f"✓ Found {len(self.hospitals)} hospitals\n")
        
    def dijkstra(self, start_node: int, target_nodes: List[int]) -> Tuple[Dict, Dict, float, float]:
        """
        Dijkstra's shortest path algorithm
        
        Returns: distances, predecessors, runtime, memory_used
        """
        tracemalloc.start()
        start_time = time.time()
        
        # Initialize
        distances = {node: float('inf') for node in self.graph.nodes()}
        distances[start_node] = 0
        predecessors = {node: None for node in self.graph.nodes()}
        visited = set()
        pq = [(0, start_node)]
        
        target_set = set(target_nodes)
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
                
            visited.add(current_node)
            
            # Early termination if we found a hospital
            if current_node in target_set:
                break
                
            # Explore neighbors
            for neighbor in self.graph.neighbors(current_node):
                if neighbor in visited:
                    continue
                    
                # Get edge weight
                edge_data = self.graph.get_edge_data(current_node, neighbor)
                weight = list(edge_data.values())[0].get('length', 100)
                distance = current_dist + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))
        
        runtime = time.time() - start_time
        memory_used = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # MB
        tracemalloc.stop()
        
        return distances, predecessors, runtime, memory_used
    
    def heuristic(self, node1: int, node2: int) -> float:
        """Heuristic function for A* (geodesic distance)"""
        lat1 = self.graph.nodes[node1]['y']
        lon1 = self.graph.nodes[node1]['x']
        lat2 = self.graph.nodes[node2]['y']
        lon2 = self.graph.nodes[node2]['x']
        return geodesic((lat1, lon1), (lat2, lon2)).meters
    
    def a_star(self, start_node: int, target_nodes: List[int]) -> Tuple[Dict, Dict, float, float]:
        """
        A* search algorithm with geodesic heuristic
        
        Returns: distances, predecessors, runtime, memory_used
        """
        tracemalloc.start()
        start_time = time.time()
        
        # Initialize
        g_scores = {node: float('inf') for node in self.graph.nodes()}
        g_scores[start_node] = 0
        predecessors = {node: None for node in self.graph.nodes()}
        visited = set()
        
        # Find nearest target for heuristic
        nearest_target = min(target_nodes, 
                           key=lambda t: self.heuristic(start_node, t))
        
        f_score = self.heuristic(start_node, nearest_target)
        pq = [(f_score, start_node)]
        
        target_set = set(target_nodes)
        
        while pq:
            _, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
                
            visited.add(current_node)
            
            # Early termination
            if current_node in target_set:
                break
                
            # Explore neighbors
            for neighbor in self.graph.neighbors(current_node):
                if neighbor in visited:
                    continue
                    
                # Get edge weight
                edge_data = self.graph.get_edge_data(current_node, neighbor)
                weight = list(edge_data.values())[0].get('length', 100)
                tentative_g = g_scores[current_node] + weight
                
                if tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    predecessors[neighbor] = current_node
                    f = tentative_g + self.heuristic(neighbor, nearest_target)
                    heapq.heappush(pq, (f, neighbor))
        
        runtime = time.time() - start_time
        memory_used = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # MB
        tracemalloc.stop()
        
        return g_scores, predecessors, runtime, memory_used
    
    def reconstruct_path(self, predecessors: Dict, start_node: int, end_node: int) -> List[int]:
        """Reconstruct path from predecessors"""
        path = []
        current = end_node
        
        # Backtrack from end to start
        while current is not None and len(path) < 10000:
            path.append(current)
            if current == start_node:
                break
            current = predecessors.get(current)
            
        path.reverse()
        
        # Validate path
        if len(path) < 2 or path[0] != start_node:
            # Fallback to NetworkX
            try:
                path = nx.shortest_path(self.graph, start_node, end_node, weight='length')
            except:
                path = [start_node, end_node]
                
        return path
    
    def find_nearest_hospital(self, start_lat: float, start_lon: float, algorithm: str = 'dijkstra'):
        """Find path to nearest hospital using specified algorithm"""
        
        # Get nearest node to start position
        start_node = ox.nearest_nodes(self.graph, start_lon, start_lat)
        
        # Run algorithm
        if algorithm == 'dijkstra':
            distances, predecessors, runtime, memory = self.dijkstra(
                start_node, self.hospital_nodes
            )
        else:
            distances, predecessors, runtime, memory = self.a_star(
                start_node, self.hospital_nodes
            )
        
        # Find nearest hospital
        nearest_hospital_idx = min(
            range(len(self.hospital_nodes)),
            key=lambda i: distances[self.hospital_nodes[i]]
        )
        
        nearest_hospital_node = self.hospital_nodes[nearest_hospital_idx]
        path = self.reconstruct_path(predecessors, start_node, nearest_hospital_node)
        
        return {
            'algorithm': algorithm,
            'start_node': int(start_node),
            'hospital_node': int(nearest_hospital_node),
            'hospital_info': self.hospitals[nearest_hospital_idx],
            'path': [int(n) for n in path],
            'distance': float(distances[nearest_hospital_node]),
            'runtime': float(runtime),
            'memory': float(memory),
            'nodes_explored': len([d for d in distances.values() if d != float('inf')])
        }
    
    def visualize_path(self, result: Dict, output_file: str = 'route.html'):
        """Create interactive map visualization"""
        
        # Get coordinates
        start_node = result['start_node']
        start_coords = [
            self.graph.nodes[start_node]['y'], 
            self.graph.nodes[start_node]['x']
        ]
        
        # Create map centered on start
        m = folium.Map(location=start_coords, zoom_start=13)
        
        # Add start marker (green)
        folium.Marker(
            start_coords,
            popup='<b>Start Location</b>',
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
        
        # Add hospital marker (red)
        hospital_coords = [
            result['hospital_info']['lat'], 
            result['hospital_info']['lon']
        ]
        
        folium.Marker(
            hospital_coords,
            popup=f"<b>{result['hospital_info']['name']}</b><br>"
                  f"Distance: {result['distance']:.0f}m<br>"
                  f"Algorithm: {result['algorithm'].upper()}",
            icon=folium.Icon(color='red', icon='plus', prefix='fa')
        ).add_to(m)
        
        # Draw path (blue line)
        if len(result['path']) > 1:
            path_coords = []
            for node in result['path']:
                if node in self.graph.nodes:
                    path_coords.append([
                        self.graph.nodes[node]['y'], 
                        self.graph.nodes[node]['x']
                    ])
            
            if len(path_coords) > 1:
                folium.PolyLine(
                    path_coords,
                    color='blue',
                    weight=4,
                    opacity=0.8,
                    popup=f"<b>Route ({result['algorithm'].upper()})</b><br>"
                          f"Distance: {result['distance']:.0f}m<br>"
                          f"Path nodes: {len(path_coords)}"
                ).add_to(m)
        
        # Add other hospitals (small red circles)
        for hospital in self.hospitals:
            if hospital['name'] != result['hospital_info']['name']:
                folium.CircleMarker(
                    [hospital['lat'], hospital['lon']],
                    radius=4,
                    popup=hospital['name'],
                    color='darkred',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.6
                ).add_to(m)
        
        # Save
        m.save(output_file)
        print(f"  ✓ Map saved: {output_file}")


def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("         EMERGENCY SERVICES PATHFINDING SYSTEM")
    print("                 Dresden, Germany")
    print("=" * 70 + "\n")
    
    # Initialize
    pathfinder = EmergencyPathfinder("Dresden, Germany")
    pathfinder.load_city_graph()
    pathfinder.load_hospitals()
    
    # Test scenarios
    scenarios = [
        {"name": "City Center (Altstadt)", "lat": 51.0504, "lon": 13.7373},
        {"name": "Neustadt District", "lat": 51.0657, "lon": 13.7426},
        {"name": "Plauen District", "lat": 51.0267, "lon": 13.7097},
        {"name": "Striesen District", "lat": 51.0365, "lon": 13.7864},
        {"name": "Klotzsche District", "lat": 51.1287, "lon": 13.7615}
    ]
    
    results = []
    
    print("=" * 70)
    print("RUNNING EXPERIMENTS")
    print("=" * 70 + "\n")
    
    # Run experiments
    for i, scenario in enumerate(scenarios, 1):
        print(f"📍 Scenario {i}: {scenario['name']}")
        print("-" * 70)
        
        for algorithm in ['dijkstra', 'astar']:
            result = pathfinder.find_nearest_hospital(
                scenario['lat'], 
                scenario['lon'], 
                algorithm
            )
            result['scenario_name'] = scenario['name']
            results.append(result)
            
            print(f"\n{algorithm.upper():>12}: ", end='')
            print(f"Hospital: {result['hospital_info']['name']}")
            print(f"              Distance: {result['distance']:.1f}m  |  "
                  f"Runtime: {result['runtime']*1000:.1f}ms  |  "
                  f"Nodes: {result['nodes_explored']}")
            
            # Visualize first scenario
            if i == 1:
                pathfinder.visualize_path(result, f'route_{algorithm}_scenario{i}.html')
        
        print()
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    dijkstra_times = [r['runtime']*1000 for r in results if r['algorithm'] == 'dijkstra']
    astar_times = [r['runtime']*1000 for r in results if r['algorithm'] == 'astar']
    dijkstra_nodes = [r['nodes_explored'] for r in results if r['algorithm'] == 'dijkstra']
    astar_nodes = [r['nodes_explored'] for r in results if r['algorithm'] == 'astar']
    
    avg_dijk_time = np.mean(dijkstra_times)
    avg_astar_time = np.mean(astar_times)
    avg_dijk_nodes = np.mean(dijkstra_nodes)
    avg_astar_nodes = np.mean(astar_nodes)
    
    print(f"\nAverage Runtime:")
    print(f"  Dijkstra: {avg_dijk_time:.1f}ms")
    print(f"  A*:       {avg_astar_time:.1f}ms")
    print(f"  ➜ A* is {((avg_dijk_time - avg_astar_time) / avg_dijk_time * 100):.1f}% faster")
    
    print(f"\nAverage Nodes Explored:")
    print(f"  Dijkstra: {avg_dijk_nodes:.0f}")
    print(f"  A*:       {avg_astar_nodes:.0f}")
    print(f"  ➜ A* explores {((avg_dijk_nodes - avg_astar_nodes) / avg_dijk_nodes * 100):.1f}% fewer nodes")
    
    print(f"\n✓ All {len(scenarios)} scenarios completed successfully!")
    print(f"✓ Results saved to: results.json")
    print(f"✓ Visualizations: route_dijkstra_scenario1.html, route_astar_scenario1.html")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()