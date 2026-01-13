# 🚀 Emergency Pathfinding

A Python-based solution for computing optimal emergency evacuation routes using classic graph search algorithms. This project includes performance benchmarking, route visualization, and interactive output to assist in evaluating **A*** and **Dijkstra** pathfinding approaches in emergency scenarios.

---

## 📑 Table of Contents
- [About](#about)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Comparison](#performance-comparison)
- [Example Outputs](#example-outputs)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

---

## 📖 About
**Emergency Pathfinding** is an open-source project designed to explore and compare shortest-path algorithms in the context of emergency response and evacuation planning. The system allows simulation of routing scenarios, benchmarking of algorithms, and visualization of computed paths. 

This repository is intended for developers, researchers, and students interested in algorithmic route optimization and emergency navigation systems.

---

## ✨ Features
* **Algorithm Support:**
    * **A* Search:** Heuristic-based approach for faster, goal-oriented search.
    * **Dijkstra’s Algorithm:** Guaranteed shortest path via uniform cost search.
* **Emergency Simulation:** Model-specific routing for crisis scenarios.
* **Performance Benchmarking:** Side-by-side metrics comparison.
* **Interactive Visualization:** HTML-based maps for route inspection.
* **Structured Data:** Exportable results in `.json` format for further analysis.
* **Modular Code:** Easily extend with new algorithms or map data.

---

## 📁 Repository Structure
```text
.
├── main.py                    # Core pathfinding logic
├── run.py                     # Script to run simulations
├── visualization.py           # Route visualization (HTML output)
├── performance_comparison.png # Algorithm performance comparison chart
├── route_*.html               # Generated interactive visualizations
├── results.json               # Output data and metrics
├── req.txt                    # Python dependencies
├── run_clean.sh               # Cleanup script for generated files
└── README.md                  # Project documentation
```

⚙️ Installation
1. Clone the Repository
git clone https://github.com/finddawood/emergency-pathfinding.git
cd emergency-pathfinding

2. Set Up the Environment

Python 3.8 or higher is recommended.

# Create virtual environment
python3 -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate

# Install dependencies
pip install -r req.txt

🚀 Usage
Run the Simulation

Execute the main script:

python run.py

What Happens During Execution?

Predefined emergency scenarios are loaded

Pathfinding algorithms are executed

Performance metrics are saved to results.json

Interactive HTML route visualizations are generated

View Results

Open any generated HTML file in your web browser:

route_astar_*.html
route_dijkstra_*.html

📊 Performance Comparison

The system generates a performance_comparison.png file that visually compares execution time and efficiency between the implemented algorithms across multiple scenarios.

📄 Example Outputs

results.json – Runtime, path length, nodes visited, and metadata

route_*.html – Interactive map-based route visualizations

performance_comparison.png – Benchmark comparison chart

🛠️ Requirements

Python: 3.8 or higher

Dependencies: Listed in req.txt

Install all requirements with:

pip install -r req.txt
