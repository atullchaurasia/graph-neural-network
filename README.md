# 🚦 Traffic-GNN: Dynamic Traffic Prediction & Route Optimization

> **Spatio-Temporal Graph Neural Networks (STGNN) on Real-World Maps** 🌍📈

An end-to-end Machine Learning pipeline that predicts future traffic speeds and dynamically optimizes city routes using a hybrid Spatio-Temporal Graph Neural Network architecture (GAT + LSTM) alongside OpenStreetMap integration.

---

## 🌟 Overviews

Traditional routing algorithms calculate the shortest path based on static distances. However, real-world travel relies heavily on **dynamic traffic conditions**. 

This system uses Graph Neural Networks to ingest historical traffic context, predict what traffic will look like in the future, and layer those speed predictions over city graphs to run intelligent route optimization (A* Search).

### 🎯 Key Features
* 🧠 **Spatio-Temporal Prediction Model:** Combines Graph Attention Networks (GAT) to understand spatial road topology and Long Short-Term Memory (LSTM) layers to understand time dynamics.
* 🗺️ **Real-World Map Integration:** Automatically pulls city street networks (e.g., Indore, India) via `osmnx`.
* 📍 **Robust Geocoding:** Translates natural language places (e.g., "Rajwada, Indore") to precise GPS coordinates.
* 🚦 **Graph-Based Routing:** Derives dynamic edge limits, adjusting constraints based on traffic node capacity, and solves for the most efficient path.
* 📊 **Interactive Heatmaps:** Generates interactive [Folium](https://python-visualization.github.io/folium/) HTML maps demonstrating node-level predictions and optimal routes.

---

## 🛠 Flow & Architecture

The pipeline consists of three major stages: Geospatial Extraction, STGNN Prediction, and Route Optimization.

### 1. Spatial Structure & Graph Build
To run predictions on a map, we need a mathematical graph representation of the streets:
* Uses `OpenStreetMap` via `osmnx` to download a drive network for a given city.
* Every intersection becomes a **Node** and every road segment becomes an **Edge**.
* Translates starting points (e.g., "Airport") and destinations to the nearest graph nodes dynamically.

### 2. Spatio-Temporal Graph Neural Network (STGNN)
The core intelligence engine designed to process `[Batch, Time, Nodes, Features]`:
* **Spatial Block (GAT):** Uses Graph Attention Networks (`models/gat_conv.py`). Unlike static convolutions, GAT allows the network to "attend" (give more weight/importance) to heavily congested adjacent nodes instead of treating all connecting roads equally.
* **Temporal Block (LSTM):** Processes the sequence of spatial encodings to predict temporal variations. Traffic at `T=0` influences traffic at `T+15mins`. The `models/temporal.py` builds the memory needed to extrapolate future edge speeds.

### 3. Dynamic Optimization Pathing (A\*)
* The STGNN generates a predicted speed tensor for every single road in the city.
* The `RouteOptimizer` maps these predicted speeds alongside Haversine distances to calculate a realistic **travel time** per edge.
* An **A* Search Algorithm** explores the graph. Using the dynamic travel times as weights (and geographical distance as a heuristic), it discovers the optimal minimum-time path.

---

## 🚀 Quick Start Guide

Want to run this yourself? Follow these steps:

### Installation
```bash
# Clone the repository and install dependencies
git clone <repository_url>
pip install -r requirements.txt
```

### Try Route Prediction
You can immediately predict routes dynamically over the real-world street network.
```bash
# Predicts traffic globally and finds the fastest route
python predict_indore.py --start_loc "Devi Ahilyabai Holkar Airport, Indore" --goal_loc "Musakhedi, Indore"
```
Once completed, simply open `visualizations/indore_custom_route.html` in your web browser!

---

## 📚 Interview Prep: Core Concepts to Learn 

If you are using this project for an interview or portfolio piece, make sure you can explain these concepts natively:

1. **Why Graph Neural Networks (GNN)?** 
   * *Answer:* Standard CNNs work perfectly on a rigid grid (like an image). City roads, however, are non-Euclidean structures (graphs). A GNN naturally adapts to vertices (intersections) and edges (roads), allowing the ML model to pass "traffic messages" along connected pathways.
2. **What is Graph Attention (GAT) vs Graph Convolution (GCN)?** 
   * *Answer:* A traditional GCN uses static adjacency matrices to pool neighbor data. A GAT applies an Attention Mechanism, assigning dynamic weights to neighbor nodes (e.g., recognizing that an accident on Main St impacts traffic far more than normal flow on Side St).
3. **How does A* differ from Dijkstra here?** 
   * *Answer:* We use A* because it incorporates an informed heuristic (straight-line Haversine distance to the goal) to guide the search forward, dramatically reducing search time over large 60,000+ node city graphs compared to Dijkstra.
4. **How do you evaluate standard Traffic Models?**
   * *Answer:* You typically look at Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Mean Absolute Percentage Error (MAPE). The goal is minimizing error across 15m, 30m, and 60m time horizons.

---

## 🐦 Twitter / X Sharing Hook

**Check out this sample thread template to show off your project on Twitter:**

> 🚀 Just built an end-to-end Spatio-Temporal Graph Neural Network (STGNN) to predict dynamic city traffic and optimize travel routes! 🚦🗺️
> 
> Here’s how I combined Deep Learning and Graph Theory to predict real-world congestion: 🧵👇
>
> 1️⃣ Used standard CNN? Nope. Cities aren't grids. I used a Graph Attention Network (GAT) to analyze the complex topology of road networks. 
> 2️⃣ Handled time data naturally by layering an LSTM to predict multi-step horizons (future traffic blocks).
> 3️⃣ Piped the predictions right into OpenStreetMap and plotted dynamic routing logic via A* search constraints. 
>
> Code is open source! #MachineLearning #GNN #Python #DataScience #AI

*(Feel free to attach a small screen recording of your Folium Map route generation alongside this tweet!)*
