from matplotlib import colors
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from cdlib import algorithms, evaluation
import community as community_louvain
import matplotlib.cm as cm
import warnings

warnings.filterwarnings("ignore")

# === Load Data ===
nodes_df = pd.read_csv("fb-pages-tvshow.nodes", sep=",")
edges_df = pd.read_csv("fb-pages-tvshow.edges", sep=",", header=None, names=["source", "target"])

G = nx.Graph()

for _, row in nodes_df.iterrows():
    G.add_node(row['new_id'], name=row['name'])

for _, row in edges_df.iterrows():
    G.add_edge(row['source'], row['target'])

print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# === Part 2: Network Analysis ===

# Degree distribution
degree_sequence = [deg for _, deg in G.degree()]
plt.figure(figsize=(8, 5))
plt.hist(degree_sequence, bins=50, color='skyblue')
plt.title("Degree Distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("degree_distribution.png")
plt.show()
plt.close()

# Connected components
components = list(nx.connected_components(G))
largest_component = G.subgraph(max(components, key=len)).copy()
print(f"Number of components: {len(components)}")
print(f"Largest component size: {largest_component.number_of_nodes()} nodes")

# Shortest paths (on largest component)
avg_shortest_path_length = nx.average_shortest_path_length(largest_component)
print(f"Average shortest path length: {avg_shortest_path_length}")

# Clustering coefficient and density
clustering = nx.average_clustering(G)
density = nx.density(G)
print(f"Average clustering coefficient: {clustering}")
print(f"Network density: {density}")

# Centrality
deg_cent = nx.degree_centrality(G)
bet_cent = nx.betweenness_centrality(G)
print("Top 5 Degree Centrality:", sorted(deg_cent.items(), key=lambda x: -x[1])[:5])
print("Top 5 Betweenness Centrality:", sorted(bet_cent.items(), key=lambda x: -x[1])[:5])

# === Part 3: Community Detection ===

# Louvain
louvain_comm = algorithms.louvain(G)
print(f"Louvain: {len(louvain_comm.communities)} communities")

# Label Propagation
label_prop = algorithms.label_propagation(G)
print(f"Label Propagation: {len(label_prop.communities)} communities")

# Infomap
infomap_comm = algorithms.infomap(G)
print(f"Infomap: {len(infomap_comm.communities)} communities")

# Compare modularity scores
modularity_scores = {
    "Louvain": evaluation.newman_girvan_modularity(G, louvain_comm).score,
    "Label Propagation": evaluation.newman_girvan_modularity(G, label_prop).score,
    "Infomap": evaluation.newman_girvan_modularity(G, infomap_comm).score
}
for method, score in modularity_scores.items():
    print(f"{method} Modularity: {score:.4f}")

# === Visualization: Louvain Communities (Top 100 Nodes) ===

# Use top 100 nodes by degree
top_n = 100
top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_n]
subgraph = G.subgraph([n for n, _ in top_nodes])

# Louvain partition
partition = community_louvain.best_partition(subgraph)

# Plot
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm

pos = nx.spring_layout(subgraph, seed=42, k=0.5, iterations=200)  # More spacing

# Coloring nodes by community
colors = [cm.get_cmap('tab20')(partition[n] % 20) for n in subgraph.nodes()]

plt.figure(figsize=(16, 12))  # Bigger figure
nx.draw_networkx_edges(subgraph, pos, alpha=0.3, width=1)

# Draw nodes with edgecolor (outline) and partial transparency
nx.draw_networkx_nodes(
    subgraph, pos, node_size=500, node_color=colors,
    edgecolors='black', linewidths=1, alpha=0.9
)

# Label only top central nodes or high degree ones
top_nodes = sorted(subgraph.degree, key=lambda x: -x[1])[:10]
labels = {
    n: nodes_df[nodes_df['new_id'] == n]['name'].values[0]
    for n, _ in top_nodes
}

# Plot labels slightly above nodes with white background for readability
for node, (x, y) in pos.items():
    if node in labels:
        plt.text(
            x, y + 0.05, labels[node], fontsize=10, ha='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.3')
        )

plt.title("Louvain Communities (Top 100 Nodes)", fontsize=18)
plt.axis('off')
plt.tight_layout()
plt.savefig("louvain_fixed_cleaned.png")
plt.show()
