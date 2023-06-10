import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec

G = nx.random_tree(40)

nx.draw_spring(G)
plt.show()

# Create a Node2Vec object
node2vec = Node2Vec(G, dimensions=2, walk_length=10, num_walks=200, p=1, q=1)

# Embed nodes
model = node2vec.fit(window=10, min_count=1)

# Get the embeddings for all nodes
embeddings = {}
for node in G.nodes():
    embeddings[node] = model.wv[node]

# Print the embedding of a specific nodef
for i in range(len(embeddings)):
    print(embeddings[i])
