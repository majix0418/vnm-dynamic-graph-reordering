import random

graphInfo = input().split()
numberOfVertices = int(graphInfo[0])
numberOfEdges = int(graphInfo[2])
# number of edges from/to the vertex
vertexInfo = []
edges = []

for i in range(numberOfVertices):
    # [node number, from, to]
    vertexInfo.append([i+1, 0, 0])

for i in range(numberOfEdges):
    edgeInfo = input().split()
    vertexInfo[int(edgeInfo[0])-1][1] += 1
    vertexInfo[int(edgeInfo[1]) - 1][2] += 1
    edges.append([int(edgeInfo[0]), int(edgeInfo[1])])

def basicGen():
    a = random.randint(1, numberOfVertices)
    b = random.randint(1, numberOfVertices)
    while [a,b] in edges:
        a = random.randint(1, numberOfVertices)
        b = random.randint(1, numberOfVertices)
    return [a,b]

# maybe weighted generation using vertex info later

# replace "10" for any number
for i in range (10):
    newEdge = basicGen()
    edges.append(newEdge)
    vertexInfo[newEdge[0]-1][1] += 1
    vertexInfo[newEdge[0]-1][2] += 1