import random

graphInfo = input().split()
numberOfVertices = int(graphInfo[0])
numberOfEdges = int(graphInfo[2])
# number of edges from/to the vertex
vertexInfo = []
edges = []

# adjacency list
adj = [set() for _ in range(numberOfVertices + 1)]

for i in range(numberOfVertices):
    # [node number, outdegree, indegree]
    vertexInfo.append([i+1, 0, 0])

for i in range(numberOfEdges):
    edgeInfo = input().split()
    u = int(edgeInfo[0])
    v = int(edgeInfo[1])
    vertexInfo[u - 1][1] += 1
    vertexInfo[v - 1][2] += 1
    edges.append([u,v])
    adj[u].add(v)

def realGen():
    # chat remark: degree^2 + 1?
    outdegrees = [vertex[1]+1 for vertex in vertexInfo]
    indegrees = [vertex[2]+1 for vertex in vertexInfo]
    sumOut = sum(outdegrees)
    sumIn = sum(indegrees)
    outNum = random.randint(1, sumOut)
    inNum = random.randint(1, sumIn)
    currentSum = 0
    outIndex = 0
    inIndex = 0
    for i in range(len(outdegrees)):
        currentSum += outdegrees[i]
        if currentSum >= outNum:
            outIndex = i
            break
    currentSum = 0
    for i in range(len(indegrees)):
        currentSum += indegrees[i]
        if currentSum >= inNum:
            inIndex = i
            break
    return(outIndex + 1, inIndex + 1)

def neighborGen():
    u = random.randint(1, numberOfVertices)
    while len(adj[u]) <= 1: # remark: might not work if each vertex has <= 1 edge
        u = random.randint(1, numberOfVertices)
    outv = random.choice(list(adj[u]))
    inv = random.choice(list(adj[u]))
    attempts = 10
    while (inv in adj[outv] or inv == outv) and attempts >= 0:
        outv = random.choice(list(adj[u]))
        inv = random.choice(list(adj[u]))
        attempts -= 1
    if attempts < 0:
        return(realGen())
    return(outv, inv)

def edgeExists(outv, inv):
    if inv in adj[outv]:
        return True
    return False

def bestGen():
    p = random.randint(1,5)
    if p == 1:
        edge = realGen()
        while edgeExists(edge[0], edge[1]) or edge[0] == edge[1]:
            edge = realGen()
    else:
        edge = neighborGen()
        while edgeExists(edge[0], edge[1]) or edge[0] == edge[1]:
            edge = neighborGen()

    edges.append([edge[0], edge[1]])
    vertexInfo[edge[0] - 1][1] += 1  # outdegree
    vertexInfo[edge[1] - 1][2] += 1  # indegree
    adj[edge[0]].add(edge[1])
    return edge