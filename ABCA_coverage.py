""" A Python Class
A simple Python graph class, demonstrating the essential 
facts and functionalities of graphs.
"""
#import queue
import math 
from random import choice
#import copy
import sys
import time
from collections import deque
#from numba import jit



class Graph(object):

    def __init__(self, graph_dict=None):
        """ initializes a graph object 
            If no dictionary or None is given, 
            an empty dictionary will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self.__graph_dict = graph_dict





    def self_dict(self):
        return self.__graph_dict
    
    def bfs_dict(self):
        return self.__bfs_dict


    def vertices(self):
        """ returns the vertices of a graph """
        return list(self.__graph_dict.keys())

    def edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()

    def num_vertices(self):
        """ returns the number  of vertices of a graph """
        return len(self.__graph_dict.keys())

    def num_edges(self):
        """ returns the number of edges of a graph """
        return len(self.__generate_edges())



    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in 
            self.__graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary. 
            Otherwise nothing has to be done. 
        """
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex] = {}

    def delete_vertex(self,vertex):
        if vertex not in self.__graph_dict.keys():
            print("The vertex is not in the graph")
        else:
            for node in self.__graph_dict[vertex]:
                self.__graph_dict[node].remove(vertex)
            self.__graph_dict.pop(vertex)


    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple or list; 
            between two vertices can be multiple edges! 
        """
        edge = set(edge)
        (vertex1, vertex2) = tuple(edge)
    
        if vertex1 in self.__graph_dict.keys() and vertex2 in self.__graph_dict.keys():
            if vertex2 in self.__graph_dict[vertex1] and vertex1 in self.__graph_dict[vertex2]:
                return
            self.__graph_dict[vertex1].add(vertex2)
            self.__graph_dict[vertex2].add(vertex1)
        elif vertex1 not in self.__graph_dict.keys() and vertex2 in self.__graph_dict.keys():
            self.__graph_dict[vertex1] = {vertex2}
            self.__graph_dict[vertex2].add(vertex1)
        elif vertex1 in self.__graph_dict.keys() and vertex2 not in self.__graph_dict.keys():
            self.__graph_dict[vertex2] = {vertex1}
            self.__graph_dict[vertex1].add(vertex2)
        else:
            self.__graph_dict[vertex1] = {vertex2}
            self.__graph_dict[vertex2] = {vertex1}


    def delete_edge(self, edge):
        edge = set(edge)
        (vertex1, vertex2) = tuple(edge) 
        if vertex1 in self.__graph_dict.keys() and vertex2 in self.__graph_dict[vertex1]:
            self.__graph_dict[vertex1].remove(vertex2)
        #if vertex2 in self.__graph_dict.keys() and vertex1 in self.__graph_dict[vertex2]:
            self.__graph_dict[vertex2].remove(vertex1)
        else:
            print("This edge is not in the graph.")

        

    def __generate_edges(self):
        """ A static method generating the edges of the 
            graph "graph". Edges are represented as sets 
            with one (a loop back to the vertex) or two 
            vertices 
        """
        edges = []
        for vertex in self.__graph_dict:
            for neighbor in self.__graph_dict[vertex]:
                if {neighbor, vertex} not in edges:
                    edges.append({vertex, neighbor})
        return edges

# the bfs_dict need to be renewed every time the node changed in graph

    def bfs(self, vertex_s):
        """
        use bfs explore graph from a single vertex
        return a shortest path tree from that vertex
        """
        nd_list = list(self.vertices())
        visited = dict((node, 0) for node in nd_list)

        nq = deque()
        pre_dict, dist = {}, {}
        nq.append(vertex_s)
        visited[vertex_s]=1
        dist[vertex_s]  = 0

        #loop_counts = 0
        while nq:
            s = nq.popleft()
            for node in self.__graph_dict[s]: # for each child/neighbour of current node 's'
                #loop_counts += 1
                
                #if not node in visited:
                if not visited[node]:
                    nq.append(node) # let 'node' in queue
                    pre_dict[node] = [s] # the 'parent' (in terms of shortest path from 'root') of 'node' is 's'
                    dist[node] = dist[s] + 1 # shortest path to 'root'
                    visited[node] = 1 # 'node' is visted
                #if node in visited and dist[node] == dist[s] + 1: # still within the shortest path
                if  visited[node] and dist[node] == dist[s] + 1: # still within the shortest path
                    if s not in pre_dict[node]: # if this path have NOT been recorded, let's do that now
                        pre_dict[node].append(s) 
                    
                if  visited[node] and dist[node] > dist[s] + 1: # the previous 'recorded' path is longer than our current path (via node 's'); let's update that path and distance
                    pre_dict[node] = [s]
                    dist[node] = dist[s] + 1
        #print("  #loops: %d" %loop_counts)
        #current_bfs[vertex_s] = pre_dict
            
        return pre_dict
   
            
        

    def read_edgelist(self, file):
        f = open(file, 'r')
        while True:
            line = f.readline()
            if not line:
                break
            v1, v2 = line.strip().split()
            if v1 != v2: # no self loop
                self.add_edge({v1,v2})

    def is_connect(self, s, t):
        pre_map = self.bfs(s)
        if t in pre_map:
            return [True, pre_map]
        return[False, pre_map]

    def __str__(self):
        res = "vertices: "
        for k in self.__graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res
    


class Node(object):
    """Generic tree."""
    def __init__(self, name='', children=None):
        self.name = name
        if children is None:
            children = {}
        self.children = children

    
    def add_child(self, child):
        self.children.append(child)
    


####not in graph class#############
def bfs_counting(graph, root_vertex, bottom_vertex): # perform analysis twice: 1) set root_vertex = 't'; 2) set root_vertex = 's'
    """
    use bfs explore graph from a single vertex
    return a shortest path tree from that vertex
    """

    #visited = dict()
    nd_list = graph.keys()
    visited = dict((node, 0) for node in nd_list)
    visited[bottom_vertex]=0

    nq = deque()# queue for recording current nodes
    pre_dict, dist, parents, node_count_dict  = {}, {}, {}, {}
    
    nq.append(root_vertex)
    visited[root_vertex]=1
    dist[root_vertex]  = 0
    parents[root_vertex]=['fake_root']
    node_count_dict['fake_root']=1
    while nq:
        s = nq.popleft() # dequeue
       
        node_count_dict[s] = 0
        for p in parents[s]: # count is defined as the sum of counts from all parents
            node_count_dict[s] += node_count_dict[p]

        if not s in graph.keys():
            continue
        for node in graph[s]:

            #if not node in visited:
            if not visited[node]:
                nq.append(node) # let 'node' in queue
                pre_dict[node] = [s] # the 'parent' (in terms of shortest path from 'root') of 'node' is 's'
                dist[node] = dist[s] + 1 # shortest path to 'root'
                visited[node]=1 # 'node' is visted
                parents[node]=[s] # record 'parents' of this node
            else:
                parents[node].append(s) # record 'parents' of this node
                pre_dict[node].append(s)
                
    node_count_dict.pop('fake_root')
    return [pre_dict, node_count_dict]  # two returns: 1) tree; 2) node count dictionary



def dfs(root, total_count):
    #visited = []
    leaf_count = dict()
    #total_count = dict()
    dfs_helper(root, leaf_count, total_count)
    n = leaf_count['root']
    for k in total_count.keys():
        total_count[k] = total_count[k]/n
    return total_count

def dfs_helper(v, leaf_count, total_count): 
      
    # Set current to root of binary tree
    #visited.append(v.name)
    if len(v.children) == 0: 
        leaf_count[v.name] = 1    
    else:
        leaf_count[v.name] = 0
        for nd in v.children:
            dfs_helper(nd, leaf_count, total_count)
            leaf_count[v.name] += leaf_count[nd.name]                
            total_count[nd.name] += leaf_count[nd.name]
    return
        

def add_branch(tree_map, current_node, total_count):
    total_count[current_node.name] = 0
    if current_node.name not in tree_map.keys(): 
        return
    children = tree_map[current_node.name]
    for child in children:
        child_node = Node(child)
        current_node.add_child(child_node)
        add_branch(tree_map, child_node, total_count)

    return

            
def set_m(graph, eta):
    m = int(math.log2((graph.num_vertices()**2)/(eta**2)))
    print("m = %d" %m)
    return m

def cal_bc(graph, m, s_list, t_list): # m must be much smaller than the number of edges
    nd_list = list(graph.vertices())
    bc_dict = dict((node, 0) for node in nd_list)

    for i in range(m):
        if len(nd_list) >=2:
            s = choice(nd_list)
            s_list.add(s)
            nd_list.remove(s)
            t = choice(nd_list)
            t_list.add(t)
            connect, pre_g = graph.is_connect(s, t)
            if connect:
                dfsts1 = time.time()
                pre_g_rev, count1 = bfs_counting(pre_g,t,s)
                dfsts2 = time.time()
                pre_g_rev2, count2 = bfs_counting(pre_g_rev,s,t)
                count = dict((node, count1[node] * count2[node]) for node in count1.keys())
                count = dict((node, count1[node] / count[t]) for node in count1.keys())
                count.pop(s)
                count.pop(t)
                for node in count.keys():
                    bc_dict[node] += count[node]
                dfsts2 = time.time()
                print(" BFS counting: Duration: %f ms" %((dfsts2-dfsts1)*1000))
            else: 
                continue
        else:
            break
    return [bc_dict, s_list, t_list]


def maxBC(dict): #to be finished, default remove highest 1
    #return sorted(dict, key=dict.get, reverse=True)[:n]
    max_value = max(dict.values())  # maximum value
    max_keys = [k for k, v in dict.items() if v == max_value]
    #print(max_keys)
    return max_keys

def get_set(g1, percent = 1.0, eta = 0.1):
    #print(g1)
    high_bc_set = []
    vertice_list_s = {-1}
    vertice_list_t = {-1}
    m = set_m(g1, eta)
    init, vertice_list_s, vertice_list_t = cal_bc(g1, m, vertice_list_s, vertice_list_t) # first calculation
    num_nodes = g1.num_vertices()
    for i in range(num_nodes**2): # set to 2
        print (i)
        vi = maxBC(init)
        high_bc_set += vi
        if len(vertice_list_s) >= percent * num_nodes and len(vertice_list_t) >= percent * num_nodes:
            #for i in high_bc_set:
                #print(i)
            break
        for node in vi:
            #print(node)
            g1.delete_vertex(node)
        init, vertice_list_s, vertice_list_t = cal_bc(g1, m, vertice_list_s, vertice_list_t)
        if len(init) == 0:
            break
    return high_bc_set          
        

if __name__ == "__main__":
    



    graph = Graph()
    #path = "email-Eu-core.txt"
    path = sys.argv[1]
    #path = "compositePPI_bindingOnly_edges.txt"
    graph.read_edgelist(path)
    #graph.read_edgelist("compositePPI_bindingOnly_edges.txt")

    print("Vertices of graph:")
    #print(graph.vertices())
    print(graph.num_vertices())

    print("Edges of graph:")
    

    ts1 = time.time()
    m = set_m(graph, 0.1)
    #bc_set  = get_set(graph, 0.5, 0.1)
    percentage = float(sys.argv[2])
    bc_set  = get_set(graph, percentage, 0.1)
    ts2 = time.time()
    print("Duration: %d seconds" %(ts2-ts1))
    outfile = open(path.split('.')[0] + "_" + str(percentage) +"percentage_high_bc_set.txt", "w")
    for i in bc_set:
        outfile.write(i) 
        outfile.write('\n') 

    outfile.close()


    
    



