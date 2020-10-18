# An algorithm to choose a set of vertice in a graph with largest BC based on iGraph.
# version 0.0
# 07/13/20
from igraph import *
import math 
from random import choice
import copy
import sys
import time

#print(igraph.__version__)

def read_udgraph(path):
    print(path)
    g = Graph.Read_Ncol(path, directed = False)
    g.vs["name"] = ["{0}".format(i) for i in range(g.vcount())]
    print (g.vcount())
    print(g.ecount())
    print(g.transitivity_undirected())
    return g

#print(read_udgraph('test_graph.txt'))

#caculate m
def set_m(graph, eta):
    m = int(math.log2((graph.vcount()**2)/(eta**2)))
    return m
#g = read_udgraph('test_graph.txt')
#print(set_m(g, 0.1))




def cal_bc(graph, m, s_list, t_list): # m must be much smaller than the number of edges
    bc_dict = dict()
    #print(VertexSeq(graph))
    
    #nd_list = list(VertexSeq(graph))
    nd_list = list(graph.vs["name"])
    #print(nd_list)
    for node in nd_list:
        bc_dict[node] = 0
    #print(bc_dict)

    for i in range(m):
        ndl_copy = copy.deepcopy(nd_list)
        if len(ndl_copy) >=2:
            s = choice(ndl_copy)
        
            if not(s in s_list):
                s_list.append(s)
            ndl_copy.remove(s)
            t = choice(ndl_copy)
            #check not been chosen
            if not (t  in t_list):
                t_list.append(t)
            counter = 0
            #ts1 = time.time()
            path_list = graph.get_all_shortest_paths(s, to=t, weights=None, mode=ALL)
            count_path = len(path_list)
            #print("nodes in s to t:")
            #print(count_path)  # return Ids
            #counter = 0
            ts1 = time.time()
            for pt in path_list:
                #counter = counter + 1
                for n in pt[1:-1]:
                    counter = counter + 1
                    #print(graph.vs[n]["name"])
                    bc_dict[graph.vs[n]["name"]] = bc_dict[graph.vs[n]["name"]]+ 1/count_path
            ts2 = time.time()
            print("using time: %f ms" %((ts2-ts1)*1000))
            print(counter)
    
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
    #g1 = read_udgraph('Wiki-Vote.txt')
    #bc_dict = dict()
    high_bc_set = []
    vertice_list_s = []
    vertice_list_t = []
    m = set_m(g1, eta)
    
    #print(m)
    init, vertice_list_s, vertice_list_t = cal_bc(g1, m, vertice_list_s, vertice_list_t)
    #print(init)
    num_nodes = g1.vcount()
    print("Need run %d times" %(num_nodes**2))
    for i in range(num_nodes**2): # set to 2
        print("%d th run " %(i))


        if len(init) == 0:
            print(high_bc_set)
            break
        vi = maxBC(init)
        #print(vi)
        high_bc_set += vi
        #print(high_bc_set)
        if len(vertice_list_s) >= percent * num_nodes and len(vertice_list_t) >= percent * num_nodes:
            break
    
        for nd in vi:
            g1.delete_vertices(nd)
            
        #print(g1)

        init, vertice_list_s, vertice_list_t = cal_bc(g1, m, vertice_list_s, vertice_list_t)
        #print(init)
        if len(init) == 0:
            break
        
    return high_bc_set


def main():
    print(sys.argv[0])
    path = sys.argv[1]
    #path =  'email-Eu-core.txt' # 'PP-Pathways_ppi.txt'# # 'test_graph.txt' #
    coverage = float(sys.argv[2])
    #coverage = 0.9
    

    test_graph =  read_udgraph(path)

    #print(test_graph)
    #test_graph.delete_vertices(5)
    #print(test_graph.vcount())
    #print(test_graph)
    
    ts1 = time.time()
    m = set_m(test_graph, 0.1)
    bc_set = get_set(test_graph,coverage, 0.1)
    ts2 = time.time()
    print("Duration: %d seconds" %(ts2-ts1))
    outfile = open(path.split('.')[0] +"_high_bc_set.txt", "w")
    for i in bc_set:
                outfile.write(i) 
                outfile.write('\n') 

    outfile.close()
    #print(m)
    '''
    high_bc_set = []
    vertice_list_s = []
    vertice_list_t = []
    percent = 0.1
    init, vertice_list_s, vertice_list_t = cal_bc(test_graph, m, vertice_list_s, vertice_list_t)
    
    #print(init)
    #print(maxBC(init, 5))
    
    #get_set(test_graph)
    num_nodes = test_graph.number_of_nodes()
    for i in range(num_nodes**2):
        vi = maxBC(init, 20)
        print(vi)
        high_bc_set.append(vi)
        if len(vertice_list_s) >= percent * num_nodes and len(vertice_list_t) >= percent * num_nodes:
            print(high_bc_set)
            break
        for node in vi:
            #print(node)
            test_graph.remove_node(node)
            #print(test_graph)
        #m = set_m(new_graph, eta)
        init, vertice_list_s, vertice_list_t = cal_bc(test_graph, m, vertice_list_s, vertice_list_t)
        #print(vertice_list_s)
        #print(vertice_list_t)
    
    
    print(init)
    '''



if __name__ == 'main':
    main()

main()
