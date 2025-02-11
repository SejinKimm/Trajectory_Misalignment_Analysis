import arcle
import action_sequence_parser
import json
import numpy as np
import time
from pyvis.network import Network
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt


df = action_sequence_parser.preprocess()


def tostr(grid, sel):
    unsel=['⬛','🟦','🟥','🟩','🟨','🔲','⬜','🟧','🟪','🟫']
    oksel=['🖤','🩵','❤️','💚','💛','🩶','🤍','🧡','💜','🤎']
    s = ''
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if sel[i,j] == 1:
                s+=oksel[grid[i,j]]
            else:
                s+=unsel[grid[i,j]]
        s+='\n'
    return s

ents = []
ns = []
od = []
for i in range(1,401):
    prob_id = i

    probdf= df[df['taskId']==prob_id]
    print(f'{len(probdf)} records found for problem {prob_id}.')

    graphnodes = defaultdict(int)
    graphedges = defaultdict(int)

    graphneighbours = defaultdict(lambda : defaultdict(int))

    graph_indegrees = defaultdict(int)
    graph_outdegrees = defaultdict(int)
    graphnodeend = defaultdict(int)
    nodedata = defaultdict(lambda : {'grid':None, 'sel':None, 'op':None})
    edgedata = defaultdict(str)
    errorcount = 0
    for trajind in range(len(probdf)):
        #print(trajind)
        success = probdf.iloc[trajind]['success']
        traj= json.loads(probdf.iloc[trajind]['actionSequence'])
        cursor = 0
        prevhash = 0
        graphnodes[prevhash] +=1
        
        graph_outdegrees[prevhash] +=1

        try:
            for act in traj:
                sel = np.zeros((30,30),dtype=int)
                op = act['operation']
                grid = np.array(act['grid'])
                h, w = grid.shape
                for aaa in act['object']:
                    sel[aaa['y'],aaa['x']]=1
                    if 0<=aaa['y']<h and 0<=aaa['x']<w:
                        grid[aaa['y'],aaa['x']]=aaa['color']
                
                sel = sel[:h, :w]

                statehash = hash(str(grid))
                graphnodes[statehash]+=1
                nodedata[statehash] = {'grid':grid, 'sel':sel, 'op':op}

                graphedges[(prevhash, hash(str(op)), statehash)]+=1
                if prevhash != statehash:
                    graphneighbours[prevhash][statehash] +=1
                    graph_outdegrees[prevhash] +=1
                    graph_indegrees[statehash] +=1
                edgedata[(prevhash, hash(str(op)), statehash)] = op
                if act['operation'] == 'Submit':
                    
                    graphnodeend[statehash]= +1 if success else -1
                prevhash = statehash
        except:
            errorcount+=1
    
    n_trajs = len(probdf) - errorcount
    nodesizedist = defaultdict(int)
    outdegdist =    defaultdict(int)
    totout = 0
    totin = 0
    for node in graphnodes.keys():
        nodesizedist[graph_indegrees[node]]+=1
        totin+=1
        for neighbour in graphneighbours[node].keys():
            outdegdist[graphneighbours[node][neighbour]]+=1
            totout += 1
    
    avg_nodesize = 0
    for key in nodesizedist.keys():
        avg_nodesize+=key * nodesizedist[key]/totin

    avg_nodesize /=n_trajs
    ns.append(avg_nodesize)
    
    avg_outdeg = 0
    for key in outdegdist.keys():
        avg_outdeg += key*outdegdist[key]/ totout
    avg_outdeg /= n_trajs
    od.append(avg_outdeg)

    print(errorcount,'errors')
    sizef = lambda s: 5+10*np.log(s)
    widthf = lambda s: 3+5*np.log(s)
    g = Network(height='1080px', width='100%', directed=True)
    g.inherit_edge_colors(False)
    g.force_atlas_2based()
    




    g.add_node(0, label='start', size=sizef(graphnodes[0]), color='blue')
    for node in graphnodes.keys():
        prop = nodedata[node]
        grid = prop['grid']
        sel = prop['sel']
        
        if grid is not None:
            title=tostr(grid,np.zeros(grid.shape,dtype=int))
        else:
            title=''
        if graphnodeend[node] == -1:
            g.add_node(node, label="wrong",color='red',title=title, value=graphnodes[node])
        elif graphnodeend[node] == 1:
            g.add_node(node, label="correct", color='green',title=title, value=graphnodes[node])
        else:
            g.add_node(node,color='gray',title=title, value=graphnodes[node])

    for edge in graphedges.keys():
        if edge[0]==edge[2]:
            continue
        g.add_edge(edge[0], edge[2], label=edgedata[edge],title=f'{edgedata[edge]}\n{graphedges[edge]} transitions', value=graphedges[edge],color='rgba(128,128,128,0.5)',arrowStrikethrough=False)
    

    g.show_buttons()
    g.save_graph(f'graphs_wosel/{prob_id}.html')
    