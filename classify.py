import parse
import json
import numpy as np
import time
import arckit
import arcle
import json
import numpy as np
import time
from pyvis.network import Network
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt

df = parse.preprocess()


def popularity_predicate(n_trajectory, avg_indegree, node_indegree):
    return node_indegree >= np.sqrt(n_trajectory)


validtrajectory = []
trajwise = []


def tostr(grid, sel):
    unsel = ["⬛", "🟦", "🟥", "🟩", "🟨", "🔲", "⬜", "🟧", "🟪", "🟫"]
    oksel = ["🖤", "🩵", "❤️", "💚", "💛", "🩶", "🤍", "🧡", "💜", "🤎"]
    s = ""
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if sel[i, j] == 1:
                s += oksel[grid[i, j]]
            else:
                s += unsel[grid[i, j]]
        s += "\n"
    return s


def classify(prob_id):
    init_grid = arckit.load_single(id="train" + str(prob_id - 1)).test[0][0]

    probdf = df[df["taskId"] == prob_id]
    print(f"{len(probdf)} records found for problem {prob_id}.")

    # Graph Data Processing -------------------------------------------------------
    graphnodes = defaultdict(int)
    graphedges = defaultdict(int)
    graphneighbours = defaultdict(lambda: defaultdict(int))
    graph_indegrees = defaultdict(int)
    graph_outdegrees = defaultdict(int)
    graphnodeend = defaultdict(int)
    nodedata = defaultdict(lambda: {"grid": None, "sel": None, "op": None})
    edgedata = defaultdict(str)
    hash_seqs = []
    succ_ = []
    errorcount = 0
    fraudcount = 0
    starthash = hash(str(init_grid))
    for trajind in range(len(probdf)):

        success = probdf.iloc[trajind]["success"]
        tt = probdf.iloc[trajind]["dt"].total_seconds()

        traj = json.loads(probdf.iloc[trajind]["actionSequence"])
        yes_copy = False
        yes_paste = False
        for act in traj:
            op = act["operation"]
            if op == "Copy":
                yes_copy = True
            elif op == "Paste":
                yes_paste = True

        if (
            tt < 10 and success and len(traj) <= 6 and (not yes_copy and yes_paste)
        ):  # fraud detection
            # print("FRAUD at ----------------------------------", trajind, tt)'
            fraudcount += 1
            continue
        prevhash = starthash
        graphnodes[prevhash] += 1

        graph_outdegrees[prevhash] += 1

        try:
            traj_hashseq = [prevhash]
            for act in traj:
                sel = np.zeros((30, 30), dtype=int)
                op = act["operation"]
                grid = np.array(act["grid"])
                h, w = grid.shape
                for aaa in act["object"]:
                    sel[aaa["y"], aaa["x"]] = 1
                    if 0 <= aaa["y"] < h and 0 <= aaa["x"] < w and aaa["color"] > 0:
                        grid[aaa["y"], aaa["x"]] = aaa["color"]

                sel = sel[:h, :w]

                if act["operation"] == "Copy" and act["source"] == "input":
                    op = "Copy-I"
                elif act["operation"] == "Copy" and act["source"] == "output":
                    op = "Copy-O"
                # if act['operation'] == 'Copy':
                #     maxy, maxx = 0, 0
                #     clip = np.zeros((30,30),dtype=int)
                #     for pix in act['special']:
                #         maxy = max(maxy, pix['y'])
                #         maxx = max(maxx, pix['x'])
                #         clip[pix['y'], pix['x']] = pix['color']
                #     clip = clip[:maxy+1, :maxx+1]

                statehash = hash(
                    str(grid)
                )  # +str(sel)+('submit' if act['operation'] == 'Submit' else ''))
                traj_hashseq.append(statehash)
                graphnodes[statehash] += 1
                nodedata[statehash] = {"grid": grid, "sel": sel, "op": op}

                graphedges[(prevhash, hash(str(op)), statehash)] += 1
                if prevhash != statehash:
                    graphneighbours[prevhash][statehash] += 1
                    graph_outdegrees[prevhash] += 1
                    graph_indegrees[statehash] += 1
                edgedata[(prevhash, hash(str(op)), statehash)] = op
                if act["operation"] == "Submit":

                    graphnodeend[statehash] = +1 if success else -1
                prevhash = statehash
            hash_seqs.append(traj_hashseq)
            succ_.append(success)
            validtrajectory.append(
                [probdf.iloc[trajind]["logId"], probdf.index.values[trajind]]
            )
        except:
            # print("error")
            errorcount += 1
    # print(errorcount, "errors")

    # ---------------------------------------------------------------------------

    # Node/Edge Statistics-------------------------------------------------------
    adj = np.zeros((len(graphnodes), len(graphnodes)), dtype=int)
    tot_edges = 0
    for edge in graphedges.keys():
        if edge[0] == edge[2]:
            continue
        adj[
            list(graphnodes.keys()).index(edge[0]),
            list(graphnodes.keys()).index(edge[2]),
        ] += graphedges[edge]
        tot_edges += graphedges[edge]

    # random walk
    totent = 0
    lenkey = 0
    totdeg = 0
    for node in graphnodes.keys():
        for neighbour in graphneighbours[node].keys():
            prob = graphneighbours[node][neighbour] / graph_outdegrees[node]
            if prob > 0:
                totent += -prob * np.log2(prob)
        totdeg += graph_outdegrees[node]
        lenkey += 1

    ent = totent / (lenkey * np.log2(lenkey - 1))

    n_trajs = len(probdf) - errorcount - fraudcount
    print(
        f"{errorcount} error / {fraudcount} fraud trajectories; Use {n_trajs} trajectories for problem {prob_id}."
    )
    nodesizedist = defaultdict(int)
    outdegdist = defaultdict(int)
    totout = 0
    totin = 0
    for node in graphnodes.keys():
        if graph_indegrees[node] == 0:
            continue
        nodesizedist[graph_indegrees[node]] += 1
        totin += 1
        for neighbour in graphneighbours[node].keys():
            outdegdist[graphneighbours[node][neighbour]] += 1
            totout += 1

    avg_nodesize = 0
    for key in nodesizedist.keys():
        avg_nodesize += key * nodesizedist[key] / totin
    ns_unscaled = avg_nodesize
    avg_nodesize /= n_trajs
    ns = avg_nodesize

    avg_outdeg = 0
    for key in outdegdist.keys():
        avg_outdeg += key * outdegdist[key] / totout
    od_unscaled = avg_outdeg
    avg_outdeg /= n_trajs
    od = avg_outdeg
    intention_nodes = set()
    # -------------------------------------------------------------------------

    # Graph visualization-------------------------------------------------------
    sizef = lambda s: 5 + 10 * np.log(s)

    g = Network(height="1080px", width="100%", directed=True)
    g.inherit_edge_colors(False)
    g.force_atlas_2based()

    g.add_node(
        starthash,
        label="start",
        title=tostr(init_grid, np.zeros(init_grid.shape, dtype=int)),
        size=sizef(graphnodes[starthash]),
        color="blue",
    )
    intention_nodes.add(starthash)
    for node in graphnodes.keys():
        prop = nodedata[node]
        grid = prop["grid"]
        sel = prop["sel"]

        if grid is not None:
            title = f"Indegree={graph_indegrees[node]}\n" + tostr(
                grid, np.zeros(grid.shape, dtype=int)
            )
        else:
            title = ""
        pop = False

        if (
            popularity_predicate(n_trajs, avg_nodesize, graph_indegrees[node])
            and graphnodeend[node] != 1
        ):
            intention_nodes.add(node)
            pop = True

        if graphnodeend[node] == -1:
            g.add_node(
                node, label="wrong", color="red", title=title, value=graphnodes[node]
            )

        elif graphnodeend[node] == 1:
            g.add_node(
                node,
                label="correct",
                color="green",
                title=title,
                value=graphnodes[node],
            )
            intention_nodes.add(node)
        elif pop:
            g.add_node(
                node,
                color="orange",
                title=title,
                value=graphnodes[node],
            )
        else:
            g.add_node(node, color="gray", title=title, value=graphnodes[node])

    for edge in graphedges.keys():
        if edge[0] == edge[2]:
            continue
        g.add_edge(
            edge[0],
            edge[2],
            label=edgedata[edge],
            title=f"{edgedata[edge]}\n{graphedges[edge]} transitions",
            value=graphedges[edge],
            color="rgba(128,128,128,0.5)",
            arrowStrikethrough=False,
        )

    g.show_buttons()
    g.save_graph(f"graphs_wosel_wofraud/{prob_id}.html")

    # g.show(f"{prob_id}.html", notebook=False)
    # print(intention_nodes)

    # misalignment_classification ---------------------------------------------------
    tot_nomiss = 0
    tot_nomiss_length = 0
    tot_tool_task = 0
    tot_user_tool = 0
    tot_user_task = 0
    tot_tool_task_length = 0
    tot_user_tool_length = 0
    tot_user_task_length = 0
    tot_intention = 0
    tot_intention_length = 0

    for success, traj_seq in zip(succ_, hash_seqs):

        prev_intention_node = starthash
        last_intention_distance = 0

        intention_segments = []
        nomiss = 0
        nomiss_length = 0
        usertoolmiss = 0
        usertoolmiss_length = 0
        tooltaskmiss = 0
        tooltaskmiss_length = 0
        usertaskmiss = 0
        usertaskmiss_length = 0

        for i in range(len(traj_seq) - 1):
            last_intention_distance += 1

            begin_node = traj_seq[i]
            end_node = traj_seq[i + 1]
            # print(begin_node, end_node, last_intention_distance)

            begin_is_intention = begin_node in intention_nodes
            end_is_intention = end_node in intention_nodes

            # print(begin_is_intention, end_is_intention)

            if i == len(traj_seq) - 2 and graphnodeend[end_node] == -1:  # wrong!
                # print("Wrong trajectory detected!")
                intention_segments.append(last_intention_distance)
                usertaskmiss += 1
                usertaskmiss_length += last_intention_distance

            elif begin_is_intention and end_is_intention:
                prev_intention_node = end_node
                # Great! No misalignment
                intention_segments.append(last_intention_distance)
                nomiss += 1
                nomiss_length += last_intention_distance
                last_intention_distance = 0

            elif begin_is_intention and (not end_is_intention):
                # keep and check later
                prev_intention_node = begin_node
                last_intention_distance = 1

            elif (not begin_is_intention) and end_is_intention:

                if last_intention_distance > 1:
                    if end_node in graphneighbours[prev_intention_node]:
                        # there is a direct path; user is babo
                        # print(
                        #     "User-Tool misalignment detected!", last_intention_distance
                        # )
                        intention_segments.append(last_intention_distance)
                        usertoolmiss += 1
                        usertoolmiss_length += last_intention_distance
                    else:
                        # no direct path; function deficiency
                        # print(
                        #     "Tool-Task misalignment detected!", last_intention_distance
                        # )
                        intention_segments.append(last_intention_distance)
                        tooltaskmiss += 1
                        tooltaskmiss_length += last_intention_distance

                    # misalignment
                    prev_intention_node = end_node
                    last_intention_distance = 0

        # pprint(sum(intention_segments), len(intention_segments), len(traj_seq) - 1)
        assert sum(intention_segments) == len(traj_seq) - 1
        tot_nomiss += nomiss
        tot_nomiss_length += nomiss_length
        tot_user_task += usertaskmiss
        tot_user_task_length += usertaskmiss_length
        tot_user_tool += usertoolmiss
        tot_user_tool_length += usertoolmiss_length
        tot_tool_task += tooltaskmiss
        tot_tool_task_length += tooltaskmiss_length
        tot_intention += len(intention_segments)
        tot_intention_length += len(traj_seq) - 1
        trajwise.append(
            (
                len(intention_segments),
                sum(intention_segments),
                nomiss,
                usertoolmiss,
                usertaskmiss,
                tooltaskmiss,
                nomiss_length,
                usertoolmiss_length,
                usertaskmiss_length,
                tooltaskmiss_length,
            )
        )

        # print(
        #     f"Length/ Correct: {nomiss_length/(len(traj_seq)-1)*100:.2f}%, User-Tool: {usertoolmiss_length/(len(traj_seq)-1)*100:.2f}%, Tool-Task: {tooltaskmiss_length/(len(traj_seq)-1)*100:.2f}%, , User-Task: {usertaskmiss_length/(len(traj_seq)-1)*100:.2f}%"
        # )
        # print(
        #     f"Count/  Correct: {nomiss/len(intention_segments)*100:.2f}%, User-Tool: {usertoolmiss/len(intention_segments)*100:.2f}%, Tool-Task: {tooltaskmiss/len(intention_segments)*100:.2f}%, User-Task: {usertaskmiss/len(intention_segments)*100:.2f}%"
        # )
        # print()
    return (
        n_trajs,
        (ent, ns, ns_unscaled, od, od_unscaled),
        (
            tot_nomiss,
            tot_nomiss_length,
            tot_user_tool,
            tot_user_tool_length,
            tot_user_task,
            tot_user_task_length,
            tot_tool_task,
            tot_tool_task_length,
            tot_intention,
            tot_intention_length,
        ),
    )
    # --------------------------------------------------------------------------------


""" 
def detect(prob_id):

    init_grid = arckit.load_single(id="train" + str(prob_id - 1)).test[0][0]

    probdf = df[df["taskId"] == prob_id]
    # print(f"{len(probdf)} records found for problem {prob_id}.")

    # Graph Data Processing -------------------------------------------------------
    graphnodes = defaultdict(int)
    graphedges = defaultdict(int)
    graphneighbours = defaultdict(lambda: defaultdict(int))
    graph_indegrees = defaultdict(int)
    graph_outdegrees = defaultdict(int)
    graphnodeend = defaultdict(int)
    nodedata = defaultdict(lambda: {"grid": None, "sel": None, "op": None})
    edgedata = defaultdict(str)
    hash_seqs = []
    succ_ = []

    errorcount = 0
    fraudcount = 0
    starthash = hash(str(init_grid))
    for trajind in range(len(probdf)):

        success = probdf.iloc[trajind]["success"]
        tt = probdf.iloc[trajind]["dt"].total_seconds()

        traj = json.loads(probdf.iloc[trajind]["actionSequence"])
        yes_copy = False
        yes_paste = False
        for act in traj:
            op = act["operation"]
            if op == "Copy":
                yes_copy = True
            elif op == "Paste":
                yes_paste = True

        if (
            tt < 10 and success and len(traj) <= 6 and (not yes_copy and yes_paste)
        ):  # fraud detection
            # print("FRAUD at ----------------------------------", trajind, tt)
            print(probdf.iloc[trajind]["userId"])
            continue

"""
if __name__ == "__main__":

    with open("misalignments.csv", "w") as f1:
        f1.write(
            "Task_ID,Total_Traj,entropy,nodesize,nodesize_unscaled,outdegree,outdegree_unscaled,Total_Intentions,Total_Intention_Length,No_Miss_Rate,User_Tool_Rate,User_Task_Rate,Tool_Task_Rate,No_Miss_Length_Rate,User_Tool_Length_Rate,User_Task_Length_Rate,Tool_Task_Length_Rate,No_Miss,User_Tool,User_Task,Tool_Task,No_Miss_Length,User_Tool_Length,User_Task_Length,Tool_Task_Length\n"
        )
        results = []
        for i in range(1, 401):
            prob_id = i
            n, stat, rs = classify(prob_id)
            (
                nomiss,
                nomiss_l,
                usertool,
                usertool_l,
                usertask,
                usertask_l,
                tooltask,
                tooltask_l,
                n_intention,
                n_intention_l,
            ) = rs
            (ent, ns, ns_unscaled, od, od_unscaled) = stat
            assert (nomiss + usertask + usertool + tooltask) == n_intention and (
                nomiss_l + usertask_l + usertool_l + tooltask_l
            ) == n_intention_l
            print(
                f"\
                Count / Correct: {nomiss/n_intention*100:.2f}%, User-Tool: {usertool/n_intention*100:.2f}%, Tool-Task: {tooltask/n_intention*100:.2f}%, User-Task: {usertask/n_intention*100:.2f}\n\
                Length/ Correct: {nomiss_l/n_intention_l*100:.2f}%, User-Tool: {usertool_l/n_intention_l*100:.2f}%, Tool-Task: {tooltask_l/n_intention_l*100:.2f}%, User-Task: {usertask_l/n_intention_l*100:.2f}%"
            )
            print()
            f1.write(
                f"{prob_id},{n},{ent},{ns},{ns_unscaled},{od},{od_unscaled},"
                + ",".join(
                    map(
                        str,
                        list(rs[-2:])
                        + [x / n_intention * 100 for x in rs[0:8:2]]
                        + [x / n_intention_l * 100 for x in rs[1:8:2]]
                        + list(rs[0:8:2])
                        + list(rs[1:8:2]),
                    )
                )
                + "\n"
            )

            results.append(rs)

    with open("valid.csv", "w") as f:
        f.write(
            ",".join(
                [
                    "logId",
                    "id",
                    "Intentions",
                    "Intentions_Length",
                    "No_Miss",
                    "User_Tool",
                    "User_Task",
                    "Tool_Task",
                    "No_Miss_Length",
                    "User_Tool_Length",
                    "User_Task_Length",
                    "Tool_Task_Length",
                ]
            )
            + "\n"
        )
        for i, j in zip(validtrajectory, trajwise):
            f.write(str(i[1]) + "," + str(i[0]) + "," + ",".join(map(str, j)) + "\n")
