import networkx as nx
import numpy as np
import pandas as pd
import random as rand
from collections import Counter

#Spreading Animation Related
import os
from pathlib import Path
import shutil
import imageio as iio

#Graphics
import plotly
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML


def generate_minimal_graph(n_nodes):
    """
    Generates a graph following the minimal model with the desired number of nodes (n_nodes)
    """
    g = nx.Graph()
    g.add_edges_from([(1,2), (2,3), (3,1)])
    for i in range(4, n_nodes+1):
        edges = list(g.edges)
        chosen_edge = rand.choice(edges)
        g.add_edge(chosen_edge[0], i)
        g.add_edge(chosen_edge[1], i)
    return g


def maki_thompson_rumour_model(graph: nx.Graph, gamma: float, alpha: float, time_limit: int, number_infected=1, all_time_values=False, starting_nodes=[], outputDictQ=True, animation=False, gif_name="spreading", gif_duration=10):
    """
    Simulates the spreading of a rumor, using the maki-thompson model with the desired parameters.

    graph: the network where we wish to spread the rumor
    gamma: the probability of a spreader to spread the rumor to an ignorant node
    alpha: the probability of a spreader becoming a stiffler when in contact with a spreader/stiffler
    time_limit: The desired time limit for the simulation
    number_infected: Number of initial spreaders desired
    all_time_values: If true, even when the simulation stop eraly, the return always return a list with length equal to the time_limit
    starting_nodes: option for selecting the starting spreader nodes
    outputDictQ: True if we want the population output as dict type, array c.c.
    animation: True if we wish to see an animation of the spreading (This will make the simulation much slower)
    gif_name: name of the generated gif file
    gif_duration: duration of the generated gif
    """
    #Clear status of the nodes of the given graph
    #"status" tells the type of node (I - Ignorant, S - Spreader, R - Stifler)
    #"visit" tells if the node should be visited in a cycle or not

    node_status={key: "I" for key in graph.nodes}
    node_visit={key: False for key in graph.nodes}
    nx.set_node_attributes(graph, node_status, "status")
    nx.set_node_attributes(graph, node_visit, "visit")

    #Obtain a random sample of size "number_infected" out of the nodes of the given graph
    if not(starting_nodes):
        starting_nodes = rand.sample(list(graph.nodes), number_infected)

    for node in starting_nodes:
        graph.nodes[node]["status"]="S"
        graph.nodes[node]["visit"]=True

        #All neighbours of spreaders should be visited in the cycle for possible changes
        for neighbor in graph.neighbors(node):
            graph.nodes[neighbor]["visit"]=True

    t=0

    #Counter for every status type, for each time value "t"
    counter=Counter(nx.get_node_attributes(graph, "status").values())
    if outputDictQ:
        status_count = [{"I": counter["I"], "S": counter["S"], "R": counter["R"]}]
    else:
        status_count= [[counter["I"], counter["S"], counter["R"]]]

    #Creates directory for the images used in the animation gif
    if animation:
        centrality = nx.degree_centrality(graph)
        centrality = np.fromiter(centrality.values(), float)
        dir=os.getcwd()
        folder_name="gif_images"
        folder_path=os.path.join(dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        color_array = ["grey" if x == "I" else "red" if x == "S" else "green" for x in list(nx.get_node_attributes(graph, "status").values())]
        fig = plt.figure()
        nx.draw(graph, node_color=color_array, ax=fig.add_subplot(), pos=nx.random_layout(graph, seed=16), node_size=centrality*1e3)
        matplotlib.use("Agg")
        fig.savefig(os.path.join(folder_path, str(t)))

    load=0
    #Checks if there is still nodes worth visiting in the next iteration
    continue_visitingQ=True
    while t < time_limit-1 and continue_visitingQ:
        #Only apply node_changes at the end of the cycle, the changes have to be done all at the same time.
        node_changes=[]
        for node in graph.nodes(data=True):
            if node[1]["visit"]==True:
                #Counting the amount of each type of nodes in the neighbourhood of the visited node
                neighbours_status_count={"I":0, "S":0, "R":0}
                for neighbor in graph.neighbors(node[0]):
                    neighbours_status_count[graph.nodes[neighbor]["status"]]+=1

                #If this value is less than the probability of transforming then the status of the node changes
                transform_value = rand.uniform(0,1)

                #Transformation of the node status
                if node[1]["status"]=="I":
                    #(1-gamma)**neighbours_status_count["S"] gives the probability of the event that no spreader transmits the rumour. We only need one of the spreaders to pass the rumour to transform the ignorant into spreader
                    transform_threshold = 1-(1-gamma)**neighbours_status_count["S"]
                    if transform_value <= transform_threshold:
                        node_changes.append((node[0], "S", True))
                        for neighbor in graph.neighbors(node[0]):
                            if graph.nodes[neighbor]["status"]!="R":
                                node_changes.append((neighbor, graph.nodes[neighbor]["status"], True))
                    load+=neighbours_status_count["S"]
                elif node[1]["status"]=="S":
                     #Whenever we are transforming a spreader into a stiffler the stifflers and the spreaders contribute the same to the probability of transforming the spreader
                    transform_threshold = 1-(1-alpha)**(neighbours_status_count["S"] + neighbours_status_count["R"])
                    if transform_value <= transform_threshold:
                        node_changes.append((node[0], "R", False))

                    load+=neighbours_status_count["S"]+neighbours_status_count["R"]
        
        #Applying all changes to the graph 
        for node_change in node_changes:
            graph.nodes[node_change[0]]["status"]=node_change[1]
            graph.nodes[node_change[0]]["visit"]=node_change[2]

            #Changes colors in color array for the nodes that were changed
            if animation:
                if node_change[1]=="I":
                    color_array[node_change[0]]="grey"
                elif node_change[1]=="S":
                    color_array[node_change[0]]="red"
                else:
                    color_array[node_change[0]]="green"
        
        counter=Counter(nx.get_node_attributes(graph, "status").values())

        if outputDictQ:
            status_count.append({"I": counter["I"], "S": counter["S"], "R": counter["R"]})
        else:
            status_count.append([counter["I"], counter["S"], counter["R"]])

        #Stops the simulation when there are no more spreaders
        if counter["S"]==0:
            if not(all_time_values):
                continue_visitingQ=False
            else:
                for i in range(time_limit-t-2):
                    if outputDictQ:
                        status_count.append({"I": counter["I"], "S": counter["S"], "R": counter["R"]})
                    else:
                        status_count.append([counter["I"], counter["S"], counter["R"]])
                continue_visitingQ=False

        t+=1

        #Creates the plot of the graph for the current timestamp 
        if animation:
            fig = plt.figure()
            nx.draw(graph, node_color=color_array, ax=fig.add_subplot(), pos=nx.random_layout(graph, seed=16), node_size=centrality*1e3)
            matplotlib.use("Agg")
            fig.savefig(os.path.join(folder_path, str(t)))

    load=load/len(graph)

    #Joins the created images into a gif and deletes the directory 
    if animation:
        images = []
        img_names_order = sorted(list(Path(folder_path).iterdir()),
                            key=lambda path: int(path.stem))
        for img in img_names_order:
            images.append(iio.v3.imread(img))
        iio.v2.mimsave(gif_name+".gif", images, duration = gif_duration)
        shutil.rmtree(folder_path)
    return status_count, load, [i[1] for i in list(graph.degree())], list(nx.get_node_attributes(graph, "status").values())


def simulation(name, n_nodes, gamma_name, gamma=1, ws_prob=0.01, n_graph_per_iteration=10):
    """
    Simulates multiple instances of rumor spreading, with the desired parameters, for multiple alphas, and saves the results to multiple numpy arrays
    """

    population_results=[]
    degrees=[]
    mean_loads=[]
    status=[]
    #It shouldn't have started at 0
    for alpha in np.linspace(0,1,20):
        alpha_population_results=[]
        alpha_degrees=[]
        alpha_mean_loads=[]
        alpha_status=[]
        for j in range(n_graph_per_iteration):
            if name=="BA":
                graph=nx.barabasi_albert_graph(n=n_nodes,m=2)
            elif name=="MM":
                graph=generate_minimal_graph(n_nodes=n_nodes)
            else:
                graph=nx.watts_strogatz_graph(n=n_nodes, k=4, p=ws_prob)
            sim,load,degree,final_status=maki_thompson_rumour_model(graph, gamma=gamma, alpha=alpha, time_limit=1000, number_infected=1, all_time_values=True, outputDictQ=False)
            alpha_population_results.append(sim)
            alpha_degrees.append(degree)
            alpha_mean_loads.append(load)
            alpha_status.append(final_status)
        population_results.append(alpha_population_results)
        degrees.append(alpha_degrees)
        mean_loads.append(alpha_mean_loads)
        status.append(alpha_status)

    dir=os.getcwd()
    folder_name="gamma"+gamma_name
    folder_path = os.path.join(dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(os.path.join(folder_path,name+"_population_"+gamma_name),np.array(population_results))
    np.save(os.path.join(folder_path,name+"_degrees_"+gamma_name), np.array(degrees))
    np.save(os.path.join(folder_path,name+"_loads_"+gamma_name), np.array(mean_loads))
    np.save(os.path.join(folder_path,name+"_status_"+gamma_name), np.array(status))


def simulation_graph_size(name, alpha, gamma, ws_prob=0.1, n_graph_per_iteration=10):
    """Similar to simulation function, but instead changes the network size instead of the alpha parameter"""
    population_results=[]
    mean_loads=[]

    for n_nodes in list(map(int,np.linspace(0,10000,11)[1:])):
        alpha_population_results=[]
        alpha_mean_loads=[]
        for j in range(n_graph_per_iteration):
            if name=="BA":
                graph=nx.barabasi_albert_graph(n=n_nodes,m=2)
            elif name=="MM":
                graph=generate_minimal_graph(n_nodes=n_nodes)
            else:
                graph=nx.watts_strogatz_graph(n=n_nodes, k=4, p=ws_prob)
            sim,load,degree,final_status=maki_thompson_rumour_model(graph, gamma=gamma, alpha=alpha, time_limit=1000, number_infected=1, all_time_values=True, outputDictQ=False)
            alpha_population_results.append(sim)
            alpha_mean_loads.append(load)
        population_results.append(alpha_population_results)
        mean_loads.append(alpha_mean_loads)

    dir=os.getcwd()
    folder_name="variable_graph_size"
    folder_path = os.path.join(dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(os.path.join(folder_path,name+"_population"),np.array(population_results))
    np.save(os.path.join(folder_path,name+"_loads"), np.array(mean_loads))


def plot_line_chart(x, y1, y2, y3, y4=np.array([]), title="", x_title="", y_title=""):

    plotly.offline.init_notebook_mode()
    display(HTML(
    '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
    ))

    fig = go.Figure()

    # Add the first line
    fig.add_trace(go.Scatter(x=x, y=y1,mode='lines+markers', name='BA'))

    # Add the second line
    fig.add_trace(go.Scatter(x=x, y=y2,mode='lines+markers', name='WS'))

    #Add the third line
    fig.add_trace(go.Scatter(x=x, y=y3,mode='lines+markers', name='MM'))

    # Add the forth line
    if y4.size>0:
        fig.add_trace(go.Scatter(x=x, y=y4, mode='lines+markers', name='WS1'))

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template='plotly',
        plot_bgcolor='white',
        barmode='stack',
        width=600,
        height=500
    )

    fig.update_xaxes(
        mirror=False,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=False,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    return fig


def plot_3_line_chart(x, y1, y2, y3, title="", x_title="", y_title=""):
    fig = go.Figure()

    # Add the first line
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Ignorant'))

    # Add the second line
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Spreader'))

    # Add the second line
    fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', name='Stifler'))

    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)

    fig.update_layout(
        title=title,
        template='plotly',
        plot_bgcolor='white',
        width=800,
        height=400
    )

    fig.update_xaxes(
        mirror=False,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=False,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    return fig


def plot_stacked_area_chart(x, y1, y2, y3, title="", x_title="", y_title=""):
    fig = go.Figure()

    # Add the first line
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', fill='tozeroy', name='Ignorant', stackgroup='one',
    groupnorm='percent'))

    # Add the second line
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', fill='tonexty', name='Spreader', stackgroup='one'))

    # Add the second line
    fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', fill='tonexty', name='Stifler', stackgroup='one'))

    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)

    fig.update_layout(
        title=title,
        template='plotly',
        plot_bgcolor='white',
        barmode='stack',
        width=800,
        height=400
    )

    fig.update_xaxes(
        mirror=False,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=False,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    return fig


def get_rk(status_matrix, degree_matrix):
    res = {}
    for a in range(np.shape(status_matrix)[0]):
        r_k = {}
        total_k = {}
        for sim in range(np.shape(status_matrix)[1]):
            statuses = status_matrix[a,sim,:]
            degs = degree_matrix[a,sim,:]
            for sts, degree in zip(statuses, degs):
                if degree in total_k.keys():
                    total_k[degree] += 1
                else:
                    total_k[degree] = 1
                if sts == 'R':
                    if degree in r_k.keys():
                        r_k[degree] += 1
                    else:
                        r_k[degree] = 1
        res[a] = [(deg, 1 - (count / total_k[deg])) for deg, count in r_k.items()]
        
    for k in res.keys():
        res[k].sort(key=lambda t: t[0])

    return res