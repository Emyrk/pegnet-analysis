# Homework 09 PCA
# Authors:
#   Steven Masley
#   Pablo Ordorica

import argparse
import pandas as pd
import numpy as np
import math
from scipy.cluster import hierarchy
from sklearn import cluster
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.spatial import Delaunay
from sklearn import preprocessing


import pdb

PLOT_PROJECTION = False
PRINT_LATEX_TABLES = False
PLOT_KMEANS_SSE = False

COLOR_MAP = [
    'blue',
    'saddlebrown',
    'cyan',
    'green',
    'orange',
    'm',
    'k',
    'grey',
    'teal',
    'w',
    'yellow',
    'hotpink', 'blueviolet', 'deepskyblue', 'olive', 'burlywood', 'linen']

def main():
    # Argparse for various cli flags to change program behavior
    parser = argparse.ArgumentParser(description='PCA HW.')
    parser.add_argument('--data', metavar='FILENAME', type=str,
                        help='File where training data can be found', default='206422.csv')
    parser.add_argument("-a", action='store_true',
                    help="Run agglomeration. This will prevent the PCA from running", default=False)
    parser.add_argument("-p", action='store_true',
                    help="Plot Projection", default=False)
    parser.add_argument("-l", action='store_true',
                    help="Print latex tables", default=False)
    parser.add_argument("-k", action='store_true',
                    help="Plot Kmeans sse graph", default=False)
    args = parser.parse_args()

    global PRINT_LATEX_TABLES, PLOT_PROJECTION, PLOT_KMEANS_SSE, PLOT_CLASSIFY
    PRINT_LATEX_TABLES = args.l
    PLOT_PROJECTION = args.p
    PLOT_KMEANS_SSE = args.k

    # Data to be used
    print("Using data from file: %s" % args.data)
    df = pd.read_csv(args.data)
    if args.a:
        agglomeration(df)
    else:
        df.drop(df.columns[0], axis=1, inplace=True)
        pts = df.values
        # pts = pts[:50]
        # Normalize all columns
        pts = preprocessing.normalize(pts)

        pts_t = pts.transpose()

        # Find eigen vectors
        covariance_matrix = np.cov(pts_t)
        (eig_vl, eig_vt) = np.linalg.eig(covariance_matrix)
        eig_z = list(zip(eig_vl, eig_vt))
        eig_z.sort(key=lambda x: abs(x[0]), reverse=True)

        # Prints out the latex for the table in the writeup
        latex_table_eig_vectors(list(df.columns), eig_z)

        # Projected Data
        proj = project_onto(pts, [eig_vt[0], eig_vt[2], eig_vt[3]])
        print(len(pts), len(proj))
        (clusters, df_centroids) = scattergram(proj, pts, eig_vt, list(df.columns))
    plt.show()

# project_onto 3d
def project_onto(data, eigen_vectors):
    result = []
    # For each data point in data, project it
    for i in range(len(data)):
        tmp =[]
        for j in range(len(eigen_vectors)):
            # Dot product
            tmp.append(np.dot(data[i], eigen_vectors[j]))
        result.append(tmp)
    return result

# scattergram plots the 3d scatter plot
def scattergram(proj, pts, eig_vt, columns):
    df = pd.DataFrame(np.array(proj), columns=["EigenVector 1", "EigenVector 3", "EigenVector 4"])
    fig, ax = None, None
    if PLOT_PROJECTION:
        fig = plt.figure()
        ax = Axes3D(fig)

    # Used to determine the number of clusters
    if PLOT_KMEANS_SSE:
        kmeans_sse_plot(proj)

    # Find best kmeans
    best_res = None
    # try:
    proj = np.array(proj).astype(float)
    for i in range(50):
        res = kmeans(proj, 1)
        if best_res is None:
            best_res = res
        if res[1] < best_res[1]:
            best_res = res
    # except:
    #     print("Kmeans failed")
    #     best_res =[
    #         [[0, 0, 0]],
    #         0
    #     ]


    dfRes = pd.DataFrame(np.array(best_res[0]), columns=["EigenVector 1", "EigenVector 3", "EigenVector 4"])

    # Get centroids and mediods from kmeans
    centroids = np.array(best_res[0])
    groups = [ [] for i in range(len(centroids))]
    orig_group_size = [ 0 for i in range(len(centroids))]
    medoid_dist = [ math.inf for i in range(len(centroids))]
    medoids = [ 0 for i in range(len(centroids))]
    medoids_t_avg = [ 0 for i in range(len(centroids))]

    cnt = 0
    # Find the mediods and group the data to color the clusters
    #   To classify, find closest cluster centroid to a point
    for p in proj:
        best = 0
        smallest_dist = math.inf
        for i in range(len(centroids)):
            dist = np.linalg.norm(centroids[i]-p)
            if dist < smallest_dist:
                smallest_dist = dist
                best = i
        groups[best].append(p)
        # Find medoids
        if smallest_dist < medoid_dist[best]:
            medoid_dist[best] = smallest_dist
            medoids[best] = cnt
        medoids_t_avg[best] += sum(pts[cnt])
        cnt += 1

    # Send mediod stats for printing
    #   Mainly used to make latex tables
    medoid_stat = [ [] for i in range(len(centroids))]
    for i in range(len(groups)):
        medoid_stat[i] = {"group_size":len(groups[i]), "med_id":medoids[i], "pt":pts[medoids[i]], "grp_avg":medoids_t_avg[i]/len(groups[i])}
    medoid_stat.sort(key=lambda x: x["group_size"])

    latex_table_medioids(medoid_stat, columns)

    # Sort by cluster size so the coloring is consistent across runs
    groups.sort(key=lambda x: len(x))

    # For the latex writeup
    table_info = []
    for i in range(len(groups)):
        tmp = [len(groups[i])]
        tmp.extend(centroids[i])
        table_info.append(tmp)


    # Will print the latex table about the kmeans clusters to put
    #   in the writeup
    latex_table_cluster_size(table_info)

    if PLOT_PROJECTION:
        # Plot Centroids
        ax.scatter(dfRes["EigenVector 1"], dfRes["EigenVector 3"], dfRes["EigenVector 4"], color="r", s=100)
        # Plot each cluster as it's own color
        for i in range(len(groups)):
            # Add jitter
            g = np.array(groups[i])
            noise = [
                np.random.normal(0, abs(np.mean(g[:,0]))/100000,len(groups[i])),
                np.random.normal(0, abs(np.mean(g[:,1]))/100000,len(groups[i])),
                np.random.normal(0, abs(np.mean(g[:,2]))/100000,len(groups[i]))
            ]
            # pdb.set_trace()

            jitterGrp = g + np.array(noise).transpose()

            dfg = pd.DataFrame(np.array(jitterGrp[:50]), columns=["EigenVector 1", "EigenVector 3", "EigenVector 4"])
            ax.scatter(dfg["EigenVector 1"], dfg["EigenVector 3"], dfg["EigenVector 4"], color=COLOR_MAP[i], s=2)

            dfg = pd.DataFrame(np.array(jitterGrp[50:]), columns=["EigenVector 1", "EigenVector 3", "EigenVector 4"])
            ax.scatter(dfg["EigenVector 1"], dfg["EigenVector 3"], dfg["EigenVector 4"], color=COLOR_MAP[i+1], s=1)

        ax.set_xlabel("EigenVector 1")
        ax.set_ylabel("EigenVector 3")
        ax.set_zlabel("EigenVector 4")
        ax.set_title("20D data projected onto Eigen Vectors 1, 3, and 4")
        # plt.show()

    return (list(zip(COLOR_MAP, groups)), dfRes)

def kmeans_sse_plot(data):
    xAxis = np.arange(1, 10)
    sse = []
    for i in xAxis:
        tmp = []
        for _ in range(100):
            (centroids, kmeans_sse) = kmeans(data, i)
            tmp.append(kmeans_sse)
        sse.append(min(tmp))

    df = pd.DataFrame(sse, xAxis[:len(sse)])
    graph = df.plot(style="-o", title="Sum of Squared Errors with KMeans")

    # Label axes
    graph.set(xlabel="# of clusters", ylabel="Sum of Squared Errors")

    # plt.show()

# Latex table formatting
def latex_table_medioids(medoids, columns):
    if not PRINT_LATEX_TABLES:
        return

    clusters = [ COLOR_MAP[i] for i in range(len(medoids))]
    table = """
\\begin{tabular}{|c|c|c|c|c|c|c|}
\\hline
Cluster & %s & %s & %s & %s & %s & %s \\\\ \\hline
    """ % (COLOR_MAP[0],COLOR_MAP[1],COLOR_MAP[2],COLOR_MAP[3],COLOR_MAP[4],COLOR_MAP[5])

    for i in range(len(columns)):
        # print(medoids[i])
        table += "%s & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f  \\\\ \\hline\n" % (columns[i],
            medoids[0]["pt"][i],
            medoids[1]["pt"][i],
            medoids[2]["pt"][i],
            medoids[3]["pt"][i],
            medoids[4]["pt"][i],
            medoids[5]["pt"][i])
    # Summation
    table += "%s & %.1f & %.1f & %.1f & %.1f & %.1f & %.1f  \\\\ \\hline\n" % ("Total",
            medoids[0]["grp_avg"],
            medoids[1]["grp_avg"],
            medoids[2]["grp_avg"],
            medoids[3]["grp_avg"],
            medoids[4]["grp_avg"],
            medoids[5]["grp_avg"],)

    table += "\n\\end{tabular}"
    print(table)

# Latex table formatting
def latex_table_cluster_size(clusters):
    if not PRINT_LATEX_TABLES:
        return
    clusters.sort(key=lambda x: abs(x[0]))
    table = """
\\begin{tabular}{|c|c|c|c|}
\\hline
Cluster Size & Eigen Vector 1   & Eigen Vector 3    & Eigen Vector 4 \\\\ \\hline
"""

    for i in range(len(clusters)):
        table += "%d & %.1f & %.1f & %.1f  \\\\ \\hline\n" % (clusters[i][0], clusters[i][1],clusters[i][2],clusters[i][3])

    table += "\n\\end{tabular}"
    print(table)

# Latex table formatting
def latex_table_eig_vectors(columns, eig_z):
    if not PRINT_LATEX_TABLES:
        return
    table = """
\\begin{tabular}{|c|c|c|c|c|c|}
\\hline
Eigen Vector & 1    & 2   & 3    & 4 \\\\ \\hline
    """

    for i in range(len(columns)):
        table += "%s & %.1f & %.1f & %.1f & %.1f  \\\\ \\hline\n" % (columns[i], eig_z[0][1][i], eig_z[1][1][i], eig_z[2][1][i], eig_z[3][1][i])

    table += "\n\\end{tabular}"
    print(table)

# Dendrogram graph
def agglomeration(df):
    df.drop(df.columns[0], axis=1, inplace=True)
    pts = df.values
    lm = hierarchy.linkage(pts, method="centroid")
    graph_dendrogram(lm, len(pts), "central")
    # last_smallest(lm, 20, len(pts))

# Print the last n clusters and the minium size of the clusters merged
def last_smallest(lm, n, data_amt):
    for lmi in lm[-20:]:
        c1 = lm[int(lmi[0]) - data_amt][3]
        if int(lmi[0]) - data_amt < 0:
            c1 = 1
        c2 = lm[int(lmi[1]) - data_amt][3]
        if int(lmi[1]) - data_amt < 0:
            c2 = 1

def graph_dendrogram(lm, amt, link):
    hierarchy.dendrogram(lm, labels=list(range(amt)), p=50, truncate_mode='lastp') #p=5, truncate_mode='lastp')
    # plot_dendrogram(lm)
    plt.title('Top 50 Clusters of Agglomerative Clustering with ' +
              link.capitalize() + ' Linkage', fontsize=20)
    plt.xlabel('Cluster ID')
    plt.ylabel('Euclidean Distance')
    plt.show()


# Launch program
if __name__ == "__main__":
    main()