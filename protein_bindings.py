import numpy as np 
import pandas as pd 
import gudhi as gd
from pylab import *
from IPython.display import Image 
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn import manifold 


##Import file list (even indicies are closed, odd are open)
files_list = ['/users/kevinweiss/masters_project/1anf.corr_1.txt', '/users/kevinweiss/masters_project/1ez9.corr_1.txt',
         '/users/kevinweiss/masters_project/1fqc.corr_2.txt', '/users/kevinweiss/masters_project/1fqb.corr_3.txt','/users/kevinweiss/masters_project/1mpd.corr_4.txt',
         '/users/kevinweiss/masters_project/3hpi.corr_5.txt']


##Convert data to CSV
corr_list = [pd.read_csv(u , header=None, delim_whitespace=True) for u in files_list]

##Construct distance matrix
dist_list = [1-np.abs(c) for c in corr_list]


##Initialize arrays
D = [0]*len(dist_list)
skeleton_protein = [0]*len(dist_list)
Rips_Simplex_Tree_Protein = [0]*len(dist_list)
BarCodes_Rips = [0]*len(dist_list)
Pers_Interval = [0]*len(dist_list)

##Evaluate persistence and persistence intervals from Vietoris-Rips Complex 
for i in range(0,len(dist_list)):
    D[i] = dist_list[i]

    skeleton_protein[i] =  gd.RipsComplex(
    distance_matrix = D[i].values, 
    max_edge_length =1.0)

    Rips_Simplex_Tree_Protein[i] = skeleton_protein[i].create_simplex_tree(max_dimension = 2)

    BarCodes_Rips[i] = Rips_Simplex_Tree_Protein[i].persistence()

    Pers_Interval[i] = Rips_Simplex_Tree_Protein[i].persistence_intervals_in_dimension(0)


#Plot persistence diagrams of a closed MBP
gd.plot_persistence_diagram(BarCodes_Rips[0])
plt.show()


##Plot persistence diagram of open MBP
gd.plot_persistence_diagram(BarCodes_Rips[3])
plt.show()




print("Bottleneck dist for two open configurations:",gd.bottleneck_distance(Pers_Interval[5],Pers_Interval[4]))

print("Bottleneck dist for one open and one closed configurations:",gd.bottleneck_distance(Pers_Interval[0],Pers_Interval[3]))

persistence_list0 = []
persistence_list1 = []

i = 0 

########################## Calculate persistence in 0 and 1 dimension ##########################
########## Use multidimensional scaling to visualize different persistences ###################

persistence_list0 = []
persistence_list1 = []

i=0
for d in dist_list:
    #print(i)
    rips_complex = gd.RipsComplex(
        distance_matrix = d.values, 
        max_edge_length = 0.8
    )
    simplex_tree = rips_complex.create_simplex_tree(max_dimension = 2)
    diag = simplex_tree.persistence()
    persistence_list0.append(simplex_tree.persistence_intervals_in_dimension(0))
    persistence_list1.append(simplex_tree.persistence_intervals_in_dimension(1))
    i = i + 1

l = len(files_list)
B0 = np.zeros((l,l))
B1 = np.zeros((l,l))


for i in range(l):
    for j in range(i):
        B0[i,j] = gd.bottleneck_distance(persistence_list0[i], persistence_list0[j])
        B1[i,j] = gd.bottleneck_distance(persistence_list1[i], persistence_list1[j])

B0 = B0 + B0.transpose()
B1 = B1 + B1.transpose()


mds = manifold.MDS(
    n_components = 2, 
    max_iter = 3000, 
    eps = 1e-9, 
    dissimilarity = "precomputed", 
    n_jobs = 1
)



pos = mds.fit(B0).embedding_

plt.scatter(pos[0:3, 0], pos[0:3, 1], color = 'red' , label = "closed")
plt.scatter(pos[3:l, 0], pos[3:l, 1], color = 'blue', label = "open")
plt.legend(loc = 3, borderaxespad = 1)
plt.show()

pos2 = mds.fit(B1).embedding_

plt.scatter(pos2[0:3, 0], pos2[0:3, 1], color = 'red' , label = "closed")
plt.scatter(pos2[3:l, 0], pos2[3:l, 1], color = 'blue', label = "open")
plt.legend(loc = 3, borderaxespad = 1)
plt.show()