
import networkx as nx
from RJW import *
import seaborn as sns
from custom_svc import *
from ot_distances import RJW_distance
import matplotlib.pyplot as plt
from graph import  Graph
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np




np.random.seed(50)

X_2_train = []
for _ in range(200): 
    
    points = np.random.rand(15,2)
    # graph avec 3nn sur ce jeu de points
    G = nx.Graph()
    G.add_nodes_from(range(len(points)))
    for i in range(len(points)):
        distces = np.linalg.norm(points-points[i],axis=1)
        distces[i] = np.inf
        nn = np.argsort(distces)[:2]
        G.add_edges_from([(i,nn[0]),(i,nn[1])])
    
    y = points
    nx.set_node_attributes(G, dict(zip(G.nodes(),[points[i] for i in G.nodes()])), 'y')
    X_2_train.append(G)

X_3_train = []

for _ in range(200):
    
    points = np.random.rand(15,2)
    # graph avec 3nn sur ce jeu de points
    G = nx.Graph()
    G.add_nodes_from(range(len(points)))
    for i in range(len(points)):
        distces = np.linalg.norm(points-points[i],axis=1)
        distces[i] = np.inf
        nn = np.argsort(distces)[:3]
        G.add_edges_from([(i,nn[0]),(i,nn[1]), (i,nn[2])])
    y = points
    nx.set_node_attributes(G, dict(zip(G.nodes(),[points[i] for i in G.nodes()])), 'y')
    X_3_train.append(G)

X_2_test = []


for _ in range(50):
    
    points = np.random.rand(15,2)
    # graph avec 3nn sur ce jeu de points
    G = nx.Graph()
    G.add_nodes_from(range(len(points)))
    for i in range(len(points)):
        distces = np.linalg.norm(points-points[i],axis=1)
        distces[i] = np.inf
        nn = np.argsort(distces)[:2]
        G.add_edges_from([(i,nn[0]),(i,nn[1])])
    y = points
    nx.set_node_attributes(G, dict(zip(G.nodes(),[points[i] for i in G.nodes()])), 'y')
    X_2_test.append(G)


X_3_test = []


for _ in range(50):
    
    points = np.random.rand(15,2)
    # graph avec 3nn sur ce jeu de points
    G = nx.Graph()
    G.add_nodes_from(range(len(points)))
    for i in range(len(points)):
        distces = np.linalg.norm(points-points[i],axis=1)
        distces[i] = np.inf
        nn = np.argsort(distces)[:3]
        G.add_edges_from([(i,nn[0]),(i,nn[1]), (i,nn[2])])
    y = points
    nx.set_node_attributes(G, dict(zip(G.nodes(),[points[i] for i in G.nodes()])), 'y')
    X_3_test.append(G)




X_train, y_train = (X_2_train + X_3_train),np.array( [0]*len(X_2_train) + [1]*len(X_3_train))

X_test, y_test = (X_2_test + X_3_test), np.array([0]*len(X_2_test) + [1]*len(X_3_test))



def convert_list_in_array(l): 
    array = np.empty(len(l), dtype=object)
    for i in range(len(l)):
        array[i] = l[i]
    return array


X_train_bis = []
X_test_bis = []

for g in X_train: 
    my_graph = Graph(g)
    my_graph.distance_matrix()
    X_train_bis.append(my_graph)
for g in X_test: 
    my_graph = Graph(g)
    my_graph.distance_matrix()
    X_test_bis.append(my_graph)


X_train_bis = convert_list_in_array(X_train_bis)
X_test_bis = convert_list_in_array(X_test_bis)






if __name__ =='__main__' : 
    for g_w in [0.1, 0.5, 1, 2, 5, 10]:
        for g_s in [0.1, 0.5, 1, 2, 5, 10]:
            with open('./params.txt', 'w') as f : 
                f.write(str(g_w) + '\n')
                f.write(str(g_s) + '\n')
            print('g_w =', g_w, 'g_s =', g_s)
            classifier = Graph_RJW_SVC_Classifier()
            t1 = time.time()
            classifier.fit(X_train_bis, y_train)
            prediction_was = classifier.predict(X_test_bis)
            t2 = time.time()
            print('prediction : ',prediction_was)
            print('true labels : ',y_test)
            print('f1 score : ',f1_score(y_test, prediction_was))
            print('accuracy', np.mean(prediction_was == y_test))
            with open('./results.txt', 'a') as f : 
                f.write('g_w = ' + str(g_w) + ' g_s = ' + str(g_s) + ' f1 score : ' + str(f1_score(y_test, prediction_was)) + ' accuracy : ' + str(np.mean(prediction_was == y_test)) + ' time : ' + str(t2-t1) + '\n')
            