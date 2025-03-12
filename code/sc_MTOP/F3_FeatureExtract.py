import json
import os
from WSIGraph import constructGraphFromDict
from collections import defaultdict
import numpy as np
from utils_xml import get_windows
from dataset import get_dataset
import pdb

def fun3(json_path, wsi_path, output_path, xml_path=None, n_process = 1, \
    cal_morph = True, cal_texture = True, cal_graph = True, 
    dataset = "pannuke", \
    restrict_type = [], restrict_edge = []):
    # json_path = '../Part1_HoverNet/COWH.json'
    # wsi_path = '../Part1_HoverNet/COWH.ndpi'
    # xml_path = '../Part1_HoverNet/COWH.xml'
    # output_path = './'
    distanceThreshold = 100
    level = 0
    k = 5

    sample_name = os.path.basename(wsi_path).split('.')[0]
    with open(json_path) as fp:
        print(f"{'Loading json':*^30s}")
        nucleusInfo = json.load(fp)


    globalgraph, edge_info = constructGraphFromDict(wsi_path, nucleusInfo, distanceThreshold,
                                                    k, level, n_process = n_process,
                                                    dataset = dataset,
                                                    cal_morph = cal_morph,
                                                    cal_texture = cal_texture,
                                                    cal_graph = cal_graph,
                                                    restrict_type = restrict_type,
                                                    restrict_edge = restrict_edge)
    vertex_dataframe = globalgraph.get_vertex_dataframe()
    centroid = np.array(vertex_dataframe['Centroid'].tolist())

    if xml_path is not None:
        window_bbox = np.array(get_windows(xml_path))
        index = np.zeros((len(centroid), len(window_bbox)), dtype=np.bool_)
        for i in range(len(window_bbox)):
            index[:, i] = ((window_bbox[i, 0, 0]<centroid[:, 0]) & (centroid[:,0]<window_bbox[i, 1, 0])) &\
                ((window_bbox[i, 0, 1]<centroid[:,1]) & (centroid[:,1]<window_bbox[i, 1, 1]))
        index_x, index_y = np.where(index)
        vertex_dataframe = vertex_dataframe.iloc[index_x]

    col_dist = defaultdict(list)
    graph_dataset = get_dataset(name = dataset)
    cellType = graph_dataset.cell_type
    for featname in vertex_dataframe.columns.values:
        if 'Graph' not in featname:
            # public feature, including cell information, Morph feature and GLCM feature
            for cell in cellType:
                col_dist[cell] += [featname] if featname != 'Contour' else []
        else:
            # Graph feature, format like 'Graph_T-I_Nsubgraph'
            for cell in cellType:
                featype = featname.split('_')[1]  # Graph feature type like 'T-T', 'T-I'
                col_dist[cell] += [featname] if cell in featype else []

    cellType_save = graph_dataset.cellType_save

    output_path = os.path.join(output_path, sample_name)
    try:
        os.makedirs(output_path)
    except:
        pass
    for i in col_dist.keys():
        vertex_csvfile = os.path.join(output_path, sample_name + '_Feats_' + i + '.csv')
        save_index = vertex_dataframe['CellType'].isin(cellType_save[i]).values
        vertex_dataframe.iloc[save_index].to_csv(vertex_csvfile, index=False, columns=col_dist[i])
    edge_csvfile = os.path.join(output_path, sample_name + '_Edges.csv')
    edge_info.to_csv(edge_csvfile, index=False)