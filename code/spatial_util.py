# Customized functions for spatial statistical analyses

import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from functools import reduce
import matplotlib.pyplot as plt
import networkx as nx
import pdb


def get_distance_mean_df(adata, matrix = "spatial_distance", 
                    phenotype = "phenotype", 
                    subset_phenotype = None,
                    subset_col = None, subset_value = None,
                    log = True):
    
    distance_map = adata.uns[matrix].copy()
    distance_map.index = distance_map.index.astype(str)

    if subset_col is not None and subset_value is not None:
        cells_to_subset = list(adata.obs[adata.obs[subset_col] == subset_value]['cell_id']) # 133531 cells
        overlap_cells = list(set(distance_map.index) & set(cells_to_subset))
        assert (len(overlap_cells) == len(cells_to_subset)) == True
        distance_map = distance_map.loc[distance_map.index.intersection(cells_to_subset)]

    if log is True:
        distance_map = np.log1p(distance_map)

    data = pd.DataFrame({'phenotype': adata.obs[phenotype], 'cell_id': adata.obs['cell_id']})
    data = data.set_index('cell_id')
    data = data.merge(distance_map, left_index=True, right_index=True)
    k = data.groupby(['phenotype'], observed=False).mean()  # collapse the whole dataset into mean distance
    d = k[k.index]

    if subset_phenotype is not None:
        d = d.loc[subset_phenotype, subset_phenotype]
    d = d.transpose()
    return d

def plot_spatial_network(df_z, 
                        df_p, 
                        subset_phenotype = None, 
                        replace_phenotype = None, 
                        min_z = None, max_z = None, 
                        min_p = None, max_p = None,
                        pos = None,
                        nodeSize = None, nodeColor = '#22333b', 
                        alpha = 0.9,
                        figsize = (10,8), 
                        fontSize = 18, 
                        fontColor = 'white'):
    
    df = df_z.stack().reset_index()
    df.columns = ['neighbour_phenotype', 'phenotype', 'z score']
    df['p-value'] = df_p.stack().reset_index()[0]

    if replace_phenotype is not None:
        df['phenotype'] = df['phenotype'].replace(replace_phenotype)
        df['neighbour_phenotype'] = df['neighbour_phenotype'].replace(replace_phenotype)

    if subset_phenotype is not None:
        df = df[df['phenotype'].isin(subset_phenotype) & df['neighbour_phenotype'].isin(subset_phenotype)]
        df['phenotype'] = df['phenotype'].astype('str').astype('category')
        df['neighbour_phenotype'] = df['neighbour_phenotype'].astype('str').astype('category')

    df = df[~(df['phenotype'] == df['neighbour_phenotype'])] # remove self-looping edges
    G = nx.from_pandas_edgelist(df, 'phenotype', 'neighbour_phenotype', ['z score', 'p-value'], create_using=nx.DiGraph())

    # Normalize z-scores for edge color
    z_scores = nx.get_edge_attributes(G, 'z score')
    if min_z is None:
        min_z = min(z_scores.values())
    if max_z is None:
        max_z = max(z_scores.values())

    # Apply normalization for coloring
    colors = [plt.cm.coolwarm((z_scores[edge] - min_z) / (max_z - min_z)) for edge in G.edges()]

    p_values = nx.get_edge_attributes(G, 'p-value')

    if min_p is None:
        min_p = min(p_values.values())
    if max_p is None:
        max_p = max(p_values.values())
    print(f'Min p-value: {min_p}, Max p-value: {max_p}')

    thicknesses = [10 * (1 - (p_values[edge] - min_p) / (max_p - min_p)) + 1 for edge in G.edges()]

    if pos is None:
        pos = nx.spring_layout(G, weight='weight')

    if nodeSize is None:
        node_count = G.number_of_nodes()
        nodeSize = 2000 / (node_count / 14)  # Example scaling, adjust the divisor as needed

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=nodeSize, node_color=nodeColor, alpha=alpha)
    nx.draw_networkx_edges(G, pos, ax=ax, width=thicknesses, edge_color=colors, arrowstyle='->')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=fontSize, font_color=fontColor)

    # Setup the ScalarMappable for the colorbar reflecting z-scores
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=min_z, vmax=max_z))
    #sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=plt.Normalize(vmin=min_z, vmax=max_z))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.03, aspect=12)
    cbar.set_label('Normalized interaction score', fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    ax.axis('off')
    return fig, ax


def get_interaction_df(
    adata,
    spatial_interaction='spatial_interaction',
    subset_phenotype=None,
    subset_neighbour_phenotype=None,
):
    # Copy the interaction results from anndata object
    try:
        interaction_map = adata.uns[spatial_interaction].copy()
    except KeyError:
        raise ValueError(
            'spatial_interaction not found- Please run sm.tl.spatial_interaction first'
        )

    # subset the data if user requests
    if subset_phenotype is not None:
        if isinstance(subset_phenotype, str):
            subset_phenotype = [subset_phenotype]
        # subset the phenotype
        interaction_map = interaction_map[
            interaction_map['phenotype'].isin(subset_phenotype)
        ]

    if subset_neighbour_phenotype is not None:
        if isinstance(subset_neighbour_phenotype, str):
            subset_neighbour_phenotype = [subset_neighbour_phenotype]
        # subset the phenotype
        interaction_map = interaction_map[
            interaction_map['neighbour_phenotype'].isin(subset_neighbour_phenotype)
        ]

    # Seperate Interaction intensity from P-value
    p_value = interaction_map.filter(regex='pvalue_')
    p_val_df = pd.concat(
        [interaction_map[['phenotype', 'neighbour_phenotype']], p_value],
        axis=1,
        join='outer',
    )
    p_val_df = p_val_df.set_index(['phenotype', 'neighbour_phenotype'])
    interaction_map = interaction_map[
        interaction_map.columns.difference(p_value.columns)
    ]
    interaction_map = interaction_map.set_index(['phenotype', 'neighbour_phenotype'])
    # If multiple images are present, take the average of interactions
    interaction_map['mean'] = interaction_map.mean(axis=1).values
    interaction_map = interaction_map[['mean']]  # keep only the mean column
    interaction_map = interaction_map['mean'].unstack()
    # Do the same for P-values
    p_val_df['mean'] = p_val_df.mean(axis=1).values
    p_val_df = p_val_df[['mean']]  # keep only the mean column
    # set the P-value threshold
    #p_val_df.loc[p_val_df[p_val_df['mean'] > p_val].index, 'mean'] = np.NaN
    p_val_df = p_val_df['mean'].unstack()

    # change to the order passed in subset
    if subset_phenotype is not None:
        interaction_map = interaction_map.reindex(subset_phenotype)
        p_val_df = p_val_df.reindex(subset_phenotype)
    if subset_neighbour_phenotype is not None:
        interaction_map = interaction_map.reindex(
            columns=subset_neighbour_phenotype
        )
        p_val_df = p_val_df.reindex(columns=subset_neighbour_phenotype)
    
    interaction_map = interaction_map.transpose()
    p_val_df = p_val_df.transpose()
    return interaction_map, p_val_df


def spatial_interaction_count(adata,
                         x_coordinate='X_centroid',
                         y_coordinate='Y_centroid',
                         z_coordinate=None,
                         phenotype='phenotype',
                         method='radius', 
                         radius=30, 
                         knn=10,
                         permutation=1000,
                         imageid='imageid',
                         subset=None,
                         pval_method='zscore',
                         verbose=True,
                         label='spatial_interaction_count'):
    """
Parameters:
        adata (anndata.AnnData):  
            Annotated data matrix or path to an AnnData object, containing spatial gene expression data.

        x_coordinate (str, required):  
            Column name in `adata` for the x-coordinates.

        y_coordinate (str, required):  
            Column name in `adata` for the y-coordinates.

        z_coordinate (str, optional):  
            Column name in `adata` for the z-coordinates, for 3D spatial data analysis.

        phenotype (str, required):  
            Column name in `adata` indicating cell phenotype or any categorical cell classification.

        method (str, optional):  
            Method to define neighborhoods: 'radius' for fixed distance, 'knn' for K nearest neighbors.

        radius (int, optional):  
            Radius for neighborhood definition (applies when method='radius').

        knn (int, optional):  
            Number of nearest neighbors to consider (applies when method='knn').

        permutation (int, optional):  
            Number of permutations for p-value calculation.

        imageid (str, required):  
            Column name in `adata` for image identifiers, useful for analysis within specific images.

        subset (str, optional):  
            Specific image identifier for targeted analysis.

        pval_method (str, optional):  
            Method for p-value calculation: 'abs' for absolute difference, 'zscore' for z-score based significance.

        verbose (bool):  
            If set to `True`, the function will print detailed messages about its progress and the steps being executed.

        label (str, optional):  
            Custom label for storing results in `adata.obs`.

Returns:
        adata (anndata.AnnData):  
            Updated `adata` object with spatial interaction results in `adata.obs[label]`.

Example:
        ```python

        # Radius method for 2D data with absolute p-value calculation
        adata = sm.tl.spatial_interaction(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                                    method='radius', radius=50, permutation=1000, pval_method='abs',
                                    label='interaction_radius_abs')

        # KNN method for 2D data with z-score based p-value calculation
        adata = sm.tl.spatial_interaction(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                                    method='knn', knn=15, permutation=1000, pval_method='zscore',
                                    label='interaction_knn_zscore')

        # Radius method for 3D data analysis
        adata = sm.tl.spatial_interaction(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                                    z_coordinate='Z_centroid', method='radius', radius=60, permutation=1000,
                                    pval_method='zscore', label='interaction_3D_zscore')

        ```
    """


    def spatial_interaction_internal (adata_subset,x_coordinate,y_coordinate,
                                      z_coordinate,
                                      phenotype,
                                      method, radius, knn,
                                      permutation, 
                                      imageid,subset,
                                      pval_method):
        if verbose:
            print("Processing Image: " + str(adata_subset.obs[imageid].unique()))

        # Create a dataFrame with the necessary information
        if z_coordinate is not None:
            if verbose:
                print("Including Z -axis")
            data = pd.DataFrame({'x': adata_subset.obs[x_coordinate], 'y': adata_subset.obs[y_coordinate], 'z': adata_subset.obs[z_coordinate], 'phenotype': adata_subset.obs[phenotype]})
        else:
            data = pd.DataFrame({'x': adata_subset.obs[x_coordinate], 'y': adata_subset.obs[y_coordinate], 'phenotype': adata_subset.obs[phenotype]})


        # Identify neighbourhoods based on the method used
        # a) KNN method
        if method == 'knn':
            if verbose:
                print("Identifying the " + str(knn) + " nearest neighbours for every cell")
            if z_coordinate is not None:
                tree = BallTree(data[['x','y','z']], leaf_size= 2)
                ind = tree.query(data[['x','y','z']], k=knn, return_distance= False)
            else:
                tree = BallTree(data[['x','y']], leaf_size= 2)
                ind = tree.query(data[['x','y']], k=knn, return_distance= False)
            neighbours = pd.DataFrame(ind.tolist(), index = data.index) # neighbour DF
            neighbours.drop(0, axis=1, inplace=True) # Remove self neighbour

        # b) Local radius method
        if method == 'radius':
            if verbose:
                print("Identifying neighbours within " + str(radius) + " pixels of every cell")
            if z_coordinate is not None:
                kdt = BallTree(data[['x','y','z']], metric='euclidean') 
                ind = kdt.query_radius(data[['x','y','z']], r=radius, return_distance=False)
            else:
                kdt = BallTree(data[['x','y']], metric='euclidean') 
                ind = kdt.query_radius(data[['x','y']], r=radius, return_distance=False)

            for i in range(0, len(ind)): ind[i] = np.delete(ind[i], np.argwhere(ind[i] == i))#remove self
            neighbours = pd.DataFrame(ind.tolist(), index = data.index) # neighbour DF

        # Map Phenotypes to Neighbours
        # Loop through (all functionized methods were very slow)
        phenomap = dict(zip(list(range(len(ind))), data['phenotype'])) # Used for mapping
        if verbose:
            print("Mapping phenotype to neighbors")
        for i in neighbours.columns:
            neighbours[i] = neighbours[i].dropna().map(phenomap, na_action='ignore')

        # Drop NA
        neighbours = neighbours.dropna(how='all')

        # Collapse all the neighbours into a single column
        n = pd.DataFrame(neighbours.stack(), columns = ["neighbour_phenotype"])
        n.index = n.index.get_level_values(0) # Drop the multi index

        # Merge with real phenotype
        n = n.merge(data['phenotype'], how='inner', left_index=True, right_index=True)

        n_freq = n.groupby(['phenotype','neighbour_phenotype'],observed=False).size().unstack().fillna(0).stack() 
        n_freq = pd.DataFrame(n_freq)
        n_freq.columns = [str(adata_subset.obs[imageid].unique()[0]) + '_count']
        n_freq = n_freq.reset_index()
        
        # return
        return n_freq


    # subset a particular subset of cells if the user wants else break the adata into list of anndata objects
    if subset is not None:
        adata_list = [adata[adata.obs[imageid] == subset]]
    else:
        adata_list = [adata[adata.obs[imageid] == i] for i in adata.obs[imageid].unique()]


    # Apply function to all images and create a master dataframe
    # Create lamda function 
    r_spatial_interaction_internal = lambda x: spatial_interaction_internal (adata_subset=x, x_coordinate=x_coordinate, y_coordinate=y_coordinate, 
                                                                             z_coordinate=z_coordinate, phenotype=phenotype, method=method,  radius=radius, knn=knn, permutation=permutation, imageid=imageid,subset=subset,pval_method=pval_method) 
    all_data = list(map(r_spatial_interaction_internal, adata_list)) # Apply function 


    # Merge all the results into a single dataframe    
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['phenotype', 'neighbour_phenotype'], how='outer'), all_data)


    # Add to anndata
    adata.uns[label] = df_merged

    # return
    return adata

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Wed Aug 19 15:00:39 2020
# @author: Ajit Johnson Nirmal
"""
!!! abstract "Short Description"
    `sm.tl.spatial_count` computes a neighborhood matrix from spatial data using categorical variables, 
    such as cell types, to identify local cell clusters. It offers two neighborhood definition methods:

    - **Radius Method**: Identifies neighbors within a specified radius for each cell, allowing for 
    the exploration of spatial relationships based on physical proximity.
    - **KNN Method**: Determines neighbors based on the K nearest neighbors, focusing on the closest 
    spatial associations irrespective of physical distance.
    
    The generated neighborhood matrix is stored in `adata.uns`, providing a basis for further analysis. 
    To uncover Recurrent Cellular Neighborhoods (RCNs) that share similar spatial patterns, users can 
    cluster the neighborhood matrix using the `spatial_cluster` function. This approach enables the 
    identification of spatially coherent cell groups, facilitating insights into the cellular 
    architecture of tissues.

## Function
"""

# Import library
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

# Function
def spatial_count (adata,
                   x_coordinate='X_centroid',
                   y_coordinate='Y_centroid',
                   z_coordinate=None,
                   phenotype='phenotype',
                   method='radius',
                   radius=30,knn=10,
                   imageid='imageid',
                   subset=None,
                   verbose=True,
                   label='spatial_count'):
    """
Parameters:
        adata (anndata.AnnData):  
            Annotated data matrix with spatial information.
        
        x_coordinate (str, required):  
            Column name containing x-coordinates.
        
        y_coordinate (str, required):  
            Column name containing y-coordinates.
        
        z_coordinate (str, optional):  
            Column name containing z-coordinates, for 3D spatial data.
        
        phenotype (str, required):  
            Column name containing phenotype or any categorical cell classification.
        
        method (str, optional):  
            Neighborhood definition method: 'radius' for fixed distance, 'knn' for K nearest neighbors.
        
        radius (int, optional):  
            Radius used to define neighborhoods (applicable when method='radius').
        
        knn (int, optional):  
            Number of nearest neighbors to consider (applicable when method='knn').
        
        imageid (str, optional):  
            Column name containing image identifiers, for analyses limited to specific images.
        
        subset (str, optional):  
            Specific image identifier for subsetting data before analysis.
        
        verbose (bool, optional):  
            If True, prints progress and informational messages.
        
        label (str, optional):  
            Key for storing results in `adata.uns`.

Returns:
        adata (anndata.AnnData):  
            Updated AnnData object with the neighborhood matrix stored in `adata.uns[label]`.

Example:
    ```python
    
    # Analyze spatial relationships using the radius method
    adata = sm.tl.spatial_count(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                          phenotype='phenotype', method='radius', radius=50,
                          label='neighborhood_radius50')

    # Explore spatial neighborhoods with KNN
    adata = sm.tl.spatial_count(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                          phenotype='phenotype', method='knn', knn=15,
                          label='neighborhood_knn15')

    # 3D spatial analysis using a radius method
    adata = sm.tl.spatial_count(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                          z_coordinate='Z_centroid', phenotype='phenotype', method='radius', radius=30,
                          label='neighborhood_3D_radius30')
    
    ```
    """

    def spatial_count_internal (adata_subset,x_coordinate,y_coordinate,z_coordinate,phenotype,method,radius,knn,
                                imageid,subset,label):
        
        # Create a dataFrame with the necessary inforamtion
        if z_coordinate is not None:
            if verbose:
                print("Including Z -axis")
            data = pd.DataFrame({'x': adata_subset.obs[x_coordinate], 'y': adata_subset.obs[y_coordinate], 'z': adata_subset.obs[z_coordinate], 'phenotype': adata_subset.obs[phenotype]})
        else:
            data = pd.DataFrame({'x': adata_subset.obs[x_coordinate], 'y': adata_subset.obs[y_coordinate], 'phenotype': adata_subset.obs[phenotype]})


        # Create a DataFrame with the necessary inforamtion
        #data = pd.DataFrame({'x': adata_subset.obs[x_coordinate], 'y': adata_subset.obs[y_coordinate], 'phenotype': adata_subset.obs[phenotype]})
        
        # Identify neighbourhoods based on the method used
        # a) KNN method
        if method == 'knn':
            if verbose:
                print("Identifying the " + str(knn) + " nearest neighbours for every cell")
            if z_coordinate is not None:
                tree = BallTree(data[['x','y','z']], leaf_size= 2)
                ind = tree.query(data[['x','y','z']], k=knn, return_distance= False)
            else:
                tree = BallTree(data[['x','y']], leaf_size= 2)
                ind = tree.query(data[['x','y']], k=knn, return_distance= False)
            neighbours = pd.DataFrame(ind.tolist(), index = data.index) # neighbour DF
            neighbours.drop(0, axis=1, inplace=True) # Remove self neighbour
        
        # b) Local radius method
        if method == 'radius':
            if verbose:
                print("Identifying neighbours within " + str(radius) + " pixels of every cell")
            if z_coordinate is not None:
                kdt = BallTree(data[['x','y','z']], metric='euclidean') 
                ind = kdt.query_radius(data[['x','y','z']], r=radius, return_distance=False)
            else:
                kdt = BallTree(data[['x','y']], metric='euclidean') 
                ind = kdt.query_radius(data[['x','y']], r=radius, return_distance=False)
                
            for i in range(0, len(ind)): ind[i] = np.delete(ind[i], np.argwhere(ind[i] == i))#remove self
            neighbours = pd.DataFrame(ind.tolist(), index = data.index) # neighbour DF
            
        # Map phenotype
        phenomap = dict(zip(list(range(len(ind))), data['phenotype'])) # Used for mapping
        
        # Loop through (all functionized methods were very slow)
        for i in neighbours.columns:
            neighbours[i] = neighbours[i].dropna().map(phenomap, na_action='ignore')
        
        # Drop NA
        #n_dropped = neighbours.dropna(how='all')
           
        # Collapse all the neighbours into a single column
        n = pd.DataFrame(neighbours.stack(), columns = ["neighbour_phenotype"])
        n.index = n.index.get_level_values(0) # Drop the multi index
        n = pd.DataFrame(n)
        n['order'] = list(range(len(n)))

        # Merge with real phenotype
        n_m = n.merge(data['phenotype'], how='inner', left_index=True, right_index=True)
        n_m['neighbourhood'] = n_m.index
        n = n_m.sort_values(by=['order'])

        pdb.set_trace()
        
        # Normalize based on total cell count
        k = n.groupby(['neighbourhood','neighbour_phenotype']).size().unstack().fillna(0)
        k = k.div(k.sum(axis=1), axis=0)
        
        # return the normalized neighbour occurance count
        return k
    
    # Subset a particular image if needed
    if subset is not None:
        adata_list = [adata[adata.obs[imageid] == subset]]
    else:
        adata_list = [adata[adata.obs[imageid] == i] for i in adata.obs[imageid].unique()]
    
    # Apply function to all images and create a master dataframe
    # Create lamda function 
    r_spatial_count_internal = lambda x: spatial_count_internal(adata_subset=x,x_coordinate=x_coordinate,
                                                   y_coordinate=y_coordinate,
                                                   z_coordinate=z_coordinate,
                                                   phenotype=phenotype,
                                                   method=method,radius=radius,knn=knn,
                                                   imageid=imageid,subset=subset,label=label) 
    all_data = list(map(r_spatial_count_internal, adata_list)) # Apply function 
    
    
    # Merge all the results into a single dataframe    
    result = []
    for i in range(len(all_data)):
        result.append(all_data[i])
    result = pd.concat(result, join='outer')  
    
    # Reindex the cells
    result = result.reindex(adata.obs.index)
    result = result.fillna(0)
    
    # Add to adata
    adata.uns[label] = result
    
    # Return        
    return adata


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Mon Oct 19 20:03:01 2020
# @author: Ajit Johnson Nirmal

"""
!!! abstract "Short Description"
    `sm.tl.spatial_interaction`: This function quantifies the spatial interactions 
    between cell types, assessing their co-localization beyond random chance, with 
    support for both 2D and 3D datasets. By comparing observed adjacency frequencies 
    to a random distribution, it helps uncover significant cellular partnerships 
    within tissue contexts.
    
## Function
"""

# Import library
import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np
from joblib import Parallel, delayed
import scipy
from functools import reduce


# Function
def spatial_interaction (adata,
                         x_coordinate='X_centroid',
                         y_coordinate='Y_centroid',
                         z_coordinate=None,
                         phenotype='phenotype',
                         method='radius', 
                         radius=30, 
                         knn=10,
                         permutation=1000,
                         imageid='imageid',
                         subset=None,
                         pval_method='zscore',
                         verbose=True,
                         label='spatial_interaction'):
    """
Parameters:
        adata (anndata.AnnData):  
            Annotated data matrix or path to an AnnData object, containing spatial gene expression data.

        x_coordinate (str, required):  
            Column name in `adata` for the x-coordinates.

        y_coordinate (str, required):  
            Column name in `adata` for the y-coordinates.

        z_coordinate (str, optional):  
            Column name in `adata` for the z-coordinates, for 3D spatial data analysis.

        phenotype (str, required):  
            Column name in `adata` indicating cell phenotype or any categorical cell classification.

        method (str, optional):  
            Method to define neighborhoods: 'radius' for fixed distance, 'knn' for K nearest neighbors.

        radius (int, optional):  
            Radius for neighborhood definition (applies when method='radius').

        knn (int, optional):  
            Number of nearest neighbors to consider (applies when method='knn').

        permutation (int, optional):  
            Number of permutations for p-value calculation.

        imageid (str, required):  
            Column name in `adata` for image identifiers, useful for analysis within specific images.

        subset (str, optional):  
            Specific image identifier for targeted analysis.

        pval_method (str, optional):  
            Method for p-value calculation: 'abs' for absolute difference, 'zscore' for z-score based significance.
        
        verbose (bool):  
            If set to `True`, the function will print detailed messages about its progress and the steps being executed.

        label (str, optional):  
            Custom label for storing results in `adata.obs`.

Returns:
        adata (anndata.AnnData):  
            Updated `adata` object with spatial interaction results in `adata.obs[label]`.

Example:
        ```python
        
        # Radius method for 2D data with absolute p-value calculation
        adata = sm.tl.spatial_interaction(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                                    method='radius', radius=50, permutation=1000, pval_method='abs',
                                    label='interaction_radius_abs')
    
        # KNN method for 2D data with z-score based p-value calculation
        adata = sm.tl.spatial_interaction(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                                    method='knn', knn=15, permutation=1000, pval_method='zscore',
                                    label='interaction_knn_zscore')
    
        # Radius method for 3D data analysis
        adata = sm.tl.spatial_interaction(adata, x_coordinate='X_centroid', y_coordinate='Y_centroid',
                                    z_coordinate='Z_centroid', method='radius', radius=60, permutation=1000,
                                    pval_method='zscore', label='interaction_3D_zscore')
        
        ```
    """
    
    
    def spatial_interaction_internal (adata_subset,x_coordinate,y_coordinate,
                                      z_coordinate,
                                      phenotype,
                                      method, radius, knn,
                                      permutation, 
                                      imageid,subset,
                                      pval_method):
        if verbose:
            print("Processing Image: " + str(adata_subset.obs[imageid].unique()))
        
        # Create a dataFrame with the necessary inforamtion
        if z_coordinate is not None:
            if verbose:
                print("Including Z -axis")
            data = pd.DataFrame({'x': adata_subset.obs[x_coordinate], 'y': adata_subset.obs[y_coordinate], 'z': adata_subset.obs[z_coordinate], 'phenotype': adata_subset.obs[phenotype]})
        else:
            data = pd.DataFrame({'x': adata_subset.obs[x_coordinate], 'y': adata_subset.obs[y_coordinate], 'phenotype': adata_subset.obs[phenotype]})

        
        # Identify neighbourhoods based on the method used
        # a) KNN method
        if method == 'knn':
            if verbose:
                print("Identifying the " + str(knn) + " nearest neighbours for every cell")
            if z_coordinate is not None:
                tree = BallTree(data[['x','y','z']], leaf_size= 2)
                ind = tree.query(data[['x','y','z']], k=knn, return_distance= False)
            else:
                tree = BallTree(data[['x','y']], leaf_size= 2)
                ind = tree.query(data[['x','y']], k=knn, return_distance= False)
            neighbours = pd.DataFrame(ind.tolist(), index = data.index) # neighbour DF
            neighbours.drop(0, axis=1, inplace=True) # Remove self neighbour
            
        # b) Local radius method
        if method == 'radius':
            if verbose:
                print("Identifying neighbours within " + str(radius) + " pixels of every cell")
            if z_coordinate is not None:
                kdt = BallTree(data[['x','y','z']], metric='euclidean') 
                ind = kdt.query_radius(data[['x','y','z']], r=radius, return_distance=False)
            else:
                kdt = BallTree(data[['x','y']], metric='euclidean') 
                ind = kdt.query_radius(data[['x','y']], r=radius, return_distance=False)
                
            for i in range(0, len(ind)): ind[i] = np.delete(ind[i], np.argwhere(ind[i] == i))#remove self
            neighbours = pd.DataFrame(ind.tolist(), index = data.index) # neighbour DF
            
        # Map Phenotypes to Neighbours
        # Loop through (all functionized methods were very slow)
        phenomap = dict(zip(list(range(len(ind))), data['phenotype'])) # Used for mapping
        if verbose:
            print("Mapping phenotype to neighbors")
        for i in neighbours.columns:
            neighbours[i] = neighbours[i].dropna().map(phenomap, na_action='ignore')
            
        # Drop NA
        neighbours = neighbours.dropna(how='all')
        
        # Collapse all the neighbours into a single column
        n = pd.DataFrame(neighbours.stack(), columns = ["neighbour_phenotype"])
        n.index = n.index.get_level_values(0) # Drop the multi index
        
        # Merge with real phenotype
        n = n.merge(data['phenotype'], how='inner', left_index=True, right_index=True)
        
        # Permutation
        if verbose:
            print('Performing '+ str(permutation) + ' permutations')
    
        def permutation_pval (data):
            data = data.assign(neighbour_phenotype=np.random.permutation(data['neighbour_phenotype']))
            #data['neighbour_phenotype'] = np.random.permutation(data['neighbour_phenotype'])
            data_freq = data.groupby(['phenotype','neighbour_phenotype'],observed=False).size().unstack()
            data_freq = data_freq.fillna(0).stack().values 
            return data_freq

        # Apply function
        final_scores = Parallel(n_jobs=-1)(delayed(permutation_pval)(data=n) for i in range(permutation)) 
        perm = pd.DataFrame(final_scores).T
        
        # Consolidate the permutation results
        if verbose:
            print('Consolidating the permutation results')
        # Calculate P value
        # real
        n_freq = n.groupby(['phenotype','neighbour_phenotype'],observed=False).size().unstack().fillna(0).stack() 
        # permutation
        mean = perm.mean(axis=1)
        std = perm.std(axis=1)
        # P-value calculation
        if pval_method == 'abs':
            # real value - prem value / no of perm 
            p_values = abs(n_freq.values - mean) / (permutation+1)
            p_values = p_values[~np.isnan(p_values)].values
        if pval_method == 'zscore':
            z_scores = (n_freq.values - mean) / std        
            z_scores[np.isnan(z_scores)] = 0
            p_values = scipy.stats.norm.sf(abs(z_scores))*2
            p_values = p_values[~np.isnan(p_values)]
            
        # Compute Direction of interaction (interaction or avoidance)
        direction = ((n_freq.values - mean) / abs(n_freq.values - mean)).fillna(1)

        pdb.set_trace()

        # Normalize based on total cell count
        k = n.groupby(['phenotype','neighbour_phenotype'],observed=False).size().unstack().fillna(0)
        # add neighbour phenotype that are not present to make k a square matrix
        columns_to_add = dict.fromkeys(np.setdiff1d(k.index,k.columns), 0)
        k = k.assign(**columns_to_add)

        total_cell_count = data['phenotype'].value_counts()
        total_cell_count = total_cell_count[k.columns].values # keep only cell types that are present in the column of k

        # total_cell_count = total_cell_count.reindex(k.columns).values # replaced by above
        k_max = k.div(total_cell_count, axis = 0)
        k_max = k_max.div(k_max.max(axis=1), axis=0).stack()

        # DataFrame with the neighbour frequency and P values
        count = (k_max.values * direction).values # adding directionality to interaction
        neighbours = pd.DataFrame({'count': count,'p_val': p_values}, index = k_max.index)
        #neighbours.loc[neighbours[neighbours['p_val'] > p_val].index,'count'] = np.NaN
        #del neighbours['p_val']
        neighbours.columns = [adata_subset.obs[imageid].unique()[0], 'pvalue_' + str(adata_subset.obs[imageid].unique()[0])]
        neighbours = neighbours.reset_index()
        #neighbours = neighbours['count'].unstack()
        
        # return
        return neighbours
          
      
    # subset a particular subset of cells if the user wants else break the adata into list of anndata objects
    if subset is not None:
        adata_list = [adata[adata.obs[imageid] == subset]]
    else:
        adata_list = [adata[adata.obs[imageid] == i] for i in adata.obs[imageid].unique()]
    
    
    # Apply function to all images and create a master dataframe
    # Create lamda function 
    r_spatial_interaction_internal = lambda x: spatial_interaction_internal (adata_subset=x, x_coordinate=x_coordinate, y_coordinate=y_coordinate, 
                                                                             z_coordinate=z_coordinate, phenotype=phenotype, method=method,  radius=radius, knn=knn, permutation=permutation, imageid=imageid,subset=subset,pval_method=pval_method) 
    all_data = list(map(r_spatial_interaction_internal, adata_list)) # Apply function 
    

    # Merge all the results into a single dataframe    
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['phenotype', 'neighbour_phenotype'], how='outer'), all_data)
    

    # Add to anndata
    adata.uns[label] = df_merged
    
    # return
    return adata