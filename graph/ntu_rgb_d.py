"""
ntu_rgb_d.py - Graph Definition for NTU RGB+D Skeleton Data

This module defines the human skeleton graph structure for the NTU RGB+D dataset.
It provides adjacency matrices for spatial-temporal graph convolutional networks
used in skeleton-based action recognition.

Dataset Reference: 
    NTU RGB+D 60/120: A Large-Scale Dataset for 3D Human Activity Analysis
    https://arxiv.org/abs/1604.02808

Key Features:
    - Defines skeleton with 25 joints following NTU RGB+D standard
    - Supports multiple graph labeling modes
    - Compatible with InfoGCN and standard spatial configurations

Graph Structure:
    Joint indices (0-24) correspond to NTU RGB+D skeleton joints:
        0:  pelvis (base spine)
        1:  spine (middle spine)
        2:  neck
        3:  head
        4-20: limbs and extremities (see inward_ori_index for mapping)

Usage:
    >>> from ntu_rgb_d import Graph
    >>> graph = Graph(labeling_mode='spatial')
    >>> A = graph.A  # Get adjacency matrix
    >>> print(graph.num_node)  # 25 joints
    
Adjacency Matrices:
    - 'spatial': Standard spatial configuration with bidirectional edges
    - Additional modes can be implemented in get_adjacency_matrix()

Attributes:
    num_node (int): Number of skeleton joints (25)
    self_link (list): Self-connections [(0,0), (1,1), ..., (24,24)]
    inward (list): Edges toward body center
    outward (list): Edges away from body center  
    neighbor (list): All neighbor connections (inward + outward)
    A (np.ndarray): Primary adjacency matrix
    A_outward_binary (np.ndarray): Binary outward adjacency for InfoGCN

Author: [Your Name/Project]
Created: [Date]
Version: 1.0
"""


import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools
from graph.infogcn import tools as tools_infogcn

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward
#infogcn
inward_ori_index_infogcn = [
    (2, 1), (2, 21), (21, 3), (3, 4), #head
    (21, 5), (5, 6), (6, 7), (7, 8), (8, 23), (23, 22), # left arm
    (21, 9), (9, 10), (10, 11), (11, 12), (12, 25), (25, 24), # right arm
    (1, 13), (13, 14), (14, 15),(15, 16), # left leg
    (1, 17), (17, 18),  (18, 19),  (19, 20) # right leg
]

inward_infogcn = [(i - 1, j - 1) for (i, j) in inward_ori_index_infogcn]
outward_infogcn = [(j, i) for (i, j) in inward_infogcn]
neighbor_infogcn = inward_infogcn + outward_infogcn

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.outward_infogcn = outward_infogcn
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A_outward_binary = tools_infogcn.get_adjacency_matrix(self.outward_infogcn, self.num_node)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
