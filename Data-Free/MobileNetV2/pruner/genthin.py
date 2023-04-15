import torch
import torch.nn as nn
import functools
import numpy as np
import copy
import scipy.sparse as sparse
from torchsummary import summary
from numba import jit
from concurrent.futures import ProcessPoolExecutor as Pool
import os
import time
import copy
import math
from random import random
from tqdm import tqdm

# Utility functions for pruning

def rsetattr(obj, attr, val): # sets the value of an attribute in a class using a string
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args): # fetches the value of an attribute in a class using a string
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

@jit(nopython=True)
def _get_conv_matrix(input_height, input_width, weights, stride, pad, groups):
    """
    Modified from conv_forward of Source
    Source: https://datascience-enthusiast.com/DL/Convolution_model_Step_by_Stepv2.html
    Finds equivalent matrix A for a Conv2d layer of network such that
    A @ x.flatten() == Conv2d(x, bias = False).flatten() 
    
    Arguments:
    input_height = height of image input to the conv2d layer
    input_width = width of image input to the conv2d layer
    weights -- weights of the conv2d layer. Can be fetched as conv2d_layer_name.weights
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    conv_matrix -- equivalent matrix, numpy array of shape (n_C*n_H*n_W, n_C_prev*n_H_prev*n_W_prev)
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from given parameters (≈2 line)  
    
    (n_C, n_C_prev, f, f) = weights.shape # number of output channels, number of input channels, kernel size, kernel size
    if groups == n_C and n_C_prev == 1:
        n_C_prev = n_C
    n_H_prev, n_W_prev  = input_height, input_width
    
    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
    n_H = int((n_H_prev + 2*pad - f)/stride) + 1
    n_W =int((n_W_prev + 2*pad - f)/stride) + 1
    
    # Initialize the equivalent convolution matrix to zeros. (≈1 line)
    rows = [np.int64(x) for x in range(0)]
    columns = [np.int64(x) for x in range(0)]
    vals = [np.float32(x) for x in range(0)]
    for i in range(n_C):                           # loop over output channels
        if groups == 1:
            j_range = range(n_C_prev)
        elif groups == n_C:
            j_range = range(i, i+1)
        for j in j_range:                  # loop over input channels
            if groups == 1:
                conv_kernel = weights[i,j,:,:]
            elif groups == n_C:
                conv_kernel = weights[i,0,:,:]
            i_offset, j_offset = i*n_H*n_W, j*n_H_prev*n_W_prev
            per_conv_matrix(rows, columns, vals, i_offset, j_offset, \
                                    n_H, n_W, n_H_prev, n_W_prev, stride, pad, conv_kernel)
    return vals, rows, columns

@jit(nopython=True)
def per_conv_matrix(rows, columns, vals, i_offset, j_offset, \
                                    n_H, n_W, n_H_prev, n_W_prev, stride, pad, conv_kernel):
    # fetch kernel size
    kernel_size = conv_kernel.shape[0]

    # To find the equivalent convolution matrix, 
    for h in range(n_H):
        for w in range(n_W):
            vert_start = h*stride
            vert_end = h*stride + kernel_size
            horiz_start = w*stride 
            horiz_end = w*stride + kernel_size

            for i in range(vert_start, vert_end):
                for j in range(horiz_start, horiz_end):
                    true_i = i - pad
                    true_j = j - pad
                    if true_i >= 0 and true_j >= 0 and true_i < n_H_prev and true_j < n_W_prev:
                        h_index = n_W*h + w
                        w_index = n_W_prev*true_i + true_j
                        rows.append(h_index + i_offset)
                        columns.append(w_index + j_offset)
                        vals.append(conv_kernel[i-vert_start, j-horiz_start].item())

# Pruner Class

class GenThinPruner():
    def __init__(self, model, args):
        # Find geometric constraints / DFCs in model
        self.model = copy.deepcopy(model)
        self.geometric_constraints = self.MobileNetV2GeometricConstraints(self.model)
        self.num_processors = args.num_processors
        # fetch size of input and output tensors to each layer.
        self.model_info = summary(self.model, (3,32, 32))
        self.args = args
        del self.model

    def ComputeSaliencyScores(self, model):
        self.scores = []
        if 'l1' in self.args.strategy:
            for i in range(len(self.geometric_constraints)):
                constraint = self.geometric_constraints[i]
                num_channels = self.get_num_channels(constraint, model)
                gc_scores = [0.0]*num_channels
                for layer_name in constraint['A']:
                    layer = rgetattr(model, layer_name)
                    t = layer.weight.cpu().detach().numpy()
                    t = t.reshape(t.shape[0],t.shape[1]*t.shape[2]*t.shape[3])
                    channel_norms = np.linalg.norm(t, axis = 1,ord=1)
                    res = list(channel_norms.flatten())
                    for i in range(num_channels):
                        gc_scores[i] += res[i]
                self.scores.append(gc_scores)
        elif 'random' in self.args.strategy:
            for i in range(len(self.geometric_constraints)):
                constraint = self.geometric_constraints[i]
                num_channels = self.get_num_channels(constraint, model)
                gc_scores = []
                for _ in range(num_channels):
                    gc_scores.append(random())
                self.scores.append(gc_scores)
        else:
            self.scores = self.MobileNetV2SaliencyScores(model)

    def Prune(self, model):
        prune_coupled = 1 # no non-coupled convolutional layers in MobileNet-v2
        #find geometric constraint and channel with the lowest saliency score
        min_score_gc = 0 # index of geometric constraint with the lowest saliency score
        min_score_channel = 0 # index of channel with the lowest saliency score in the geometric constraint
        if not prune_coupled:
            min_score_gc = 1
        while(len(self.scores[min_score_gc]) == 1):
            min_score_gc += 1

        # simple brute-force search to find the indices with minimum saliency score (mean normalised)
        for i in range(len(self.scores)):
            if len(self.scores[i]) > 1 and ((prune_coupled) or (len(self.geometric_constraints[i]['A']) == 1 and len(self.geometric_constraints[i]['B']) == 1)):
                for j in range(len(self.scores[i])):
                    if self.scores[i][j] < self.scores[min_score_gc][min_score_channel]:
                        min_score_gc = i
                        min_score_channel = j
        
        constraint = self.geometric_constraints[min_score_gc] # geometric constraint with the least saliency.
        index_list = torch.tensor(list(set(range(len(self.scores[min_score_gc])))\
                                    -set([min_score_channel]))) # list of indices of channels to keep

        # prune the channel with the lowest saliency score
        for layer_name in constraint['A'] + constraint['dependent_layers']:
            layer = rgetattr(model, layer_name)
            layer_state_dict = layer.state_dict()
            if isinstance(layer, nn.Conv2d):
                new_groups = 1
                new_in_channels = layer.in_channels
                if layer.groups > 1:
                    new_groups = layer.groups - 1
                    new_in_channels = layer.in_channels - 1
                new_layer = nn.Conv2d(in_channels = new_in_channels, out_channels = layer.out_channels-1, \
                    kernel_size = layer.kernel_size, stride = layer.stride, padding = layer.padding, \
                    dilation = layer.dilation, groups = new_groups, bias = (layer.bias is not None), padding_mode = layer.padding_mode)
                for key in layer_state_dict:
                    layer_state_dict[key] = torch.index_select(layer_state_dict[key], dim = 0, index = index_list)
            elif isinstance(layer, nn.Linear):
                new_layer = nn.Linear(in_features = layer.in_features, out_features = layer.out_features-1, \
                    bias = (layer.bias is not None))
                for key in layer_state_dict:
                    layer_state_dict[key] = torch.index_select(layer_state_dict[key], dim = 0, index = index_list)
            else:
                raise ValueError('Layer {} is of unknown type'.format(layer_name))
            new_layer.load_state_dict(layer_state_dict)
            rsetattr(model, layer_name, new_layer)

            if 'conv' in layer_name:
                bn_layer_name = layer_name.replace('conv', 'bn')
            elif 'shortcut' in  layer_name:
                bn_layer_name = layer_name[:-1] + '1'
            else:
                continue
            bn_layer = rgetattr(model, bn_layer_name)
            bn_layer_state_dict = bn_layer.state_dict()
            if isinstance(bn_layer, nn.BatchNorm2d):
                bn_new_layer = nn.BatchNorm2d(num_features = bn_layer.num_features-1, eps = bn_layer.eps, \
                    momentum = bn_layer.momentum, affine = bn_layer.affine, track_running_stats = True)
                for key in bn_layer_state_dict:
                    if len(bn_layer_state_dict[key].shape) > 0:
                        bn_layer_state_dict[key] = torch.index_select(bn_layer_state_dict[key], dim = 0, index = index_list)
            else:
                raise ValueError('Layer {} is of unknown type'.format(layer_name))
            bn_new_layer.load_state_dict(bn_layer_state_dict)
            rsetattr(model, bn_layer_name, bn_new_layer)

        for layer_name in constraint['B']:
            layer = rgetattr(model, layer_name)
            layer_state_dict = layer.state_dict()
            if isinstance(layer, nn.Conv2d):
                new_layer = nn.Conv2d(in_channels = layer.in_channels-1, out_channels = layer.out_channels, \
                    kernel_size = layer.kernel_size, stride = layer.stride, padding = layer.padding, \
                    dilation = layer.dilation, groups = layer.groups, bias = (layer.bias is not None), padding_mode = layer.padding_mode)
                layer_state_dict['weight'] = torch.index_select(layer_state_dict['weight'], dim = 1, index = index_list)
            elif isinstance(layer, nn.Linear):
                new_layer = nn.Linear(in_features = layer.in_features-1, out_features = layer.out_features, \
                    bias = (layer.bias is not None))
                layer_state_dict['weight'] = torch.index_select(layer_state_dict['weight'], dim = 1, index = index_list)
            else:
                raise ValueError('Layer {} is of unknown type'.format(layer_name))
            new_layer.load_state_dict(layer_state_dict)
            rsetattr(model, layer_name, new_layer)
        
        self.scores[min_score_gc].pop(min_score_channel)
        return min_score_gc, min_score_channel
    
    def MobileNetV2GeometricConstraints(self, model):
        geometric_constraints = {}

        for block_index in range(17):
            block_name = 'layers.{}'.format(block_index)
            geometric_constraints[len(geometric_constraints)] = {'A': [block_name+'.conv1'], 'B': [block_name+'.conv3'],
                                                                        'dependent_layers': [block_name+'.conv2']}

        geometric_constraints[len(geometric_constraints)] = {'A': ['conv1'], 'B': ['layers.0.conv1', 'layers.0.shortcut.0'],
                                                                'dependent_layers': []}
        geometric_constraints[len(geometric_constraints)] = {'A': ['layers.0.conv3', 'layers.0.shortcut.0'],
                                                                'B': ['layers.1.conv1', 'layers.1.shortcut.0'],
                                                                'dependent_layers': []}
        geometric_constraints[len(geometric_constraints)] = {'A': ['layers.1.conv3', 'layers.1.shortcut.0', 'layers.2.conv3'],
                                                                'B': ['layers.2.conv1', 'layers.3.conv1'],
                                                                'dependent_layers': []}
        geometric_constraints[len(geometric_constraints)] = {'A': ['layers.3.conv3', 'layers.4.conv3', 'layers.5.conv3'],
                                                                'B': ['layers.4.conv1', 'layers.5.conv1', 'layers.6.conv1'],
                                                                'dependent_layers': []}
        geometric_constraints[len(geometric_constraints)] = {'A': ['layers.6.conv3', 'layers.7.conv3', 
                                                                                    'layers.8.conv3', 'layers.9.conv3'], 
                                                                'B': ['layers.7.conv1', 'layers.8.conv1', 
                                                                    'layers.9.conv1', 'layers.10.conv1', 'layers.10.shortcut.0'],
                                                                'dependent_layers': []}
        geometric_constraints[len(geometric_constraints)] = {'A': ['layers.10.shortcut.0', 'layers.10.conv3', 'layers.11.conv3',
                                                                    'layers.12.conv3'],
                                                                'B': ['layers.11.conv1', 'layers.12.conv1', 'layers.13.conv1'],
                                                                'dependent_layers': []}
        geometric_constraints[len(geometric_constraints)] = {'A': ['layers.13.conv3', 'layers.14.conv3', 'layers.15.conv3'],
                                                                'B': ['layers.14.conv1', 'layers.15.conv1', 'layers.16.conv1',
                                                                        'layers.16.shortcut.0'],
                                                                'dependent_layers': []}
        geometric_constraints[len(geometric_constraints)] = {'A': ['layers.16.conv3', 'layers.16.shortcut.0'], 
                                                                'B': ['conv2'], 'dependent_layers': []}
        
        geometric_constraints[len(geometric_constraints)] = {'A': ['conv2'], 'B': ['linear'], 'dependent_layers': []}
            
        return geometric_constraints
    

    def get_conv_matrix(self, input_height, input_width, weights, stride, pad, groups=1):
        # Get entries to fill into the sparse matrix
        vals, rows, columns = _get_conv_matrix(input_height, input_width, weights, stride, pad, groups)
        (n_C, n_C_prev, f, f) = weights.shape # number of output channels, number of input channels, kernel size, kernel size
        if groups == n_C and n_C_prev == 1:
            n_C_prev = n_C
        n_H_prev, n_W_prev  = input_height, input_width
        
        # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
        n_H = int((n_H_prev + 2*pad - f)/stride) + 1
        n_W =int((n_W_prev + 2*pad - f)/stride) + 1
        return sparse.coo_matrix((vals, (rows, columns)), shape=(n_H*n_W*n_C, n_H_prev*n_W_prev*n_C_prev)).tocsr()
    
    def _score_matrix_maxpool2d(self, module, input_shape):
        # only supports maxpool2d layers with dilation = 1
        num_channels = input_shape[1]
        weights = torch.ones([num_channels, num_channels, module.kernel_size, module.kernel_size])\
                    /(module.kernel_size*module.kernel_size)
        return self.get_conv_matrix(input_height=input_shape[2], input_width=input_shape[3], \
                            weights=weights.cpu().detach().numpy(), stride = module.stride, pad = module.padding)

    def _score_matrix_conv2d(self, module, input_shape):
        # only supports conv2d layers with dilation = 1
        return self.get_conv_matrix(input_height=input_shape[2], input_width=input_shape[3], \
                            weights=module.weight.cpu().detach().numpy(), stride = module.stride[0], pad = module.padding[0],
                            groups=module.groups)

    def _score_matrix_linear(self, module):
        return sparse.csr_matrix(module.weight.cpu().detach().numpy())

    def _score_matrix_bn2d(self, module, input_shape):
        num_els_per_channel = input_shape[2]*input_shape[3]
        num_channels = input_shape[1]
        v = []
        for i in range(num_channels):
            val = np.abs(module.weight[i].item()/np.sqrt(module.eps + module.running_var[i].item()))
            for _ in range(num_els_per_channel):
                v.append(val)
        return sparse.diags(diagonals = v, format='csr')

    def _score_matrix_avgpool2d(self, input_shape):
        # only deals with adaptiveAvgpool2d layers with output_size=(1, 1)
        num_channels = input_shape[1]
        input_elements_per_channel = input_shape[2]*input_shape[3]
        A = sparse.csr_matrix((num_channels, num_channels*input_elements_per_channel))
        val_matrix = np.ones((1,input_elements_per_channel))/input_elements_per_channel
        A_lil = A.tolil()
        for i in range(num_channels):
            A_lil[i,i*input_elements_per_channel:(i+1)*input_elements_per_channel] = val_matrix
        return A_lil.tocsr()
    
    def _get_layer_score_matrix(self, module, layer_info):
        if isinstance(module, nn.Conv2d):
            return abs(self._score_matrix_conv2d(module, layer_info['input_shape']))
        elif isinstance(module, nn.Linear):
            return abs(self._score_matrix_linear(module))
        elif isinstance(module, nn.MaxPool2d):
            return abs(self._score_matrix_maxpool2d(module, layer_info['input_shape']))
        elif isinstance(module, nn.BatchNorm2d):
            return abs(self._score_matrix_bn2d(module, layer_info['input_shape']))
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            return abs(self._score_matrix_avgpool2d(layer_info['input_shape']))
        else:
            raise ValueError('Module {} is of unknown type'.format(module))
    
    def get_num_channels(self, constraint, model):
        module = rgetattr(model, constraint['A'][0])
        if isinstance(module, nn.Conv2d):
            return module.weight.shape[0]
        elif isinstance(module, nn.Linear):
            return module.weight.shape[0]

    def pool_compute_score_matrix(self, layer_data):
        module = layer_data[0]
        info = layer_data[1]
        name = layer_data[2]
        return [name, self._get_layer_score_matrix(module, info)]
    
    def pool_multiply_matrices(self, matrices):
        matrix1 = matrices[0]
        matrix2 = matrices[1]
        name = matrices[2]
        return [name, matrix1 @ matrix2]
    
    def pool_compute_constraint_score_matrices(self, constraint, model, score_matrices):
        ips = []
        multiply_ips = []

        for key in constraint.keys():
            for layer_name in constraint[key]:
                layer = rgetattr(model, layer_name)
                ips.append([layer, self.model_info[layer_name], layer_name])
        if 'conv2' in constraint['A']:
            channels = model.conv2.out_channels
            eq_conv_layer = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(4,4), stride=(1, 1), bias=False, 
                            groups=channels)
            eq_conv_layer.weight = nn.Parameter(torch.ones(eq_conv_layer.weight.shape)/(eq_conv_layer.kernel_size[0]*eq_conv_layer.kernel_size[1]))
            ips.append([eq_conv_layer, {'input_shape': self.model_info['conv2']['output_shape']}, 'avgpool'])

        for layer_name in constraint['A']+constraint['dependent_layers']:
            if 'conv' in layer_name:
                bn_layer_name = layer_name.replace('conv', 'bn')
            elif 'shortcut' in  layer_name:
                bn_layer_name = layer_name[:-1] + '1'
            else:
                continue
            bn_layer = rgetattr(model, bn_layer_name)
            ips.append([bn_layer, self.model_info[bn_layer_name], bn_layer_name])
            multiply_ips.append([bn_layer_name, layer_name, layer_name+'_joint'])

        with Pool(max_workers=self.num_processors) as pool:
            for result in pool.map(self.pool_compute_score_matrix, ips):
                score_matrices[result[0]] = result[1]
        
        for i in range(len(multiply_ips)):
            bn_layer_name, layer_name = multiply_ips[i][0], multiply_ips[i][1]
            multiply_ips[i][0] = score_matrices[multiply_ips[i][0]]
            multiply_ips[i][1] = score_matrices[multiply_ips[i][1]]
            del score_matrices[bn_layer_name], score_matrices[layer_name]
        
        with Pool(max_workers=self.num_processors) as pool:
            for result in pool.map(self.pool_multiply_matrices, multiply_ips):
                score_matrices[result[0]] = result[1]
        
        if len(constraint['dependent_layers']):
            multiply_ips2 = []
            layer_name_A = constraint['A'][0]+'_joint'
            layer_name_dependent = constraint['dependent_layers'][0]+'_joint'
            multiply_ips2.append([score_matrices[layer_name_dependent],
                                    score_matrices[layer_name_A],
                                    layer_name_A])
            del score_matrices[layer_name_A], score_matrices[layer_name_dependent]

            with Pool(max_workers=self.num_processors) as pool:
                for result in pool.map(self.pool_multiply_matrices, multiply_ips2):
                    score_matrices[result[0]] = result[1]
        elif 'conv2' in constraint['A']:
            multiply_ips2 = []
            layer_name_A = constraint['A'][0]+'_joint'
            layer_name_dependent = 'avgpool'
            multiply_ips2.append([score_matrices[layer_name_dependent],
                                    score_matrices[layer_name_A],
                                    layer_name_A])
            del score_matrices[layer_name_A], score_matrices[layer_name_dependent]
            with Pool(max_workers=self.num_processors) as pool:
                for result in pool.map(self.pool_multiply_matrices, multiply_ips2):
                    score_matrices[result[0]] = result[1]

    
    def pool_compute_channel_score(self, data):
        matrix1 = data[0]
        matrix2 = data[1]
        return abs(matrix1 @ matrix2).sum()
    
    def pool_compute_channels_score_from_matrices(self, data):
        matrix1 = data[0].tocsc()
        matrix2 = data[1]
        num_channels = data[2]
        features_per_channel = matrix1.shape[1]//num_channels
        ips = []
        for i in range(num_channels):
            ips.append([matrix1[:,int(i*features_per_channel):int((i+1)*features_per_channel)],\
                matrix2[int(i*features_per_channel):int((i+1)*features_per_channel),:]])
        scores = []
        with Pool(max_workers=self.num_processors) as pool:
            for result in pool.map(self.pool_compute_channel_score, ips):
                scores.append(result)
        return np.array(scores)

    def pool_compute_constraint_scores(self, data):
        constraint = data[0]
        model = data[1]
        score_matrices = {}
        num_channels = self.get_num_channels(constraint, model)
        self.pool_compute_constraint_score_matrices(constraint, model, score_matrices)
        constraint_scores = 0
        ips = []
        for layer_name_B in constraint['B']:
            for layer_name_A in constraint['A']:
                if layer_name_B > layer_name_A or layer_name_B in ['conv2', 'linear']:
                    ips.append([score_matrices[layer_name_B], score_matrices[layer_name_A+'_joint'], num_channels])
        with Pool(max_workers=self.num_processors) as pool:
            for result in pool.map(self.pool_compute_channels_score_from_matrices, ips):
                constraint_scores += result
        return list(constraint_scores.flat)

    def MobileNetV2SaliencyScores(self, model):
        scores = []
        
        ips = []
        for i in range(len(self.geometric_constraints)):
            constraint = self.geometric_constraints[i]
            ips.append([constraint, model])
        
        with Pool(max_workers=self.num_processors) as pool:
            for result in pool.map(self.pool_compute_constraint_scores, ips):
                scores.append(result)
        
        return scores
    
    def total_channels(self, model):
        count = 0
        for i in range(len(self.geometric_constraints)):
            count += self.get_num_channels(self.geometric_constraints[i], model)
        return count
