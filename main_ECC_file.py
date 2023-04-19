#!/usr/bin/env python
# coding: utf-8

# In[529]:


# Imports all the required packages
import os
import argparse
import numpy as np
import os.path as osp
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,Data)

import torch
#from torcheval.metrics import R2Score # To be implemented
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import DataStructs
from torch_geometric.loader import DataLoader

import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
import torch.optim as optim
from torch_geometric.nn import GCNConv,GraphConv,NNConv
from torch_geometric.nn import global_add_pool,knn_graph

try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import rdBase
    from rdkit.Chem.rdchem import HybridizationType
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    from rdkit.Chem.rdchem import BondType as BT
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import MolsToGridImage
    rdBase.DisableLog('rdApp.error')
except ImportError:
    print("rdkit = None")


# In[530]:


# Model definition used for regression

class GCNlayer(nn.Module):
    
    def __init__(self, n_features, conv_dims, concat_dim, dropout):
        super(GCNlayer, self).__init__()
        self.n_features = n_features
        self.conv_dims = conv_dims
        self.concat_dim =  concat_dim
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.gru=nn.ModuleList()
        #args.conv_dims=[args.n_features,128,512,4092]
        self.lin0 = torch.nn.Linear(args.n_features, self.conv_dims[0]) # Changed from 128
        for i in range(len(self.conv_dims)-1):
            kk = nn.Sequential(Linear(6, self.conv_dims[i+1]), nn.ReLU(), Linear(self.conv_dims[i+1], self.conv_dims[i+1]*self.conv_dims[i+1]))
            conv = NNConv(self.conv_dims[i], self.conv_dims[i+1],kk,aggr='add')
            self.convs.append(conv)
            bn = BatchNorm1d(self.conv_dims[i+1])
            self.bns.append(bn)
            self.gru.append(nn.GRU(self.conv_dims[i+1], self.conv_dims[i+1])) # Changed
        self.conv_last = GraphConv(self.conv_dims[-1],self.conv_dims[-1])  # Last layer
        self.bn_last = BatchNorm1d(self.concat_dim)
        
    def forward(self, data):
        x, edge_index,edge_attr= data.x, data.edge_index,data.edge_attr # Changed . added edge_attr
        out = F.relu(self.lin0(x))
        h = out.unsqueeze(0)
        for i in range(len(self.convs)):
            m = F.relu(self.convs[i](out, edge_index,edge_attr)) # Changed . added edge_attr
            m = self.bns[i](m)
            out, h = self.gru[i](m.unsqueeze(0), h)  # Changed here. Added [i]
            out = out.squeeze(0)

        x = F.relu(self.conv_last(out, edge_index))# Changed . added edge_attr
        x = self.bn_last(x)
        x = global_add_pool(x,batch=data.batch)   #global_add_pool  cHANGED
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
class FClayer(nn.Module):
    
    def __init__(self, concat_dim, pred_dims, out_dim, dropout):
        super(FClayer, self).__init__()
        self.concat_dim = concat_dim
        self.pred_dims = pred_dims
        self.out_dim = out_dim
        self.dropout = dropout

        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()
        # Linear layer has bias default=True
        for i in range(len(self.pred_dims)-1):
            fc = Linear(self.pred_dims[i], self.pred_dims[i+1])
            self.fcs.append(fc)
            bn = BatchNorm1d(self.pred_dims[i+1])
            self.bns.append(bn)
        self.fc_last = Linear(self.pred_dims[-1], self.out_dim)
    
    def forward(self, data):
        x = data
        for i in range(len(self.fcs)):
            x = F.relu(self.fcs[i](x))
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc_last(x)
        return x

# Model definition  
class GCN_Model_reg(nn.Module):
    def __init__(self, args):
        super(GCN_Model_reg, self).__init__()
        
        # Convolutional Layer call
        self.conv = GCNlayer(args.n_features,args.conv_dims,args.concat_dim,args.dropout)

        # Fully connected layer call
        self.fc = FClayer(args.concat_dim,args.pred_dims,args.out_dim,args.dropout)
        
    def forward(self, data):
        x = self.conv(data) # Calling the convolutional layercpu
        x = self.fc(x) # Calling the Fully Connected Layer
        return x


# In[531]:


# pytorch will be run on cpu
seed=200
paser = argparse.ArgumentParser()
args = paser.parse_args("")
np.random.seed(200)
torch.manual_seed(seed)
device='cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
#'cuda' if torch.cuda.is_available() else
device


# In[532]:


# Inputs required by the program
args.epoch = 400
args.lr = 0.0001
args.optim = 'Adam'
args.step_size = 25
args.gamma = 0.8
args.dropout = 0 # Tested for 0.2 but max r2 got 0.874. For 0.1 I got 0.897
args.n_features = 30  # Mass added else 29. Without atomic mass r2 0.899
args.concat_dim=128
args.conv_dims=[128,128,128] # Removed 1 layer
args.pred_dims=[128,512,128]
args.out_dim = 1


# In[533]:


# encoding of all variables

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
 
def get_intervals(l):
    """For list of lists, gets the cumulative products of the lengths"""
    intervals = len(l) * [0]
    # Initalize with 1
    intervals[0] = 1
    for k in range(1, len(l)):
        intervals[k] = (len(l[k]) + 1) * intervals[k - 1]
    return intervals

def safe_index(l, e):
    """Gets the index of e in l, providing an index of len(l) if not found"""
    try:
        return l.index(e)
    except:
        print("Add value to list of length {}".format(len(l)))
        return len(l)


# In[534]:


# Atom features

possible_atom_list = ['H','C', 'O', 'F','N','Cl','P','S','Si','Br','I']  # Atomic symbol 11

aromatic=[0,1] # Aromatic 1
isring=[0,1]   # Ring 1
possible_numH_list = [0, 1, 2, 3, 4] # Total Number of bonded hydrogen atoms possible 5
num_bonds = [0, 1, 2, 3, 4, 5]  # Total Number of Hs a carbon can bond / Total number of bonds an atom make 6
possible_formal_charge_list = [-4,-3, -2, -1, 0, 1, 2, 3, 4]

# sp3d is removed because it doesnot vary according to the paper . Hybridization 4
possible_hybridization_list = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,Chem.rdchem.HybridizationType.SP3,Chem.rdchem.HybridizationType.SP3D2]

# storing all the features in a detailed list
reference_lists = [possible_atom_list, possible_numH_list,possible_formal_charge_list, num_bonds,aromatic,isring,possible_hybridization_list]

intervals = get_intervals(reference_lists)
print(intervals)


# - Total number of atom features used here is = 29

# In[535]:


# Concatenate the entire feature list
def get_feature_list(atom):
    features = 5 * [0]
    features[0] = safe_index(possible_atom_list, atom.GetSymbol())
    features[1] = safe_index(possible_numH_list, atom.GetTotalNumHs())
    features[2] = safe_index(num_bonds, atom.GetImplicitValence())
    features[3] = safe_index(possible_formal_charge_list, atom.GetFormalCharge())    
    features[4] = safe_index(possible_hybridization_list, atom.GetHybridization())
    return features


# In[536]:


def atom_features(atom,bool_id_feat=False,explicit_H=False):
    from rdkit import Chem
    results = np.array(one_of_k_encoding_unk(atom.GetSymbol(),possible_atom_list) + 
                           one_of_k_encoding_unk(atom.GetImplicitValence(), num_bonds) + 
                           [atom.GetFormalCharge()] + 
                           one_of_k_encoding_unk(atom.GetHybridization(), possible_hybridization_list) + 
                           [atom.GetIsAromatic()]+[atom.IsInRing()]+[atom.GetAtomicNum()])  # Atomic mass added  atom.GetAtomicNum() r2=0.907
    if not explicit_H:
        results = np.array(results.tolist() + one_of_k_encoding_unk(atom.GetTotalNumHs(),possible_numH_list))  # include neighbouring Hs
    return np.array(results)


# In[537]:


# Bond Features

def bond_features(bond):
    from rdkit import Chem
    bt = bond.GetBondType()
    bond_feats = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                  bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                  bond.GetIsConjugated(),bond.IsInRing()]
    
    # Include stereo bond features as it effects the boiling point
    #bond_feats = bond_feats + one_of_k_encoding_unk(str(bond.GetStereo()),["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats)


# - In the above bond features we have commented the stereo feature because we have seen that the smiles string has not been encoded to represent the information of the stereo state of the chemical formula.

# In[538]:


# Create  atom pair of two connected molecules to pass messages

def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[],[]]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res

# create a graph data structure comprising of x=node_f, edge_index = bond pair info, 
# Here we are taking converted molecule from smiles as string
def mol2vec(mol):
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f= [atom_features(atom) for atom in atoms]
    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond) for bond in bonds]

    for bond in bonds:
        edge_attr.append(bond_features(bond))
        
    # Graph data to be used
    # changes made in edge index format 
    data = Data(x=torch.tensor(node_f, dtype=torch.float),edge_index=torch.tensor(edge_index, dtype=torch.long),edge_attr=torch.tensor(edge_attr,dtype=torch.float))
    
    #print(data)
    return data,edge_attr,node_f


# In[539]:


# Return mol info
def make_mol(df_1):
    mols = {}
    for i in range(df_1.shape[0]):
        mols[Chem.MolFromSmiles(df_1['Smiles'].iloc[i])] = df_1['Tb'].iloc[i]
    return mols


# In[540]:


# Convert the dataset with X values and y values to be trained/tested
def make_vec(mols):
    X = []
    for m in mols.keys():
        s,_,_=mol2vec(m) 
        X.append(s)
    for i, data in enumerate(X):
        y = list(mols.values())[i]
        data.y = torch.tensor([y], dtype=torch.float)
    return X


# In[541]:


# Pca on graph data
#import umap
from sklearn.decomposition import PCA
def tsne_g(n,e):
    #print(e[0],e[1])
    model=PCA(n_components=20)
    # apply padding
    max_len_n = max(x.shape[0] for x in n)
    target_shape=np.array([0 for i in range(6)])
    for i in range(len(n)):
        c=max_len_n-n[i].shape[0]
        for j in range(c):
          n[i]=np.vstack((n[i], target_shape))
    #print(n)


    max_len_e = max(x.shape[0] for x in e)
    target_shape=np.array([0 for i in range(29)])
    for i in range(len(e)):
        c=max_len_n-e[i].shape[0]
        for j in range(c):
          e[i]=np.vstack((e[i], target_shape))
    #print(e)
    
    n=np.array(n)
    e=np.array(e)
    print(n.shape,e.shape)
    feature_matrix = np.concatenate((n,e), axis=2)

    # apply t-SNE
    #tsne = TSNE(n_components=5,perplexity=30)

    transformed_matrix = model.fit_transform(feature_matrix)

    print("Original feature matrix:")
    print(feature_matrix)
    print("\nTransformed feature matrix:")
    print(transformed_matrix)
    


# In[542]:


def call_pca(mols):
    n=[]
    e=[]
    for m in mols.keys():
        _,i,j=mol2vec(m) 
        n.append(np.array(i))
        e.append(np.array(j))
    tsne_g(n,e)


# ### Importing dataset

# In[543]:


# Importing Dataset
dataset = pd.read_csv('data/raw_data.csv', low_memory=False)
dataset = pd.concat([dataset['Smiles'], dataset['Tb']], axis=1)
dataset


# In[544]:


# Checking for duplicate values or duplicate smile string
duplicate = dataset[dataset.duplicated('Smiles')]
 
print("Duplicate Rows based on City :")
 
# Print the resultant Dataframe
print(duplicate)


# - So as we can see from the data there are 187 duplicate smiles strings. So we are removing all of them as we didnot find any relevant feature difference.

# In[545]:


# Removing the duplicates
dataset.drop_duplicates(subset="Smiles", inplace=True)
dataset


# - So after removing all the duplicates we are left with a dataset of shape (5089,2)

# In[546]:


# Standardize the Boiling Point values
mean=dataset['Tb'].mean()
std=dataset['Tb'].std()
dataset['Tb']=(dataset['Tb']-mean)/std
dataset.describe()
print(mean,std)


# In[547]:


# Scale the dataset to values between 0 and 1

min_val=dataset['Tb'].min()
max_val=dataset['Tb'].max()
dataset['Tb']=(dataset['Tb']-min_val)/(max_val-min_val)
dataset.describe()
print(max_val,min_val)


# In[548]:


# Remove noise using Boxplot
# Finding the outliers 
import seaborn as sns

# Plot the boxplot of all columns
sns.boxplot(dataset['Tb'])

# Calculate the upper and lower bounds for each column using the interquartile range (IQR)
Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 2 * IQR
upper_bound = Q3 + 2 * IQR

# Drop the rows containing the outliers in each column
dataset = dataset[~((dataset < lower_bound) | (dataset > upper_bound)).any(axis=1)]



 



def R2_score(y_pred_i,Y):
    #Next we shall compute sigma_yy, which is the variance of y 
    sigma_yy = 0
    for i in range(Y.shape[0]): #access each row from the data set 
       y_i = float(Y[i])
       sigma_yy += (y_i-Y.mean())**2

    #now we can compute sum of squared residuals 
    #print('sigma_yy:', sigma_yy)
    sum_sq_residuals=0
    for i in range(Y.shape[0]): #access each row from the data set 
         y_i = Y[i] #access i-th row of y
         e_i = y_i - y_pred_i[i] #compute the difference between the actual observation y_i and prediction y_pred_i

         sum_sq_residuals += (e_i)**2

    #print('sum of squared residuals:', sum_sq_residuals)

    #Then we will compute the R^2 quantity
    R_sq = 1-sum_sq_residuals/sigma_yy
    return R_sq


def test(model, device, test_loader, args):
    test_score = 0
    y_score=[]
    y_test=[]
    with torch.no_grad():
     for i, data in enumerate(test_loader):
        data = data.to(device)
        targets = data.y.to(device)
        outputs = model(data)
        #print(outputs)
        #print(targets)
        outputs.require_grad = False
        #_, predicted = torch.max(outputs.data, 1)
        #train_total += targets.size(0)

        correct_dim_output=outputs.squeeze() # done to reshape the outputs and match with targets
        #y_score.append(outputs.cpu().numpy())
        #y_test.append(targets.cpu().numpy())
        test_score=R2_score(outputs.cpu().numpy(),targets.cpu().numpy())
        print("R2 Score Validation test set:",R2_score(outputs.cpu().numpy(),targets.cpu().numpy()))
    return test_score,model,outputs.cpu().numpy(),targets.cpu().numpy()

#model = GCN_Model_reg(args)











