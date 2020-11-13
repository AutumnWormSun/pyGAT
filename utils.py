import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}  # 将content文件中行的序号作为Key，将对应文章的ID作为Value，便于后面edges数组的构造
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    # 将sites文件中文章的编号替换为content文件中行的序号，与前面构造的标签数组中特征的标签顺序一致
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    # coo_matrix()，元组中第一个参数为矩阵元素数组，第二个参数为元素对应的行与列索引数值，矩阵大小通过shape控制
    # 行与列索引值可以重复出现，最后矩阵对应的元素会被求和，可能是通过shape的赋值进行操作的，没有查到相关使用说明
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    # 有向图邻接矩阵转换为无向图邻接矩阵，网上说这个对后面实验效果有很大影响，表达式实际就是按主对角线翻转非零元素
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))  # 返回密集存储的Numpy数组，然后转换为CUDA上的torch.float32类型
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D(-1/2)
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.  # 意思和特征行规范化一致
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)  # 意思和特征行规范化一致
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)  # (AD(-1/2))TD(-1/2)=D(-1/2)AD(-1/2)，D是对角阵，它的逆是自身


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.  # 如果索引对应的元素为无穷小则赋值为0
    r_mat_inv = sp.diags(r_inv)  # 以每行元素和的倒数作为对角阵上的元素 
    mx = r_mat_inv.dot(mx)  # 两个矩阵的乘法，如果是两个向量则是点积操作
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

