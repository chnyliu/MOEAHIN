import os
import numpy as np
import pickle as pkl


def preprocess(dataset):
    prefix = os.path.join("data/", dataset)
    with open(os.path.join(prefix, "edges.pkl"), "rb") as f:
        edges = pkl.load(f)  # 共19645*2+14328*2=67946条边
        f.close()

    node_types = np.zeros((edges[0].shape[0],), dtype=np.int32)  # ndarray:(18405,)

    a = np.unique(list(edges[0].tocoo().row) + list(edges[2].tocoo().row))  # 统计所有paper的编号，共14328个
    b = np.unique(edges[0].tocoo().col)  # 统计所有author的编号，共4057个
    c = np.unique(edges[2].tocoo().col)  # 统计所有conference的编号，共20个
    print(a.shape[0], b.shape[0], c.shape[0])
    assert (a.shape[0] + b.shape[0] + c.shape[0] == node_types.shape[0])
    assert (np.unique(np.concatenate((a, b, c))).shape[0] == node_types.shape[0])

    node_types[a.shape[0]:a.shape[0] + b.shape[0]] = 1  # author点，14328：18385的点类型为1
    node_types[a.shape[0] + b.shape[0]:] = 2  # conference点，18385：18405的点类型为2, 【自然，paper点，0:14328的点类型为0】
    assert (node_types.sum() == b.shape[0] + 2 * c.shape[0])
    np.save(os.path.join(prefix, "node_types"), node_types)


if __name__ == "__main__":
    preprocess("DBLP")
    # main("ACM")
    # main("IMDB")
