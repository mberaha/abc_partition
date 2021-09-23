import pickle
import numpy as np
import abcpp as abc

from joblib import Parallel, delayed


def run_one(ndata):
    clus1 = np.zeros((ndata, 2))
    clus2 = np.zeros((ndata, 2))

    for i in range(clus1.shape[0]):
        clus1[i, :] = abc.rand_gandk(
            0.5, [-3, -3], [0.75, 0.75], [-0.9, -0.9], [0.1, 0.1])
        clus2[i, :] = abc.rand_gandk(
            0.5, [3, 3], [0.5, 0.5], [0.4, 0.4], [0.5, 0.5])
        
    out_wass = abc.run_gandk(
        np.vstack([clus1, clus2]), 0.5, 20000, 10000, 1.0, 0.2, 100, 100, 2, 
        "wasserstein", [], False)
    out_sink = abc.run_gandk(
        np.vstack([clus1, clus2]), 0.5, 20000, 10000, 1.0, 0.2, 100, 100, 2, 
        "sinkhorn", [], False)
    out_green = abc.run_gandk(
        np.vstack([clus1, clus2]), 0.5, 20000, 10000, 1.0, 0.2, 100, 100, 2, 
        "greenkhorn", [], False)

    return out_wass, out_sink, out_green


if __name__ == "__main__":
    
    fd = delayed(run_one)
    out100 = Parallel(n_jobs=19)([fd(50) for _ in range(50)])
    with open("gnk_results100.pickle", "wb") as fp:
        pickle.dump(out100, fp)

    out250 = Parallel(n_jobs=19)([fd(125) for _ in range(50)])
    with open("gnk_results250.pickle", "wb") as fp:
        pickle.dump(out250, fp)

    out1000 = Parallel(n_jobs=19)([fd(500) for _ in range(50)])
    with open("gnk_results1000.pickle", "wb") as fp:
        pickle.dump(out1000, fp)
