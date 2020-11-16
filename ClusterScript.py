from mpi4py import MPI
import numpy as np
from ArrayScript import RealArray

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
recvbuf = None

if rank == 0:
    data = np.load('./data/cluster_data.npy')
    chin = np.load('./data/cluster_chin.npy')
    recvbuf = np.empty([size, 1], dtype=np.float)
else:
    data = np.empty(4950, dtype=np.complex128)
    chin = np.empty(4950, dtype=np.complex128)
    
comm.Bcast(data, root=0)
comm.Bcast(chin, root=0)

arr = RealArray(10, 3)
arr.set_data(data)
arr.set_chin(chin)
arr.create_fit(3, 0)
isolve = arr.itersolve
fin_score = isolve[-1][-1]

comm.Gather(fin_score, recvbuf, root=0)
if rank == 0:
    print(recvbuf.flatten())
    np.save('./data/histscores_nside10', recvbuf.flatten())
