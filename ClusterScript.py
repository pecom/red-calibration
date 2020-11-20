from mpi4py import MPI
import numpy as np
from ArrayScript import RealArray
import pickle

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
recvbuf = None

Nside = 15
data_len = int((Nside**4 - Nside**2)/2) 

if rank == 0:
    data = np.load('./data/cluster_data.npy')
    chin = np.load('./data/cluster_chin.npy')
#    beam = np.load('./data/cluster_beams.npy')
    recvbuf = np.empty([size, 1], dtype=np.float)
else:
    data = np.empty(data_len, dtype=np.complex128)
    chin = np.empty(data_len, dtype=np.complex128)
#    beam = np.empty((Nside**2, 3, 3), dtype=np.complex128)
    
comm.Bcast(data, root=0)
comm.Bcast(chin, root=0)
#comm.Bcast(beam, root=0)

arr = RealArray(Nside, 29)
arr.set_data(data)
arr.set_chin(chin)
print(rank)
#arr.create_fit(3, 0, ib=beam)
arr.create_fit(3, 0)
fname = "cluster_pickle_Nside"+str(Nside)+"_rank"+str(rank) + ".obj"
pick_file = open('./data/' + fname, 'wb')
pickle.dump(arr, pick_file)
pick_file.close()
isolve = arr.itersolve
fin_score = isolve[-1][-1]

comm.Gather(fin_score, recvbuf, root=0)
if rank == 0:
    print(recvbuf.flatten())
    np.save('./data/histscores_nside'+str(Nside)+'_m15', recvbuf.flatten())
