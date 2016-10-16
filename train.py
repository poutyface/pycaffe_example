import caffe
import numpy as np

caffe.set_mode_cpu()
#caffe.set_device(0)
#caffe.set_mode_gpu()


#net = caffe.Net("models/train.proto", caffe.TEST)
solver = caffe.SGDSolver("models/solver.proto")
net = solver.net

print net.blobs["data"].data.shape
print net.blobs["conv"].data.shape

while solver.iter < 100:
    net.blobs["data"].data[...] = np.zeros(net.blobs["data"].data.shape, dtype=np.float32)
    net.blobs["label"].data[...] = np.ones(net.blobs["conv"].data.shape, dtype=np.float32)

    print "------- data --------"
    print net.blobs["data"].data
    print "------- label --------"
    print net.blobs["label"].data


    solver.step(1)
    #solver.solve()
    print "------- loss --------"
    print net.blobs["loss"].data
    print "------- conv --------"
    print net.blobs["conv"].data

