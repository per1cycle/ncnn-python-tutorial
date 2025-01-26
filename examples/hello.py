import ncnn

print(ncnn.__version__)

nnMat = ncnn.Mat(3, 4, 5)
nnLayer = ncnn.Layer()

print(nnMat)
print(nnLayer)
