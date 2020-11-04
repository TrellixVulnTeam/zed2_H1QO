import numpy as n

# generate some random test points
m = 20 # number of points
delta = 0.01 # size of random displacement
origin = n.random.rand(3,1) # random origin for the plane
basis = n.random.rand(3,2) # random basis vectors for the plane
coefficients = n.random.rand(2,m) # random coefficients for points on the plane

# generate random points on the plane and add random
points = n.dot(basis,coefficients)\
         + n.dot(origin,n.full((1,m),1))\
         + delta * n.random.rand(3,m) #displacement

# now find the best-fitting plane for the test points

# subtract out the centroid
points = n.transpose(n.transpose(points) - n.sum(points,1) / m)

# singular value decomposition
svd = n.transpose(n.linalg.svd(points))

n.transpose(svd [0])
# its dot product with the basis vectors of the plane is approximately zero
n.dot(n.transpose(svd [0]),basis)