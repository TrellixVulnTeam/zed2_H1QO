import numpy
import time

n  = 2000
A  = numpy.arange(0, n * n).reshape(n, n) + numpy.identity(n) # ランク落ちを防ぐため
b  = numpy.arange(0, n)

t1 = time.time()
numpy.linalg.inv(A)
t2 = time.time()

t3 = time.time()
numpy.linalg.solve(A, numpy.identity(n))
t4 = time.time()

t5 = time.time()
numpy.dot(numpy.linalg.inv(A), b)
t6 = time.time()

t7 = time.time()
numpy.linalg.solve(A, b)
t8 = time.time()