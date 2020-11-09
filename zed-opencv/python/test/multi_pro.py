
import multiprocessing
from  multiprocessing import Pool
import time
import numpy as np


def Foo(i, j):
    t = np.random.random_integers(0, 10, 1)
    time.sleep(t)
    # print(i + 100)
    print('process :%d, loop:%d is called' % (i + 100, j))
    return i + 100


def __Bar(arg):
    print(arg)


def __looptake(i,kk):
    j = 0
    print(kk)
    while True:
        Foo(i, j)
        print('process :%d, loop:%d is started' % (i + 100, j))
        time.sleep(0.2)
        j = j + 1
def take_data():
    class multi_take:
      def __init__(self, interval, pron):
        self.pool = Pool(processes=pron)
        self.interval=interval
      def looptake(self,work,i):
        while True:
            work(i)
            time.sleep(0.2)
            print(i)
        pass
      def start(self,work,n,cbk):
        for i in range(n):
            kk="cam:%d"%(i)
            self.pool.apply_async(func=work,
                                  args=(i,kk,),
                                  callback=cbk)
            print("process: %d is started!"%(i))
      def terminate(self):
        self.pool.close()
        self.pool.terminate()

    mp = multi_take(0.2, 5)
    while True:
        comm = input('Please enter command(t: take data, q:quit, [0-2]: reinit camera): ')
        if not comm in ['t', 'q', '0', '1', '2']:
            continue
        if comm == 't':
            print("take start")
            mp.start(__looptake, 10, __Bar)
        elif comm == 'q':
            if mp is not None:
                print("end multi take")
            break
    mp.pool.close()
    mp.terminate()
if __name__ == "__main__":
    take_data()