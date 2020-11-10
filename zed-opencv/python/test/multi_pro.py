
import multiprocessing
from  multiprocessing import Pool
import time
import numpy as np
from easydict import EasyDict


def Foo(i, j):
    t = np.random.randint(0, 10)
    time.sleep(t)
    # print(i + 100)
    print('process :%d, loop:%d is called' % (i + 100, j))
    return i + 100


def __Bar(arg):
    t = np.random.randint(0, 10)
    time.sleep(t)
    print(arg)

def Bar2(arg):
    print("I am __Bar2: %s"%arg)

def __looptake(*args,**kwargs):
    i=args[0]
    menu=kwargs
    # print("__looptake",i)
    # print("__looptake",menu)
    j=0
    while True:
        # Foo(i, j)
        Bar2("__looptake")
        print('process :%d, loop:%d is started' % (i + 100, j))
        time.sleep(0.2)
        j = j + 1
def take_data():
    class multi_take:
      def __init__(self, interval, pron):
        self.pool = Pool(processes=pron)
        self.interval=interval
      def start(self,work,menus,cbk):
        for i ,menu in enumerate(menus):
            kk=["cam:%d"%(i),"cam:%d"%(i)]
            self.pool.apply_async(func=work,
                                  args=(i,),
                                  kwds=menu,
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
            menus=[]
            def get_menu():
                menu = EasyDict({})
                menu.init = False
                return menu

            menus.append(get_menu())
            menus.append(get_menu())
            menus.append(get_menu())
            mp.start(__looptake, menus, __Bar)
        elif comm == 'q':
            if mp is not None:
                print("end multi take")
            break
    mp.pool.close()
    mp.terminate()
if __name__ == "__main__":
    take_data()