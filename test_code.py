import sys
import time
gpus = sys.argv[1]
# gpus = [int(gpus.split(','))]
batch_size = sys.argv[2]
def print1():

    print("batch_size", batch_size)
def print2():
    print("gpus", gpus)

if __name__ == "__main__":
    print1()    
    time.sleep(5)
    print2()