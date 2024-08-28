#learning_profiling.py

#import modules
import time
import cProfile
import pstats

#mock program

def short():
    time.sleep(0.01)

def long():
    time.sleep(0.1)

def call_short():
    for i in range(100):
        short()

def call_long():
    for i in range(100):
        long()

cProfile.run('call_short()', 'C:\\Users\\josie\\short.stats')

stats = pstats.Stats('C:\\Users\\josie\\short.stats')
