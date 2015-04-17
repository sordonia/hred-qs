__docformat__ = 'restructedtext en'
__authors__ = ("Alessandro Sordoni")
__contact__ = "Alessandro Sordoni <sordonia@iro.umontreal>"

import numpy as np
import os, gc
import cPickle
import copy
import logging

import threading
import Queue

import collections

logger = logging.getLogger(__name__)

class SSFetcher(threading.Thread):
    def __init__(self, parent):
        threading.Thread.__init__(self)
        self.parent = parent
        self.indexes = np.arange(parent.data_len)

    def run(self):
        diter = self.parent
        self.parent.rng.shuffle(self.indexes)
        
        offset = 0 
        # Take groups of 10000 sentences and group by length
        while not diter.exit_flag:
            last_batch = False
            
            sessions = []
            ranks = []
             
            while len(sessions) < diter.batch_size:        
                if offset == diter.data_len:
                    if not diter.use_infinite_loop:
                        last_batch = True
                        break
                    else:
                        # Infinite loop here, we reshuffle the indexes
                        # and reset the offset 
                        self.parent.rng.shuffle(self.indexes)
                        offset = 0
                
                index = self.indexes[offset]
                s = diter.data[index]
                offset += 1
                
                if len(s) > diter.max_len: 
                    continue

                # Append tuple if rank file is specified
                sessions.append(s)
                if diter.has_ranks:
                    r = diter.rank_data[index] 
                    ranks.append(r)

            if len(sessions):
                if diter.has_ranks:
                    diter.queue.put((sessions, ranks)) 
                else:
                    diter.queue.put(sessions)
            if last_batch:
                diter.queue.put(None)
                return

class SSIterator(object):
    def __init__(self,
                 rng,
                 batch_size,
                 session_file=None,
                 rank_file=None,
                 dtype="int32",
                 can_fit=False,
                 queue_size=100,
                 cache_size=100,
                 shuffle=True,
                 use_infinite_loop=True,
                 max_len=1000):

        args = locals()
        args.pop("self")
        self.__dict__.update(args)
        self.has_ranks = rank_file is not None
        self.load_files()
        self.exit_flag = False

    def load_files(self):
        self.data = cPickle.load(open(self.session_file, 'r'))
        self.data_len = len(self.data)
        logger.debug('Data len is %d' % self.data_len) 
        
        if self.has_ranks:
            self.rank_data = cPickle.load(open(self.rank_file, 'r'))
            self.rank_data_len = len(self.rank_data) 

            assert self.rank_data_len == self.data_len
            logger.debug('Rank data len is %d' % self.rank_data_len)

    def start(self):
        self.exit_flag = False
        self.queue = Queue.Queue(maxsize=self.queue_size)
        self.gather = SSFetcher(self)
        self.gather.daemon = True
        self.gather.start()

    def __del__(self):
        if hasattr(self, 'gather'):
            self.gather.exitFlag = True
            self.gather.join()

    def __iter__(self):
        return self

    def next(self):
        if self.exit_flag:
            return None
        batch = self.queue.get()
        if not batch:
            self.exit_flag = True
        return batch
