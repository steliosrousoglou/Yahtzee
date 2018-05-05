import random

class Multiset:
    ''' A multiset of {0, ..., n-1}
    '''

    def __init__(self, poss):
        ''' Creates an empty multiset that can contain values up to poss-1.

            poss -- a positive integer
        '''
        if poss <= 0:
            raise ValueError("number of possible values must be positive %d" % sides)

        self.maximum = poss

        # initially empty
        self.freq = [0] * poss
        self._size = 0


    def add_random(self, count):
        ''' Adds the given number of uniformly randomly selected values to this multiset.

            count -- a non-negative integer
        '''
        if count < 0:
            raise ValueError("number of elements must be non-negative %d" % count)

        outcomes = range(len(self.freq))
        for d in range(count):
            elt = random.choice(outcomes)
            self.freq[elt] += 1
        self._size += count


    def add(self, num):
        ''' Adds one occurrence of the given number to this multiset.
        
            num -- a valid element for this multiset
        '''
        if num < 0 or num >= len(self.freq):
            raise ValueError("element out of range: %d" % num)
            
        self.freq[num] += 1
        self._size += 1


    def count(self, num):
        ''' Returns the number of the given element in this multiset.
            
            num -- an integer
        '''
        if num < 0 or num >= len(self.freq):
            return 0
        else:
            return self.freq[num]


    def size(self):
        ''' Returns the number of elements in this multiset.
        '''
        return self._size


    def subset(self, other):
        ''' Determines if this multiset is a subset of the given multiset.
        
            other -- a multiset
        '''
        for elt, count in enumerate(self.freq):
            if other.count(elt) < count:
                return False
        return True

        
    def total(self):
        ''' Returns the total of the items in this multiset.
        '''
        # should probably implement an iterator so we can just use standard Python fxns
        # instead of implementing them ourselves
        return sum([elt * count for elt, count in enumerate(self.freq)])


    def as_list(self):
        ''' Returns a list of the elements in this multiset.
        '''
        # again, an iterator would allow us to just use list
        elements = []
        for elt, count in enumerate(self.freq):
            elements.extend([elt] * count)
        return elements


    def __str__(self):
        ''' Returns a string representation of this multiset.
        '''
        return str(self.as_list())
