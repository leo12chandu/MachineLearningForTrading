
import numpy as np
import time


def test_run():
    #1D array
    #print np.array([2, 3, 4])

    #2D array
    #print np.array([(2, 3, 4), (5, 6, 7)])

    #Empty array
    #print np.empty(5)
    #print np.empty((5, 4, 3))

    #Array of Ones
    #print np.ones((5, 4))
    #print np.ones((5, 4), dtype=np.int8)

    #Array of zeros
    #print np.zeros((5, 4), dtype=np.int8)

    #Generate an array full of random numbers, uniformly sampled from [0.0, 1.0) --> Zero inclusive and 1 exclusive
    #print np.random.random((5,4))
    #print np.random.rand(5,4) #--> same

    #Sample numbers from a Gaussian (normal) distribution
    #print np.random.normal(size=(2, 3)) # "standard normal" (mean = 0, s.d. = 1)
    #print np.random.normal(50, 10, size=(2, 3)) # "standard normal" (mean = 50, s.d. = 10)

    #Generate random integers
    #print np.random.randint(10) # a single integer in [0, 10)
    #print np.random.randint(0, 10) #Same as above
    #print np.random.randint(0, 10, size=5) #5 random integers as a 1D array
    #print np.random.randint(0, 10, size=(2,3)) # 2*3 array of random integers

    #shape of the array
    #a =  np.random.random((5,4))
    #print a
    #print a.shape
    #print 'Number of Rows = ', a.shape[0], '; Number of Columns', a.shape[1]
    #print 'Number of dimensions of the array :', len(a.shape)
    #print 'Number of elements of the array :', a.size
    #print 'datatype of the arrary is ', a.dtype
    
    
    #first 3 rows and 2nd and 3rd columns of data.
    #np1[0:3, 1:3]

    #All rows but on 4th column
    #np1[:, 3]

    #access numpy arrary from the bottom
    #np1[-1, 1:3]
    print 'End'

def array_operations():
    np.random.seed(693)
    a = np.random.randint(0, 10, size=(5, 4))
    print 'Array:\n', a

    print 'Sum of all elements: ', a.sum()
    print "Sum of each column:\n", a.sum(axis=0)
    print "Sum of each row:\n", a.sum(axis=1)

    print "Minimum of each column:\n", a.min(axis=0)
    print "Maximum of each row:\n", a.max(axis=1)
    print "Mean of all elements:", a.mean()
    print "Index of maximum value:\n", a.argmax()

def time_functions():
    t1 = time.time()
    print "ML4T"
    t2 = time.time()

    print "The time taken by print statement is ", t2 - t1, "seconds"

def how_long(func, *args):
    t0 = time.time()
    result = func(*args)
    t1 = time.time()
    return result, t1 - t0

def manual_mean(arr):
    sum=0
    for i in xrange(0, arr.shape[0]):
        for j in xrange(0, arr.shape[1]):
            sum = sum + arr[i, j]

    return sum / arr.size

def numpy_mean(arr):
    return arr.argmax()

def time_mean_functions():
    nd1 = np.random.random((1000, 10000)) #Large Array

    res_manual, t_manual = how_long(manual_mean, nd1)
    res_numpy, t_numpy = how_long(numpy_mean, nd1)

    print "Manual mean result: ", res_manual, "; Time taken: ", t_manual
    print "Numpy mean result: ", res_numpy, "; Time taken: ", t_numpy

def access_array():
    a = np.random.rand(5, 4)
    print "Array:\n", a

    element = a[3, 2]
    print element

    print a[0, 1:3]
    print a[0:2, 0:2]

    ## These ranges work just like slices and python lists. n:m:t specifies a range
    ## that starts at n, and stops before m, in steps of size t. If any of these 
    ## are left off, they're assumed to be the start, the end+1, and 1 respectively
    print a[:, 0:3:2]

def modify_array():
    a = np.random.rand(5,4)
    print "Array:\n", a

    a[0, 0] = 1
    print a

    #All columns of row 1 with value 2.
    a[0, :] = 2
    print a

    #Assign 4th column with multiple values each row a different value.
    a[:, 3] = [1, 2, 3, 4, 5]
    print a

def index_array_with_another():
    a = np.random.rand(5)

    indices = np.array([1, 1, 2, 3])
    print a[indices]

def bool_mask_index_arrays():
    a = np.array([(20, 25, 10, 23, 26, 32, 10, 5, 0), (0, 2, 50, 20, 0, 1, 28, 5, 0)])
    print a

    mean = a.mean()
    print mean

    print a[a<mean]

    #Masking
    a[a<mean] = mean
    print a

def arithmetic_operations():
    a = np.array([(1, 2, 3, 4, 5), (10, 20, 30, 40, 50)])
    print "Original Array a:\n", a

    print "Multuply a by 2:\n", 2 * a
    print "Divide a by 2: \n", a / 2
    print "Divide a by 2.0 the float value: \n", a / 2.0

    b = np.array([(100, 200, 300, 400, 500), (1, 2, 3, 4, 5)])
    print "Original Array b:\n ", b
    print "Add a and b:\n ", a + b

    #NOT MATRIX MULTIPLICATION
    print "Array multiplication: \n", a * b

    #Division of 2 arrays
    print "Array division: \n", a / b
    
    
if __name__ == "__main__":
    #test_run()
    #array_operations()
    #time_functions()
    #time_mean_functions()
    #access_array()
    #modify_array()
    #index_array_with_another()
    #bool_mask_index_arrays()
    arithmetic_operations()
