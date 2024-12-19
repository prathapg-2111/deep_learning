import numpy as np
import matplotlib.pyplot as plt 
import torch

# 1 Declaring an array in numpy, tensorflow or pytorch framework is used for operations

a = np.array(  [4,5,2,6,8] ) 
print(a)
print("\n")

#2 Declaring an array in numpy, with data type set to float32
b = np.array([1, 3, 2, 5] , dtype='float32' ) 
print(b)
print("\n")

# 3 Declaring 2D arrays and using it in numpy
list_of_lists = [[1, 2, 3], 
                 [4, 4, 5] , 
                 [6, 2, 11]] 
       
c = np.array(list_of_lists)
print(c)
print("\n")

#4  Initialize an array d with zeros of given size  10, used in matrix multiplications, 
# sometimes they may get initialized with random values, set to all zeros
d = np.zeros(10, dtype=int) 
print(d)
print("\n")

##5 Create a 4X6 matrix of type float using np ones
## Librar
e = np.ones((4, 6), dtype=float) 
print(e)
print("\n")

## 6create any array with specific number using np.full of given size
f = np.full((3, 3), 42) 
print(f)
print("\n")

#7 np arrange is useful to create numpy arrays with different data in them.
# Below code gives us numpy array with 10 values in the range between 1 to 28 with step of size 3
g = np.arange(1, 30, 3) 
print(g)
print("\n")

## 8np linspace another way to creating numpy arrays with data in them

h = np.linspace(0, 1, 20) 
print(h)
print("\n")

## 9Generate random data like the following. Below is the 4X4 matrix with random values in it
g = np.random.random((4,4))
print(g)
print("\n")


## 10 Generate random data with a mean of 0 and standard deviation of 1 
## mean 0 and standard deviation 1
h = np.random.normal(0, 1, (4,4)) 
print(h)
print("\n")

## 11 Generate identity matrix and a transpose
## Generate 5X5 matrix like the following with 
## one values only on the diagonal
## Several matrix operation rules hold when using identity matrix
## which can be used when doing tensor operations for deep learning.
## for any square matrix A the following relation holds
#AI = IA = A
#AA(to the power -1) = A
## using transpose we have AA(Transpose)  = I
i = np.eye(5) 
print(i)
print("\n")
## 12 identity matrix
## Transpose operations are sometimes necessary for matrix multiplications
## It can rotate row vectors in to column vectors.
## Sometimes when doing linear algebra operations you need special matrices like 
# identity matrix.
# The transpose operations are sometimes necessary for matrix multiplications
#[[1, 2, 3],             [ [1, 4]
# [4, 5, 6]]        =      [2, 5]
#                          [3, 6] ]
## GPUs were invented for computer graphics, in video games, you have a mesh
## of all the values in a 3D space, and when you want to move that object of pixels
## from one precision to another precision if you run that with a CPU what would happen
## is we will get the tracing effect of lines on the screen, GPU are invented basically 
## all the points are rendered at the same time for vision.
## In linear algebra we can it stretching and rotation, it a very efficient operation.
## so that you dont have lag in machine learning (ex:w times x is equal y)
## Assuming A and B are matrics and K is a scalar, the following should hold for a transpose
## A scalar is just a vector with one value in it 
#     (A+B)Transpose = A(Transpose) + B(Transpose)
#       (kA)Transpose  = kA(Tranpose)
#      (A(Transpose))Transpose = A
##      (AB)Transpose = B(Transpose)A(Transpose)
## using numpy proving identity and transpose operations. creating matrix with data in them
# pay close attention to dimentions
j =  np.array([[ 1,  2,  3 ],
               [ 4,  5,  6 ]])
p_tra = np.transpose(j)
print(p_tra)


# 13. Identity Matrix Operations
print("1. Identity Matrix Operations")
print("-" * 50)

# 14 Create a 5x5 identity matrix
I = np.eye(5)
print("5x5 Identity Matrix:")
print(I)
print("\n")

# 15 Create a sample 5x5 matrix
A = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
])
print("Matrix A:")
print(A)
print("\n")

# 16 Demonstrate AI = IA = A
print("Proving AI = IA = A:")
print("AI =")
print(np.dot(A, I))
print("\nIA =")
print(np.dot(I, A))
print("\nOriginal A =")
print(A)
print("\n")

# 17. Transpose Operations
print("17. Transpose Operations")
print("-" * 50)

# 18 Create a 2x3 matrix
B = np.array([[1, 2, 3],
              [4, 5, 6]])
print("Original Matrix B (2x3):")
print(B)
print("\nTranspose of B (3x2):")
print(B.T)
print("\n")

# 19. Proving Transpose Properties
print("19. Proving Transpose Properties")
print("-" * 50)

# (A+B)ᵀ = Aᵀ + Bᵀ
C = np.array([[1, 2],
              [3, 4]])
D = np.array([[5, 6],
              [7, 8]])

print("Matrix C:")
print(C)
print("\nMatrix D:")
print(D)
print("\nProving (C+D)ᵀ = Cᵀ + Dᵀ")
print("Left side (C+D)ᵀ =")
print((C + D).T)
print("\nRight side Cᵀ + Dᵀ =")
print(C.T + D.T)
print("\n")

# 20. Scalar Multiplication
print("20. Scalar Multiplication")
print("-" * 50)

k = 2
print(f"Proving (kC)ᵀ = kCᵀ with k = {k}")
print("Left side (kC)ᵀ =")
print((k * C).T)
print("\nRight side kCᵀ =")
print(k * C.T)
print("\n")

# 21. Double Transpose
print("21. Double Transpose")
print("-" * 50)
print("Proving (Cᵀ)ᵀ = C")
print("Original matrix C:")
print(C)
print("\nDouble transpose (Cᵀ)ᵀ:")
print(C.T.T)
print("\n")

# 22. Matrix Multiplication Transpose
print("22. Matrix Multiplication Transpose")
print("-" * 50)
print("Proving (CD)ᵀ = DᵀCᵀ")
print("Left side (CD)ᵀ =")
print(np.dot(C, D).T)
print("\nRight side DᵀCᵀ =")
print(np.dot(D.T, C.T))
print("\n")

## 23 Matrix dimentions 
#Below code generate several tensors with different dimensions
b1 = np.random.randint(20, size=6)
b2 = np.random.randint(20, size=(3,4)) 
b3 = np.random.randint(20, size=(2,4,6))
print("23. Matrix dimensions")
print(b2)
print(b3)
print("b2 dims ", b2.ndim) 
print("b3 shape ", b3.shape) 
print("b2 size ", b2.size) 
print("data type of b3 ", b3.dtype)
print("\n")

## 24 Indexing operations or slicing operations
# knowing how to index a numpy array is very important.
# here is example of how to extract values of index.

## indexing
a = np.array([1, 3, 2, 5] , dtype='float32' ) 

print(a)
print("first ", a[0])
print("third ", a[2])
print("last ", a[-1]) 
print("before last ", a[-2])
print("\n")


## 25 indexing
a = np.array([
              [1, 2, 3, 4], 
              [5, 6, 7, 8],
              [9, 10, 11, 12]
            ] )
            
print(a)
print("first ", a[0,0])
print("last ", a[2, -1]) 
print("\n")

# 26 slicing 
## One important concept when dealing with numpy arrays or tensors is slicing.
## slicing helps us extract slices of data from a matrix like extracting 2 middle column vectors in  a matrix.

x = np.arange(15)
print(x)
print("first 4 elemets ", x[:4])
print("all after 3 ", x[3:])
print("even indeces ", x[::2] )
print("uneven indeces ", x[1::2]) ## starts at 1
print("reverse ", x[::-1] ) ## step value is negative starts at last element