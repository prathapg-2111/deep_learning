import numpy as np
import matplotlib.pyplot as plt 
import torch


#1 Slicing a sub matrix from a larger matrix can be done as follows
## slicing here from a 3X4 matrix we get 2X2 matrix after slicing.
## when you have i and a j index, you use something like , for seperation
# below slicing extracts rows from 0 to 2 with 2 not included and 
# column 0 to 2 with 2 not included.
a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]] ) 
print(a)
print(a[:2,:2])
print("\n")

# 2 slicing last value
# whenever we want the last value of the matrix, vector or tensor, we can just use the -1 index.
a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]] ) 
## slicing with following index scheme we get a 2X3 matrix but not including last column and last row.
a[:-1,:-1]
# another example of slicing with -1 index: we get a 2X2 matrix.    
# the rank of the tensor, the dimensions 1D 2D 3D i
a[:-1,:-2]

#3 shifting
#sometimes especially with transformers in language translation
# we want to shift a sentence or a line of data to the left or right to better align or disalign 
# with another line or sentence 

# Here we can just combine slicing with -1 last index to create shifting.
# given an input [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14]
# after shifting we get
# shift right [0 1 2 3 4 5 6 7 8 9 10 11 12 13]
# shift left[1 2 3 4 5 6 7 8 9 10 11 12 13 14]

# Basic shifting using array slicing
# Circular shifting using np.roll()
# Shifting with zero padding
# Shifting in 2D arrays (useful for image processing)
# Shifting with custom padding values

# Key points about these shifting operations:

# Simple slicing (arr[1:] or arr[:-1]) reduces array length by 1
# np.roll() maintains array length but wraps elements around
# Padding allows maintaining array length by filling with specified values
# These operations are particularly useful in:

# Natural Language Processing (sentence alignment)
# Signal Processing (time series analysis)
# Image Processing (feature detection)
# Sequence Analysis (pattern matching)


#4 Create a sample array
arr = np.arange(15)  # [0 1 2 3 4 5 6 7 8 9 10 11 12 13 14]
print("Original array:")
print(arr)
print("\n")

# 5. Basic Shifting using slicing
print("1. Basic Shifting using slicing")
print("-" * 50)
# Shift right (lose last element)
shift_right = arr[:-1]
print("Shift right (using slicing):")
print(shift_right)

# 6 Shift left (lose first element)
shift_left = arr[1:]
print("\nShift left (using slicing):")
print(shift_left)
print("\n")

# 7. Using np.roll() - Circular shifting
print("2. Circular Shifting using np.roll()")
print("-" * 50)
# Shift right with roll (wraps around)
roll_right = np.roll(arr, 1)
print("Roll right (circular):")
print(roll_right)

#8 Shift left with roll (wraps around)
roll_left = np.roll(arr, -1)
print("\nRoll left (circular):")
print(roll_left)
print("\n")

# 9. Padding with zeros
print("3. Shifting with Zero Padding")
print("-" * 50)
# Shift right with zero padding
pad_right = np.zeros_like(arr)
pad_right[1:] = arr[:-1]
print("Shift right with zero padding:")
print(pad_right)

# 10Shift left with zero padding
pad_left = np.zeros_like(arr)
pad_left[:-1] = arr[1:]
print("\nShift left with zero padding:")
print(pad_left)
print("\n")

# 11. 2D Array Shifting Example (useful for image processing)
print("4. 2D Array Shifting")
print("-" * 50)
arr_2d = np.arange(16).reshape(4, 4)
print("Original 2D array:")
print(arr_2d)
print("\n")

# 12 Shift columns right
shift_cols_right = arr_2d[:, :-1]
print("\nShift columns right:")
print(shift_cols_right)

# 13 Shift rows down
shift_rows_down = arr_2d[:-1, :]
print("\nShift rows down:")
print(shift_rows_down)
print("\n")

# 14. Shifting with custom padding value
print("5. Shifting with Custom Padding")
print("-" * 50)
pad_value = -1
# Shift right with custom padding
custom_right = np.full_like(arr, pad_value)
custom_right[1:] = arr[:-1]
print("Shift right with custom padding (-1):")
print(custom_right)

# 15 Shift left with custom padding
custom_left = np.full_like(arr, pad_value)
custom_left[:-1] = arr[1:]
print("\nShift left with custom padding (-1):")
print(custom_left)


## 16 slicing column vectors
## try to slice the second column. the input and the result are below
# [[1 2 3 4]
#  [5 6 7 8]
#  [9 10 11 12]
# ]

# output
# [2 6 10]

#Create the matrix
matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

# Slice the second column (index 1)
second_column = matrix[:, 1]  # Gets [2, 6, 10]

print("Original matrix:")
print(matrix)
print("\nSecond column:")
print(second_column)
print("\n")
## slicing the first 2 rows and the 2nd column
first_two_rows = matrix[:2,1]
print("Original matrix:")
print(matrix)
print("\nFirst 2 rows of the Second column:")
print(first_two_rows)
print("\n")
# extract all rows of the matrix
X = matrix[:,:3]
print(X)
print("\n")
X.shape
print("\n")

#extract last column of the matrix using transpose if will become column vector
Y = matrix[:,3]
print(Y)
print("\n")

# extract 1st row of the matrix
matrix[1, :]

#slicing in numpy do not copy to a new array but instead it still modifies  the original array.
# To copy and create a new matrix, you may want to use .copy()

#After slicing the matrix of size 3X4 we assign a new value  
new_matrix = matrix[:2, :2]

print(new_matrix)
print("\n")

new_matrix[0,0] = 42

print("new matrix")
print(new_matrix)
print("\n")
print("orignal matrix")
print(matrix)
print("\n")

#here the value at index 0,0 in original matrix as well got replaced with 42

# 17 once you start getting in to deep learning algorithms like CNNs
# we will use reshape operations.
# here is how you can reshape  from a vector of size[1,10] to a matrix of size 3X3 as can be seen below
a = np.array( [1, 2, 3, 4, 5, 6, 7, 8, 9] )
print("original matrix of size 1,10`")
print(a)
b = np.reshape(a, (3, 3)) 
print("reshared matrix with 3X3")
print(b)
print("\n")

## 18 Besides using reshape operation,  we can sometimes use np.newaxis.
# the np.newaxis function is critical wehn we wish to make a row vector in to a column vector
# the np.newaxis function is extensively used in numpy and pytorch for broadcasting operations.

a = np.array( [1, 2, 3, 4, 5] )

print(a.shape)
print("matrix a after reshaping using reshape")
b = np.reshape(a, (1, 5)) 
print(b.shape)
print("\n")
print(b)
print("\n")

print("matrix a after reshaping using newaxis on row vector ")
a = np.array( [5, 5, 5, 5, 5] )
print(a.shape)
a = a[ np.newaxis, :]   
print(a.shape)
print("\n")

print("matrix a after reshaping using newaxis to change row vector to column vector ")
a = np.array( [5, 5, 5, 5, 5] )
a=a[:, np.newaxis]
print( a )
print( a.shape)
print("\n")

#Reshape a matrix(m) of size 3X4 into  a matrix(m) of size[3,1,1,4] using 
# m[:,np.newaxis,np.newaxis, :]

# 19 Create the matrix "m" and reshape it. the values do not matter, only the dimensions, use
#print(n.shape)
#to view the results

# Create a 3x4 matrix
m = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

print("Original matrix shape:")
print(m.shape)  # Will show (3, 4)
print("\nOriginal matrix:")
print(m)
print("\n")

# Reshape using np.newaxis
n = m[:, np.newaxis, np.newaxis, :]

print("\nReshaped matrix shape:")
print(n.shape)  # Will show (3, 1, 1, 4)
print("\nReshaped matrix:")
print(n)
print("\n")


# 20 concatenation 
#Another important numpy array or tensor technique is concatenation.
#NLP approaches use concatenation extensively on auto regressive models.
#for example language translation.
# in the following code example, see how we can use concatenation with 
#numpy function np.concatenate.

a =   np.array(   [1, 2, 3, 4]  )
b =   np.array(   [5, 6, 7, 8]  )
c =   np.array(   [9, 10, 11]  )


res1 = np.concatenate((a, b), axis=0)

print( res1 )

res2 = np.concatenate((res1, c), axis=0)
print( res2 )
print("\n")


#Try concatenation with matrices. Here we will need to use axis 0 or 1 to indicate 
# on what dimension to concatenate.
#concatenate them on axis 0. 
#given the following 2 matrices
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print( matrix )
matrix1 = np.array([
    [10, 11, 12],
    [13, 14, 15],
    [16, 17, 18]
])
print( matrix1 )
print("\n")
#The result should look like this
res1 = np.concatenate((matrix, matrix1), axis=0)

print( res1 )

res2 = np.concatenate((matrix, matrix1), axis=1)

print( res2 )