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


# 21 Try concatenation with matrices. Here we will need to use axis 0 or 1 to indicate 
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
print("\n")

# 22 Another approach to concatenate matrices is using hstack and vstack.

a =   np.array(   [1, 2, 3, 4]  )
b =   np.array(   [5, 6, 7, 8]  )
c =   np.array(   [9, 10, 11]  )

print( np.vstack((a,b)) )
print( np.hstack((a,b)) )
print("\n")

#23 Math operations using numpy
x = np.array( [1, 2, 3, 4] )
print(x + 10)
print(x * 10)
print(x / 2)
print(-x)
print(x ** 3 )
print(np.log(x))
print(np.log2(x) )
print(np.log10(x))
print("\n")


# 24 Implement f(x) = x³ - 3x² + 7 using math operators
# Create array of x values for plotting
x = np.linspace(-5, 5, 100)  # Creates 100 points between -5 and 5

# Calculate f(x) = x³ - 3x² + 7
y = np.power(x, 3) - 3*np.power(x, 2) + 7

# Print some sample points
print("Sample points (x, f(x)):")
for i in range(0, len(x), 20):  # Print every 20th point
    print(f"x = {x[i]:.2f}, f(x) = {y[i]:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.grid(True)
plt.title('f(x) = x³ - 3x² + 7')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.show()
print("\n")

# 25 Trigonometric functions like sines and coses are available in Numpy
# as can be seen here. Interestingly, sines and coses are used 
# in transformation models to capture the sequence of words in sentences.

a = np.array( [0. ,        0.44444444, 0.88888889, 1.33333333, 1.77777778, 2.22222222,
                2.66666667, 3.11111111, 3.55555556, 4.        ] )

print( a )

print( np.sin(a) )

# 26 The following code allows you to plot sine and cosine waves
# Change the number values of samples generated and note the change in the produced image.

position = torch.arange(0, 90, dtype=torch.float) ##.unsqueeze(1)

res = torch.sin(  position )
res2 = torch.cos( position )

plt.plot  (position, res,  label = "sine", color='blue') 
plt.plot(  position, res2, label = "cos", color='red') 
plt.legend() 
plt.grid(True)  # Added grid for better visualization
plt.title('Sine and Cosine Waves')
plt.xlabel('Position')
plt.ylabel('Value')
plt.show()

#27 numpy aggregate functions
#aggregates in numpy are ways in which you can perform an operation 
# and reduce the result 
#much like torch.reduce_sum() in pytorch. Below are aggregte functions 

x = np.array( [1, 2, 3, 4, 5] )
print(x)  
print(np.add.reduce(x) )
print(np.multiply.reduce(x) )
print(np.sum(x))
print(np.min(x))
print(np.max(x) )
print("\n")


#28 numpy min and max functions
#sometimes it is useful to extract minimum and maximum values across dimentions.
# we can do that using np.min np.sum etc.
m = np.array(
                                  [[ 1,  2,  3,  4],
                                   [ 5, 6,  7,  8],
                                   [ 9, 10, 11, 12],
                                   [13, 14, 15, 16]]
)
print(m)
print(m.sum())
print(np.min(m, axis=0))
print(np.min(m, axis=1))
print(np.min(m, axis=-1))
print("\n")

#Broadcasting  is one of the extremely useful concept in numpy and pytorch.
#Broadcasting allows you to perform an operation element wise between matrices 
# and vectors when all we want is a smaller numpy array to be repeated.

#Transformers rely heaviliy on broadcasting. 

m = np.array(
    [[1., 1., 1.],
     [1., 1., 1.],
     [1., 1., 1.]]
)

print(m)

a = np.array(
    [0, 1, 2]
)

print(a)
print(m+a)

#Example broacasting
a = a[:, np.newaxis]
print(a)
print(m+a)

#Example broacasting
v1 = np.array( [1, 1, 1] )
print(v1)

v2 =  np.array( [[0],
                 [1],
                 [2]]
              )
print(v2)

print(v1+v2)

print(v1*v2)
print(m)

zz = v1*v2

print(zz)

print(m*zz)

#Example broacasting
v1 = np.array(  [1, 1, 1],
             
             )
print(v1)

v1 = v1[ :, np.newaxis]

print(v1.shape)

v2 = np.array( [0, 1, 2])
print( v2.shape )

v2 = v2[np.newaxis, :]
print( v2.shape )

print(v1*v2)

print(v1+v2)


## Example broadcasting 
#create a matrix m of size 150X4 with random data. Assume that you also have a vector
#mean of size 1X4. perform a broadcast opperation that would let you scale the data 
#with following equation
#m_scaled = m - mean


# Create a 150x4 matrix with random data using numpy
m = np.random.randn(150, 4)

# Create a 1x4 mean vector (calculated as mean along columns)
mean = np.mean(m, axis=0)  # This gives us a vector of size 1x4
# Alternative: create custom mean vector
# mean = np.array([1, 2, 3, 4])

# Print shapes to understand the broadcasting
print("Matrix shape:", m.shape)      # (150, 4)
print("Mean shape:", mean.shape)     # (4,)

# Perform broadcasting subtraction
m_scaled = m - mean

# Verify the operation
print("\nOriginal matrix first row:", m[0])
print("Mean vector:", mean)
print("Scaled matrix first row:", m_scaled[0])

# Verify shapes remain correct
print("\nScaled matrix shape:", m_scaled.shape)  # Should still be (150, 4)


# Create a 150x4 matrix with random data using pytorch
m = torch.randn(150, 4)

# Create mean vector
mean = torch.mean(m, dim=0)  # Calculate mean along columns

# Perform broadcasting subtraction
m_scaled = m - mean

print("Matrix shape:", m.shape)
print("Mean shape:", mean.shape)
print("Scaled matrix shape:", m_scaled.shape)