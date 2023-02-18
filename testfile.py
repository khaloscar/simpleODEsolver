# For testing purposes
import numpy as np

A = np.array([[1, 2],[1, 2]])
B = [1]
C = np.array([5, 5])

print(A)
B = np.append(A, np.array([C]), axis=0)
A = np.append(A, np.array(np.transpose([C])), axis=1)

print(f'A is a {np.shape(A)} matrix: \n{A}\n\n' + 
      f'B is a {np.shape(B)} matrix: \n{B}]')

print(f'Transpose of A should give B:\n' + 
      f'{A.T} == {B}')

""" if type(A) == list:
    
    # Error
    # Do not want lists, raise error for requirements or proposed solution (i.e specify input argument...)
if A.ndim != 1:
    break
    # Error
    # Require vector to be of length (N,) (one dimensional...)
    # but what happens if the vector is nested with a metrix or something...
    # seems like array.ndim handles it perfectly... """