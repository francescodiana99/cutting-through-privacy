import numpy as np 

import numpy as np

def rref(B, tol=1e-15, debug=False):
  A = B.copy()
  rows, cols = A.shape
  r = 0
  pivots_pos = []
  row_exchanges = np.arange(rows)
  for c in range(cols):
    if debug: 
        print(f"Now at row {r} and col {c} with matrix:{A}")

    ## Find the pivot row:
    pivot = np.argmax (np.abs (A[r:rows,c])) + r
    m = np.abs(A[pivot, c])
    if debug: 
        print(f"Found pivot {m} in row {pivot}")
    if m <= tol:
      ## Skip column c, making sure the approximately zero terms are
      ## actually zero.
      A[r:rows, c] = np.zeros(rows-r)
      if debug: 
        print (f"All elements at and below ({r},{c}) are zero.. moving on..")
    else:
      ## keep track of bound variables
      pivots_pos.append((r,c))

      if pivot != r:
        ## Swap current row and pivot row
        A[[pivot, r], c:cols] = A[[r, pivot], c:cols]
        row_exchanges[[pivot,r]] = row_exchanges[[r,pivot]]
        
        if debug: 
            print (f"Swap row {r} with row {pivot}. \n Now:{A}")

      ## Normalize pivot row
      A[r, c:cols] = A[r, c:cols] / A[r, c];

      ## Eliminate the current column
      v = A[r, c:cols]
      ## Above (before row r):
      if r > 0:
        ridx_above = np.arange(r)
        A[ridx_above, c:cols] = A[ridx_above, c:cols] - np.outer(v, A[ridx_above, c]).T
        if debug: 
            print ("Elimination above performed:")
            print(A)
      ## Below (after row r):
      if r < rows-1:
        ridx_below = np.arange(r+1,rows)
        A[ridx_below, c:cols] = A[ridx_below, c:cols] - np.outer(v, A[ridx_below, c]).T
        if debug: 
            print("Elimination below performed:")
            print(A)
      r += 1
    ## Check if done
    if r == rows:
      break;
  return (A, pivots_pos, row_exchanges)



def generate_large_magnitude_diff_matrix(m, n, c, min_exp=-15, max_exp=15):
    """
    Generates a matrix of shape (m, n) with values that have a large difference in magnitude.
    
    Parameters:
    m (int): Number of rows.
    n (int): Number of columns.
        c (int): Number of rows to be replaced by linear combinations.
    min_exp (int): Minimum exponent for the magnitude (default -10, e.g. 1e-10).
    max_exp (int): Maximum exponent for the magnitude (default 10, e.g. 1e+10).
    
    Returns:
    np.ndarray: A matrix of shape (m, n) with values ranging from 10^min_exp to 10^max_exp.
    """
    
    # Generate random exponents between min_exp and max_exp
    exponents = np.random.uniform(min_exp, max_exp, size=(m, n))
    
    # Generate random signs (-1 or 1)
    signs = np.random.choice([-1, 1], size=(m, n))
    
    # Compute the matrix entries as random values with large magnitude differences
    matrix = signs * 10 ** exponents
    
    # Step 2: Replace c rows with linear combinations of other rows
    for _ in range(c):
        # Randomly select two distinct rows to form the linear combination
        row_indices = np.random.choice(m, size=2, replace=False)
        
        # Generate random coefficients for the linear combination
        
        # Create the new row as a linear combination of the two selected rows
        new_row =  matrix[row_indices[0], :]
        
        # Replace one of the rows with this new row
        matrix[row_indices[1], :] = new_row
        # print(matrix)
    return matrix



def test_rref():
    for _ in range(100):
        A = generate_large_magnitude_diff_matrix(512, 512, 100)
        rref_A, pivots, row_exchanges = rref(A)

        if len(pivots) != np.linalg.matrix_rank(A):
            print("Warning")
            print("Rank of A is: ", len(pivots))
            print(f"Rank with matrix_rank is: {np.linalg.matrix_rank(A)}")
        else:
            print("Success")
            print("Rank of A is: ", len(pivots))
            print(f"Rank with matrix_rank is: {np.linalg.matrix_rank(A)}")
            
    


test_rref()