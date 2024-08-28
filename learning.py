import multiprocessing
import numpy as np

def func1(matrix):
    # Example manipulation: matrix multiplication
    return np.dot(matrix, matrix.T)

def func2(matrix):
    # Example manipulation: element-wise addition
    return matrix + 2

def manipulate_matrix(matrix):
    result1 = func1(matrix)
    result2 = func2(result1)
    return result2

def process_list_of_matrices(matrices):
    # Apply manipulation to each matrix in the list
    processed_matrices = [manipulate_matrix(matrix) for matrix in matrices]
    
    # Combine the processed matrices (e.g., sum them)
    combined_matrix = np.sum(processed_matrices, axis=0)
    
    return combined_matrix

if __name__ == "__main__":
    # Create four lists of random matrices
    list_of_matrices = [
        [np.random.rand(1000, 1000) for _ in range(5)] for _ in range(4)
    ]

    # Set up a pool of workers
    with multiprocessing.Pool(processes=4) as pool:
        # Apply processing to each list of matrices in parallel
        combined_matrices = pool.map(process_list_of_matrices, list_of_matrices)

    # `combined_matrices` is a list of the combined matrices
    for i, combined_matrix in enumerate(combined_matrices):
        print(f"Combined Matrix {i+1} Shape:", combined_matrix.shape)
