# Batch generator
import numpy as np


def get_batches(X, Y, batch_size=100, shuffle = True):
    """
    Yielding generator of batches (X,Y) test data at `batch_size` per call
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], Y[batch_idx]
        
# Visualize Matrix

# Reusable heatmap creation function
def init_heatmap(ax, matrix, cmap='viridis'):
    matrix = np.array(matrix)
    rows, cols = matrix.shape

    cax = ax.imshow(matrix, cmap=cmap, origin='upper')
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")
    ax.set_title("Matrix Heatmap")

    # Create annotations
    annotations = []
    for i in range(rows):
        row_texts = []
        for j in range(cols):
            txt = ax.text(j, i, f"{matrix[i, j]:.2f}",
                          ha='center', va='center',
                          color='white' if matrix[i, j] > matrix.max()/2 else 'black')
            row_texts.append(txt)
        annotations.append(row_texts)

    annotations_flat = [txt for row in annotations for txt in row]

    return cax, annotations_flat

def update_heatmap(matrix, cax, annotations):
    matrix = np.array(matrix)
    cax.set_data(matrix)
    rows, cols = matrix.shape

    # Update text annotations
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            annotations[idx].set_text(f"{matrix[i, j]:.2f}")
            annotations[idx].set_color('white' if matrix[i, j] > matrix.max()/2 else 'black')

    return [cax] + annotations