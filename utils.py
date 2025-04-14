import numpy as np
import os
import plotly.graph_objs as go
import plotly.io as pio


def plotting(X, U, c):
    """
    Plot the data points and the cone.
    Args:
        X (numpy.array): Data matrix.
        U (numpy.array): Cone vertex rays matrix.
        c (int): Plot identifier for filename.
    """
    # Prepare data for Plotly
    points = go.Scatter3d(
        x=X[0, :],
        y=X[1, :],
        z=X[2, :],
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Points from X'
    )
    vectors = []
    for i in range(U.shape[1]):
        vectors.append(
            go.Scatter3d(
                x=[0, U[0, i]],
                y=[0, U[1, i]],
                z=[0, U[2, i]],
                mode='lines',
                line=dict(width=5, color='red'),
                name=f'Vector {i + 1} from U'
            )
        )
    # Define the vertices of the shaded area between all vectors
    vertices_x = [0] + list(U[0, :])
    vertices_y = [0] + list(U[1, :])
    vertices_z = [0] + list(U[2, :])
    # Create triangles between the origin and adjacent vectors
    triangles_i = []
    triangles_j = []
    triangles_k = []
    for i in range(1, U.shape[1]):
        # Create triangles with the origin and two consecutive vectors
        triangles_i.append(0)  # Origin index (0)
        triangles_j.append(i)  # Current vector index
        triangles_k.append(i + 1)  # Next vector index
    # Close the surface by connecting the last vector back to the first one
    triangles_i.append(0)
    triangles_j.append(U.shape[1])
    triangles_k.append(1)
    # Create a mesh for the shaded region
    shaded_region = go.Mesh3d(
        x=vertices_x,
        y=vertices_y,
        z=vertices_z,
        i=triangles_i,
        j=triangles_j,
        k=triangles_k,
        color='lightblue',
        opacity=0.5,
        name='Shaded Region'
    )
    # Add the mesh to the list of plots
    vectors.append(shaded_region)
    # Combine the points and vectors
    data = [points] + vectors
    # Define layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    # Create figure and plot
    fig = go.Figure(data=data, layout=layout)

    # Make sure the plots directory exists
    os.makedirs('plots', exist_ok=True)
    pio.write_html(fig, f'plots/{c}.html')


def generate_synthetic_data(m=3, n=100, r=2, noise_level=0.01):
    """Generate synthetic data for NMF."""
    # Create a random nonnegative basis
    U_true = np.abs(np.random.rand(m, r))

    # Create random nonnegative coefficients
    V_true = np.abs(np.random.rand(r, n))

    # Generate data
    X = np.dot(U_true, V_true)

    # Add small noise
    X += noise_level * np.abs(np.random.rand(m, n))

    return X, U_true, V_true