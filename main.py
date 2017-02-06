'''
A function that computes the Barnes hut t-SNE approximation of the Mnist dataset.
The computation is done in the memory efficient c++ code in this folder.
The positions of the projected samples are returned for each iteration, which enables the progress of the approach to be displayed as a movie with moviepy
'''

import numpy as np
from sklearn.datasets import load_digits

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy 

from bhtsne import run_bh_tsne

digits = load_digits()
print("digits shape", digits.data.shape)

# We first reorder the data points according to the handwritten numbers.
X = np.vstack([digits.data[digits.target==i]
               for i in range(10)])
y = np.hstack([digits.target[digits.target==i]
               for i in range(10)])

#Set the perplexity
perplexity = 50
#Set the max iterations
max_iter = 2000

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


print("Clusterring with t-SNE")
'''Perform Barnes-hut t-SNE approximation on the encoded inputs
    first input is an NxD array, where N is the number of samples
    no_dims => Number of dimensions to reduce the data to.
    initial_dims => number of principle components to extract with PCA
    perplexity => 2^(shannon entropy). A fair dice with k sides has a perplexity of k.

    This is much faster and has a similar accuracy to the standard t-SNE
    PCA can be used to reduce the dimensionality before performing the clustering
        This speeds up the computation of pairwise distances and supresses some of the noise
'''
#Call the python wrapper for the c++ implementation
clusters, positions = run_bh_tsne(X, no_dims=2, perplexity=perplexity,verbose=True,initial_dims=50, use_pca=False, max_iter=max_iter)

print("position shape", positions[0].shape)
X_iter = np.dstack(position.reshape(-1, 2) for position in positions)

f, ax, sc, txts = scatter(X_iter[..., -1], y)

def make_frame_mpl(t):
    i = int(t*40)
    x = X_iter[..., i]
    sc.set_offsets(x)
    for j, txt in zip(range(10), txts):
        xtext, ytext = np.median(x[y == j, :], axis=0)
        txt.set_x(xtext)
        txt.set_y(ytext)
    return mplfig_to_npimage(f)

animation = mpy.VideoClip(make_frame_mpl,duration=X_iter.shape[2]/40.)
animation.write_gif("Images/animation.gif", fps=20)

#Plot the results
f2, ax2, sc2, txts2 = scatter(clusters, y)

plt.show()