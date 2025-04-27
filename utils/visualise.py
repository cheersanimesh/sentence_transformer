import matplotlib.pyplot as plt
import numpy as np

def plot_projection(proj, title, file_dest):
    plt.figure()
    xs, ys = proj[:,0], proj[:,1]
    plt.scatter(xs, ys)    # default colors/styles
    for i, sent in enumerate(sentences):
        plt.annotate(f"{sentences[i]}", (xs[i], ys[i]))
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
	plt.savfig(file_dest)
    plt.show()