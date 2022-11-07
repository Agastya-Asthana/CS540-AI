from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    yale_file = np.load(filename)
    yale_file = yale_file - np.mean(yale_file, axis=0)
    return yale_file


def get_covariance(dataset):
    transpose = np.transpose(dataset)
    covariance = np.dot(transpose, dataset)/(len(dataset)-1)
    return covariance


def get_eig(S, m):
    w, v = eigh(S, subset_by_index=[len(S)-m, len(S)-1])
    w = np.flip(np.diag(w))
    v = np.flip(v[::-1])
    return w, v


def get_eig_prop(S, prop):
    min_eigen = np.trace(S)
    min_eigen = min_eigen * prop
    w, v = eigh(S, subset_by_value=[min_eigen, np.inf])
    w = np.flip(np.diag(w))
    v = np.flip(v[::-1])
    return w, v


def project_image(image, U):
    alpha = np.dot(np.transpose(U), image)
    xpca = np.dot(alpha, np.transpose(U))
    return xpca


def display_image(orig, proj):
    orig = np.reshape(orig, (32, 32))
    proj = np.reshape(proj, (32, 32))
    figure, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("Original")
    ax2.set_title("Projection")
    orig = np.rot90(orig, k=1, axes=(1, 0))
    proj = np.rot90(proj, k=1, axes=(1, 0))
    bar1 = ax1.imshow(orig)
    bar2 = ax2.imshow(proj)
    figure.colorbar(bar1, ax=ax1)
    figure.colorbar(bar2, ax=ax2)
    plt.show()


x = load_and_center_dataset("YaleB_32x32.npy")
print(len(x))
print(len(x[0]))
print(np.average(x))
S = get_covariance(x)
Lambda, U = get_eig(S, 2)
print(Lambda)
print(U)
Lambda, U = get_eig_prop(S, 0.07)
print(Lambda)
print(U)
projection = project_image(x[0], U)
print(projection)
display_image(x[0], projection)
