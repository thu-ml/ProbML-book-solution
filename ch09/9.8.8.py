# ref: https://zhuanlan.zhihu.com/p/462014474?utm_medium=social&utm_oi=1240215594773229568&utm_id=0
import numpy as np
import struct
from matplotlib import pyplot as plt
from PIL import Image


def decode_idx3_ubyte(file_path):
    """
    说明：加载的file_path必须是解压文件，以.idx3-ubyte结尾的文件才能解码正确
    :param file_path:
    :return: images
    """
    bin_data = open(file_path, "rb").read()

    offset = 0
    magics, numimgs, rows, cols = struct.unpack_from('>IIII', bin_data, offset)
    # print("magic number %d, number of images: %d, size of images: %d * %d" % (magics, numimgs, rows, cols))

    img_size = rows * cols
    offset += struct.calcsize(">iiii")
    # print(offset)
    img_fmt = ">" + str(img_size) + "B"
    # print(img_fmt)

    images = np.empty((numimgs, rows, cols))
    for i in range(numimgs):
        if (i + 1) % 10000 == 0:
            print("%d images decoded" % (i + 1))
        images[i] = np.array(struct.unpack_from(img_fmt, bin_data, offset)).reshape((rows, cols))
        offset += struct.calcsize(img_fmt)
    return images


def decode_idx1_ubyte(file_path):
    bin_data = open(file_path, "rb").read()

    offset = 0
    header_fmt = ">II"
    magics, nums = struct.unpack_from(header_fmt, bin_data, offset)
    # print("magic: %d, number of labels: %d" % (magics, nums))

    offset += struct.calcsize(header_fmt)
    img_fmt = ">B"
    labels = np.empty(nums)
    for i in range(nums):
        if (i + 1) % 10000 == 0:
            print("%d labels decoded" % (i + 1))
        labels[i] = struct.unpack_from(img_fmt, bin_data, offset)[0]
        offset += struct.calcsize(img_fmt)
    return labels


def pca(x):
    N, p = x.shape
    # S = np.matmul(x.reshape(N, p, 1), x.reshape(N, 1, p)).mean(axis=0)
    S = np.zeros((p, p))
    batch_size = 1000
    for i in range(N // batch_size):
        batch = x[batch_size * i: min(batch_size * (i + 1), N)]
        if len(batch) == 0:
            break
        S += np.matmul(batch.reshape(-1, p, 1), batch.reshape(-1, 1, p)).sum(axis=0)
    S /= N
    v, u = np.linalg.eigh(S)
    u = u[:, np.argsort(-v)]
    v = v[np.argsort(-v)]
    return v, u


def reconstruct(image, u, d=1):
    image_recon = np.zeros_like(image)
    for i in range(d):
        ui = u[:, i]
        image_recon += np.dot(ui, image) * ui
    error = np.linalg.norm(image - image_recon)
    print(f"d={d}, error={error}")
    return image_recon


def main():
    images = decode_idx3_ubyte("train-images.idx3-ubyte")
    labels = decode_idx1_ubyte("train-labels.idx1-ubyte")
    label_3_indexes = np.where(labels == 3)[0]
    label_3_images = images[label_3_indexes].reshape(len(label_3_indexes), -1)
    mean_label_3_images = label_3_images.mean(axis=0, keepdims=True)
    _, u = pca(label_3_images - mean_label_3_images)
    for d in range(1, 10):
        image_recon = reconstruct(label_3_images[0] - mean_label_3_images[0], u, d) + mean_label_3_images[0]
        image_recon = image_recon.reshape(28, 28)
        Image.fromarray(image_recon).convert('L').save(f"9.8.8_{d}.png")


if __name__ == "__main__":
    main()
