import numpy as np
import skimage


def check_shape_equal(*mats):
    mat0 = mats[0]
    if not all([m.shape == mat0.shape for m in mats[1:]]):
        raise ValueError('Input images must have the same dimensions.')
    return


def cal_mutual_info(arr1, arr2, bins=256, ranges=((0, 255), (0, 255))):
    arr1_flat = np.ravel(arr1)
    arr2_flat = np.ravel(arr2)

    assert arr1.shape == arr2.shape, f"{arr1.shape = } and {arr2.shape = }"

    joint_hist, x_edges, y_edges = np.histogram2d(
        arr1_flat, arr2_flat, bins=bins, range=ranges
    )

    joint_prob = joint_hist / joint_hist.sum()
    nzx, nzy = np.nonzero(joint_prob)
    px = joint_prob.sum(axis=1).take(nzx)
    py = joint_prob.sum(axis=0).take(nzy)
    joint_prob = joint_prob[nzx, nzy]
    tmp = px * py
    tmp = -np.log(tmp) + np.log(joint_prob)
    tmp = joint_prob * tmp
    mi = tmp.sum()

    return mi


def cal_mse(arr1, arr2):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    assert arr1.shape == arr2.shape

    return ((arr1 - arr2) ** 2).sum() / np.prod(arr1.shape)


def cal_ncc(arr1, arr2):
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    # assert check_shape_equal(arr1, arr2), f"{arr1.shape = } and {arr2.shape = }"

    cc = np.sum((arr1 - np.mean(arr1)) * (arr2 - np.mean(arr2)))
    sta_dev = np.sqrt(np.sum(arr1**2) * np.sum(arr2**2)) + np.finfo(float).eps

    return cc / sta_dev

def cal_ssim(*args, **kwargs):
    return skimage.metrics.structural_similarity(*args, **kwargs)


def cal_psnr(*args, **kwargs):
    return skimage.metrics.peak_signal_noise_ratio(*args, **kwargs)


if __name__ == "__main__":
    print(cal_ncc(np.array([1, 2, 4]), np.array([1, 2, 4])))
