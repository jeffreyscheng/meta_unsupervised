from meta_framework import *


def partition(n):
    sum_to_n = []
    for v_i in range(n + 1):
        for w_ij in range(n + 1):
            for v_j in range(n + 1):
                if v_i + w_ij + v_j == n:
                    sum_to_n.append([v_i, w_ij, v_j])
    return sum_to_n


def get_poly_approx(deg):
    pass


for deg in range(10):
    print(len(partition(deg)))


def evaluate_func_diff(f1, f2, pts):
    diff = [f1(pt) - f2(pt) for pt in pts]
    return np.mean(diff)
