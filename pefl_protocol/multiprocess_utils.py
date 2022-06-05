from multiprocessing import Pool

from pefl_protocol.configs import Configs


def number_add(numbers):
    return numbers[0] + numbers[1]


def number_sub(numbers):
    return numbers[0] - numbers[1]


def number_mul(numbers):
    return numbers[0] * numbers[1]


def arr_operation(arr1: [], arr2: [], op) -> []:
    arr3 = list(zip(arr1, arr2))
    with Pool(Configs.PROCESS_COUNT) as pool:
        ret = pool.map(op, arr3)

    return ret


def arr_add(arr1: [], arr2: []) -> []:
    return arr_operation(arr1, arr2, number_add)


def arr_sub(arr1: [], arr2: []) -> []:
    return arr_operation(arr1, arr2, number_sub)


def arr_mul(arr1: [], arr2: []) -> []:
    return arr_operation(arr1, arr2, number_mul)

