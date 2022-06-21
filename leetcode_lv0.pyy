import time
from typing import List


def timer(func, inputs, repeat=(1, 1)):
    """
    :param func:    函数
    :param inputs:  函数处理的数据
    :param repeat:  性能验证参数: (函数重复执行次数, 函数处理的数据规模扩大倍数)
    :return:        '函数返回值，执行时间'
    """
    t0, rtn = time.time(), None
    times, size = repeat
    if size > 1 and isinstance(inputs, (list, str)):
        inputs = [i for _ in range(size) for i in inputs]
    for i in range(times):
        rtn = func(*inputs) if isinstance(inputs, tuple) else func(inputs)

    return '\033[37m{:.2f}ms\033[0m\t{}'.format((time.time() - t0)*1000, rtn)


class No_0088_Merge:
    """
    两个按 非递减顺序 排列的整数数组 nums1 和 nums2，另有两个整数 m 和 n ，分别表示 nums1 和 nums2 中的元素数目
    合并 nums2 到 nums1 中，使合并后的数组同样按 非递减顺序 排列
    """
    @staticmethod
    def sol_two_points(nums1, nums2):
        # 没有完全按照题目中的m和n写代码, 另外指针也可以从后往前，没有本质区别。
        m, n = len(nums1), len(nums2)
        p0, p1, p2 = 0, 0, 0

        rtn = [0] * (m+n)      # 可以直接往某个数组末尾写入, 降低空间复杂度, 这里是避免修改原数组
        while p1 < m and p2 < n:
            if nums1[p1] <= nums2[p2]:
                rtn[p0] = nums1[p1]
                p1 += 1
            else:
                rtn[p0] = nums2[p2]
                p2 += 1
            p0 += 1
        if p1 < m:
            rtn[p0:] = nums1[p1:]
        elif p2 < n:
            rtn[p0:] = nums2[p2:]
        return rtn

    @classmethod
    def run(cls):
        nums1 = [1, 2, 3]
        nums2 = [2, 5, 6]
        repeat = (10, 100)    # 重复次数, 数据扩规模
        print('\033[7;34mQuest: {}\033[0m'.format(cls.__name__))

        print('双指针:', timer(cls.sol_two_points, (nums1, nums2), repeat))
        print('')
