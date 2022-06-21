from typing import List
from leetcode_lv0 import timer


class No_0004_FindMedianSortedArrays:
    @staticmethod
    def sol_01_traversal(nums1: List[int], nums2: List[int]):
        # 合并数据, 排序
        all_nums = nums1 + nums2
        all_nums.sort()
        size = len(all_nums)
        if size < 2:
            return all_nums[0]
        elif size % 2 == 0:
            return (all_nums[int(size/2)-1] + all_nums[int(size/2)]) / 2
        else:
            return all_nums[int(size/2)]

    @staticmethod
    def sol_02_two_points(nums1: List[int], nums2: List[int]):
        """
        暴力解完全没有使用到数组有序的条件，可以参考88题合并有序数组, 时间复杂O(m+n)
        """
        # 没有完全按照题目中的m和n写代码, 另外指针也可以从后往前，没有本质区别。
        m, n = len(nums1), len(nums2)
        mid = (m+n) / 2
        p0, p1, p2 = 0, 0, 0

        all_nums = [0] * (m+n)      # 可以直接往某个数组末尾写入, 降低空间复杂度, 这里是避免修改原数组
        while (p1 < m and p2 < n) and p1+p2 <= mid+1:  # 只需要到一半就可以退出循环了
            if nums1[p1] <= nums2[p2]:
                all_nums[p0] = nums1[p1]
                p1 += 1
            else:
                all_nums[p0] = nums2[p2]
                p2 += 1
            p0 += 1
        if p1+p2 <= mid+1:
            # 至少已经遍历完一个数组还没有达到一半总长度
            if p1 < m:
                all_nums[p0:] = nums1[p1:]
            elif p2 < n:
                all_nums[p0:] = nums2[p2:]

        if mid < 1:
            return all_nums[0]
        elif mid % 1 == 0:
            return (all_nums[int(mid-1)] + all_nums[int(mid)]) / 2
        else:
            return all_nums[int(mid)]

    @staticmethod
    def sol_03_split_nums(nums1: List[int], nums2: List[int]):
        """
        划分数组：属于特定问题的特定解法，需要理解中位数的推导
        从位置i将A划分为数量近似相等的两部分 A[0], A[1], ..., A[i-1]  |  A[i], A[i+1], ..., A[m-1]，称为A左集和A右集
        从位置j将B划分为数量近似相等的两部分 B[0], B[1], ..., B[j-1]  |  B[j], B[j+1], ..., B[n-1]，称为B左集和B右集

        分别合并左集和右集, 两种情况：
        1、m+n为奇数：i+j = m-i + n-j + 1，∴ i+j = (m+n+1)/2，∴  j = (m+n+1)/2 - i，让
        2、m+n为偶数 i+j = m-i + n-j，∴ i+j = (m+n)/2 ，整数除法可以补上1，值不变，这样和上面的式子就统一表达，可以忽略奇偶性

        中位数的第二个性质，左集任意元素<=右集任意元素：
        1、判断：A左集的最大元素 <= B右集合的最小值
        2、判断：B左集的最大元素 <= A右集合的最小值
        如果不满足的话，也要调整分割线

        为减少处理分支，规定较短的数组为A，较长的数组为B，方面后续编码
        前一部分的最小值，和后一部分的最大值，就对应了合并后的中位数
        """
        m, n = len(nums1), len(nums2)
        if m > n:
            nums1, nums2 = nums2, nums1  # 交换数组，始终保持短数组在前,这样在[0, m]区间找分隔线
            m, n = n, m

        tol_left = int((m + (n-m+1)) / 2)  # 分隔线左侧要满足的元素个数， 这样写可以防止整形溢出

        # 定义num1的分隔线划在下标i元素的左侧, 所以下标i即分隔线左侧的元素个数，同理nums2的分割线为j
        # 结果要满足 nums1[i-1] <= nums2[j] and nums1[j-1] <= nums1[i]
        left, right = 0, m
        while left < right:     # 退出循环必然left == right
            i = int(left + (right - left + 1)/2)     # 二分查找
            j = tol_left - i

            if nums1[i-1] > nums2[j]:
                # 左集最大大于右集最小, 分隔线左移
                right = i - 1
            else:
                left = i

        i = left    # 退出循环的时候找到满足nums1[i-1] <= nums2[j] and nums1[j-1] <= nums1[i]的位置
        j = tol_left-i

        # 分隔线左右的4个元素
        nums1_left_max = nums1[i-1] if i != 0 else float('-inf')  # 当i等于0时, 分隔线没有意义, 最小值保证比较结果
        nums1_right_min = nums1[i] if i != m else float('inf')   # 当i等于m时, 分隔线没有意义, 最大值保证比较结果
        nums2_left_max = nums2[j-1] if j != 0 else float('-inf')
        nums2_right_min = nums2[j] if j != m else float('inf')

        if (m+n) % 2 == 0:
            return (max(nums1_left_max, nums2_left_max) + min(nums1_right_min, nums2_right_min))/2
        else:
            return max(nums1_left_max, nums2_left_max)

    @classmethod
    def run(cls):
        nums1 = [1, 2]
        nums2 = [3, 4]
        repeat = (10, 1)    # 重复次数, 数据扩规模
        print('\033[7;34mQuest: {}\033[0m'.format(cls.__name__))

        print('暴力解:', timer(cls.sol_01_traversal, (nums1, nums2), repeat))
        print('双指针:', timer(cls.sol_02_two_points, (nums1, nums2), repeat))
        print('数学分割:', timer(cls.sol_03_split_nums, (nums1, nums2), repeat))
        print('')



class No_0032_LongestValidParentheses:
    """给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。"""
    @staticmethod
    def sol_01_traversal(s: str):
        """
        暴力解法: 从最长字符串开始递减，每次递减长度遍历子串匹配括号，当括号匹配成功时返回当前的长度
        """
        def valid_parentheses(sub_s):
            """括号匹配：遇到左进栈，遇右弹出。栈顶"""
            stack = []
            for c in sub_s:
                if c == '(':
                    stack.append(c)
                elif stack:     # 因为不存在别的字符, 省略stack[-1] == '('判断
                    stack.pop()
                else:
                    return False
            return stack == []

        size = len(s)
        if size < 2: return 0

        for i in range(size if size % 2 == 0 else size-1, 0, -2):   # 有效的括号序列必然是偶数
            for j in range(size-i+1):
                if valid_parentheses(s[j: j+i]):
                    return i
        return 0

    @staticmethod
    def sol_02_dp(s: str):
        """
        动态规格：状态转移方程（相邻阶段的递归关系），边界条件，一般需要一个与原数组相同大小的dp数组
        假设已经完成前一个字符的最长有效括号数识别，那么当前字符的有效括号数：
        1、前字符为')', 如果前一个字符的最长有效括号数之前的一个字符，即对称位置为'('，那么当前最长值+2，否则为0
           对称位置：i-dp[i-1]-1
        2、叠加之前的有效长度:  dp[i-dp[i-1]-2]
        """
        size = len(s)
        if size < 2:
            return 0

        dp = [0] * size
        for i in range(size):
            if s[i] == ')' and i-dp[i-1]-1 >= 0 and s[i-dp[i-1]-1] == '(':
                # 要对')'计数 and 对称位置为'(' and 判断下标有效
                dp[i] = dp[i-1] + 2 + dp[i-dp[i-1]-2]
        return max(dp)

    @staticmethod
    def sol_03_stack(s: str):
        """
        栈（下标）：保持栈底元素为当前已经遍历过的元素中「最后一个没有被匹配的右括号的下标」
        1、遇到'('入栈
        2、遇到')'出栈
            :如果栈为空，则按「最后一个没有被匹配的右括号的下标」入栈
            :如果栈不为空，当前下标减去栈顶元素就是「以该右括号为结尾的最长有效括号的长度」，栈顶元素可以视为一个新的开始点
        """
        stack = [-1]    # 哨兵
        size = len(s)
        max_len = 0
        for i in range(size):
            if s[i] == '(':
                stack.append(i)
            else:
                stack.pop()
                if stack:
                    max_len = max(max_len, i - stack[-1])
                    # 当栈内始终有元素时, 除了栈底元素, 其他元素必定是'('
                else:
                    stack.append(i)
                    # 有一个不成对的')'保持在栈底. 可以理解为这里多出了一个无效的符号, 从这个位置重新计算
        return max_len

    @staticmethod
    def sol_04_go_and_back(s: str):
        """
        前向遍历：当左括号个数和右括号相等时, 最大长度为对数*2， 当右括号数量超过左括号时计数归0； 反向相反
        :param s:
        :return:
        """
        size = len(s)
        left = right = max_len = 0
        # 前向遍历
        for c in s:
            if c == '(':
                left += 1
            else:
                right += 1
            if left == right:
                max_len = max(max_len, right * 2)
            elif right > left:
                left = right = 0
        # 反向遍历
        left = right = 0
        for i in range(size-1, -1, -1):
            if s[i] == ')':
                right += 1
            else:
                left += 1
            if left == right:
                max_len = max(max_len, left * 2)
            elif left > right:
                left = right = 0
        return max_len

    @classmethod
    def run(cls):
        s = ")((((((()())))))))()()))()()()()()"
        repeat = (10, 10000)    # 重复次数, 数据扩规模
        print('\033[7;34mQuest: {}\033[0m'.format(cls.__name__))

        # print('暴力解:', timer(cls.sol_01_traversal, s, repeat))
        print('动态规划:', timer(cls.sol_02_dp, s, repeat))
        print('栈:', timer(cls.sol_03_stack, s, repeat))
        print('左右遍历:', timer(cls.sol_04_go_and_back, s, repeat))
        print('')


class No_0042_TrapRain:
    """
    给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水（凹的地方接水）
    关键理解：
    对于数组中的每个元素，它两边最大高度的较小值减去当前高度的值，就是当前元素上方积水能到达的最大高度，
    如果最高就是它自己，那么这个逻辑的结果就是0
    """
    @staticmethod
    def sol_01_traversal(heights):
        """
        暴力解法1: 遍历每个元素，找到其左右的最大高度
        """
        ans = 0
        size = len(heights)
        for i in range(size):
            max_left, max_right = 0, 0
            for j in range(i, -1, -1):
                max_left = max(max_left, heights[j])
            for j in range(i, size):
                max_right = max(max_right, heights[j])
            ans += min(max_left, max_right) - heights[i]
        return ans

    @staticmethod
    def sol_02_traversal(heights):
        """
        暴力解法2: 按行求解，找出每一层的中间空格的个数进行累积
        """
        ans = 0
        size = len(heights)
        max_height = max(heights)
        for i in range(max_height):
            flag_start_accu = False
            j, temp = 0, 0  # 遍历索引，临时累积, 遇到右边界才计入ans
            while j < size:
                if heights[j] > i:
                    flag_start_accu = True  # 遇到第一个左边界时置为True
                    break
                j += 1
            for k in range(j, size):
                if flag_start_accu and heights[k] > i:  # 遇到右边界
                    ans += temp
                    temp = 0
                else:
                    temp += 1
        return ans

    @staticmethod
    def sol_03_two_arrays(heights):
        """
        双数组: 先遍历一次从左向右看的最大值，再遍历一次从右向左看的最大值，最后遍历一次每个元素的蓄水量
        """
        ans = 0
        size = len(heights)
        left_max = [heights[0] for _ in range(size)]
        for i in range(1, size):
            left_max[i] = max(heights[i], left_max[i-1])
        right_max = [heights[-1] for _ in range(size)]
        for i in range(size-2, -1, -1):
            right_max[i] = max(heights[i], right_max[i+1])

        for i in range(size):
            ans += min(left_max[i], right_max[i]) - heights[i]
        return ans

    @staticmethod
    def sol_04_two_points(heights):
        """
        双指针（空间复杂度低）: 双数组遍历了3次（左一次，右一次，计算一次），双指针把左右合一
        """
        ans = 0
        size = len(heights)
        i, j = 0, size-1    # 双指针从头尾开始
        i_left_max, j_right_max = heights[0], heights[-1]

        while i < j:    # 两个指针相遇时结束遍历
            if heights[i] < heights[j]:
                ans += i_left_max - heights[i]
                i += 1  # 左指针向右移动
                i_left_max = max(heights[i], i_left_max)

            else:
                ans += j_right_max - heights[j]
                j -= 1  # 右指针向左移动
                j_right_max = max(heights[j], j_right_max)
        return ans

    @staticmethod
    def sol_05_stack(heights):
        """
        积水只能在低洼处形成，使用单调栈递减存储左柱，当遇到大于左柱的右柱时，弹出计算
        """
        ans = 0
        size = len(heights)
        stack = []
        for i in range(size):
            while stack and heights[i] > heights[stack[-1]]:
                bottom_idx = stack.pop()   # 弹出值为当前坑底，栈顶为坑的左侧
                if not stack:
                    break  # 如果栈为空, 没有左侧柱挡水
                width = i - stack[-1] - 1
                depth = min(heights[i], heights[stack[-1]]) - heights[bottom_idx]
                ans += width * depth
            stack.append(i)
        return ans

    @classmethod
    def run(cls):
        heights = [4, 2, 0, 3, 2, 5]
        repeat = (100, 10)    # 重复次数, 数据扩规模
        print('\033[7;34mQuest: {}\033[0m'.format(cls.__name__))

        print('暴力解(列):', timer(cls.sol_01_traversal, heights, repeat))
        print('暴力解(行):', timer(cls.sol_02_traversal, heights, repeat))  # 对数据规模不敏感, 对数值大小敏感
        print('双数组:\t', timer(cls.sol_03_two_arrays, heights, repeat))
        print('双指针:\t', timer(cls.sol_04_two_points, heights, repeat))
        print('栈:\t\t', timer(cls.sol_05_stack, heights, repeat))
        print('')


class No_0084_LargestRectangleArea:
    """
    动态规划经典
    给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
    求在该柱状图中，能够勾勒出来的矩形的最大面积。
    """
    @staticmethod
    def sol_01_traversal(heights: List[int]):
        """
        暴力解法：遍历每个柱形，向左右扩展
        """
        bar_tol = len(heights)
        max_s = 0
        for i in range(bar_tol):
            width = 1
            j = i - 1
            while j >= 0:
                # 如果高于或等于当前柱形，矩形可以扩展
                if heights[j] >= heights[i]:
                    width += 1
                    j -= 1  # 向左
                else:
                    break
            j = i + 1
            while j <= bar_tol-1:
                if heights[j] >= heights[i]:
                    width += 1
                    j += 1  # 向右
                else:
                    break
            max_s = max(heights[i] * width, max_s)
        return max_s

    @staticmethod
    def sol_02_stack(heights: List[int]):
        """
        空间换时间思路：
        矩形向右扩展, 当右侧不低于左侧时, 面积不确定（有继续扩展的可能），但是当右侧低于左侧时，矩形就不能继续扩展了
        所以有：先得到靠右的面积，再得到靠左的面积（即后进先出，所以可以考虑使用栈进行辅助）
        不能确定矩形宽度时，入栈；当矩形高度小于栈顶元素时，出栈直到大于栈顶元素并计算出最大矩形面积
        :param heights:
        :return:
        """
        area = 0
        length = len(heights)
        if length == 0:
            return 0
        else:
            stack = []
            # 遍历
            for i in range(len(heights)):   # 遍历一次数组
                while stack and heights[stack[-1]] > heights[i]:
                    height = heights[stack.pop()]
                    while stack and heights[stack[-1]] == height:
                        # 跳过高度相同的柱形, 可以省略（因为回溯到中间某个柱形的面积一定小于最前一个）, 但是可以节省运算
                        stack.pop()
                    if stack:
                        width = i - stack[-1] - 1
                    else:
                        width = i
                    area = max(area, height * width)
                stack.append(i)
            # 清空栈
            while stack:
                height = heights[stack.pop()]
                while stack and heights[stack[-1]] == height:
                    stack.pop()
                    pass
                if stack:
                    width = length - stack[-1] - 1
                else:
                    width = length  # 扩展到边缘
                area = max(area, height * width)
            return area

    @staticmethod
    def sol_03_stack_with_sentinal(heights: List[int]):
        """
        空间换时间思路的代码优化：
        在数组两端增加高度为0的柱形，使“栈为空”和“完成遍历后栈中还有元素”具有统一形式，增加的柱形就称为"哨兵"
        头哨兵可以保证栈不为空，尾哨兵可以在遍历后必然空栈
        """
        area = 0
        length = len(heights)
        if length == 0:
            return 0
        else:
            # 哨兵入队
            new_heights = [0]
            new_heights.extend(heights)
            new_heights.append(0)
            heights = new_heights

            stack = [0]  # 先放入一个哨兵, 省略非空判断
            for i in range(1, len(heights)):   # 遍历一次数组
                while heights[stack[-1]] > heights[i]:
                    height = heights[stack.pop()]
                    while heights[stack[-1]] == height:
                        # 跳过高度相同的柱形
                        stack.pop()
                    width = i - stack[-1] - 1
                    area = max(area, height * width)
                stack.append(i)
            return area

    @classmethod
    def run(cls):
        heights = [2, 1, 5, 6, 2, 3]
        repeat = (100, 10)    # 重复次数, 数据扩规模
        print('\033[7;34mQuest: {}\033[0m'.format(cls.__name__))

        print('暴力解:\t', timer(cls.sol_01_traversal, heights, repeat))
        print('栈:\t\t', timer(cls.sol_02_stack, heights, repeat))
        print('栈(哨兵):', timer(cls.sol_03_stack_with_sentinal, heights, repeat))
        print('')


class No_0085_LargestRectangleArea:
    """
    给定一个仅包含 0 和 1 、大小为 rows x cols 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积
    """
    @staticmethod
    def sol_01_traversal(matrix: List[List[str]]):
        """
        暴力解法：
        遍历每个点，遍历这个点为右下角的所有矩形面积
        """
        row_num = len(matrix)
        col_num = len(matrix[0])

        width = [[0 for _i in range(col_num)] for _j in range(row_num)]
        area = 0

        for row in range(row_num):
            for col in range(col_num):
                if matrix[row][col] == "1":
                    if col == 0:
                        width[row][col] = 1
                    else:
                        # 使用空间换时间, 记录累积宽度
                        width[row][col] = width[row][col-1] + 1

                # 记录当前跨越的行中最小的宽度作为矩形的宽
                min_width = width[row][col]
                up_row = row        # 从当前行开始向上扩展
                while up_row >= 0:  # 序号0包括在内
                    height = row - up_row + 1
                    min_width = min(min_width, width[up_row][col])
                    if min_width == 0:
                        break

                    area = max(area, min_width * height)    # 更新最大面积
                    up_row -= 1
        return area

    @staticmethod
    def sol_02_call_stack(matrix: List[List[str]]):
        """
        空间换时间思路：结合84题
        把当前行当做柱形图，传入No84LargestRectangleArea的栈，逐行遍历
        """
        row_num = len(matrix)
        col_num = len(matrix[0])

        heights = [0] * col_num     # 空间换时间, 累计更新
        area = 0
        for row in range(row_num):
            # 更新每一列的柱形高度
            for col in range(col_num):
                if matrix[row][col] == '1':
                    heights[col] += 1
                else:
                    heights[col] = 0
            area = max(area, No_0084_LargestRectangleArea.sol_02_stack(heights))
        return area

    @staticmethod
    def sol_03_stack(matrix: List[List[str]]):
        """
        空间换时间思路： 把栈结合进来重构代码, 与上一方法的逻辑相同
        """
        # 把栈结合进来重构代码, 逻辑是一样的
        row_num = len(matrix)
        col_num = len(matrix[0])
        area = 0

        heights = [0 for _ in range(col_num+1)]  # 队尾哨兵, 保证所有元素出栈
        for row in range(row_num):
            stack = []
            for col in range(col_num+1):    # 哨兵要进入循环
                if col < col_num:           # 哨兵不用累积高度
                    if matrix[row][col] == '1':
                        heights[col] += 1
                    else:
                        heights[col] = 0

                # 栈机制不需要一次性提供所有高度，每更新一个高度就可以操作栈
                if (not stack) or heights[col] >= heights[stack[-1]]:
                    stack.append(col)
                else:
                    while stack and heights[col] < heights[stack[-1]]:
                        height = heights[stack.pop()]
                        width = col - (stack[-1] if stack else -1) - 1
                        # right = col       # 当前柱形右侧第一个比当前高度矮的位置
                        # left = stack[-1]  # 当前柱形左侧第一个比当前高度矮的位置
                        area = max(area, height * width)
                    stack.append(col)
        return area

    @staticmethod
    def sol_04_dp(matrix: List[List[str]]):
        """
        动态规划：
        思考上述代码在计算矩形宽度时right, left的意义
            # right = col       # 当前柱形右侧第一个比当前高度矮的位置
            # left = stack[-1]  # 当前柱形左侧第一个比当前高度矮的位置
        随矩阵层数向下扩展时，如果下层增加的值全部为1，那么矩形的left, right实际上不会变化
        如果矩阵下层出现了0，那么和上一层的0比较，新的left位置，应该更接为更接近当前柱子的0位置, right同理
        动态更新：每增加一层，更新一次left_less_min和right_less_min的列表
        """
        row_num = len(matrix)
        col_num = len(matrix[0])
        area = 0

        heights = [0 for _ in range(col_num)]
        # 全部初始化为最左和最右
        lefts, rights = [-1 for _ in range(col_num)], [col_num for _ in range(col_num)]

        for row in range(row_num):
            for col in range(col_num):
                if matrix[row][col] == '1':
                    heights[col] += 1
                else:
                    heights[col] = 0

            boundary = -1    # 记录上一次的0位置, 初始化-1
            for col in range(col_num):
                if matrix[row][col] == '1':
                    lefts[col] = max(boundary, lefts[col])
                else:
                    lefts[col] = -1
                    boundary = col

            # 右侧更新与左侧相反
            boundary = col_num      # 记录上一次的0位置, 初始化最大索引
            for col in range(col_num-1, -1, -1):
                if matrix[row][col] == '1':
                    rights[col] = min(boundary, rights[col])
                else:
                    rights[col] = col_num
                    boundary = col

            # 计算面积
            for col in range(col_num):
                area = max((rights[col] - lefts[col] -1) * heights[col], area)
        return area

    @classmethod
    def run(cls):
        matrix = [["1", "0", "1", "0", "0"],
                  ["1", "0", "1", "1", "1"],
                  ["1", "1", "1", "1", "1"],
                  ["1", "0", "0", "1", "0"]]
        repeat = (100, 10)    # 重复次数, 数据扩规模
        print('\033[7;34mQuest: {}\033[0m'.format(cls.__name__))

        print('暴力解:\t', timer(cls.sol_01_traversal, matrix, repeat))
        print('栈:\t\t', timer(cls.sol_02_call_stack, matrix, repeat))
        print('栈优化:\t', timer(cls.sol_03_stack, matrix, repeat))
        print('动态规划:', timer(cls.sol_04_dp, matrix, repeat))
        print('')


class No_0321_MaxNumber:
    """
    给定长度分别为m和n的两个数组，其元素由0-9构成，表示两个自然数各位上的数字。
    现在从这两个数组中选出 k (k <= m + n)个数字拼接成一个新的数，要求从同一个数组中取出的数字保持其在原数组中的相对顺序。
    求满足该条件的最大数。结果返回一个表示该最大数的长度为k的数组。

    关联-：
    No402_RemoveKDigits 解决一个数组的问题, 这里是两个数组

    思路：
    1、将k分解为从第一个数组选出k1个数，从第二个数组中选出k2个数，解法已经有了
    2、将k1个数和k2个数合并为最大值，就是顺序从两个数组中选择较大值
    3、遍历k分为k1和k2的组合
    """
    @staticmethod
    def sol_01_stack(nums1, nums2, k):
        def pick_max(nums, k):  # 选k个最大
            stack = []
            _k = len(nums) - k
            for n in nums:
                if _k and stack and stack[-1] < n:
                    stack.pop()
                    _k -= 1
                stack.append(n)
            return stack[:k]

        def merge_max(arr_a, arr_b):
            ans = []
            while arr_a or arr_b:
                larger = arr_a if arr_a > arr_b else arr_b  # 理解数组的比较规则
                ans.append(larger.pop(0))   # pop(0)效率低
            return ans
        # def merge_max(arr_a, arr_b):    # 针对pop(0)效率低改的指针操作
        #     ans = []
        #     pa, pb = 0, 0
        #     size_a, size_b = len(arr_a), len(arr_b)
        #     while (pa < size_a) or (pb < size_b):
        #         if arr_a[pa:] > arr_b[pb:]:
        #             ans.append(arr_a[pa])
        #             pa += 1
        #         else:
        #             ans.append(arr_b[pb])
        #             pb += 1
        #     return ans

        return max(merge_max(pick_max(nums1, i), pick_max(nums2, k-i))
                   for i in range(k+1) if i <= len(nums1) and k-i <= len(nums2))    # 遍历k的分解方式

    @classmethod
    def run(cls):
        nums1 = [3, 4, 6, 5]
        nums2 = [9, 1, 2, 5, 8, 3]
        k = 5

        repeat = (1, 1)    # 重复次数, 数据扩规模
        print('\033[7;34mQuest: {}\033[0m'.format(cls.__name__))

        print('单调栈:', timer(cls.sol_01_stack, (nums1, nums2, k), repeat))
        print('')




