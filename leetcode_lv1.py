from typing import List
from leetcode_lv0 import timer


class No_0316_RemoveDuplicateLetters:
    """
    请去除字符串s中重复的字母，使得每个字母只出现一次。需保证, 返回结果的字典序最小（要求不能打乱其他字符的相对位置)
    相同题目1081: 返回 s 字典序最小的子序列，该子序列包含 s 的所有不同字符，且只包含一次。
    """
    @staticmethod
    def sol_01_stack(s: str):
        """
        贪心+单调栈
        思路：顺序保存字符，当前再次出现已出现过的字符时，如果字典序正确，则丢弃历史记录，保存新的记录
        栈实现：遍历字符，栈为空则入栈。关键！如果当前字符比栈顶字符的字典序更前，而且剩余字符中存在该字符，则弹出该字符。
        """
        # 限定字符为26个字母
        asc_a = ord('a')  # a的ASCII码方便运算
        size = len(s)
        stack = []

        char_last_idx = [-1 for _ in range(26)]     # 记录每个字符最后一次出现的位置
        asc_arr = [-1 for _ in range(size)]         # 转换为ASCII方便计算
        for i in range(size):
            asc = ord(s[i]) - asc_a
            char_last_idx[asc] = i
            asc_arr[i] = asc

        char_visited = [False for _ in range(26)]       # 字符进栈的标志
        for i in range(size):
            if char_visited[asc_arr[i]]:                # 跳过栈内元素
                continue
            # 栈不为空 & 当前字符在栈顶字符之前 & 后面还会出现栈顶字符
            while stack and stack[-1] > asc_arr[i] and char_last_idx[stack[-1]] > i:
                char_visited[stack.pop()] = False    # 出栈

            stack.append(asc_arr[i])   # 字符进栈
            char_visited[asc_arr[i]] = True

        return ''.join([chr(asc+asc_a) for asc in stack])

    @staticmethod
    def sol_02_stack(s: str):
        """
        逻辑相同，只是用内置函数简化代码，但是字符比较相对与数值比较更低效。
        """
        import collections
        stack = []
        remain_counter = collections.Counter(s)  # 利用内置函数实现识别字符最后一次出现的位置
        char_in_stack = {key: False for key in remain_counter}

        for c in s:
            if not char_in_stack[c]:
                while stack and c < stack[-1] and remain_counter[stack[-1]] > 0:
                    char_in_stack[stack.pop()] = False
                stack.append(c)
                char_in_stack[c] = True
            remain_counter[c] -= 1
        return ''.join(stack)

    @classmethod
    def run(cls):
        s = "cbacdcbc"
        repeat = (10, 100)    # 重复次数, 数据扩规模
        print('\033[7;34mQuest: {}\033[0m'.format(cls.__name__))

        print('单调栈:', timer(cls.sol_01_stack, s, repeat))
        print('单调栈:', timer(cls.sol_02_stack, s, repeat))
        print('')


class No_0739_DailyTemperatures:
    """
    给定一个整数数组temperatures，表示每天的温度，返回一个数组answer，
    其中answer[i]是指在第 i 天之后，才会有更高的温度。如果气温在这之后都不会升高，请在该位置用0 来代替
    """

    @staticmethod
    def sol_01_traversal(temperatures: List[int]):
        """
        暴力解: 遍历
        """
        size = len(temperatures)
        ans = [0 for _ in temperatures]
        for i in range(size):
            for j in range(i+1, size):
                if temperatures[j] > temperatures[i]:
                    ans[i] = j - i
                    break
        return ans

    @staticmethod
    def sol_02_stack(temperatures):
        """
        栈: 栈为空或栈顶温度小于等于当天温度，入栈；说明遇到升温日，出栈计算
        """
        size = len(temperatures)
        ans = [0 for _ in temperatures]
        stack = []
        for day in range(size):
            while stack and temperatures[day] > temperatures[stack[-1]]:
                pre_day = stack.pop()
                ans[pre_day] = day - pre_day
            stack.append(day)
        return ans

    @classmethod
    def run(cls):
        temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
        repeat = (1, 1000)    # 重复次数, 数据扩规模
        print('\033[7;34mQuest: {}\033[0m'.format(cls.__name__))

        print('暴力解:\t', timer(cls.sol_01_traversal, temperatures, repeat))
        print('单调栈:\t', timer(cls.sol_02_stack, temperatures, repeat))
        print('')


class No_0402_RemoveKDigits:
    """
    一个以字符串表示的非负整数num和一个整数 k ，移除这个数中的k位数字，使得剩下的数字最小。以字符串形式返回这个最小的数字。
    思路：
    两个相同位数的数字大小关系取决于第一个不同的数的大小，
    如果一个数字的右侧值比它更大，那么这个数据就不能丢弃，因为丢弃后会使得这个位数上的数值变大。如果是一个递增数列，则丢弃最后的几位。
    由于删除中间元素的操作效率较低，逆转上述思路，就是保留n-k个元素，使用栈存储，当前遍历的元素比栈顶小，则出栈

    关联+：
    No321_MaxNumber
    """
    @staticmethod
    def sol_01_stack(num: str, k: int):
        stack = []
        rm = 0
        for str_n in num:
            n = int(str_n)
            if stack and stack[-1] > n:
                stack.pop()
                rm += 1
            stack.append(n)
        while rm < k:   # 对于升序队列还可能需要从尾部删起
            stack.pop()
            rm += 1
        return ''.join([str(i) for i in stack])

    @staticmethod
    def sol_02_stack(num: str, k: int):
        stack = []
        remain = len(num) - k
        for str_n in num:
            n = int(str_n)
            if stack and stack[-1] > n:
                stack.pop()
                k -= 1
            stack.append(n)
        return ''.join([str(i) for i in stack[:remain]])  # 逻辑基本相同, 用切片解决不足的部分

    @classmethod
    def run(cls):
        num = "1432219"
        # num = "12345678"
        k = 3
        repeat = (1000, 1)    # 重复次数, 数据扩规模
        print('\033[7;34mQuest: {}\033[0m'.format(cls.__name__))
        print('栈:', timer(cls.sol_01_stack, (num, k), repeat))
        print('栈:', timer(cls.sol_02_stack, (num, k), repeat))
        print('')
