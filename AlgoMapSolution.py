from typing import List, Optional
from collections import Counter, defaultdict
from math import ceil, floor
# https://algomap.io/


# 33 MinStack
class MinStack:
    def __init__(self):
        self.stk = []
        self.min_stk = []

    def push(self, val: int) -> None:
        self.stk.append(val)
        if not self.min_stk:
            self.min_stk.append(val)
        elif self.min_stk[-1] < val:
            self.min_stk.append(self.min_stk[-1])
        else:
            self.min_stk.append(val)

    def pop(self) -> None:
        self.stk.pop()
        self.min_stk.pop()

    def top(self) -> int:
        return self.stk[-1]

    def getMin(self) -> int:
        return self.min_stk[-1]



# https://leetcode.com/problems/remove-duplicates-from-sorted-list/

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class AlgoMapSolution:
    def __init__(self):
        pass

    # 1 findClosestNumber #########################

    def findClosestNumber(self, nums:List[int])->int:
        closest = nums[0] # the integer at position index 0
        for x in nums:
            if abs(x)<abs(closest):
                closest = x
        if closest < 0 and abs(closest) in nums:
            return abs(closest)
        else:
            return closest

    # 9 Merge Intervals #########################

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda interval: interval[0])
        merged = []

        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1] = [merged[-1][0], max(merged[-1][1], interval[1])]

        return merged

    # 10 spiralOrder #########################

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m, n = len(matrix), len(matrix[0])
        ans = []
        i, j = 0, 0
        UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
        direction = RIGHT

        UP_WALL = 0
        RIGHT_WALL = n
        DOWN_WALL = m
        LEFT_WALL = -1

        while len(ans) != m * n:
            if direction == RIGHT:
                while j < RIGHT_WALL:
                    ans.append(matrix[i][j])
                    j += 1
                i, j = i + 1, j - 1
                RIGHT_WALL -= 1
                direction = DOWN
            elif direction == DOWN:
                while i < DOWN_WALL:
                    ans.append(matrix[i][j])
                    i += 1
                i, j = i - 1, j - 1
                DOWN_WALL -= 1
                direction = LEFT
            elif direction == LEFT:
                while j > LEFT_WALL:
                    ans.append(matrix[i][j])
                    j -= 1
                i, j = i - 1, j + 1
                LEFT_WALL += 1
                direction = UP
            else:
                while i > UP_WALL:
                    ans.append(matrix[i][j])
                    i -= 1
                i, j = i + 1, j + 1
                UP_WALL += 1
                direction = RIGHT

        return ans
        # Time: O(m*n)
        # Space: O(1)

    # 11 rotate #########################

    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)

        # Tranpose
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

        # Reflection
        for i in range(n):
            for j in range(n // 2):
                matrix[i][j], matrix[i][n - j - 1] = matrix[i][n - j - 1], matrix[i][j]

        # Time: O(n^2)
        # Space: O(1)

# 12 numJewelsInStones #########################

    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        # O(n + m)
        s = set(jewels)
        count = 0
        for stone in stones:
            if stone in s:
                count += 1
        return count

# 13 containsDuplicate #########################

    def containsDuplicate(self, nums: list[int]) -> bool:
        h = set()
        for num in nums:
            if num in h:
                return True
            else:
                h.add(num)
        return False


# 14 ransomNote #########################

    def canConstruct(self, ransomNote: str, magazine: str) -> bool:

        #-------
        # Creation of a Counter Class object using
        # string as an iterable data container
        x = Counter("geeksforgeeks")

        # printing the elements of counter object
        for i in x.elements():
            print(i, end=" ")

        # g g e e e e k k s s f o r
        #-------

        # Creating a Counter class object using list as an iterable data container
        a = [12, 3, 4, 3, 5, 11, 12, 6, 7]

        y = Counter(a)

        # directly printing whole y
        print(y)

        # We can also use .keys() and .values() methods to access Counter class object
        for i in y.keys():
            print(i, ":", y[i])

        # We can also make a list of keys and values of y
        y_keys = list(y.keys())
        y_values = list(y.values())

        print(y_keys)
        print(y_values)

        # Counter({12: 2, 3: 2, 4: 1, 5: 1, 11: 1, 6: 1, 7: 1})
        # 12: 2
        # 3: 2
        # 4: 1
        # 5: 1
        # 11: 1
        # 6: 1
        # 7: 1
        # [12, 3, 4, 5, 11, 6, 7]
        # [2, 2, 1, 1, 1, 1, 1]

        #-------

        hashmap = Counter(magazine)  # TC for Counter is O(n)
        for ch in ransomNote:
            if hashmap[ch] > 0:
                hashmap[ch] -= 1
            else:
                return False
        return True

# 15 valid anagram #########################

    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        s_dict = Counter(s)
        t_dict = Counter(t)

        return s_dict == t_dict

# 16 maxNumberOfBalloons #########################

    def maxNumberOfBalloons(self, text: str) -> int:
        counter = defaultdict(int)
        balloon = "balloon"

        for c in text:
            if c in balloon:
                counter[c] += 1

        if any(c not in counter for c in balloon):
            return 0
        else:
            return min(counter["b"], counter["a"], counter["l"] // 2, counter["o"] // 2, counter["n"])

# 17 twoSum #########################

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        h = {}
        for i in range(len(nums)):
            h[nums[i]] = i

        for i in range(len(nums)):
            y = target - nums[i]

            if y in h and h[y] != i:
                return [i, h[y]]

# 18 isValidSudoku #########################

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # Validate Rows
        for i in range(9):
            s = set()
            for j in range(9):
                item = board[i][j]
                if item in s:
                    return False
                elif item != '.':
                    s.add(item)

        # Validate Cols
        for i in range(9):
            s = set()
            for j in range(9):
                item = board[j][i]
                if item in s:
                    return False
                elif item != '.':
                    s.add(item)

        # Validate Boxes
        starts = [(0, 0), (0, 3), (0, 6),
                  (3, 0), (3, 3), (3, 6),
                  (6, 0), (6, 3), (6, 6)]

        for i, j in starts:
            s = set()
            for row in range(i, i + 3):
                for col in range(j, j + 3):
                    item = board[row][col]
                    if item in s:
                        return False
                    elif item != '.':
                        s.add(item)
        return True

# 19 groupAnagrams #########################

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        anagrams_dict = defaultdict(list)

        print("before: anagrams_dict = ", anagrams_dict)
        # before: anagrams_dict =  defaultdict(<class 'list'>, {})

        for s in strs:  # n
            count = [0] * 26
            for c in s:
                count[ord(c) - ord("a")] += 1
            key = tuple(count)
            anagrams_dict[key].append(s)

        print("after: anagrams_dict = ", anagrams_dict)

        # after: anagrams_dict = defaultdict( <class 'list'>,
        # {(1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0):
        # ['eat', 'tea', 'ate'], (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        # 1, 0, 0, 0, 0, 0, 0): ['tan', 'nat'], (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0): ['bat']})


        print("ord('a') = ", ord('a'))
        # ord('a') =  97

        return anagrams_dict.values()

# 20 majorityElement #########################

    def majorityElement(self, nums: List[int]) -> int:
        candidate = None
        count = 0

        for num in nums:
            if count == 0:
                candidate = num

            count += 1 if candidate == num else -1

        return candidate

# 21 longestConsecutive #########################
    def longestConsecutive(self, nums: List[int]) -> int:
        s = set(nums)
        longest = 0

        for num in s:
            if num - 1 not in s:
                next_num = num + 1
                length = 1
                while next_num in s:
                    length += 1
                    next_num += 1
                longest = max(longest, length)

        return longest



# 22 sortedSquares #########################

    def sortedSquares(self, nums: List[int]) -> List[int]:
        left = 0
        right = len(nums) - 1
        result = []

        while left <= right:
            if abs(nums[left]) > abs(nums[right]):
                result.append(nums[left] ** 2)
                left += 1
            else:
                result.append(nums[right] ** 2)
                right -= 1

        result.reverse()

        return result


# 23 reverseString #########################

    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        n = len(s)
        left = 0
        right = len(s) - 1

        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1

        return s

    # 24 twoSum2 #########################

    def twoSum2(self, numbers: List[int], target: int) -> List[int]:
        n = len(numbers)
        left = 0
        right = len(numbers) - 1

        while left < right:
            summ = numbers[left] + numbers[right]
            if summ == target:
                return [left + 1, right + 1]
            elif summ < target:
                left += 1
            else:
                right -= 1

# 24 twoSum #########################

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        h = {}
        for i in range(len(nums)): # i is index, not value
            h[nums[i]] = i

        print("h = ", h)
        # h =  {2: 0, 7: 1, 3: 2, 6: 3, 11: 4, 15: 5}

        for i in range(len(nums)):
            y = target - nums[i] # 7 = 9 - 2, i = 0

            if y in h and h[y] != i: # y = 7, h[7] = 1, i = 0; not of the same index
                return [i, h[y]]

    # 25 isPalindrome #########################

    def isPalindrome(self, s: str) -> bool:
        n = len(s)
        left = 0
        right = len(s) - 1

        while left < right:
            if not s[left].isalnum():
                left += 1
                continue

            if not s[right].isalnum():
                right -= 1
                continue

            if s[left].lower() != s[right].lower():
                return False

            left += 1
            right -= 1

        return True


# x xxxxxxxxxxxxx #########################

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        h = {}
        n = len(nums)
        s = set()

        for i, num in enumerate(nums):
            h[num] = i

        for i in range(n):
            for j in range(i + 1, n):
                desired = -nums[i] - nums[j]
                if desired in h and h[desired] != i and h[desired] != j:
                    s.add(tuple(sorted([nums[i], nums[j], desired])))

        return s

    def threeSumTwoPointers(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        answer = []
        for i in range(n):
            if nums[i] > 0:
                break
            elif i > 0 and nums[i] == nums[i - 1]:
                continue
            lo, hi = i + 1, n - 1
            while lo < hi:
                summ = nums[i] + nums[lo] + nums[hi]
                if summ == 0:
                    answer.append([nums[i], nums[lo], nums[hi]])
                    lo, hi = lo + 1, hi - 1
                    while lo < hi and nums[lo] == nums[lo - 1]:
                        lo += 1
                    while lo < hi and nums[hi] == nums[hi + 1]:
                        hi -= 1
                elif summ < 0:
                    lo += 1
                else:
                    hi -= 1

        return answer


# 27 Container With Most Water #########################

    def maxArea(self, height: List[int]) -> int:
        n = len(height)
        left = 0
        right = len(height) - 1
        max_area = 0

        while left < right:
            w = right - left
            h = min(height[left], height[right])
            a = w * h
            max_area = max(max_area, a)

            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return max_area


# 28 Trapping Rain Water #########################

    def trap(self, height: List[int]) -> int:
        l_wall = r_wall = 0
        n = len(height)
        max_left = [0] * n
        max_right = [0] * n

        for i in range(n):
            j = -i - 1
            max_left[i] = l_wall
            max_right[j] = r_wall
            l_wall = max(l_wall, height[i])
            r_wall = max(r_wall, height[j])

        summ = 0
        for i in range(n):
            pot = min(max_left[i], max_right[i])
            summ += max(0, pot - height[i])

        return summ

    # 29 Baseball Game #########################

    def calPoints(self, operations: List[str]) -> int:
        stk = []

        for op in operations:
            if op == "+":
                stk.append(stk[-1] + stk[-2])
            elif op == "D":
                stk.append(stk[-1] * 2)
            elif op == "C":
                stk.pop()
            else:
                stk.append(int(op))

        return sum(stk)


# 30 Valid Parentheses #########################
    def isValid(self, s: str) -> bool:
        hashmap = {")": "(", "}": "{", "]": "["}
        stk = []

        for c in s:
            if c not in hashmap:
                stk.append(c)
            else:
                if not stk:
                    return False
                else:
                    popped = stk.pop()
                    if popped != hashmap[c]:
                        return False

        return not stk

# 30 Valid Parentheses #########################

# from math import ceil, floor
    def evalRPN(self, tokens: List[str]) -> int:
        stk = []
        for t in tokens:
            if t in "+-*/":
                b, a = stk.pop(), stk.pop()

                if t == "+":
                    stk.append(a + b)
                elif t == "-":
                    stk.append(a - b)
                elif t == "*":
                    stk.append(a * b)
                else:
                    division = a / b
                    if division < 0:
                        stk.append(ceil(division))
                    else:
                        stk.append(floor(division))
            else:
                stk.append(int(t))

        return stk[0]

# 32 dailyTemperatures #########################

    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        temps = temperatures
        n = len(temps)
        answer = [0] * n
        stk = []

        for i, t in enumerate(temps):
            print("stk_outer = ", stk)
            while stk and stk[-1][0] < t:
                print("stk = " , stk)

                # temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
                # stk_outer = []
                # stk_outer = [(73, 0)]
                # stk = [(73, 0)]
                # stk_outer = [(74, 1)]
                # stk = [(74, 1)]
                # stk_outer = [(75, 2)]
                # stk_outer = [(75, 2), (71, 3)]
                # stk_outer = [(75, 2), (71, 3), (69, 4)]
                # stk = [(75, 2), (71, 3), (69, 4)]
                # stk = [(75, 2), (71, 3)]
                # stk_outer = [(75, 2), (72, 5)]
                # stk = [(75, 2), (72, 5)]
                # stk = [(75, 2)]
                # stk_outer = [(76, 6)]

                stk_t, stk_i = stk.pop()
                answer[stk_i] = i - stk_i

            stk.append((t, i))
        return answer

# 34 Remove Duplicates from sorted list #########################
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head

        while cur and cur.next:
            if cur.val == cur.next.val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return head

    def deleteDuplicatesUpdated(self, nodeList: List[int]) -> Optional[ListNode]:

        n = len(nodeList)
        head = ListNode(nodeList[n-1], None)
        i = n-2
        while i >=0:
            head = ListNode(nodeList[i], head)
            i -= 1

        cur = head

        while cur and cur.next:
            if cur.val == cur.next.val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return head

    def printListNode(self, head: Optional[ListNode]):
        cur = head
        resultList = []

        while cur != None:
            resultList.append(cur.val)
            cur = cur.next

        print(resultList)

    # 35 reverseList #########################

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head
        prev = None

        while cur:
            temp = cur.next
            cur.next = prev
            prev = cur
            cur = temp

        return prev


# 36 Merged Two Sorted Lists #########################
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        d = ListNode()
        cur = d

        while list1 and list2:
            if list1.val < list2.val:
                cur.next = list1
                cur = list1
                list1 = list1.next
            else:
                cur.next = list2
                cur = list2
                list2 = list2.next

        cur.next = list1 if list1 else list2

        return d.next

    # 37 Linked List Cycle #########################

    def hasCycle(self, head: Optional[ListNode]) -> bool:
        dummy = ListNode()
        dummy.next = head
        slow = dummy
        fast = dummy

        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

            if slow is fast:
                return True

        return False

    # 38 Middle of the linked list #########################
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = head
        fast = head

        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

        return slow

    def middleNodeUpdated(self, head: Optional[ListNode]) -> int:
        slow = head
        fast = head

        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next

        return slow.val

    # 80 Word Search #########################



    # xx xxxxxxxxxxxxxxxxx #########################


# 1 findClosestNumber #########################

algoMapClass = AlgoMapSolution()

integerArray = [-2,9,-3, 2, 4,6]
print("findClosestNumber = ", algoMapClass.findClosestNumber(integerArray))
# findClosestNumber = 2

# 18 isValidSudoku #########################

board = [["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]

print("isValidSudoku = ", algoMapClass.isValidSudoku(board))
# isValidSudoku = True


# 19 groupAnagrams #########################
strs = ["eat","tea","tan","ate","nat","bat"]

print("groupAnagrams = ", algoMapClass.groupAnagrams(strs))
# groupAnagrams = dict_values([['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']])


# 20 majorityElement #########################

nums = [2,2,1,1,1,2,2]

print("majorityElement = ", algoMapClass.majorityElement(nums))
# majorityElement =  2


# 21 longestConsecutive #########################

nums = [100,4,200,1,3,2]

print("longestConsecutive = ", algoMapClass.longestConsecutive(nums))
# longestConsecutive =  4


# 22 sortedSquares #########################

nums = [-4,-1,0,3,10]

print("sortedSquares = ", algoMapClass.sortedSquares(nums))
# sortedSquares =  [0, 1, 9, 16, 100]


# 23 reverseString #########################

s = ["h","e","l","l","o"]

print("reverseString = ", algoMapClass.reverseString(s))
# reverseString =  [0, 1, 9, 16, 100]

# 24 twoSum2 #########################
numbers2 = [2,7, 11,15]
target2 = 9
print("twoSum2 = ", algoMapClass.twoSum2(numbers2, target2))
# twoSum2 = [1, 2]

# 24 twoSum #########################
numbers1 = [2,7,3, 6, 11,15]
target1 = 9
print("twoSum = ", algoMapClass.twoSum(numbers1, target1))
# twoSum =  [0, 1]


# 25 isPalindrome #########################
s = "A man, a plan, a canal: Panama"
print("isPalindrome = ", algoMapClass.isPalindrome(s))
# isPalindrome =  True

# 26 threeSum #########################
nums = [-1,0,1,2,-1,-4]
print("threeSum = ", algoMapClass.threeSum(nums))
# threeSum =  {(-1, 0, 1), (-1, -1, 2)}

# 26 threeSumTwoPointers #########################
print("threeSumTwoPointers = ", algoMapClass.threeSumTwoPointers(nums))
# threeSumTwoPointers =  [[-1, -1, 2], [-1, 0, 1]]


# 27 Container with Most Water #########################
height = [1,8,6,2,5,4,8,3,7]
print("maxArea = ", algoMapClass.maxArea(height))
# maxArea =  49


# 28 Trapping Rain Water #########################
height = [0,1,0,2,1,0,1,3,2,1,2,1]

print("trap = ", algoMapClass.trap(height))
# trap =  6


# 29 Baseball Game #########################
ops = ["5","2","C","D","+"]

print("calPoints = ", algoMapClass.calPoints(ops))
# calPoints =  30

# 30 Valid Parentheses #########################
s = "([])"

print("isValid = ", algoMapClass.isValid(s))
# isValid =  True

# 31 Evaluate Reverse Polish Notation #########################

tokens = ["2","1","+","3","*"]

print("evalRPN = ", algoMapClass.evalRPN(tokens))
# evalRPN =  9

# 32 dailyTemperatures #########################

temperatures = [73,74,75,71,69,72,76,73]

print("dailyTemperatures = ", algoMapClass.dailyTemperatures(temperatures))
# dailyTemperatures =  [1, 1, 4, 2, 1, 1, 0, 0]


# 33 MinStack #########################

myMinStack = MinStack()
myMinStack.push(-2)
myMinStack.push(0)
myMinStack.push(-3)
print("getMin() = ", myMinStack.getMin())
myMinStack.pop()
print("top() = ", myMinStack.top())
print("getMin() = ", myMinStack.getMin())

# getMin() =  -3
# top() =  0
# getMin() =  -2


# 34 Remove Duplicates from sorted list #########################
L5 = ListNode(3, None)
L4 = ListNode(3, L5)
L3 = ListNode(2, L4)
L2 = ListNode(1, L3)
L1 = ListNode(1, L2)
algoMapClass.printListNode(algoMapClass.deleteDuplicates(L1))

nodeList = [1,1,2,3,3]
algoMapClass.printListNode(algoMapClass.deleteDuplicatesUpdated(nodeList))
algoMapClass.printListNode(algoMapClass.deleteDuplicatesUpdated([1,1,1,2,2,2,3,3,3,4,4,4]))


# 35 reverseList #########################

L5 = ListNode(5, None)
L4 = ListNode(4, L5)
L3 = ListNode(3, L4)
L2 = ListNode(2, L3)
L1 = ListNode(1, L2)
algoMapClass.printListNode(algoMapClass.reverseList(L1))

# [5, 4, 3, 2, 1]


# 36 Merged Two Sorted Lists #########################



L1_3 = ListNode(4, None)
L1_2 = ListNode(2, L1_3)
L1_1 = ListNode(1, L1_2)

L2_3 = ListNode(4, None)
L2_2 = ListNode(3, L2_3)
L2_1 = ListNode(1, L2_2)


algoMapClass.printListNode(algoMapClass.mergeTwoLists(L1_1, L2_1))
# [1, 1, 2, 3, 4, 4]


# 37 Linked List Cycle #########################

L_4 = ListNode(-4, None)
L_3 = ListNode(0, L_4)
L_2 = ListNode(2, L_3)
L_1 = ListNode(3, L_2)

L_4.next = L_2

print(algoMapClass.hasCycle(L_1))
# True


# 38 Middle of the Linkedlist #########################
head = [1,2,3,4,5]

L_5 = ListNode(5, None)
L_4 = ListNode(4, L_5)
L_3 = ListNode(3, L_4)
L_2 = ListNode(2, L_3)
L_1 = ListNode(1, L_2)

algoMapClass.printListNode(algoMapClass.middleNode(L_1)) # odd numbers
# [3, 4, 5]

print("middleNodeUpdated = ", algoMapClass.middleNodeUpdated(L_1))
# middleNodeUpdated =  3

L_6 = ListNode(6, None)
L_5 = ListNode(5, L_6)
L_4 = ListNode(4, L_5)
L_3 = ListNode(3, L_4)
L_2 = ListNode(2, L_3)
L_1 = ListNode(1, L_2)

algoMapClass.printListNode(algoMapClass.middleNode(L_1))#even numbers
# [4, 5, 6]

print("middleNodeUpdated = ", algoMapClass.middleNodeUpdated(L_1))
# middleNodeUpdated =  4

# xx xxxxxxxxxxxxxxxxx #########################


# x xxxxxxxxxxxxx #########################