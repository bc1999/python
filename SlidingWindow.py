class SlidingWindowInt:
    def __init__(self):
        pass

    def sumWindow(self, k, a):
        cur = best = sum(a[:k])
        my_list=[]
        my_list.append(cur)
        for r in range(k, len(a)):
            cur = cur + a[r] - a[r-k]
            best = max(best, cur)
            my_list.append(cur)
        print(my_list)
        # [18, 9, 6, 13, 12, 16, 11]
        return best


class SlidingWindowIntDynamic:
    def __init__(self):
        pass

    def sumWindow(self, s, a):

        left, cur, best = -1, 0, 0
        for r in range(len(a)):
            cur += a[r]

            while cur >= s:
                left += 1
                cur -= a[left]
            best = max(best, r-left)

        return best


swInt = SlidingWindowInt()

print(swInt.sumWindow(5, [8, 3,-2,4,5,-1,0,5,3,9,-6]))
# 18

swIntDyn = SlidingWindowIntDynamic()
print(swIntDyn.sumWindow(15, [4,5,2,0,1,8,12,3,6,9]))#
# 5
