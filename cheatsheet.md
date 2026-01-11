#### 双指针（two pointers）：

```python
#同向双指针（快慢指针）
#示例：移除数组中的特定元素
def remove_element(nums,val):
    #移除数组中所有等于val的元素#
    slow = 0 #慢指针：指向下一个有效元素的位置
    for fast in range(len(nums)): #快指针：遍历所有元素
        if nums[fast] != val:
            nums[slow] = nums[fast] #将前几个元素替换为有效的元素
            slow += 1
    return slow #返回新数组长度
nums = [3, 2, 2, 3]
new_length = remove_element(nums,3)
print(nums[:new_length]) #输出：[2,2]

#对向双指针（左右指针）
#示例：有序数组的两数之和
def two_sum_sorted(number,target):
	#在有序数组numbers中找两个数，使他们的和等于target#
    left, right = 0, len(numbers) - 1
    
    while left < right:
        current_sum = numbers[left] + numbers[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1 #和太小，左指针右移
        else:
            right -= 1 #和太大，右指针左移
    return [] #如果找不到返回空列表

numbers=[2, 7, 11, 15]
print(two_sum_sorted(numbers,9)) #输出：[0,1]

#滑动窗口
#示例：长度最小的子数组
def min_subarray_len(target, nums):
    #找到和>=target的长度最小的连续子数组#
    left = 0
    current_sum = 0
    min_len = float('inf')
    for right in range(len(nums)):
        current_sum += nums[right] #扩大窗口
        while current_sum >= target: #满足条件时收缩窗口
            minlen = min(min_len, right-left+1)
            current_sum -= nums[left]
            left+=1   
    return 0 if min_len == float('inf') else min_len
nums = [2, 3, 1, 2, 4, 3]
print(min_subarray_len(7,nums)) #输出：2

#滑动窗口最大值：双指针+滑动窗口+deque
#给定一个数组nums和一个整数k，请输出每个长度为k的子数组中的最大值
nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
#>> [3, 3, 5, 5, 6, 7]
from collections import deque
def maxSlidingWindow(nums, k):
    dq = deque()  # 存下标，保证对应值递减
    res = []
    for right, x in enumerate(nums):
        # step 1: 窗口右扩，保持单调递减
        while dq and nums[dq[-1]] <= x: #把所有小于x的数都弹出，因为他们不可能                                          是最大值
            dq.pop()
        dq.append(right)

        # step 2: 移除滑出窗口的左端元素
        if dq[0] <= right - k:
            dq.popleft() #所以说不能用列表代替deque

        # step 3: 当窗口形成（长度 >= k）时，记录最大值
        if right >= k - 1:
            res.append(nums[dq[0]])
    return res
```

#### 单调栈（Monotonic Stack）

```python
#应用场景：寻找下一个更大（更小）的元素，直方图中的最大矩形，滑动窗口的最大值
#工作原理：入栈操作：当一个新元素需要加入到栈中时，根据栈的性质（递增或递减），将所有不符合条件的栈顶元素弹出，然后再将新元素压入栈中。                                          出栈操作：通常情况下，出栈操作是自动发生的，即在执行入栈操作时，为了保持栈的单调性，会自动移除不满足条件的栈顶元素。

#单调递增栈模板
def monotoneIncreasingStack(nums):
    stack = []
    for i, num in enumerate(nums):
        while stack and num >= stack[-1]:
            stack.pop()      
        stack.append(num)
    return stack
#单调递减栈模板就是把>=改成<=,其中=是否保留根据题目中是否保留相同元素决定
```

#### 二分查找（binary search）

```python
def binary_search(arr, target): #找到target在arr中的下标
    left, right = 0, len(arr)-1
    
    while left <= right:
        mid = (left+right)//2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid+1
        else:
            right = mid-1
    return -1

#python内置的二分库bisect
bisect.bisect_left(arr,x,key) (bisect_right同理)
bisect.insort_left(arr,x,key)#将待插入元素x插入给定列表arr中（原地操作）
```

#### 贪心（greedy）

```python
#跳跃游戏问题
#给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false 。
def canJump(self, nums: List[int]) -> bool:
	max_s = 0
    for i in range(len(nums)):
        if i > max_s: #如果当前到达位置已经大于最大能到达位置
            return False
        max_s = max(max_s, i+nums[i]) #更新最大能到达位置
        if max_s >= len(nums)-1: #如果最大能到达位置已经大于最后一个下标
            return True

#跳跃游戏II：求跳跃的最小次数
def jump(self, nums: List[int]) -> int:
    if len(nums) <= 1:
        return 0
    current_end, current_max, cnt = 0, 0, 0
    for i in range(len(nums)):
        current_max = max(current_max, i+nums[i])#更新最大能到达位置
        if i == current_max:
            return -1 #两段之间不能衔接上，所以不能跳跃到最后
        if i == current_end: #如果说这里是区间覆盖问题，而且不要求连续，那么就是								i > current_end
            current_end = current_max #在current_end范围内找能走最多步的那步
            cnt += 1 #到达一段的边界需要再加一步
            if current_end >= len(nums)-1:
                break
    return cnt

#装箱问题
import math
rest = [0,5,3,1] #用来算装完3*3的之后还有多少空余能来装2*2
while True:
    a,b,c,d,e,f = map(int,input().split())
    if a + b + c + d + e + f == 0:
        break
    boxes = d + e + f           #装4*4, 5*5, 6*6
    boxes += math.ceil(c/4)     #填3*3
    spaceforb = 5*d + rest[c%4] #能和4*4 3*3 一起放的2*2
    if b > spaceforb:
    	boxes += math.ceil((b - spaceforb)/9)
    spacefora = boxes*36 - (36*f + 25*e + 16*d + 9*c + 4*b)     #和其他箱子一起的填的1*1
    if a > spacefora:
        boxes += math.ceil((a - spacefora)/36)
    print(boxes)
```

#### 矩阵

```python
#row:行 column:列
#矩阵乘法,A、B分别为m*n矩阵和n*p矩阵，D为A*B
D = [[0]*p for _ in range(m)]
m, n, p = len(A), len(B), len(B[0])
for i in range(m):
    for j in range(p):
		for k in range(n):
            D[i][j] += A[i][k]*B[k][j]
print(D)
```

#### 前缀和及Kadane算法

```python
#一维前缀和
#给定数组A[0..n-1],其前缀和数组P定义为：
P[0] = 0
P[i] = A[0] + A[1] + ... + A[i-1]
#区间[l, r]的和可以快速计算为：p[r+1] - P[l]

#二维前缀和
#P[i][j]表示从(0, 0)到(i-1, j-1)的矩形区域的元素和
P[i][j] = matrix[i-1][j-1]+P[i-1][j]+P[i][j-1]-P[i-1][j-1]
#矩形区域(x1, y1)到(x2, y2)的和为：
sum = P[x2+1][y2+1]-P[x1][y2+1]-P[x2+1][y1]+P[x1][y1]

#Kadane算法（一维最大子数组和）
def kadane(arr):
    curr_max = total_max = arr[0]
    for x in arr[1:]:
        curr_max = max(x, curr_max + x)  # 要么重新开始，要么接上前面
        total_max = max(total_max, curr_max)
    return total_max
#扩展到二维——最大子矩阵
def kadane(s):
    curr_max = total_max = s[0]
    for x in s[1:]:
        curr_max = max(x, curr_max + x)
        total_max = max(total_max, curr_max)
    return total_max

def max_sum_matrix(mat):
    max_sum = -float('inf')
    row, col = len(mat), len(mat[0])
    for top in range(row):
        col_sum = [0] * col
        for bottom in range(top, row):
            for c in range(col):
                col_sum[c] += mat[bottom][c]
            max_sum = max(max_sum, kadane(col_sum))
    return max_sum
```

#### 区间问题

```python
#1.区间合并问题：合并所有有交集的区间（包括端点处相交）
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])
        ans = []
    
        for x in intervals:
            if ans and x[0] <= ans[-1][1]:
                ans[-1][1] = max(ans[-1][1], x[1])
            else:
                ans.append(x)
        return ans

#2.选择不相交区间：选择尽可能多的区间，使这些区间互不相交（只在端点上重合的区间不算相交），求可选取的区间的最大数量
#思路：优先保留结束早的区间，这样能为后续留下更多空间，所以要按右端点排序
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key = lambda x: x[1])
        cur_inv = []
        res = 0
        for x in intervals:
            if not cur_inv or x[0] >= cur_inv[1]:
                cur_inv = [x[0], x[1]]
            elif x[0] < cur_inv[1]:
                res += 1
        return res
   
#3.区间选点问题：取尽量少的点，使得每个区间内至少有一个点（不同区间内的点可以是同一个）
#思路和代码与选择不相交区间问题一样
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key = lambda x:x[-1])
        res = 0
        cur_inv = []
        for x in points:
            if not cur_inv or x[0] > cur_inv[1]:
                cur_inv = [x[0], x[1]]
                res += 1
        return res

#4.区间覆盖问题：给出一堆区间和一个目标区间，问最少选择多少区间可以覆盖掉目标区间
#与上面的跳跃游戏问题相同，只不过是需要自己预先处理一下得到与那道题相同的条件
n = int(input())
a = list(map(int, input().split()))
far = [0] * (n + 2)
for i, x in enumerate(a, start=1):
    L = max(1, i - x)
    R = min(n, i + x)
    far[L] = max(far[L], R)
ans = 0
covered = 0
best = 0

for i in range(1, n + 1):
    best = max(best, far[i])
    if i > covered:
        ans += 1
        covered = best
print(ans)

#5.区间分组问题：给出一堆区间问最少可以将这些区间分成多少组使得每个组内的区间互不相交
#思路：将问题转化为最多有多少个区间能同时重叠。将进入区间和走出区间转化为事件，进入区间时区间数+1，走出区间时区间数-1，当同时有进入区间和走出区间事件时先处理走出区间事件。
from typing import List

class Solution:
    def minmumNumberOfHost(self, n: int, startEnd: List[List[int]]) -> int:
        # 将每个活动的开始时间和结束时间转换为事件
        events = []
        for i in range(n):
            start, end = startEnd[i]
            events.append((start, 1))  # 活动开始，+1主持人
            events.append((end, -1))  # 活动结束，-1主持人
        # 对事件按照时间排序，如果时间相同，先处理结束事件
        events.sort(key=lambda x: (x[0], x[1]))
        min_hosts = 0
        current_hosts = 0
        # 遍历所有事件，计算需要的主持人数
        for time, event in events:
            current_hosts += event
            min_hosts = max(min_hosts, current_hosts)
        return min_hosts

#6.覆盖连续区间：购物问题，有n中不同面值的硬币，每种硬币无限个，求能组合出1到x之间任意值需要携带多少个硬币
#思路：从大到小遍历硬币面值，当遇到比目前能达到的最大面值大1的硬币（coins[i]）为止，加上这枚硬币之后能达到的最大面值就变成了cur+coins[i]，直到能达到目标值为止。
x, n = map(int, input().split())
coins = list(map(int, input().split()))
if 1 not in coins:
    print(-1)
else:
    coins.sort(reverse = True)
    cur = res = 0
    while cur < x:
        tmp = 0
        for i in range(n):
            if coins[i] <= cur+1:
                tmp = coins[i]
                res += 1
                break
        cur += tmp
    print(res)
```

#### 动态规划（recursion）：

```python
# 01背包：每个物品最多选一次
n,b=map(int, input().split())
price=[0]+[int(i) for i in input().split()]
weight=[0]+[int(i) for i in input().split()]
bag=[[0]*(b+1) for _ in range(n+1)] #表示前i个物品装容量为j的背包的最大值
for i in range(1,n+1):
    for j in range(1,b+1):
        if weight[i]<=j:
            bag[i][j]=max(price[i]+bag[i-1][j-weight[i]], bag[i-1][j])
        else:
            bag[i][j]=bag[i-1][j]
print(bag[-1][-1])

def knapsack_01(n, C, w, v):
    #n是物品数量，C是背包最大承重
    dp = [0] * (C + 1) #表示背包重量为i时的最大价值，初始值为0表示什么也不干的时候最大价值是0
    for i in range(n):
        for j in range(C, w[i]-1, -1):  # 倒序！
            dp[j] = max(dp[j], dp[j-w[i]] + v[i])
    return dp[C]

# 完全背包：每个物品无限次
def complete_knapsack(n, C, w, v):
    dp = [0] * (C + 1)
    for i in range(n):
        for j in range(w[i], C+1):  # 正序！
            dp[j] = max(dp[j], dp[j-w[i]] + v[i])
    return dp[C]

#恰好型01背包
t,n=map(int,input().split())
dp=[0]+[-1]*(t+1) #初始值为-1表示有可能做不到恰好填满背包
for i in range(n):
    k,w=map(int,input().split())
    for j in range(t,k-1,-1):
        if dp[j-k]!=-1:
            dp[j]=max(dp[j-k]+w,dp[j])
print(dp[t])

#最长公共子序列 一个长为a，一个长为b
dp = [[0]*(b+1) for i in range(a+1)]#dp[i][j]为a序列前i个元素和b序列前j个元素                                      中相同元素的个数
for i in range(1, a+1):
    for j in range(1, b+1):
        if a[i-1] == b[j-1]:
            dp[i][j] = dp[i-1][j-1] + 1
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
print(dp[a][b])

# 二维费用背包：两个限制条件
def two_dim_knapsack(n, C1, C2, w, v, c):
    dp = [[0]*(C2+1) for _ in range(C1+1)]
    for i in range(n):
        for j in range(C1, w[i]-1, -1):
            for k in range(C2, c[i]-1, -1):
                dp[j][k] = max(dp[j][k], dp[j-w[i]][k-c[i]] + v[i])
    return dp[C1][C2]

# ==================== 线性DP模板 ====================
# 最长递增子序列(LIS)
def LIS(nums):
    n = len(nums)
    dp = [1] * n  #dp[i]表示以nums[i]结尾的最长上升子序列长度
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
#bisect写法
def LIS(nums):
    n = len(nums)
    dp = [1e9]*n
    for i in nums:
        dp[bisect.bisect_left(dp, i)] = i #如果允许非严格递增要用right
    return bisect.bisect_left(dp, 1e8)

# 最大子数组和 #最大连续子序列和
def max_subarray(nums):
    dp = [0] * len(nums)
    dp[0] = nums[0]
    for i in range(1, len(nums)):
        dp[i] = max(nums[i], dp[i-1] + nums[i])
    return max(dp)
#如果问最大连续子序列的最优方案
n = int(input())
*a, = map(int, input().split())
dp = [0]*n
start =[0]*n
dp[0] = a[0]
for i in range(1, n):
    if (dp[i-1] >= 0):
        dp[i] = dp[i-1] + a[i]
        start[i] = start[i-1]
    else:
        dp[i] = a[i]
        start[i] = i
max_val = max(dp)
pos = dp.index(max_val)
print(max_val, start[pos]+1, pos+1)

# 编辑距离：将word1转换成word2所使用的最少操作数
def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0]*(n+1) for _ in range(m+1)]#把word1的前i个字符变成word2的前j个                                         字符最少需要多少步
    for i in range(m+1): dp[i][0] = i #全删了
    for j in range(n+1): dp[0][j] = j #全加上
    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                #从左到右依次是删掉word1的最后一个字符，在word1最后加上一个和                   word2一样的字符，把word1的字符替换成word2的字符。
    return dp[m][n]

#求路径总数（只能向右或者向下走）
def uniquePaths(m, n):
    # 1. 建表：创建一个 m 行 n 列的二维数组
    dp = [[0] * n for _ in range(m)] #表示走到(i, j)这个格子有多少种走法
    # 2. 初始化：第一行和第一列
    # 为什么？因为沿着第一行走，只能向右，只有1种走法
    # 沿着第一列走，只能向下，也只有1种走法
    for i in range(m):
        dp[i][0] = 1
    for j in range(n):
        dp[0][j] = 1

    # 3. 填表：从 (1,1) 开始遍历（因为(0,0)及边界已经填了）
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
            #只能是从上面走下来的或者是从左边走过来的
    # 4. 返回右下角的结果
    return dp[m-1][n-1]
#如果是最小路径和（最小路费）问题，状态转移方程为(grid[i][j]为(i,j)格的路费)：
dp[i][j] = min(dp[i-1][j], dp[i][j-1])+grid[i][j]
#如果有障碍物，就是不能走的格子，就在循环里加一个if判断
# 假设 obstacleGrid[i][j] == 1 代表有障碍
for i in range(1, m):
    for j in range(1, n):
        if obstacleGrid[i][j] == 1:
            dp[i][j] = 0  # 走不通
        else:
            dp[i][j] = dp[i-1][j] + dp[i][j-1]

#多状态dp（需要创建多个dp数组）

#股票买卖问题：有股票才能卖，手里没有股票才能买
#今天的状态只与昨天的状态相关，所以可以只维护两个dp值hold和cash(hold表示今天持有股状态下的现有的钱，cash表示今天没有股状态下现有的钱)，今天持有股票hold[i]有两个来源：昨天持有今天不动:hold[i-1]，昨天没有今天刚买:cash[i-1]-price；今天不持有股票cash[i]有两个来源：昨天没股今天不动cash[i-1]和昨天有股今天卖了hold[i-1]+price，两个都要取最大值
def maxProfit(prices):
    if not prices: return 0
    n = len(prices)
    # 初始化
    # hold: 第一天买了，钱变成负数
    # cash: 第一天没买，钱是0
    hold = -prices[0]
    cash = 0
    for i in range(1, n):
        # 存一下昨天的状态，因为今天计算要用到
        prev_hold = hold
        prev_cash = cash
        # 今天的持有 = max(昨天持有, 昨天没持有 - 今天股价)
        hold = max(prev_hold, prev_cash - prices[i])
        # 今天的现金 = max(昨天现金, 昨天持有 + 今天股价)
        cash = max(prev_cash, prev_hold + prices[i])   
    return cash # 最后肯定是手里没股票赚得多

#乘积最大子数组：求子数组中所有元素乘起来最大的，同时记录当前的最大值和最小值
def maxProduct(nums):
    if not nums: return 0
    # 结果变量
    res = nums[0]
    # 当前的最大和最小（相当于两个dp数组的当前值）
    cur_max = nums[0]
    cur_min = nums[0]
    for i in range(1, len(nums)):
        num = nums[i]
        # 因为 cur_max 在下一行会被修改，所以先存个临时变量
        prev_max = cur_max
        # 核心公式：一定要把 num 自己也带上比较（因为可能之前都是0，从自己开始算）
        cur_max = max(num, prev_max * num, cur_min * num)
        cur_min = min(num, prev_max * num, cur_min * num)
        res = max(res, cur_max)
    return res

#粉刷房子：每个房子可以选择红蓝绿三种颜色，但是相邻的房子不能涂一样的颜色（有多个选择的维度）
def minCost(costs):
    if not costs: return 0
    n = len(costs)
    # 1. 定义 dp 数组，大小和 costs 一样
    dp = [[0] * 3 for _ in range(n)]
    # 2. 初始化第一间房子（刷什么色就是什么价）
    dp[0][0] = costs[0][0]
    dp[0][1] = costs[0][1]
    dp[0][2] = costs[0][2]
    # 3. 从第二间房子开始遍历
    for i in range(1, n):
        # 如果这间刷 0(红)，上间只能是 1(蓝) 或 2(绿)
        dp[i][0] = costs[i][0] + min(dp[i-1][1], dp[i-1][2])
        # 如果这间刷 1(蓝)，上间只能是 0(红) 或 2(绿)
        dp[i][1] = costs[i][1] + min(dp[i-1][0], dp[i-1][2])
        # 如果这间刷 2(绿)，上间只能是 0(红) 或 1(蓝)
        dp[i][2] = costs[i][2] + min(dp[i-1][0], dp[i-1][1])
    # 4. 最后看哪种颜色的总路径最便宜
    return min(dp[n-1][0], dp[n-1][1], dp[n-1][2])
```

### 递归常规思路：

```python
# ==================== 回溯模板 ====================
def backtrack_template(path, choices):
    # 1. 终止条件
    if meet_condition(path):
        result.append(path.copy())  # 注意要复制
        return
    # 2. 遍历所有选择
    for choice in choices:
        # 剪枝：跳过无效选择
        if not is_valid(path, choice):
            continue
        # 做出选择
        path.append(choice)
        # 更新可选范围（避免重复）
        new_choices = update_choices(choices, choice) 
        # 递归探索
        backtrack_template(path, new_choices)
        # 撤销选择（关键！）
        path.pop()
        
#马走日类型（求路径数）：
directions = [(-2, 1), (-2, -1), (2, 1), (2, -1), (-1, 2), (-1, -2), (1, 2), (1, -2)]
def dfs(x, y, steps):  #如果是求最大权值，那么需要设置一个全局变量global max_v
    					#如果报CE则在第一行加上pylint:skip-file
    if steps == m*n:    #if 到达终点 and steps == k:
        				#	return True
            			#return False
        return 1
    res = 0
    for dx, dy in directions:
        nx, ny = x+dx, y+dy
        if 0 <= nx < n and 0 <= ny < m and not visited[nx][ny]:
            visited[nx][ny] = True
            res += dfs(nx, ny, steps+1) #if dfs(nx, ny, steps+1):
            							#	return True
            visited[nx][ny] = False
    return res							#return False
n, m, x, y = map(int, input().split())
visited = [[False]*m for _ in range(n)]
visited[x][y] = True #标记起始点
print(dfs(x, y, 1)) #如果限定条件是固定步数的就是steps初始值为0

# 全排列
def permute(nums):
    result = []
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if not used[i]:
                used[i] = True
                path.append(nums[i])
                backtrack(path, used)
                path.pop()
                used[i] = False
    backtrack([], [False]*len(nums))
    return result

# 组合（选k个）
def combine(n, k):
    result = []
    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return
        for i in range(start, n+1):
            path.append(i)
            backtrack(i+1, path)
            path.pop()
    backtrack(1, [])
    return result

# 子集
def subsets(nums):
    result = []
    def backtrack(start, path):
        result.append(path[:])  # 所有路径都是解
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i+1, path)
            path.pop()
    backtrack(0, [])
    return result

#岛屿数量问题
def numIslands(grid):
    if not grid: return 0
    R, C = len(grid), len(grid[0])
    count = 0
    # === DFS 函数：负责淹没岛屿 ===
    def dfs(r, c):
        # 越界或者已经是水，直接回头
        if not (0 <= r < R and 0 <= c < C and grid[r][c] == '1'):
            return
        grid[r][c] = '0' # 1. 标记已访问（沉岛）
        # 2. 向四周扩散
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)
    # === 主循环 ===
    for i in range(R):
        for j in range(C):
            if grid[i][j] == '1': # 发现新大陆
                count += 1
                dfs(i, j) # 派DFS把这个岛全淹了
    return count

# 归并排序
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge_two_sorted_lists(left, right)

# ==================== 记忆化递归模板 ====================
def memoized_recursion(params, memo={}):
    if base_case(params):
        return base_value
    
    # 检查是否已计算
    if params in memo:
        return memo[params]
    
    # 计算并存储
    result = compute(params, memoized_recursion)
    memo[params] = result
    
    return result

# 斐波那契数列（记忆化）
def fibonacci(n, memo={}):
    if n <= 1:
        return n
    if n in memo:
        return memo[n]
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]

#递归优化两板斧
import sys
sys.setrecursionlimit(1 << 30) #将递归深度限制设置为 2^30

from functools import lru_cache
@lru_cache(maxsize=None) #可以缓存函数的返回值，避免计算相同的子问题，减少耗时，                           不用手动记忆化
#下面接递归函数程序

#波兰表达式
data = input().split()
data_iter = iter(data) #使用iter形成迭代器
def Poland():
    try:
        token = next(data_iter) #从迭代器中取出一个值，并把指针自动后挪一        位,next(iteration, default_value),当迭代器中没有值了之后会输出设置好的default_value而不会报错。
    except StopIteration: #当迭代器中没有元素了之后会报错StopIteration
        return 0
    if token == '+':
        return Poland() + Poland()
    elif token == '-':
        return Poland() - Poland()
    elif token == '*':
        return Poland() * Poland()
    elif token == '/':
        return Poland() / Poland()
    else:
        return float(token)
result = Poland()
print(f"{result:.6f}")
```

### bfs:

```python
#bfs（最短路径）
from collections import deque
  
def bfs(start, end):    
    q = deque([(0, start)])  # (step, start)
    in_q = {start}


    while q:
        step, front = q.popleft() # 取出队首元素
        if front == end:
            return step # 返回需要的结果，如：步长、路径等信息

        # 将 front 的下一层结点中未曾入队的结点全部入队q，并加入集合in_queue设置为已入队
        for neighbor in blahblah:
            if neighbor not in in_q:
                in_q.add(neighbor)
                q.append(neighbor)

#dfs的deque写法
def dfs(start, end):
    visited = set()
    stack = [start]
    result = []
    while stack:
        node = stack.pop()
       	if node not in visited:
            visited.add(node)
            result.append(node)
            
            for neighbor in reversed(blahblah): #将邻居逆序入栈
            	if neighbor not in visited:
                    stack.append(neighbor)
    return result
```

#### 并查集

```python
#宗教信仰：每行给出一组数据代表这两个编号的同学信仰同一种宗教，问学校里的宗教上限
import sys
# 增加递归深度，防止N很大时递归报错
sys.setrecursionlimit(100000)

class UnionFind:
    def __init__(self, n):
        # 初始化：parent[i] = i，代表一开始大家的老大都是自己
        # 下标从0到n，虽然0不用，但方便直接对应学生ID
        self.parent = list(range(n + 1))
        # 记录当前的集合数量（也就是宗教数的上限）
        self.count = n
    
    def find(self, x):
        # 如果x的老大就是自己，那x就是根节点
        if self.parent[x] == x:
            return x
        
        # 路径压缩（优化）：
        # 在找老大的过程中，直接把沿途所有人的老大都改成终极老大
        # 这样下次再找就快了
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        # 找到x的终极老大
        root_x = self.find(x)
        # 找到y的终极老大
        root_y = self.find(y)
        
        # 如果老大不一样，说明是两个不同的集合，需要合并
        if root_x != root_y:
            self.parent[root_x] = root_y # 让x的老大归顺y的老大
            self.count -= 1 # 集合数量减少1
            return True # 合并成功
        else:
            return False # 已经是同一个集合，无需合并

def solve():
    # 使用 sys.stdin 读取以提高速度，这道题数据量较大
    input_data = sys.stdin.read().split()
    
    if not input_data:
        return

    iterator = iter(input_data)
    case_num = 1
    
    while True:
        try:
            # 读取 n 和 m
            n = int(next(iterator))
            m = int(next(iterator))
            
            # 结束条件
            if n == 0 and m == 0:
                break
            
            # 初始化并查集，初始有 n 个宗教
            uf = UnionFind(n)
            
            # 处理 m 对关系
            for _ in range(m):
                i = int(next(iterator))
                j = int(next(iterator))
                uf.union(i, j)
            
            # 输出结果
            print(f"Case {case_num}: {uf.count}")
            case_num += 1
            
        except StopIteration:
            break

if __name__ == "__main__":
    solve()
```

### 常见语法：

不定行输入：

```python
import sys

# 方法1：逐行读取
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue  # 跳过空行
    # 处理line

# 方法2：一次性读取所有
data = sys.stdin.read().strip().split()  # 所有内容分割成列表
# 然后根据格式解析 #split('\n')代表按行分割

# 方法3：读取到列表
lines = sys.stdin.readlines()
lines = [line.strip() for line in lines]

# 方法4：try-except处理（交互式）
while True:
    try:
        line = input().strip()
        # 处理line
    except EOFError:
        break
```

格式化输出：

```python
# 基础输出
print("Hello")                    # 自动换行
print(1, 2, 3, sep=',', end='')  # 自定义分隔符和结尾

# f-string（推荐）
name = "Alice"
age = 20
print(f"{name} is {age} years old")
print(f"Pi: {3.14159:.2f}")  print(f"{num:.2f}")     # 保留2位小数
print(f"Number: {10:05d}")        # 宽度5，前面补0
```

strip（）：

```python
s = "  hello world  \n"

# 移除两端空白
s.strip()        # "hello world"
s.lstrip()       # "hello world  \n"（只去左边）
s.rstrip()       # "  hello world"（只去右边）

# 移除指定字符
"***hello***".strip('*')  # "hello"

# 重要：input()后面通常加strip()
user_input = input().strip()  # 去掉输入的前后空格
```

列表：

```python
lst = [1, 2, 3, 4, 5]
# 基本操作
lst.append(6)           # 末尾添加
lst.insert(0, 0)        # 指定位置插入
lst.pop()               # 删除末尾
lst.pop(0)              # 删除指定位置
lst.remove(3)           # 删除第一个匹配值
del lst[1:3]            # 删除切片
lst.clear()             # 清空
# 查询
if 3 in lst:            # 存在判断
    idx = lst.index(3)  # 查找索引
    cnt = lst.count(3)  # 计数
# 切片
lst[::2]      # 步长2
# 列表推导式
[x**2 for x in range(5)]                      # [0, 1, 4, 9, 16]
[x for x in range(10) if x % 2 == 0]          # 偶数
[(x, y) for x in range(3) for y in range(2)]  # 嵌套循环
```

string：

```python
s = "Hello World"

# 常用方法
s.upper()           # 大写
s.lower()           # 小写
s.title()           # 单词首字母大写
s.strip()           # 去两端空白
s.replace('H', 'J') # 替换
s.split()           # 分割成列表
','.join(['a','b']) # 连接
s.find('World') s.rfind('World') s.index('a') s.rindex('a')  # 查找索引
#带r的是从后往前找，find找不到返回-1，index找不到引发异常
s.count('l')        # 统计出现次数
s.startswith('He')  # 判断开头
s.endswith('ld')    # 判断结尾

# 检查类型
s.isdigit()     # 是否全数字
s.isalpha()     # 是否全字母
s.isalnum()     # 是否字母或数字
```

dict：

```python
d = {'a': 1, 'b': 2, 'c': 3}

# 增删改查
d['d'] = 4                      # 添加/修改
value = d.get('a', 0)           # 安全获取
value = d.setdefault('e', 5)    # 不存在则设置默认
del d['a']                      # 删除
d.pop('b')                      # 删除并返回值
d.clear()                       # 清空
# 字典推导式
{x: x**2 for x in range(5)}            # {0:0, 1:1, 4:4, 9:9, 16:16}
{v: k for k, v in d.items()}           # 交换键值
```

集合：

```python
s1 = {1, 2, 3}
s2 = {3, 4, 5}

# 基本操作
s1.add(4)           # 添加
s1.remove(1)        # 删除（不存在报错）
s1.discard(1)       # 删除（不报错）
s1.pop()            # 删除并返回任意元素
s1.clear()          # 清空

# 集合运算
s1 | s2         # 并集 {1,2,3,4,5}
s1 & s2         # 交集 {3}
s1 - s2         # 差集 {1,2}
s1 ^ s2         # 对称差集 {1,2,4,5}

# 集合推导式
{x for x in range(10) if x % 2 == 0}  # 偶数集合
```

元组：

```python
t = (1, 2, 3)       # 创建
t[0]               # 访问
a, b, c = t        # 解包
```

math：

```python
import math
math.pow(2, 3)       # 8.0
math.gcd(12, 18)     # 6（最大公约数）
math.lcm(4, 6)       # 12（最小公倍数，Python 3.9+）
math.exp(x)          # e的x次幂
math.log(x, base)    # 以base为底的x的对数，不填base就是以e为底
math.pi              # 3.14159...
math.e               # 2.71828...
```

一行代码：

```python
# 交换两个变量
a, b = b, a

# 列表扁平化
nested = [[1,2], [3,4]]
flat = [item for sublist in nested for item in sublist]  # [1,2,3,4]

# 条件赋值
x = 10
y = 5 if x > 0 else 0

# 矩阵转置
matrix = [[1,2], [3,4]]
transpose = list(zip(*matrix))  # [(1,3), (2,4)]

# 统计频率
from collections import Counter
freq = Counter("hello")  # Counter({'h':1, 'e':1, 'l':2, 'o':1})

# 删除列表中所有特定值
lst = [1, 2, 3, 2, 4]
lst = [x for x in lst if x != 2]  # [1, 3, 4]
```

内置函数：

```python
pow(2, 3)               # 8
divmod(10, 3)           # (3, 1)（商和余数）
round(3.14159, 2)       # 3.14
reversed([1,2,3])       # 反转迭代器
zip([1,2], ['a','b'])   # [(1,'a'), (2,'b')]
any([True, False])      # True
all([True, True])       # True
enumerate(['a','b'])    # (0,'a'), (1,'b')
filter(lambda x:x>0, [-1,0,1])  # [1]
map(lambda x:x*2, [1,2,3])      # [2,4,6]
```

ASCII码：

大写字母A到Z是65到90；

小写字母a到z是97到122

ord() 是用来求字符的ASCII码的，chr() 是用来找已知ASCII码对应的字符的。

```python
#Counter函数
from collections import Counter
nums = [1, 1, 1, 6, 6, 6, 7, 8]
count = Counter(nums) #count为一个字典，key为可迭代对象（如string,list,tuple）里的元素，value为元素出现次数
for k, v in count.items():
    print(k, v)
#如果key重复会报错，查找不存在的元素时会返回0
#elements方法
c = Counter({'a':1, 'b':2, 'c':3})
c2 = Counter({'a':0, 'b':-1, 'c':3})
print(list(c.elements())) #>>['a', 'b', 'b', 'c', 'c', 'c'] 这里如果用tuple输出的就是一个元组
print(list(c2.elements())) #>>['c', 'c', 'c']
```

```python
#进制转换
bin() #十进制转化为二进制，结果以'0b'为前缀
oct() #十进制转化为八进制，结果以'0c'为前缀
hex() #十进制转化为十六进制，结果以'0x'为前缀
int(num, n) #将n进制的数num转化为十进制
#递归写法
def to_str(n, base):
    # 定义用于转换的字符序列
    convert_string = "0123456789ABCDEF" #最多能转换成十六进制
    # 基准情形：如果 n 小于基数，则直接返回对应的字符
    if n < base:
        return convert_string[n]
    else:
        # 递归调用：先处理商，再处理余数
        # 通过延迟连接操作，确保结果的顺序是正确的
        return to_str(n // base, base) + convert_string[n % base]
```

```python
#埃氏筛
def eratosthenes_sieve(n):
    if n < 2:
        return []
    # 初始化标记数组，True表示是素数
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    # 从2开始筛选
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            # 将i的倍数标记为非素数
            # 从i*i开始，因为小于i*i的合数已经被更小的素数筛掉了
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    # 收集所有素数
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    return primes
#欧氏筛（线性筛）
def euler_sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    primes = []  # 存放所有素数
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
        # 用当前已找到的素数 primes[j] 去筛掉合数 i * primes[j]
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            # 关键步骤：保证每个合数只被它的最小质因子筛掉
            if i % p == 0:
                break
    return primes
```

卡特兰数：1,1,2,5,14,42,132（对应n=0,1,2,3,4,5,6）

``` python
import math
def catalan_formula(n):
    return math.comb(2*n,n)//(n+1) #comb(n,k)是求组合数C(n,k)

h(n) = h(0)h(n-1)+h(1)h(n-2)+...+h(n-1)h(0)
def catalan_dp(n):
    if n == 0:return 1
    dp = [0] * (n+1)
    dp[0] = 1
    for i in range(1, n+1):
        for j in range(i):
            dp[i] += dp[j]+dp[i-1-j]
  	return dp[n]  

#进出栈序列问题：有n个元素一次入栈，求有多少种合法的出栈序列：catalan(n)
#括号匹配问题，有n对括号，求有多少种合法的排列方式：catalan(n)
#二叉树的结构数量：n个节点，能构成多少种不同结构的二叉树：catalan(n)
#n*n的网格中，从左下角走到右上角，只能向上或向右，且路径不能越过对角线:catalan(n)
```



