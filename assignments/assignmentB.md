# Assignment #B: dp

Updated 1448 GMT+8 Nov 18, 2025

2025 fall, Complied by <mark>魏铭泽 元培学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### LuoguP1255 数楼梯

dp, bfs, https://www.luogu.com.cn/problem/P1255

思路：

状态转移方程：`dp[i]=dp[i-1]+dp[i-2]`

代码：

```python
N=int(input())
dp=[1]*(N+1)
for i in range(2,N+1):
    dp[i]=dp[i-1]+dp[i-2]
print(dp[N])
```

耗时：2min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251120163635620](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251120163635620.png)



### 27528: 跳台阶

dp, http://cs101.openjudge.cn/practice/27528/

思路：

状态转移方程：`dp[i]=sum(dp[j])`其中j<i。因为跳n级台阶，可以跳n-1阶台阶再跳1级，可以跳n-2级台阶再跳2级（跳n-2级台阶再跳1级再跳1级的情况与跳n-1级台阶中最后一步是跳1级的情况重复了），依次类推，所以应该等于前面所有的和，而且我们将dp[0]设置成1，这样一步跳上去的情况就也包含在内了。

代码：

```python
N=int(input())
dp=[1]+[0]*N
for i in range(1,N+1):
    for j in range(i):
        dp[i]+=dp[j]
print(dp[-1])
```

耗时：30min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251120172516172](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251120172516172.png)



### M23421:《算法图解》小偷背包问题

dp, http://cs101.openjudge.cn/pctbook/M23421/

思路：

二维数组的状态转移方程为`dp[i][j]=max(dp[i-1][j],dp[i-1][j-weights[i]]+values[i]`其中i代表目前有几种物品可以偷，j代表目前的最大容量。如果想要使用滚动数组，那么dp数组就应该为目前有i种物品可以偷的状态下不同最大容量可以得到的最大价值（因为i+1状态下的价值只与i状态下的价值相关），而且需要让dp数组不能被污染，所以说要从后往前遍历，先填大的，而下界是因为当j<当前物品重量时肯定不能偷当前物品，所以说最大价值不变，所以说只需要遍历到weights[i]-1。

代码：

```python
n,b=map(int,input().split())
values=list(map(int,input().split()))
weights=list(map(int,input().split()))
dp=[0]*(b+1)
for i in range(n):
    for j in range(b,weights[i]-1,-1):
        dp[j]=max(dp[j],dp[j-weights[i]]+values[i])
print(dp[-1])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251120203058663](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251120203058663.png)



### M5.最长回文子串

dp, two pointers, string, https://leetcode.cn/problems/longest-palindromic-substring/

思路：

状态转移方程：`dp[i][j]=s[i]==s[j] and dp[i+1][j-1]`，其中`dp[i][j]`为`s[i:j+1]`是否为回文子串。

代码：

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n=len(s)
        if n==1:
            return s
        dp=[[False]*n for _ in range(n)]
        for j in range(n):
            for i in range(j+1):
                if s[i]==s[j] and (j-i<=1 or dp[i+1][j-1]):
                    dp[i][j]=True
        start_index=0
        max_len=0
        for i in range(n):
            for j in range(i,n):
                if dp[i][j] and (j-i+1)>max_len:
                    max_len=j-i+1
                    start_index=i
        return s[start_index:start_index+max_len]
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251120210801444](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251120210801444.png)





### 474D. Flowers

dp, 1700 https://codeforces.com/problemset/problem/474/D

思路：

状态转移方程：`dp[i]=dp[i-1]+dp[i-k]`,dp[i-1]表示的是有i-1朵花时的方案数，可以直接在后面加上一朵红花，dp[i-k]是有i-k朵花时的方案数，可以在后面加上k朵白花。

需要注意的是要提前把前缀和算出来否则会超时。

代码：

```python
t,k=map(int,input().split())
dp=[0]*(int(1e5)+1)
s=[0]*(int(1e5)+1)
dp[0]=s[0]=1
for i in range(1,int(1e5)+1):
    if i<k:
        dp[i]=1
    else:
        dp[i]=(dp[i-1]+dp[i-k])%(int(1e9)+7)
for i in range(1,int(1e5)+1):
    s[i]=(s[i-1]+dp[i])%(int(1e9)+7)
for _ in range(t):
    a,b=map(int,input().split())
    print((s[b]-s[a-1])%(int(1e9)+7))
```

耗时：30min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251120214534412](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251120214534412.png)



### M198.打家劫舍

dp, https://leetcode.cn/problems/house-robber/

思路：

状态转移方程分两种情况：1.dp[i-1]取最大值时用到了nums[i-1]这个元素，则`dp[i]=max(dp[i-1],dp[i-2]+nums[i])`；2.dp[i-1]取最大值时没用到nums[i-1]这个元素，则`dp[i]=dp[i-2]+nums[i]`。

代码：

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n=len(nums)
        if n==1:
            return nums[0]
        dp=[0]*n
        s=[False]*n
        dp[0]=nums[0]
        if nums[0]<=nums[1]:
            dp[1]=nums[1]
            s[1]=True
        else:
            dp[1]=nums[0]
        for i in range(2,n):
            if s[i-1]:
                if dp[i-1]<=dp[i-2]+nums[i]:
                    dp[i]=dp[i-2]+nums[i]
                    s[i]=True
                else:
                    dp[i]=dp[i-1]
            else:
                dp[i]=dp[i-2]+nums[i]
                s[i]=True
        return dp[-1]
```

耗时：1h

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251120230256315](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251120230256315.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

额外练习题目：讲义中的部分练习题

学习收获：

学到了dp题目最重要的就是找状态转移方程，感觉有点像高中数学数列中的找递推公式，但是经常会找不到状态转移方程，还需要多加练习。

在Flowers那道题中了解到了前缀和这个概念。



