# Assignment #C: bfs & dp

Updated 1436 GMT+8 Nov 25, 2025

2025 fall, Complied by <mark>魏铭泽 元培学院</mark>



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



## 1. 题目

### sy321迷宫最短路径

bfs, https://sunnywhy.com/sfbj/8/2/321

思路：

思路本身没有难点，重要的在于向q添加route的时候需要注意复制一份（被这个卡了好久）

代码：

```python
from collections import deque
directions=[(-1,0),(1,0),(0,-1),(0,1)]
def bfs(maze,m,n):
    q=deque([([[1,1]],(1,1))])
    inq=set()
    inq.add((1,1))
    while q:
        route,(fx,fy)=q.popleft()
        if fx==n and fy==m:
            return route
        for dx,dy in directions:
            nx,ny=fx+dx,fy+dy
            if (nx,ny) not in inq and maze[nx][ny]==0:
                inq.add((nx,ny))
                route.append([nx,ny])
                q.append((route[:],(nx,ny)))
                route.pop()
n,m=map(int,input().split())
maze=[]
for _ in range(n):
    maze.append([1]+list(map(int,input().split()))+[1])
maze=[[1]*(m+2)]+maze+[[1]*(m+2)]
res=bfs(maze,m,n)
for i in range(len(res)):
    print(*res[i])
```

耗时：30min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251127203222648.png)



### sy324多终点迷宫问题

bfs, https://sunnywhy.com/sfbj/8/2/324

思路：

对每一个终点都进行一次bfs会超时，所以说需要采用dp的思想。

状态转移方程：`res[nx][ny]=res[fx][fy]+1`，即与(fx,fy)相邻的平地的最少步数就为(fx,fy)的最小步数+1，同时采用res这个二维数组代替之前普通的迷宫问题中的inq集合，(x,y)点的值为-1代表没有入队，不为-1代表已经入队，初始将起点(0,0)入队，也就是将其在res中的值改为0（因为从起点到起点需要0步），满足条件的(nx,ny)会根据上面的状态转移方程更改其在res中的值，使得其被标记为已入队。其他思路和普通的迷宫问题一样了。

代码：

```python
from collections import deque
directions=[(-1,0),(1,0),(0,-1),(0,1)]
def bfs(maze,n,m):
    q=deque([(0,0)])
    res=[[-1]*m for _ in range(n)]
    res[0][0]=0
    while q:
        (fx,fy)=q.popleft()
        for dx,dy in directions:
            nx,ny=fx+dx,fy+dy
            if 0<=nx<=n-1 and 0<=ny<=m-1 and res[nx][ny]==-1 and maze[nx][ny]==0:
                res[nx][ny]=res[fx][fy]+1
                q.append((nx,ny))
    return res
n,m=map(int,input().split())
maze=[list(map(int,input().split())) for _ in range(n)]
res=bfs(maze,n,m)
for i in range(n):
    print(*res[i])
```

耗时：25min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251127203154186](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251127203154186.png)



### M02945: 拦截导弹

dp, greedy http://cs101.openjudge.cn/pctbook/M02945

思路：

这个题是讲义中原题，看过答案后自己又写了一遍，思路如下：

首先将dp数组全设置为inf，对于height中的每一个h，用bisect_right函数找比它更大的数的index，然后将这个h插入到这个位置。这样就可以得到一个非严格单调递增的数组，就是我们想要的数组。

还可以不使用bisect函数，令dp[i]为以height[i]结尾的最长子序列的长度，那么dp数组最初始应该全部设为1（因为以该元素结尾的子序列一定有这个元素本身构成的序列，长度为1），所以状态转移方程为dp[i]=max(dp[j]+1,dp[i])，其中j应该遍历所有比i小的数，寻找其中height[j]>=height[i]的元素，那么就可以在以height[j]为结尾的子序列中再加一个height[i]。最后取dp中的最大值即可。

代码：

```python
from bisect import bisect_right
k=int(input())
height=list(map(int,input().split()))
height=height[::-1]
dp=[float('inf')]*k
for h in height:
    dp[bisect_right(dp,h)]=h
print(bisect_right(dp,1e9))
```

耗时：10min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251125200334107](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251125200334107.png)



### 189A. Cut Ribbon

brute force/dp, 1300, https://codeforces.com/problemset/problem/189/A

思路：

转化为完全背包问题：背包总容量为n，有重量分别为a,b,c，价值均为1的商品可以拿，刚好装满的最大价值是多少

状态转移方程为：当i<j时：`dp[i]=dp[i]`，当i>=j时，`dp[i]=max(dp[i-j]+1,dp[i])`，其中i为当前背包总容量，j为商品的重量。

注意需要将初始dp数组全设为负无穷，表示一种不可能的情况，为初始状态。

代码：

```python
n,a,b,c=map(int,input().split())
from math import inf
dp=[0]+[-float(inf)]*n
for i in range(1,n+1):
    for j in (a,b,c):
        if i>=j:
            dp[i]=max(dp[i-j]+1,dp[i])
print(dp[n])
```

耗时：10min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251125160805516](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251125160805516.png)





### M01384: Piggy-Bank

dp, http://cs101.openjudge.cn/practice/01384/

思路：

完全背包，需要注意遍历i时从weight[j]开始，否则会超时。

代码：

```python
from math import inf
t=int(input())
for _ in range(t):
    e,f=map(int,input().split())
    n=int(input())
    price,weight=[],[]
    for _ in range(n):
        p,w=map(int,input().split())
        price.append(p)
        weight.append(w)
    dp=[0]+[float(inf)]*(f-e)
    for j in range(n):
        for i in range(weight[j],f-e+1):
            dp[i]=min(dp[i-weight[j]]+price[j],dp[i])
    if dp[-1]==float('inf'):
        print('This is impossible.')
    else:
        print(f'The minimum amount of money in the piggy-bank is {dp[-1]}.')
```

耗时：30min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251125164724928](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251125164724928.png)



### M02766: 最大子矩阵

dp, kadane, http://cs101.openjudge.cn/pctbook/M02766

思路：

这道题是讲义上的例题，在上周看讲义的时候就看懂了这道题，这周写作业的时候又自己写了一遍。思路如下：

先写出kadane函数，用max_end_here记录在此处停止的最大子序列的值，用max_so_far记录目前已经算过的子序列中的最大值，当遍历到一个新的数num时，如果max_end_here小于0（也就是max_end_here+num小于num，则在num处停止的最大子序列的值就为num，反之为max_end_here+num，再更新max_so_far的值，最后return max_so_far即可。

先固定子矩阵的最左边一列，使用tmp数组记录子矩阵的值，其中tmp[i]表示子矩阵行数为n时的值，右边一列遍历剩下的列，再遍历所有的行，求出tmp数组，再使用kadane函数求出tmp的最大子序列的值，与已有的最大值maximum进行比较更新maximum。

这个题处理数据也很有趣，我这里是直接使用sys.stdin.read将数据储存在列表当中，而后遍历这个列表，元素的下标i//n就代表这个元素在矩阵的第几行，i%n就代表这个元素在该行的第几列。

代码：

```python
def max_subarray(matrix):
    def kadane(lst):
        max_end_here=max_so_far=lst[0]
        for num in lst[1:]:
            max_end_here=max(num,max_end_here+num)
            max_so_far=max(max_so_far,max_end_here)
        return max_so_far
    n=len(matrix)
    maximum=-float('inf')
    for left in range(n):
        tmp=[0]*n
        for right in range(left,n):
            for row in range(n):
                tmp[row]+=matrix[row][right]
            maximum=max(maximum,kadane(tmp))
    return maximum
n=int(input())
import sys
data=sys.stdin.read().split()
matrix=[[0]*n for _ in range(n)]
for i in range(n*n):
    matrix[i//n][i%n]=int(data[i])
print(max_subarray(matrix))
```

耗时：30min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251125194308936](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251125194308936.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2024fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

额外练习题目：

讲义上的bfs和dp的练习题目

学习收获：
目前做题的感受是bfs全是套模板的题，相比较dp和递归还是简单了不少

dp的多重背包类型题还是没有弄明白，还有点不太会二进制优化，争取这两天学会。



