# Assignment #6: 矩阵、贪心

Updated 1432 GMT+8 Oct 14, 2025

2025 fall, Complied by <mark>魏铭泽 元培学院</mark>



>**说明：**
>
>1. **解题与记录：**
>
>  对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
>2. 提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的本人头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
> 
>4. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。  
>
>请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。





## 1. 题目

### M18211: 军备竞赛

greedy, two pointers, http://cs101.openjudge.cn/pctbook/M18211



思路：

最有性价比的做法就是买最便宜的卖最贵的，所以说先把所有武器按价格从小到大排序，能买得起就买，买不起就从后往前卖。

代码

```python
# 
p=int(input())
cost=[int(x) for x in input().split()]
cost.sort()
num,left,right=0,0,len(cost)-1
while left<=right:
    if cost[left]<=p:
        num+=1
        p-=cost[left]
        left+=1
    else:
        if left==right:
            break
        p+=cost[right]
        right-=1
        num-=1
        if num<0:
            num=0
            break
print(num)
```

耗时：1h30min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251014172703398](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251014172703398.png)



### M21554: 排队做实验

greedy, http://cs101.openjudge.cn/pctbook/M21554/



思路：



代码

```python
n=int(input())
time=[int(x) for x in input().split()]
time1=sorted(time)
for i in range(n):
    print(time.index(time1[i])+1,end=' ')
    time[time.index(time1[i])]=-1
print('')
total=0
for i in range(n):
    total+=time1[i]*(n-i-1)
print('%.2f'%(total/n))
```

耗时：30min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251014175939790](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251014175939790.png)



### E23555: 节省存储的矩阵乘法

implementation, matrices, http://cs101.openjudge.cn/pctbook/E23555



思路：



代码

```python
n,m1,m2=map(int,input().split())
X,Y=[[0]*n for _ in range(n)],[[0]*n for _ in range(n)]
for _ in range(m1):
    r,c,e=map(int,input().split())
    X[r][c]=e
for _ in range(m2):
    r,c,e=map(int,input().split())
    Y[r][c]=e
XY=[[] for _ in range(n)]
for i in range(n):
    for j in range(n):
        total=0
        for k in range(n):
            total+=X[i][k]*Y[k][j]
        XY[i].append(total)
for i in range(n):
    for j in range(n):
        if XY[i][j]!=0:
            print(i,j,XY[i][j])
```

耗时：20min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251014162641795](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251014162641795.png)



### M12558: 岛屿周⻓

matices, http://cs101.openjudge.cn/pctbook/M12558


思路：

先加保护圈确保后面遍历不会越界，然后检查每个为1的元素的上下左右有几个0，就代表这块地有几个边界。

代码

```python
# 
n,m=map(int,input().split())
board=[[0]*(m+2)]
for _ in range(n):
    board.append([0]+[int(x) for x in input().split()]+[0])
board.append([0]*(m+2))
total=0
for i in range(1,n+1):
    for j in range(1,m+1):
        if board[i][j]==1:
            total+=4-(board[i-1][j]+board[i+1][j]+board[i][j-1]+board[i][j+1])
print(total)
```

耗时：30min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251014230153979](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251014230153979.png)



### M01328: Radar Installation

greedy, http://cs101.openjudge.cn/practice/01328/



思路：

先把所有的岛屿在x轴上的投影存储下来，然后按右边的点从小到大排列，如果某个岛屿没被前面设下的雷达覆盖到，就在它右边的点设置一个新雷达。

代码

```python
case_number=0
while True:
    n, d = map(int, input().split())
    if n == d == 0:
        break
    case_number += 1
    islands = []
    cnt = 0
    num = 0
    for _ in range(n):
        num+=1
        x, y = map(int, input().split())
        if y > d:
            cnt = -1
            for _ in range(n-num):
                input()
            break
        islands.append((x, y))
    if cnt != -1:
        areas = []
        from math import sqrt,inf

        for x, y in islands:
            areas.append([x - sqrt(d**2 - y ** 2), x + sqrt(d**2 - y ** 2)])
        areas.sort(key=lambda p:p[1])
        last_radar = float(-inf)
        for item in areas:
            if item[0]>last_radar:
                last_radar=item[1]
                cnt+=1
    print(f"Case {case_number}: {cnt}")
    input()
```

耗时：2h

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251016161507031](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251016161507031.png)



### 545C. Woodcutters

dp, greedy, 1500, https://codeforces.com/problemset/problem/545/C



思路：

第一棵树一定向左倒，第二棵树一定向右倒，中间的树如果能向左倒就向左倒，如果能向右倒就向右，都不能就不倒。(尝试练习了一下deque的用法)

代码

```python
n=int(input())
cnt=0
from collections import deque
from math import inf
trees=deque(maxlen=1)
trees.extend([(-inf,0,'left')])
for _ in range(n-1):
    last = trees.popleft()
    x,h=map(int,input().split())
    trees.extend([(x,h,'')])
    if last[0]+last[1]<x and last[2]!='left':
        cnt+=1
        if x-h>last[0]+last[1]:
            cnt+=1
            trees.extend([(x,h,'left')])
    else:
        if x-h>last[0]:
            cnt+=1
            trees.extend([(x,h,'left')])
x,h=map(int,input().split())
last=trees.popleft()
if last[2]!='left' and last[0]+last[1]<x:
    cnt+=2
else:
    cnt+=1
print(cnt)
```

耗时：1h

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251016181120188](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251016181120188.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2025fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

额外练习题目：这周时间有点少，没有额外练习题目orz

学习收获：

1.（可能）学会了双指针的写法，体会到双指针在处理有序序列的问题时的优越性。

2.了解到了bfs和dfs，大概知道bfs是找最短路径的（遍历每一条路径），而dfs是找环的（如果遇到端点就不遍历了，如果没遇到就一直遍历到成环为止），但是认知还是很浅显而且不太会写orz

3.了解到了deque和栈，知道deque在处理需要在序列两端频繁操作的问题时的优越性。

4.了解到了保护圈，加上保护圈可以确保遍历不越界，可以让下面的代码少一些if，elif的论述。

5.学会了print(f"")的用法，可以在引号中加{}来调用变量，调用函数，表达式求值，格式化控制(如{pi:.2f})#3.14)