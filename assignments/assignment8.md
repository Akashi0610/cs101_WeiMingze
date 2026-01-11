# Assignment #8: 递归

Updated 1315 GMT+8 Oct 21, 2025

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

### M04147汉诺塔问题(Tower of Hanoi)

dfs, http://cs101.openjudge.cn/pctbook/M04147

思路：



代码

```python
info=input().split()
num,x,y,z=int(info[0]),info[1],info[2],info[3]
def Hanoi(n,a,b,c):
    if n==1:
        print(f'{n}:{a}->{c}')
        return
    else:
        Hanoi(n-1,a,c,b)
        print(f'{n}:{a}->{c}')
        Hanoi(n-1,b,a,c)
        return
Hanoi(num,x,y,z)
```

耗时：30min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251028160641888](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251028160641888.png)



### M05585: 晶矿的个数

matrices, dfs similar, http://cs101.openjudge.cn/pctbook/M05585

思路：



代码

```python
from collections import deque
directions=[(1,0),(-1,0),(0,1),(0,-1)]
def bfs(x,y,n,m):
    node=deque([(x,y)])
    m[x][y]='#'
    while node:
        x, y = node.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0<=nx<=len(m)-1 and 0<=ny<=len(m[0])-1 and m[nx][ny] == n:
                m[nx][ny]='#'
                node.append((nx, ny))
k=int(input())
for _ in range(k):
    n=int(input())
    m=[[0]*n for _ in range(n)]
    for i in range(n):
        s = input()
        for j in range(n):
            m[i][j]=s[j]
    r_cnt,b_cnt=0,0
    for i in range(n):
        for j in range(n):
            if m[i][j]=='r':
                bfs(i,j,'r',m)
                r_cnt+=1
            if m[i][j]=='b':
                bfs(i,j,'b',m)
                b_cnt+=1
    print(r_cnt,b_cnt)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251028173524628](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251028173524628.png)



### M02786: Pell数列

dfs, dp, http://cs101.openjudge.cn/pctbook/M02786/

思路：



代码

```python
import math as m
def Pell(x):
    p=[1,2]
    for i in range(m.ceil(x/2)-1):
        p[0]=(p[1]*2+p[0])%32767
        p[1]=(p[0]*2+p[1])%32767
    if x%2:
        return p[0]
    else:
        return p[1]
n=int(input())
for _ in range(n):
    t=int(input())
    print(Pell(t))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251028173752163](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251028173752163.png)



### M46.全排列

backtracking, https://leetcode.cn/problems/permutations/


思路：



代码

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(path,used):
            if len(path)==len(nums):
                result.append(path[:])
                return
    
            for i in range(len(nums)):
                if used[i]:
                    continue
                path.append(nums[i])
                used[i]=True
        
                backtrack(path,used)
        
                path.pop()
                used[i]=False
        
        used=[False]*len(nums)
        result=[]
        backtrack([],used)
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251028203940117](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251028203940117.png)



### T02754: 八皇后

dfs and similar, http://cs101.openjudge.cn/pctbook/T02754

思路：



代码

```python
def eight_queens():
    def backtrack(row,path,minus,plus):
        if row==8:
            result.append(path[:])
            return
        for i in range(1,9):
            if str(i) in path or i-row in minus or i+row in plus:
                continue
            path.append(str(i))
            minus.append(i-row)
            plus.append(i+row)
            backtrack(row+1,path,minus,plus)
            path.pop()
            minus.pop()
            plus.pop()
    result=[]
    backtrack(0,[],[],[])
    return result
result=eight_queens()
nums=[]
for i in result:
    i=int(''.join(i))
    nums.append(i)
nums.sort()
n=int(input())
for _ in range(n):
    b=int(input())
    print(nums[b-1])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251030160923973](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251030160923973.png)



### T01958 Strange Towers of Hanoi

http://cs101.openjudge.cn/practice/01958/

思路：



代码

```python
def triHanoi(n):
    if n==1:
        return 1
    else:
        return triHanoi(n-1)*2+1
def quadHanoi(n):
    if n==0:
        return 0
    elif n==1:
        return 1
    else:
        from math import inf
        mini = inf
        for i in range(1, n + 1):
            total = quadHanoi(n - i) * 2 + triHanoi(i)
            mini = min(mini, total)
        return mini
for i in range(1,13):
    print(quadHanoi(i))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251030174936365](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251030174936365.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2025fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

额外练习题目：每日选做的部分backtrack题目。

学习收获：

1.在汉诺塔那个题因为不知道怎么打印出结果卡了十几分钟，后来才知道函数能够不return任何值，可以在函数里面写print函数。

2.仔细地写了一遍dfs和bfs。

3.学到了回溯算法，大概的算法框架是：

```python
def backtrack(结束指标（可有可无），路径，选择列表（对于八皇后问题就是1到8所以可以不写），控制列表（可有可无）)
	if 满足结束条件:
        结果中添加上路径（注意要复制一份）
        return
    for 选择 in 选择列表:
        if 不满足所需条件:
            continue #剪枝
            做选择
            backtrack(新的参数)
            撤销选择
```

