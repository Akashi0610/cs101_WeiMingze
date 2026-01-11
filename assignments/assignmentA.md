# Assignment #A: 递归、田忌赛马

Updated 2355 GMT+8 Nov 4, 2025

2025 fall, Complied by <mark>同学的姓名、院系</mark>



>**说明：**
>
>1. **解题与记录：**
>
>  对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
>2. 提交安排：提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的本人头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
> 
>4. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。  
>
>请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。





## 1. 题目

### M018160: 最大连通域面积

dfs similar, http://cs101.openjudge.cn/pctbook/M18160

思路：



代码

```python
T=int(input())
for _ in range(T):
    N,M=map(int,input().split())
    board=[]
    for _ in range(N):
        board.append(input())
    from collections import deque
    visited=[[False]*M for _ in range(N)]
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (0, -1), (1, -1), (1, 0), (1, 1)]
    def dfs(start,result):
        dq=deque([start])
        visited[start[0]][start[1]]=True
        while dq:
            node=dq.pop()
            result+=1
            for dx,dy in directions:
                nx,ny=node[0]+dx,node[1]+dy
                if 0<=nx<N and 0<=ny<M and board[nx][ny]=='W' and not visited[nx][ny]:
                    visited[nx][ny]=True
                    dq.append((nx,ny))
        return result
    maximum=0
    for i in range(N):
        for j in range(M):
            if board[i][j]=='W' and not visited[i][j]:
                area=dfs((i,j),0)
                if area>maximum:
                    maximum=area
    print(maximum)
```

耗时：40min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251106234739632](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251106234739632.png)



### sy134: 全排列III 中等

https://sunnywhy.com/sfbj/4/3/134

思路：



代码

```python
n=int(input())
nums=list(map(int,input().split()))
def backtrack(path,used):
    if len(path)==n and path not in result:
        result.append(path[:])
    for i in range(n):
        if used[i]:
            continue
        used[i]=True
        path.append(nums[i])
        backtrack(path,used)
        used[i]=False
        path.pop()
used=[False]*n
result=[]
backtrack([],used)
result.sort()
for item in result:
    print(*item)
```

耗时：15min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251107000633406](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251107000633406.png)



### sy136: 组合II 中等

https://sunnywhy.com/sfbj/4/3/136

给定一个长度为的序列，其中有n个互不相同的正整数，再给定一个正整数k，求从序列中任选k个的所有可能结果。

思路：



代码

```python
n,k=map(int,input().split())
nums=list(map(int,input().split()))
def dfs(index,path):
    if len(path)==k:
        result.append(path[:])
        return
    if len(path)>k or index>=n:
        return
    path.append(nums[index])
    dfs(index+1,path)
    path.pop()
    dfs(index+1,path)
result=[]
dfs(0,[])
result.sort()
for item in result:
    print(*item)
```

耗时：20min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251107003225884](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251107003225884.png)



### sy137: 组合III 中等

https://sunnywhy.com/sfbj/4/3/137


思路：



代码

```python
n,k=map(int,input().split())
nums=list(map(int,input().split()))
def dfs(index,path):
    if len(path)==k and path not in result:
        result.append(path[:])
        return
    if len(path)>k or index==n:
        return
    path.append(nums[index])
    dfs(index+1,path)
    path.pop()
    dfs(index+1,path)
result=[]
dfs(0,[])
result.sort()
for item in result:
    print(*item)
```

耗时：5min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251107004043959](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251107004043959.png)



### M04123: 马走日

dfs, http://cs101.openjudge.cn/pctbook/M04123

思路：

这应该算是我自己独立写出的第一个dfs题目了，只不过最后出了一些小错误没看出来问了一下ai

思路就是对每个节点周围的8个节点做dfs，然后对已经经过的节点标记一下，这里需要对visited进行回溯的原因应该是当这一层dfs已经结束之后，我要进行下一次的对另外一个点的dfs，所以说需要还原visited，以免对下一次dfs造成干扰。

代码

```python
directions=[(-2,-1),(-2,1),(-1,-2),(-1,2),(2,-1),(2,1),(1,-2),(1,2)]
def dfs(i,j,steps):
    if steps==n*m:
        return 1
    total=0
    visited[i][j]=True
    for dx,dy in directions:
        nx,ny=i+dx,j+dy
        if 0<=nx<n and 0<=ny<m and not visited[nx][ny]:
            total+=dfs(nx,ny,steps+1)
    visited[i][j]=False
    return total

T=int(input())
for _ in range(T):
    n,m,x,y=map(int,input().split())
    visited=[[False]*m for _ in range(n)]
    print(dfs(x,y,1))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251113162843584](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251113162843584.png)



### T02287: Tian Ji -- The Horse Racing

greedy, dfs http://cs101.openjudge.cn/pctbook/T02287

思路：

本质上是两句话：让田忌以最小的优势获胜，让齐王以最大的优势获胜。每次比较田忌的最慢马和齐王的最慢马，这样有三种可能：1.田忌的最慢马比齐王的最慢马快，则直接用田忌的最慢马比掉齐王的最慢马，cnt+1；2.田忌的最慢马比齐王的最慢马慢，则直接用田忌的最慢马比掉齐王的最快马，cnt-1；3.田忌的最慢马与齐王的最慢马一样快，则需要比较二者的最快马，当田忌的最快马比齐王的最快马快时，可以直接用最快马相比，最慢马相比，cnt+1（如果说此时还用田忌的最慢马比掉齐王的最快马则田忌的最快马就不能以最小优势获胜，而且还会送给齐王1分），当田忌的最快马不如齐王的最快马快时，应该用田忌的最慢马比掉齐王的最快马，这样既消耗了齐王的最快马也保留了田忌的最快马（让齐王以最大的优势获胜），但是此时还有一个问题，如果此时齐王的最快马和齐王的最慢马一样快时，说明齐王的最快马，田忌的最快马，齐王的最慢马，田忌的最慢马均相等，此时只可能平局所以说需要排除这种情况之后再把cnt-1。

代码

```python
while True:
    n=int(input())
    if n==0:
        break
    tian=sorted(map(int,input().split()))
    king=sorted(map(int,input().split()))
    cnt=0
    t_slow,t_fast,k_slow,k_fast=0,n-1,0,n-1
    for _ in range(n):
        if tian[t_slow]>king[k_slow]:
            cnt+=1
            t_slow+=1
            k_slow+=1
        elif tian[t_slow]<king[k_slow]:
            cnt-=1
            t_slow+=1
            k_fast-=1
        else:
            if tian[t_fast]>king[k_fast]:
                cnt+=1
                t_fast-=1
                k_fast-=1
            else:
                if king[k_slow]!=king[k_fast]:
                    cnt-=1
                    t_slow+=1
                    k_fast-=1
    print(cnt*200)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251117224528576](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251117224528576.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2025fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

学习收获：
1.有一些dfs的题已经能够默写下来了，但是还是不够，应当自己多做一些dfs的题加深自己对其的理解，目前已经对回溯有了一个大概的理解了，但是有时候还是会想不明白。

2.田忌赛马这道题让我练习了一下双指针这个之前每太使用过的东西，同时这个贪心思路也给了我一定的启发。



