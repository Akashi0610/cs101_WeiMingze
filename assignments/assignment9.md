# Assignment #9: Mock Exam立冬前一天

Updated 1658 GMT+8 Nov 6, 2025

2025 fall, Complied by <mark>魏铭泽 元培学院</mark>



>**说明：**
>
>1. Nov⽉考： AC3<mark>（请改为同学的通过数）</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
>
>2. 解题与记录：对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
>3. 提交安排：提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的本人头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
> 
>4. 延迟提交：如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。  
>
>请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。





## 1. 题目

### E29982:一种等价类划分问题

hashing, http://cs101.openjudge.cn/practice/29982

思路：



代码

```python
m,n,k=map(int,input().split(","))
dt={}
for i in range(m+1,n):
    total=0
    for j in str(i):
        total+=int(j)
    if not total%k:
        if total not in dt:
            dt[total]=[str(i)]
        else:
            dt[total].append(str(i))
lst=[]
for k,v in dt.items():
    lst.append((k,v))
lst.sort(key=lambda x:x[0])
for k,v in lst:
    v.sort(key=lambda x:int(x))
    print(','.join(v))
```

耗时：30min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251106180840022](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251106180840022.png)



### E30086:dance

greedy, http://cs101.openjudge.cn/practice/30086

思路：



代码

```python
N,D=map(int,input().split())
height=list(map(int,input().split()))
height.sort()
b=True
for i in range(N):
    if height[2*i+1]-height[2*i]>D:
        b=False
if b:
    print("Yes")
else:
    print("No")
```

耗时 ：5min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251106180919268](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251106180919268.png)



### M25570: 洋葱

matrices, http://cs101.openjudge.cn/practice/25570

思路：



代码

```python
n=int(input())
onion=[]
for _ in range(n):
    onion.append(list(map(int,input().split())))
from math import ceil
maximum=0
for i in range(ceil(n/2)):
    total=0
    for p in range(i,n-i):
        total+=onion[i][p]+onion[n-i-1][p]
    for p in range(i+1,n-i-1):
        total+=onion[p][i]+onion[p][n-i-1]
    if n%2 and i==ceil(n/2)-1:
        total-=onion[(n-1)//2][(n-1)//2]
    if total>maximum:
        maximum=total
print(maximum)
```

耗时：20min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251106181010371](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251106181010371.png)



### M28906:数的划分

dfs, dp, http://cs101.openjudge.cn/practice/28906


思路：

dp`[i][j]`表示i被分成j份的方案数，当i<j时一定为0，当j=1时一定为1（只有一种分法，即整数本身），当j>=2时，可以将所有方案分成两部分，其中一部分包含1，则相当于将i-1分成j-1份，所以方案数为dp`[i-1][j-1]`，当不包含1时，方案中的每个数一定大于等于2，所以可以将每个数都-1，即总数-j，此时就相当于将i-j分成j份，所以总的方案数为dp`[i-j][j]`，所以状态转移方程为`dp[i][j]=dp[i-1][j-1]+dp[i-j][j]`,进而可以填充DP表，得到`dp[n][k]`。

代码

```python
n, k = map(int, input().split())
dp=[[0]*(k+1) for i in range(n+1)]
for i in range(1,n+1):
    dp[i][1]=1
for j in range(2,k+1):
    for i in range(j,n+1):
        dp[i][j]=dp[i-1][j-1]+dp[i-j][j]
print(dp[n][k])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251106221803493](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251106221803493.png)



### M29896:购物

greedy, http://cs101.openjudge.cn/practice/29896

思路：

current代表当前能够得到的最大面值，从大到小遍历硬币的面值，当coins[i]>current+1时，加上coins[i]这个硬币后会导致current+1这个面值无法被得到，当遇到coins[i]<=current+1时，代表coins[i]以下的面值都能够得到了，也就是能得到1,2,3,4……current，所以再加上一个coins[i]也就能得到coins[i]+1到coins[i]+current的面值，所以说可以将current更新为coins[i]+current，以此类推，当current的值大于X时就找到了所有需要的硬币。

代码

```python
X,N=map(int,input().split())
coins=list(map(int,input().split()))
coins.sort(reverse=True)
current=total=0
while current<X:
    candidate=0
    for i in range(N):
        if coins[i]<=current+1:
            candidate=coins[i]
            break
    current+=candidate
    total+=1
print(total)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251106200838824](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251106200838824.png)



### T25353:排队

greedy, http://cs101.openjudge.cn/practice/25353

思路：



代码

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





## 2. 学习总结和收获

如果作业题目简单，有否额外练习题目，比如：OJ“计概2025fall每日选做”、CF、LeetCode、洛谷等网站题目。

学习收获：

1.在考试时第一道题就卡住了一段时间，后来才想起来python3.8里的字典还没有顺序。

2.接触到了dp和状态转移方程。

3.最后一道排队真是拼尽全力无法战胜了，期中周之后一定会来好好想这道题的orz

感觉现在自己对于递归、回溯、dfs和bfs的理解还是太浅了，虽然全排列之类的题已经能默写下来了但是感觉没有真正掌握核心思想，遇到其他的题还是不会做，还是需要看题解或者问ai，争取在下两周提高一下自己对这些算法的认识吧。



