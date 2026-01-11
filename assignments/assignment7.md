# Assignment #7: 矩阵、队列、贪心

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

### M12560: 生存游戏

matrices, http://cs101.openjudge.cn/pctbook/M12560/

思路：



代码

```python
n,m=map(int,input().split())
cells=[[0]*(m+2)]
for _ in range(n):
    cells.append([0]+list(map(int,input().split()))+[0])
cells.append([0]*(m+2))
from copy import deepcopy
c=deepcopy(cells)
for i in range(1,n+1):
    for j in range(1,m+1):
        case=cells[i-1][j-1]+cells[i-1][j]+cells[i-1][j+1]+cells[i][j-1]+cells[i][j+1]+cells[i+1][j-1]+cells[i+1][j]+cells[i+1][j+1]
        if cells[i][j]==1:
            if case<2 or case>3:
                c[i][j]=0
        else:
            if case==3:
                c[i][j]=1
for i in range(1,n+1):
    for j in range(1,m+1):
        print(c[i][j],end=" ")
    print('')
```

耗时：20min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251021155804489](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251021155804489.png)



### M04133:垃圾炸弹

matrices, http://cs101.openjudge.cn/pctbook/M04133/

思路：

遍历所有垃圾周围能炸到这处垃圾的点，更新他们的值，将其与最大值比较，进而更新最大值和投放点数量。

代码

```python
d=int(input())
n=int(input())
board=[[0]*1025 for _ in range(1025)]
max_v=0
cnt=0
for _ in range(n):
    x,y,z=map(int,input().split())
    for i in range(max(0,x-d),min(1024,x+d)+1):
        for j in range(max(0,y-d),min(1024,y+d)+1):
            board[i][j]+=z
            if board[i][j]>max_v:
                max_v=board[i][j]
                cnt=1
            elif board[i][j]==max_v:
                cnt+=1
print(cnt,max_v)
```

耗时：1h20min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251021172304814](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251021172304814.png)



### M02746: 约瑟夫问题

implementation, queue, http://cs101.openjudge.cn/pctbook/M02746/

思路：



代码

```python
import sys
for line in sys.stdin:
    line=line.strip()
    if line!='0 0':
        n,m=map(int,line.split())
        from collections import deque
        d=deque([i for i in range(1,n+1)])
        while True:
            if len(d)==1:
                print(d.pop())
                break
            else:
                d.rotate(-(m-1))
                d.popleft()
```

耗时：10min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251021173940569](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251021173940569.png)



### M26976:摆动序列

greedy, http://cs101.openjudge.cn/pctbook/M26976/


思路：

将每两个数之间的差算出来，对于一个数，如果两边的差的符号相反，就在总长度上+1，如果两边的差的符号相同，就先记录下来它左边的差的符号，然后看后面的序列，此时有两种情况，一种是形如3 2 5 6 5，因为5-2>0,6-5>0，所以刚才记录下来的符号也可以作为6这个元素的左边的符号，接下来和前面同理；第二种是形如3 2 5 6 4，此时按照我的代码来说，记录下来的摆动序列应该为3 2 6 4，实际上可以有两种不同的摆动序列3 2 5 4 和3 2 6 4 ，两种摆动序列的长度也是一样的，所以也无所谓，所以最后记录下来的结果就应该是正确的。

代码

```python
n=int(input())
nums=list(map(int,input().split()))
if n==1:
    print(1)
else:
    dif=[]
    for i in range(1,n):
        if nums[i]-nums[i-1]<0:
            dif.append(-1)
        elif nums[i]-nums[i-1]>0:
            dif.append(1)
        else:
            dif.append(0)
    l=1
    last=0
    for i in range(n-1):
        if dif[i]*last<0 or (last==0 and dif[i]!=0):
            l+=1
            last=dif[i]
    print(l)
```

耗时：1h30min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251021195723851](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251021195723851.png)



### T26971:分发糖果

greedy, http://cs101.openjudge.cn/pctbook/T26971/

思路：

如果将ratings这个数组画成图像，那么会有很多波峰波谷，波谷处的孩子分到的糖果一定为1。我为了使整个图像一开始和最后都是波谷，在ratings的前后分别加上了一个-1，这样波谷数就一定比波峰数多1，将所有波峰、波谷收集起来，对波峰进行遍历，在波峰的两端一边是单调递增序列，一边是单调递减序列，分别求出两个单调序列中所需要的糖数，这样的话波峰就算了两次，需要在最后的结果上减掉小的那一个值。大概的思路就是这样，但是需要对相等的点做很多复杂的说明，如果说连续几个值都相等，而且他们还是局部的极大值极小值，我这里是让他们中最左边的那个数作为极值点，剩下的数放到单调序列中，而且还要考虑如下情况：比如几个相邻的值是1 1 1 1，那么无论两边的1分到几个糖，中间的1都应该只分到1个糖。总之就是十分复杂，我在看了题解之后自叹不如，不仅比我的更简单而且还更快，于是又照着题解的思路又写了一个新的代码，由于和题解差不多，这里就不放了orz。

代码

```python
n=int(input())
ratings=[-1]+list(map(int,input().split()))+[-1]
min_v,max_v=[0],[]
#记录所有极大值极小值点,并在两端分别添加一个极小值点
for i in range(1,n+1):
    if (ratings[i]-ratings[i-1])*(ratings[i+1]-ratings[i])<0:
        if ratings[i]-ratings[i-1]<0:
            min_v.append(i)
        if ratings[i]-ratings[i-1]>0:
            max_v.append(i)
    elif (ratings[i]-ratings[i-1])*(ratings[i+1]-ratings[i])==0:
        if ratings[i]>ratings[i-1] and ratings[i]==ratings[i+1]:
            for k in range(i+1,n+1):
                if ratings[k]!=ratings[i]:
                    break
                if ratings[k]<ratings[k+1]:
                    pass
                elif ratings[k]>ratings[k+1]:
                    max_v.append(i)
        elif ratings[i]<ratings[i-1] and ratings[i]==ratings[i+1]:
            for k in range(i+1,n+1):
                if ratings[k]!=ratings[i]:
                    break
                if ratings[k]>ratings[k+1]:
                    pass
                elif ratings[k]<ratings[k+1]:
                    min_v.append(i)
min_v.append(n+1)
#极小值点得到的糖数一定为1，需要再减去上面多添加的两个极小值点
total=len(min_v)-2
#将整段数组分成多段，每段中只含有一个极大值点，分别算极大值点两端的糖数，极大值点的糖数取分别从两端得到的糖数的最大值
last=1
for i in range(len(max_v)):
    m1,m2=1,1
    last=1
    for j in range(min_v[i]+1,max_v[i]+1):
        if ratings[j-1]==-1:
            total+=last
        elif ratings[j]==ratings[j-1]:
            last=1
            total+=last
        else:
            last+=1
            total+=last
    m1=last
    last=1
    for j in range(min_v[i+1]-1,max_v[i]-1,-1):
        if ratings[j+1]==-1:
            total+=last
        elif ratings[j]==ratings[j+1]:
            last=1
            total+=last
        else:
            last+=1
            total+=last
    m2=last
    total-=min(m1,m2)
print(total)
```

耗时：不知道多久……

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251023163208268](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251023163208268.png)



### 1868A. Fill in the Matrix

constructive algorithms, implementation, 1300, https://codeforces.com/problemset/problem/1868/A

思路：

针对给的样例m=n=6入手，最简单的情况如下：

1 2 3 4 5 0

2 3 4 5 0 1

3 4 5 0 1 2

4 5 0 1 2 3 

5 0 1 2 3 4

1 2 3 4 5 0

所以说只需要每行对1 2 3 4 5 0进行旋转就可以了~~deque的rotate真好用~~

代码

```python
from collections import deque
t=int(input())
for _ in range(t):
    n,m=map(int,input().split())
    d=deque([x for x in range(m)])
    if m==1:
        print(0)
        for i in range(n):
            print(0)
    elif n>=m:
        print(m)
        cnt=n//(m-1)
        for i in range(cnt):
            for j in range(m-1):
                d.rotate(-1)
                print(*list(d))
            d.rotate(-1)
        for i in range(n%(m-1)):
            d.rotate(-1)
            print(*list(d))
    else:
        print(n+1)
        for i in range(n):
            d.rotate(-1)
            print(*list(d))
```

耗时：30min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251023154657285](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251023154657285.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2025fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

额外练习题目：期中周来临，而且感觉作业难度不小，所以只做了作业

学习收获：

1.学会了使用深度拷贝

2.学会了使用deque的rotate函数

3.在面对分发糖果那样的题时，如果发现自己的思路可能需要很复杂的讨论，那一定会有更好更简便的思路。



