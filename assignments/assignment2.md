# Assignment #2: 语法练习

Updated 1335 GMT+8 Sep 16, 2025

2025 fall, Complied by <mark>魏铭泽 元培学院</mark>



**作业的各项评分细则及对应的得分**

| 标准                                 | 等级                                                         | 得分 |
| ------------------------------------ | ------------------------------------------------------------ | ---- |
| 按时提交                             | 完全按时提交：1分<br/>提交有请假说明：0.5分<br/>未提交：0分  | 1 分 |
| 源码、耗时（可选）、解题思路（可选） | 提交了4个或更多题目且包含所有必要信息：1分<br/>提交了2个或以上题目但不足4个：0.5分<br/>少于2个：0分 | 1 分 |
| AC代码截图                           | 提交了4个或更多题目且包含所有必要信息：1分<br/>提交了2个或以上题目但不足4个：0.5分<br/>少于：0分 | 1 分 |
| 清晰头像、PDF文件、MD/DOC附件        | 包含清晰的Canvas头像、PDF文件以及MD或DOC格式的附件：1分<br/>缺少上述三项中的任意一项：0.5分<br/>缺失两项或以上：0分 | 1 分 |
| 学习总结和个人收获                   | 提交了学习总结和个人收获：1分<br/>未提交学习总结或内容不详：0分 | 1 分 |
| 总得分： 5                           | 总分满分：5分                                                |      |

>
>
>
>**说明：**
>
>1. **解题与记录：**
>
>   对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
>2. **课程平台：**课程网站位于Canvas平台（https://pku.instructure.com ）。该平台将在<mark>第2周</mark>选课结束后正式启用。在平台启用前，请先完成作业并将作业妥善保存。待Canvas平台激活后，再上传你的作业。
>
>3. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的本人头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
>4. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。  
>
>请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。





## 1. 题目

### 263A. Beautiful Matrix

implementation, 800, https://codeforces.com/problemset/problem/263/A



思路：

先构造一个矩阵，然后再找到现在矩阵中1的位置，再算其到矩阵中心的距离。

代码

```python
lst=[]
for i in range(5):
    s=input().split()
    lst.append(s)
for i in range(5):
    for j in range(5):
        if int(lst[i][j])==1:
            a,b=i,j
            break
print(abs(a-2)+abs(b-2))

```

耗时：3min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250920000939063](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250920000939063.png)



### 1328A. Divisibility Problem

math, 800, https://codeforces.com/problemset/problem/1328/A



思路：

一开始想用迭代，超时后发现能直接算。

代码

```python
n=int(input())
for i in range(n):
    a,b=map(int,input().split())
    if a%b==0:
        print(0)
    else:
        print((a//b+1)*b-a)
```

耗时：15min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250920001224492](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250920001224492.png)



### 427A. Police Recruits

implementation, 800, https://codeforces.com/problemset/problem/427/A



思路：

先设定一个total，再把每个数往total上加，当total小于0时说明有1个罪犯没有警察来抓，则times+1，此时再将total归零，继续重复上面过程，最后times就是没被抓的罪犯数。

代码

```python
n=int(input())
s=input().split()
total=0
times=0
for i in range(n):
    total+=int(s[i])
    if total<0:
        times+=1
        total=0
print(times)
```

耗时：10min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250920001506169](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250920001506169.png)



### E02808: 校门外的树

implementation, http://cs101.openjudge.cn/pctbook/E02808/


思路：

先创建一个字典，把这条路上所有的树全都标记为1，再把施工路段内的树标记为0，最后把字典里面所有值相加就是还剩下的树的数量。

代码

```python
L,M=map(int,input().split())
dt={}
for i in range(0,L+1):
    dt[i]=1
for i in range(M):
    s,t=map(int,input().split())
    for j in range(s,t+1):
        dt[j]=0
total=0
for i in dt.values():
    total+=i
print(total)
```

耗时：10min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250920002002164](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250920002002164.png)



### sy60: 水仙花数II

implementation, https://sunnywhy.com/sfbj/3/1/60



思路：

先将数字转化成字符串再用下标访问字符串的各位，得到该三位数的各位上的数，再找到水仙花数，而后转化成字符串输出（因为每两个水仙花数需要隔一个空格输出），最后再用.strip删去输出末尾的空格。

代码

```python
a,b=map(int,input().split())
s=""
for i in range(a,b+1):
    x,y,z=int(str(i)[0]),int(str(i)[1]),int(str(i)[2])
    if x**3+y**3+z**3==i:
        s+=str(i)+" "
if s=="":
    print("NO")
else:
    print(s.strip())
```

耗时：10min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250920002252063](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250920002252063.png)



### M01922: Ride to School

implementation, http://cs101.openjudge.cn/pctbook/M01922/



思路：

最重要的一点是Charley一定是和他最后跟着的骑手一起抵达燕园，所以说只需要找到在Charley之后出发的最快到达燕园的骑手所用的时间即可（当然还需要加上Charley比他先出发的时间），一开始没有想明白这一点，一直在按照高中物理运动学的思维来思考，在试图找出每两个骑手的相遇时间时被难住了很久QAQ。

代码

```python
import math
try:
    while True:
        N=int(input())
        if N==0:
            break
        min_time=math.inf
        for i in range(N):
            v,t=map(int,input().split())
            total_time=t+(4.5*3600/v)
            if total_time<min_time and t>=0:
                min_time=total_time
        print(math.ceil(min_time))
except:
    pass
```

耗时：1h

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250920002647553](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250920002647553.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2025fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

额外练习题目：0901-0912的每日选做

学习总结和收获：

通过Ride to School这道题我学会了怎么不定行输入，同时也学会了如何使用Python自带的库（比如这道题中使用到的math库）。同时在做这道题时我也感受到了M等级难度的题目对于现在的我来说还很吃力，我还需要做更多的题来提升自己。

在做0911的每日选做中03143：验证哥德巴赫猜想 这道题时我还学会了欧式筛和埃氏筛，感觉到了不同算法对于运行时间的重要影响。



