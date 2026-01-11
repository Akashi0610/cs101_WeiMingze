# Assignment #1: 自主学习

Updated 1306 GMT+8 Sep 14, 2025

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

### E02733: 判断闰年

http://cs101.openjudge.cn/pctbook/E02733/



思路：



代码

```python
a=int(input())
if a%4==0 and a%100 or a%400==0:
    print("Y")
else:
    print("N")
```

耗时：1min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250919233208967](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250919233208967.png)



### E02750: 鸡兔同笼

http://cs101.openjudge.cn/pctbook/E02750/



思路：



代码

```python
a=int(input())
if a%4==0:
    print(a//4,a//2)
elif a%4 and a%2==0:
    print(a//4+(a%4)//2,a//2)
elif a%2:
    print(0,0)
```

耗时：2min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250919233350941](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250919233350941.png)



### 50A. Domino piling

greedy, math, 800, http://codeforces.com/problemset/problem/50/A



思路：

如果两个边长均为1则肯定一块多米诺骨牌也放不下；如果两个边长中有一个是1，则能放下的多米诺骨牌数取决于另外一个边长是2的几倍；如果两个边长均大于1，放多米诺骨牌最多的方式肯定是先沿着同一个方向放尽可能多的骨牌再沿着另外一个方向将剩余的位置补上，但是最开始沿着哪个方向不一定，所以说我选择两种都打出来取最大值。~~（虽然感觉这种方法思路有点麻烦了）~~

代码

```python
s=input().split()
M,N=int(s[0]),int(s[1])
if M==1 and N==1:
    print(0)
elif M==1 or N==1:
    print(max(M,N)//2)
else:
    print(max((M//2)*N+(M%2)*(N//2),(N//2)*M+(N%2)*(M//2)))
```

耗时：2min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250919233656715](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250919233656715.png)



### 1A. Theatre Square

math, 1000, https://codeforces.com/problemset/problem/1/A



思路：



代码

```python
s=input().split()
n,m,a=int(s[0]),int(s[1]),int(s[2])
if n<=a and m<=a:
    print(1)
elif n%a==0 and m%a!=0:
    print((n//a)*(m//a+1))
elif n%a!=0 and m%a==0:
    print((n//a+1)*(m//a))
elif n%a==0 and m%a==0:
    print((n//a)*(m//a))
else:
    print((n//a+1)*(m//a+1))
```

耗时：2min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250919234455168](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250919234455168.png)



### 112A. Petya and Strings

implementation, strings, 1000, http://codeforces.com/problemset/problem/112/A



思路：



代码

```python
s1=input()
s2=input()
if s1.lower()==s2.lower():
    print(0)
elif s1.lower()<s2.lower():
    print(-1)
else:
    print(1)
```

耗时：1min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250919234635442](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250919234635442.png)



### 231A. Team

bruteforce, greedy, 800, http://codeforces.com/problemset/problem/231/A



思路：



代码

```python
n=int(input())
times=0
for i in range(1,n+1):
    s=input().split()
    total=0
    for j in s:
        total+=int(j)
    if total>=2:
        times+=1
print(times)
```

耗时：1min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250919234915978](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250919234915978.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2025fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

额外练习题目：0827-0831的每日选做题目

学习收获：我在暑假期间自学了一部分语法，经过做题我加深了对这些知识的印象，同时AC每一道题都会让我有一种成就感，让我想要继续做下一道题，~~可能是因为还没有做到难题~~，课程讨论群里面大家的积极讨论也激励着我更加努力地学习。



