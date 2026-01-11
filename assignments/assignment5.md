# Assignment #5: 20251009 cs101 Mock Exam寒露第二天

Updated 1651 GMT+8 Oct 9, 2025

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

### E29895: 分解因数

implementation, http://cs101.openjudge.cn/practice/29895/



思路：



代码

```python
# 
import math as m
N=int(input())
lst=[]
for i in range(2,m.ceil(m.sqrt(N))):
    if N%i==0:
        lst.append(N//i)
print(max(lst))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251009195755213](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251009195755213.png)



### E29940: 机器猫斗恶龙

greedy, http://cs101.openjudge.cn/practice/29940/



思路：



代码

```python
n=int(input())
initial,total=0,0
lst=[int(x) for x in input().split()]
for i in range(n):
    total+=lst[i]
    if total+initial<=0:
        initial+=-total-initial+1
    else:
        continue
print(initial)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251009195922601](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251009195922601.png)



### M29917: 牛顿迭代法

implementation, http://cs101.openjudge.cn/practice/29917/



思路：



代码

```python
import sys
def newton(x,n):
    return x-(x**2-n)/(2*x)
for line in sys.stdin:
    num=eval(line)
    total=0
    est0,est1=0,1
    while True:
        if abs(est1-est0)<=0.000001:
            print(total,"%.2f"%est1)
            break
        else:
            total+=1
            est0=est1
            est1=newton(est1,num)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251009224147301](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251009224147301.png)



### M29949: 贪婪的哥布林

greedy, http://cs101.openjudge.cn/practice/29949/


思路：



代码

```python
# 
N,M=map(int,input().split())
lst=[]
for i in range(N):
    v,w=map(int,input().split())
    lst.append((v,w))
lst.sort(key=lambda x:(-x[0]/x[1],x[1]))
total,rest=0,M
for i in range(N):
    if lst[i][1]<=rest:
        rest-=lst[i][1]
        total+=lst[i][0]
    else:
        total+=lst[i][0]*(rest/lst[i][1])
        rest=0
    if rest==0:
        break
print("%.2f"%total)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251009200243587](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251009200243587.png)



### M29918: 求亲和数

implementation, http://cs101.openjudge.cn/practice/29918/



思路：

与筛选质数的方法相似，把每一个数的所有因数全都找出来放在同一个列表中，再转换成集合进行求和，确保没有重复的因数，进而筛选亲和数对。

代码

```python
n=int(input())
lst=[[1] for i in range(n+1)]
for i in range(2,n//2+1):
    for j in range(2*i,n+1,i):
        lst[j].append(i)
        lst[j].append(j//i)
for i in range(1,n+1):
    total=sum(set(lst[i]))
    if n>=total>i and sum(set(lst[total]))==i:
        print(i,total)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251009211458964](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251009211458964.png)



### T29947:校门外的树又来了（选做）

http://cs101.openjudge.cn/practice/29947/



思路：



代码

```python
L,M=map(int,input().split())
lst=[]
for _ in range(M):
    s,t=map(int,input().split())
    lst.append([s,t])
lst.sort()
for i in range(1,M):
    if lst[i][0]<=lst[i-1][1] and lst[i-1]!=[0,-1]:
        if lst[i][1]<=lst[i-1][1]:
            lst[i][0],lst[i][1]=0,-1
        else:
            lst[i][0]=lst[i-1][1]+1
    elif lst[i-1]==[0,-1]:
        for j in range(i-2,-1,-1):
            if lst[j]!=[0,-1]:
                if lst[i][0]<=lst[j][1]:
                    if lst[i][1] <= lst[j][1]:
                        lst[i][0], lst[i][1] = 0, -1
                    else:
                        lst[i][0] = lst[j][1] + 1
total=L+1
for i in range(M):
    total-=(lst[i][1]-lst[i][0]+1)
print(total)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251009220245097](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251009220245097.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2025fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

额外练习题目：每日选做0930-1005

学习收获：

~~0.在TLE时可以直接print出得到的结果来逃课~~

1.使用lst=[[]]*n这样的语句需要注意得到的列表里面的小列表都是指向同一位置的，一旦修改一个就会修改所有的；采用列表生成式就不会有这样的问题。

2.字典会消耗大量内存，如果输入数据过大（比如校门外的树里的10^9），使用字典可能会MLE

3.保留小数点后几位的方法：

1. 字符串格式化操作符%："%.2f" % num（即保留两位小数）（输出的格式是字符串）
2. format()方法： "{:.2f}".format(num)
3. round()函数：round(num,2)（可能会出现意外情况）

