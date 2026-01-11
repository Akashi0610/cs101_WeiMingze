# Assignment #4: T-primes + 贪心

Updated 1814 GMT+8 Sep 30, 2025

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

### 34B. Sale

greedy, sorting, 900, https://codeforces.com/problemset/problem/34/B



思路：



代码

```python
n,m=map(int,input().split())
prices=[int(i) for i in input().split()]
prices.sort()
num, total = 0, 0
for i in range(n):
    if num<m:
        if prices[i]<0:
            total-=prices[i]
            num+=1
    else:
        break
print(total)
```

耗时：10min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250930201004932](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250930201004932.png)



### 160A. Twins

greedy, sortings, 900, https://codeforces.com/problemset/problem/160/A



思路：



代码

```python
n=int(input())
s=input().split()
total=0
for i in range(n):
    s[i]=int(s[i])
    total+=s[i]
s.sort(reverse=True)
money=0
for i in range(n):
    money+=s[i]
    if money>total/2:
        print(i+1)
        break
```

耗时：20min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250930201205835](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250930201205835.png)



### 1879B. Chips on the Board

constructive algorithms, greedy, 900, https://codeforces.com/problemset/problem/1879/B



思路：



代码

```python
t=int(input())
for _ in range(t):
    n=int(input())
    a=input().split()
    b=input().split()
    a1,b1=[int(i) for i in a],[int(i) for i in b]
    a1.sort()
    b1.sort()
    total_a,total_b=b1[0]*n,a1[0]*n
    for i in range(n):
        total_a+=a1[i]
        total_b+=b1[i]
    print(min(total_a,total_b))
```

耗时：30min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250930195543363](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250930195543363.png)



### M01017: 装箱问题

greedy, http://cs101.openjudge.cn/pctbook/M01017/


思路：



代码

```python
import math as m
import sys
for line in sys.stdin:
    a,b,c,d,e,f=map(int,line.split())
    if a+b+c+d+e+f==0:
        break
    else:
        total=f+e+d+m.ceil(c/4)
        lst=[0,5,3,1]
        b0=d*5+lst[c%4]
        if b>b0:
            total+=m.ceil((b-b0)/9)
        a0=36*total-4*b-9*c-16*d-25*e-36*f
        if a>a0:
            total+=m.ceil((a-a0)/36)
        print(total)
```

耗时：30min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250930204006709](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250930204006709.png)



### M01008: Maya Calendar

implementation, http://cs101.openjudge.cn/practice/01008/



思路：



代码

```python
n=int(input())
print(n)
for _ in range(n):
    data=input().split()
    monthHaab={'pop':1, 'no':2, 'zip':3, 'zotz':4, 'tzec':5, 'xul':6, 'yoxkin':7, 'mol':8, 'chen':9, 'yax':10, 'zac':11,'ceh':12, 'mac':13, 'kankin':14, 'muan':15, 'pax':16, 'koyab':17, 'cumhu':18,'uayet':19}
    noteTzolkin={1:'imix',2:'ik',3:'akbal',4:'kan',5:'chicchan',6:'cimi',7:'manik',8:'lamat',9:'muluk',10:'ok',11:'chuen',12:'eb',13:'ben',14:'ix',15:'mem',16:'cib',17:'caban',18:'eznab',19:'canac',20:'ahau'}
    d,m,y=int(data[0][:-1]),data[1],int(data[2])
    days=(monthHaab[m]-1)*20+d+y*365
    yT=days//260
    restDays=days%260
    num=restDays%13+1
    note=noteTzolkin[restDays%20+1]
    print(num,note,yT)
```

耗时：1h

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251001162338790](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251001162338790.png)



### 230B. T-primes（选做）

binary search, implementation, math, number theory, 1300, http://codeforces.com/problemset/problem/230/B



思路：

首先满足要求的数一定是一个平方数，否则因数的个数不可能为奇数，其次这个数的平方根一定是一个素数，否则还会出现除了1，自身和平方根之外的因数。但是如果对于每种情况都求一次质数表会导致内存很大用时很长，所以说需要先把小于题目设定的最大上限的素数求出来，再以此为基础找T-primes。

代码

```python
primes=[True]*(10**6+1)
primes[0],primes[1]=False,False
for j in range(2,10**3+1):
    if primes[j]:
        for p in range(j*j,10**6+1,j):
            primes[p]=False
Tprimes=set([])
for i in range(2,10**6+1):
    if primes[i]:
        Tprimes.add(i*i)
def T(x):
    if x in Tprimes:
        return 'YES'
    else:
        return 'NO'
n = int(input())
numbers = list(map(int, input().split()))
for i in range(n):
    print(T(numbers[i]))
```

耗时：1h30min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251001182030243](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251001182030243.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2025fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

额外练习题目：每日选做0921-0929

学习收获：在写每一道题时应该先思考这道题背后有没有潜在的数学逻辑，可能会大大简化代码减少运行时间。在写每一道题的时候应该仔细阅读题目的输入输出要求，~~已经在这上面吃了好几次亏了~~。



