# Assignment #D: Mock Exam下元节

Updated 1729 GMT+8 Dec 4, 2025

2025 fall, Complied by <mark>魏铭泽 元培学院</mark>



>**说明：**
>
>1. Dec⽉考： AC1<mark>（请改为同学的通过数）</mark> 。考试题⽬都在“题库（包括计概、数算题目）”⾥⾯，按照数字题号能找到，可以重新提交。作业中提交⾃⼰最满意版本的代码和截图。
>
>2. 解题与记录：对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
>3. 提交安排：提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的本人头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
> 
>4. 延迟提交：如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。  
>
>请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。





## 1. 题目

### E29945:神秘数字的宇宙旅行 

implementation, http://cs101.openjudge.cn/practice/29945

思路：



代码

```python
n=int(input())
while True:
    if n==1:
        print('End')
        break
    if n%2==0:
        print(f"{n}/2={n//2}")
        n=n//2
    else:
        print(f"{n}*3+1={n*3+1}")
        n=n*3+1
```

耗时：5min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251204194210661](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251204194210661.png)



### E29946:删数问题

monotonic stack, greedy, http://cs101.openjudge.cn/practice/29946

思路：



代码

```python
def RemoveDigits(num,k):
    stack=[]
    for digit in num:
        while k and stack and stack[-1]>digit:
            stack.pop()
            k-=1
        stack.append(digit)
    while k:
        stack.pop()
        k-=1
    return int(''.join(stack))
n=input()
num=[]
for digit in n:
    num.append(digit)
k=int(input())
print(RemoveDigits(num,k))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251208152726234](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251208152726234.png)



### E30091:缺德的图书馆管理员

greedy, http://cs101.openjudge.cn/practice/30091

思路：

最小值就是所有人都往自己的最近出口走所用时间里面的最大值，最大值就是所有人都往自己的最远出口走所用时间里面的最大值（将相遇看作交换，所有人都不回头）。

代码

```python
l=int(input())
n=int(input())
pos=list(map(int,input().split()))
mintime,maxtime=0,0
for i in pos:
    mintime=max(mintime,min(i,l+1-i))
for i in pos:
    maxtime=max(maxtime,max(i,l+1-i))
print(mintime,maxtime)
```

耗时：30min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251208155530811](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251208155530811.png)



### M27371:Playfair密码

simulation，string，matrix, http://cs101.openjudge.cn/practice/27371


思路：



代码

```python
keyword=input().strip()
n=int(input())
letters='abcdefghiklmnopqrstuvwxyz'
used=set([])
matrix=[['']*5 for i in range(5)]
l=[]
dt={}
for i in keyword:
    if i not in used:
        if i=='j' and 'i' not in used:
            l.append('i')
            used.add('i')
        elif i=='j' and 'i' in used:
            continue
        elif i!='j':
            used.add(i)
            l.append(i)
for i in letters:
    if i not in used:
        l.append(i)
        used.add(i)
for i in range(25):
    matrix[i//5][i%5]=l[i]
for i in range(5):
    for j in range(5):
        dt[matrix[i][j]]=(i,j)
for _ in range(n):
    s=input().strip()
    string=''
    for i in s:
        if i!='j':
            string+=i
        else:
            string+='i'
    twins=[]
    twin=''
    for i in string:
        if twin=='':
            twin+=i
        elif len(twin)==1 and twin[0]!=i:
            twin+=i
            twins.append(twin)
            twin=''
        elif len(twin)==1 and twin[0]==i:
            if twin[0]=='x':
                twin+='q'
                twins.append(twin)
                twin=i
            else:
                twin+='x'
                twins.append(twin)
                twin=i
    if twin!='':
        if twin[0]=='x':
            twin+='q'
            twins.append(twin)
        else:
            twin+='x'
            twins.append(twin)
    res=[]
    for t in twins:
        row1,col1,row2,col2=dt[t[0]][0],dt[t[0]][1],dt[t[1]][0],dt[t[1]][1]
        if row1==row2:
            res.append(matrix[row1][(col1+1)%5]+matrix[row2][(col2+1)%5])
        elif col1==col2:
            res.append(matrix[(row1+1)%5][col1]+matrix[(row2+1)%5][col2])
        elif row1!=row2 and col1!=col2:
            res.append(matrix[row1][col2]+matrix[row2][col1])
    print(''.join(res))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20251208152919113](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20251208152919113.png)



### T30201:旅行售货商问题

dp,dfs, http://cs101.openjudge.cn/practice/30201

思路：



代码

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### T30204:小P的LLM推理加速

greedy, http://cs101.openjudge.cn/practice/30204

思路：



代码

```python

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





## 2. 学习总结和收获

如果作业题目简单，有否额外练习题目，比如：OJ“计概2025fall每日选做”、CF、LeetCode、洛谷等网站题目。

本次考试只做对了一个题。删数问题不会做是因为自己没有掌握单调栈。playfair密码这道题在考场上做了很久，但是依旧没能想清楚所有的情况，导致一直RE，考完试之后才发现哪里做错了。图书管理员那道题在考场中没有想明白时间最大的情况应该怎么写，回来之后发现这个题想清楚之后真的很简单。

旅行售货问题写了一个dfs的版本但是超时了，状压dp还没有掌握，争取下周想清楚。

最后一道题有一定思路，但是还没看出来为什么会wa，下周一定完成！

期末机考争取ac4



