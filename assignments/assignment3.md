# Assignment #3: 语法练习

Updated 1440 GMT+8 Sep 23, 2025

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

### E28674:《黑神话：悟空》之加密

http://cs101.openjudge.cn/pctbook/E28674/



思路：



代码

```python
k=int(input())
s=input()
n=''
for i in s:
    if 65<=ord(i)<=90 and 65<=ord(i)-k<=90:
        n+=chr(ord(i)-k)
    elif 97<=ord(i)<=122 and 97<=ord(i)-k<=122:
        n+=chr(ord(i)-k)
    elif 90 >= ord(i) >= 65 > ord(i)-k:
        if not (65-(ord(i)-k))%26:
            n+='A'
        else:
            n+=chr(90-(65-(ord(i)-k))%26+1)
    elif 122>=ord(i)>=97>ord(i)-k:
        if not (97-(ord(i)-k))%26:
            n+='a'
        else:
            n+=chr(122-(97-(ord(i)-k))%26+1)
print(n)
```

耗时：30min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250923155815576](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250923155815576.png)



### E28691: 字符串中的整数求和

http://cs101.openjudge.cn/pctbook/E28691/



思路：



代码

```python
s=input().split()
total=int(s[0][0]+s[0][1])+int(s[1][0]+s[1][1])
print(total)
```

耗时：1min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250923160435029](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250923160435029.png)



### M28664: 验证身份证号 

http://cs101.openjudge.cn/pctbook/M28664/



思路：



代码

```python
n=int(input())
dt={0:'1',1:'0',2:'X',3:'9',4:'8',5:'7',6:'6',7:'5',8:'4',9:'3',10:'2'}
for _ in range(n):
    ID=input()
    total=(int(ID[0])+int(ID[10]))*7+(int(ID[1])+int(ID[11]))*9+(int(ID[2])+int(ID[12]))*10+(int(ID[3])+int(ID[13]))*5+(int(ID[4])+int(ID[14]))*8+(int(ID[5])+int(ID[15]))*4+(int(ID[6])+int(ID[16]))*2+int(ID[7])+int(ID[8])*6+int(ID[9])*3
    if dt[total%11]==ID[-1]:
        print('YES')
    else:
        print('NO')
```

耗时：5min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250923161549051](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250923161549051.png)



### M28678: 角谷猜想

http://cs101.openjudge.cn/pctbook/M28678/


思路：



代码

```python
n=int(input())
while n!=1:
    if n%2==0:
        print(str(n)+'/'+'2'+'='+str(n//2))
        n=n//2
    elif n%2!=0:
        print(str(n)+'*'+'3'+'+'+'1'+'='+str(n*3+1))
        n=n*3+1
else:
    print('End')

```

耗时：5min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250923162330260](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250923162330260.png)



### M28700: 罗马数字与整数的转换

http://cs101.openjudge.cn/pctbook/M28700/



思路：

整数转罗马数字部分：

先将特殊罗马数字及其对应的整数以降序放在字典里（我这里应该使用列表，因为我的python版本字典还没有顺序，~~我在打这段话的时候才反应过来这一点~~），然后看整数n里分别包含多少个这些特殊的数字，最后将这些数字组装起来就是所要的罗马数字；

罗马数字转整数部分：

除遇到‘I’，‘X’，‘C'这几个字符时，都可以直接将字符对应的数字加到total上，当遇到‘I’，‘X’，‘C'时，需要判断他们的下一位是不是比他们更大，如果更大就需要减掉这个字符对应的数字，反之成立，最后就能得到所要的整数。

代码

```python
n=input()
if n[0].isdigit():
    num=int(n)
    Rome=''
    dt={1000:'M',900:'CM',500:'D',400:'CD',100:'C',90:'XC',50:'L',40:'XL',10:'X',9:'IX',5:'V',4:'IV',1:'I'}
    for i in dt.items():
        count=num//i[0]
        if count:
            Rome+=i[1]*count
            num=num%i[0]
        if num==0:
            break
    print(Rome)
else:
    dt={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
    pre_v=0
    total=0
    for i in reversed(n):
        if dt[i]<pre_v:
            total-=dt[i]
        else:
            total+=dt[i]
        pre_v=dt[i]
    print(total)
```

耗时：1h30min

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250923231643674](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250923231643674.png)



### 158B. Taxi

*special problem, greedy, implementation, 1100,  https://codeforces.com/problemset/problem/158/B



思路：

我一开始就想要把所有情况全部列出来，但是总是落下几种情况，后来想了一种循环套循环的方法，写在最后的学习收获当中了，虽然这种方法能够正确输出，但是会在test59，也就是输入了100000个1的那组数据TLE，所以说我只好回到列出所有情况的方法，最后一点一点试错才得出正确的代码。

代码

```python
import math
n=int(input())
s=input().split()
dt={1:0,2:0,3:0,4:0}
for i in range(n):
    dt[int(s[i])]+=1
total=dt[4]
if dt[2]==dt[3]==0:
    total+=math.ceil(dt[1]/4)
elif dt[1]==dt[2]==0:
    total+=dt[3]
elif dt[1]==dt[3]==0:
    total+=math.ceil(dt[2]/2)
elif dt[3]>=dt[1]>=0:
    total+=dt[3]+math.ceil(dt[2]/2)
elif 0<=dt[3]<dt[1] and math.ceil((dt[1]-dt[3])/2)>=dt[2]>=0:
    total+=dt[3]+dt[2]+math.ceil((dt[1]-dt[3]-dt[2]*2)/4)
elif 0<=dt[3]<dt[1] and 0<=math.ceil((dt[1]-dt[3])/2)<dt[2]:
    total+=dt[3]+math.ceil((dt[1]-dt[3])/2)+math.ceil((dt[2]-math.ceil((dt[1]-dt[3])/2))/2)
print(total)
```

耗时：3h

代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250923231534192](C:\Users\huawei\AppData\Roaming\Typora\typora-user-images\image-20250923231534192.png)



## 2. 学习总结和收获

<mark>如果作业题目简单，有否额外练习题目，比如：OJ“计概2025fall每日选做”、CF、LeetCode、洛谷等网站题目。</mark>

额外练习题目：每日选做0913-0920

通过E20742:泰波拿契數这道题，我学到了使用动态规划来降低递归过程中的时间复杂度。

通过M02786:Pell数列这道题我又学会了在动态规划中使用滚动数组来储存已经算过的值，以此来降低空间复杂度。

我在E28674:《黑神话：悟空》之加密这道题就卡了一段时间，原因是我不够熟悉ASCII码，通过这道题我对英文字母的ASCII码有了更深的理解。

在写M28700: 罗马数字与整数的转换这道题时，我一开始想要用穷举的方法来做，结果不出所料地超时了，以后在做题的时候我应该克服自己的惰性，不能只想着穷举做出来就可以了，应该先思考有没有数学上的或者算法上的方法。通过这道题我也学到了reversed（）这个函数。

在写158B. Taxi这道题时，我一开始就想要把所有的可能情况都列举出来，但是一直落下某些情况，所以我就转换思路写了一个循环套循环的方法，代码如下：

```python
n=int(input())
s=input().split()
total=n
for i in range(n-1):
    b=False
    if s[i]!='4':
        for j in range(i+1,n):
            if int(s[i])+int(s[j])==4:
                s[j]=4
                total-=1
                b=True
                break
        if not b:
            for j in range(i+1,n):
                if int(s[i])+int(s[j])==3:
                    s[j]=3
                    total-=1
                    b=True
                    break
        if not b:
            for j in range(i+1,n):
                if int(s[i])+int(s[j])==2:
                    s[j]=2
                    total-=1
                    b=True
                    break
print(total)
```

虽然这段代码能够对测试数据正确输出，但是在test59，也就是输入了100000个1的那组数据会TLE，所以我又被迫回到列举所有情况的方法，最终ac，刚刚看了题解只有三行感觉被降维打击了，还是需要再多培养数学思维。

