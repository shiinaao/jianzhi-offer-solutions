[TOC]

# 1.二维数组中的查找

在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

```python
# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        flag = False
        for i in array:
            if target in i:
            	flag = True
        return flag
```

# 2.替换空格

请实现一个函数，将一个字符串中的空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

```py
# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        s = s.replace(' ', '%20')
        return s
```

# 3.从尾到头打印链表

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        res = []
        while listNode:
            res.append(listNode.val)
            listNode = listNode.next
        return res[::-1]
```

# 4.重建二叉树

输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if len(pre) == 0:
            return None
        cur = TreeNode(pre[0])
        index = tin.index(pre[0])
        cur.left = self.reConstructBinaryTree(pre[1:index+1], tin[:index])
        cur.right = self.reConstructBinaryTree(pre[index+1:], tin[index+1:])
        return cur
```

# 5.用两个栈实现队列

用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack = []
    def push(self, node):
        # write code here
        self.stack.append(node)
    def pop(self):
        # return xx
        res = self.stack[0]
        self.stack = self.stack[1:]
        return res
```

# 6.旋转数组的最小数字

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

```python
# -*- coding:utf-8 -*-
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        #return min(rotateArray)
        if len(rotateArray) == 0:
            return 0
        for i in range(len(rotateArray)):
            if rotateArray[i] > rotateArray[i+1]:
                return rotateArray[i+1]
        else:
            return -1

```

# 7.斐波那契数列(循环实现)

```python
# -*- coding:utf-8 -*-
class Solution:
    def Fibonacci(self, n):
        # write code here
        if n <= 1:
            return n
        a, b = 0, 1
        for i in range(n):
            a, b = b, a + b
        return a
```

# 8.跳台阶

一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

```python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloor(self, number):
        # write code here
        #if number <= 2:
        #    return number
        #else:
        #    return self.jumpFloor(number - 1) + self.jumpFloor(number - 2)
        if number <= 1:
            return number
        a, b = 0, 1
        for i in range(number):
            a, b = b, a+b
        return b
```

# 9.变态跳台阶

一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。[公式求解参考](http://blog.csdn.net/hackbuteer1/article/details/6686747)

```python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloorII(self, number):
        # write code here
        if number <= 2:
            return number
        else:
            return 2 * self.jumpFloorII(number - 1)
```

# 10.矩形覆盖(斐波那契变形)

我们可以用`2*1`的小矩形横着或者竖着去覆盖更大的矩形。请问用n个`2*1`的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

# 11.二进制中1的个数

输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。

```python
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1(self, n):
        # write code here
        return sum([(n>>i & 1) for i in range(0,32)])
```

# 12.数值的整数次方

```python
# -*- coding:utf-8 -*-
class Solution:
    def Power(self, base, exponent):
        # write code here
        return base ** exponent
```

# 13.调整数组顺序使奇数位于偶数前面

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

```python
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        # write code here
        oddL=filter(lambda x: x%2,array)
        evenL=filter(lambda x: not x%2,array)
        L = oddL + evenL
        return L
```

# 14.链表中倒数第k个节点

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        li = []
        while head:
            li.append(head)
            head = head.next
        if k>len(li) or k<1:
            return None
        return li[-k]
```

# 15.翻转链表

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        if not pHead or not pHead.next:
            return pHead
        pre = pHead
        cur = pHead.next
        pre.next = None
        while cur:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        return pre
```

# 16.合并两个排序链表

输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        mergeHead = ListNode(90)
        p = mergeHead
        while pHead1 and pHead2:
            if pHead1.val >= pHead2.val:
                mergeHead.next = pHead2
                pHead2 = pHead2.next
            else:
                mergeHead.next = pHead1
                pHead1 = pHead1.next
                  
            mergeHead = mergeHead.next
        if pHead1:
            mergeHead.next = pHead1
        elif pHead2:
            mergeHead.next = pHead2
        return p.next
```

# 17.树的子结构

输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        if not pRoot1 or not pRoot2:
            return False
        return self.is_subtree(pRoot1, pRoot2) or self.HasSubtree(pRoot1.left, pRoot2) or self.HasSubtree(pRoot1.right, pRoot2)
     
    def is_subtree(self, A, B):
        if not B:
            return True
        if not A or A.val != B.val:
            return False
        return self.is_subtree(A.left,B.left) and self.is_subtree(A.right, B.right)
```

# 18.二叉树镜像

二叉树的镜像定义：

​	源二叉树 
​    	    8
​    	   /  \
​    	  6   10
​    	 / \  / \
​    	5  7 9 11
​    	镜像二叉树
​    	    8
​    	   /  \
​    	  10   6
​    	 / \  / \
​    	11 9 7  5

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        # write code here
        if not root:
            return None
        root.left, root.right = root.right, root.left
        if root.left:
        	self.Mirror(root.left)
        if root.right:
        	self.Mirror(root.right)
        return root
```

# 19.顺时针打印矩阵

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

```python
# -*- coding:utf-8 -*-
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        # write code here
        result = []
        while(matrix):
            result += matrix.pop(0)
            if not matrix or not matrix[0]:
                break
            matrix = list(reversed(list(zip(*matrix))))
        return result
```

# 20.包含min函数的栈

定义栈的数据结构，请在该类型中实现一个能够得到栈最小元素的min函数。

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack = []
    def push(self, node):
        # write code here
        self.stack.append(node)
    def pop(self):
        # write code here
        return self.stack.pop(-1)
    def top(self):
        # write code here
        return self.stack[0]
    def min(self):
        # write code here
        return min(self.stack)
```

# 21.栈的压入, 弹出序列(没看懂

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4，5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

```python
# -*- coding:utf-8 -*-
class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here
        # 真心没看懂
        if not pushV or len(pushV) != len(popV):
            return False
        stack = []
        for i in pushV:
            stack.append(i)
            while len(stack) and stack[-1] == popV[0]:
                stack.pop()
                popV.pop(0)
        if len(stack):
            return False
        return True
```

# 22.从上往下打印二叉树

从上往下打印出二叉树的每个节点，同层节点从左至右打印。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        if not root:
            return []
        q = [root]
        res = []
        while q:
            cur = q.pop(0)
            res.append(cur.val)
            if cur.left:
                q.append(cur.left)
            if cur.right:
                q.append(cur.right)
        return res
```

# 23.二叉搜索树的后序遍历序列(有点懂了 ^&

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

```python
# -*- coding:utf-8 -*-
class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        s = sequence
        if not s:
            return False
        if len(s) <= 2:
            return True
        # 后续遍历最后一个节点是 根节点
        root = s[-1]
        for i in range(len(s)):
            if s[i] > root:
                break
        # 此时 [:i] 为左子树, [i:-1] 为右子树
        for j in range(i, len(s)):
            if s[j] < root:
                return False
        left = right = True
        if i > 0:
            left = self.VerifySquenceOfBST(s[:i])
        if i < len(s)-1:
            right = self.VerifySquenceOfBST(s[i:-1])
        return left and right
```

# 24.二叉树和为某一值的路径

输入一颗二叉树和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        # write code here
        if not root:
            return []
        if root and not root.left and not root.right and root.val == expectNumber:
            return [[root.val]]
        res = []
        left = self.FindPath(root.left, expectNumber-root.val)
        right = self.FindPath(root.right, expectNumber-root.val)
        for i in left+right:
            res.append([root.val]+i)
        return res
```

# 25.复杂链表的复制 ^&

输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

```python
# -*- coding:utf-8 -*-
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
def iter_node(root):
    while root:
        yield root
        root = root.next

class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        # 使用字典 mem 索引链表位置
        mem = dict()
        for i, n in enumerate(iter_node(pHead)):
            mem[id(n)] = i
        li = [RandomListNode(n.label) for n in iter_node(pHead)]
        for r, d in zip(iter_node(pHead), li):
            if r.next:
                d.next = li[mem[id(r.next)]]
            if r.random:
                d.random = li[mem[id(r.random)]]
        return li[0] if li else None
```



# 26.二叉搜索树与双向链表 ^&

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

> 中序遍历获取排序后的顺序, 然后按顺序遍历修改指针指向

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def midOrder(self, root):
        if not root:
            return []
        return self.midOrder(root.left) + [root] + self.midOrder(root.right)
    def Convert(self, pRootOfTree):
        # write code here
        mid = self.midOrder(pRootOfTree)
        if not mid:
            return None
        if len(mid) == 1:
            return pRootOfTree
        mid[0].left = mid[-1].right = None
        mid[0].right = mid[1]
        mid[-1].left = mid[-2]
        for item in range(1, len(mid)-1):
            mid[item].left = mid[item - 1]
            mid[item].right = mid[item + 1]
        return mid[0]
```



# 27.字符串的排列

输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。 

```python
# -*- coding:utf-8 -*-
from itertools import permutations
class Solution:
    def Permutation(self, ss):
        # write code here
        if not ss:
            return []
        res = []
        for i in permutations(ss):
            res.append(''.join(i))
        res = list(set(res))
        return sorted(res)
```

# 28.数组中出现次数超过一半的数字

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0

```python
# -*- coding:utf-8 -*-
from collections import Counter
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        c = Counter(numbers)
        if c.most_common(1)[0][1] > len(numbers)//2:
            return c.most_common(1)[0][0]
        else:
            return 0
```

# 29.最小的k个数

输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        if not tinput or not k or len(tinput) < k:
            return []
        t = sorted(tinput)
        return t[:k]
```

# 30.连续子数组的最大和

HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。你会不会被他忽悠住？(子向量的长度至少是1)

```python
# -*- coding:utf-8 -*-
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        if not array:
            return 0
        res = array[0]
        sumn = 0
        for i in array:
            if sumn <= 0:
                sumn = i
            else:
                sumn += i
            if sumn > res:
                res = sumn
        return res
```

# 31.整数中1出现的次数

求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数。

```python
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        # write code here
        s = ''.join(map(str, range(n+1)))
        return s.count('1')
```

# 32.把数组排成最小数

输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

```python
# -*- coding:utf-8 -*-
from itertools import permutations
class Solution:
    def PrintMinNumber(self, numbers):
        # write code here
        if not numbers:
            return ''
        li = []
        for item in permutations(map(str, numbers)):
            li.append(int(''.join(item)))
        return min(li)
```

# 33.丑数(这很数学

把只包含因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetUglyNumber_Solution(self, index):
        # write code here
        if index < 1:
            return 0
          
        res = [1]
        t2 = t3 = t5 = 0
        nextIdx = 1
        while nextIdx < index:
            minNum = min(res[t2]*2,res[t3]*3,res[t5]*5)
            res.append(minNum)
              
            while res[t2]*2 <= minNum:
                t2 += 1
            while res[t3]*3 <= minNum:
                t3 += 1
            while res[t5]*5 <= minNum:
                t5 += 1
            nextIdx += 1
        return res[nextIdx-1]
```

# 34.第一个只出现一次的字符

在一个字符串(1<=字符串长度<=10000，全部由大写字母组成)中找到第一个只出现一次的字符,并返回它的位置

```python
# -*- coding:utf-8 -*-
class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code here
        if not s:
            return -1
        for i, v in enumerate(s):
            if s.count(v) == 1:
                return i
        return -1
```

# 35.数组中的逆序对(归并排序)

**牛客网环境不通过, 可能是递归太深导致超时**

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007 

输入例子: 1,2,3,4,5,6,7,0

输出例子: 7

```python
# -*- coding:utf-8 -*-
class Solution:
    def InversePairs(self, data):
        # write code here
        self.num = 0
        self.merge_sort(data)
        return self.num

    def merge_sort(self, data):
        if len(data) <= 1:
            return data
        mid = len(data) // 2
        left = self.merge_sort(data[:mid])
        right = self.merge_sort(data[mid:])
        return self.merge(left, right)

    def merge(self, left, right):
        li = []
        while len(left) > 0 and len(right) > 0:
            if left[-1] > right[-1]:
                self.num += len(right)
                # li.append(left.pop())
                li.insert(0, left.pop())
            else:
                # li.append(right.pop())
                li.insert(0, right.pop())
        if len(left) > 0:
            li = left + li
        else:
            li = right + li
        return li
```

# 36.两个链表的第一个公共节点

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        l1 = []
        while pHead1:
            l1.append(pHead1.val)
            pHead1 = pHead1.next
        while pHead2:
            if pHead2.val in l1:
                return pHead2
            else:
                pHead2 = pHead2.next
```

# 37.数字在排序数组中出现的次数

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        return data.count(k)
```

# 38.二叉树深度

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def TreeDepth(self, pRoot):
        # write code here
        if not pRoot:
            return 0
        return max(self.TreeDepth(pRoot.left), self.TreeDepth(pRoot.right)) + 1
```

# 39.平衡二叉树

输入一棵二叉树，判断该二叉树是否是平衡二叉树。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def TreeDepth(self, pRoot):
        # write code here
        if not pRoot:
            return 0
        return max(self.TreeDepth(pRoot.left), self.TreeDepth(pRoot.right)) + 1
    
    def IsBalanced_Solution(self, pRoot):
        # write code here
        if not pRoot:
            return True
        if abs(self.TreeDepth(pRoot.left) - self.TreeDepth(pRoot.right)) <= 1:
            return self.IsBalanced_Solution(pRoot.left) and self.IsBalanced_Solution(pRoot.right)
        else:
            return False
        
```

# 40.数组中只出现一次的数字

一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。

```python
# -*- coding:utf-8 -*-
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        # write code here
        res = []
        for item in array:
            if array.count(item) == 1:
                res.append(item)
        if len(res) == 2:
            return res
        else:
            return None
```

# 41.和为S的连续整数序列

小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck! 

```python
# -*- coding:utf-8 -*-
class Solution:
    def FindContinuousSequence(self, tsum):
        # write code here
        if tsum < 3:
            return []
        small = 1
        big = 2
        mid = tsum + 1 // 2
        cursum = small + big
        res = []
        while small < mid:
            if cursum == tsum:
                res.append(range(small, big+1))
                big += 1
                cursum += big
            elif cursum > tsum:
                cursum -= small
                small += 1
            else:
                big += 1
                cursum += big
        return res
```

# 42.和为S的两个数字

输入一个递增排序的数组和一个数字S，在数组中查找两个数，是的他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。 

```python
# -*- coding:utf-8 -*-
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        array = sorted(array)
        res = []
        for i, v in enumerate(array):
            for v1 in array[i:]:
                if (v + v1) == tsum:
                    res.append([v, v1])
        if res:
            return res[0]
        else:
            return res
```

# 43.左旋转字符串

汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！

```python
# -*- coding:utf-8 -*-
class Solution:
    def LeftRotateString(self, s, n):
        # write code here
        return s[n:] + s[:n]
```

# 44.翻转单词顺序

牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？

```python
# -*- coding:utf-8 -*-
class Solution:
    def ReverseSentence(self, s):
        # write code here
        res = s.split(' ')
        return ' '.join(res[::-1])
```

# 45.扑克牌顺子

LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何。为了方便起见,你可以认为大小王是0。

```python
# -*- coding:utf-8 -*-
class Solution:
    def IsContinuous(self, numbers):
        # write code here
        if len(numbers) != 5:
            return False
        numbers.sort()
        num_of_zero = numbers.count(0)
        num_of_gap = 0
        small = num_of_zero
        big = small + 1
        while big < len(numbers):
            if numbers[small] == numbers[big]:
                return False
            num_of_gap += numbers[big] - numbers[small] - 1
            small = big
            big += 1
        if num_of_gap > num_of_zero:
            return False
        return True
```

# 46.圆圈中最后剩下的数

每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)

```python
# -*- coding:utf-8 -*-
class Solution:
    def LastRemaining_Solution(self, n, m):
        # write code here
        if n == 0:
            return -1
        li = range(n)
        i = 0
        for num in range(n, 1, -1):
            i = (i + m - 1) % num
            li.pop(i)
        return li[0]
```

# 47.求1+2+3+...+n

求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）

```python
# -*- coding:utf-8 -*-
class Solution:
    def Sum_Solution(self, n):
        # write code here
        return sum(range(n+1))
```

# 48.不用加减乘除做加法

```python
# -*- coding:utf-8 -*-
class Solution:
    def Add(self, num1, num2):
        # write code here
        return sum([num1, num2])
```

# 49.把字符串转换成整数

将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0 

```python
# -*- coding:utf-8 -*-
class Solution:
    def StrToInt(self, s):
        # write code here
        try:
            return int(s)
        except Exception as e:
            return 0
```

# 50.数组中重复的数字

在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是重复的数字2或者3。

```python
# -*- coding:utf-8 -*-
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # write code here
        for item in numbers:
            if numbers.count(item) > 1:
                duplication[0] = item
                return True
        else:
            return False
```

# 51.构建乘积数组(reduce)

给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。

```python
# -*- coding:utf-8 -*-
class Solution:
    def multiply(self, A):
        # write code here
        # 复制的, 666
        B = []
        for i in range(len(A)):
            B.append(reduce(lambda x,y:x*y, (A[:i] + A[i+1:])))
        return B
```

# 52.正则表达式匹配 ^&

请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配

```python
# -*- coding:utf-8 -*-
class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        # write code here
        if not pattern and not s:
            return True
        if not pattern and s:
            return False
        if len(pattern) > 1 and pattern[1] == '*':
            if s and (pattern[0] == s[0] or pattern[0] == '.' and s):
                # .* 或者 a* 的后续匹配情况
                # 依次为 [.* 匹配 ''] or [.* 匹配字符多个字符] or [.* 匹配一个字符]
                return self.match(s, pattern[2:]) \
                    or self.match(s[1:], pattern) \
                    or self.match(s[1:], pattern[2:])
            else:
                # a* 不匹配的情况, 栗子: 'a*' != 'b'
                return self.match(s, pattern[2:])
        # 普通匹配的情况, 栗子: 'a' == 'a'
        if s and (pattern[0] == s[0] or pattern[0] == '.' and s):
            return self.match(s[1:], pattern[1:])
        else:
            return False
```



# 53.表示数值的字符串

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。

```python
# -*- coding:utf-8 -*-
class Solution:
    # s字符串
    def isNumeric(self, s):
        # write code here
        try:
            float(s)
        except Exception as e:
            return False
        else:
            return True
```

# 54.字符流中第一个不重复的字符

请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。 

```python
# -*- coding:utf-8 -*-
class Solution:
    # 返回对应char
    def __init__(self):
        self.s = ''
        self.dc = {}
        
    def FirstAppearingOnce(self):
        # write code here
        for i in self.s:
            if self.dc[i] == 1:
                return i
        return '#'
        
    def Insert(self, char):
        # write code here
        self.s += char
        if char in self.dc:
            self.dc[char] += 1
        else:
            self.dc[char] = 1
```

# 55.链表中环的入口节点

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def EntryNodeOfLoop(self, pHead):
        # write code here
        li = []
        while pHead:
            if pHead in li:
                return pHead
            li.append(pHead)
            pHead = pHead.next
        return None
```

# 56.删除链表中重复的节点(去重,删除)

在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

```python
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        # 去重
        #h = pHead
        #while pHead and pHead.next:
        #    while pHead.val == pHead.next.val:
        #        pHead.next = pHead.next.next
        #    pHead = pHead.next
        #return h
        if pHead is None or pHead.next is None:
            return pHead
        pre = ListNode(0)
        pre.next = pHead
        pHead = pre
         
        while pHead.next and pHead.next.next:
            if pHead.next.val == pHead.next.next.val:
                value = pHead.next.val
                while pHead.next and pHead.next.val == value:
                    pHead.next = pHead.next.next
            else:
                pHead = pHead.next
        return pre.next
```

# 57.二叉树的下一个节点

给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

```python
# -*- coding:utf-8 -*-
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None
class Solution:
    def __init__(self):
        self.mid = []
        
    def midOrder(self, root):
        if root.left:
            self.midOrder(root.left)
        self.mid.append(root)
        if root.right:
            self.midOrder(root.right)
    
    def GetNext(self, node):
        root = node
        while root.next:
            root = root.next
        self.midOrder(root)
        for i, v in enumerate(self.mid):
            if v == node and i != len(self.mid)-1:
                return self.mid[i + 1]
        return None
            
```

# 58.对称二叉树

请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def isSymmetrical(self, root):
        # write code here
        if not root:
            return True
        return self.isEq(root.left, root.right)
        
    def isEq(self, l, r):
        if not l and not r:
            return True
        if l and r:
            if l.val == r.val:
                return self.isEq(l.left, r.right) and self.isEq(l.right, r.left)
        return False
```

# 59.按之自形顺序打印二叉树

请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Print(self, root):
        # write code here
        if not root:
            return []
        q = [root]
        res = []
        k = 1
        while q:
            tmp = []
            x = []
            for item in q:
                tmp.append(item.val)
                if item.left:
                    x.append(item.left)
                if item.right:
                    x.append(item.right)
            q = x
            if k == -1:
                tmp.reverse()
            res.append(tmp)
            k = -k
        return res
```

# 60.把二叉树打印成多行

从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, root):
        # write code here
        if not root:
            return []
        q = [root]
        res = []
        while q:
            tmp = []
            x = []
            for item in q:
                tmp.append(item.val)
                if item.left:
                    x.append(item.left)
                if item.right:
                    x.append(item.right)
            q = x
            res.append(tmp)
        return res
```

# 61.序列化二叉树

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

def inner_deserialize(text, index):
    if index[0] >= len(text) or text[index[0]] == '#':
        index[0] += 1
        return None
    node = TreeNode(int(text[index[0]]))
    index[0] += 1
    node.left = inner_deserialize(text, index)
    node.right = inner_deserialize(text, index)
    return node

class Solution:
    def Serialize(self, root):
        # write code here
        if not root:
            return '#,'
        else:
            return str(root.val) + ',' + self.Serialize(root.left) + self.Serialize(root.right)
        
    def Deserialize(self, s):
        # write code here
        # 序列化勉强能看明白, 反序列化理解不能
        s = s.split(',')
        return inner_deserialize(s, [0])
```

# 62.二叉搜索树的第k个节点

给定一颗二叉搜索树，请找出其中的第k大的结点。例如， 5 / \ 3 7 /\ /\ 2 4 6 8 中，按结点数值大小顺序第三个结点的值为4。

```python
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回对应节点TreeNode
    def __init__(self):
        self.mid = []
    
    def midOrder(self, root):
        if not root:
            return None
        if root.left:
            self.midOrder(root.left)
        self.mid.append(root)
        if root.right:
            self.midOrder(root.right)
    
    def KthNode(self, root, k):
        # write code here
        self.midOrder(root)
        if k <= 0 or k > len(self.mid):
            return None
        # 不是第k大, 而是第k小
        return self.mid[k-1]

```

# 63.数据流中的中位数

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.li = []
    
    def Insert(self, num):
        # write code here
        self.li.append(num)
    
    # *arg 参数没有作用, 但是后台验证时会添加一个参数
    def GetMedian(self, *arg):
        # write code here
        self.li.sort()
        n = len(self.li)
        
        if n % 2 == 1:
            return self.li[n//2]
        else:
            return (self.li[n//2-1] + self.li[n//2]) / 2.0
```

# 64.滑动窗口最大值

给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。

```python
# -*- coding:utf-8 -*-
class Solution:
    def maxInWindows(self, num, size):
        # write code here
        if not num or size < 1:
            return []
        res = []
        l = len(num) - size + 1
        for i in range(l):
            res.append(max(num[i:i+size]))
        return res
```

# 65.矩阵中的路径

请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。 例如[a b c e s f c s a d e e]是3*4矩阵，其包含字符串"bcced"的路径，但是矩阵中不包含“abcb”路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。

```python
# -*- coding:utf-8 -*-
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        # write code here
        if not matrix or rows < 1 or cols < 1 or not path:
            return False
        visited = [0] * (rows * cols)
        cur = 0
        matrix = list(matrix)
        for row in range(rows):
            for col in range(cols):
                if self.haspath_core(matrix, rows, cols, row, col, path, cur, visited):
                    return True
        return False

    def haspath_core(self, matrix, rows, cols, row, col, path, cur, visited):
        if len(path) == cur:
            return True
        has_path = False
        if 0 <= row < rows and 0 <= col < cols and matrix[row*cols+col] == path[cur] \
                and not visited[row*cols+col]:
            cur += 1
            visited[row*cols+col] = True
            has_path = self.haspath_core(matrix, rows, cols, row+1, col, path, cur, visited) \
                or self.haspath_core(matrix, rows, cols, row-1, col, path, cur, visited) \
                or self.haspath_core(matrix, rows, cols, row, col+1, path, cur, visited) \
                or self.haspath_core(matrix, rows, cols, row, col-1, path, cur, visited)
            if not has_path:
                cur -= 1
                visited[row*cols+col] = False
        return has_path
```

# 66.机器人的运动范围

地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？

```python
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.count = 0

    def movingCount(self, threshold, rows, cols):
        # write code here
        board = [[0 for _ in range(cols)] for _ in range(rows)]

        def block(row, col):
            s = sum(map(int, str(row) + str(col)))
            return s > threshold

        def traverse(r, c):
            if not (0 <= r < rows and 0 <= c < cols):
                return None
            if board[r][c] != 0:
                return None
            if board[r][c] == -1 or block(r, c):
                board[r][c] = -1
                return None
            board[r][c] = 1
            self.count += 1
            traverse(r + 1, c)
            traverse(r - 1, c)
            traverse(r, c + 1)
            traverse(r, c - 1)

        traverse(0, 0)
        return self.count
```

