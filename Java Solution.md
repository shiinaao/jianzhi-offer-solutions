剑指offer - Java

[TOC]

# **1.二维数组中的查找**

在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
方法一：从右上角开始向左下方向移动比较

    public class Solution {
        public boolean Find(int target, int [][] array) {
            int row = 0;
            int col = array[0].length-1;
            while (row <= array.length-1 && col >=0) {
                if (target == array\[row\][col]) {
                    return true;
                } else if (target > array\[row\][col]) {
                    row++;
                } else {
                    col--;
                }
            }
            return false;
        }
    }

方法二：每行进行二分查找

    public class Solution {
        public boolean Find(int target, int [][] array) {
            for(int i=0;i<array.length;i++){
                int low=0;
                int high=array[i].length-1;
                while(low<=high){
                    int mid=(low+high)/2;
                    if(target>array\[i\][mid])
                        low=mid+1;
                    else if(target<array\[i\][mid])
                        high=mid-1;
                    else
                        return true;
                }
            }
            return false;
        }
    }
# **2.替换空格**

请实现一个函数，将一个字符串中的空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

    public class Solution {
        public String replaceSpace(StringBuffer str) {
            return str.toString().replaceAll(" ", "%20");
        }
    }
# **3.从尾到头打印链表**
    /**
    *    public class ListNode {
    *        int val;
    *        ListNode next = null;
    *
    *        ListNode(int val) {
    *            this.val = val;
    *        }
    *    }
    *
    */
    import java.util.ArrayList;
    import java.util.Collections;
    public class Solution {
        public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
            ArrayList<Integer> res = new ArrayList<>();
            if (listNode == null) return res;
            while (listNode.next != null) {
                res.add(listNode.val);
                listNode = listNode.next;
            }
            res.add(listNode.val);
            Collections.reverse(res);
            return res;
        }
    }
# **4.重建二叉树**

输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

    /**
     * Definition for binary tree
     * public class TreeNode {
     *     int val;
     *     TreeNode left;
     *     TreeNode right;
     *     TreeNode(int x) { val = x; }
     * }
     */
    import java.util.*;
    import java.util.stream.Collectors;
    public class Solution {
        public TreeNode reConstructBinaryTree(int [] pre,int [] in) {
            if (pre.length == 0) return null;
            TreeNode cur = new TreeNode(pre[0]);
            //int index = Arrays.stream(in).boxed().collect(Collectors.toList()).indexOf(pre[0]);
            int index = 0;
            for (int i=0; i<in.length; i++) {
                if (in[i] == pre[0]) {
                    index = i;
                    break;
                }
            }
            cur.left = reConstructBinaryTree(Arrays.copyOfRange(pre, 1, index+1), Arrays.copyOfRange(in, 0, index));
            cur.right = reConstructBinaryTree(Arrays.copyOfRange(pre, index+1, pre.length), Arrays.copyOfRange(in, index+1, in.length));
            return cur;
        }
    }

下面这个略麻烦了。。

    /**
     * Definition for binary tree
     * public class TreeNode {
     *     int val;
     *     TreeNode left;
     *     TreeNode right;
     *     TreeNode(int x) { val = x; }
     * }
     */
    import java.util.*;
    import java.util.stream.Collectors;
    public class Solution {
        public TreeNode reConstructBinaryTree(int [] pre,int [] in) {
            List<Integer> preList = new ArrayList<>(Arrays.stream(pre).boxed().collect(Collectors.toList()));
            List<Integer> inList = new ArrayList<>(Arrays.stream(in).boxed().collect(Collectors.toList()));
            return getTreeNode(preList, inList);
        }
    
        public TreeNode getTreeNode(List<Integer> preList, List<Integer> inList) {
            if (preList.size() == 0) return null;
            TreeNode cur = new TreeNode(preList.get(0));
            int index = inList.indexOf(preList.get(0));
            cur.left = getTreeNode(preList.subList(1, index+1), inList.subList(0, index));
            cur.right = getTreeNode(preList.subList(index+1, preList.size()), inList.subList(index+1, inList.size()));
            return cur;
        }
    }
# **5.用两个栈实现队列**

用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。

    import java.util.Stack;
    
    public class Solution {
        Stack<Integer> stack1 = new Stack<Integer>();
        Stack<Integer> stack2 = new Stack<Integer>();
        
        public void push(int node) {
            stack1.push(node);
        }
    
        public int pop() {
            if (stack2.empty()) {
                while (!stack1.empty()) {
                    stack2.push(stack1.pop());
                }
            }
            return stack2.pop();
        }
    }
# **6.旋转数组的最小数字**

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

    import java.util.ArrayList;
    public class Solution {
        public int minNumberInRotateArray(int [] array) {
            for (int i=1; i<array.length; i++) {
                if (array[i-1] > array[i]) return array[i];
            }
            return array[0];
        }
    }
# **7.斐波那契数列(循环实现)**
    public class Solution {
        public int Fibonacci(int n) {
            if (n < 2) return n;
            int a = 0;
            int b = 1;
            for (int i=1; i<=n; i++) {
                int temp = a + b;
                a = b;
                b = temp;
            }
            return a;
        }
    }
# **8.跳台阶**

一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法。（斐波那契变形题）

    public class Solution {
        public int JumpFloor(int target) {
            if (target <= 2) return target;
            return JumpFloor(target-1) + JumpFloor(target-2);
        }
    }
# **9.变态跳台阶**

一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。[公式求解参考](http://blog.csdn.net/hackbuteer1/article/details/6686747)

    public class Solution {
        public int JumpFloorII(int target) {
            if (target <= 2) return target;
            else return 2 * JumpFloorII(target-1);
        }
    }
# **10.矩形覆盖(斐波那契变形)**

我们可以用`2*1`的小矩形横着或者竖着去覆盖更大的矩形。请问用n个`2*1`的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

    public class Solution {
        public int RectCover(int target) {
            if (target < 2) return target;
            int a = 0;
            int b = 1;
            for (int i=1; i<=target; i++) {
                int temp = a + b;
                a = b;
                b = temp;
            }
            return b;
        }
    }
# **11.二进制中1的个数**

输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
一开始用向右移位的方法做，但是过不了测试

    public class Solution {
        public int NumberOf1(int n) {
            int count = 0;
            while (n != 0) {
              count++;
              n &= (n-1);
            }
            return count;
        }
    }
# **12.数值的整数次方**

通过&1和>>1逐位读取1101，为1时将该位代表的乘数累乘到最终结果

    public class Solution {
        public double Power(double base, int exponent) {
            double res = 1, curr = base;
            if (exponent == 0) return 1;
            int e = exponent > 0 ? exponent : -exponent;
            while (e != 0) {
                if ((e & 1) == 1)
                    res *= curr;
                e >>= 1;
                curr *= curr;
            }
            return exponent > 0 ? res : 1/res;
      }
    }
# **13.调整数组顺序使奇数位于偶数前面**

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
遇到相邻的前偶后奇就交换，类似冒泡

    public class Solution {
        public void reOrderArray(int [] array) {
            for (int i=0; i<array.length; i++) {
                for (int j=array.length-1; j>i; j--) {
                    if (array[j]%2==1 && array[j-1]%2==0) {
                        int temp = array[j-1];
                        array[j-1] = array[j];
                        array[j] = temp;
                    }
                }
            }
        }
    }
# **14.链表中倒数第k个节点**

使用 pre，curr 两个标记，pre 先走 k 步，然后两个同步前进，pre 到达尾部时，curr 就是倒数第 k 个节点

    /*
    public class ListNode {
        int val;
        ListNode next = null;
    
        ListNode(int val) {
            this.val = val;
        }
    }*/
    public class Solution {
        public ListNode FindKthToTail(ListNode head,int k) {
            if (head == null || k < 1)
                return null;
            ListNode pre = head;
            ListNode curr = head;
            for (int i=1; i < k; i++) {
                if (pre.next != null)
                    pre = pre.next;
                else
                    return null;
            }
            while (pre.next != null) {
                pre = pre.next;
                curr = curr.next;
            }
            return curr;
        }
    }
# **15.翻转链表**
    /*
    public class ListNode {
        int val;
        ListNode next = null;
    
        ListNode(int val) {
            this.val = val;
        }
    }*/
    public class Solution {
        public ListNode ReverseList(ListNode head) {
            if (head == null)
                return null;
            ListNode pre = null;
            ListNode next = null;
            while (head != null) {
                next = head.next;
                head.next  = pre;
                pre = head;
                head = next;
            }
            return pre;
        }
    }
# **16.合并两个排序链表**

输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

    /*
    public class ListNode {
        int val;
        ListNode next = null;
    
        ListNode(int val) {
            this.val = val;
        }
    }*/
    public class Solution {
        public ListNode Merge(ListNode list1,ListNode list2) {
            ListNode pre = new ListNode(0);
            ListNode head = pre;
            while (list1 != null && list2 != null) {
                if (list1.val < list2.val) {
                    pre.next = list1;
                    list1 = list1.next;
                } else {
                    pre.next = list2;
                    list2 = list2.next;
                }
                pre = pre.next;
            }
            if (list1 != null)
                pre.next = list1;
            else if (list2 != null)
                pre.next = list2;
            return head.next;
        }
    }
# **17.树的子结构**

输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

    /**
    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;
    
        public TreeNode(int val) {
            this.val = val;
    
        }
    
    }
    */
    public class Solution {
        public boolean HasSubtree(TreeNode root1,TreeNode root2) {
            if (root1 == null || root2 == null)
                return false;
            return isSubTree(root1, root2) || isSubTree(root1.left, root2) || isSubTree(root1.right, root2);
        }
        
        public boolean isSubTree(TreeNode a, TreeNode b) {
            if (b == null)
                return true;
            if (a == null || a.val != b.val)
                return false;
            return isSubTree(a.left, b.left) && isSubTree(a.right, b.right);
        }
    }
# **18.二叉树镜像**

二叉树的镜像定义：

    /**
    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;
    
        public TreeNode(int val) {
            this.val = val;
    
        }
    
    }
    */
    public class Solution {
        public void Mirror(TreeNode root) {
            if (root != null){
                TreeNode temp = root.left;
                root.left = root.right;
                root.right = temp;
                Mirror(root.left);
                Mirror(root.right);
            }
        }
    }
# **19.顺时针打印矩阵**

输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
真让人头大.jpg 出不来了

    import java.util.ArrayList;
    public class Solution {
        public ArrayList<Integer> printMatrix(int [][] array) {
            ArrayList<Integer> result = new ArrayList<Integer> ();
            if(array.length==0) return result;
            int n = array.length, m = array[0].length;
            if(m==0) return result;
            int layers = (Math.min(n,m)-1)/2+1;//这个是层数
            for(int i=0;i<layers;i++){
                for(int k = i;k<m-i;k++) result.add(array\[i\][k]);//左至右
                for(int j=i+1;j<n-i;j++) result.add(array\[j\][m-i-1]);//右上至右下
                for(int k=m-i-2;(k>=i)&&(n-i-1!=i);k--) result.add(array\[n-i-1\][k]);//右至左
                for(int j=n-i-2;(j>i)&&(m-i-1!=i);j--) result.add(array\[j\][i]);//左下至左上
            }
            return result;
        }
    }
# **20.包含min函数的栈**

定义栈的数据结构，请在该类型中实现一个能够得到栈最小元素的min函数。
用一个栈data保存数据，用另外一个栈min保存依次入栈最小的数
比如，data中依次入栈，5, 4, 3, 8, 10, 11, 12, 1

     则min依次入栈，5, 4, 3, no, no, no, no, 1
    import java.util.Stack;
    
    public class Solution {
      Stack<Integer> data = new Stack<>();
      Stack<Integer> min = new Stack<>();
    
      public void push(int node) {
        data.push(node);
        if (min.empty() || node < min.peek())
          min.push(node);
      }
    
      public void pop() {
        int temp = data.pop();
        if (temp == min.peek())
          min.pop();
      }
    
      public int top() {
        return data.peek();
      }
    
      public int min() {
        return min.peek();
      }
    }
# **21.栈的压入, 弹出序列（ArrayList）**

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

    import java.util.ArrayList;
    
    public class Solution {
        public boolean IsPopOrder(int[] pushA, int[] popA) {
            if (pushA.length == 0 || popA.length == 0)
                return false;
            ArrayList<Integer> stack = new ArrayList<>();
            for (int i=0, j=0; i<pushA.length; i++) {
                stack.add(pushA[i]);
                while (j < popA.length && stack.get(stack.size() - 1) == popA[j]) {
                    stack.remove(stack.size()-1);
                    j++;
                }
            }
            return stack.size() == 0 ? true : false;
        }
    }
# **22.从上往下打印二叉树**

从上往下打印出二叉树的每个节点，同层节点从左至右打印。

    import java.util.ArrayList;
    import java.util.stream.Collectors;
    /**
    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;
    
        public TreeNode(int val) {
            this.val = val;
    
        }
    
    }
    */
    public class Solution {
        public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
            ArrayList<Integer> res = new ArrayList<>();
            if (root == null) 
                return res;
            ArrayList<TreeNode> nodes = new ArrayList<>();
            nodes.add(root);
            while (nodes.size() > 0) {
                ArrayList<TreeNode> temp = new ArrayList<>();
                for (TreeNode node : nodes) {
                    res.add(node.val);
                    if (node.left != null) temp.add(node.left);
                    if (node.right != null) temp.add(node.right);
                }
                //res.addAll(nodes.stream().map(x -> x.val).collect(Collectors.toList()));
                nodes = temp;
            }
            return res;
        }
    }
# **23.二叉搜索树的后序遍历序列**

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

    //非递归 
    //非递归也是一个基于递归的思想：
    //左子树一定比右子树小，因此去掉根后，数字分为left，right两部分，right部分的
    //最后一个数字是右子树的根他也比左子树所有值大，因此我们可以每次只看有子树是否符合条件
    //即可，即使到达了左子树左子树也可以看出由左右子树组成的树还想右子树那样处理
     
    //对于左子树回到了原问题，对于右子树，左子树的所有值都比右子树的根小可以暂时把他看出右子树的左子树
    //只需看看右子树的右子树是否符合要求即可
    public class Solution {
        public boolean VerifySquenceOfBST(int [] sequence) {
            int size = sequence.length;
            if(0==size)return false;
            int i = 0;
            while(--size>=0)
            {
                System.out.println(size);
                while(sequence[i++]<sequence[size]);
                while(i<size && sequence[i++]>sequence[size]);
    
                if(i<size)return false;
                i=0;
            }
            return true;
        }
    }

这个是自己写的，真的是废废的

    import java.util.Arrays;
    
    public class Solution {
        public boolean VerifySquenceOfBST(int [] sequence) {
            if (sequence.length == 0) {
                return false;
            } else if (sequence.length <= 2) {
                return true;
            }
            int root = sequence[sequence.length-1];
            int i = 0;
            for (; i<sequence.length; i++) {
                if (sequence[i]>root) break;
            }
            if (i==sequence.length) return true;
            // left:0~i, right: i~size-1, root: last
            for (int j=i; j<sequence.length; j++) {
                if (sequence[j] < root) return false;
            }
            boolean left = true, right = true;
            if (i > 0) {
                left = VerifySquenceOfBST(Arrays.copyOfRange(sequence, 0, i));
            }
            if (i < sequence.length-1) {
                right = VerifySquenceOfBST(Arrays.copyOfRange(sequence, i, sequence.length-1));
            }
            return left && right;
        }
    }
# **24.二叉树和为某一值的路径**

输入一颗二叉树和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。

    // 递归法
    import java.util.ArrayList;
    import java.util.Collections;
    /**
    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;
    
        public TreeNode(int val) {
            this.val = val;
    
        }
    
    }
    */
    public class Solution {
        public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {
            ArrayList<ArrayList<Integer>> res = new ArrayList<>();
            if (root == null) return res;
            if (root.left == null && root.right == null && root.val == target) {
                res.add(new ArrayList<Integer>(){{add(root.val);}});
                return res;
            }
            ArrayList<ArrayList<Integer>> left = FindPath(root.left, target-root.val);
            ArrayList<ArrayList<Integer>> right = FindPath(root.right, target-root.val);
            left.addAll(right);
            for (ArrayList<Integer> item : left) {
                item.add(0, root.val);
                res.add(item);
            }
            return res;
        }
    }


    // 其实这个也是递归
    import java.util.ArrayList;
    import java.util.Collections;
    /**
    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;
    
        public TreeNode(int val) {
            this.val = val;
    
        }
    
    }
    */
    public class Solution {
        private ArrayList<ArrayList<Integer>> res = new ArrayList<>();
    
        private ArrayList<Integer> temp = new ArrayList<>();
    
        public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {
            if (root == null) return res;
            temp.add(root.val); //进入节点时，将节点的值添加到路径里
            if (root.left == null && root.right == null && root.val == target) {
                res.add(new ArrayList<Integer>(temp));
            }
            FindPath(root.left, target-root.val);
            FindPath(root.right, target-root.val);
            temp.remove(temp.size()-1); //如果左右子树都没有找到可行的路径，层层退回并删除节点值
            return res;
        }
    }
# **25.复杂链表的复制**

输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

    /*
    public class RandomListNode {
        int label;
        RandomListNode next = null;
        RandomListNode random = null;
    
        RandomListNode(int label) {
            this.label = label;
        }
    }
    */
    public class Solution {
        public RandomListNode Clone(RandomListNode pHead)
        {
            if (pHead == null) return null;
            RandomListNode head = pHead;
            RandomListNode temp;
            // 复制
            while (head != null) {
                temp = new RandomListNode(head.label);
                temp.next = head.next;
                head.next = temp;
                head = temp.next;
            }
            head = pHead;
            // 给复制的节点设置 random
            while (head != null) {
                if (head.random != null) {
                    head.next.random = head.random.next;
                }
                head = head.next.next;
            }
            RandomListNode clone = pHead.next;
            head = pHead;
            // 分离链表
            while (head != null) {
                RandomListNode cc = head.next;
                head.next = cc.next;
                head = head.next;
                if (head != null) {
                    cc.next = head.next;
                }
            }
    //        为什么！！ 为什么这样写就过不了测试啊
    //        temp = clone;
    //        while (temp.next != null) {
    //            head.next = head.next.next;
    //            temp.next = temp.next.next;
    //            head = head.next;
    //            temp = temp.next;
    //        }
            return clone;
        }
    }
# **26.二叉搜索树与双向链表**  

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
别问怎么搞，问就是递归

    /**
    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;
    
        public TreeNode(int val) {
            this.val = val;
    
        }
    
    }
    */
    public class Solution {
        private TreeNode leftLast = null;
        
        public TreeNode Convert(TreeNode root) {
            if (root == null) return null;
            if (root.left == null && root.right == null) {
                leftLast = root;
                return root;
            }
            TreeNode left = Convert(root.left);
            if (left != null) {
                leftLast.right = root;
                root.left = leftLast;
            }
            leftLast = root;
            TreeNode right = Convert(root.right);
            if (right != null) {
                right.left = root;
                root.right = right;
            }
            return left != null ? left : root; 
        }
    }
# **27.字符串的排列**

输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
http://www.cnblogs.com/cxjchen/p/3932949.html

    import java.util.*;
    
    public class Solution {
        public ArrayList<String> Permutation(String str) {
            ArrayList<String> res = new ArrayList<>();
            if (str != null && str.length() > 0) {
                PermutationSearch(str.toCharArray(), 0, res);
                Collections.sort(res);
            }
            return res;
        }
    
        private void PermutationSearch(char[] chars, int i, ArrayList<String> list) {
            if (i == chars.length-1) {
                list.add(String.valueOf(chars));
            } else {
                Set<Character> charSet = new HashSet<>();
                for (int j=i; j<chars.length; j++) {
                    if (j == i || !charSet.contains(chars[j])) {
                        charSet.add(chars[j]);
                        swap(chars, i, j);
    //                    System.out.printf("%s %s\n", String.valueOf(chars), i+1);
                        PermutationSearch(chars, i+1, list);
                        swap(chars, i, j);
                    }
                }
            }
        }
    
        private void swap(char[] cs, int i, int j) {
            char temp = cs[i];
            cs[i] = cs[j];
            cs[j] = temp;
        }
    }
# **28.数组中出现次数超过一半的数字**

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0

    public class Solution {
        public int MoreThanHalfNum_Solution(int [] array) {
            if (array == null || array.length==0) return 0;
            int res = 0;
            int count = 0;
            for (int i=0; i<array.length; i++) {
                if (count == 0) {
                    res = array[i];
                    count = 1;
                } else {
                    if (array[i] == res) {
                        count++;
                    } else {
                        count--;
                    }
                }
            }
            
            count = 0;
            for (int item : array) {
                if (item == res) count++;
            }
            return count * 2 > array.length ? res : 0;
        }
    }
# **29.最小的k个数**

输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。

    import java.util.*;
    import java.util.stream.Collectors;
    
    public class Solution {
        public ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {
            ArrayList<Integer> res = new ArrayList<>();
            if (k > input.length || k == 0) return res;
            PriorityQueue<Integer> maxHeap = new PriorityQueue<>(k, (x, y) -> -x.compareTo(y));
            for (int item : input) {
                if (maxHeap.size() != k) {
                    maxHeap.offer(item);
                } else if (maxHeap.peek() > item) {
                    maxHeap.poll();
                    maxHeap.offer(item);
                }
            }
            res.addAll(maxHeap.stream().sorted().collect(Collectors.toList()));
            return res;
        }
    }
# **30.连续子数组的最大和**

HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。你会不会被他忽悠住？(子向量的长度至少是1)

    public class Solution {
        public int FindGreatestSumOfSubArray(int[] array) {
            if (array.length == 0) return 0;
            int res = array[0], sum = 0;
            for (int item : array) {
                if (sum <= 0) {
                    sum = item;
                } else {
                    sum += item;
                }
                res = sum > res ? sum : res;
            }
            return res;
        }
    }
# **31.整数中1出现的次数（**可以，这很数学**）**

求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

    public class Solution {
        public int NumberOf1Between1AndN_Solution(int n) {
            int count = 0;
            for (int i=1; i<=n; i*=10) {
                int a = n/i, b = n%i;
                count += (a+8)/10*i+ (a%10==1?1:0)*(b+1);
            }
            return count;
        }
    }
# **32.把数组排成最小数**

输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

    import java.util.*;
    import java.util.stream.IntStream;
    import java.util.stream.Collectors;
    
    public class Solution {
        public String PrintMinNumber(int [] numbers) {
            if (numbers == null || numbers.length == 0) return "";
            List<String> str = IntStream.of(numbers).boxed()
                    .map(x -> String.valueOf(x)).collect(Collectors.toList());
            str.sort((x, y) -> (x+y).compareTo(y+x));
            return str.stream().reduce("", (x, y) -> x+y);
        }
    }
# **33.丑数(这很数学**

把只包含因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

    public class Solution {
        public int GetUglyNumber_Solution(int index) {
            if (index < 7) return index;
            int[] res = new int[index];
            res[0] = 1;
            int t2 = 0, t3 = 0, t5 = 0;
            for (int i=1; i<index; i++) {
                res[i] = Math.min(res[t2]*2, Math.min(res[t3]*3, res[t5]*5));
                if (res[i]==res[t2]*2) t2++;
                if (res[i]==res[t3]*3) t3++;
                if (res[i]==res[t5]*5) t5++;
            }
            return res[index-1];
        }
    }
# **34.第一个只出现一次的字符**

在一个字符串(1<=字符串长度<=10000，全部由大写字母组成)中找到第一个只出现一次的字符,并返回它的位置

    public class Solution {
        public int FirstNotRepeatingChar(String str) {
            char[] chars = str.toCharArray();
            int[] temp = new int['~'];
            for (char c : chars) {
                temp[(int)c]++;
            }
            for (int i=0; i < chars.length; i++) {
                if (temp[(int)chars[i]] == 1)
                    return i;
            }
            return -1;
        }
    }
# **35.数组中的逆序对(归并排序)**

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007
输入例子: 1,2,3,4,5,6,7,0
输出例子: 7

    // 挠头.jpg
# **36.两个链表的第一个公共节点**
    /*
    public class ListNode {
        int val;
        ListNode next = null;
    
        ListNode(int val) {
            this.val = val;
        }
    }*/
    public class Solution {
        public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
            int len1 = GetLinkedListLenght(pHead1);
            int len2 = GetLinkedListLenght(pHead2);
            if (len1 > len2) {
                pHead1 = GetListNode(pHead1, len1-len2);
            } else {
                pHead2 = GetListNode(pHead2, len2-len1);
            }
            while (pHead1 != null) {
                if (pHead1.val == pHead2.val) break;
                pHead1 = pHead1.next;
                pHead2 = pHead2.next;
            }
            return pHead1;
        }
        
        public int GetLinkedListLenght(ListNode head) {
            int len = 0;
            while (head != null) {
                head = head.next;
                len++;
            }
            return len;
        }
        
        public ListNode GetListNode(ListNode head, int num) {
            for (int i=0; i<num; i++) {
                head = head.next;
            }
            return head;
        }
    }
# **37.数字在排序数组中出现的次数**
    public class Solution {
        public int GetNumberOfK(int [] array , int k) {
           int count = 0;
           for (int item : array) {
               if (item == k) count++;
               else if (item > k) break;
           }
           return count;
        }
    }
# **38.二叉树深度**
    public class Solution {
        public int TreeDepth(TreeNode root) {
            if (root == null) return 0;
            return Math.max(TreeDepth(root.left), TreeDepth(root.right))+1;
        }
    }
# **39.平衡二叉树**

输入一棵二叉树，判断该二叉树是否是平衡二叉树。

    public class Solution {
        public boolean IsBalanced_Solution(TreeNode root) {
            if (root == null) return true;
            if (Math.abs(TreeDepth(root.left) - TreeDepth(root.right)) <= 1) {
                return IsBalanced_Solution(root.left) && IsBalanced_Solution(root.right);
            } else {
                return false;
            }
        }
        
        public int TreeDepth(TreeNode root) {
            if (root == null) return 0;
            return Math.max(TreeDepth(root.left), TreeDepth(root.right)) + 1;
        }
    }
# **40.数组中只出现一次的数字**

一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。

    import java.util.*;
    
    //num1,num2分别为长度为1的数组。传出参数
    //将num1[0],num2[0]设置为返回结果
    public class Solution {
        public void FindNumsAppearOnce(int [] array,int num1[] , int num2[]) {
            if (array.length < 2) return;
            int myxor = 0;
            int flag = 1;
            for (int i = 0; i < array.length; ++i)
                myxor ^= array[i];
            while ((myxor & flag) == 0) flag <<= 1;
            for (int i = 0; i < array.length; ++i) {
                if ((flag & array[i]) == 0) num2[0] ^= array[i];
                else num1[0] ^= array[i];
    // 我好菜啊
    //        HashMap<Integer,Integer> map = new HashMap<>();
    //        for (Integer item : array) {
    //            map.merge(item, 1, Integer::sum);
    //        }
    //        int[] temp = map.entrySet().stream().filter(x -> x.getValue() == 1).mapToInt(x -> x.getKey()).toArray();
    //        num1[0] = temp[0];
    //        num2[0] = temp[1];
        }
    }
# **41.和为S的连续整数序列**

小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!

    import java.util.ArrayList;
    public class Solution {
        public ArrayList<ArrayList<Integer> > FindContinuousSequence(int sum) {
            ArrayList<ArrayList<Integer>> res = new ArrayList<>();
            int low = 1, high = 2;
            while (high > low) {
                int cur = (high + low) * (high - low + 1) / 2;
                if (cur == sum) {
                    ArrayList<Integer> list = new ArrayList<>();
                    for (int i=low; i<=high; i++) {
                        list.add(i);
                    }
                    res.add(list);
                    low++;
                } else if (cur < sum) {
                    high++; // 和小右移
                } else {
                    low++; // 和大左移
                }
            }
            return res;
        }
    }
# **42.和为S的两个数字**

输入一个递增排序的数组和一个数字S，在数组中查找两个数，是的他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

    import java.util.ArrayList;
    public class Solution {
        public ArrayList<Integer> FindNumbersWithSum(int [] array,int sum) {
            ArrayList<Integer> res = new ArrayList<>();
            int left = 0, right = array.length - 1;
            while (left < right) {
                int cur = array[left] + array[right];
                if (cur == sum) {
                    res.add(array[left]);
                    res.add(array[right]);
                    break;
                } else if (cur > sum) {
                    right--;
                } else {
                    left++;
                }
            }
            return res;
        }
    }
# **43.左旋转字符串**

汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！

    public class Solution {
        public String LeftRotateString(String str,int n) {
            if (str.length() == 0 || n == 0) return str;
            StringBuilder sb = new StringBuilder();
            int move = n % str.length();
            for (int i=move; i<str.length(); i++) sb.append(str.charAt(i));
            for (int j=0; j<move; j++) sb.append(str.charAt(j));
            return sb.toString();
        }
    }
# **44.翻转单词顺序**

牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？

    public class Solution {
        public String ReverseSentence(String str) {
            if (str.trim().equals("")) return str;
            String[] list = str.split(" ");
            for (int left=0, right=list.length-1; left < right; left++, right--) {
                String temp = list[left];
                list[left] = list[right];
                list[right] = temp;
            }
            return String.join(" ", list);
        }
    }
# **45.扑克牌顺子**

LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何。为了方便起见,你可以认为大小王是0。

    public class Solution {
        public boolean isContinuous(int [] numbers) {
            if (numbers.length != 5) return false;
            int[] flag = new int[14];
            int min = 14, max = -1;
            for (int i=0; i<numbers.length; i++) {
                flag[numbers[i]]++;
                if (numbers[i] == 0) continue;
                if (flag[numbers[i]] > 1) return false;
                if (numbers[i] > max) max = numbers[i];
                if (numbers[i] < min) min = numbers[i];
            }
            return max - min > 5 ? true : false;
        }
    }
# **46.圆圈中最后剩下的数**

每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)

    import java.util.LinkedList;
    
    public class Solution {
        public int LastRemaining_Solution(int n, int m) {
            if (n == 0) return -1;
            LinkedList<Integer> list = new LinkedList<>();
            for (int i=0; i<n; i++) list.add(i);
            int index = 0;
            while (list.size() > 1) {
                index = (index + m - 1) % list.size();
                list.remove(index);
            }
            return list.get(0);
        }
    }
# **47.求1+2+3+...+n**

求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）
利用逻辑与的短路特性实现递归终止

    public class Solution {
        public int Sum_Solution(int n) {
            int sum = n;
            boolean flag = (sum > 0) && (sum+=Sum_Solution(--n)) > 0;
            return sum;
        }
    }
# **48.不用加减乘除做加法(数学题)**
    public class Solution {
        public int Add(int num1,int num2) {
            while (num2 != 0) {
                int temp = num1 ^ num2;
                num2 = (num1 & num2) << 1;
                num1 = temp;
            }
            return num1;
        }
    }
# **49.把字符串转换成整数**

将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0

    public class Solution {
        public int StrToInt(String str) {
            int res = 0;
            for (int i=str.length()-1, base=1; i>=0; i--, base*=10) {
                int n = str.charAt(i);
                if (n>='0' && n<='9')
                    res += (n - '0') * base;
                else if (n == '+' && i==0) break;
                else if (n == '-' && i==0) res = -res;
                else {
                    res = 0;
                    break;
                }
            }
            return res;
        }
    }
# **50.数组中重复的数字**

在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是重复的数字2或者3。

    import java.util.*;
    
    public class Solution {
        // Parameters:
        //    numbers:     an array of integers
        //    length:      the length of array numbers
        //    duplication: (Output) the duplicated number in the array number,length of duplication array is 1,so using duplication[0] = ? in implementation;
        //                  Here duplication like pointor in C/C++, duplication[0] equal *duplication in C/C++
        //    这里要特别注意~返回任意重复的一个，赋值duplication[0]
        // Return value:       true if the input is valid, and there are some duplications in the array number
        //                     otherwise false
        public boolean duplicate(int numbers[],int length,int [] duplication) {
            HashSet<Integer> v = new HashSet<>();
            for (int i=0; i<length; i++) {
                if (v.contains(numbers[i])) {
                    duplication[0] = numbers[i];
                    return true;
                } else {
                    v.add(numbers[i]);
                }
            }
            duplication[0] = -1;
            return false;
        }
    }
# **51.构建乘积数组(reduce)**

给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。
B[] 展开成方格, 放个左上角到右下角斜线换成 1

    import java.util.ArrayList;
    public class Solution {
        public int[] multiply(int[] A) {
            int[] res = new int[A.length];
            if (A.length==0) return res;
            res[0] = 1;
            for (int i=1; i<A.length; i++) {
                res[i] = res[i-1] * A[i-1];
            }
            int temp = 1;
            for (int j=A.length-2; j>=0; j--) {
                res[j] *= (temp *= A[j+1]);
            }
            return res;
        }
    }
# **52.正则表达式匹配 ^&**

请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配

    public class Solution {
        public boolean match(char[] str, char[] pattern) {
            return check(str, 0, pattern, 0);
        }
    
        public boolean check(char[] str, int i, char[] pattern, int j) {
            if (i == str.length && j == pattern.length) return true;
            if (i != str.length && j == pattern.length) return false;
            if (j+1 < pattern.length && pattern[j+1] == '*') {
                if (i != str.length && (str[i] == pattern[j] || pattern[j] == '.')) {
                    return check(str, i, pattern, j+2) // * 匹配0次
                            || check(str, i+1, pattern, j+2) // * 匹配1次
                            || check(str, i+1, pattern, j); // * 匹配1个,然后继续递归
                } else {
                    return check(str, i, pattern, j+2); // * 匹配0次
                }
            }
            // 逐个匹配判断
            if (i != str.length && (str[i] == pattern[j] || pattern[j] == '.')) {
                return check(str, i+1, pattern, j+1);
            } else {
                return false;
            }
        }
    }
# **53.表示数值的字符串**

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
别花里胡哨，直接匹配完事了

    public class Solution {
        public boolean isNumeric(char[] str) {
            String s = String.valueOf(str);
            return s.matches("\[\\+\\-]?\\d*(\\.\\d+)?([eE\][\\+\\-]?\\d+)?");
        }
    }
# **54.字符流中第一个不重复的字符**

请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。

    import java.util.*;
    
    class Solution {
        ArrayList<Character> chars = new ArrayList<>();
        ArrayList<Character> first = new ArrayList<>();
        //Insert one char from stringstream
        public void Insert(char ch)
        {
            if (chars.contains(ch)) {
                first.remove(Character.valueOf(ch));
            } else {
                first.add(ch);
            }
            chars.add(ch);
        }
        //return the first appearence once char in current stringstream
        public char FirstAppearingOnce()
        {
            return first.size() > 0 ? first.get(0) : '#';
        }
    }
# **55.链表中环的入口节点**
    /*
     public class ListNode {
        int val;
        ListNode next = null;
    
        ListNode(int val) {
            this.val = val;
        }
    }
    */
    public class Solution {
    
        public ListNode EntryNodeOfLoop(ListNode pHead)
        {
            ListNode slow = pHead, fast = pHead;
            while (fast.next != null) {
                slow = slow.next;
                fast = fast.next.next;
                if (slow == fast) {
                    fast = pHead;
                    while (slow != fast) {
                        slow = slow.next;
                        fast = fast.next;
                    }
                    return slow;
                }
            }
            return null;
        }
    }
# **56.删除链表中重复的节点(去重,删除)**

在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

    /*
     public class ListNode {
        int val;
        ListNode next = null;
    
        ListNode(int val) {
            this.val = val;
        }
    }
    */
    public class Solution {
        public ListNode deleteDuplication(ListNode pHead)
        {
            ListNode head = pHead, pre = new ListNode(0);
            pre.next = head;
            ListNode save = pre;
            while (head != null && head.next != null) {
                if (head.val == head.next.val) {
                    ListNode temp = head.next;
                    while (temp != null && head.val == temp.val) {
                        temp = temp.next;
                    }
                    head = temp;
                    pre.next = head;
                } else {
                    head = head.next;
                    pre = pre.next;
                }
            }
            return save.next;
        }
    }
# **57.二叉树的下一个节点**

给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

    /*
    public class TreeLinkNode {
        int val;
        TreeLinkNode left = null;
        TreeLinkNode right = null;
        TreeLinkNode next = null;
    
        TreeLinkNode(int val) {
            this.val = val;
        }
    }
    */
    public class Solution {
        public TreeLinkNode GetNext(TreeLinkNode pNode)
        {
            // 有右子树, 下一个节点是右子树的左叶节点
            if (pNode.right != null) {
                TreeLinkNode p = pNode.right;
                while (p.left != null) {
                    p = p.left;
                }
                return p;
            }
            // 无右子树, 找到第一个当前节点是父节点左叶节点的节点
            while (pNode.next != null) {
                if (pNode.next.left == pNode) return pNode.next;
                pNode = pNode.next;
            }
            return null;
        }
    }
# **58.对称二叉树**

请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

    /*
    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;
    
        public TreeNode(int val) {
            this.val = val;
    
        }
    
    }
    */
    public class Solution {
        boolean isSymmetrical(TreeNode pRoot)
        {
            if (pRoot == null) return true;
            return isEqual(pRoot.left, pRoot.right);
        }
        
        boolean isEqual(TreeNode left, TreeNode right) {
            if (left == null && right == null) return true;
            if (left == null || right == null) return false;
            if (left.val == right.val) {
                return isEqual(left.left, right.right) && isEqual(left.right, right.left);
            }
            return false;
        }
    }
# **59.按之自形顺序打印二叉树**

请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

    import java.util.ArrayList;
    import java.util.Collections;
    /*
    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;
    
        public TreeNode(int val) {
            this.val = val;
    
        }
    
    }
    */
    public class Solution {
        public ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
            ArrayList<ArrayList<Integer>> res = new ArrayList<>();
            if (pRoot == null) return res;
            boolean even = false;
            ArrayList<TreeNode> row = new ArrayList<TreeNode>(){{add(pRoot);}};
            while (row.size() > 0) {
                ArrayList<Integer> rowVal = new ArrayList<>();
                ArrayList<TreeNode> temp = new ArrayList<>();
                for (TreeNode item : row) {
                    rowVal.add(item.val);
                    if (item.left != null) temp.add(item.left);
                    if (item.right != null) temp.add(item.right);
                }
                if (even) Collections.reverse(rowVal);
                even = !even;
                res.add(rowVal);
                row = temp;
            }
            return res;
        }
    }
# **60.把二叉树打印成多行**

从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。

    import java.util.ArrayList;


​    
​    /*
​    public class TreeNode {
​        int val = 0;
​        TreeNode left = null;
​        TreeNode right = null;
​    
        public TreeNode(int val) {
            this.val = val;
    
        }
    
    }
    */
    public class Solution {
        public ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
            ArrayList<ArrayList<Integer>> res = new ArrayList<>();
            if (pRoot == null) return res;
            ArrayList<TreeNode> row = new ArrayList<TreeNode>(){{add(pRoot);}};
            while (row.size() > 0) {
                ArrayList<Integer> rowVal = new ArrayList<>();
                ArrayList<TreeNode> temp = new ArrayList<>();
                for (TreeNode item : row) {
                    rowVal.add(item.val);
                    if (item.left != null) temp.add(item.left);
                    if (item.right != null) temp.add(item.right);
                }
                res.add(rowVal);
                row = temp;
            }
            return res;
        }
    }
# **61.序列化二叉树**

好难啊, 什么序列化呀

    /*
    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;
    
        public TreeNode(int val) {
            this.val = val;
    
        }
    
    }
    */
    public class Solution {
        String Serialize(TreeNode root) {
            if (root == null) return "#,";
            return String.format("%s,%s%s", root.val, Serialize(root.left), Serialize(root.right));
        }
    
        TreeNode Deserialize(String str) {
            String[] text = str.split(",");
            return innerDeserialize(text);
        }
    
        private int index = -1;
    
        TreeNode innerDeserialize(String[] text) {
            index++;
            if (index >= text.length || text[index].equals("#")) {
               return null;
            }
            TreeNode node = new TreeNode(Integer.valueOf(text[index]));
            node.left = innerDeserialize(text);
            node.right = innerDeserialize(text);
            return node;
        }
    }
# **62.二叉搜索树的第k个节点**

给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。

    /*
    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;
    
        public TreeNode(int val) {
            this.val = val;
    
        }
    
    }
    */
    public class Solution {
        private int index = 0;
        TreeNode KthNode(TreeNode pRoot, int k)
        {
            if (pRoot != null) {
                TreeNode node = KthNode(pRoot.left, k);
                if (node != null) return node;
                index++;
                if (index == k) return pRoot;
                node = KthNode(pRoot.right, k);
                if (node != null) return node;
            }
            return null;
        }
    }
# **63.数据流中的中位数**

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。

    import java.util.PriorityQueue;
    
    public class Solution {
    
        private int count = 0;
        private PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        private PriorityQueue<Integer> maxHeap = new PriorityQueue<>((x,y) -> -x.compareTo(y));
    
        public void Insert(Integer num) {
            // 从小到大排序, 最大堆记录左半部分, 最小堆记录右半部分
            if (count % 2 == 0) {
                // 添加前有偶数个元素时, num -> 最大堆, 最大堆.最大元素 -> 最小堆
                maxHeap.offer(num);
                minHeap.offer(maxHeap.poll());
            } else {
                minHeap.offer(num);
                maxHeap.offer(minHeap.poll());
            }
            count++;
        }
    
        public Double GetMedian() {
            if (count % 2 == 0) {
                return Double.valueOf((minHeap.peek() + maxHeap.peek())) / 2;
            } else {
                return Double.valueOf(minHeap.peek());
            }
        }
    
    }
# **64.滑动窗口最大值**

给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。

    import java.util.ArrayList;
    
    public class Solution {
        public ArrayList<Integer> maxInWindows(int [] num, int size)
        {
            ArrayList<Integer> res = new ArrayList<>();
            if (num.length == 0 || size == 0) return res;
            for (int i=0; i<=num.length-size; i++) {
                int temp =num[i];
                for (int j=i+1; j<i+size; j++) {
                    if (num[j] > temp) temp = num[j];
                }
                res.add(temp);
            }
            return res;
        }
    }
# **65.矩阵中的路径**

请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。 例如[a b c e s f c s a d e e]是3*4矩阵，其包含字符串"bcced"的路径，但是矩阵中不包含“abcb”路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。
声明一个数组用来记录是否访问，遍历每一个位置开始的情况，判断是否可行

    public class Solution {
        public boolean hasPath(char[] matrix, int rows, int cols, char[] str)
        {
            int[] visited = new int[matrix.length];
            for (int r=0; r<rows; r++) {
                for (int c=0; c<cols; c++) {
                    if (serchPath(matrix, rows, cols, r, c, str, 0, visited)) {
                        return true;
                    }
                }
            }
            return false;
        }
    
        public boolean serchPath(char[] matrix, int rows, int cols, int r, int c, char[] str, int cur, int[] visited) {
            int index = r * cols + c;
            if (r < 0 || r >= rows || c < 0 || c >= cols || matrix[index] != str[cur] 
                || visited[index] == 1) return false;
            if (cur == str.length - 1) return true;
            visited[index] = 1;
            if (serchPath(matrix, rows, cols, r - 1, c, str, cur + 1, visited)
                || serchPath(matrix, rows, cols, r + 1, c, str, cur + 1, visited)
                || serchPath(matrix, rows, cols, r, c - 1, str, cur + 1, visited)
                || serchPath(matrix, rows, cols, r, c + 1, str, cur + 1, visited))
                return true;
            // 这里把访问标记清除后，可以重复使用此数组
            visited[index] = 0;
            return false;
        }
    }
# **66.机器人的运动范围**

地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？
生命一个数组标记是否访问，然后向上下左右探索

    public class Solution {
        public int movingCount(int threshold, int rows, int cols)
        {
            int\[][] visited = new int[rows\][cols];
            return searchCount(threshold, rows, cols, 0, 0, visited);
        }
    
        public int searchCount(int threshold, int rows, int cols, int r, int c, int[][] visited) {
            if (r < 0 || r >= rows || c < 0 || c >= cols 
                    || !thresholdCheck(threshold, r, c) || visited\[r\][c] == 1) return 0;
            visited\[r\][c] = 1;
            return searchCount(threshold, rows, cols, r - 1, c, visited)
                    + searchCount(threshold, rows, cols, r + 1, c, visited)
                    + searchCount(threshold, rows, cols, r, c - 1, visited)
                    + searchCount(threshold, rows, cols, r, c + 1, visited) + 1;
        }
    
        public boolean thresholdCheck(int threshold, int r, int c) {
            int sum = 0;
            while (r > 0) {
                sum += r % 10;
                r /= 10;
            }
            while (c > 0) {
                sum += c % 10;
                c /= 10;
            }
            return sum <= threshold;
        }
    }

