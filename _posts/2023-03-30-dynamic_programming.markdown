---
layout: post
title:  "Dynamic Programming"
date:   2023-03-29 21:59:00 +0000
categories: Software Development
---
Dynamic programming is a powerful approach in computer programming for solving complex problems by breaking them down into simpler components. This post provides a brief summary of the theory underlying this method as well as a short example highlighting its application.

### Overview

**Dynamic programming** is a technique for solving problems by recursively decomposing them into more manageable sub-problems and combining their solutions. There are two main requirements for a problem to be solvable via dynamic programming: 
- **Optimal Sub-structure**  - the solution must be obtainable from its sub-problem solutions.
- **Overlapping Sub-problems** - sub-problems must be reusable in solving other sub-problems. 

These qualities can be more clearly illustrated by discussing dynamic programming in the context of the Fibonacci sequence, in which each value is the summation of the previous two values. In this setting, the problem of calculating the $$n^{\text{th}}$$ Fibonacci value $$F(n)$$, can be obtained by solving the two sub-problems $$F(n - 1)$$ and $$F(n - 2)$$. In this respect, the problem has an optimal sub-structure as the solution to $$F(n)$$ can be computed from the summation of $$F(n - 1)$$ and $$F(n - 2)$$. Similarly, the solution to $$F(n - 1)$$ can be obtained from the summation of $$F(n - 2)$$ and $$F(n - 3)$$. From this observation, it can be concluded that the problem has overlapping sub-problems, as both $$F(n)$$ and the sub-problem $$F(n - 1)$$ require the solution to the sub-problem $$F(n - 2)$$. In dynamic programming, the solution for each overlapping sub-problem is calculated only once; reducing the time-complexity of the task from exponential, when solved by the naive recursive approach, to polynomial, when solved via dynamic programming. 

### Example

Below is an illustration of how dynamic programming can be applied to a practical problem:

*There are n boards of length {A1, A2, ..., An} and k painters. Each board takes 1 unit of time per unit of board to paint and each painter can only be used to paint adjacent board segments, e.g. A1 and A2. What is the optimal way of allocating k painters to achieve the minimum total painting time?*

To identify if dynamic programming is suitable for this problem, it is first important to search for a naive recursive solution. In this instance, the simplest method would be to determine all the possible partition configurations and compute the painting time of each, selecting the minimum.


```python
def partition(arr, n, k):
    """Get the minimum partition sum"""
     
    # if only one painter
    if k == 1: 
        return sum(arr[0:n])
    
    # if only a single board
    if n == 1: 
        return arr[0]
     
    best = 1e5
    for i in range(1, n + 1):

        # calculate sum of new and all previous partitions
        right_part_sum = sum(arr[i:n])
        left_part_sum = partition(arr, i, k-1)

        # select the minimum painter combination
        best = min(best, max(left_part_sum, right_part_sum))

    return best
```


This naive solution iterates through the length of the board; comparing the painting time of new board segments to the recursively-calculated painting time of all prior segments. More explicitly, the board solution with $$k$$ painters relies on the board solution with $$k-1$$ painters and simply requires a comparison to be made between the lowest sum of the prior partitions and the newest partition. From this statement it is clear that the problem has optimal sub-structure and overlapping sub-problems, as the solutions for $$k$$ and $$k-1$$ are obtainable from the solution of $$k-2$$ and all smaller partitions. The satisfaction of these conditions implies dynamic programming is a applicable to this problem and consequently it should be possible to optimise the solution by storing and re-using overlapping sub-problems solutions.


```python
def dynamic_partition(arr, n, k):
    """Optimise minimum partition sum with dynamic programming."""

    # create table for storing values
    saved_values = [[0 for i in range(n+1)] for j in range(k+1)]

    # if only one painter (set partition values to cumulative sum)
    for i in range(1, n + 1):
        saved_values[1][i] = sum(arr[0:i])
 
    # if only a single board (set painter number to board sum)
    for j in range(1, k + 1):
        saved_values[j][1] = arr[0]

    # iteratively span greater num of boards and painters
    # select the minimum greatest sum of each partition
    # use prior partitions to calculate future partitions
    # storing values in the table
    for i in range(2, n+1): # boards
        for j in range(2, k+1): # painters
            best = 1e5
            for l in range(1, i+1):
                right_part_sum = sum(arr[l:i])
                left_part_sum = saved_values[j-1][l]
                best = min(best, max(left_part_sum, right_part_sum))            
            saved_values[j][i] = best

    return saved_values[k][n]
```

This solution differs from the previous in that it uses a bottom-up approach to compute the lowest sum of lesser partitions and stores these values for faster computation in future calculations. For more practice problems, such as the one described above, refer to [geeksforgeeks.org](https://www.geeksforgeeks.org/dynamic-programming/). 
