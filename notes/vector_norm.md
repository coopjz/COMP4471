# Norm

1. Vector Norm
2. Vector L1 Norm
3. Vector L2 Norm
4. Vector Max Norm

## Vector Norm

The length of the vector is referred to as the vector norm or the vector’s magnitude.

> The length of a vector is a nonnegative number that describes the extent of the vector in space, and is sometimes referred to as the vector’s magnitude or the norm.

Notations are used to represent the vector norm in broader calculations and the type of vector norm calculation almost always has its own unique notation.

## L1 Norm

The notation for the L1 norm of a vector is $||v||_1$ , where 1 is a subscript. As such, this length is sometimes called the taxicab norm or the **Manhattan norm**.

The L1 norm is calculated as the sum of the absolute vector values, where the absolute value of a scalar uses the notation $|a_1|$. In effect, the norm is a calculation of the Manhattan distance from the origin of the vector space.

$$
||v||_1 = |a_1| + |a_2| + |a_3|
$$

```python
 
# l1 norm of a vector
from numpy import array
from numpy.linalg import norm
a = array([1, 2, 3])
print(a)
l1 = norm(a, 1)
print(l1)
# output 6
```

## L2 Norm

The length of a vector can be calculated using the L2 norm, where the 2 is a superscript of the L, e.g. L^2.

The notation for the L2 norm of a vector is$||v||_2$ where 2 is a subscript.

The L2 norm calculates the distance of the vector coordinate from the origin of the vector space. As such, it is also known as the Euclidean norm as it is calculated as the Euclidean distance from the origin. The result is a positive distance value.

The L2 norm is calculated as the square root of the sum of the squared vector values.

$$
||v||_2 = \sqrt{(a_1^2 + a_2^2 + a_3^2)}
$$

```python
# l2 norm of a vector

from numpy import array
from numpy.linalg import norm
a = array([1, 2, 3])
print(a)
l2 = norm(a)
print(l2)  #3.74
```

Like the L1 norm, the L2 norm is often used when fitting machine learning algorithms as a regularization method, e.g. a method to keep the coefficients of the model small and, in turn, the model less complex.

By far, the L2 norm is more commonly used than other vector norms in machine learning.

## Max Norm

The length of a vector can be calculated using the maximum norm, also called max norm.

Max norm of a vector is referred to as L^inf where inf is a superscript and can be represented with the infinity symbol. The notation for max norm is ||x||inf, where inf is a subscript.

$$
||v||_{inf} = max(|a_1|, |a_2|, |a_3|)
$$

```python

# max norm of a vector
from numpy import inf
from numpy import array
from numpy.linalg import norm
a = array([1, 2, 3])
print(a)
maxnorm = norm(a, inf)
print(maxnorm)  # 3
```

Max norm is also used as a regularization in machine learning, such as on neural network weights, called max norm regularization.
