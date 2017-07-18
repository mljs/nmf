#!/usr/bin/env python

from numpy import *
from nmf import *

w1 = array([[1,2,3],[4,5,6]])
h1 = array([[1,2],[3,4],[5,6]])
w2 = array([[1,1,3],[4,5,6]])
h2 = array([[1,1],[3,4],[5,6]])

v = dot(w1,h1)

(wo,ho) = nmf(v, w2, h2, 0.001, 10, 10)
print wo
print ho
