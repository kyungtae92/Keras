import numpy as np

aaa = np.array([1,2,3,4,5])
print(aaa.shape) # (5,) 유사함 (1, 5) 벡터

aaa = aaa.reshape(1, 5)
print(aaa.shape)  # (1, 5)   행렬

bbb = np.array([[1,2,3], [4,5,6]])
print(bbb.shape)  # (2, 3)

ccc = np.array([[1,2],[3,4],[5,6]])
print(ccc.shape)  # (3, 2)

ddd = ccc.reshape(3,1,2,1)
print(ddd.shape)

print(aaa)
print(bbb)
print(ccc)
print(ddd)


A = np.array([[1, 2, 3], [4, 5, 6]])
print(A)
B = A.reshape((3, 2))
print(B)

'''
스칼라 : 하나의 숫자
벡터 : 스칼라(숫자)의 배열
행렬 : 2차원 배열
텐서 : 2차원 이상 배열
scalar   vector   matrix         tensor
1        [1,2]    [[1,2],[2,3]]  [[[1,2],[2,3]],[[3,2],[4,6]]]


행렬 계산법
[ [1,2,3], [4,5,6] ] = 2,3
[ [[1],[2],[3]], [[4],[5],[6]] ] = 2,3,1
[ [1,2], [3,4], [5,6] ] = 3,2
[ [[1],[2]], [[3],[4]], [[5],[6]] ] = 3,2,1
[ [[[1],[2]]], [[[3],[4]]], [[[5],[6]]] ] = 3,1,2,1
'''