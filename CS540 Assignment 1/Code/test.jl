A = [1 2 3 4;3 4 5 6; 4 5 6 7; 3 3 3 3]

using Printf
(n,d) = size(A)

show(A[1:Int(n/2),:])
