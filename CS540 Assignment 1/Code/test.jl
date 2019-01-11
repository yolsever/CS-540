using LinearAlgebra, Base

A = [0 2 3 4;3 4 5 3; 4 5 6 7; 3 3 3 3]
# show(A')
B = [1 2 3 4]

(val,indx) = findmax(A, dims=2)
show(fill(Inf, (3,3)))
show(length([A; A]))
show(fill(Inf, (length([B; B]),1)))
# global ind = zeros(size(A)[1],1)
# for i in 1:size(indx)[1]
#     global ind[i] = indx[i][2]
# end
# show(ind)
using Printf
(n,d) = size(A)


# show(size(unique(A)))

denom =2
global res = 5

res = 10

for i in 1:10
    global res += denom
end

# show(res)

#
# function ret(X)
#     global a = 2
#     return X
# end
#
# function kad(X)
#     X = a *2
#     return X
# end
# show(ret(2))
# show(kad(2))
