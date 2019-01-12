include("misc.jl")
using LinearAlgebra, MathProgBase, GLPKMathProgInterface, Base

function leastAbsolutes(X,y)

	# Add bias column
	n = size(X,1)
	Z = [ones(n,1) X]

	# Find regression weights minimizing squared error
	# w = (Z'*Z)\(Z'*y)
	r = ones(1, size(Z)[1])
	w = zeros(1, size(Z)[2]) # size (502,)
    #
	# show(size([r w]))
	solution = linprog(vec([r w]),[I Z; I -Z],vec([y; -y]),vec(fill(Inf,(1,length([y; y])))),vec([zeros((1,length(r))) fill(-Inf,(1,length(w)))]),vec(fill(Inf, (1, length(y) + length(w)))),GLPKSolverLP())

	# solution = linprog([r w],0,-inf,inf,[lb fill(-inf,size[Z][2])],[fill(inf,size(Z)[1]+size[Z][2]],GLKPSolverLP()) #(r,w)
    x = solution.sol
	w = x[501:502]

	predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w

	# Return model
	return LinearModel(predict,w)
end
