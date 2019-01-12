include("misc.jl")
using LinearAlgebra, MathProgBase, GLPKMathProgInterface, Base

function leastMax(X,y)

	# Add bias column
	n = size(X,1)
	Z = [ones(n,1) X]

	# Find regression weights minimizing squared error
	# w = (Z'*Z)\(Z'*y)
    ex = ones((1,1))
	r = zeros(1, size(Z)[1])
	w = zeros(1, size(Z)[2])
    exz = zeros((size(Z)[1],1))
    exo = ones((size(Z)[1],1))
    top = [exo -I zeros((size(Z)[1],size(Z)[2]))]
    # show(top[:,3])

    # show(size(top))
    # show(size([exz I Z]))
    # show(size([exz I Z; exz I -Z]))
    #
	# show(size([r w]))
    # linprog(c,A,d,b,lb,ub,GLPKSolverLP())
	solution = linprog(vec([ex r w]),[top ;exz I Z;exz I -Z],
    vec([exz; y; -y]),vec(fill(Inf,(1,length([exz;y; y])))),vec([0 zeros((1,length(r))) fill(-Inf,(1,length(w)))]),
    vec(fill(Inf, (1, length(ex)+ length(y) + length(w)))),GLPKSolverLP())

	# solution = linprog([r w],0,-inf,inf,[lb fill(-inf,size[Z][2])],[fill(inf,size(Z)[1]+size[Z][2]],GLKPSolverLP()) #(r,w)
    x = solution.sol
	w = x[502:503]

	predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w

	# Return model
	return LinearModel(predict,w)
end
