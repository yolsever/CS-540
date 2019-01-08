include("misc.jl")
using LinearAlgebra, MathProgBase

function leastAbsolutes(X,y)

	# Add bias column
	n = size(X,1)
	Z = [ones(n,1) X]

	# Find regression weights minimizing squared error
	# w = (Z'*Z)\(Z'*y)
    # linprog(c, A, sense, b, l, u, solver)


	# Make linear prediction function
	predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w

	# Return model
	return LinearModel(predict,w)
end
