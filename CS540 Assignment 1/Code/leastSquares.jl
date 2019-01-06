include("misc.jl")
using LinearAlgebra

function leastSquares(X,y)

	# Add bias column
	n = size(X,1)
	Z = [ones(n,1) X]

	# Find regression weights minimizing squared error
	w = (Z'*Z)\(Z'*y)

	# Make linear prediction function
	predict(Xtilde) = [ones(size(Xtilde,1),1) Xtilde]*w

	# Return model
	return LinearModel(predict,w)
end

function leastSquaresRBFL2(X, y, lambda, sigma)

	# Add bias column
	n = size(X,1)

	X_rbf = rbfBasis(X, X, sigma)

	Z = [ones(n,1) X_rbf]

	# Find regression weights minimizing squared error
	w = (Z'*Z + lambda * I)\(Z'*y)

	# Make linear prediction function
	predict(Xtilde) = [ones(size(Xtilde,1),1) rbfBasis(Xtilde, X,sigma)]*w

	# Return model
	return LinearModel(predict,w)
end

function rbfBasis(X, y, sigma)
(n,d) = size(X)
(m,e) = size(y)

Z = zeros(n,m)

for i in 1:n
	for j in 1:m
		Z[i,j] = exp(-norm(X[i,:] - y[j,:])^2/(2*sigma^2))
	end
end

return Z
end
