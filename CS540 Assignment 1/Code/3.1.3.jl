# Load X and y variable
using JLD
using Random
data = load("nonLinear.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

(n,d) = size(X)

a = shuffle(1:n)

X = X[a, :]
y = y[a, :]

(X_training, X_valid,y_training, y_valid) = (X[1:Int(n/2),:],X[Int(n/2)+1:end,:],y[1:Int(n/2),:],y[Int(n/2)+1:end,:])
(n,d) = size(X_training)
(m,e) = size(X_valid)

# Compute number of training examples and number of features

# Fit least squares model
include("leastSquares.jl")

bestError = 9999999999999
bestsigma = 1
bestlambda =1
for lambda in [1/8 1/4 1/2 1 2 4 8]
    for sigma in [1/8 1/4 1/2 1 2 4 8]
        model = leastSquaresRBFL2(X_training,y_training,lambda,sigma)
        t = size(X_valid,1)
        yhat_valid = model.predict(X_valid)
        testError = sum((yhat_valid - y_valid).^2)/t
        if testError < bestError
            global bestError, bestlambda, bestsigma
            bestError = testError
            bestlambda = lambda
            bestsigma = sigma
        end
    end
end
@show(bestlambda)
@show(bestsigma)

model = leastSquaresRBFL2(X, y, bestlambda,bestsigma)

# Report the error on the test set
using Printf
t = size(Xtest,1)
yhat = model.predict(Xtest)
testError = sum((yhat - ytest).^2)/t
@printf("TestError = %.2f\n",testError)

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
plot(Xtest,ytest,"g.")
Xhat = minimum(X):.1:maximum(X)
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
yhat = model.predict(Xhat)
plot(Xhat[:],yhat,"r")
ylim((-300,400))
