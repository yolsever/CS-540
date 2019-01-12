# Load X and y variable
using JLD, Statistics, Printf
data = load("outliersData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])
y = reshape(y, length(y))

# Fit a least squares model
include("leastMax.jl")
model = leastMax(X,y)

# Evaluate training error
# show(size(X))
yhat = model.predict(X)
trainError = mean((yhat - y).^2)
@printf("Squared train Error with least max: %.3f\n",trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean((yhat - ytest).^2)
@printf("Squared test Error with least max: %.3f\n",testError)

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
Xhat = minimum(X):.01:maximum(X)
yhat = model.predict(Xhat)
plot(Xhat,yhat,"g")
