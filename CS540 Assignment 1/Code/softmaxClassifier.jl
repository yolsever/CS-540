using LinearAlgebra, Base

function softmaxClassifier(X,y)
		global (n, d) = size(X)
        global n_classes = size(unique(y))[1]
        # Initial guess
        global w = zeros(n_classes,d)
		global (wn,wd) = size(w)
        w = reshape(w, (wn*wd,1))
		include("findMin.jl")
		funObj(w) = softmaxObj(w,X,y)
        w = findMin(funObj, w, derivativeCheck=true)
		return LinearModel(predict,w)
end

function predict(X)
	# self.w= self.w.reshape((self.n_classes, X.shape[1]))
	# return np.argmax(X@self.w.T, axis=1)
    global w= reshape(w, (n_classes, d))
	# show((n,d,wn,wd))
	dprod = X*w'
	(maxval, index) = findmax(dprod, dims=2)
	global ind = zeros(size(X)[1],1)
	for i in 1:size(index)[1]
	    global ind[i] = index[i][2]
	end
	return convert(Array{Int64,2}, ind) ## one of these was @ instead of dot
end

function softmaxObj(w, X, y)
    # Calculate the function value
    w= reshape(w, (n_classes, d))

    global denom = zeros((n,1))

    for i in 1:n
        for c_bar in 1:n_classes
            global denom[i] += exp(dot(w[c_bar,:],X[i,:]))
		end
	end

    global f=0
    for i in 1:n
        global f += dot(-w[y[i],:], X[i,:]) + log(denom[i])
	end

        # Calculate the gradient value

    global g = zeros((n_classes,d))

    for c in 1:n_classes
        for j in 1:d
            for i in 1:n
                global g[c,j] += X[i,j] * (exp(dot(w[c,:],X[i,:])) / denom[i] - (y[i] == c))
			end
		end
	end

    g = reshape(g, (n_classes*d,1))
    return f, g
end
