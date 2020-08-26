
D(f) = x -> ForwardDiff.derivative(f, float(x))
D(f, k) = k == 0 ? f : k == 1 ? D(f) : k>1 ? D(D(f),k-1) : print("Differential operator not defined for k<0")
