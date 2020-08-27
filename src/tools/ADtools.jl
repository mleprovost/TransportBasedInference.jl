export Diff


Diff(f) = x -> ForwardDiff.derivative(f, float(x))
Diff(f, k) = k == 0 ? f : k == 1 ? Diff(f) : k>1 ? Diff(Diff(f),k-1) : print("Differential operator not defined for k<0")
