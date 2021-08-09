# @testset "Verify log_pdf, grad_x_log_pdf and hess_x_log_pdf functions" begin
#     atol = 1e-8
#     Ne = 50
#     m = 10
#     Blist = [ProHermiteBasis(8); PhyHermiteBasis(8); CstProHermiteBasis(8)]#; CstPhyHermiteBasis(8)]#; CstLinProHermiteBasis(8); CstLinPhyHermiteBasis(8)]
#     for Nx = 1:3
#         for b in Blist
#             X = randn(Nx, Ne)
#             B = MultiBasis(b, Nx)
#             Nψ = 5
#             idx = vcat(zeros(Int64, 1, Nx), rand(0:2,Nψ-1, Nx))
#             coeff = randn(Nψ)
#             C = HermiteMapComponent(m, Nx, idx, coeff; b = string(typeof(b)))
#
# 			logC = log_pdf(C, X)
#             dxlogC  = grad_x_log_pdf(C, X)
#             d2xlogC = hess_x_log_pdf(C, X)
#
#
#             function evaluatef0(x)
#                 y = copy(x)
#                 y[end] = 0.0
#                 return C.I.f(y)
#             end
#
#             integrand(t,x) = C.I.g(ForwardDiff.gradient(y->C.I.f(y), vcat(x[1:end-1],t*x[end]))[end])
#
#             function Ct(x)
#                 lb = 0.0
#                 ub = 1.0
#                 prob = QuadratureProblem(integrand,lb,ub,x)
#                 out = evaluatef0(x) + x[end]*solve(prob,QuadGKJL())[1]
#                 return out
#             end
#
#             log_pdfCt(x) = log_pdf(Ct(x)) + log(C.I.g(ForwardDiff.gradient(z->C.I.f(z),x)[end]))
# 			@show "hello"
#             @inbounds for i=1:Ne
# 				@test norm(log_pdfCt(X[:,i]) - logC[i])<atol
#                 @test norm(ForwardDiff.gradient(log_pdfCt, X[:,i]) - dxlogC[i,:])<atol
#                 # Works but too costly, use FiniteDiff instead
# #                 @test norm(ForwardDiff.fhessian(log_pdfCt, X[:,i]) - d2xlogC[i,:,:])<atol
#                 @test norm(FiniteDiff.finite_difference_hessian(log_pdfCt, X[:,i]) - d2xlogC[i,:,:])<1000*atol
#             end
#             end
#
#         end
#     end
# end


@testset "Verify that initial map is identity" begin
	Blist = ["ProHermiteBasis"; "PhyHermiteBasis";
            "CstProHermiteBasis"; "CstPhyHermiteBasis";
            "CstLinProHermiteBasis"; "CstLinPhyHermiteBasis"]
		for b in Blist
			H = HermiteMapComponent(3, 2; α = 1e-6, b = b)
			x = randn(2,1)
			Hx = evaluate(H.I, x)
			@test abs(Hx[1] - x[2])<1e-10
		end
end

@testset "Verify getcoeff, setcoeff!, clearcoeff!" begin
	m = 5
	Nx = 10
	Nψ = 5
	idx = rand(0:m, Nψ, Nx)

	coeff = randn(Nψ)

	M = HermiteMapComponent(m, Nx, idx, deepcopy(coeff))
	@test norm(getcoeff(M) - coeff)<1e-12

	coeff2 = randn(Nψ)
	setcoeff!(M, deepcopy(coeff2))

	@test norm(getcoeff(M) - coeff2)<1e-12

	clearcoeff!(M)

	@test norm(getcoeff(M) - zeros(Nψ))<1e-12
end

# @testset "Verify loss function and its gradient" begin
#
#     Nx = 2
#     Ne = 8
#     m = 5
#
#     X     =  [0.267333   1.43021;
#               0.364979   0.607224;
#              -1.23693    0.249277;
#              -2.0526     0.915629;
#              -0.182465   0.415874;
#               0.412907   1.01672;
#               1.41332   -0.918205;
#               0.766647  -1.00445]';
#     B = MultiBasis(CstProHermiteBasis(3), Nx)
#
#     idx = [0 0; 0 1; 1 0; 2 1; 1 2]
#     Nψ = 5
#
#     coeff =   [0.6285037650645056;
#      -0.4744029092496623;
#       1.1405280011620331;
#      -0.7217760771894809;
#       0.11855056306742319]
#     f = ExpandedFunction(B, idx, coeff)
#     R = IntegratedFunction(f)
#     H = HermiteMapComponent(R; α = 0.0)
#     S = Storage(H.I.f, X);
#
#    res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, H, X)), coeff, Optim.BFGS())
#    coeff_opt = Optim.minimizer(res)
#
#     @test norm(coeff_opt - [3.015753764546621;
#                           -2.929908252283099;
#                           -3.672401233483867;
#                            2.975554571687243;
#                            1.622308437415610])<1e-4
#
#     # Verify with L-2 penalty term
#
#     H = HermiteMapComponent(R; α = 0.1)
#     S = Storage(H.I.f, X);
#
#     res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S, H, X)), coeff, Optim.BFGS())
#     coeff_opt = Optim.minimizer(res)
#
#     @test norm(coeff_opt - [ -0.11411931034615422;
#                              -0.21942146397348522;
#                              -0.17368042128948974;
#                               0.37348086659548607;
#                               0.02434745831060741])<1e-4
# end
#
# @testset "Verify log_pdf function" begin
#
#   Nx = 2
#   X  =  Matrix([0.267333   1.43021;
#           0.364979   0.607224;
#          -1.23693    0.249277;
#          -2.0526     0.915629;
#          -0.182465   0.415874;
#           0.412907   1.01672;
#           1.41332   -0.918205;
#           0.766647  -1.00445]');
#   B = MultiBasis(CstProHermiteBasis(6), Nx)
#
#   idx = [0 0; 0 1; 1 0; 0 2; 2 0; 1 1]
#
#   Nψ = 5
#
#   coeff = [ -0.9905841755746164;
#         0.6771992097558741;
#        -2.243695806805015;
#        -0.34591297359447354;
#        -1.420159186008486;
#        -0.5361337327704369]
#   f = ExpandedFunction(B, idx, coeff)
#   C = HermiteMapComponent(f)
#
#   @test norm(log_pdf(C, X) - [  -1.572509004118956
#                                 -2.870725221050853
#                                 -1.285696671132943
#                                 -1.088115266085997
#                                 -2.396567575345843
#                                 -2.238446803642176
#                                 -5.999143198546611
#                                 -7.082679248037675])<1e-8
# end


@testset "Verify evaluation of HermiteMapComponent" begin

	Ne = 500
	Blist = [CstProHermiteBasis(8); CstPhyHermiteBasis(8); CstLinProHermiteBasis(8); CstLinPhyHermiteBasis(8)]

	for Nx = 1:4
		for b in Blist
			X = randn(Nx, Ne)
			B = MultiBasis(b, Nx)
			Nψ = 5
			idx = vcat(zeros(Int64, 1, Nx), rand(0:3,Nψ-1, Nx))

			coeff = randn(Nψ)
			f = ExpandedFunction(B, idx, coeff)
			R = IntegratedFunction(f)

			C = HermiteMapComponent(R)

			# Test evaluate
			ψt = zeros(Ne)

			for i=1:Ne
				x = view(X,:,i)
				ψt[i] = R.f(vcat(x[1:end-1], 0.0)) + quadgk(t->R.g(ForwardDiff.gradient(y->R.f(y), vcat(x[1:end-1],t))[end]), 0, x[end])[1]
			end

			ψ = evaluate(C, X)

			@test norm(ψ - ψt)<1e-10
		end
	end
end


# @testset "Verify log_pdf, grad_x_log_pdf and hess_x_log_pdf functions" begin
#     atol = 1e-8
#     Ne = 50
#     m = 10
#     Blist = [ProHermiteBasis(8); PhyHermiteBasis(8); CstProHermiteBasis(8)]#; CstPhyHermiteBasis(8)]#; CstLinProHermiteBasis(8); CstLinPhyHermiteBasis(8)]
#     for Nx = 1:3
#         for b in Blist
#             X = randn(Nx, Ne)
#             B = MultiBasis(b, Nx)
#             Nψ = 5
#             idx = vcat(zeros(Int64, 1, Nx), rand(0:2,Nψ-1, Nx))
#             coeff = randn(Nψ)
#             C = HermiteMapComponent(m, Nx, idx, coeff; b = string(typeof(b)))
#
# 			logC = log_pdf(C, X)
#             dxlogC  = grad_x_log_pdf(C, X)
#             d2xlogC = hess_x_log_pdf(C, X)
#
#
#             function evaluatef0(x)
#                 y = copy(x)
#                 y[end] = 0.0
#                 return C.I.f(y)
#             end
#
#             integrand(t,x) = C.I.g(ForwardDiff.gradient(y->C.I.f(y), vcat(x[1:end-1],t*x[end]))[end])
#
#             function Ct(x)
#                 lb = 0.0
#                 ub = 1.0
#                 prob = QuadratureProblem(integrand,lb,ub,x)
#                 out = evaluatef0(x) + x[end]*solve(prob,QuadGKJL())[1]
#                 return out
#             end
#
#             log_pdfCt(x) = log_pdf(Ct(x)) + log(C.I.g(ForwardDiff.gradient(z->C.I.f(z),x)[end]))
#
#             @inbounds for i=1:Ne
# 				@test norm(log_pdfCt(X[:,i]) - logC[i])<atol
#                 @test norm(ForwardDiff.gradient(log_pdfCt, X[:,i]) - dxlogC[i,:])<atol
#                 # Works but too costly, use FiniteDiff instead
# #                 @test norm(ForwardDiff.fhessian(log_pdfCt, X[:,i]) - d2xlogC[i,:,:])<atol
#                 @test norm(FiniteDiff.finite_difference_hessian(log_pdfCt, X[:,i]) - d2xlogC[i,:,:])<1000*atol
#             end
#             end
#
#         end
#     end
# end


@testset "Test loss function and its gradient" begin

Nx = 2
Ne = 100

atol = 1e-8

X = Matrix([   0.096932076386797  -0.062113573952594
   1.504907056818036  -1.270761688825393
   0.122588515870727   0.138604054973047
   0.095040305032907  -0.097989669097228
   0.056360334325948  -0.098106117790437
   0.034757567319988  -0.152864122871058
   0.260831995265119  -0.564465060524812
   0.191690179163063   0.097123146831467
   0.131835643270215  -0.303532750891873
   0.121404994711150  -0.414223062306829
  -0.046027217149733   0.267918560101165
   0.222891154723706   0.054734106232808
   0.264238681579693  -0.561044970267127
   0.021345749974379   0.023020769850203
   0.395423632217738  -0.619284581169702
   0.231607199656522   0.319089487692097
   0.389505867820041   0.572610562725449
  -0.146845693861564  -0.007959126279965
   0.138489687840940   0.330481872227252
   1.617286792209003  -1.272765331253319
   0.013078993247482   0.006243503157725
   0.304266481947197  -0.508127858160603
  -0.134224572142272  -0.057784550290700
   0.058245459428341  -0.388163133932979
   0.143773871017238  -0.569911233902741
  -0.020852128497605   0.159888469685458
   0.042924979621615  -0.285726823344180
   0.566514904142701  -0.815010017214964
   0.022782788493659  -0.468191560991914
  -0.072005082647238  -0.139317736738881
  -0.088861617714439  -0.042323531263036
   0.146019714866274  -0.441835803838725
   0.062827433058116   0.255887684932700
   0.024887528661752  -0.036316627235643
   0.079288422911736   0.009625876863572
   0.124929172288805  -0.196711126223702
   0.886085483369468   1.012837297288688
  -0.062868708331948   0.065147939699519
   0.117047537973873  -0.301180703268764
   0.147117619773736   0.466610279392604
   0.690523534483231   0.722717906410571
  -0.110919157441807   0.000008657527041
   2.055034349742512   1.423365847218695
   0.091130503033388  -0.192933472923564
   0.075510323340039   0.075087088602955
  -0.181121769738736  -0.158594842114751
  -0.080866378396288  -0.081393537160189
   0.184561993178819   0.409693133491638
  -0.032018126656783   0.040851821749292
   0.702808256935611   0.978185567019995
  -0.029041940950512   0.097268795862442
   1.406479923500753  -1.166340130567743
  -0.109758856235453  -0.297180107753266
   0.236494055349260   0.548223372779986
  -0.108550945852467  -0.097740533128956
   0.908726493797006  -0.957752761353643
   0.178092997560103  -0.359297653971968
   1.103683368372646  -1.064482193617671
   1.907065203740822   1.349542162065178
   0.523722913693736   0.725695151794695
   0.261131020580618   0.576363120484564
   0.118044539009197   0.196304662470752
   0.289261334786348   0.399639383890177
   0.902906400981006  -0.957301599045371
  -0.054657884786803  -0.292760625531884
  -0.021735291600425   0.029650166664389
   0.065200888050752  -0.295894159582647
   1.486186253485923  -1.217814186300608
   0.889545420155124   0.939789761164950
  -0.174386606410644  -0.092037014472893
  -0.065037226616579   0.009771974040525
   0.074486430996400  -0.287910597788305
   0.174336742307535  -0.400464726305446
   0.096781997529499  -0.153178844921250
   0.796408810115516   0.881930856163525
   0.005874471676882   0.067750993468743
   0.156654113730103  -0.239182272065197
   0.333688106104332  -0.629954291766549
   0.086388606764696   0.305488995071947
   0.211268899950691  -0.299878322704640
   0.104223240397571   0.199354790284364
   0.336858958710283  -0.620166113933000
   0.145071750152222   0.250136305618056
   0.032242317206686  -0.233223578816564
   0.064558616046395  -0.007577632839606
  -0.055022872335109  -0.190212128003969
  -0.169436992810515  -0.206948615170099
   0.150088939478557   0.090560438773547
   0.256256842094403   0.598874371523965
   0.340882741111244  -0.516594535669757
   0.278186325120137  -0.547968005821968
   0.645979568807173  -0.827959899268083
   0.436535833804569   0.689957746461832
   0.268437499571141   0.341281325011944
   0.120485843238972  -0.301999984465851
   0.160365386980321  -0.202012022307596
   0.154560496835611   0.244912011403144
   0.117966782622546   0.342990099492354
   0.280465408470057  -0.526206432878627
   1.174002195932550   1.136790584581798]')

L = LinearTransform(X)
X = transform(L, X)

Nψ = 5

idx = [0 0; 0 1;  1 0; 1 1; 1 2]
coeff =  [  -0.08490347737829938;
              1.0107939341310863;
              0.8792816228540664;
             -0.2535482121318712;
              0.7462002599336889]

	Blist = [ProHermiteBasis(8); PhyHermiteBasis(8); CstProHermiteBasis(8); CstPhyHermiteBasis(8); CstLinProHermiteBasis(8); CstLinPhyHermiteBasis(8)]

	for b in Blist
		B = MultiBasis(b, 2)
		f = ExpandedFunction(B, idx, coeff)
		C = HermiteMapComponent(IntegratedFunction(f); α = 0.0)
		S = Storage(C.I.f, X)

		ψoff = evaluate_offdiagbasis(C.I.f, X)
		ψd0 = evaluate_diagbasis(C.I.f, vcat(X[1:end-1,:], zeros(1, Ne)))
		∂ψd = grad_xd_diagbasis(C.I.f, X)

		function loss_coeff(c)
		    R = (ψoff .* ψd0)*c +
		    X[end,:] .* quadgk(t-> C.I.g.(ψoff .*
		    grad_xd_diagbasis(C.I.f, vcat(X[1:end-1,:], t*X[end:end,:]))*c), 0.0, 1.0)[1]
		    δ = 1e-9
		    R .+= δ*X[end,:]
		    quad = 1/Ne*sum(x->0.5*x^2+ 0.5*log(2*π), R)
		    logterm = 1/Ne*(-1.0)*sum(log.(C.I.g.( (ψoff .* ∂ψd)*c) .+ δ))
		    return quad + logterm
		end

		Jt = loss_coeff(coeff)
		dJt = ForwardDiff.gradient(loss_coeff, coeff)

		dJ = zeros(Nψ)
		J = negative_log_likelihood!(0.0, dJ, getcoeff(C), S, C, X)

		@test norm(Jt - J)<atol
		@test norm(dJt - dJ)<atol
	end
end

# Code for optimization with Hessian
# X = randn(Nx, Ne) .* randn(Nx, Ne)
# S = Storage(H.I.f, X; hess = true);
#
# J = 0.0
# dJ = zeros(Nψ)
# d2J = zeros(Nψ, Nψ)
# hess_negative_log_likelihood!(J, dJ, d2J, coeff, S, H, X)
#
# res = Optim.optimize(Optim.only_fgh!(hess_negative_log_likelihood!($S, $H, $X)), $coeff, Optim.NewtonTrustRegion())
