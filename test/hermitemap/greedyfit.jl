
@testset "Test function update_coeffs I" begin

    Nx = 2
    Ne = 8
    m = 5
    X = zeros(Nx, Ne)

    X .=  [0.267333   1.43021;
              0.364979   0.607224;
             -1.23693    0.249277;
             -2.0526     0.915629;
             -0.182465   0.415874;
              0.412907   1.01672;
              1.41332   -0.918205;
              0.766647  -1.00445]';
    # Initialize map with zero coefficients
    C_old = HermiteMapComponent(m, Nx; α = 1e-6, b= "CstProHermiteBasis");

    setcoeff!(C_old, [1.5])
    reduced_margin = getreducedmargin(getidx(C_old))
    @show reduced_margin
    idx_new = vcat(getidx(C_old), reduced_margin)

    C_new = HermiteMapComponent(m, Nx, idx_new, vcat(getcoeff(C_old), zeros(2)));

    coeff_new, coeff_idx_added, idx_added = update_coeffs(C_old, C_new)

    @test norm(coeff_new - [1.5; 0.0; 0.0])<1e-8
    @test coeff_idx_added == [2; 3]
    @test idx_added == [0 1; 1 0]
end

@testset "Test function update_coeffs II" begin

    Nx = 2
    Ne = 8
    m = 5
    X = zeros(Nx, Ne)

    X    .=  [0.267333   1.43021;
              0.364979   0.607224;
             -1.23693    0.249277;
             -2.0526     0.915629;
             -0.182465   0.415874;
              0.412907   1.01672;
              1.41332   -0.918205;
              0.766647  -1.00445]';
    # Initialize map with zero coefficients
    C_old = HermiteMapComponent(m, Nx, [0 0; 1 0; 2 0; 0 1; 0 2; 1 1], randn(6); α = 1e-6);

    reduced_margin = getreducedmargin(getidx(C_old))
    idx_new = vcat(getidx(C_old), reduced_margin)
    C_new = HermiteMapComponent(m, Nx, idx_new, vcat(getcoeff(C_old), zeros(4)));
    coeff_new, coeff_idx_added, idx_added = update_coeffs(C_old, C_new)

    @test norm(coeff_new - vcat(getcoeff(C_old), zeros(4)))<1e-8
    @test coeff_idx_added == [7; 8; 9; 10]
    @test idx_added == [0 3; 1 2; 2 1; 3 0]
end


@testset "Test function update_coeffs III" begin
    Nx = 3
    m = 5
    # Initialize map with zero coefficients
    C_old = HermiteMapComponent(m, Nx, [0 0 0; 0 0 1; 0 0 2; 0 1 0; 0 2 0], randn(5); α = 1e-6);

    reduced_margin = getreducedmargin(getidx(C_old))
    idx_new = vcat(getidx(C_old), reduced_margin)
    C_new = HermiteMapComponent(m, Nx, idx_new, vcat(getcoeff(C_old), zeros(4)));
    coeff_new, coeff_idx_added, idx_added = update_coeffs(C_old, C_new)

    @test norm(coeff_new - vcat(getcoeff(C_old), zeros(4)))<1e-8
    @test coeff_idx_added == [6; 7; 8; 9]
    @test idx_added == reduced_margin
end

@testset "Test update_component I" begin
    Nx = 2
    Ne = 8
    m = 5
    X = zeros(Nx, Ne)

    X .=     [0.267333   1.43021;
              0.364979   0.607224;
             -1.23693    0.249277;
             -2.0526     0.915629;
             -0.182465   0.415874;
              0.412907   1.01672;
              1.41332   -0.918205;
              0.766647  -1.00445]';
    X = X
    # Initialize map with zero coefficients
    C_old = HermiteMapComponent(m, Nx; α = 1e-6);

    S = Storage(C_old.I.f, X)

    dJ_old = zeros(1)
    J_old = 0.0
    J_old = negative_log_likelihood!(J_old, dJ_old, getcoeff(C_old), S, C_old, X)

    @test abs(J_old - 1.317277788545110)<1e-10

    @test norm(dJ_old - [0.339034875000000])<1e-10


    # setcoeff!(C_old, [1.5])
    reduced_margin0 = getreducedmargin(getidx(C_old))
    idx_old0 = getidx(C_old)
    idx_new0 = vcat(idx_old0, reduced_margin0)

    # Define updated map
    f_new = ExpandedFunction(C_old.I.f.MB, idx_new0, vcat(getcoeff(C_old), zeros(size(reduced_margin0,1))))
    C_new = HermiteMapComponent(f_new; α = 1e-6)
    idx_new, reduced_margin = update_component!(C_old, X, reduced_margin0, S)

    dJ_new = zeros(3)
    J_new = 0.0
    S = update_storage(S, X, reduced_margin0)
    J_new = negative_log_likelihood!(J_new, dJ_new, getcoeff(C_new), S, C_new, X)


    @test abs(J_new -    1.317277788545110)<1e-8

    @test norm(dJ_new - [0.339034875000000;   0.228966410748600;   0.192950409869679])<1e-8


    @test findmax(abs.(dJ_new[2:3]))[2] == 2-1

    @test idx_new0[3,:] == reduced_margin0[2,:]

    @test reduced_margin == updatereducedmargin(getidx(C_old), reduced_margin0, 1)[2]
    @test idx_new == updatereducedmargin(getidx(C_old), reduced_margin0, 1)[1]
end

@testset "Test update_component II" begin
    Nx = 2
    Ne = 8
    m = 5
    X = zeros(Nx, Ne)

    X .=     [0.267333   1.43021;
              0.364979   0.607224;
             -1.23693    0.249277;
             -2.0526     0.915629;
             -0.182465   0.415874;
              0.412907   1.01672;
              1.41332   -0.918205;
              0.766647  -1.00445]';
    X = X

    idx = [0 0; 0 1; 1 0; 0 2; 2 0; 1 1]

    coeff = [ -0.9905841755746164;
          0.6771992097558741;
         -2.243695806805015;
         -0.34591297359447354;
         -1.420159186008486;
         -0.5361337327704369]

    C_old = HermiteMapComponent(m, Nx, idx, coeff; α = 1e-6);

    S = Storage(C_old.I.f, X)

    dJ_old = zeros(6)
    J_old = 0.0
    J_old = negative_log_likelihood!(J_old, dJ_old, getcoeff(C_old), S, C_old, X)

    @test abs(J_old - 3.066735373495125)<1e-5

    dJt_old  = [-1.665181693060798;
                -0.919442416275569;
                -0.900922248322526;
                -0.119522662066802;
                -0.472794468187974;
                -0.496862354324658]

    @test norm(dJ_old - dJt_old)<1e-5



    reduced_margin0 = getreducedmargin(getidx(C_old))
    idx_old0 = getidx(C_old)
    idx_new0 = vcat(idx_old0, reduced_margin0)

    # Define updated map
    f_new = ExpandedFunction(C_old.I.f.MB, idx_new0, vcat(getcoeff(C_old), zeros(size(reduced_margin0,1))))
    C_new = HermiteMapComponent(f_new; α = 1e-6)
    idx_new, reduced_margin = update_component!(C_old, X, reduced_margin0, S)

    dJ_new = zeros(10)
    J_new = 0.0
    S = update_storage(S, X, reduced_margin0)
    J_new = negative_log_likelihood!(J_new, dJ_new, getcoeff(C_new), S, C_new, X)

     Jt_new  = 3.066735373495125
    dJt_new  = [-1.665181693060798;
                -0.919442416275569;
                -0.900922248322526;
                -0.119522662066802;
                -0.472794468187974;
                -0.496862354324658;
                 0.338831623096620;
                -0.083502724667314;
                -0.281130533740555;
                 0.271587190562667]

    @test abs(J_new -    Jt_new)<1e-5
    @test norm(dJ_new - dJt_new)<1e-5


    @test findmax(abs.(dJ_new[7:10]))[2] == 7-6

    @test idx_new0[8,:] == reduced_margin0[2,:]

    @test reduced_margin == updatereducedmargin(getidx(C_old), reduced_margin0, 1)[2]
    @test idx_new == updatereducedmargin(getidx(C_old), reduced_margin0, 1)[1]
end

@testset "Test greedy optimization on training set only without QR " begin

Nx = 2
m = 15

X = Matrix([0.267333   1.43021;
          0.364979   0.607224;
         -1.23693    0.249277;
         -2.0526     0.915629;
         -0.182465   0.415874;
          0.412907   1.01672;
          1.41332   -0.918205;
          0.766647  -1.00445]')
L = LinearTransform(X)
# transform!(L, X)
C_new, error = greedyfit(m, Nx, X, 6; withconstant = false, withqr = false, verbose = false)

@test norm(error -  [1.317277788545110;
                     1.240470006245235;
                     1.239755181933741;
                     1.191998917636778;
                     0.989821300578785;
                     0.956587763340890;
                     0.848999846549310])<1e-4


@test norm(getcoeff(C_new) - [ 2.758739330966050;
                               4.992420909046952;
                               0.691952693078807;
                               5.179459837781270;
                              -3.837569097976292;
                               1.660285776681347])<1e-4

@test norm(getidx(C_new) -   [0  1;
                              0  2;
                              0  3;
                              0  4;
                              1  0;
                              2  0])<1e-10
end

@testset "Test greedy optimization on training set only with QR " begin

Nx = 2
m = 15

X = Matrix([0.267333   1.43021;
          0.364979   0.607224;
         -1.23693    0.249277;
         -2.0526     0.915629;
         -0.182465   0.415874;
          0.412907   1.01672;
          1.41332   -0.918205;
          0.766647  -1.00445]')
L = LinearTransform(X)
# transform!(L, X)
C_new, error = greedyfit(m, Nx, X, 6; withconstant = false, withqr = true, verbose = false)

@test norm(error -  [1.317277788545110;
                     1.240470006245235;
                     1.239755181933741;
                     1.191998917636778;
                     0.989821300578785;
                     0.956587763340890;
                     0.848999846549310])<1e-4


@test norm(getcoeff(C_new) - [ 2.758739330966050;
                               4.992420909046952;
                               0.691952693078807;
                               5.179459837781270;
                              -3.837569097976292;
                               1.660285776681347])<1e-4

@test norm(getidx(C_new) -   [0  1;
                              0  2;
                              0  3;
                              0  4;
                              1  0;
                              2  0])<1e-10
end

@testset "Test greedy optimization on train dataset without/with QR" begin

Nx = 2
m = 10
Ne = 10^3
X = randn(2, Ne).^2 + 0.1*randn(2, Ne)
L = LinearTransform(X)
transform!(L, X)

C_noqr, error_noqr = greedyfit(m, Nx, X, 10; withconstant = true, withqr = false, verbose = false);
C_qr, error_qr = greedyfit(m, Nx, X, 10; withconstant = true, withqr = true, verbose = false);

train_error_noqr = error_noqr
train_error_qr   = error_qr


C_test = deepcopy(C_noqr)
setcoeff!(C_test, zero(getcoeff(C_noqr)));

S_test = Storage(C_test.I.f, X)
coeff_test = getcoeff(C_test)

res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S_test, C_test, X)), coeff_test, Optim.BFGS())

@test norm(getcoeff(C_noqr)-Optim.minimizer(res))<5e-5
@test norm(getcoeff(C_qr)-Optim.minimizer(res))<5e-5


@test norm(train_error_noqr[end] - res.minimum)<5e-5
@test norm(train_error_qr[end]   - res.minimum)<5e-5

end

@testset "Test greedy optimization on train/valid dataset without/with QR" begin

Nx = 2
m = 10
Ne = 10^3
X = randn(2, Ne).^2 + 0.1*randn(2, Ne)
L = LinearTransform(X)
transform!(L, X)

X_train = X[:,1:800]
X_valid = X[:,801:1000]


C_noqr, error_noqr = greedyfit(m, Nx, X_train, X_valid, 10; withconstant = true, withqr = false, verbose = false);
C_qr, error_qr = greedyfit(m, Nx, X_train, X_valid, 10; withconstant = true, withqr = true, verbose = false);

train_error_noqr, valid_error_noqr = error_noqr
train_error_qr, valid_error_qr = error_qr

C_test = deepcopy(C_noqr)
setcoeff!(C_test, zero(getcoeff(C_noqr)));

S_test = Storage(C_test.I.f, X_train)
coeff_test = getcoeff(C_test)

res = Optim.optimize(Optim.only_fg!(negative_log_likelihood(S_test, C_test, X_train)), coeff_test, Optim.BFGS())

@test norm(getcoeff(C_noqr) - Optim.minimizer(res))<5e-5
@test norm(getcoeff(C_qr)  - Optim.minimizer(res))<5e-5


@test norm(train_error_noqr[end] - res.minimum)<5e-5
@test norm(train_error_qr[end] - res.minimum)<5e-5
@test norm(valid_error_noqr[end] - valid_error_qr[end])<5e-5

end

@testset "Test greedyfit routine with/without QR, with/without preconditioner" begin

    Nx = 2
    Ne = 100

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

    nterms = 15

	# For b = "ProHermiteBasis"

    Cnoqrnoprecond, error_noqrnoprecond = greedyfit(10, 2, X, nterms; α = 0.0, withconstant = false,
	                   withqr = false,
                       hessprecond = false, b = "ProHermiteBasis",
                       ATMcriterion = "gradient")

	Cqrnoprecond, error_qrnoprecond = greedyfit(10, 2, X, nterms; α = 0.0, withconstant = false,
	                  withqr = true,
	                  hessprecond = false, b = "ProHermiteBasis",
	                  ATMcriterion = "gradient")


	Cnoqrprecond, error_noqrprecond = greedyfit(10, 2, X, nterms; α = 0.0, withconstant = false,
  	                     withqr = false,
                         hessprecond = true, b = "ProHermiteBasis",
                         ATMcriterion = "gradient")

  	Cqrprecond, error_qrprecond = greedyfit(10, 2, X, nterms; α = 0.0, withconstant = false,
  	                  withqr = true,
  	                  hessprecond = true, b = "ProHermiteBasis",
  	                  ATMcriterion = "gradient")

	error_true =   [1.413939;
					1.410124;
					1.403666;
					1.398928;
					1.398540;
					1.374076;
					1.374064;
					1.368187;
					1.308003;
					1.233256;
					1.230376;
					1.177314;
					1.155683;
					1.138491;
					1.117992]

	idx_true =      [0     0;
				     0     1;
				     0     2;
				     1     0;
				     0     3;
				     0     4;
				     2     0;
				     3     0;
				     4     0;
				     5     0;
				     0     5;
				     0     6;
				     0     7;
				     0     8;
				     6     0]
	coeff_true = [0.105975175137040; -64.829184645818373;   3.644736570204142;  34.333722919244586; -42.641654439858939;  10.491202988540742;   6.170682598059267;
				  15.243539772142837; 13.033494780747679;  3.833643021385606; -25.133002059654828;  12.504145228058636;  -7.625230653938331;   4.946600220335136;   7.531771217294497]				

	# For b = "CstProHermiteBasis"
end
