# Estimation of dynamic discrete choice model with real data
    # Steel manufacturing
    # With selection on unobservables
    # Monopolistic Competition
# Emmanuel Murray Leclair
# August 2023

## Home directory
computer = gethostname() ;
cd("/project/6001227/emurrayl/DynamicDiscreteChoice") # Sharcnet
sharcnet = true;

empty!(DEPOT_PATH)
push!(DEPOT_PATH, "/project/6001227/emurrayl/julia")

## Make auxiliary directores
Fig_Folder = "/project/6001227/emurrayl/DynamicDiscreteChoice/Figures"
mkpath(Fig_Folder)
Result_Folder = "/project/6001227/emurrayl/DynamicDiscreteChoice/Results"
mkpath(Result_Folder)

# Load packages
using SparseArrays, Interpolations, Dierckx, ForwardDiff, Optim, Roots, Parameters, Kronecker, Plots, StatsPlots, NLopt, Distributions, QuantEcon, HDF5
using CSV, DataFrames, DiscreteMarkovChains, StructTypes, StatsBase, Distributed, SharedArrays, DelimitedFiles, NLsolve, ParallelDataTransfer, CUDA_Runtime_jll, CUDA
using FiniteDiff, BenchmarkTools, Distances, JLD2, FileIO

# Load externally written functions
include("/project/6001227/emurrayl/DynamicDiscreteChoice/VFI_Toolbox.jl")


println(" ")
println("-----------------------------------------------------------------------------------------")
println("Dynamic production and fuel set choice in Julia - Steel Manufacturing Data With Selection")
println(        "Estimation routine with Pipeline network and Monopolistic Competition"            )
println("                     Discount factor related to real interest rate"                       )
println("-----------------------------------------------------------------------------------------")
println(" ")

#-----------------------------------------------------------
#-----------------------------------------------------------
# 1. Parameters, Model and Data Structure

# Generate structure for parameters using Parameters module
    # Import Parameters from Steel manufacturing data
    @with_kw struct Par
        p_all = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_all.txt", DataFrame)[:,2:(end-1)]          ;
        p_cov = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_cov.txt", DataFrame)[:,2:(end-1)]          ;
        # Production function parameters
            σ=p_all[1,1];
            η=p_all[1,2];
            αk=p_all[1,3];
            αl=p_all[1,4];
            αm=p_all[1,5];
            αe=p_all[1,6];
            λ=p_all[1,7];
            βo=p_all[1,8];
            βg=p_all[1,9];
            βc=p_all[1,10];
            βe=p_all[1,11];
        #
        # State transition: hicks-neutral productivity z
            ρz=p_all[2,1];
            σz=p_all[2,2];
            μz=p_all[2,3];
            μz_t = μz .+ [0,p_all[2,4],p_all[2,5],p_all[2,6],p_all[2,7],p_all[2,8],p_all[2,9]];
        #
        # State transition: productivity of electricity ψe
            ρ_ψe=p_all[3,1];
            σ_ψe=p_all[3,2];
            μψe=p_all[3,3];
            μψe_t = μψe .+ [0,p_all[3,4],p_all[3,5],p_all[3,6],p_all[3,7],p_all[3,8],p_all[3,9]];
            cov_peψe=p_all[3,11];
        #
        # State transition: price of electricity pe
            ρ_pe=p_all[4,1];
            σ_pe=p_all[4,2];
            μpe=p_all[4,3];
            μpe_t = μpe .+ [0,p_all[4,4],p_all[4,5],p_all[4,6],p_all[4,7],p_all[4,8],p_all[4,9]];
        #
        # State transition: productivity of oil
            ρ_ψo=p_all[5,1];
            σ_ψo=p_all[5,2];
            μψo=p_all[5,3];
            μψo_t = μψo .+ [0,p_all[5,4],p_all[5,5],p_all[5,6],p_all[5,7],p_all[5,8],p_all[5,9]];
        #
        # Price of oil
            po = [p_all[6,1],p_all[6,2],p_all[6,3],p_all[6,4],p_all[6,5],p_all[6,6],p_all[6,7],p_all[6,8]];
        #
        # State transition: productivity of gas (no RE)
            σ_ψg=p_all[7,1];
            μψg=p_all[7,2];
            μψg_t = μψg .+ [0,p_all[7,3],p_all[7,4],p_all[7,5],p_all[7,6],p_all[7,7],p_all[7,8],p_all[7,9]];
            cov_pgψg = p_all[7,10];
        #
        # State transition: price of gas (no RE)
            σ_pg=p_all[8,1];
            μpg=p_all[8,2];
            μpg_t = μpg .+ [0,p_all[8,3],p_all[8,4],p_all[8,5],p_all[8,6],p_all[8,7],p_all[8,8],p_all[8,9]];
        #
        # First guess of mean and variance of gas RE
            μgre_init = p_all[7,11];
            σgre_init = p_all[7,12];
        #
        # State transition: price over productivity of gas (no RE)
            μpgψg = μpg-μψg;
            μpgψg_t = μpg_t.-μψg_t;
            σ_pgψg = σ_ψg+σ_pg-(2*cov_pgψg);
        #
        # State transition: productivity of coal (no RE)
            σ_ψc=p_all[9,1];
            μψc=p_all[9,2];
            μψc_t = μψc .+ [0,p_all[9,3],p_all[9,4],p_all[9,5],p_all[9,6],p_all[9,7],p_all[9,8],p_all[9,9]];
            cov_pcψc = p_all[9,10];
        #
        # State transition: price of coal (no RE)
            σ_pc=p_all[10,1];
            μpc=p_all[10,2];
            μpc_t = μpc .+ [0,p_all[10,3],p_all[10,4],p_all[10,5],p_all[10,6],p_all[10,7],p_all[10,8],p_all[10,9]];
        #
        # First guess of mean and variance of gas RE
            μcre_init = p_all[9,11];
            σcre_init = p_all[9,12];
        #
        # State transition: price over productivity of coal (no RE)
            μpcψc = μpc-μψc;
            μpcψc_t = μpc_t.-μψc_t;
            σ_pcψc = σ_ψc+σ_pc-(2*cov_pcψc);
        #
        # State transition: price of materials
            ρ_pm=p_all[11,1];
            σ_pm=p_all[11,2];
            μpm=p_all[11,3];
            μpm_t = μpm .+ [0,p_all[11,4],p_all[11,5],p_all[11,6],p_all[11,7],p_all[11,8],p_all[11,9]];
        #
        # Price of materials (used if not a state variable)
            pm = [p_all[12,1],p_all[12,2],p_all[12,3],p_all[12,4],p_all[12,5],p_all[12,6],p_all[12,7],p_all[12,8]];
        #
        # Wages, rental rate of capital, output prices and emission factors
            w =  [p_all[13,1],p_all[13,2],p_all[13,3],p_all[13,4],p_all[13,5],p_all[13,6],p_all[13,7],p_all[13,8]];
            rk = [p_all[14,1],p_all[14,2],p_all[14,3],p_all[14,4],p_all[14,5],p_all[14,6],p_all[14,7],p_all[14,8]];
            pout_struc = [p_all[15,1],p_all[15,2],p_all[15,3],p_all[15,4],p_all[15,5],p_all[15,6],p_all[15,7],p_all[15,8]];
            pout_init = [p_all[15,1],p_all[15,2],p_all[15,3],p_all[15,4],p_all[15,5],p_all[15,6],p_all[15,7],p_all[15,8]];
            Ygmean = [p_all[16,1],p_all[16,2],p_all[16,3],p_all[16,4],p_all[16,5],p_all[16,6],p_all[16,7],p_all[16,8]];
            γ = [p_all[17,1],p_all[17,2],p_all[17,3],p_all[17,4]]; # In order: electricity, natural gas, coal and oil
            γe = γ[1];
            γg = γ[2];
            γc = γ[3];
            γo = γ[4];
        #
        # Geometric mean of fuel quantities 
            fgmean = [p_all[18,1],p_all[18,2],p_all[18,3],p_all[18,4]];
            ogmean = fgmean[1];
            ggmean = fgmean[2];
            cgmean = fgmean[3];
            egmean = fgmean[4];
        #
        # Demand parameters
            ρ=p_all[19,1];
            θ=p_all[19,2];
            d_t = [0,p_all[19,4],p_all[19,5],p_all[19,6],p_all[19,7],p_all[19,8],p_all[19,9]];
        #
        # Number of plants by year
            N_t = [0,p_all[20,2],p_all[20,3],p_all[20,4],p_all[20,5],p_all[20,6],p_all[20,7]];
        #
        # Convergence parameters 
        max_iter::Int64     = 20000  ; # Maximum number of iterations
        dist_tol::Float64   = 1E-4   ; # Tolerance for distance
        dist_tol_Δ::Float64 = 1E-11  ; # Tolerance for change in distance 
        pout_tol::Float64   = 1E-15  ; # Tolerance for fixed point in aggregate output price index
        υ                   = 0.0    ; # Dampen factor
        # misc 
        F_tot               = 4      ; # Total number of fuels
        β                   = 0.9    ; # Discount factor 
        T_pers              = size(μψe_t,1);    # number of years, persistent variables
        T_nonpers           = size(po,1);       # Number of years, non persistent variables
        T                   = T_pers-1;                  # Actual number of years where I kept data
    end
    p = Par();
#

# Import the data
    Data = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/MainData_wPipeline-Steel.csv", DataFrame) ;
    Data[!,:id] = 1:size(Data,1);
    Data_year = groupby(Data,:year);
#

# Generate structure of model objects
    # Model : Dynamic production and fuel set choice - With Selection
    @with_kw struct Model
        # Parameters
        p::Par = Par(); # Model parameters in their own structure
        NT = size(Data,1); # Total number of observations
        N = NT;

        # Multivariate markov chain for all persistent state variables
        nstate = 5;     # Number of states in multivariate markov chain
        ngrid = 4;      # Number of grid points per 
        Πs = nothing;            # Transition matrix
        lnSgrid = nothing;       # grid in logs
        Sgrid = nothing;         # grid in levels
        
        # Price/productivity of gas non-persistent process
        n_g    = 4                                              ;  # Size of grid
        Π_g = nothing                                           ;  # Initialize transition matrix
        lng_grid = nothing                                      ;  # Initialize grid in logs
        g_grid = nothing                                        ;  # Initialize grid in levels
        lnpg_grid = nothing;
        lnψg_grid = nothing;
        pg_grid = nothing;
        ψg_grid = nothing;
        # productivity of gas random effects
        ng_re    = 3                                            ;  # Size of grid
        Πg_re = nothing                                         ;  # Initialize transition matrix
        lng_re_grid = nothing                                   ;  # Initialize grid in logs
        g_re_grid = nothing                                     ;  # Initialize grid in levels
        μg_re = nothing                                         ;  # Mean
        σg_re = nothing                                         ;  # Variance 
        # Price/productivity of coal non-persistent residual process
        n_c    = 4                                              ;  # Size of grid
        Π_c = nothing                                           ;  # Initialize transition matrix
        lnc_grid = nothing                                      ;  # Initialize grid in logs
        c_grid = nothing                                        ;  # Initialize grid in levels       
        lnpc_grid = nothing;
        lnψc_grid = nothing;
        pc_grid = nothing;
        ψc_grid = nothing;
        # productivity of coal random effects
        nc_re    = 3                                            ;  # Size of grid
        Πc_re = nothing                                         ;  # Initialize transition matrix
        lnc_re_grid = nothing                                   ;  # Initialize grid in logs
        c_re_grid = nothing                                     ;  # Initialize grid in levels     
        μc_re = nothing                                         ;  # Mean
        σc_re = nothing                                         ;  # Variance           
        # Productivity of coal and gas, joint unconditional distribution of random effects
        π_uncond = nothing         
        # Current Period profit               
        πgrid_oe = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                       ;
        πgrid_oge = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                      ;
        πgrid_oce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                      ;
        πgrid_ogce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                     ;
        # Expected Value Functions (single agent)
        V_itermax = 1000          
        nconnect=2;            # Number of possible pipeline connections (connected and not connected)                                                        ;
        Wind = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                     ; 
        Wind1 = Array{Float64}(undef,p.F_tot,ngrid^nstate,n_c,n_g,nc_re,ng_re)                        ; 
        Wind0 = Array{Float64}(undef,(ngrid^nstate)*n_c*n_g,p.F_tot)                                ;
        Wind0_full = Array{Float64}(undef,(ngrid^nstate)*n_c*n_g,p.F_tot,p.T*nc_re*ng_re)              ;
        Wind0_year = Array{Float64}(undef,(ngrid^nstate)*n_c*n_g,p.F_tot,p.T)                         ;
        W_new = Array{Float64}(undef,nconnect,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                     ; 
        # Choice-specific value function  
        vgrid_oe = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                       ;
        vgrid_oge = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                      ;
        vgrid_oce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                      ;
        vgrid_ogce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                     ;
        # Fuel prices
        pEgrid_oe = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                   ;
        pEgrid_oge = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                  ;
        pEgrid_oce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                  ;
        pEgrid_ogce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                 ;
        # Misc
        loglik_uncond = nothing;
    end
    M = Model();
#

### Store of expected value function
    mutable struct VF_bench
        W;
    end
#

# Include all relevant functions to current script
    include("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Simulation_func.jl");
    include("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Simulation_func-CUDA.jl");
#

# First guess of random effects distribution parameters (from selected empirical distribution)
    θre = [p.μgre_init,p.μcre_init,p.σgre_init,p.σcre_init];
    M = ParamRE_func(θre,M);
    test=1;
#

# Generate Function that generates state transition markov chains (multivariate markov chains)
    M = StateTransition_func_multivariate(p,M);
    test=1;
#

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# 2. Preliminary simulations to see if grid points match the generated data

# Find closest point on the grid for each state variable
    grid_indices,dist_grid_smallest = MatchGridData_pre(M,Data);
#

## Use moments of the prices and productivity for coal and gas to discretize distribution of price separately from productivity
    M = DiscretizeGasCoal_distr(p,M);
    test=1;
#

### Find equilibrium aggregate output price
    # With state variables from the grid
        # p,M = OutputPrice_func(M,Data,grid_indices,p);
    #
    # With state variables from the data
        τnotax = [0,0,0,0];
        p,M = OutputPrice_func_nogrid(M,Data,p,τnotax);
        test=1;
    #
#

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# 3. Simulation and Functions used for estimation

#  Compute Static Profits Next Period under all combination of state variables
    # Main 
        @time M = StaticProfit_grid_pre(M)   ;
        test=1;
    #
#
# Compute predicted profit under state variables in the data 
    πpred = StaticProfit_data(M,Data,τnotax);
#

### Compute and store expected value functions
    function W_max(M::Model,Κ,σ,l::VF_bench)
        ## Given a guess of the fixed cost parameters, this function returns the expected value functions
        # Initialized expected value functions
        #W_new = VFI_discrete_faster(M,Κ,l) ;  
        #W_new = VFI_discrete_evenfaster(M,Κ,σ,l) ;  
        W_new = VFI_discrete_evenfaster_zfc(M,Κ,σ,l) ;  
        # Update Model
        M = Model(M; W_new=copy(convert(Array,W_new))) ;
        return M;
    end
#

### Inner optimization (iterate log-likelihood over fixed costs)
    function (l::VF_bench)(M::Model,Data::DataFrame,grid_indices::DataFrame,π_cond,param)
        @unpack p,nc_re,ng_re = M;
        N = size(Data,1);
        Nfirms = size(unique(Data.IDnum))[1];
        Κ = param[1:8];
        σ = param[9];
        # Update expected value function
        #W_new = VFI_discrete_faster(M,Κ,l) ;  
        W_new = VFI_discrete_evenfaster_zfc(M,Κ,σ,l) ;  
        l.W = copy(W_new) ;
        # Update choice probability conditional on random effects
        Prnext_long = Array{Float64}(undef,N,nc_re,ng_re);
        for j = 1:(nc_re*ng_re)
            ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
            Prnext_long[:,ic_re,ig_re] = choicePR_func_zfc(M,Data,grid_indices,Κ,ig_re,ic_re,W_new);
        end
        # Evalutate the current conditional log-likelihood
        indloglik = Array{Float64}(undef,Nfirms,nc_re,ng_re);
        #Data_ID = groupby(Data,:IDnum);
        for j = 1:(nc_re*ng_re)
            ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
            Data.Prnext = Prnext_long[:,ic_re,ig_re];
            Data_ID = groupby(Data,:IDnum);
            for ifirm = 1:Nfirms
                # Individual likelihood (by unique plants)
                indloglik[ifirm,ic_re,ig_re] = π_cond[ifirm,ic_re,ig_re]*sum(log.(Data_ID[ifirm].Prnext));
            end
        end
        loglik = sum(indloglik);
        # Evaluate the current unconditional log-likelihood
        π_uncond = zeros(nc_re,ng_re);
        for j = 1:(nc_re*ng_re)
            ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
            π_uncond[ic_re,ig_re] = (1/Nfirms)*sum(π_cond[:,ic_re,ig_re]);
        end
        # if sum(π_uncond) ≈ 1 != true
        #     error("Probability do not sum to 1")
        # end
        indloglik_uncond = zeros(Nfirms,nc_re,ng_re); 
        Data_ID = groupby(Data,:IDnum);
        for i = 1:Nfirms
            itotal = Data_ID[i].id;
            pr_aux = zeros(size(itotal,1),nc_re,ng_re);
            for ii in eachindex(itotal)
                pr_aux[ii,:,:] = Prnext_long[itotal[ii],:,:];
            end
            aux_1 = zeros(nc_re,ng_re);
            for j = 1:(nc_re*ng_re)
                ic_re,ig_re = Tuple(CartesianIndices((nc_re,ng_re))[j]);
                aux_1[ic_re,ig_re] = π_uncond[ic_re,ig_re]*prod(pr_aux[:,ic_re,ig_re]);
            end
            indloglik_uncond[i] = log(sum(aux_1));
        end
        loglik_uncond = sum(indloglik_uncond);
        println("----------------------------------------------------------------------------")
        println("--------------------  CURRENT LIKELIHOOD EVALUATION ------------------------")
        println("Current conditional log likelihood = $loglik")
        println("Current unconditional log likelihood = $loglik_uncond")
        println("Current fixed costs = $(Κ[1:6])")
        println("Current fixed costs as a function of (log) productivity = $(Κ[7:8])")
        println("Current unit conversion factor = $σ")
        println("----------------------------------------------------------------------------")
        return -loglik;
    end
#

#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# 4. Estimation of fixed costs and distribution of comparative advantages

### Outer objective function evaluation
    function OuterEstimation_eval(M::Model,Data::DataFrame,grid_indices::DataFrame,Κ,σ)
        @unpack p = M;
        # Update value function
        if isnan(sum(M.W_new)) == true || sum(M.W_new) == 0
            W0=VF_bench(nothing);  
        else
            W0 = VF_bench(M.W_new);
        end
        M = W_max(M,Κ,σ,W0);
        # Update posterior conditional and unconditional random effect probability and choice probability
        M,π_cond,Prnext_post = PosteriorRE_func(M,Data,grid_indices,Κ);
        # Inner optimization: estimate conditional likelihood given posterior random effect probability
            nparam=7;
            if isnan(sum(M.W_new)) == true || sum(M.W_new) == 0
                loglik_eval=VF_bench(nothing);  
            else
                loglik_eval = VF_bench(M.W_new);
            end
            #opt=Opt(:LN_NEWUOA,nparam);
            opt=Opt(:LN_BOBYQA,nparam);
            function objfunc_temp(x::Vector,g::Vector)
                if length(g) > 0
                    ForwardDiff.gradient!(g, loglik_eval(M,Data,grid_indices,π_cond,x), x);
                end
                return loglik_eval(M,Data,grid_indices,π_cond,x);
            end
            opt.min_objective = objfunc_temp;
            opt.xtol_abs=1e-6;
            opt.ftol_rel=1e-6;;
            opt.lower_bounds = [-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,0.0];
            opt.upper_bounds = [Inf,Inf,Inf,Inf,Inf,Inf,Inf];
            opt.maxtime=25200; # 7h max time
            fc_est = zeros(7);
            (loglik,fc_est[:],ret) = NLopt.optimize(opt, [Κ;σ]);
            Κest = fc_est[1:6];
            σest = fc_est[7];
        #
        # Compute unconditional likelihood
            Nfirms = size(unique(Data.IDnum))[1];
            indloglik_uncond = zeros(Nfirms,M.nc_re,M.ng_re); 
            Data_ID = groupby(Data,:IDnum);
            # Update choice probability conditional on random effects
            l=VF_bench(nothing);
            W_new = VFI_discrete_evenfaster(M,Κest,σest,l) ;  
            Prnext_long = Array{Float64}(undef,M.NT,M.nc_re,M.ng_re);
            for j = 1:(M.nc_re*M.ng_re)
                ic_re,ig_re = Tuple(CartesianIndices((M.nc_re,M.ng_re))[j]);
                Prnext_long[:,ic_re,ig_re] = choicePR_func(M,Data,grid_indices,fc_est,ig_re,ic_re,W_new);
            end
            π_uncond = zeros(M.nc_re,M.ng_re);
            for j = 1:(M.nc_re*M.ng_re)
                ic_re,ig_re = Tuple(CartesianIndices((M.nc_re,M.ng_re))[j]);
                π_uncond[ic_re,ig_re] = (1/Nfirms)*sum(π_cond[:,ic_re,ig_re]);
            end
            for i = 1:Nfirms
                itotal = Data_ID[i].id;
                pr_aux = zeros(size(itotal,1),M.nc_re,M.ng_re);
                for ii in eachindex(itotal)
                    pr_aux[ii,:,:] = Prnext_long[itotal[ii],:,:]
                end
                aux_1 = zeros(M.nc_re,M.ng_re);
                for j = 1:(M.nc_re*M.ng_re)
                    ic_re,ig_re = Tuple(CartesianIndices((M.nc_re,M.ng_re))[j]);
                    aux_1[ic_re,ig_re] = π_uncond[ic_re,ig_re]*prod(pr_aux[:,ic_re,ig_re]);
                end
                indloglik_uncond[i] = log(sum(aux_1));
            end
            loglik_uncond = sum(indloglik_uncond);
        #
        println("conditional likelihood converged")
        println("conditional log-likelihood = $(-loglik)")
        println("unconditional log-likelihood = $(-loglik_uncond)")
        return M,fc_est,loglik_uncond;
    end
    function OuterEstimation_eval_zfc(M::Model,Data::DataFrame,grid_indices::DataFrame,Κ,σ)
        @unpack p = M;
        # Update value function
        if isnan(sum(M.W_new)) == true || sum(M.W_new) == 0
            W0= VF_bench(nothing);  
        else
            W0 = VF_bench(M.W_new);
        end
        M = W_max(M,Κ,σ,W0);
        # Update posterior conditional and unconditional random effect probability and choice probability
        M,π_cond,Prnext_post = PosteriorRE_func_zfc(M,Data,grid_indices,Κ);
        # Inner optimization: estimate conditional likelihood given posterior random effect probability
            nparam=9;
            if isnan(sum(M.W_new)) == true || sum(M.W_new) == 0
                loglik_eval=VF_bench(nothing);  
            else
                loglik_eval = VF_bench(M.W_new);
                test=1;
            end
            #opt=Opt(:LN_NEWUOA,nparam);
            opt=Opt(:LN_BOBYQA,nparam);
            function objfunc_temp(x::Vector,g::Vector)
                if length(g) > 0
                    ForwardDiff.gradient!(g, loglik_eval(M,Data,grid_indices,π_cond,x), x);
                end
                return loglik_eval(M,Data,grid_indices,π_cond,x);
            end
            opt.min_objective = objfunc_temp;
            opt.xtol_abs=1e-6;
            opt.ftol_rel=1e-6;;
            opt.lower_bounds = [-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,0.0];
            opt.upper_bounds = [Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf];
            opt.maxtime=25200; # 7h max time
            fc_est = zeros(9);
            (loglik,fc_est[:],ret) = NLopt.optimize(opt, [Κ;σ]);
            Κest = fc_est[1:8];
            σest = fc_est[9];
        #
        # Compute unconditional likelihood
            Nfirms = size(unique(Data.IDnum))[1];
            indloglik_uncond = zeros(Nfirms,M.nc_re,M.ng_re); 
            Data_ID = groupby(Data,:IDnum);
            # Update choice probability conditional on random effects
            l=VF_bench(nothing);
            W_new = VFI_discrete_evenfaster_zfc(M,Κest,σest,l) ;  
            Prnext_long = Array{Float64}(undef,M.NT,M.nc_re,M.ng_re);
            for j = 1:(M.nc_re*M.ng_re)
                ic_re,ig_re = Tuple(CartesianIndices((M.nc_re,M.ng_re))[j]);
                Prnext_long[:,ic_re,ig_re] = choicePR_func_zfc(M,Data,grid_indices,fc_est,ig_re,ic_re,W_new);
            end
            π_uncond = zeros(M.nc_re,M.ng_re);
            for j = 1:(M.nc_re*M.ng_re)
                ic_re,ig_re = Tuple(CartesianIndices((M.nc_re,M.ng_re))[j]);
                π_uncond[ic_re,ig_re] = (1/Nfirms)*sum(π_cond[:,ic_re,ig_re]);
            end
            for i = 1:Nfirms
                itotal = Data_ID[i].id;
                pr_aux = zeros(size(itotal,1),M.nc_re,M.ng_re);
                for ii in eachindex(itotal)
                    pr_aux[ii,:,:] = Prnext_long[itotal[ii],:,:]
                end
                aux_1 = zeros(M.nc_re,M.ng_re);
                for j = 1:(M.nc_re*M.ng_re)
                    ic_re,ig_re = Tuple(CartesianIndices((M.nc_re,M.ng_re))[j]);
                    aux_1[ic_re,ig_re] = π_uncond[ic_re,ig_re]*prod(pr_aux[:,ic_re,ig_re]);
                end
                indloglik_uncond[i] = log(sum(aux_1));
            end
            loglik_uncond = sum(indloglik_uncond);
        #
        println("conditional likelihood converged")
        println("conditional log-likelihood = $(-loglik)")
        println("unconditional log-likelihood = $(-loglik_uncond)")
        return M,fc_est,loglik_uncond;
    end
#

# Initialize estimation
Κinit = [7.212483451,5.357563894,3.745502072,3.313797851,4.793929774,1.907971774];
σinit = 0.5;
Minit = M;
loglik_dist = 1000;
outer_init = 1
# Start estimation
loglik_uncond_init = 10000;
while loglik_dist > p.dist_tol
    if outer_init == 1
        outer_iter = 1
        println("outer iter = $outer_init")
        global outer_init = 0;
    else
        println("outer iter = $outer_iter")
    end
    println("  ")
    println("--------------------------------------------------")
    if outer_iter == 1
        Mnew,fc_est_new,loglik_uncond_new = OuterEstimation_eval_zfc(Minit,Data,grid_indices,Κinit,σinit);
    else
        Mnew,fc_est_new,loglik_uncond_new = OuterEstimation_eval_zfc(Mold,Data,grid_indices,Κold,σold);
    end
    println("--------------------------------------------------")
    println("  ")
    # Save parameters to csv file (fixed costs, distribution of comparative advantages, and log likelihood)
    CSV.write("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Results/main/FC_est_$outer_iter.csv",Tables.table(fc_est_new),header=false);
    CSV.write("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Results/main/loglik_uncond_$outer_iter.csv",Tables.table([loglik_uncond_new]),header=false);
    CSV.write("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Results/main/reDist_uncond_$outer_iter.csv", Tables.table(Mnew.π_uncond),header=false);
    Mold = Mnew;
    Κold = copy(fc_est_new[1:6]);
    σold = copy(fc_est_new[7]);
    if outer_iter == 1
        dist_tol = sqrt(norm(loglik_uncond_new-loglik_uncond_init));
    else
        dist_tol = sqrt(norm(loglik_uncond_new-loglik_uncond_old));
    end
    println("--------------------------------------------------------------------")
    println("outer iter = $outer_iter, current likelihood distance = $dist_tol")
    println("--------------------------------------------------------------------")
    loglik_uncond_old = copy(loglik_uncond_new);
    outer_iter += 1
end
CSV.write("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Results/main/lnc_re_grid.csv",Tables.table(Mnew.lnc_re_grid),header=false);
CSV.write("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Results/main/lng_re_grid.csv",Tables.table(Mnew.lng_re_grid),header=false);
