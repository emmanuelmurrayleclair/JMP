# Policy counterfactual of dynamic discrete choice model with real data
    # Steel manufacturing
    # With selection on unobservables
    # Monopolistic Competition
    # Carbon tax with fixed cost subsidy
# Emmanuel Murray Leclair
# June 2023

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
using CSV, DataFrames, DiscreteMarkovChains, StructTypes, StatsBase, Distributed, SharedArrays, DelimitedFiles, NLsolve, ParallelDataTransfer,CUDA_Runtime_jll, CUDA
using FiniteDiff, BenchmarkTools, Distances, JLD2, FileIO

# Load externally written functions
include("/project/6001227/emurrayl/DynamicDiscreteChoice/VFI_Toolbox.jl")

println(" ")
println("-----------------------------------------------------------------------------------------")
println("Dynamic production and fuel set choice in Julia - Steel Manufacturing Data With Selection")
println("            Counterfactuals - Model comparison (with Inter-temporal Switching)           ")
println("-----------------------------------------------------------------------------------------")
println(" ")

#-----------------------------------------------------------
#-----------------------------------------------------------
# 1. Parameters, Model and Data Structure

# Generate structure for parameters using Parameters module
# Import Parameters from Steel manufacturing data (with and without fuel productivity)
    @with_kw struct Par
        p_all = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_all.txt", DataFrame)[:,2:(end-1)]          ;
        p_cov = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_cov.txt", DataFrame)[:,2:(end-1)]          ;
        p_cov_g = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_cov_g.txt", DataFrame)[:,2:(end-1)]      ;
        p_cov_c = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_cov_c.txt", DataFrame)[:,2:(end-1)]      ;
        p_cov_gc = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_cov_gc.txt", DataFrame)[:,2:(end-1)]    ;
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
        # Dynamic discrete choice parameter (fixed costs and distribution of comparative advantages)
            fc_est = Array(CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Results/main/FC_est.csv", DataFrame,header=false));
            Κ = fc_est[1:8];
            σunit = fc_est[9];
            π_uncond = Array(CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Results/main/reDist_uncond.csv", DataFrame,header=false));
            ng_re = 3;
            nc_re = 3;
            π_uncond_marginal_c = [sum(π_uncond[1,:]),sum(π_uncond[2,:]),sum(π_uncond[3,:])];
            π_uncond_marginal_g = [sum(π_uncond[:,1]),sum(π_uncond[:,2]),sum(π_uncond[:,3])];
            lnc_re_grid = Array(CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Results/main/lnc_re_grid.csv", DataFrame,header=false));
            lng_re_grid = Array(CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Results/main/lng_re_grid.csv", DataFrame,header=false));
            g_re_grid = exp.(lng_re_grid);  
            c_re_grid = exp.(lnc_re_grid);
            μg_re = sum(π_uncond_marginal_g.*lng_re_grid);
            μc_re = sum(π_uncond_marginal_c.*lnc_re_grid);
            σg_re = sum(((lng_re_grid.-μg_re).^2).*π_uncond_marginal_g);
            σc_re = sum(((lnc_re_grid.-μc_re).^2).*π_uncond_marginal_c);
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
            γe = γ[1];  # 1 mmBtu of electricity to metric ton of co2e 
            γg = γ[2];  # 1 mmBtu of gas to metric ton of co2e   
            γc = γ[3];  # 1 mmBtu of coal to metric ton of co2e 
            γo = γ[4];  # 1 mmBtu of oil to metric ton of co2e 
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
        # Social cost of carbon (metric ton in million USD)
            unit = 1000000;               # Unit of measurement (1mil rupee)
            SCC_fowlie = 51;              # SCC in USD/metric ton (Fowlie et al. 2016) 
            SCC_nature = 185;             # SCC in USD/metric ton (Nature 2022)
            SCC_india = 5.74;             # SCC in USD/metric ton (Tol 2019)
            # To get SCC in million dollars, divide on of the three SCC by unit
        #
        # Convergence parameters 
        max_iter::Int64     = 20000  ; # Maximum number of iterations
        dist_tol::Float64   = 1E-4   ; # Tolerance for distance
        dist_tol_Δ::Float64 = 1E-11  ; # Tolerance for change in distance 
        pout_tol::Float64   = 1E-10  ; # Tolerance for fixed point in aggregate output price index
        υ                   = 0.0    ; # Dampen factor
        # misc 
        F_tot               = 4      ; # Total number of fuels
        β                   = 0.9    ; # Discount factor
        T_pers              = size(μψe_t,1);    # number of years, persistent variables
        T_nonpers           = size(po,1);       # Number of years, non persistent variables
        T                   = T_pers-1;         # Actual number of years where I kept data
        Tf                  = 40      ;         # Number of years of forward simulation (baseline: 40)
        S                   = 50      ;         # Number of simulations for CCP (baseline: 50)
    end
    p = Par();
#

# Import the data
    Data = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/MainData_wPipeline-Steel.csv", DataFrame) ;
    Data[!,:id] = 1:size(Data,1);
    Data_year = groupby(Data,:year);
#

# Generate structure of model objects
    # Model : Dynamic production and fuel set choice - With Selection and Fuel Productivity
    @with_kw struct Model
        # Parameters
        p::Par = Par(); # Model parameters in their own structure
        ng_re = 3;
        nc_re = 3;
        N = size(Data,1); # Total number of observations
        # Multivariate markov chain for all persistent state variables
        nstate = 5;     # Number of states in multivariate markov chain
        ngrid = 4;      # Number of grid points per 
        Πs = nothing;            # Transition matrix
        lnSgrid = nothing; # grid in logs
        Sgrid = nothing;          # grid in levels  
        # Conditional probability of comparative advantages
        π_cond = nothing;      
        π_uncond = nothing;
        # Price/productivity of gas non-persistent process
        n_g    = 4                                              ;  # Size of grid
        Π_g = nothing                                           ;  # Initialize transition matrix
        lng_grid = nothing                                      ;  # Initialize grid in logs
        g_grid = nothing                                        ;  # Initialize grid in levels
        lnpg_grid = nothing;
        lnψg_grid = nothing;
        pg_grid = nothing;
        ψg_grid = nothing;
        # Price/productivity of coal non-persistent residual process
        n_c    = 4                                              ;  # Size of grid
        Π_c = nothing                                           ;  # Initialize transition matrix
        lnc_grid = nothing                                      ;  # Initialize grid in logs
        c_grid = nothing                                        ;  # Initialize grid in levels       
        lnpc_grid = nothing;
        lnψc_grid = nothing;
        pc_grid = nothing;
        ψc_grid = nothing;
        # All forward simulation draws
        FcombineF_draw = nothing                                ;  # Type 1 extreme value draw for each fuel set
            FcombineF_fs = Array{Int64}(undef,N,p.S,p.Tf+1)     ;  
        #state_fs = Array{Int64}(undef,nstate,N,p.S,p.Tf)        ;  # Simulation draws: multivariate states (z,pm,pe,ψe,ψo)
        z_fs = Array{Float64}(undef,N,p.S,p.Tf)                 ;
            shock_z = Array{Float64}(undef,N,p.S,p.Tf)          ;  # Simulation draw: shock to z
        pm_fs = Array{Float64}(undef,N,p.S,p.Tf)                ;
            shock_pm = Array{Float64}(undef,N,p.S,p.Tf)         ;  # Simulation draw: shock to pm
        pe_fs = Array{Float64}(undef,N,p.S,p.Tf)                ;
            shock_pe = Array{Float64}(undef,N,p.S,p.Tf)         ;  # Simulation draw: shock to pe
        ψe_fs = Array{Float64}(undef,N,p.S,p.Tf)                ;
            shock_ψe = Array{Float64}(undef,N,p.S,p.Tf)         ;  # Simulation draw: shock to ψe
        ψo_fs = Array{Float64}(undef,N,p.S,p.Tf)                ;
            shock_ψo = Array{Float64}(undef,N,p.S,p.Tf)         ;  # Simulation draw: shock to ψo
        pg_fs = Array{Float64}(undef,N,p.S,p.Tf)                ;  # Simulation draw: price of gas
            shock_pg = Array{Float64}(undef,N,p.S,p.Tf)         ;  # Simulation draw: shock to pg
        lnψg_fs = Array{Float64}(undef,N,p.S,p.Tf)              ;  # Simulation draw: productivity of gas
            shock_lnψg = Array{Float64}(undef,N,p.S,p.Tf)       ;  # Simulation draw: shock to lnψg
        pc_fs = Array{Float64}(undef,N,p.S,p.Tf)                ;  # Simulation draw: price of coal
            shock_pc = Array{Float64}(undef,N,p.S,p.Tf)         ;  # Simulation draw: shock to pc
        lnψc_fs = Array{Float64}(undef,N,p.S,p.Tf)              ;  # Simulation draw: productivity of coal
            shock_lnψc = Array{Float64}(undef,N,p.S,p.Tf)       ;  # Simulation draw: shock to lnψc
        gre_draw = nothing                                      ;  # Fixed uniform draw for gas comparative advantage
            gre_fs = Array{Int64}(undef,N,p.S)                  ;
        cre_draw = nothing                                      ;  # Fixed uniform draw for coal comparative advantage
            cre_fs = Array{Int64}(undef,N,p.S)                  ;
        # Current Period profit               
        πgrid_oe = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                      ;
        πgrid_oge = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                      ;
        πgrid_oce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                      ;
        πgrid_ogce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                      ;
        # # Fuel quantities
        # fqty_oe = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                 ;
        #     fqty_oe_norm = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)        ;
        # fqty_oge = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                ;
        #     fqty_oge_norm = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)       ;
        # fqty_oce = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                ;
        #     fqty_oce_norm = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)       ;
        # fqty_ogce = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)               ;
        #     fqty_ogce_norm = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)      ;

        # Expected Value Functions (single agent)
        V_itermax = 1000                                                                            ;
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
        # # Fuel prices
        # pEgrid_oe = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                   ;
        # pEgrid_oge = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                  ;
        # pEgrid_oce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                  ;
        # pEgrid_ogce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                 ;
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

# Simulate state transition markov chain (used for main model only)
    M = StateTransition_func_multivariate(p,M);
#

# Find closest point on the grid for each state variable (used for main model only)
    grid_indices,dist_grid_smallest = MatchGridData(M,Data);
#

# Get distribution for price and productivity of coal/gas serparately
    M = DiscretizeGasCoal_distr(p,M);
# 

#-----------------------------------------------------------
#-----------------------------------------------------------
# 2. Simulation functions (output price, profits and forward simulated draws)

### Find equilibrium aggregate output price
    τnotax = [0,0,0,0];
    p,M = OutputPrice_func_nogrid(M,Data,p,τnotax);
#
# Compute Static Profits under all combination of state variables given tax rates
    @time M = StaticProfit_grid(M,τnotax)   ;
    test=1;
#
# Compute predicted profit under state variables in the data 
    πpred = StaticProfit_data(M,Data,τnotax);
#

### Function that compute and store expected value functions
    function W_max(M::Model,Κ,σ,l::VF_bench)
        ## Given a guess of the fixed cost parameters, this function returns the expected value functions
        # Initialized expected value functions
        W_new = VFI_discrete_evenfaster_zfc(M,Κ,σ,l) ;  
        # Update Model
        M = Model(M; W_new=copy(convert(Array,W_new))) ;
        return M;
    end
    loglik_eval=VF_bench(nothing);
    @time M = W_max(M,p.Κ,p.σunit,loglik_eval);
    test=1;
#

### Update posterior conditional and unconditional probabilities of comparative advantages
    M,π_cond,Prnext_long = PosteriorRE_func_zfc_cf(M,Data,grid_indices,p.Κ);
#

#--------------------------------------------------------------
#--------------------------------------------------------------
# 3. Forward Simulation Draws

# Get simulation draws
    seed = 13421;
    # seed = 14123;
    @time M,state_resdraw1 = ForwardSimul_draws(p,M,Data,seed);
    test=1;
#

#--------------------------------------------------------------
#--------------------------------------------------------------
# 4. Compute counterfactuals

# Test each method without any tax (see if I can recover levels)
    τnotax = [0,0,0,0];
    # 1. Full model
        function Welfare_ForwardSimul(M::Model,Data::DataFrame,p::Par,τ)
            @unpack N,z_fs,pm_fs,pe_fs,ψe_fs,ψo_fs,pg_fs,lnψg_fs,pc_fs,lnψc_fs,gre_fs,cre_fs = M;
            @unpack S,Tf,T,γg,γc,γo,γe,β,Κ,σunit,θ=p;
            τg = τ[1];
            τc = τ[2];
            τo = τ[3];
            τe = τ[4];
            # Fixed costs
            κg_nopipe   = Κ[1];
            κc          = Κ[2];
            γg_nopipe   = Κ[3];
            γc          = Κ[4];
            κg_pipe     = Κ[5];
            γg_pipe     = Κ[6];
            κz          = Κ[7];
            γz          = Κ[8];
            # Fixed cost matrix (oe,oge,oce,ogce)
            Κmat_nopipe = [0 -κg_nopipe -κc -κg_nopipe-κc;
                        γg_nopipe 0 γg_nopipe-κc -κc;
                        γc γc-κg_nopipe 0 -κg_nopipe;
                        γg_nopipe+γc γg_nopipe γc 0];
            Κmat_pipe = [0 -κg_pipe -κc -κg_pipe-κc;
                        γg_pipe 0 γg_pipe-κc -κc;
                        γc γc-κg_pipe 0 -κg_pipe;
                        γg_pipe+γc γg_pipe γc 0];
            Κmat_z = [0 -κz -κz -κz;
                     γz 0 -κz + γz -κz;
                     γz -κz + γz 0 -κz;
                     γz γz γz 0];
            # Start static of social welfare
            pout_agg = Array{Float64}(undef,T,S,Tf+1);
            gas = Array{Float64}(undef,N,S,Tf+1);
            coal = Array{Float64}(undef,N,S,Tf+1);
            oil = Array{Float64}(undef,N,S,Tf+1);
            elec = Array{Float64}(undef,N,S,Tf+1);
            y = Array{Float64}(undef,N,S,Tf+1);
            E = Array{Float64}(undef,N,S,Tf+1);
                δ = 0.0;
                r = 0.150838;
            profit = Array{Float64}(undef,N,S,Tf+1);
            CS = Array{Float64}(undef,T,S,Tf+1);
            FC = Array{Float64}(undef,N,S,Tf+1);
            pE = Array{Float64}(undef,N,S,Tf+1);
            println("      ")
            println("--------------------------------------------------------------")
            println("--------      BEGINNING FORWARD SIMULATION     ---------------")
            println("---------     Main model - With Switching - With fuel prod ---")
            println("---------              Tax (g,c,o,e) = $τ           ----------")
            println("--------------------------------------------------------------")
            println("      ")
            # Update aggregate output price
            p,M = OutputPrice_func_nogrid(M,Data,p,τ);
            # Update static profits for all grid points
            M = StaticProfit_grid(M,τ);
            # Find value function
            loglik_eval=VF_bench(nothing);
            M = W_max(M,Κ,σunit,loglik_eval);
            # Get a grid of value functions for different values of aggregate output price

            ##############################################################################

            for s = 1:S
                println("      ")
                println("--------------------------------------------------------------")
                println("-------------      Simulation number = $s    -----------------")
                println("      ")
                ### INITIAL YEAR ###
                # Get aggregate output price index
                pout_agg[:,s,1] = p.pout_struc[2:7];
                # Update comparative advantages
                Data_s = copy(Data);
                grid_indices_init = copy(grid_indices);
                for i = 1:N
                    if Data_s.gas[i] == 0
                        grid_indices_init.g_re[i] = gre_fs[i,s];
                    end
                    if Data_s.coal[i] == 0
                        grid_indices_init.c_re[i] = cre_fs[i,s];
                    end
                end
                # Get aggregate output price index and consumer surplus
                p,M = OutputPrice_func_nogrid(M,Data_s,p,τ);
                pout_agg[:,s,1] = p.pout_struc[2:7];
                CS[:,s,1] = ((1-θ)/θ)*(pout_agg[:,s,1].^(-θ/(1-θ)));
                ###
                # UPDATE VALUE FUNCTION OR INTERPOLATE FROM GRID OF VALUE FUNCTIONS HERE (LATER)
                ###
                # Get choice-specific value functions for starting period
                vchoicef = choiceVF_func_zfc(M,Data_s,grid_indices_init,Κ);      # Choice-specific value function
                FcombineF = Array{Int64}(undef,N);                                      # Fuel chosen for next period
                fc_shock = Array{Float64}(undef,N);  
                combineF = copy(Data_s.combineF);
                combineF_num = copy(Data_s.combineF);
                combineF_num[combineF.==12] .= 1;
                combineF_num[combineF.==124] .= 2;
                combineF_num[combineF.==123] .= 3;
                combineF_num[combineF.==1234] .= 4;
                # Draw choice for next period
                voe = vchoicef[:,1] .+ M.FcombineF_draw[1,:,s,1];
                voge = vchoicef[:,2] .+ M.FcombineF_draw[2,:,s,1];
                voce = vchoicef[:,3] .+ M.FcombineF_draw[3,:,s,1];
                vogce = vchoicef[:,4] .+ M.FcombineF_draw[4,:,s,1];
                vtilde = maximum([voe voge voce vogce],dims=2);
                FcombineF[vtilde[:].==voe] .= 1;
                fc_shock[vtilde[:].==voe] .=  M.FcombineF_draw[1,vtilde[:].==voe,s,1];
                FcombineF[vtilde[:].==voge] .= 2;
                fc_shock[vtilde[:].==voge] .=  M.FcombineF_draw[2,vtilde[:].==voge,s,1];
                FcombineF[vtilde[:].==voce] .= 3;
                fc_shock[vtilde[:].==voce] .=  M.FcombineF_draw[3,vtilde[:].==voce,s,1];
                FcombineF[vtilde[:].==vogce] .= 4;
                fc_shock[vtilde[:].==vogce] .=  M.FcombineF_draw[4,vtilde[:].==vogce,s,1];
                for i = 1:N
                    if Data_s.Connection[i] == "3"
                        FC[i,s,1] = σunit*(fc_shock[i] + Κmat_nopipe[combineF_num[i],FcombineF[i]] + Κmat_z[combineF_num[i],FcombineF[i]]*Data_s.lnz[i]);
                    elseif Data_s.Connection[i] == "direct" || Data.Connection[i] == "indirect"
                        FC[i,s,1] = σunit*(fc_shock[i] + Κmat_pipe[combineF_num[i],FcombineF[i]] + Κmat_z[combineF_num[i],FcombineF[i]]*Data_s.lnz[i]);
                    end
                end
                # Year of data: Get firm-level objects: output, fuel prices and productivity, fuel quantities
                for i = 1:N
                    t = Data.year[i]-2009;
                    pm = exp(Data.logPm[i]);
                    z = exp(Data.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean;
                    pe = exp(Data.lnpelec_tilde[i]) + τe*p.egmean;
                    # fuel productivity
                    ψo = exp(Data.lnfprod_o[i]);
                    ψe = exp(Data.lnfprod_e[i]);
                    ψg = exp(Data.lnfprod_g[i]);
                    ψc = exp(Data.lnfprod_c[i]);
                    # input prices indices
                    poψo = po/ψo;
                    peψe = pe/ψe;
                    if Data.combineF[i] == 12
                        pfψf = [poψo,peψe];
                        pE[i,s,1] = pE_func(12,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE[i,s,1],p);
                    elseif Data.combineF[i] == 123
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc/ψc;
                        pfψf = [poψo,peψe,pcψc];
                        pE[i,s,1] = pE_func(123,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE[i,s,1],p);
                    elseif Data.combineF[i] == 124
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg/ψg;
                        pfψf = [poψo,peψe,pgψg];
                        pE[i,s,1] = pE_func(124,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE[i,s,1],p);
                    elseif Data.combineF[i] == 1234
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc/ψc;
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg/ψg;
                        pfψf = [poψo,peψe,pcψc,pgψg];
                        pE[i,s,1] = pE_func(1234,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE[i,s,1],p);
                    end
                    # Output
                    y[i,s,1] = output_func_monopolistic(z,pin,p.pout_struc[t+1],t,p);
                    # profit
                    profit[i,s,1] = profit_func_monopolistic(z,pin,t,p);
                    # Energy
                    E[i,s,1] = ((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE[i,s,1])^p.σ);
                    # Fuel quantitiy
                    oil[i,s,1] = (p.ogmean*E[i,s,1])*((p.βo/po)^p.λ)*(ψo^(p.λ-1))*(pE[i,s,1]^p.λ);
                    elec[i,s,1] = (p.egmean*E[i,s,1])*((p.βe/pe)^p.λ)*(ψe^(p.λ-1))*(pE[i,s,1]^p.λ);
                    if Data.combineF[i] == 12
                        gas[i,s,1] = 0;
                        coal[i,s,1] = 0;
                    elseif Data.combineF[i] == 123
                        gas[i,s,1] = 0;
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc)^p.λ)*(ψc^(p.λ-1))*(pE[i,s,1]^p.λ);
                    elseif Data.combineF[i] == 124
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg)^p.λ)*(ψg^(p.λ-1))*(pE[i,s,1]^p.λ);
                        coal[i,s,1] = 0;
                    elseif Data.combineF[i] == 1234
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg)^p.λ)*(ψg^(p.λ-1))*(pE[i,s,1]^p.λ);
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc)^p.λ)*(ψc^(p.λ-1))*(pE[i,s,1]^p.λ);
                    end
                end
                ### FORWARD SIMULATION ###
                for tf = 1:Tf
                    # Update current fuel set
                    Data_fs = copy(Data);
                    combineF_num = copy(FcombineF);
                    Data_fs.combineF = copy(FcombineF);
                    Data_fs.combineF[Data_fs.combineF.==1] .= 12;
                    Data_fs.combineF[Data_fs.combineF.==2] .= 124;
                    Data_fs.combineF[Data_fs.combineF.==3] .= 123;
                    Data_fs.combineF[Data_fs.combineF.==4] .= 1234;
                    # Get new aggregate price index
                    for i = 1:N
                        t = Data_fs.year[i]-2009;
                        Data_fs.lnz[i] = log(z_fs[i,s,tf]);
                        Data_fs.logPm[i] = log(pm_fs[i,s,tf]);
                        # fuel prices (multiplied by geometric mean of fuel)
                        po = p.po[t+1]; 
                        Data_fs.lnpelec_tilde[i] = log(pe_fs[i,s,tf]);
                        # fuel productivity
                        ψo = ψo_fs[i,s,tf]; Data_fs.lnfprod_o[i] = log(ψo);
                        ψe = ψe_fs[i,s,tf]; Data_fs.lnfprod_e[i] = log(ψe);
                        # input prices indices
                        if Data_fs.combineF[i] == 12
                            Data_fs.lnfprod_c[i] = missing; Data_fs.lnpc_tilde[i] = missing;
                            Data_fs.lnfprod_g[i] = missing; Data_fs.lnpg_tilde[i] = missing;
                        elseif Data_fs.combineF[i] == 123
                            Data_fs.lnfprod_c[i] = lnψc_fs[i,s,tf] + p.lnc_re_grid[grid_indices_init.c_re[i]];
                            Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                            Data_fs.lnfprod_g[i] = missing; Data_fs.lnpg_tilde[i] = missing;
                            Data_fs.res_lnpg_prodg[i] = missing
                        elseif Data_fs.combineF[i] == 124
                            Data_fs.lnfprod_g[i] = lnψg_fs[i,s,tf] + p.lng_re_grid[grid_indices_init.g_re[i]];
                            Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                            Data_fs.lnfprod_c[i] = missing; Data_fs.lnpc_tilde[i] = missing;
                            Data_fs.res_lnpc_prodc[i] = missing
                        elseif Data_fs.combineF[i] == 1234
                            Data_fs.lnfprod_c[i] = lnψc_fs[i,s,tf] + p.lnc_re_grid[grid_indices_init.c_re[i]];
                            Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                            Data_fs.lnfprod_g[i] = lnψg_fs[i,s,tf] + p.lng_re_grid[grid_indices_init.g_re[i]];
                            Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                        end
                    end
                    p_fs,M = OutputPrice_func_nogrid(M,Data_fs,p,τ);
                    pout_agg[:,s,tf+1] = p_fs.pout_struc[2:7];
                    CS[:,s,tf+1] = ((1-θ)/θ)*(pout_agg[:,s,tf+1].^(-θ/(1-θ)));
                    ###
                    # UPDATE VALUE FUNCTION OR INTERPOLATE FROM GRID OF VALUE FUNCTIONS HERE (LATER)
                    ###
                    # Get closest point to grid
                    grid_indices_fs,dist_grid_smallest = MatchGridData_fs(M,Data_fs);
                    grid_indices_fs.g_re .= grid_indices_init.g_re;
                    grid_indices_fs.c_re .= grid_indices_init.c_re;
                    # Draw new fuel set for next period 
                    vchoicef = choiceVF_func_zfc(M,Data_fs,grid_indices_fs,Κ);     # Choice-specific value function
                    FcombineF = Array{Int64}(undef,N);                              # Fuel chosen for next period
                    fc_shock = Array{Float64}(undef,N);                             # Shock to fixed costs
                    voe = vchoicef[:,1] .+ M.FcombineF_draw[1,:,s,tf+1];
                    voge = vchoicef[:,2] .+ M.FcombineF_draw[2,:,s,tf+1];
                    voce = vchoicef[:,3] .+ M.FcombineF_draw[3,:,s,tf+1];
                    vogce = vchoicef[:,4] .+ M.FcombineF_draw[4,:,s,tf+1];
                    vtilde = maximum([voe voge voce vogce],dims=2);
                    FcombineF[vtilde[:].==voe] .= 1;
                    fc_shock[vtilde[:].==voe] .=  M.FcombineF_draw[1,vtilde[:].==voe,s,tf+1];
                    FcombineF[vtilde[:].==voge] .= 2;
                    fc_shock[vtilde[:].==voge] .=  M.FcombineF_draw[2,vtilde[:].==voge,s,tf+1];
                    FcombineF[vtilde[:].==voce] .= 3;
                    fc_shock[vtilde[:].==voce] .=  M.FcombineF_draw[3,vtilde[:].==voce,s,tf+1];
                    FcombineF[vtilde[:].==vogce] .= 4;
                    fc_shock[vtilde[:].==vogce] .=  M.FcombineF_draw[4,vtilde[:].==vogce,s,tf+1];
                    for i = 1:N
                        if Data_s.Connection[i] == "3"
                            FC[i,s,tf+1] = σunit*(fc_shock[i] + Κmat_nopipe[combineF_num[i],FcombineF[i]] + Κmat_z[combineF_num[i],FcombineF[i]]*Data_fs.lnz[i]);
                        elseif Data_s.Connection[i] == "direct" || Data.Connection[i] == "indirect"
                            FC[i,s,tf+1] = σunit*(fc_shock[i] + Κmat_pipe[combineF_num[i],FcombineF[i]] + Κmat_z[combineF_num[i],FcombineF[i]]*Data_fs.lnz[i]);
                        end
                    end
                    # Get firm-level objects: output, fuel prices and productivity, fuel quantities
                    for i = 1:N
                        t = Data_fs.year[i]-2009; 
                        pm = exp(Data_fs.logPm[i]);
                        z = exp(Data_fs.lnz[i]);
                        # fuel prices (multiplied by geometric mean of fuel)
                        po = p.po[t+1] + τo*p.ogmean;
                        pe = exp(Data_fs.lnpelec_tilde[i]) + τe*p.egmean; 
                        pc = exp(Data_fs.lnpc_tilde[i]) + τc*p.cgmean;
                        pg = exp(Data_fs.lnpg_tilde[i]) + τg*p.ggmean;
                        # fuel productivity
                        ψo = ψo_fs[i,s,tf];
                        ψe = ψe_fs[i,s,tf];
                        # input prices indices
                        poψo = po/ψo;
                        peψe = pe/ψe;
                        if Data_fs.combineF[i] == 12
                            pfψf = [poψo,peψe];
                            pE[i,s,tf+1] = pE_func(12,pfψf,p);
                            pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE[i,s,tf+1],p);
                        elseif Data_fs.combineF[i] == 123
                            pcψc = pc/exp(Data_fs.lnfprod_c[i]);
                            pfψf = [poψo,peψe,pcψc];
                            pE[i,s,tf+1] = pE_func(123,pfψf,p);
                            pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE[i,s,tf+1],p);
                        elseif Data_fs.combineF[i] == 124
                            pgψg = pg/exp(Data_fs.lnfprod_g[i]);
                            pfψf = [poψo,peψe,pgψg];
                            pE[i,s,tf+1] = pE_func(124,pfψf,p);
                            pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE[i,s,tf+1],p);
                        elseif Data_fs.combineF[i] == 1234
                            pcψc = pc/exp(Data_fs.lnfprod_c[i]);
                            pgψg = pg/exp(Data_fs.lnfprod_g[i]);
                            pfψf = [poψo,peψe,pcψc,pgψg];
                            pE[i,s,tf+1] = pE_func(1234,pfψf,p);
                            pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE[i,s,tf+1],p);
                        end
                        # Output
                        y[i,s,tf+1] = output_func_monopolistic(z,pin,p_fs.pout_struc[t+1],t,p_fs);
                        # profit
                        profit[i,s,tf+1] = profit_func_monopolistic(z,pin,t,p_fs);
                        # Energy
                        E[i,s,tf+1] = ((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE[i,s,tf+1])^p.σ);
                        # Fuel quantitiy
                        oil[i,s,tf+1] = (p.ogmean*E[i,s,tf+1])*((p.βo/po)^p.λ)*(ψo^(p.λ-1))*(pE[i,s,tf+1]^p.λ);
                        elec[i,s,tf+1] = (p.egmean*E[i,s,tf+1])*((p.βe/pe)^p.λ)*(ψe^(p.λ-1))*(pE[i,s,tf+1]^p.λ);
                        if Data_fs.combineF[i] == 12
                            gas[i,s,tf+1] = 0;
                            coal[i,s,tf+1] = 0;
                        elseif Data_fs.combineF[i] == 123
                            gas[i,s,tf+1] = 0;
                            coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc)^p.λ)*(exp(Data_fs.lnfprod_c[i])^(p.λ-1))*(pE[i,s,tf+1]^p.λ);
                        elseif Data_fs.combineF[i] == 124
                            gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg)^p.λ)*(exp(Data_fs.lnfprod_g[i])^(p.λ-1))*(pE[i,s,tf+1]^p.λ);
                            coal[i,s,tf+1] = 0;
                        elseif Data_fs.combineF[i] == 1234
                            gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg)^p.λ)*(exp(Data_fs.lnfprod_g[i])^(p.λ-1))*(pE[i,s,tf+1]^p.λ);
                            coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc)^p.λ)*(exp(Data_fs.lnfprod_c[i])^(p.λ-1))*(pE[i,s,tf+1]^p.λ);
                        end
                    end
                end
            end
            # Return aggregates by year forward (only look at last year in data - 2016)
            y_fs = zeros(Tf+1); E_fs = zeros(Tf+1); gas_fs = zeros(Tf+1); coal_fs = zeros(Tf+1); oil_fs = zeros(Tf+1); netprofit_fs=zeros(Tf+1);
            elec_fs = zeros(Tf+1); co2_fs = zeros(Tf+1); profit_fs = zeros(Tf+1); CS_fs = zeros(Tf+1); FC_fs = zeros(Tf+1); pE_fs = zeros(Tf+1);
            Data_year = groupby(Data,:year); Nt = size(Data_year[end])[1];
            y = y[(end-Nt+1:end),:,:];
            E = E[(end-Nt+1:end),:,:];
            gas = gas[(end-Nt+1:end),:,:];
            coal = coal[(end-Nt+1:end),:,:];
            oil = oil[(end-Nt+1:end),:,:];
            elec = elec[(end-Nt+1:end),:,:];
            profit = profit[(end-Nt+1:end),:,:];
            FC = FC[(end-Nt+1:end),:,:];
            pE = pE[(end-Nt+1):end,:,:];
            pout_agg = pout_agg[end,:,:];
            CS = CS[end,:,:];
            co2 = p.γg*gas .+ p.γc*coal .+ p.γo*oil .+ p.γe*elec;
            netprofit = profit .+ FC;
            # Aggregate output (from rep consumer CES demand)
            Nfirms = Data_year[end].N[1];
            demand_shock = exp(p.d_t[end]);
            rhoterm = (p.ρ-1)/p.ρ;
            y_s_fs = zeros(p.S,p.Tf+1);
            for s = 1:p.S
                for tf = 1:(p.Tf+1)
                    y_s_fs[s,tf] = ((demand_shock/Nfirms)*(sum(y[:,s,tf].^rhoterm)))^(1/rhoterm);
                end
            end
            # Consumer surplus
            for tf = 1:(p.Tf+1)
                CS_fs[tf] = mean(CS[:,tf]);
            end
            # Average across simulations
            for tf = 1:Tf+1
                pE_fs[tf] = mean(pE[:,:,tf]);
                # pin_fs[tf] = mean(pin[:,:,tf]);
                y_fs[tf] = mean(y_s_fs[:,tf]);
                for i = 1:Nt
                    E_fs[tf] += mean(E[i,:,tf]);
                    gas_fs[tf] += mean(gas[i,:,tf]);
                    coal_fs[tf] += mean(coal[i,:,tf]);
                    oil_fs[tf] += mean(oil[i,:,tf]);
                    elec_fs[tf] += mean(elec[i,:,tf]);
                    co2_fs[tf] += mean(co2[i,:,tf]);
                    profit_fs[tf] += mean(profit[i,:,tf]);
                    FC_fs[tf] += mean(FC[i,:,tf]);
                    netprofit_fs[tf] += mean(netprofit[i,:,tf]);
                end
            end
            return y_fs,E_fs,gas_fs,coal_fs,oil_fs,elec_fs,co2_fs,profit_fs,netprofit_fs,CS_fs,FC_fs,pE_fs,gas,coal,pout_agg;
        end
    #
    # 2. Restricted switching (cannot switch as response to carbon tax)
        function Welfare_ForwardSimul_restricted(M::Model,Data::DataFrame,p::Par,τ)
            @unpack N,z_fs,pm_fs,pe_fs,ψe_fs,ψo_fs,pg_fs,lnψg_fs,pc_fs,lnψc_fs,gre_fs,cre_fs = M;
            @unpack S,Tf,T,γg,γc,γo,γe,β,Κ,σunit,θ=p;
            τg = τ[1];
            τc = τ[2];
            τo = τ[3];
            τe = τ[4];
            # Fixed costs
            κg_nopipe   = Κ[1];
            κc          = Κ[2];
            γg_nopipe   = Κ[3];
            γc          = Κ[4];
            κg_pipe     = Κ[5];
            γg_pipe     = Κ[6];
            κz          = Κ[7];
            γz          = Κ[8];
            # Fixed cost matrix (oe,oge,oce,ogce)
            Κmat_nopipe = [0 -κg_nopipe -κc -κg_nopipe-κc;
                        γg_nopipe 0 γg_nopipe-κc -κc;
                        γc γc-κg_nopipe 0 -κg_nopipe;
                        γg_nopipe+γc γg_nopipe γc 0];
            Κmat_pipe = [0 -κg_pipe -κc -κg_pipe-κc;
                        γg_pipe 0 γg_pipe-κc -κc;
                        γc γc-κg_pipe 0 -κg_pipe;
                        γg_pipe+γc γg_pipe γc 0];
            Κmat_z = [0 -κz -κz -κz;
                     γz 0 -κz + γz -κz;
                     γz -κz + γz 0 -κz;
                     γz γz γz 0];
            # Start static of social welfare
            pout_agg = Array{Float64}(undef,T,S,Tf+1);
            gas = Array{Float64}(undef,N,S,Tf+1);
            coal = Array{Float64}(undef,N,S,Tf+1);
            oil = Array{Float64}(undef,N,S,Tf+1);
            elec = Array{Float64}(undef,N,S,Tf+1);
            # pE = SharedArray{Float64}(N,S,Tf+1);
            y = Array{Float64}(undef,N,S,Tf+1);
            E = Array{Float64}(undef,N,S,Tf+1);
                δ = 0.0;
                r = 0.150838;
            profit = Array{Float64}(undef,N,S,Tf+1);
            CS = Array{Float64}(undef,T,S,Tf+1);
            FC = Array{Float64}(undef,N,S,Tf+1);
            pE = Array{Float64}(undef,N,S,Tf+1);
            println("      ")
            println("--------------------------------------------------------------")
            println("--------      BEGINNING FORWARD SIMULATION     ---------------")
            println("---------     Main model - With Switching - With fuel prod ---")
            println("---------              Tax (g,c,o,e) = $τ           ----------")
            println("--------------------------------------------------------------")
            println("      ")
            ### UPDATE FUEL SET - AS IF NO TAX (VALUE FUNCTION GIVEN EXTERNALLY) ###
            for s = 1:S
                println("      ")
                println("--------------------------------------------------------------")
                println("-------------      Simulation number = $s    -----------------")
                println("      ")
                ### INITIAL YEAR ###
                # Update comparative advantages
                Data_s = copy(Data);
                grid_indices_init = copy(grid_indices);
                for i = 1:N
                    if Data_s.gas[i] == 0
                        grid_indices_init.g_re[i] = gre_fs[i,s];
                    end
                    if Data_s.coal[i] == 0
                        grid_indices_init.c_re[i] = cre_fs[i,s];
                    end
                end
                # Get aggregate output price index and consumer surplus
                p,M = OutputPrice_func_nogrid(M,Data_s,p,τ);
                pout_agg[:,s,1] = p.pout_struc[2:7];
                CS[:,s,1] = ((1-θ)/θ)*(pout_agg[:,s,1].^(-θ/(1-θ)));
                ###
                # UPDATE VALUE FUNCTION OR INTERPOLATE FROM GRID OF VALUE FUNCTIONS HERE (LATER)
                ###
                # Get choice-specific value functions for starting period
                vchoicef = choiceVF_func_zfc(M,Data_s,grid_indices_init,Κ);             # Choice-specific value function
                FcombineF = Array{Int64}(undef,N);                                      # Fuel chosen for next period
                fc_shock = Array{Float64}(undef,N);  
                combineF = copy(Data_s.combineF);
                combineF_num = copy(Data_s.combineF);
                combineF_num[combineF.==12] .= 1;
                combineF_num[combineF.==124] .= 2;
                combineF_num[combineF.==123] .= 3;
                combineF_num[combineF.==1234] .= 4;
                # Draw choice for next period
                voe = vchoicef[:,1] .+ M.FcombineF_draw[1,:,s,1];
                voge = vchoicef[:,2] .+ M.FcombineF_draw[2,:,s,1];
                voce = vchoicef[:,3] .+ M.FcombineF_draw[3,:,s,1];
                vogce = vchoicef[:,4] .+ M.FcombineF_draw[4,:,s,1];
                vtilde = maximum([voe voge voce vogce],dims=2);
                FcombineF[vtilde[:].==voe] .= 1;
                fc_shock[vtilde[:].==voe] .=  M.FcombineF_draw[1,vtilde[:].==voe,s,1];
                FcombineF[vtilde[:].==voge] .= 2;
                fc_shock[vtilde[:].==voge] .=  M.FcombineF_draw[2,vtilde[:].==voge,s,1];
                FcombineF[vtilde[:].==voce] .= 3;
                fc_shock[vtilde[:].==voce] .=  M.FcombineF_draw[3,vtilde[:].==voce,s,1];
                FcombineF[vtilde[:].==vogce] .= 4;
                fc_shock[vtilde[:].==vogce] .=  M.FcombineF_draw[4,vtilde[:].==vogce,s,1];
                for i = 1:N
                    if Data_s.Connection[i] == "3"
                        FC[i,s,1] = σunit*(fc_shock[i] + Κmat_nopipe[combineF_num[i],FcombineF[i]] + Κmat_z[combineF_num[i],FcombineF[i]]*Data_s.lnz[i]);
                    elseif Data_s.Connection[i] == "direct" || Data.Connection[i] == "indirect"
                        FC[i,s,1] = σunit*(fc_shock[i] + Κmat_pipe[combineF_num[i],FcombineF[i]] + Κmat_z[combineF_num[i],FcombineF[i]]*Data_s.lnz[i]);
                    end
                end
                # Year of data: Get firm-level objects: output, fuel prices and productivity, fuel quantities
                for i = 1:N
                    t = Data.year[i]-2009;
                    pm = exp(Data.logPm[i]);
                    z = exp(Data.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean;
                    pe = exp(Data.lnpelec_tilde[i]) + τe*p.egmean;
                    # fuel productivity
                    ψo = exp(Data.lnfprod_o[i]);
                    ψe = exp(Data.lnfprod_e[i]);
                    ψg = exp(Data.lnfprod_g[i]);
                    ψc = exp(Data.lnfprod_c[i]);
                    # input prices indices
                    poψo = po/ψo;
                    peψe = pe/ψe;
                    if Data.combineF[i] == 12
                        pfψf = [poψo,peψe];
                        pE[i,s,1] = pE_func(12,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE[i,s,1],p);
                    elseif Data.combineF[i] == 123
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc/ψc;
                        pfψf = [poψo,peψe,pcψc];
                        pE[i,s,1] = pE_func(123,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE[i,s,1],p);
                    elseif Data.combineF[i] == 124
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg/ψg;
                        pfψf = [poψo,peψe,pgψg];
                        pE[i,s,1] = pE_func(124,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE[i,s,1],p);
                    elseif Data.combineF[i] == 1234
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc/ψc;
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg/ψg;
                        pfψf = [poψo,peψe,pcψc,pgψg];
                        pE[i,s,1] = pE_func(1234,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE[i,s,1],p);
                    end
                    # Output
                    y[i,s,1] = output_func_monopolistic(z,pin,p.pout_struc[t+1],t,p);
                    # profit
                    profit[i,s,1] = profit_func_monopolistic(z,pin,t,p);
                    # Energy
                    E[i,s,1] = ((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE[i,s,1])^p.σ);
                    # Fuel quantitiy
                    oil[i,s,1] = (p.ogmean*E[i,s,1])*((p.βo/po)^p.λ)*(ψo^(p.λ-1))*(pE[i,s,1]^p.λ);
                    elec[i,s,1] = (p.egmean*E[i,s,1])*((p.βe/pe)^p.λ)*(ψe^(p.λ-1))*(pE[i,s,1]^p.λ);
                    if Data.combineF[i] == 12
                        gas[i,s,1] = 0;
                        coal[i,s,1] = 0;
                    elseif Data.combineF[i] == 123
                        gas[i,s,1] = 0;
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc)^p.λ)*(ψc^(p.λ-1))*(pE[i,s,1]^p.λ);
                    elseif Data.combineF[i] == 124
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg)^p.λ)*(ψg^(p.λ-1))*(pE[i,s,1]^p.λ);
                        coal[i,s,1] = 0;
                    elseif Data.combineF[i] == 1234
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg)^p.λ)*(ψg^(p.λ-1))*(pE[i,s,1]^p.λ);
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc)^p.λ)*(ψc^(p.λ-1))*(pE[i,s,1]^p.λ);
                    end
                end
                ### FORWARD SIMULATION ###
                for tf = 1:Tf
                    # println("Simulation number = $s, year fordward = $tf")
                    # Update current fuel set
                    Data_fs = copy(Data);
                    combineF_num = copy(FcombineF);
                    Data_fs.combineF = copy(FcombineF);
                    Data_fs.combineF[Data_fs.combineF.==1] .= 12;
                    Data_fs.combineF[Data_fs.combineF.==2] .= 124;
                    Data_fs.combineF[Data_fs.combineF.==3] .= 123;
                    Data_fs.combineF[Data_fs.combineF.==4] .= 1234;
                    # Get new aggregate price index
                    for i = 1:N
                        t = Data_fs.year[i]-2009;
                        Data_fs.lnz[i] = log(z_fs[i,s,tf]);
                        Data_fs.logPm[i] = log(pm_fs[i,s,tf]);
                        # fuel prices (multiplied by geometric mean of fuel)
                        po = p.po[t+1]; 
                        Data_fs.lnpelec_tilde[i] = log(pe_fs[i,s,tf]);
                        # fuel productivity
                        ψo = ψo_fs[i,s,tf]; Data_fs.lnfprod_o[i] = log(ψo);
                        ψe = ψe_fs[i,s,tf]; Data_fs.lnfprod_e[i] = log(ψe);
                        # input prices indices
                        if Data_fs.combineF[i] == 12
                            Data_fs.lnfprod_c[i] = missing; Data_fs.lnpc_tilde[i] = missing;
                            Data_fs.lnfprod_g[i] = missing; Data_fs.lnpg_tilde[i] = missing;
                        elseif Data_fs.combineF[i] == 123
                            Data_fs.lnfprod_c[i] = lnψc_fs[i,s,tf] + p.lnc_re_grid[grid_indices_init.c_re[i]];
                            Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                            Data_fs.lnfprod_g[i] = missing; Data_fs.lnpg_tilde[i] = missing;
                            Data_fs.res_lnpg_prodg[i] = missing
                        elseif Data_fs.combineF[i] == 124
                            Data_fs.lnfprod_g[i] = lnψg_fs[i,s,tf] + p.lng_re_grid[grid_indices_init.g_re[i]];
                            Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                            Data_fs.lnfprod_c[i] = missing; Data_fs.lnpc_tilde[i] = missing;
                            Data_fs.res_lnpc_prodc[i] = missing
                        elseif Data_fs.combineF[i] == 1234
                            Data_fs.lnfprod_c[i] = lnψc_fs[i,s,tf] + p.lnc_re_grid[grid_indices_init.c_re[i]];
                            Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                            Data_fs.lnfprod_g[i] = lnψg_fs[i,s,tf] + p.lng_re_grid[grid_indices_init.g_re[i]];
                            Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                        end
                    end
                    p_fs,M = OutputPrice_func_nogrid(M,Data_fs,p,τ);
                    pout_agg[:,s,tf+1] = p_fs.pout_struc[2:7];
                    CS[:,s,tf+1] = ((1-θ)/θ)*(pout_agg[:,s,tf+1].^(-θ/(1-θ)));
                    ###
                    # UPDATE FUEL SET
                    ###
                    # Get closest point to grid
                    grid_indices_fs,dist_grid_smallest = MatchGridData_fs(M,Data_fs);
                    grid_indices_fs.g_re .= grid_indices_init.g_re;
                    grid_indices_fs.c_re .= grid_indices_init.c_re;
                    # Draw new fuel set for next period 
                    vchoicef = choiceVF_func_zfc(M,Data_fs,grid_indices_fs,Κ);      # Choice-specific value function
                    FcombineF = Array{Int64}(undef,N);                              # Fuel chosen for next period
                    fc_shock = Array{Float64}(undef,N);                             # Shock to fixed costs
                    voe = vchoicef[:,1] .+ M.FcombineF_draw[1,:,s,tf+1];
                    voge = vchoicef[:,2] .+ M.FcombineF_draw[2,:,s,tf+1];
                    voce = vchoicef[:,3] .+ M.FcombineF_draw[3,:,s,tf+1];
                    vogce = vchoicef[:,4] .+ M.FcombineF_draw[4,:,s,tf+1];
                    vtilde = maximum([voe voge voce vogce],dims=2);
                    FcombineF[vtilde[:].==voe] .= 1;
                    fc_shock[vtilde[:].==voe] .=  M.FcombineF_draw[1,vtilde[:].==voe,s,tf+1];
                    FcombineF[vtilde[:].==voge] .= 2;
                    fc_shock[vtilde[:].==voge] .=  M.FcombineF_draw[2,vtilde[:].==voge,s,tf+1];
                    FcombineF[vtilde[:].==voce] .= 3;
                    fc_shock[vtilde[:].==voce] .=  M.FcombineF_draw[3,vtilde[:].==voce,s,tf+1];
                    FcombineF[vtilde[:].==vogce] .= 4;
                    fc_shock[vtilde[:].==vogce] .=  M.FcombineF_draw[4,vtilde[:].==vogce,s,tf+1];
                    for i = 1:N
                        if Data_s.Connection[i] == "3"
                            FC[i,s,tf+1] = σunit*(fc_shock[i] + Κmat_nopipe[combineF_num[i],FcombineF[i]] + Κmat_z[combineF_num[i],FcombineF[i]]*Data_fs.lnz[i]);
                        elseif Data_s.Connection[i] == "direct" || Data.Connection[i] == "indirect"
                            FC[i,s,tf+1] = σunit*(fc_shock[i] + Κmat_pipe[combineF_num[i],FcombineF[i]] + Κmat_z[combineF_num[i],FcombineF[i]]*Data_fs.lnz[i]);
                        end
                    end
                    # Get firm-level objects: output, fuel prices and productivity, fuel quantities
                    for i = 1:N
                        t = Data_fs.year[i]-2009; 
                        pm = exp(Data_fs.logPm[i]);
                        z = exp(Data_fs.lnz[i]);
                        # fuel prices (multiplied by geometric mean of fuel)
                        po = p.po[t+1] + τo*p.ogmean;
                        pe = exp(Data_fs.lnpelec_tilde[i]) + τe*p.egmean; 
                        pc = exp(Data_fs.lnpc_tilde[i]) + τc*p.cgmean;
                        pg = exp(Data_fs.lnpg_tilde[i]) + τg*p.ggmean;
                        # fuel productivity
                        ψo = ψo_fs[i,s,tf];
                        ψe = ψe_fs[i,s,tf];
                        # input prices indices
                        poψo = po/ψo;
                        peψe = pe/ψe;
                        if Data_fs.combineF[i] == 12
                            pfψf = [poψo,peψe];
                            pE[i,s,tf+1] = pE_func(12,pfψf,p);
                            pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE[i,s,tf+1],p);
                        elseif Data_fs.combineF[i] == 123
                            pcψc = pc/exp(Data_fs.lnfprod_c[i]);
                            pfψf = [poψo,peψe,pcψc];
                            pE[i,s,tf+1] = pE_func(123,pfψf,p);
                            pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE[i,s,tf+1],p);
                        elseif Data_fs.combineF[i] == 124
                            pgψg = pg/exp(Data_fs.lnfprod_g[i]);
                            pfψf = [poψo,peψe,pgψg];
                            pE[i,s,tf+1] = pE_func(124,pfψf,p);
                            pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE[i,s,tf+1],p);
                        elseif Data_fs.combineF[i] == 1234
                            pcψc = pc/exp(Data_fs.lnfprod_c[i]);
                            pgψg = pg/exp(Data_fs.lnfprod_g[i]);
                            pfψf = [poψo,peψe,pcψc,pgψg];
                            pE[i,s,tf+1] = pE_func(1234,pfψf,p);
                            pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE[i,s,tf+1],p);
                        end
                        # Output
                        y[i,s,tf+1] = output_func_monopolistic(z,pin,p_fs.pout_struc[t+1],t,p_fs);
                        # profit
                        profit[i,s,tf+1] = profit_func_monopolistic(z,pin,t,p_fs);
                        # Energy
                        E[i,s,tf+1] = ((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE[i,s,tf+1])^p.σ);
                        # Fuel quantitiy
                        oil[i,s,tf+1] = (p.ogmean*E[i,s,tf+1])*((p.βo/po)^p.λ)*(ψo^(p.λ-1))*(pE[i,s,tf+1]^p.λ);
                        elec[i,s,tf+1] = (p.egmean*E[i,s,tf+1])*((p.βe/pe)^p.λ)*(ψe^(p.λ-1))*(pE[i,s,tf+1]^p.λ);
                        if Data_fs.combineF[i] == 12
                            gas[i,s,tf+1] = 0;
                            coal[i,s,tf+1] = 0;
                        elseif Data_fs.combineF[i] == 123
                            gas[i,s,tf+1] = 0;
                            coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc)^p.λ)*(exp(Data_fs.lnfprod_c[i])^(p.λ-1))*(pE[i,s,tf+1]^p.λ);
                        elseif Data_fs.combineF[i] == 124
                            gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg)^p.λ)*(exp(Data_fs.lnfprod_g[i])^(p.λ-1))*(pE[i,s,tf+1]^p.λ);
                            coal[i,s,tf+1] = 0;
                        elseif Data_fs.combineF[i] == 1234
                            gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg)^p.λ)*(exp(Data_fs.lnfprod_g[i])^(p.λ-1))*(pE[i,s,tf+1]^p.λ);
                            coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc)^p.λ)*(exp(Data_fs.lnfprod_c[i])^(p.λ-1))*(pE[i,s,tf+1]^p.λ);
                        end
                    end
                end
            end
            # Return aggregates by year forward (only look at last year in data - 2016)
            y_fs = zeros(Tf+1); E_fs = zeros(Tf+1); gas_fs = zeros(Tf+1); coal_fs = zeros(Tf+1); oil_fs = zeros(Tf+1); netprofit_fs=zeros(Tf+1);
            elec_fs = zeros(Tf+1); co2_fs = zeros(Tf+1); profit_fs = zeros(Tf+1); CS_fs = zeros(Tf+1); FC_fs = zeros(Tf+1); pE_fs = zeros(Tf+1);
            Data_year = groupby(Data,:year); Nt = size(Data_year[end])[1];
            y = y[(end-Nt+1:end),:,:];
            E = E[(end-Nt+1:end),:,:];
            gas = gas[(end-Nt+1:end),:,:];
            coal = coal[(end-Nt+1:end),:,:];
            oil = oil[(end-Nt+1:end),:,:];
            elec = elec[(end-Nt+1:end),:,:];
            profit = profit[(end-Nt+1:end),:,:];
            FC = FC[(end-Nt+1:end),:,:];
            pE = pE[(end-Nt+1):end,:,:];
            pout_agg = pout_agg[end,:,:];
            CS = CS[end,:,:];
            co2 = p.γg*gas .+ p.γc*coal .+ p.γo*oil .+ p.γe*elec;
            netprofit = profit .+ FC;
            # Aggregate output (from rep consumer CES demand)
            Nfirms = Data_year[end].N[1];
            demand_shock = exp(p.d_t[end]);
            rhoterm = (p.ρ-1)/p.ρ;
            y_s_fs = zeros(p.S,p.Tf+1);
            for s = 1:p.S
                for tf = 1:(p.Tf+1)
                    y_s_fs[s,tf] = ((demand_shock/Nfirms)*(sum(y[:,s,tf].^rhoterm)))^(1/rhoterm);
                end
            end
            # Consumer surplus
            for tf = 1:(p.Tf+1)
                CS_fs[tf] = mean(CS[:,tf]);
            end
            # Average across simulations
            for tf = 1:Tf+1
                pE_fs[tf] = mean(pE[:,:,tf]);
                # pin_fs[tf] = mean(pin[:,:,tf]);
                y_fs[tf] = mean(y_s_fs[:,tf]);
                for i = 1:Nt
                    E_fs[tf] += mean(E[i,:,tf]);
                    gas_fs[tf] += mean(gas[i,:,tf]);
                    coal_fs[tf] += mean(coal[i,:,tf]);
                    oil_fs[tf] += mean(oil[i,:,tf]);
                    elec_fs[tf] += mean(elec[i,:,tf]);
                    co2_fs[tf] += mean(co2[i,:,tf]);
                    profit_fs[tf] += mean(profit[i,:,tf]);
                    FC_fs[tf] += mean(FC[i,:,tf]);
                    netprofit_fs[tf] += mean(netprofit[i,:,tf]);
                end
            end
            return y_fs,E_fs,gas_fs,coal_fs,oil_fs,elec_fs,co2_fs,profit_fs,netprofit_fs,CS_fs,FC_fs,pE_fs,gas,coal,pout_agg;
        end
    #
#

### Set taxes and initialize arrays
    τ = zeros(4,2);
    τ[:,2] = [p.γg*p.SCC_india/p.unit,p.γc*p.SCC_india/p.unit,p.γo*p.SCC_india/p.unit,p.γe*p.SCC_india/p.unit];
#

# Grid of aggregate output price for different tax rates (based on percentiles: 1%, 5%, 25%, 50%, 75%, 95%, 99%)
    # poutagg_grid = zeros(7,21);
    # for tau = 1:21
    #     poutagg_grid[:,tau] = percentile(SimulCompareModels_full["pout_agg"][:,:,tau][:],[1,5,25,50,75,95,99]);
    # end
#

### Perform simulations that compare models across tax rates
    function WelfareCompare_Compile(τ,model::String)
        taxeval = size(τ,2);               # Number of tax levels to evaluate
        Nt = size(Data_year[end])[1];
        y = zeros(p.Tf+1,taxeval);  
        E = zeros(p.Tf+1,taxeval); 
        gas = zeros(p.Tf+1,taxeval);
        coal = zeros(p.Tf+1,taxeval); 
        oil = zeros(p.Tf+1,taxeval);   
        elec = zeros(p.Tf+1,taxeval); 
        co2 = zeros(p.Tf+1,taxeval); 
        profit = zeros(p.Tf+1,taxeval);
        netprofit = zeros(p.Tf+1,taxeval);
        CS = zeros(p.Tf+1,taxeval); 
        FC = zeros(p.Tf+1,taxeval);
        pE= zeros(p.Tf+1,taxeval);
        pout_agg = zeros(p.S,p.Tf+1,taxeval);
        gas_ind = zeros(Nt,p.S,p.Tf+1,taxeval);
        coal_ind = zeros(Nt,p.S,p.Tf+1,taxeval);
        if model == "full"
            for tr = 1:taxeval
                y[:,tr],E[:,tr],gas[:,tr],coal[:,tr],oil[:,tr],elec[:,tr],co2[:,tr],profit[:,tr],netprofit[:,tr],CS[:,tr],FC[:,tr],pE[:,tr],gas_ind[:,:,:,tr],coal_ind[:,:,:,tr],pout_agg[:,:,tr] = Welfare_ForwardSimul(M,Data,p,τ[:,tr]);
            end
        elseif model == "restricted"
            for tr = 1:taxeval
                y[:,tr],E[:,tr],gas[:,tr],coal[:,tr],oil[:,tr],elec[:,tr],co2[:,tr],profit[:,tr],netprofit[:,tr],CS[:,tr],FC[:,tr],pE[:,tr],gas_ind[:,:,:,tr],coal_ind[:,:,:,tr],pout_agg[:,:,tr] = Welfare_ForwardSimul_restricted(M,Data,p,τ[:,tr]);
            end
        end
        return Dict("y"=>y,"E"=>E,"gas"=>gas,"coal"=>coal,"oil"=>oil,"elec"=>elec,"co2"=>co2,"profit"=>profit,"CS"=>CS,"FC"=>FC,"pE"=>pE,"gas_ind"=>gas_ind,"coal_ind"=>coal_ind,"pout_agg"=>pout_agg);
    end
    ### BASELINE ###
    # Model with Restricted switching (cannot switch as response to carbon tax)
    SimulCompareModels_rswitch = WelfareCompare_Compile(τ,"restricted");
    save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_rswitch.jld2", "SimulCompareModels_rswitch", SimulCompareModels_rswitch);
    # Main model
    SimulCompareModels_full = WelfareCompare_Compile(τ,"full");
    save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_full.jld2", "SimulCompareModels_full", SimulCompareModels_full);
#