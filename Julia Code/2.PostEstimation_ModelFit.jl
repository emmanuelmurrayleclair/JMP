# Post-estimation of dynamic discrete choice model with real data
    # Shapley decomposition of energy marginal
    # Model Fit
    # Estimation results
# Emmanuel Murray Leclair
# August 2023

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
println("                  Shapley Decomposition, Model fit and Estimation Results                ")
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
        # pout_tol::Float64   = 1E-15  ; # Tolerance for fixed point in aggregate output price index
        pout_tol::Float64   = 1E-5  ; # Tolerance for fixed point in aggregate output price index
        υ                   = 0.0    ; # Dampen factor
        # misc 
        F_tot               = 4      ; # Total number of fuels
        β                   = 0.9    ; # Discount factor
        T_pers              = size(μψe_t,1);    # number of years, persistent variables
        T_nonpers           = size(po,1);       # Number of years, non persistent variables
        T                   = T_pers-1;         # Actual number of years where I kept data
        Tf                  = 40      ;         # Number of years of forward simulation
        S                   = 50      ;         # Number of simulations for CCP
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
        # Fuel quantities
        fqty_oe = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                 ;
            fqty_oe_norm = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)        ;
        fqty_oge = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                ;
            fqty_oge_norm = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)       ;
        fqty_oce = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                ;
            fqty_oce_norm = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)       ;
        fqty_ogce = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)               ;
            fqty_ogce_norm = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)      ;

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
        # Fuel prices
        pEgrid_oe = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                   ;
        pEgrid_oge = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                  ;
        pEgrid_oce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                  ;
        pEgrid_ogce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                 ;
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

## Generates state transition markov chains (multivariate markov chains)
    M = StateTransition_func_multivariate(p,M);
    test=1;
#

# Find closest point on the grid for each state variable
    grid_indices,dist_grid_smallest = MatchGridData(M,Data);
#

## Use moments of the prices and productivity for coal and gas to discretize distribution of price separately from productivity
    M = DiscretizeGasCoal_distr(p,M);
    test=1;
#

#-----------------------------------------------------------
#-----------------------------------------------------------
# 2. Simulations

### Find equilibrium aggregate output price
    τnotax = [0,0,0,0];
    p,M = OutputPrice_func_nogrid(M,Data,p,τnotax);
    test=1;
#
###  Compute Static Profits under all combination of state variables given tax rates
    @time M = StaticProfit_grid(M,τnotax)   ;
    test=1;
#
### Compute predicted profit under state variables in the data 
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
    loglik_eval=VF_bench(nothing);
    @time M = W_max(M,p.Κ,p.σunit,loglik_eval);
    test=1;
#

### Update posterior conditional and unconditional probabilities of random effects
    M,π_cond,Prnext_long = PosteriorRE_func_zfc_cf(M,Data,grid_indices,p.Κ);
#

# Get simulation draws
    # Without bounds
    seed = 13421;
    # seed = 14123;
    @time M,state_resdraw1 = ForwardSimul_draws(p,M,Data,seed);
    test=1;
#

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------
# 3. Create and export dataset with gas and coal productivity for counterfactual fuel sets

# Initialize dataset
Data_prod = DataFrame(IDnum = Data.IDnum, year = Data.year, lnfprod_g_old = Data.lnfprod_g, lnfprod_c_old = Data.lnfprod_c);
# Add gas and coal productivity for each simulation (wide format, convert to long in stata)
lnfprod_g = zeros(M.N,p.S);
lnfprod_c = zeros(M.N,p.S);
for i = 1:M.N
    for s = 1:p.S
        lnfprod_g[i,s] = M.lnψg_fs[i,s,1] + p.lng_re_grid[M.gre_fs[i,s]];
        lnfprod_c[i,s] = M.lnψc_fs[i,s,1] + p.lnc_re_grid[M.cre_fs[i,s]];
    end
end
for s = 1:p.S
    local snum = Int(s);
    Data_prod[!, "lnfprod_g$snum"] = lnfprod_g[:,s];
    Data_prod[!, "lnfprod_c$snum"] = lnfprod_c[:,s];
end

### Get expected choice probability (over the conditional distribution of random effects)
    Prnext_all = Array{Float64}(undef,M.N,4,p.nc_re,p.ng_re);
    vfchoice_all = choiceVF_func_grid_zfc(M,Data,grid_indices,p.Κ);
    for i = 1:M.N
        for j = 1:(M.nc_re*M.ng_re)
            ic_re,ig_re = Tuple(CartesianIndices((M.nc_re,M.ng_re))[j]);
            vtilde = maximum([vfchoice_all[i,1,ic_re,ig_re],vfchoice_all[i,2,ic_re,ig_re],vfchoice_all[i,3,ic_re,ig_re],vfchoice_all[i,4,ic_re,ig_re]]);
            aux_2 = exp(vfchoice_all[i,1,ic_re,ig_re]-vtilde) + exp(vfchoice_all[i,2,ic_re,ig_re]-vtilde) + exp(vfchoice_all[i,3,ic_re,ig_re]-vtilde) + exp(vfchoice_all[i,4,ic_re,ig_re]-vtilde) ;
            Prnext_all[i,1,ic_re,ig_re] = exp(vfchoice_all[i,1,ic_re,ig_re]-vtilde)/aux_2;
            Prnext_all[i,2,ic_re,ig_re] = exp(vfchoice_all[i,2,ic_re,ig_re]-vtilde)/aux_2;
            Prnext_all[i,3,ic_re,ig_re] = exp(vfchoice_all[i,3,ic_re,ig_re]-vtilde)/aux_2;
            Prnext_all[i,4,ic_re,ig_re] = exp(vfchoice_all[i,4,ic_re,ig_re]-vtilde)/aux_2;
        end
    end
    Eprnext = zeros(p.F_tot,M.N);
    Eprnext_uncond = zeros(p.F_tot,M.N);
    Data_ID = groupby(Data,:IDnum);
    Nfirms = size(unique(Data.IDnum))[1];
    for f = 1:4
        for ifirm = 1:Nfirms
            for j = 1:size(Data_ID[ifirm],1)
                i = Data_ID[ifirm].id[j];
                Eprnext[f,i] = sum( M.π_cond[ifirm,:,:].*Prnext_all[i,f,:,:] );# Version with conditional distr 
                Eprnext_uncond[f,i] = sum( p.π_uncond.*Prnext_all[i,f,:,:] );  # Version with unconditional distr
            end
        end
    end
    # Add probabilities to dataset
    Data_prod[!, "Proe_cond"] = Eprnext[1,:];
    Data_prod[!, "Proge_cond"] = Eprnext[2,:];
    Data_prod[!, "Proce_cond"] = Eprnext[3,:];
    Data_prod[!, "Progce_cond"] = Eprnext[4,:];
    Data_prod[!, "Proe_uncond"] = Eprnext_uncond[1,:];
    Data_prod[!, "Proge_uncond"] = Eprnext_uncond[2,:];
    Data_prod[!, "Proce_uncond"] = Eprnext_uncond[3,:];
    Data_prod[!, "Progce_uncond"] = Eprnext_uncond[4,:];
#

# Export dataset
CSV.write("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Results/Data_prod.csv",Data_prod);


#--------------------------------------------------------------
#--------------------------------------------------------------
# 4. Table of Fixed Costs Estimates (no standard error)

# Change units of parameter estimates
    Κdollar = p.Κ*p.σunit;
    # Display
    global κg_np = "$(round(Κdollar[1],digits=2))";
    global κc = "$(round(Κdollar[2],digits=2))";
    global γg_np = "$(round(Κdollar[3],digits=2))";
    global γc = "$(round(Κdollar[4],digits=2))";
    global κg_p = "$(round(Κdollar[5],digits=2))";
    global γg_p = "$(round(Κdollar[6],digits=2))";
    global κz = "$(round(Κdollar[7],digits=2))";
    global γz = "$(round(Κdollar[8],digits=2))";
    global Nobs = Int(size(Data,1));
    # Write table
        tex_table = """
        \\begin{tabular}{@{}lccc@{}}
        \\toprule
        & \\textbf{} & \\textbf{\\begin{tabular}[c]{@{}c@{}}Fixed Costs\\\\ (Million USD)\\end{tabular}} & \\textbf{\\begin{tabular}[c]{@{}c@{}}Salvage Values\\\\ (Million USD)\\end{tabular}} \\\\ \\midrule
        \\multirow{2}{*}{Natural Gas} & Pipeline Access & $κg_p & $γg_p \\\\
        & No Pipeline Access & $κg_np & $γg_np \\\\
        \\multicolumn{2}{l}{Coal} & $κc & $γc \\\\
        \\multicolumn{2}{l}{Effect of Total Factor Productivity} & $κz & $γz \\\\ \\midrule
        \\multicolumn{2}{l}{Observations} & \\multicolumn{2}{c}{$Nobs} \\\\ \\bottomrule
        \\end{tabular}
        """
        fname = "FC_estimates.tex"
        dirpath = "Results/Tables"
        fpath = joinpath(dirpath,fname);
        open(fpath, "w") do file
            write(file, tex_table)
        end
    #

#

#--------------------------------------------------------------
#--------------------------------------------------------------
# 5. Graphs of model fit

### Create graphs of model fit
    #1. Graphs of overall distribution
        setF = combine(groupby(Data,:FcombineF), nrow => :data)           ;
        setF.data = setF.data./M.N;
        setF = sort(setF);
        setF[!,:model].=0.0;
        setF.model[setF.FcombineF.==12] .= mean(Eprnext[1,:]);      # oe
        setF.model[setF.FcombineF.==123] .= mean(Eprnext[3,:]);     # oce
        setF.model[setF.FcombineF.==124] .= mean(Eprnext[2,:]);     # oge
        setF.model[setF.FcombineF.==1234] .= mean(Eprnext[4,:]);    # ogce
        setF = coalesce.(setF,0.0)                                             ;
        setF = stack(setF,[:data,:model])                                      ;
        setF.FcombineF = string.(setF[!,:FcombineF])                           ;
        nn = size(setF)[1]                                                     ;
        for i = 1:nn
            if setF.FcombineF[i] == "12.0" || setF.FcombineF[i] == "12"
                setF.FcombineF[i] = "oe"    ; 
            elseif setF.FcombineF[i] == "123.0" || setF.FcombineF[i] == "123"
                setF.FcombineF[i] = "oce"   ;
            elseif setF.FcombineF[i] == "124.0" || setF.FcombineF[i] == "124"
                setF.FcombineF[i] = "oge"   ;
            elseif setF.FcombineF[i] == "1234.0" || setF.FcombineF[i] == "1234"
                setF.FcombineF[i] = "ogce"  ;
            end
        end
        plotd = @df setF groupedbar(:FcombineF,:value,group =:variable,framestyle=:box,xlabel = "Fuel Sets", ylabel = "Probability",lw = 1) ;
        savefig(plotd,"Results/Figures/Distr_overall.pdf");test=1;
    #
    #2. Graph that counts probability of fuel switching from OE
        setF = combine(groupby(Data,[:combineF,:FcombineF]), nrow => :data)            ;
        setF[!,:model].=0.0;
        setF = sort(setF,:combineF);
        setF_byset = groupby(setF,:combineF);
        setF_fromOE = DataFrame(setF_byset[1]);
        Noe = sum(setF_fromOE.data);
        setF_fromOE.data = setF_fromOE.data./Noe;
        setF_fromOE.model[setF_fromOE.FcombineF.==12] .= mean(Eprnext[1,Data.combineF.==12]);       # oe
        setF_fromOE.model[setF_fromOE.FcombineF.==123] .= mean(Eprnext[3,Data.combineF.==12]);      # oce
        setF_fromOE.model[setF_fromOE.FcombineF.==124] .= mean(Eprnext[2,Data.combineF.==12]);      # oge
        setF_fromOE.model[setF_fromOE.FcombineF.==1234] .= mean(Eprnext[4,Data.combineF.==12]);     # ogce
        setF_fromOE.FcombineF = string.(setF_byset[1][!,:FcombineF])                                  ;
        setF_fromOE.combineF = string.(setF_byset[1][!,:combineF])                                    ;
        nn = size(setF_fromOE)[1];
        for i = 1:nn
            if setF_fromOE.FcombineF[i] == "12.0" || setF_fromOE.FcombineF[i] == "12"
                setF_fromOE.FcombineF[i] = "to oe"    ; 
            elseif setF_fromOE.FcombineF[i] == "123.0" || setF_fromOE.FcombineF[i] == "123"
                setF_fromOE.FcombineF[i] = "to oce"   ;
            elseif setF_fromOE.FcombineF[i] == "124.0" || setF_fromOE.FcombineF[i] == "124"
                setF_fromOE.FcombineF[i] = "to oge"   ;
            elseif setF_fromOE.FcombineF[i] == "1234.0" || setF_fromOE.FcombineF[i] == "1234"
                setF_fromOE.FcombineF[i] = "to ogce"  ;
            end
        end
        setF_fromOE = stack(setF_fromOE,[:data,:model])                                      ;
        @df setF_fromOE groupedbar(:FcombineF,:value,group =:variable,xlabel="From Oil and Electricity (oe)",ylabel = "Probability",framestyle=:box) ;
        savefig("Results/Figures/Distr_oe.pdf");
    #
    #3. Graph that counts probability of fuel switching from OCE
        setF_fromOCE = DataFrame(setF_byset[2]);
        Noce = sum(setF_fromOCE.data);
        setF_fromOCE.data = setF_fromOCE.data./Noce;
        setF_fromOCE = sort(setF_fromOCE,:FcombineF);
        setF_fromOCE.model[setF_fromOCE.FcombineF.==12] .= mean(Eprnext[1,Data.combineF.==123]);        # oe
        setF_fromOCE.model[setF_fromOCE.FcombineF.==123] .= mean(Eprnext[3,Data.combineF.==123]);       # oce
        setF_fromOCE.model[setF_fromOCE.FcombineF.==124] .= mean(Eprnext[2,Data.combineF.==123]);       # oge
        setF_fromOCE.model[setF_fromOCE.FcombineF.==1234] .= mean(Eprnext[4,Data.combineF.==123]);      # ogce
        setF_fromOCE.FcombineF = string.(setF_fromOCE[!,:FcombineF])                                  ;
        setF_fromOCE.combineF = string.(setF_fromOCE[!,:combineF])                                    ;
        nn = size(setF_fromOCE)[1];
        for i = 1:nn
            if setF_fromOCE.FcombineF[i] == "12.0" || setF_fromOCE.FcombineF[i] == "12"
                setF_fromOCE.FcombineF[i] = "to oe"    ; 
            elseif setF_fromOCE.FcombineF[i] == "123.0" || setF_fromOCE.FcombineF[i] == "123"
                setF_fromOCE.FcombineF[i] = "to oce"   ;
            elseif setF_fromOCE.FcombineF[i] == "124.0" || setF_fromOCE.FcombineF[i] == "124"
                setF_fromOCE.FcombineF[i] = "to oge"   ;
            elseif setF_fromOCE.FcombineF[i] == "1234.0" || setF_fromOCE.FcombineF[i] == "1234"
                setF_fromOCE.FcombineF[i] = "to ogce"  ;
            end
        end
        setF_fromOCE = stack(setF_fromOCE,[:data,:model])                                      ;
        @df setF_fromOCE groupedbar(:FcombineF,:value,group =:variable,xlabel="From Oil, Coal and Electricity (oce)",ylabel = "Probability",framestyle=:box) ;
        savefig("Results/Figures/Distr_oce.pdf");
    #
    #4. Graph that counts probability of fuel switching from OGE
        setF_fromOGE = DataFrame(setF_byset[3]);
        Noge = sum(setF_fromOGE.data);
        setF_fromOGE.data = setF_fromOGE.data./Noge;
        setF_fromOGE = sort(setF_fromOGE,:FcombineF);
        setF_fromOGE.model[setF_fromOGE.FcombineF.==12] .= mean(Eprnext[1,Data.combineF.==124]);        # oe
        setF_fromOGE.model[setF_fromOGE.FcombineF.==123] .= mean(Eprnext[3,Data.combineF.==124]);       # oce
        setF_fromOGE.model[setF_fromOGE.FcombineF.==124] .= mean(Eprnext[2,Data.combineF.==124]);       # oge
        setF_fromOGE.model[setF_fromOGE.FcombineF.==1234] .= mean(Eprnext[4,Data.combineF.==124]);      # ogce
        setF_fromOGE.FcombineF = string.(setF_fromOGE[!,:FcombineF])                                  ;
        setF_fromOGE.combineF = string.(setF_fromOGE[!,:combineF])                                    ;
        nn = size(setF_fromOGE)[1];
        for i = 1:nn
            if setF_fromOGE.FcombineF[i] == "12.0" || setF_fromOGE.FcombineF[i] == "12"
                setF_fromOGE.FcombineF[i] = "to oe"    ; 
            elseif setF_fromOGE.FcombineF[i] == "123.0" || setF_fromOGE.FcombineF[i] == "123"
                setF_fromOGE.FcombineF[i] = "to oce"   ;
            elseif setF_fromOGE.FcombineF[i] == "124.0" || setF_fromOGE.FcombineF[i] == "124"
                setF_fromOGE.FcombineF[i] = "to oge"   ;
            elseif setF_fromOGE.FcombineF[i] == "1234.0" || setF_fromOGE.FcombineF[i] == "1234"
                setF_fromOGE.FcombineF[i] = "to ogce"  ;
            end
        end
        setF_fromOGE = stack(setF_fromOGE,[:data,:model])                                      ;
        @df setF_fromOGE groupedbar(:FcombineF,:value,group =:variable,xlabel="From Oil, Gas and Electricity (oge)",ylabel = "Probability",framestyle=:box) ;
        savefig("Results/Figures/Distr_oge.pdf");
    #
    #5. Graph that counts probability of fuel switching from OGCE 
        setF_fromOGCE = DataFrame(setF_byset[4]);
        Nogce = sum(setF_fromOGCE.data);
        setF_fromOGCE.data = setF_fromOGCE.data./Nogce;
        setF_fromOGCE = sort(setF_fromOGCE,:FcombineF);
        setF_fromOGCE.model[setF_fromOGCE.FcombineF.==12] .= mean(Eprnext[1,Data.combineF.==1234]);      # oe
        setF_fromOGCE.model[setF_fromOGCE.FcombineF.==123] .= mean(Eprnext[3,Data.combineF.==1234]);     # oce
        setF_fromOGCE.model[setF_fromOGCE.FcombineF.==124] .= mean(Eprnext[2,Data.combineF.==1234]);     # oge
        setF_fromOGCE.model[setF_fromOGCE.FcombineF.==1234] .= mean(Eprnext[4,Data.combineF.==1234]);    # ogce
        setF_fromOGCE.FcombineF = string.(setF_fromOGCE[!,:FcombineF])                                  ;
        setF_fromOGCE.combineF = string.(setF_fromOGCE[!,:combineF])                                    ;
        nn = size(setF_fromOGCE)[1];
        for i = 1:nn
            if setF_fromOGCE.FcombineF[i] == "12.0" || setF_fromOGCE.FcombineF[i] == "12"
                setF_fromOGCE.FcombineF[i] = "to oe"    ; 
            elseif setF_fromOGCE.FcombineF[i] == "123.0" || setF_fromOGCE.FcombineF[i] == "123"
                setF_fromOGCE.FcombineF[i] = "to oce"   ;
            elseif setF_fromOGCE.FcombineF[i] == "124.0" || setF_fromOGCE.FcombineF[i] == "124"
                setF_fromOGCE.FcombineF[i] = "to oge"   ;
            elseif setF_fromOGCE.FcombineF[i] == "1234.0" || setF_fromOGCE.FcombineF[i] == "1234"
                setF_fromOGCE.FcombineF[i] = "to ogce"  ;
            end
        end
        setF_fromOGCE = stack(setF_fromOGCE,[:data,:model])                                      ;
        @df setF_fromOGCE groupedbar(:FcombineF,:value,group =:variable,xlabel="From Oil, Gas, Coal and Electricity (ogce)",ylabel = "Probability",framestyle=:box) ;
        savefig("Results/Figures/Distr_ogce.pdf");
    #
#


#----------------------------------------------------------------------
#----------------------------------------------------------------------
# 6. Shapley Decomposition of Energy Marginal Cost

# Initialize
    Data_combineF = groupby(Data,:combineF);
    N = size(Data,1);
    po_avg = mean(exp.(Data_combineF[1].lnpo_tilde)); po_norm = fill(po_avg,N);
    pe_avg = mean(exp.(Data_combineF[1].lnpelec_tilde)); pe_norm = fill(pe_avg,N);
    pg_avg = mean(exp.(skipmissing(Data.lnpg_tilde))); pg_norm = fill(pg_avg,N);
    pc_avg = mean(exp.(skipmissing(Data.lnpc_tilde))); pc_norm = fill(pc_avg,N);
    prodo_avg = mean(exp.(Data_combineF[1].lnfprod_o)); prodo_norm = fill(prodo_avg,N);
    prode_avg = mean(exp.(Data_combineF[1].lnfprod_e)); prode_norm = fill(prode_avg,N);
    prodg_avg = mean(exp.(skipmissing(Data.lnfprod_g))); prodg_norm = fill(prodg_avg,N);
    prodc_avg = mean(exp.(skipmissing(Data.lnfprod_c))); prodc_norm = fill(prodc_avg,N);
    prodg_avg_selec = mean(exp.(skipmissing(Data.lnfprod_g))); prodg_norm_select = fill(prodg_avg_selec,N);
    prodc_avg_selec = mean(exp.(skipmissing(Data.lnfprod_c))); prodc_norm_select = fill(prodc_avg_selec,N);
#

# Function that returns a submodel
    function MCEnergy_submodel(pf,pf_oe,ψf,ψf_oe,F,N,Noe,p::Par)
        # Initialize inputs: prices and productivity
        po = pf[:,1]; po_oe = pf_oe[:,1];
        pe = pf[:,2]; pe_oe = pf_oe[:,2];
        ψo = ψf[:,1]; ψo_oe = ψf_oe[:,1];
        ψe = ψf[:,2]; ψe_oe = ψf_oe[:,2];
        pE_oe = zeros(Noe);
        for i = 1:Noe
            pfψf_oe = [po_oe[i]/ψo_oe[i],pe_oe[i]/ψe_oe[i]];
            pE_oe[i] = pE_func(12,pfψf_oe,p);
        end
        pE_setF = zeros(N);
        if F==12
            for i = 1:N
                pfψf = [po[i]/ψo[i],pe[i]/ψe[i]];
                pE_setF[i] = pE_func(F,pfψf,p);
            end
        elseif F==123
            pc = pf[:,3];
            ψc = ψf[:,3];
            for i = 1:N
                pfψf = [po[i]/ψo[i],pe[i]/ψe[i],pc[i]/ψc[i]];
                pE_setF[i] = pE_func(F,pfψf,p);
            end
        elseif F==124
            pg = pf[:,3];
            ψg = ψf[:,3];
            for i = 1:N
                pfψf = [po[i]/ψo[i],pe[i]/ψe[i],pg[i]/ψg[i]];
                pE_setF[i] = pE_func(F,pfψf,p);
            end
        elseif F==1234
            pc = pf[:,3];
            ψc = ψf[:,3];
            pg = pf[:,4];
            ψg = ψf[:,4];
            for i = 1:N
                pfψf = [po[i]/ψo[i],pe[i]/ψe[i],pc[i]/ψc[i],pg[i]/ψg[i]];
                pE_setF[i] = pE_func(F,pfψf,p);
            end
        end
        return mean(pE_setF)-mean(pE_oe);
        # return median(pE_setF)-median(pE_oe);
    end
#
# initialize total difference in energy marginal cost
# OE
    Noe = size(Data_combineF[1])[1];
    pE_oe = zeros(Noe);
    for i = 1:Noe
        po = exp(Data_combineF[1].lnpo_tilde[i]); ψo = exp(Data_combineF[1].lnfprod_o[i]);
        pe = exp(Data_combineF[1].lnpelec_tilde[i]); ψe = exp(Data_combineF[1].lnfprod_e[i]);
        pfψf = [po/ψo,pe/ψe];
        pE_oe[i] = pE_func(12,pfψf,p);
    end
#
# OCE
    Noce = size(Data_combineF[2])[1];
    pE_oce = zeros(Noce);
    for i = 1:Noce
        po = exp(Data_combineF[2].lnpo_tilde[i]); ψo = exp(Data_combineF[2].lnfprod_o[i]);
        pe = exp(Data_combineF[2].lnpelec_tilde[i]); ψe = exp(Data_combineF[2].lnfprod_e[i]);
        pc = exp(Data_combineF[2].lnpc_tilde[i]); ψc = exp(Data_combineF[2].lnfprod_c[i]);
        pfψf = [po/ψo,pe/ψe,pc/ψc];
        pE_oce[i] = pE_func(123,pfψf,p);
    end
#
# OGE
    Noge = size(Data_combineF[3])[1];
    pE_oge = zeros(Noge);
    for i = 1:Noge
        po = exp(Data_combineF[3].lnpo_tilde[i]); ψo = exp(Data_combineF[3].lnfprod_o[i]);
        pe = exp(Data_combineF[3].lnpelec_tilde[i]); ψe = exp(Data_combineF[3].lnfprod_e[i]);
        pg = exp(Data_combineF[3].lnpg_tilde[i]); ψg = exp(Data_combineF[3].lnfprod_g[i]);
        pfψf = [po/ψo,pe/ψe,pg/ψg];
        pE_oge[i] = pE_func(124,pfψf,p);
    end
#
# OGCE
    Nogce = size(Data_combineF[4])[1];
    pE_ogce = zeros(Nogce);
    for i = 1:Nogce
        po = exp(Data_combineF[4].lnpo_tilde[i]); ψo = exp(Data_combineF[4].lnfprod_o[i]);
        pe = exp(Data_combineF[4].lnpelec_tilde[i]); ψe = exp(Data_combineF[4].lnfprod_e[i]);
        pc = exp(Data_combineF[4].lnpc_tilde[i]); ψc = exp(Data_combineF[4].lnfprod_c[i]);
        pg = exp(Data_combineF[4].lnpg_tilde[i]); ψg = exp(Data_combineF[4].lnfprod_g[i]);
        pfψf = [po/ψo,pe/ψe,pc/ψc,pg/ψg];
        pE_ogce[i] = pE_func(1234,pfψf,p)
    end
#
# Perform the decomposition accounting for selection on unobservables
    Noe = size(Data_combineF[1])[1];
    po_oe = [po_norm[Data.combineF.==12] exp.(Data_combineF[1].lnpo_tilde)];
    pe_oe = [pe_norm[Data.combineF.==12] exp.(Data_combineF[1].lnpelec_tilde)];
    ψo_oe = [prodo_norm[Data.combineF.==12] exp.(Data_combineF[1].lnfprod_o)];
    ψe_oe = [prode_norm[Data.combineF.==12] exp.(Data_combineF[1].lnfprod_e)];
    # OCE
        F = 123;
        Noce = size(Data_combineF[2])[1];
        po_oce = [po_norm[Data.combineF.==123] exp.(Data_combineF[2].lnpo_tilde)];
        pe_oce = [pe_norm[Data.combineF.==123] exp.(Data_combineF[2].lnpelec_tilde)];
        pc_oce = [pc_norm[Data.combineF.==123] exp.(Data_combineF[2].lnpc_tilde)];
        ψo_oce = [prodo_norm[Data.combineF.==123] exp.(Data_combineF[2].lnfprod_o)];
        ψe_oce = [prode_norm[Data.combineF.==123] exp.(Data_combineF[2].lnfprod_e)];
        ψc_oce = [prodc_norm[Data.combineF.==123] exp.(Data_combineF[2].lnfprod_c)];
        po = zeros(Noce,2,2,2); po_oce_oe = zeros(Noe,2,2,2);
        pe = zeros(Noce,2,2,2); pe_oce_oe = zeros(Noe,2,2,2);
        pc = zeros(Noce,2,2,2);
        ψo = zeros(Noce,2,2,2); ψo_oce_oe = zeros(Noe,2,2,2); 
        ψe = zeros(Noce,2,2,2); ψe_oce_oe = zeros(Noe,2,2,2);
        ψc = zeros(Noce,2,2,2);
        MCE_subm = zeros(2,2,2);
        for j = 1:(2^3)
            ip,iψ,iF = Tuple(CartesianIndices((2,2,2))[j]);
            for i = 1:Noce
                po[i,ip,iψ,iF] = po_oce[i,ip];
                pe[i,ip,iψ,iF] = pe_oce[i,ip];
                pc[i,ip,iψ,iF] = pc_oce[i,ip];
                ψo[i,ip,iψ,iF] = ψo_oce[i,iψ];
                ψe[i,ip,iψ,iF] = ψe_oce[i,iψ];
                ψc[i,ip,iψ,iF] = ψc_oce[i,iψ];
            end
            for i = 1:Noe
                po_oce_oe[i,ip,iψ,iF] = po_oe[i,ip];
                pe_oce_oe[i,ip,iψ,iF] = pe_oe[i,ip];
                ψo_oce_oe[i,ip,iψ,iF] = ψo_oe[i,iψ];
                ψe_oce_oe[i,ip,iψ,iF] = ψe_oe[i,iψ];
            end
            pf_oce = [po[:,ip,iψ,iF] pe[:,ip,iψ,iF] pc[:,ip,iψ,iF]];
            ψf_oce = [ψo[:,ip,iψ,iF] ψe[:,ip,iψ,iF] ψc[:,ip,iψ,iF]];
            pf_oe = [po_oce_oe[:,ip,iψ,iF] pe_oce_oe[:,ip,iψ,iF]];
            ψf_oe = [ψo_oce_oe[:,ip,iψ,iF] ψe_oce_oe[:,ip,iψ,iF]];
            if iF == 1
                F = 12;
            elseif iF == 2
                F = 123;
            end
            MCE_subm[ip,iψ,iF] = MCEnergy_submodel(pf_oce,pf_oe,ψf_oce,ψf_oe,F,Noce,Noe,p);
        end 
        CF_oce = (1/3)*(MCE_subm[1,1,2]-MCE_subm[1,1,1]) + (1/6)*(MCE_subm[2,1,2]-MCE_subm[2,1,1]) + (1/6)*(MCE_subm[1,2,2]-MCE_subm[1,2,1]) + (1/3)*(MCE_subm[2,2,2]-MCE_subm[2,2,1]);
        Cψ_oce = (1/3)*(MCE_subm[1,2,1]-MCE_subm[1,1,1]) + (1/6)*(MCE_subm[2,2,1]-MCE_subm[2,1,1]) + (1/6)*(MCE_subm[1,2,2]-MCE_subm[1,1,2]) + (1/3)*(MCE_subm[2,2,2]-MCE_subm[2,1,2]);
        Cp_oce = (1/3)*(MCE_subm[2,1,1]-MCE_subm[1,1,1]) + (1/6)*(MCE_subm[2,2,1]-MCE_subm[1,2,1]) + (1/6)*(MCE_subm[2,1,2]-MCE_subm[1,1,2]) + (1/3)*(MCE_subm[2,2,2]-MCE_subm[1,2,2]);
        # Total difference
        C_oce = mean(pE_oce)-mean(pE_oe);
        # C_oce = median(pE_oce)-median(pE_oe);
    #
    # OGE
        F = 124;
        Noge = size(Data_combineF[3])[1];
        po_oge = [po_norm[Data.combineF.==124] exp.(Data_combineF[3].lnpo_tilde)];
        pe_oge = [pe_norm[Data.combineF.==124] exp.(Data_combineF[3].lnpelec_tilde)];
        pg_oge = [pg_norm[Data.combineF.==124] exp.(Data_combineF[3].lnpg_tilde)];
        ψo_oge = [prodo_norm[Data.combineF.==124] exp.(Data_combineF[3].lnfprod_o)];
        ψe_oge = [prode_norm[Data.combineF.==124] exp.(Data_combineF[3].lnfprod_e)];
        ψg_oge = [prodg_norm[Data.combineF.==124] exp.(Data_combineF[3].lnfprod_g)];
        po = zeros(Noge,2,2,2); po_oge_oe = zeros(Noe,2,2,2);
        pe = zeros(Noge,2,2,2); pe_oge_oe = zeros(Noe,2,2,2);
        pg = zeros(Noge,2,2,2);
        ψo = zeros(Noge,2,2,2); ψo_oge_oe = zeros(Noe,2,2,2); 
        ψe = zeros(Noge,2,2,2); ψe_oge_oe = zeros(Noe,2,2,2);
        ψg = zeros(Noge,2,2,2);
        MCE_subm = zeros(2,2,2);
        for j = 1:(2^3)
            ip,iψ,iF = Tuple(CartesianIndices((2,2,2))[j]);
            for i = 1:Noge
                po[i,ip,iψ,iF] = po_oge[i,ip];
                pe[i,ip,iψ,iF] = pe_oge[i,ip];
                pg[i,ip,iψ,iF] = pg_oge[i,ip];
                ψo[i,ip,iψ,iF] = ψo_oge[i,iψ];
                ψe[i,ip,iψ,iF] = ψe_oge[i,iψ];
                ψg[i,ip,iψ,iF] = ψg_oge[i,iψ];
            end
            for i = 1:Noe
                po_oge_oe[i,ip,iψ,iF] = po_oe[i,ip];
                pe_oge_oe[i,ip,iψ,iF] = pe_oe[i,ip];
                ψo_oge_oe[i,ip,iψ,iF] = ψo_oe[i,iψ];
                ψe_oge_oe[i,ip,iψ,iF] = ψe_oe[i,iψ];
            end
            pf_oge = [po[:,ip,iψ,iF] pe[:,ip,iψ,iF] pg[:,ip,iψ,iF]];
            ψf_oge = [ψo[:,ip,iψ,iF] ψe[:,ip,iψ,iF] ψg[:,ip,iψ,iF]];
            pf_oe = [po_oge_oe[:,ip,iψ,iF] pe_oge_oe[:,ip,iψ,iF]];
            ψf_oe = [ψo_oge_oe[:,ip,iψ,iF] ψe_oge_oe[:,ip,iψ,iF]];
            if iF == 1
                F = 12;
            elseif iF == 2
                F = 124;
            end
            MCE_subm[ip,iψ,iF] = MCEnergy_submodel(pf_oge,pf_oe,ψf_oge,ψf_oe,F,Noge,Noe,p);
        end 
        CF_oge = (1/3)*(MCE_subm[1,1,2]-MCE_subm[1,1,1]) + (1/6)*(MCE_subm[2,1,2]-MCE_subm[2,1,1]) + (1/6)*(MCE_subm[1,2,2]-MCE_subm[1,2,1]) + (1/3)*(MCE_subm[2,2,2]-MCE_subm[2,2,1]);
        Cψ_oge = (1/3)*(MCE_subm[1,2,1]-MCE_subm[1,1,1]) + (1/6)*(MCE_subm[2,2,1]-MCE_subm[2,1,1]) + (1/6)*(MCE_subm[1,2,2]-MCE_subm[1,1,2]) + (1/3)*(MCE_subm[2,2,2]-MCE_subm[2,1,2]);
        Cp_oge = (1/3)*(MCE_subm[2,1,1]-MCE_subm[1,1,1]) + (1/6)*(MCE_subm[2,2,1]-MCE_subm[1,2,1]) + (1/6)*(MCE_subm[2,1,2]-MCE_subm[1,1,2]) + (1/3)*(MCE_subm[2,2,2]-MCE_subm[1,2,2]);
        # Total difference
        C_oge = mean(pE_oge)-mean(pE_oe);
        # C_oge = median(pE_oge)-median(pE_oe);
    #
    # OGCE
        F = 1234;
        Nogce = size(Data_combineF[4])[1];
        po_ogce = [po_norm[Data.combineF.==1234] exp.(Data_combineF[4].lnpo_tilde)];
        pe_ogce = [pe_norm[Data.combineF.==1234] exp.(Data_combineF[4].lnpelec_tilde)];
        pc_ogce = [pc_norm[Data.combineF.==1234] exp.(Data_combineF[4].lnpc_tilde)];
        pg_ogce = [pg_norm[Data.combineF.==1234] exp.(Data_combineF[4].lnpg_tilde)];
        ψo_ogce = [prodo_norm[Data.combineF.==1234] exp.(Data_combineF[4].lnfprod_o)];
        ψe_ogce = [prode_norm[Data.combineF.==1234] exp.(Data_combineF[4].lnfprod_e)];
        ψc_ogce = [prodc_norm[Data.combineF.==1234] exp.(Data_combineF[4].lnfprod_c)];
        ψg_ogce = [prodg_norm[Data.combineF.==1234] exp.(Data_combineF[4].lnfprod_g)];
        po = zeros(Nogce,2,2,2); po_ogce_oe = zeros(Noe,2,2,2);
        pe = zeros(Nogce,2,2,2); pe_ogce_oe = zeros(Noe,2,2,2);
        pc = zeros(Nogce,2,2,2);
        pg = zeros(Nogce,2,2,2);
        ψo = zeros(Nogce,2,2,2); ψo_ogce_oe = zeros(Noe,2,2,2); 
        ψe = zeros(Nogce,2,2,2); ψe_ogce_oe = zeros(Noe,2,2,2);
        ψc = zeros(Nogce,2,2,2);
        ψg = zeros(Nogce,2,2,2);
        MCE_subm = zeros(2,2,2);
        for j = 1:(2^3)
            ip,iψ,iF = Tuple(CartesianIndices((2,2,2))[j]);
            for i = 1:Nogce
                po[i,ip,iψ,iF] = po_ogce[i,ip];
                pe[i,ip,iψ,iF] = pe_ogce[i,ip];
                pc[i,ip,iψ,iF] = pc_ogce[i,ip];
                pg[i,ip,iψ,iF] = pg_ogce[i,ip];
                ψo[i,ip,iψ,iF] = ψo_ogce[i,iψ];
                ψe[i,ip,iψ,iF] = ψe_ogce[i,iψ];
                ψc[i,ip,iψ,iF] = ψc_ogce[i,iψ];
                ψg[i,ip,iψ,iF] = ψg_ogce[i,iψ];
            end
            for i = 1:Noe
                po_ogce_oe[i,ip,iψ,iF] = po_oe[i,ip];
                pe_ogce_oe[i,ip,iψ,iF] = pe_oe[i,ip];
                ψo_ogce_oe[i,ip,iψ,iF] = ψo_oe[i,iψ];
                ψe_ogce_oe[i,ip,iψ,iF] = ψe_oe[i,iψ];
            end
            pf_ogce = [po[:,ip,iψ,iF] pe[:,ip,iψ,iF] pc[:,ip,iψ,iF] pg[:,ip,iψ,iF]];
            ψf_ogce = [ψo[:,ip,iψ,iF] ψe[:,ip,iψ,iF] ψc[:,ip,iψ,iF] ψg[:,ip,iψ,iF]];
            pf_oe = [po_ogce_oe[:,ip,iψ,iF] pe_ogce_oe[:,ip,iψ,iF]];
            ψf_oe = [ψo_ogce_oe[:,ip,iψ,iF] ψe_ogce_oe[:,ip,iψ,iF]];
            if iF == 1
                F = 12;
            elseif iF == 2
                F = 1234;
            end
            MCE_subm[ip,iψ,iF] = MCEnergy_submodel(pf_ogce,pf_oe,ψf_ogce,ψf_oe,F,Nogce,Noe,p);
        end 
        CF_ogce = (1/3)*(MCE_subm[1,1,2]-MCE_subm[1,1,1]) + (1/6)*(MCE_subm[2,1,2]-MCE_subm[2,1,1]) + (1/6)*(MCE_subm[1,2,2]-MCE_subm[1,2,1]) + (1/3)*(MCE_subm[2,2,2]-MCE_subm[2,2,1]);
        Cψ_ogce = (1/3)*(MCE_subm[1,2,1]-MCE_subm[1,1,1]) + (1/6)*(MCE_subm[2,2,1]-MCE_subm[2,1,1]) + (1/6)*(MCE_subm[1,2,2]-MCE_subm[1,1,2]) + (1/3)*(MCE_subm[2,2,2]-MCE_subm[2,1,2]);
        Cp_ogce = (1/3)*(MCE_subm[2,1,1]-MCE_subm[1,1,1]) + (1/6)*(MCE_subm[2,2,1]-MCE_subm[1,2,1]) + (1/6)*(MCE_subm[2,1,2]-MCE_subm[1,1,2]) + (1/3)*(MCE_subm[2,2,2]-MCE_subm[1,2,2]);
        # Total difference
        C_ogce = mean(pE_ogce)-mean(pE_oe);
        # C_ogce = median(pE_ogce)-median(pE_oe);
    #
#

# Write Table 
    # Save all elements of the table 
        global TotDiff_oce = "$(round(C_oce*1000,digits=2))"
        global TotDiff_oge = "$(round(C_oge*1000,digits=2))"
        global TotDiff_ogce = "$(round(C_ogce*1000,digits=2))"
        global TotDiff_oce_perc = "$(round(C_oce/mean(pE_oe)*100,digits=2))"
        global TotDiff_oge_perc = "$(round(C_oge/mean(pE_oe)*100,digits=2))"
        global TotDiff_ogce_perc = "$(round(C_ogce/mean(pE_oe)*100,digits=2))"

        global Diff_F_oce = "$(round(CF_oce*1000,digits=2))"
        global Diff_F_oge = "$(round(CF_oge*1000,digits=2))"
        global Diff_F_ogce = "$(round(CF_ogce*1000,digits=2))"
        global Diff_F_oce_perc = "$(round(CF_oce/C_oce*100,digits=2))"
        global Diff_F_oge_perc = "$(round(CF_oge/C_oge*100,digits=2))"
        global Diff_F_ogce_perc = "$(round(CF_ogce/C_ogce*100,digits=2))"

        global Diff_p_oce = "$(round(Cp_oce*1000,digits=2))"
        global Diff_p_oge = "$(round(Cp_oge*1000,digits=2))"
        global Diff_p_ogce = "$(round(Cp_ogce*1000,digits=2))"
        global Diff_p_oce_perc = "$(round(Cp_oce/C_oce*100,digits=2))"
        global Diff_p_oge_perc = "$(round(Cp_oge/C_oge*100,digits=2))"
        global Diff_p_ogce_perc = "$(round(Cp_ogce/C_ogce*100,digits=2))"

        global Diff_ψ_oce = "$(round(Cψ_oce*1000,digits=2))"
        global Diff_ψ_oge = "$(round(Cψ_oge*1000,digits=2))"
        global Diff_ψ_ogce = "$(round(Cψ_ogce*1000,digits=2))"
        global Diff_ψ_oce_perc = "$(round(Cψ_oce/C_oce*100,digits=2))"
        global Diff_ψ_oge_perc = "$(round(Cψ_oge/C_oge*100,digits=2))"
        global Diff_ψ_ogce_perc = "$(round(Cψ_ogce/C_ogce*100,digits=2))"

        # global Diff_F_oce_sel = "$(round(CF_oce_sel*1000,digits=2))"
        # global Diff_F_oge_sel = "$(round(CF_oge_sel*1000,digits=2))"
        # global Diff_F_ogce_sel = "$(round(CF_ogce_sel*1000,digits=2))"
        # global Diff_F_oce_perc_sel = "$(round(CF_oce_sel/C_oce*100,digits=2))"
        # global Diff_F_oge_perc_sel = "$(round(CF_oge_sel/C_oge*100,digits=2))"
        # global Diff_F_ogce_perc_sel = "$(round(CF_ogce_sel/C_ogce*100,digits=2))"

        # global Diff_p_oce_sel = "$(round(Cp_oce_sel*1000,digits=2))"
        # global Diff_p_oge_sel = "$(round(Cp_oge_sel*1000,digits=2))"
        # global Diff_p_ogce_sel = "$(round(Cp_ogce_sel*1000,digits=2))"
        # global Diff_p_oce_perc_sel = "$(round(Cp_oce_sel/C_oce*100,digits=2))"
        # global Diff_p_oge_perc_sel = "$(round(Cp_oge_sel/C_oge*100,digits=2))"
        # global Diff_p_ogce_perc_sel = "$(round(Cp_ogce_sel/C_ogce*100,digits=2))"

        # global Diff_ψ_oce_sel = "$(round(Cψ_oce_sel*1000,digits=2))"
        # global Diff_ψ_oge_sel = "$(round(Cψ_oge_sel*1000,digits=2))"
        # global Diff_ψ_ogce_sel = "$(round(Cψ_ogce_sel*1000,digits=2))"
        # global Diff_ψ_oce_perc_sel = "$(round(Cψ_oce_sel/C_oce*100,digits=2))"
        # global Diff_ψ_oge_perc_sel = "$(round(Cψ_oge_sel/C_oge*100,digits=2))"
        # global Diff_ψ_ogce_perc_sel = "$(round(Cψ_ogce_sel/C_ogce*100,digits=2))"
    #
    # Make Table (baseline)
        tex_table = """
        \\begin{tabular}{@{}llccc@{}}
        \\toprule\\hline
        \\multicolumn{2}{l}{} & OCE & OGE & OGCE \\\\ \\midrule
        Total Difference & Percent (\\%) Difference with OE & $TotDiff_oce_perc & $TotDiff_oge_perc & $TotDiff_ogce_perc  \\\\ \\midrule
        Option Value & \\multirow{3}{*}{Percent (\\%) of Total} & $Diff_F_oce_perc & $Diff_F_oge_perc & $Diff_F_ogce_perc  \\\\ 
        Fuel Productivity &  & $Diff_ψ_oce_perc & $Diff_ψ_oge_perc & $Diff_ψ_ogce_perc  \\\\ 
        Fuel Prices & & $Diff_p_oce_perc & $Diff_p_oge_perc & $Diff_p_ogce_perc  \\\\ \\hline\\bottomrule
        \\end{tabular}
        """
        fname = "ShapleyDecomp_MCE.tex"
        dirpath = "Counterfactuals/Tables"
        fpath = joinpath(dirpath,fname);
        open(fpath, "w") do file
            write(file, tex_table)
        end
    #
    # Make Table (both accounting for selection and not)
        tex_table = """
        \\begin{tabular}{@{}llllllll@{}}
        \\toprule\\hline
        \\multicolumn{2}{l}{\\multirow{2}{*}{}} & \\multicolumn{3}{l}{\\begin{tabular}[c]{@{}l@{}}Accounting for Selection \\\\ in Comparative adv. of gas and coal\\end{tabular}} & \\multicolumn{3}{l}{\\begin{tabular}[c]{@{}l@{}}Not accounting for Selection \\\\ in Comparative adv. of gas and coal\\end{tabular}} \\\\ \\cmidrule(l){3-8} 
        \\multicolumn{2}{l}{} & OCE & OGE & OGCE & OCE & OGE & OGCE \\\\ \\midrule
        \\multirow{2}{*}{Total Difference} & Level & $TotDiff_oce & $TotDiff_oge & $TotDiff_ogce & $TotDiff_oce & $TotDiff_oge & $TotDiff_ogce \\\\
        & Percent (\\%) Difference & $TotDiff_oce_perc & $TotDiff_oge_perc & $TotDiff_ogce_perc & $TotDiff_oce_perc & $TotDiff_oge_perc & $TotDiff_ogce_perc \\\\ \\midrule
        \\multirow{2}{*}{Gains from Variety} & Level & $Diff_F_oce & $Diff_F_oge & $Diff_F_ogce & $Diff_F_oce_sel & $Diff_F_oge_sel & $Diff_F_ogce_sel \\\\
        & Percent (\\%)  of Total & $Diff_F_oce_perc & $Diff_F_oge_perc & $Diff_F_ogce_perc & $Diff_F_oce_perc_sel & $Diff_F_oge_perc_sel & $Diff_F_ogce_perc_sel \\\\ \\midrule
        \\multirow{2}{*}{Fuel Productivity} & Level & $Diff_ψ_oce & $Diff_ψ_oge & $Diff_ψ_ogce & $Diff_ψ_oce_sel & $Diff_ψ_oge_sel & $Diff_ψ_ogce_sel \\\\
        & Percent (\\%)  of Total & $Diff_ψ_oce_perc & $Diff_ψ_oge_perc & $Diff_ψ_ogce_perc & $Diff_ψ_oce_perc_sel & $Diff_ψ_oge_perc_sel & $Diff_ψ_ogce_perc_sel \\\\ \\midrule
        \\multirow{2}{*}{Fuel Prices} & Level & $Diff_p_oce & $Diff_p_oge & $Diff_p_ogce & $Diff_p_oce_sel & $Diff_p_oge_sel & $Diff_p_ogce_sel \\\\
        & Percent (\\%)  of Total & $Diff_p_oce_perc & $Diff_p_oge_perc & $Diff_p_ogce_perc & $Diff_p_oce_perc_sel & $Diff_p_oge_perc_sel & $Diff_p_ogce_perc_sel \\\\ \\hline\\bottomrule
        \\end{tabular}
        """
        fname = "ShapleyDecomp_MCE.tex"
        dirpath = "Counterfactuals/Tables"
        fpath = joinpath(dirpath,fname);
        open(fpath, "w") do file
            write(file, tex_table)
        end
    #
#














