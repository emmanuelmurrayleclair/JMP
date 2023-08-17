# Policy counterfactual of dynamic discrete choice model with real data
    # Steel manufacturing
    # With selection on unobservables
    # Monopolistic Competition
    # Carbon tax with fixed cost subsidy
# Emmanuel Murray Leclair
# June 2023

# Distributed computing
using Distributed, SharedArrays 
addprocs(Sys.CPU_THREADS-2) ;

## Home directory
computer = gethostname() ;
@everywhere cd("/project/6001227/emurrayl/DynamicDiscreteChoice") # Sharcnet
sharcnet = true;

@everywhere empty!(DEPOT_PATH)
@everywhere push!(DEPOT_PATH, "/project/6001227/emurrayl/julia")

## Make auxiliary directores
@everywhere Fig_Folder = "/project/6001227/emurrayl/DynamicDiscreteChoice/Figures"
@everywhere mkpath(Fig_Folder)
@everywhere Result_Folder = "/project/6001227/emurrayl/DynamicDiscreteChoice/Results"
@everywhere mkpath(Result_Folder)

# Load packages
# using CUDA_Runtime_jll, CUDA
@everywhere using SparseArrays, Interpolations, Dierckx, ForwardDiff, Optim, Roots, Parameters, Kronecker, Plots, StatsPlots, NLopt, Distributions, QuantEcon, HDF5
@everywhere using CSV, DataFrames, DiscreteMarkovChains, StructTypes, StatsBase, Distributed, SharedArrays, DelimitedFiles, NLsolve, ParallelDataTransfer
@everywhere using FiniteDiff, BenchmarkTools, Distances, JLD2, FileIO
#plotlyjs()

# Load externally written functions
@everywhere include("/project/6001227/emurrayl/DynamicDiscreteChoice/VFI_Toolbox.jl")

println(" ")
println("-----------------------------------------------------------------------------------------")
println("Dynamic production and fuel set choice in Julia - Steel Manufacturing Data With Selection")
println("                    Counterfactuals - Model comparison (No Switching)                    ")
println("-----------------------------------------------------------------------------------------")
println(" ")

#-----------------------------------------------------------
#-----------------------------------------------------------
# 1. Parameters, Model and Data Structure

# Generate structure for parameters using Parameters module
# Import Parameters from Steel manufacturing data (with and without fuel productivity)
    # With fuel productivity (full model)
    @everywhere @with_kw struct Par
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
            # lnc_re_grid = [-1.0864664729029823,-0.006641199999999974,1.0731840729029822];
            # lng_re_grid = [-1.3233458340099815,0.024497739999999858,1.3723413140099814];
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
        pout_tol::Float64   = 1E-15  ; # Tolerance for fixed point in aggregate output price index
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
    # With energy productivity only (different estimation of energy prod function)
    @everywhere @with_kw struct Par_Eprod
        p_all = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_Eprod_all.txt", DataFrame)[:,2:(end-1)]          ;
        p_cov = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_Eprod_cov.txt", DataFrame)[:,2:(end-1)]          ;
        p_cov_g = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_Eprod_cov_g.txt", DataFrame)[:,2:(end-1)]      ;
        p_cov_c = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_Eprod_cov_c.txt", DataFrame)[:,2:(end-1)]      ;
        p_cov_gc = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_Eprod_cov_gc.txt", DataFrame)[:,2:(end-1)]    ;
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
        # State transition: productivity of energy ψE
            ρ_ψE=p_all[3,1];
            σ_ψE=p_all[3,2];
            μψE=p_all[3,3];
            μψE_t = μψE .+ [0,p_all[3,4],p_all[3,5],p_all[3,6],p_all[3,7],p_all[3,8],p_all[3,9]];
        #
        # State transition: price of electricity pe
            ρ_pe=p_all[4,1];
            σ_pe=p_all[4,2];
            μpe=p_all[4,3];
            μpe_t = μpe .+ [0,p_all[4,4],p_all[4,5],p_all[4,6],p_all[4,7],p_all[4,8],p_all[4,9]];
        #
        # Price of oil
            po = [p_all[5,1],p_all[5,2],p_all[5,3],p_all[5,4],p_all[5,5],p_all[5,6],p_all[5,7],p_all[5,8]];
        #
        # State transition: price of gas (no RE)
            σ_pg=p_all[6,1];
            μpg=p_all[6,2];
            μpg_t = μpg .+ [0,p_all[6,3],p_all[6,4],p_all[6,5],p_all[6,6],p_all[6,7],p_all[6,8],p_all[6,9]];
        #
        # State transition: price of coal (no RE)
            σ_pc=p_all[7,1];
            μpc=p_all[7,2];
            μpc_t = μpc .+ [0,p_all[7,3],p_all[7,4],p_all[7,5],p_all[7,6],p_all[7,7],p_all[7,8],p_all[7,9]];
        #
        # State transition: price of materials
            ρ_pm=p_all[8,1];
            σ_pm=p_all[8,2];
            μpm=p_all[8,3];
            μpm_t = μpm .+ [0,p_all[8,4],p_all[8,5],p_all[8,6],p_all[8,7],p_all[8,8],p_all[8,9]];
        #
        # Price of materials (used if not a state variable)
            pm = [p_all[9,1],p_all[9,2],p_all[9,3],p_all[9,4],p_all[9,5],p_all[9,6],p_all[9,7],p_all[9,8]];
        #
        # Wages, rental rate of capital, output prices and emission factors
            w =  [p_all[10,1],p_all[10,2],p_all[10,3],p_all[10,4],p_all[10,5],p_all[10,6],p_all[10,7],p_all[10,8]];
            rk = [p_all[11,1],p_all[11,2],p_all[11,3],p_all[11,4],p_all[11,5],p_all[11,6],p_all[11,7],p_all[11,8]];
            pout_struc = [p_all[12,1],p_all[12,2],p_all[12,3],p_all[12,4],p_all[12,5],p_all[12,6],p_all[12,7],p_all[12,8]];
            pout_init = [p_all[12,1],p_all[12,2],p_all[12,3],p_all[12,4],p_all[12,5],p_all[12,6],p_all[12,7],p_all[12,8]];
            Ygmean = [p_all[13,1],p_all[13,2],p_all[13,3],p_all[13,4],p_all[13,5],p_all[13,6],p_all[13,7],p_all[13,8]];
            γ = [p_all[14,1],p_all[14,2],p_all[14,3],p_all[14,4]]; # In order: electricity, natural gas, coal and oil
            γe = γ[1];  # 1 mmBtu of electricity to metric ton of co2e 
            γg = γ[2];  # 1 mmBtu of gas to metric ton of co2e   
            γc = γ[3];  # 1 mmBtu of coal to metric ton of co2e 
            γo = γ[4];  # 1 mmBtu of oil to metric ton of co2e 
        #
        # Geometric mean of fuel quantities 
            fgmean = [p_all[15,1],p_all[15,2],p_all[15,3],p_all[15,4]];
            ogmean = fgmean[1];
            ggmean = fgmean[2];
            cgmean = fgmean[3];
            egmean = fgmean[4];
        #
        # Demand parameters
            ρ=p_all[16,1];
            θ=p_all[16,2];
            d_t = [0,p_all[16,4],p_all[16,5],p_all[16,6],p_all[16,7],p_all[16,8],p_all[16,9]];
        #
        # Number of plants by year
            N_t = [0,p_all[17,2],p_all[17,3],p_all[17,4],p_all[17,5],p_all[17,6],p_all[17,7]];
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
        pout_tol::Float64   = 1E-15  ; # Tolerance for fixed point in aggregate output price index
        υ                   = 0.0    ; # Dampen factor
        # misc 
        F_tot               = 4      ; # Total number of fuels
        β                   = 0.9    ; # Discount factor
        T_pers              = size(μψE_t,1);    # number of years, persistent variables
        T_nonpers           = size(po,1);       # Number of years, non persistent variables
        T                   = T_pers-1;         # Actual number of years where I kept data
        Tf                  = 40      ;         # Number of years of forward simulation
        S                   = 50      ;         # Number of simulations for CCP
    end
    p_Eprod = Par_Eprod();
    # Without fuel productivity (but otherwise same as full model) 
    @everywhere @with_kw struct Par_nfp
        p_all = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_nofprod_all.txt", DataFrame)[:,2:(end-1)]          ;
        p_cov = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_nofprod_cov.txt", DataFrame)[:,2:(end-1)]          ;
        p_cov_g = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_nofprod_cov_g.txt", DataFrame)[:,2:(end-1)]      ;
        p_cov_c = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_nofprod_cov_c.txt", DataFrame)[:,2:(end-1)]      ;
        p_cov_gc = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_nofprod_cov_gc.txt", DataFrame)[:,2:(end-1)]    ;
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
            Κ = fc_est[1:6];
            σunit = fc_est[7];
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
        pout_tol::Float64   = 1E-15  ; # Tolerance for fixed point in aggregate output price index
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
    p_nfp = Par_nfp();
    # Without fuel productivity (simple nested CES, new estimation) 
    @everywhere @with_kw struct Par_simpleCES
        p_all = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_simpleCES_all.txt", DataFrame)[:,2:(end-1)]          ;
        p_cov = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_simpleCES_cov.txt", DataFrame)[:,2:(end-1)]          ;
        p_cov_g = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_simpleCES_cov_g.txt", DataFrame)[:,2:(end-1)]      ;
        p_cov_c = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_simpleCES_cov_c.txt", DataFrame)[:,2:(end-1)]      ;
        p_cov_gc = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/param_simpleCES_cov_gc.txt", DataFrame)[:,2:(end-1)]    ;
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
        # State transition: price of electricity pe
            ρ_pe=p_all[3,1];
            σ_pe=p_all[3,2];
            μpe=p_all[3,3];
            μpe_t = μpe .+ [0,p_all[4,4],p_all[4,5],p_all[4,6],p_all[4,7],p_all[4,8],p_all[4,9]];
        #
        # Price of oil
            po = [p_all[4,1],p_all[4,2],p_all[4,3],p_all[4,4],p_all[4,5],p_all[4,6],p_all[4,7],p_all[4,8]];
        #
        # State transition: price of gas (no RE)
            σ_pg=p_all[5,1];
            μpg=p_all[5,2];
            μpg_t = μpg .+ [0,p_all[5,3],p_all[5,4],p_all[5,5],p_all[5,6],p_all[5,7],p_all[5,8],p_all[5,9]];
        #
        # State transition: price of coal (no RE)
            σ_pc=p_all[6,1];
            μpc=p_all[6,2];
            μpc_t = μpc .+ [0,p_all[6,3],p_all[6,4],p_all[6,5],p_all[6,6],p_all[6,7],p_all[6,8],p_all[6,9]];
        #
        # State transition: price of materials
            ρ_pm=p_all[7,1];
            σ_pm=p_all[7,2];
            μpm=p_all[7,3];
            μpm_t = μpm .+ [0,p_all[7,4],p_all[7,5],p_all[7,6],p_all[7,7],p_all[7,8],p_all[7,9]];
        #
        # Price of materials (used if not a state variable)
            pm = [p_all[8,1],p_all[8,2],p_all[8,3],p_all[8,4],p_all[8,5],p_all[8,6],p_all[8,7],p_all[8,8]];
        #
        # Wages, rental rate of capital, output prices and emission factors
            w =  [p_all[9,1],p_all[9,2],p_all[9,3],p_all[9,4],p_all[9,5],p_all[9,6],p_all[9,7],p_all[9,8]];
            rk = [p_all[10,1],p_all[10,2],p_all[10,3],p_all[10,4],p_all[10,5],p_all[10,6],p_all[10,7],p_all[10,8]];
            pout_struc = [p_all[11,1],p_all[11,2],p_all[11,3],p_all[11,4],p_all[11,5],p_all[11,6],p_all[11,7],p_all[11,8]];
            pout_init = [p_all[11,1],p_all[11,2],p_all[11,3],p_all[11,4],p_all[11,5],p_all[11,6],p_all[11,7],p_all[11,8]];
            Ygmean = [p_all[12,1],p_all[12,2],p_all[12,3],p_all[12,4],p_all[12,5],p_all[12,6],p_all[12,7],p_all[12,8]];
            γ = [p_all[13,1],p_all[13,2],p_all[13,3],p_all[13,4]]; # In order: electricity, natural gas, coal and oil
            γe = γ[1];  # 1 mmBtu of electricity to metric ton of co2e 
            γg = γ[2];  # 1 mmBtu of gas to metric ton of co2e   
            γc = γ[3];  # 1 mmBtu of coal to metric ton of co2e 
            γo = γ[4];  # 1 mmBtu of oil to metric ton of co2e 
        #
        # Geometric mean of fuel quantities 
            fgmean = [p_all[14,1],p_all[14,2],p_all[14,3],p_all[14,4]];
            ogmean = fgmean[1];
            ggmean = fgmean[2];
            cgmean = fgmean[3];
            egmean = fgmean[4];
        #
        # Demand parameters
            ρ=p_all[15,1];
            θ=p_all[15,2];
            d_t = [0,p_all[15,4],p_all[15,5],p_all[15,6],p_all[15,7],p_all[15,8],p_all[15,9]];
        #
        # Number of plants by year
            N_t = [0,p_all[16,2],p_all[16,3],p_all[16,4],p_all[16,5],p_all[16,6],p_all[16,7]];
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
        pout_tol::Float64   = 1E-15  ; # Tolerance for fixed point in aggregate output price index
        υ                   = 0.0    ; # Dampen factor
        # misc 
        F_tot               = 4      ; # Total number of fuels
        β                   = 0.9    ; # Discount factor
        T_pers              = size(μz_t,1);    # number of years, persistent variables
        T_nonpers           = size(po,1);       # Number of years, non persistent variables
        T                   = T_pers-1;         # Actual number of years where I kept data
        Tf                  = 40      ;         # Number of years of forward simulation
        S                   = 50      ;         # Number of simulations for CCP
    end
    p_simpleCES = Par_simpleCES();
#

# Import the data
    # With fuel productivity
    @everywhere Data = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/MainData_wPipeline-Steel.csv", DataFrame) ;
    @everywhere Data[!,:id] = 1:size(Data,1);
    @everywhere Data_year = groupby(Data,:year);
    # Without fuel productivity (same model)
    @everywhere Data_nfp = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/MainData_wPipeline_nofprod-Steel.csv", DataFrame) ;
    @everywhere Data_nfp[!,:id] = 1:size(Data_nfp,1);
    @everywhere Data_nfp_year = groupby(Data_nfp,:year);
    # With energy productivity only (different estimation of energy prod function)
    @everywhere Data_Eprod = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/MainData_wPipeline_Eprod-Steel.csv", DataFrame) ;
    @everywhere Data_Eprod[!,:id] = 1:size(Data_Eprod,1);
    @everywhere Data_Eprod_year = groupby(Data_Eprod,:year);
    # Without fuel productivity (simple nested CES, new estimation) 
    @everywhere Data_simpleCES = CSV.read("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Data/MainData_wPipeline_simpleCES-Steel.csv", DataFrame) ;
    @everywhere Data_simpleCES[!,:id] = 1:size(Data_simpleCES,1);
    @everywhere Data_simpleCES_year = groupby(Data_simpleCES,:year);
#

# Generate structure of model objects
    # Model : Dynamic production and fuel set choice - With Selection and Fuel Productivity
    @everywhere @with_kw struct Model
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
        # πgrid_oe = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                      ;
        # πgrid_oge = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                      ;
        # πgrid_oce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                      ;
        # πgrid_ogce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                      ;
        # # Fuel quantities
        # fqty_oe = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                 ;
        #     fqty_oe_norm = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)        ;
        # fqty_oge = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                ;
        #     fqty_oge_norm = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)       ;
        # fqty_oce = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                ;
        #     fqty_oce_norm = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)       ;
        # fqty_ogce = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)               ;
        #     fqty_ogce_norm = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)      ;

        # # Expected Value Functions (single agent)
        # V_itermax = 1000                                                                            ;
        # nconnect=2;            # Number of possible pipeline connections (connected and not connected)                                                        ;
        # Wind = Array{Float64}(undef,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                     ; 
        # Wind1 = Array{Float64}(undef,p.F_tot,ngrid^nstate,n_c,n_g,nc_re,ng_re)                        ; 
        # Wind0 = Array{Float64}(undef,(ngrid^nstate)*n_c*n_g,p.F_tot)                                ;
        # Wind0_full = Array{Float64}(undef,(ngrid^nstate)*n_c*n_g,p.F_tot,p.T*nc_re*ng_re)              ;
        # Wind0_year = Array{Float64}(undef,(ngrid^nstate)*n_c*n_g,p.F_tot,p.T)                         ;
        # W_new = Array{Float64}(undef,nconnect,p.F_tot,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                     ; 
        # # Choice-specific value function  
        # vgrid_oe = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                       ;
        # vgrid_oge = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                      ;
        # vgrid_oce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                      ;
        # vgrid_ogce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                     ;
        # # Fuel prices
        # pEgrid_oe = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                   ;
        # pEgrid_oge = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                  ;
        # pEgrid_oce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                  ;
        # pEgrid_ogce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                 ;
    end
    M = Model();
    # Model : Dynamic production and fuel set choice - No fuel productivity (same as main model)
    @everywhere @with_kw struct Model_nfp
        # Parameters
        p::Par_nfp = Par_nfp(); # Model parameters in their own structure
        ng_re = 3;
        nc_re = 3;
        N = size(Data_nfp,1); # Total number of observations
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
    end
    M_nfp = Model_nfp();
    # Model : Dynamic production and fuel set choice - Energy productivity
    @everywhere @with_kw struct Model_Eprod
        # Parameters
        p::Par_Eprod = Par_Eprod(); # Model parameters in their own structure
        N = size(Data_Eprod,1); # Total number of observations
        # All forward simulation draws
        z_fs = Array{Float64}(undef,N,p.S,p.Tf)                 ;
            shock_z = Array{Float64}(undef,N,p.S,p.Tf)          ;  # Simulation draw: shock to z
        pm_fs = Array{Float64}(undef,N,p.S,p.Tf)                ;
            shock_pm = Array{Float64}(undef,N,p.S,p.Tf)         ;  # Simulation draw: shock to pm
        pe_fs = Array{Float64}(undef,N,p.S,p.Tf)                ;
            shock_pe = Array{Float64}(undef,N,p.S,p.Tf)         ;  # Simulation draw: shock to pe
        ψE_fs = Array{Float64}(undef,N,p.S,p.Tf)                ;
            shock_ψE = Array{Float64}(undef,N,p.S,p.Tf)         ;  # Simulation draw: shock to ψE
        pg_fs = Array{Float64}(undef,N,p.S,p.Tf)                ;  # Simulation draw: price of gas
            shock_pg = Array{Float64}(undef,N,p.S,p.Tf)         ;  # Simulation draw: shock to pg
        pc_fs = Array{Float64}(undef,N,p.S,p.Tf)                ;  # Simulation draw: price of coal
            shock_pc = Array{Float64}(undef,N,p.S,p.Tf)         ;  # Simulation draw: shock to pc
    end
    M_Eprod = Model_Eprod();
    # Model : Dynamic production and fuel set choice - No fuel productivity (re-estimate simple nested CES)
    @everywhere @with_kw struct Model_simpleCES
        # Parameters
        p::Par_simpleCES = Par_simpleCES(); # Model parameters in their own structure
        ng_re = 3;
        nc_re = 3;
        N = size(Data_simpleCES,1); # Total number of observations
        # All forward simulation draws
        z_fs = Array{Float64}(undef,N,p.S,p.Tf)                 ;
            shock_z = Array{Float64}(undef,N,p.S,p.Tf)          ;  # Simulation draw: shock to z
        pm_fs = Array{Float64}(undef,N,p.S,p.Tf)                ;
            shock_pm = Array{Float64}(undef,N,p.S,p.Tf)         ;  # Simulation draw: shock to pm
        pe_fs = Array{Float64}(undef,N,p.S,p.Tf)                ;
            shock_pe = Array{Float64}(undef,N,p.S,p.Tf)         ;  # Simulation draw: shock to pe
        pg_fs = Array{Float64}(undef,N,p.S,p.Tf)                ;  # Simulation draw: price of gas
            shock_pg = Array{Float64}(undef,N,p.S,p.Tf)         ;  # Simulation draw: shock to pg
        pc_fs = Array{Float64}(undef,N,p.S,p.Tf)                ;  # Simulation draw: price of coal
            shock_pc = Array{Float64}(undef,N,p.S,p.Tf)         ;  # Simulation draw: shock to pc
    end
    M_simpleCES = Model_simpleCES();
#

### Store of expected value function
    @everywhere mutable struct VF_bench
        W;
    end
#

# Include all relevant functions to current script
    @everywhere include("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Simulation_func.jl");
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
    # With fuel productivity
        # Main model 
            @everywhere τnotax = [0,0,0,0];
            p,M = OutputPrice_func_nogrid(M,Data,p,τnotax);
        #
        # Energy Productivity 
            @everywhere function OutputPrice_func_Eprod(M::Model_Eprod,Data::DataFrame,p::Par_Eprod,τ)
                # This function does fixed point iteration to find the aggregate output price index
                @unpack T,η,ρ,σ,pout_struc,N_t = p;
                τg = τ[1]*p.ggmean;
                τc = τ[2]*p.cgmean;
                τo = τ[3]*p.ogmean;
                τe = τ[4]*p.egmean;
                N = size(Data,1);
                Data_year = groupby(Data,:year);
                pout_struc = copy(p.pout_struc);
                # Initial guess and initial distance
                pout_agg_init = copy(p.pout_init);
                dist = 100;
                #println("Current distance = $dist");
                pout_agg_old = copy(pout_agg_init);
                pout_agg_new = copy(pout_agg_init);
                iter = 1;
                while dist > p.pout_tol 
                    for t = 1:T
                        # Get predicted individual output given state variables and current guess of aggregate price index
                        NT = size(Data_year[t],1);
                        pout = Array{Float64}(undef,NT);
                        rk = p.rk[t+1];
                        w = p.w[t+1];
                        for i = 1:NT
                            ii = Data_year[t].id[i];
                            lnz = Data.lnz[ii]; z = exp(lnz);
                            lnpm = Data.logPm[ii]; pm = exp(lnpm);
                            lnpe = log(exp(Data.lnpelec_tilde[ii])+τe); pe = exp(lnpe);
                            lnψE = Data.lnfprod_E[ii]; ψE = exp(lnψE);
                            pe = pe;
                            po = (p.po[t+1]+τo);
                            if Data.combineF[ii] == 12
                                pf = [po,pe];
                                pE =  (1/ψE)*pE_func(12,pf,p);
                            elseif Data.combineF[ii] == 123
                                pc = exp(Data.lnpc_tilde[ii]) + τc;
                                pc = pc;
                                pf = [po,pe,pc];
                                pE = (1/ψE)*pE_func(123,pf,p);
                            elseif Data.combineF[ii] == 124
                                pg = exp(Data.lnpg_tilde[ii]) + τg;
                                pg = pg;
                                pf = [po,pe,pg];
                                pE = (1/ψE)*pE_func(124,pf,p);
                            elseif Data.combineF[ii] == 1234
                                pc = exp(Data.lnpc_tilde[ii]) + τc;
                                pc = pc;
                                pg = exp(Data.lnpg_tilde[ii]) + τg;
                                pg = pg;
                                pf = [po,pe,pc,pg];
                                pE = (1/ψE)*pE_func(1234,pf,p);
                            end
                            pind = pinput_func(rk,pm,w,pE,p);
                            Y = output_func_monopolistic(z,pind,pout_agg_old[t+1],t,p);
                            pout[i] = outprice_func_monopolistic_ind(Y,pout_agg_old[t+1],t,p);
                        end
                        # Get new aggregate output price
                        pout_agg_new[t+1] = ((1/N_t[t+1])*sum(pout.^(1-ρ)))^(1/(1-ρ));
                    end
                    iter += 1;
                    if iter >= 10000
                        error("10,000 iterations reached, Output price did not converge")
                    end
                    # Check for convergence
                    #println("Current distance = $(sqrt(sum((pout_agg_new-pout_agg_old).^2)))");
                    dist = sqrt(sum((pout_agg_new-pout_agg_old).^2));
                    if mod(iter,100)==0
                        println("aggregate output price iter = $iter, distance = $dist")
                    end
                    # Update for new iteration
                    pout_agg_old = copy(pout_agg_new);
                end
                #println("output price converged after $iter iterations")
                # Update output price index to parameter struc
                p = Par_Eprod(p; pout_struc = copy(pout_agg_new));
                M = Model_Eprod(M; p = p);
                return p,M;
            end
            p_Eprod,M_Eprod = OutputPrice_func_Eprod(M_Eprod,Data_Eprod,p_Eprod,τnotax);
        #
        # No fuel productivity (main model otherwise unchanged)
            @everywhere function OutputPrice_func_nfp(M::Model_nfp,Data::DataFrame,p::Par_nfp,τ)
                # This function does fixed point iteration to find the aggregate output price index
                @unpack T,η,ρ,σ,pout_struc,N_t = p;
                τg = τ[1]*p.ggmean;
                τc = τ[2]*p.cgmean;
                τo = τ[3]*p.ogmean;
                τe = τ[4]*p.egmean;
                N = size(Data,1);
                Data_year = groupby(Data,:year);
                pout_struc = copy(p.pout_struc);
                # Initial guess and initial distance
                pout_agg_init = copy(p.pout_init);
                dist = 100;
                #println("Current distance = $dist");
                pout_agg_old = copy(pout_agg_init);
                pout_agg_new = copy(pout_agg_init);
                iter = 1;
                ψe = exp(mean(Data.lnfprod_e));
                ψo = exp(mean(Data.lnfprod_o));
                ψg = exp(mean(Data.lnfprod_g[Data.gas.>0]));
                ψc = exp(mean(Data.lnfprod_c[Data.coal.>0]));
                while dist > p.pout_tol 
                    for t = 1:T
                        # Get predicted individual output given state variables and current guess of aggregate price index
                        NT = size(Data_year[t],1);
                        pout = Array{Float64}(undef,NT);
                        rk = p.rk[t+1];
                        w = p.w[t+1];
                        for i = 1:NT
                            ii = Data_year[t].id[i];
                            lnz = Data.lnz[ii]; z = exp(lnz);
                            lnpm = Data.logPm[ii]; pm = exp(lnpm);
                            lnpe = log(exp(Data.lnpelec_tilde[ii])+τe); pe = exp(lnpe);
                            peψe = pe/ψe;
                            poψo = (p.po[t+1]+τo)/ψo;
                            if Data.combineF[ii] == 12
                                pfψf = [poψo,peψe];
                                pE =  pE_func(12,pfψf,p);
                            elseif Data.combineF[ii] == 123
                                pc = exp(Data.lnpc_tilde[ii]) + τc;
                                pcψc = pc/ψc;
                                pfψf = [poψo,peψe,pcψc];
                                pE = pE_func(123,pfψf,p);
                            elseif Data.combineF[ii] == 124
                                pg = exp(Data.lnpg_tilde[ii]) + τg;
                                pgψg = pg/ψg;
                                pfψf = [poψo,peψe,pgψg];
                                pE = pE_func(124,pfψf,p);
                            elseif Data.combineF[ii] == 1234
                                pc = exp(Data.lnpc_tilde[ii]) + τc;
                                pcψc = pc/ψc;
                                pg = exp(Data.lnpg_tilde[ii]) + τg;
                                pgψg = pg/ψg;
                                pfψf = [poψo,peψe,pcψc,pgψg];
                                pE = pE_func(1234,pfψf,p);
                            end
                            pind = pinput_func(rk,pm,w,pE,p);
                            Y = output_func_monopolistic(z,pind,pout_agg_old[t+1],t,p);
                            pout[i] = outprice_func_monopolistic_ind(Y,pout_agg_old[t+1],t,p);
                        end
                        # Get new aggregate output price
                        pout_agg_new[t+1] = ((1/N_t[t+1])*sum(pout.^(1-ρ)))^(1/(1-ρ));
                    end
                    iter += 1;
                    if iter >= 10000
                        error("10,000 iterations reached, Output price did not converge")
                    end
                    # Check for convergence
                    #println("Current distance = $(sqrt(sum((pout_agg_new-pout_agg_old).^2)))");
                    dist = sqrt(sum((pout_agg_new-pout_agg_old).^2));
                    if mod(iter,100)==0
                        println("aggregate output price iter = $iter, distance = $dist")
                    end
                    # Update for new iteration
                    pout_agg_old = copy(pout_agg_new);
                end
                #println("output price converged after $iter iterations")
                # Update output price index to parameter struc
                p = Par_nfp(p; pout_struc = copy(pout_agg_new));
                M = Model_nfp(M; p = p);
                return p,M;
            end
            p_nfp,M_nfp = OutputPrice_func_nfp(M_nfp,Data_nfp,p_nfp,τnotax);
        #
        # No fuel productivity (re-estimate simple nested CES)
            @everywhere function OutputPrice_func_simpleCES(M::Model_simpleCES,Data::DataFrame,p::Par_simpleCES,τ)
                # This function does fixed point iteration to find the aggregate output price index
                @unpack T,η,ρ,σ,pout_struc,N_t = p;
                # p = Par_simpleCES(p; σ=1.7999572,λ=2.2235012);
                τg = τ[1]*p.ggmean;
                τc = τ[2]*p.cgmean;
                τo = τ[3]*p.ogmean;
                τe = τ[4]*p.egmean;
                N = size(Data,1);
                Data_year = groupby(Data,:year);
                pout_struc = copy(p.pout_struc);
                # Initial guess and initial distance
                pout_agg_init = copy(p.pout_init);
                dist = 100;
                #println("Current distance = $dist");
                pout_agg_old = copy(pout_agg_init);
                pout_agg_new = copy(pout_agg_init);
                iter = 1;
                while dist > p.pout_tol 
                    for t = 1:T
                        # Get predicted individual output given state variables and current guess of aggregate price index
                        NT = size(Data_year[t],1);
                        pout = Array{Float64}(undef,NT);
                        rk = p.rk[t+1];
                        w = p.w[t+1];
                        for i = 1:NT
                            ii = Data_year[t].id[i];
                            lnz = Data.lnz[ii]; z = exp(lnz);
                            lnpm = Data.logPm[ii]; pm = exp(lnpm);
                            lnpe = log(exp(Data.lnpelec_tilde[ii])+τe); pe = exp(lnpe);
                            po = (p.po[t+1]+τo);
                            if Data.combineF[ii] == 12
                                pf = [po,pe];
                                pE =  pE_func(12,pf,p);
                            elseif Data.combineF[ii] == 123
                                pc = exp(Data.lnpc_tilde[ii]) + τc;
                                pc = pc;
                                pf = [po,pe,pc];
                                pE = pE_func(123,pf,p);
                            elseif Data.combineF[ii] == 124
                                pg = exp(Data.lnpg_tilde[ii]) + τg;
                                pg = pg;
                                pf = [po,pe,pg];
                                pE = pE_func(124,pf,p);
                            elseif Data.combineF[ii] == 1234
                                pc = exp(Data.lnpc_tilde[ii]) + τc;
                                pc = pc;
                                pg = exp(Data.lnpg_tilde[ii]) + τg;
                                pg = pg;
                                pf = [po,pe,pc,pg];
                                pE = pE_func(1234,pf,p);
                            end
                            pind = pinput_func(rk,pm,w,pE,p);
                            Y = output_func_monopolistic(z,pind,pout_agg_old[t+1],t,p);
                            pout[i] = outprice_func_monopolistic_ind(Y,pout_agg_old[t+1],t,p);
                        end
                        # Get new aggregate output price
                        pout_agg_new[t+1] = ((1/N_t[t+1])*sum(pout.^(1-ρ)))^(1/(1-ρ));
                    end
                    iter += 1;
                    if iter >= 10000
                        error("10,000 iterations reached, Output price did not converge")
                    end
                    # Check for convergence
                    #println("Current distance = $(sqrt(sum((pout_agg_new-pout_agg_old).^2)))");
                    dist = sqrt(sum((pout_agg_new-pout_agg_old).^2));
                    if mod(iter,100)==0
                        println("aggregate output price iter = $iter, distance = $dist")
                    end
                    # Update for new iteration
                    pout_agg_old = copy(pout_agg_new);
                end
                #println("output price converged after $iter iterations")
                # Update output price index to parameter struc
                p = Par_simpleCES(p; pout_struc = copy(pout_agg_new));
                M = Model_simpleCES(M; p = p);
                return p,M;
            end
            p_simpleCES,M_simpleCES = OutputPrice_func_simpleCES(M_simpleCES,Data_simpleCES,p_simpleCES,τnotax);
        #
    #
#

# Compute predicted profit under state variables in the data 
    # Full model
    πpred = StaticProfit_data(M,Data,τnotax);
    # Energy productivity 
    function StaticProfit_Eprod(M::Model_Eprod,Data::DataFrame,τ)
        @unpack p = M;
        τg = τ[1]*p.ggmean;
        τc = τ[2]*p.cgmean;
        τo = τ[3]*p.ogmean;
        τe = τ[4]*p.egmean;
        # Initialize static profit under grid points (order of states: z,pm,pe,ψe,ψo)
        N = size(Data,1);
        Data_year = groupby(Data,:year);
        πpred = Array{Float64}(undef,N);
        for t = 1:p.T
            # Get predicted individual output given state variables and current guess of aggregate price index
            NT = size(Data_year[t],1);
            rk = p.rk[t+1];
            w = p.w[t+1];
            for i = 1:NT
                ii = Data_year[t].id[i];
                lnz = Data.lnz[ii]; z = exp(lnz);
                lnpm = Data.logPm[ii]; pm = exp(lnpm);
                lnpe = log(exp(Data.lnpelec_tilde[ii])+τe); pe = exp(lnpe);
                lnψE = Data.lnfprod_E[ii]; ψE = exp(lnψE);
                peψe = pe;
                poψo = (p.po[t+1]+τo);
                if Data.combineF[ii] == 12
                    pfψf = [poψo,peψe];
                    pE =  (1/ψE)*pE_func(12,pfψf,p);
                elseif Data.combineF[ii] == 123
                    pc = exp(Data.lnpc_tilde[ii])+τc;
                    pcψc = pc;
                    pfψf = [poψo,peψe,pcψc];
                    pE = (1/ψE)*pE_func(123,pfψf,p);
                elseif Data.combineF[ii] == 124
                    pg = exp(Data.lnpg_tilde[ii])+τg;
                    pgψg = pg;
                    pfψf = [poψo,peψe,pgψg];
                    pE = (1/ψE)*pE_func(124,pfψf,p);
                elseif Data.combineF[ii] == 1234
                    pc = exp(Data.lnpc_tilde[ii])+τc;
                    pcψc = pc;
                    pg = exp(Data.lnpg_tilde[ii])+τg;
                    pgψg = pg;
                    pfψf = [poψo,peψe,pcψc,pgψg];
                    pE = (1/ψE)*pE_func(1234,pfψf,p);
                end
                pind = pinput_func(rk,pm,w,pE,p);
                πpred[ii] = profit_func_monopolistic(z,pind,t,p);
            end
        end
        return πpred;
    end
    πpred_Eprod = StaticProfit_Eprod(M_Eprod,Data_Eprod,τnotax);
    # No fuel productivity (main model otherwise unchanged)
    function StaticProfit_nfp(M::Model_nfp,Data::DataFrame,τ)
        @unpack p = M;
        τg = τ[1]*p.ggmean;
        τc = τ[2]*p.cgmean;
        τo = τ[3]*p.ogmean;
        τe = τ[4]*p.egmean;
        # Initialize static profit under grid points (order of states: z,pm,pe,ψe,ψo)
        N = size(Data,1);
        Data_year = groupby(Data,:year);
        πpred = Array{Float64}(undef,N);
        ψe = exp(mean(Data.lnfprod_e));
        ψo = exp(mean(Data.lnfprod_o));
        ψg = exp(mean(Data.lnfprod_g[Data.gas.>0]));
        ψc = exp(mean(Data.lnfprod_c[Data.coal.>0]));
        for t = 1:p.T
            # Get predicted individual output given state variables and current guess of aggregate price index
            NT = size(Data_year[t],1);
            rk = p.rk[t+1];
            w = p.w[t+1];
            for i = 1:NT
                ii = Data_year[t].id[i];
                lnz = Data.lnz[ii]; z = exp(lnz);
                lnpm = Data.logPm[ii]; pm = exp(lnpm);
                lnpe = log(exp(Data.lnpelec_tilde[ii])+τe); pe = exp(lnpe);
                peψe = pe/ψe;
                poψo = (p.po[t+1]+τo)/ψo;
                if Data.combineF[ii] == 12
                    pfψf = [poψo,peψe];
                    pE =  pE_func(12,pfψf,p);
                elseif Data.combineF[ii] == 123
                    pc = exp(Data.lnpc_tilde[ii])+τc;
                    pcψc = pc/ψc;
                    pfψf = [poψo,peψe,pcψc];
                    pE = pE_func(123,pfψf,p);
                elseif Data.combineF[ii] == 124
                    pg = exp(Data.lnpg_tilde[ii])+τg;
                    pgψg = pg/ψg;
                    pfψf = [poψo,peψe,pgψg];
                    pE = pE_func(124,pfψf,p);
                elseif Data.combineF[ii] == 1234
                    pc = exp(Data.lnpc_tilde[ii])+τc;
                    pcψc = pc/ψc;
                    pg = exp(Data.lnpg_tilde[ii])+τg;
                    pgψg = pg/ψg;
                    pfψf = [poψo,peψe,pcψc,pgψg];
                    pE = pE_func(1234,pfψf,p);
                end
                pind = pinput_func(rk,pm,w,pE,p);
                πpred[ii] = profit_func_monopolistic(z,pind,t,p);
            end
        end
        return πpred;
    end
    πpred_nfp = StaticProfit_nfp(M_nfp,Data_nfp,τnotax);
    # No fuel productivity (re-estimate simple nested CES)
    function StaticProfit_simpleCES(M::Model_simpleCES,Data::DataFrame,τ)
        @unpack p = M;
        # p = Par_simpleCES(p; σ=1.7999572,λ=2.2235012);
        τg = τ[1]*p.ggmean;
        τc = τ[2]*p.cgmean;
        τo = τ[3]*p.ogmean;
        τe = τ[4]*p.egmean;
        # Initialize static profit under grid points (order of states: z,pm,pe,ψe,ψo)
        N = size(Data,1);
        Data_year = groupby(Data,:year);
        πpred = Array{Float64}(undef,N);
        for t = 1:p.T
            # Get predicted individual output given state variables and current guess of aggregate price index
            NT = size(Data_year[t],1);
            rk = p.rk[t+1];
            w = p.w[t+1];
            for i = 1:NT
                ii = Data_year[t].id[i];
                lnz = Data.lnz[ii]; z = exp(lnz);
                lnpm = Data.logPm[ii]; pm = exp(lnpm);
                lnpe = log(exp(Data.lnpelec_tilde[ii])+τe); pe = exp(lnpe);
                peψe = pe;
                poψo = (p.po[t+1]+τo);
                if Data.combineF[ii] == 12
                    pfψf = [poψo,peψe];
                    pE =  pE_func(12,pfψf,p);
                elseif Data.combineF[ii] == 123
                    pc = exp(Data.lnpc_tilde[ii])+τc;
                    pcψc = pc;
                    pfψf = [poψo,peψe,pcψc];
                    pE = pE_func(123,pfψf,p);
                elseif Data.combineF[ii] == 124
                    pg = exp(Data.lnpg_tilde[ii])+τg;
                    pgψg = pg;
                    pfψf = [poψo,peψe,pgψg];
                    pE = pE_func(124,pfψf,p);
                elseif Data.combineF[ii] == 1234
                    pc = exp(Data.lnpc_tilde[ii])+τc;
                    pcψc = pc;
                    pg = exp(Data.lnpg_tilde[ii])+τg;
                    pgψg = pg;
                    pfψf = [poψo,peψe,pcψc,pgψg];
                    pE = pE_func(1234,pfψf,p);
                end
                pind = pinput_func(rk,pm,w,pE,p);
                πpred[ii] = profit_func_monopolistic(z,pind,t,p);
            end
        end
        return πpred;
    end
    πpred_simpleCES = StaticProfit_simpleCES(M_simpleCES,Data_simpleCES,τnotax);
#

#--------------------------------------------------------------
#--------------------------------------------------------------
# 3. Forward Simulation Draws

## Function that draws  relevant variables for all forward simulations
    # Without bounds - no fuel productivity (main model otherwise unchanged)
        @everywhere function ForwardSimul_draws_nfp(p::Par_nfp,M::Model_nfp,Data::DataFrame,seed)
            Random.seed!(seed);
            @unpack N = M;
            @unpack S,Tf = p;
            z = Array{Float64}(undef,N,S,Tf);
            pm = Array{Float64}(undef,N,S,Tf);
            pe = Array{Float64}(undef,N,S,Tf);
            ψe = Array{Float64}(undef,N,S,Tf);
            ψo = Array{Float64}(undef,N,S,Tf);
            pg = Array{Float64}(undef,N,S,Tf);
            lnψg = Array{Float64}(undef,N,S,Tf);
            pc = Array{Float64}(undef,N,S,Tf);
            lnψc = Array{Float64}(undef,N,S,Tf);
            # Order of states (z,pm,pe,ψe,ψo)
            # Construct large variance-covariance matrix for all state variables (selected and unselected)
                # order of rows and columns: ψg,pg,ψc,pc,z,pm,pe,ψe,ψo
                # There are four different covariance matrices that give this information
                # INITIALIZE 
                nstate_tot = 9;
                cov_all = zeros(nstate_tot,nstate_tot);
                # Add non-selected ones
                cov_all[end-4:end,end-4:end] = Array(p.p_cov);
                # Add diagonal of selected ones (variance)
                cov_all[1,1] = Array(p.p_cov_g)[1,1]; # ψg
                cov_all[2,2] = Array(p.p_cov_g)[2,2]; # pg
                cov_all[3,3] = Array(p.p_cov_c)[1,1]; # ψc
                cov_all[4,4] = Array(p.p_cov_c)[2,2]; # pc
                # Add off-diagonals
                cov_all[2,1] = Array(p.p_cov_g)[2,1]; cov_all[1,2] = Array(p.p_cov_g)[1,2];    # ψg,pg
                cov_all[3,1] = Array(p.p_cov_gc)[3,1]; cov_all[1,3] = Array(p.p_cov_gc)[1,3];  # ψg,ψc
                cov_all[4,1] = Array(p.p_cov_gc)[4,1]; cov_all[1,4] = Array(p.p_cov_gc)[1,4];  # ψg,pc
                cov_all[5:end,1] = Array(p.p_cov_g)[3:end,1]; cov_all[1,5:end] = Array(p.p_cov_g)[1,3:end]; # ψg,z etc.
                cov_all[3,2] = Array(p.p_cov_gc)[3,2]; cov_all[2,3] = Array(p.p_cov_gc)[2,3]; # pg,ψc
                cov_all[4,2] = Array(p.p_cov_gc)[4,2]; cov_all[2,4] = Array(p.p_cov_gc)[2,4]; # pg,pc
                cov_all[5:end,2] = Array(p.p_cov_g)[3:end,2]; cov_all[2,5:end] = Array(p.p_cov_g)[2,3:end]; # pg,z etc.
                cov_all[4,3] = Array(p.p_cov_c)[2,1]; cov_all[3,4] = Array(p.p_cov_c)[1,2]; # ψc,pc
                cov_all[5:end,3] = Array(p.p_cov_c)[3:end,1]; cov_all[3,5:end] = Array(p.p_cov_c)[1,3:end]; # ψc,z etc.
                cov_all[5:end,4] = Array(p.p_cov_c)[3:end,2]; cov_all[4,5:end] = Array(p.p_cov_c)[2,3:end]; # pc,z etc.
            #
            state_rand = MvNormal(zeros(nstate_tot),cov_all);
            # state_resdraw = rand(state_rand,N,p.S,p.Tf);
            state_resdraw1 = rand(state_rand,N*p.S*p.Tf);
            # Reshape
            state_resdraw = [reshape(state_resdraw1[1,:],N,p.S,p.Tf),reshape(state_resdraw1[2,:],N,p.S,p.Tf),reshape(state_resdraw1[3,:],N,p.S,p.Tf),
                            reshape(state_resdraw1[4,:],N,p.S,p.Tf),reshape(state_resdraw1[5,:],N,p.S,p.Tf),reshape(state_resdraw1[6,:],N,p.S,p.Tf),
                            reshape(state_resdraw1[7,:],N,p.S,p.Tf),reshape(state_resdraw1[8,:],N,p.S,p.Tf),reshape(state_resdraw1[9,:],N,p.S,p.Tf)];
            # add all state variables. order: ψg,pg,ψc,pc,z,pm,pe,ψe,ψo
            #                                  1,2 ,3 ,4 ,5,6 ,7 ,8 ,9
            for i = 1:N
                for s = 1:S
                    # First year forward
                    t = Data.year[i]-2009;
                    lnz = p.μz_t[t+1] + p.ρz*Data.res_lnz[i] + state_resdraw[5][i,s,1]; z[i,s,1] = exp(lnz);
                    lnpm = p.μpm_t[t+1] + p.ρ_pm*Data.res_pm[i] + state_resdraw[6][i,s,1]; pm[i,s,1] = exp(lnpm); 
                    # lnpm = p.μpm_t[t+1] + p.ρ_pm*Data.logPm[i] + state_resdraw[6][i,s,1]; pm[i,s,1] = exp(lnpm); 
                    lnpe = p.μpe_t[t+1] + p.ρ_pe*Data.res_pe[i] + state_resdraw[7][i,s,1]; pe[i,s,1] = exp(lnpe);
                    lnψe = p.μψe_t[t+1] + state_resdraw[8][i,s,1]; ψe[i,s,1] = exp(lnψe);
                    lnψo = p.μψo_t[t+1] + state_resdraw[9][i,s,1]; ψo[i,s,1] = exp(lnψo);
                    lnpg = p.μpg_t[t+1] + state_resdraw[2][i,s,1]; pg[i,s,1] = exp(lnpg);
                    lnψg[i,s,1] = p.μψg_t[t+1] + state_resdraw[1][i,s,1];
                    lnpc = p.μpc_t[t+1] + state_resdraw[4][i,s,1]; pc[i,s,1] = exp(lnpc);
                    lnψc[i,s,1] = p.μψc_t[t+1] + state_resdraw[3][i,s,1];
                    # Multiple years forward
                    for tf=2:Tf
                        res_lnz = log(z[i,s,tf-1]) - p.μz_t[t+1];
                            lnz = p.μz_t[t+1] + p.ρz*res_lnz + state_resdraw[5][i,s,tf]; z[i,s,tf] = exp(lnz);
                        res_pm = log(pm[i,s,tf-1]) - p.μpm_t[t+1];
                        # res_pm = log(pm[i,s,tf-1]);
                            lnpm = p.μpm_t[t+1] + p.ρ_pm*res_pm + state_resdraw[6][i,s,tf]; pm[i,s,tf] = exp(lnpm); 
                        res_pe = log(pe[i,s,tf-1]) - p.μpe_t[t+1];
                            lnpe = p.μpe_t[t+1] + p.ρ_pe*res_pe + state_resdraw[7][i,s,tf]; pe[i,s,tf] = exp(lnpe);
                        lnψe = p.μψe_t[t+1] + state_resdraw[8][i,s,tf]; ψe[i,s,tf] = exp(lnψe);
                        lnψo = p.μψo_t[t+1] + state_resdraw[9][i,s,tf]; ψo[i,s,tf] = exp(lnψo);
                        lnpg = p.μpg_t[t+1] + state_resdraw[2][i,s,tf]; pg[i,s,tf] = exp(lnpg);
                        lnψg[i,s,tf] = p.μψg_t[t+1] + state_resdraw[1][i,s,tf];
                        lnpc = p.μpc_t[t+1] + state_resdraw[4][i,s,tf]; pc[i,s,tf] = exp(lnpc);
                        lnψc[i,s,tf] = p.μψc_t[t+1] + state_resdraw[3][i,s,tf];
                    end
                end
            end
            # Update model
            M = Model_nfp(M; z_fs=copy(z),pm_fs=copy(pm),pe_fs=copy(pe),ψe_fs=copy(ψe),ψo_fs=copy(ψo),pg_fs=copy(pg),pc_fs=copy(pc),
                    lnψg_fs=copy(lnψg),lnψc_fs=copy(lnψc));
            return M,state_resdraw1;
        end
    #
    # Without bounds - Energy productivity 
        @everywhere function ForwardSimul_draws_Eprod(p::Par_Eprod,M::Model_Eprod,Data::DataFrame,seed)
            Random.seed!(seed);
            @unpack N = M;
            @unpack S,Tf = p;
            z = Array{Float64}(undef,N,S,Tf);
            pm = Array{Float64}(undef,N,S,Tf);
            pe = Array{Float64}(undef,N,S,Tf);
            ψE = Array{Float64}(undef,N,S,Tf);
            pg = Array{Float64}(undef,N,S,Tf);
            pc = Array{Float64}(undef,N,S,Tf);
            # Order of states (z,pm,pe,ψE)
            # Construct large variance-covariance matrix for all state variables (selected and unselected)
                # order of rows and columns: pg,pc,z,pm,pe,ψE
                # There are four different covariance matrices that give this information
                # INITIALIZE 
                nstate_tot = 6;
                cov_all = zeros(nstate_tot,nstate_tot);
                # Add non-selected ones
                cov_all[end-3:end,end-3:end] = Array(p.p_cov);
                # Add coal price
                cov_all[2,2:end] = Array(p.p_cov_c)[1,:];
                cov_all[2:end,2] = Array(p.p_cov_c)[:,1];
                # Add covariance between coal and gas price
                cov_all[1,1] = Array(p.p_cov_g)[2,2]; # var(pg)
                cov_all[2,2] = Array(p.p_cov_c)[1,1]; # var(pc)
                cov_all[2,1] = Array(p.p_cov_gc)[2,1]; cov_all[1,2] = Array(p.p_cov_gc)[1,2]; # cov(pg,pc)
                # Add gas price
                cov_all[1,3:end] = Array(p.p_cov_g)[1,2:end];
                cov_all[3:end,1] = Array(p.p_cov_g)[2:end,1];
            #
            state_rand = MvNormal(zeros(nstate_tot),cov_all);
            # state_resdraw = rand(state_rand,N,p.S,p.Tf);
            state_resdraw1 = rand(state_rand,N*p.S*p.Tf);
            # Reshape
            state_resdraw = [reshape(state_resdraw1[1,:],N,p.S,p.Tf),reshape(state_resdraw1[2,:],N,p.S,p.Tf),reshape(state_resdraw1[3,:],N,p.S,p.Tf),
                            reshape(state_resdraw1[4,:],N,p.S,p.Tf),reshape(state_resdraw1[5,:],N,p.S,p.Tf),reshape(state_resdraw1[6,:],N,p.S,p.Tf)];
            # add all state variables. order: pg,pc,z,pm,pe,ψE
            #                                  1,2, 3, 4,5, 6 
            for i = 1:N
                for s = 1:S
                    # First year forward
                    t = Data.year[i]-2009;
                    lnz = p.μz_t[t+1] + p.ρz*Data.res_lnz[i] + state_resdraw[3][i,s,1]; z[i,s,1] = exp(lnz);
                    lnpm = p.μpm_t[t+1] + p.ρ_pm*Data.res_pm[i] + state_resdraw[4][i,s,1]; pm[i,s,1] = exp(lnpm); 
                    lnpe = p.μpe_t[t+1] + p.ρ_pe*Data.res_pe[i] + state_resdraw[5][i,s,1]; pe[i,s,1] = exp(lnpe);
                    lnψE = p.μψE_t[t+1] + p.ρ_ψE*Data.res_prodE[i] + state_resdraw[6][i,s,1]; ψE[i,s,1] = exp(lnψE);
                    lnpg = p.μpg_t[t+1] + state_resdraw[1][i,s,1]; pg[i,s,1] = exp(lnpg);
                    lnpc = p.μpc_t[t+1] + state_resdraw[2][i,s,1]; pc[i,s,1] = exp(lnpc);
                    # Multiple years forward
                    for tf=2:Tf
                        res_lnz = log(z[i,s,tf-1]) - p.μz_t[t+1];
                            lnz = p.μz_t[t+1] + p.ρz*res_lnz + state_resdraw[3][i,s,tf]; z[i,s,tf] = exp(lnz);
                        res_pm = log(pm[i,s,tf-1]) - p.μpm_t[t+1];
                            lnpm = p.μpm_t[t+1] + p.ρ_pm*res_pm + state_resdraw[4][i,s,tf]; pm[i,s,tf] = exp(lnpm); 
                        res_pe = log(pe[i,s,tf-1]) - p.μpe_t[t+1];
                            lnpe = p.μpe_t[t+1] + p.ρ_pe*res_pe + state_resdraw[5][i,s,tf]; pe[i,s,tf] = exp(lnpe);
                        res_ψE = log(ψE[i,s,tf-1]) - p.μψE_t[t+1];
                            lnψE = p.μψE_t[t+1] + p.ρ_ψE*res_ψE + state_resdraw[6][i,s,tf]; ψE[i,s,tf] = exp(lnψE);
                        lnpg = p.μpg_t[t+1] + state_resdraw[1][i,s,tf]; pg[i,s,tf] = exp(lnpg);
                        lnpc = p.μpc_t[t+1] + state_resdraw[2][i,s,tf]; pc[i,s,tf] = exp(lnpc);
                    end
                end
            end
            # Update model
            M = Model_Eprod(M; z_fs=copy(z),pm_fs=copy(pm),pe_fs=copy(pe),ψE_fs=copy(ψE),pg_fs=copy(pg),pc_fs=copy(pc));
            return M,state_resdraw1;
        end
    #
    # Without bounds - no fuel productivity (re-estimate simple nested CES)
        @everywhere function ForwardSimul_draws_simpleCES(p::Par_simpleCES,M::Model_simpleCES,Data::DataFrame,seed)
            Random.seed!(seed);
            @unpack N = M;
            @unpack S,Tf = p;
            z = Array{Float64}(undef,N,S,Tf);
            pm = Array{Float64}(undef,N,S,Tf);
            pe = Array{Float64}(undef,N,S,Tf);
            pg = Array{Float64}(undef,N,S,Tf);
            pc = Array{Float64}(undef,N,S,Tf);
            # Order of states (z,pm,pe)
            # Construct large variance-covariance matrix for all state variables (selected and unselected)
                # order of rows and columns: pg,pc,z,pm,pe
                # There are four different covariance matrices that give this information
                # INITIALIZE 
                nstate_tot = 5;
                cov_all = zeros(nstate_tot,nstate_tot);
                # Add non-selected ones
                cov_all[end-2:end,end-2:end] = Array(p.p_cov);
                # Add coal price
                cov_all[2,2:end] = Array(p.p_cov_c)[1,:];
                cov_all[2:end,2] = Array(p.p_cov_c)[:,1];
                # Add covariance between coal and gas price
                cov_all[1,1] = Array(p.p_cov_g)[2,2]; # var(pg)
                cov_all[2,2] = Array(p.p_cov_c)[1,1]; # var(pc)
                cov_all[2,1] = Array(p.p_cov_gc)[2,1]; cov_all[1,2] = Array(p.p_cov_gc)[1,2]; # cov(pg,pc)
                # Add gas price
                cov_all[1,3:end] = Array(p.p_cov_g)[1,2:end];
                cov_all[3:end,1] = Array(p.p_cov_g)[2:end,1];
            #
            state_rand = MvNormal(zeros(nstate_tot),cov_all);
            # state_resdraw = rand(state_rand,N,p.S,p.Tf);
            state_resdraw1 = rand(state_rand,N*p.S*p.Tf);
            # Reshape
            state_resdraw = [reshape(state_resdraw1[1,:],N,p.S,p.Tf),reshape(state_resdraw1[2,:],N,p.S,p.Tf),reshape(state_resdraw1[3,:],N,p.S,p.Tf),
                            reshape(state_resdraw1[4,:],N,p.S,p.Tf),reshape(state_resdraw1[5,:],N,p.S,p.Tf)];
            # add all state variables. order: pg,pc,z,pm,pe
            #                                  1,2, 3, 4,5
            for i = 1:N
                for s = 1:S
                    # First year forward
                    t = Data.year[i]-2009;
                    lnz = p.μz_t[t+1] + p.ρz*Data.res_lnz[i] + state_resdraw[3][i,s,1]; z[i,s,1] = exp(lnz);
                    lnpm = p.μpm_t[t+1] + p.ρ_pm*Data.res_pm[i] + state_resdraw[4][i,s,1]; pm[i,s,1] = exp(lnpm); 
                    lnpe = p.μpe_t[t+1] + p.ρ_pe*Data.res_pe[i] + state_resdraw[5][i,s,1]; pe[i,s,1] = exp(lnpe);
                    lnpg = p.μpg_t[t+1] + state_resdraw[1][i,s,1]; pg[i,s,1] = exp(lnpg);
                    lnpc = p.μpc_t[t+1] + state_resdraw[2][i,s,1]; pc[i,s,1] = exp(lnpc);
                    # Multiple years forward
                    for tf=2:Tf
                        res_lnz = log(z[i,s,tf-1]) - p.μz_t[t+1];
                            lnz = p.μz_t[t+1] + p.ρz*res_lnz + state_resdraw[3][i,s,tf]; z[i,s,tf] = exp(lnz);
                        res_pm = log(pm[i,s,tf-1]) - p.μpm_t[t+1];
                            lnpm = p.μpm_t[t+1] + p.ρ_pm*res_pm + state_resdraw[4][i,s,tf]; pm[i,s,tf] = exp(lnpm); 
                        res_pe = log(pe[i,s,tf-1]) - p.μpe_t[t+1];
                            lnpe = p.μpe_t[t+1] + p.ρ_pe*res_pe + state_resdraw[5][i,s,tf]; pe[i,s,tf] = exp(lnpe);
                        lnpg = p.μpg_t[t+1] + state_resdraw[1][i,s,tf]; pg[i,s,tf] = exp(lnpg);
                        lnpc = p.μpc_t[t+1] + state_resdraw[2][i,s,tf]; pc[i,s,tf] = exp(lnpc);
                    end
                end
            end
            # Update model
            M = Model_simpleCES(M; z_fs=copy(z),pm_fs=copy(pm),pe_fs=copy(pe),pg_fs=copy(pg),pc_fs=copy(pc));
            return M,state_resdraw1;
        end
    #
#

# Get simulation draws
    seed = 13421;
    # seed = 14123;
    # Without bounds - fuel productivity
        @time M,state_resdraw1 = ForwardSimul_draws_noswitch(p,M,Data,seed);
        test=1;
    #
    # Without bounds - fuel productivity
        @time M_nfp,state_resdraw1_nfp = ForwardSimul_draws_nfp(p_nfp,M_nfp,Data_nfp,seed);
        test=1;
    #
    # Without bounds - fuel productivity
        @time M_Eprod,state_resdraw1_Eprod = ForwardSimul_draws_Eprod(p_Eprod,M_Eprod,Data_Eprod,seed);
        test=1;
    #
    # Without bounds - fuel productivity
        @time M_simpleCES,state_resdraw1_simpleCES = ForwardSimul_draws_simpleCES(p_simpleCES,M_simpleCES,Data_simpleCES,seed);
        test=1;
    #
#

#--------------------------------------------------------------
#--------------------------------------------------------------
# 3. Compute counterfactuals

# Test each method without any tax (see if I can recover levels)
τnotax = [0,0,0,0];
    # 1. Full model without switching
        # Including forward simulation
        @everywhere function Welfare_ForwardSimul_noswitch(M::Model,Data::DataFrame,p::Par,τ,d_elast::Float64=p.ρ,rscale::Float64=p.η,perfcompl=false)
            @unpack N,z_fs,pm_fs,pe_fs,ψe_fs,ψo_fs,pg_fs,lnψg_fs,pc_fs,lnψc_fs = M;
            @unpack S,Tf,T,γg,γc,γo,γe,β,Κ=p;
            τg = τ[1];
            τc = τ[2];
            τo = τ[3];
            τe = τ[4];
            p = Par(p; ρ=d_elast, η=rscale);
            M = Model(M; p =p);
            println("Demand elasticity = $(p.ρ)")
            # Update aggregate output price
            p,M = OutputPrice_func_nogrid(M,Data,p,τ);
            # Start static of social welfare
            pout = SharedArray{Float64}(T,S,Tf+1);
            gas = SharedArray{Float64}(N,S,Tf+1);
            coal = SharedArray{Float64}(N,S,Tf+1);
            oil = SharedArray{Float64}(N,S,Tf+1);
            elec = SharedArray{Float64}(N,S,Tf+1);
            y = SharedArray{Float64}(N,S,Tf+1);
            E = SharedArray{Float64}(N,S,Tf+1);
                δ = 0.0;
                r = 0.150838;
            profit = SharedArray{Float64}(N,S,Tf+1);
            println("      ")
            println("--------------------------------------------------------------")
            println("--------      BEGINNING FORWARD SIMULATION     ---------------")
            println("---------      Main model                ----------")
            println("---------      No Switching - No fuel prod - Tax (g,c,o,e) = $τ    ----------")
            println("--------------------------------------------------------------")
            println("      ")
            # Year of data: Get firm-level objects: output, fuel prices and productivity, fuel quantities
            for s = 1:S
                pout[:,s,1] = p.pout_struc[2:7];
            end
            for i = 1:N
                for s = 1:S
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
                        pE = pE_func(12,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 123
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc/ψc;
                        pfψf = [poψo,peψe,pcψc];
                        pE = pE_func(123,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 124
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg/ψg;
                        pfψf = [poψo,peψe,pgψg];
                        pE = pE_func(124,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 1234
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc/ψc;
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg/ψg;
                        pfψf = [poψo,peψe,pcψc,pgψg];
                        pE = pE_func(1234,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    end
                    # Output
                    y[i,s,1] = output_func_monopolistic(z,pin,p.pout_struc[t+1],t,p);
                    # profit
                    profit[i,s,1] = profit_func_monopolistic(z,pin,t,p);
                    # Energy
                    E[i,s,1] = ((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE)^p.σ);
                    # Fuel quantitiy
                    oil[i,s,1] = (p.ogmean*E[i,s,1])*((p.βo/po)^p.λ)*(ψo^(p.λ-1))*(pE^p.λ);
                    elec[i,s,1] = (p.egmean*E[i,s,1])*((p.βe/pe)^p.λ)*(ψe^(p.λ-1))*(pE^p.λ);
                    if Data.combineF[i] == 12
                        gas[i,s,1] = 0;
                        coal[i,s,1] = 0;
                    elseif Data.combineF[i] == 123
                        gas[i,s,1] = 0;
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc)^p.λ)*(ψc^(p.λ-1))*(pE^p.λ);
                    elseif Data.combineF[i] == 124
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg)^p.λ)*(ψg^(p.λ-1))*(pE^p.λ);
                        coal[i,s,1] = 0;
                    elseif Data.combineF[i] == 1234
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg)^p.λ)*(ψg^(p.λ-1))*(pE^p.λ);
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc)^p.λ)*(ψc^(p.λ-1))*(pE^p.λ);
                    end
                end
            end
            # Years of forward simulation (starting one year ahead)
            @sync @distributed for j = 1:(Tf*S)
                tf,s = Tuple(CartesianIndices((Tf,S))[j]);
                # Get new aggregate price index
                Data_fs = deepcopy(Data);
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
                    if Data.combineF[i] == 123
                        Data_fs.lnfprod_c[i] = lnψc_fs[i,s,tf] + p.lnc_re_grid[grid_indices.c_re[i]];
                        Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                    elseif Data.combineF[i] == 124
                        Data_fs.lnfprod_g[i] = lnψg_fs[i,s,tf] + p.lng_re_grid[grid_indices.g_re[i]];
                        Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                    elseif Data.combineF[i] == 1234
                        Data_fs.lnfprod_c[i] = lnψc_fs[i,s,tf] + p.lnc_re_grid[grid_indices.c_re[i]];
                        Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                        Data_fs.lnfprod_g[i] = lnψg_fs[i,s,tf] + p.lng_re_grid[grid_indices.g_re[i]];
                        Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                    end
                end
                p_fs,M = OutputPrice_func_nogrid(M,Data_fs,p,τ);
                pout[:,s,tf+1] = p_fs.pout_struc[2:7];
                # Get everything else
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
                    if Data.combineF[i] == 12
                        pfψf = [poψo,peψe];
                        pE = pE_func(12,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 123
                        # ψc_fs[i,s,tf] = exp(Data_fs.lnfprod_c[i]);
                        pcψc = pc/exp(Data_fs.lnfprod_c[i]);
                        pfψf = [poψo,peψe,pcψc];
                        pE = pE_func(123,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 124
                        # ψg_fs[i,s,tf] = exp(Data_fs.lnfprod_g[i]);
                        pgψg = pg/exp(Data_fs.lnfprod_g[i]);
                        pfψf = [poψo,peψe,pgψg];
                        pE = pE_func(124,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 1234
                        # ψc_fs[i,s,tf] = exp(Data_fs.lnfprod_c[i]);
                        # ψg_fs[i,s,tf] = exp(Data_fs.lnfprod_g[i]);
                        pcψc = pc/exp(Data_fs.lnfprod_c[i]);
                        pgψg = pg/exp(Data_fs.lnfprod_g[i]);
                        pfψf = [poψo,peψe,pcψc,pgψg];
                        pE = pE_func(1234,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    end
                    # Output
                    y[i,s,tf+1] = output_func_monopolistic(z,pin,p_fs.pout_struc[t+1],t,p_fs);
                    # profit
                    profit[i,s,tf+1] = profit_func_monopolistic(z,pin,t,p_fs);
                    # Energy
                    E[i,s,tf+1] = ((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE)^p.σ);
                    # Fuel quantitiy
                    oil[i,s,tf+1] = (p.ogmean*E[i,s,tf+1])*((p.βo/po)^p.λ)*(ψo^(p.λ-1))*(pE^p.λ);
                    elec[i,s,tf+1] = (p.egmean*E[i,s,tf+1])*((p.βe/pe)^p.λ)*(ψe^(p.λ-1))*(pE^p.λ);
                    if Data.combineF[i] == 12
                        gas[i,s,tf+1] = 0;
                        coal[i,s,tf+1] = 0;
                    elseif Data.combineF[i] == 123
                        gas[i,s,tf+1] = 0;
                        coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc)^p.λ)*(exp(Data_fs.lnfprod_c[i])^(p.λ-1))*(pE^p.λ);
                    elseif Data.combineF[i] == 124
                        gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg)^p.λ)*(exp(Data_fs.lnfprod_g[i])^(p.λ-1))*(pE^p.λ);
                        coal[i,s,tf+1] = 0;
                    elseif Data.combineF[i] == 1234
                        gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg)^p.λ)*(exp(Data_fs.lnfprod_g[i])^(p.λ-1))*(pE^p.λ);
                        coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc)^p.λ)*(exp(Data_fs.lnfprod_c[i])^(p.λ-1))*(pE^p.λ);
                    end
                end
            end
            # Return aggregates by year forward (only look at last year in data - 2016)
            y_fs = zeros(Tf+1); E_fs = zeros(Tf+1); gas_fs = zeros(Tf+1); coal_fs = zeros(Tf+1); oil_fs = zeros(Tf+1); elec_fs = zeros(Tf+1); co2_fs = zeros(Tf+1); profit_fs = zeros(Tf+1); CS_fs = zeros(Tf+1);
            Data_year = groupby(Data,:year); Nt = size(Data_year[end])[1];
            # pE = pE[end-Nt+1:end,:,:];
            y = y[end-Nt+1:end,:,:];
            E = E[end-Nt+1:end,:,:];
            gas = gas[end-Nt+1:end,:,:];
            coal = coal[end-Nt+1:end,:,:];
            oil = oil[end-Nt+1:end,:,:];
            elec = elec[end-Nt+1:end,:,:];
            profit = profit[end-Nt+1:end,:,:];
            pout = pout[end,:,:];
            co2 = p.γg*gas .+ p.γc*coal .+ p.γo*oil .+ p.γe*elec;
            # Aggregate output (from rep consumer CES demand)
            Nfirms = Data_year[end].N[1];
            demand_shock = exp(p.d_t[end]);
            rhoterm = (p.ρ-1)/p.ρ;
            y_s_fs = zeros(p.S,p.Tf+1);
            for s = 1:p.S
                for tf = 1:(p.Tf+1)
                    if perfcompl == true
                        y_s_fs[s,tf] = demand_shock*minimum(y[:,s,tf]);
                    else
                        y_s_fs[s,tf] = ((demand_shock/Nfirms)*(sum(y[:,s,tf].^rhoterm)))^(1/rhoterm);
                    end
                end
            end
            # Consumer surplus
            CS_s_fs = zeros(p.S,p.Tf+1);
            thetaterm = p.θ/(1-p.θ)
            for tf = 1:(p.Tf+1)
                for s = 1:p.S
                    CS_s_fs[s,tf] = thetaterm*(pout[s,tf]^-thetaterm);
                end
                CS_fs[tf] = mean(CS_s_fs[:,tf]);
            end
            # Average across simulations
            for tf = 1:Tf+1
                y_fs[tf] = mean(y_s_fs[:,tf]);
                for i = 1:Nt
                    E_fs[tf] += mean(E[i,:,tf]);
                    gas_fs[tf] += mean(gas[i,:,tf]);
                    coal_fs[tf] += mean(coal[i,:,tf]);
                    oil_fs[tf] += mean(oil[i,:,tf]);
                    elec_fs[tf] += mean(elec[i,:,tf]);
                    co2_fs[tf] += mean(co2[i,:,tf]);
                    profit_fs[tf] += mean(profit[i,:,tf]);
                end
            end
            return y_fs,E_fs,gas_fs,coal_fs,oil_fs,elec_fs,co2_fs,profit_fs,CS_fs;
        end
        # No forward simulation
        @everywhere function Welfare_noswitch(M::Model,Data::DataFrame,p::Par,τ,d_elast::Float64=p.ρ,rscale::Float64=p.η,perfcompl=false)
            @unpack N,z_fs,pm_fs,pe_fs,ψe_fs,ψo_fs,pg_fs,lnψg_fs,pc_fs,lnψc_fs = M;
            @unpack S,Tf,T,γg,γc,γo,γe,β,Κ=p;
            τg = τ[1];
            τc = τ[2];
            τo = τ[3];
            τe = τ[4];
            p = Par(p; ρ=d_elast, η=rscale);
            M = Model(M; p =p);
            # println("Demand elasticity = $(p.ρ)")
            # Update aggregate output price
            p,M = OutputPrice_func_nogrid(M,Data,p,τ);
            # Start static of social welfare
            pout = SharedArray{Float64}(T);
            gas = SharedArray{Float64}(N);
            coal = SharedArray{Float64}(N);
            oil = SharedArray{Float64}(N);
            elec = SharedArray{Float64}(N);
            y = SharedArray{Float64}(N);
            E = SharedArray{Float64}(N);
                δ = 0.0;
                r = 0.150838;
            profit = SharedArray{Float64}(N);
            # Year of data: Get firm-level objects: output, fuel prices and productivity, fuel quantities
            pout = p.pout_struc[2:7];
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
                        pE = pE_func(12,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 123
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc/ψc;
                        pfψf = [poψo,peψe,pcψc];
                        pE = pE_func(123,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 124
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg/ψg;
                        pfψf = [poψo,peψe,pgψg];
                        pE = pE_func(124,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 1234
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc/ψc;
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg/ψg;
                        pfψf = [poψo,peψe,pcψc,pgψg];
                        pE = pE_func(1234,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    end
                    # Output
                    y[i] = output_func_monopolistic(z,pin,p.pout_struc[t+1],t,p);
                    # profit
                    profit[i] = profit_func_monopolistic(z,pin,t,p);
                    # Energy
                    E[i] = ((y[i]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE)^p.σ);
                    # Fuel quantitiy
                    oil[i] = (p.ogmean*E[i])*((p.βo/po)^p.λ)*(ψo^(p.λ-1))*(pE^p.λ);
                    elec[i] = (p.egmean*E[i])*((p.βe/pe)^p.λ)*(ψe^(p.λ-1))*(pE^p.λ);
                    if Data.combineF[i] == 12
                        gas[i] = 0;
                        coal[i] = 0;
                    elseif Data.combineF[i] == 123
                        gas[i] = 0;
                        coal[i] = (p.cgmean*E[i])*((p.βc/pc)^p.λ)*(ψc^(p.λ-1))*(pE^p.λ);
                    elseif Data.combineF[i] == 124
                        gas[i] = (p.ggmean*E[i])*((p.βg/pg)^p.λ)*(ψg^(p.λ-1))*(pE^p.λ);
                        coal[i] = 0;
                    elseif Data.combineF[i] == 1234
                        gas[i] = (p.ggmean*E[i])*((p.βg/pg)^p.λ)*(ψg^(p.λ-1))*(pE^p.λ);
                        coal[i] = (p.cgmean*E[i])*((p.βc/pc)^p.λ)*(ψc^(p.λ-1))*(pE^p.λ);
                    end
            end
            # Return aggregates by year forward (only look at last year in data - 2016)
            Data_year = groupby(Data,:year); Nt = size(Data_year[end])[1];
            # pE = pE[end-Nt+1:end,:,:];
            y = y[(end-Nt+1):end];
            E = E[(end-Nt+1):end];
            gas = gas[(end-Nt+1):end];
            coal = coal[(end-Nt+1):end];
            oil = oil[(end-Nt+1):end];
            elec = elec[(end-Nt+1):end];
            profit = profit[(end-Nt+1):end];
            pout = pout[end];
            co2 = p.γg*gas .+ p.γc*coal .+ p.γo*oil .+ p.γe*elec;
            # Aggregate output (from rep consumer CES demand)
            Nfirms = Data_year[end].N[1];
            demand_shock = exp(p.d_t[end]);
            rhoterm = (p.ρ-1)/p.ρ;
            if perfcompl == true
                y_fs = demand_shock*minimum(y);
            else
                y_fs = ((demand_shock/Nfirms)*(sum(y[:].^rhoterm)))^(1/rhoterm);
            end
            # Consumer surplus
            thetaterm = p.θ/(1-p.θ)
            CS_fs = thetaterm*(pout^-thetaterm);
            # Other variables
            E_fs = sum(E);
            gas_fs = sum(gas);
            coal_fs = sum(coal);
            oil_fs = sum(oil);
            elec_fs = sum(elec);
            co2_fs = sum(co2);
            profit_fs = sum(profit);
            return y_fs,E_fs,gas_fs,coal_fs,oil_fs,elec_fs,co2_fs,profit_fs,CS_fs;
            # return pE,pin,y,E,gas,gasspend,coal,coalspend,oil,oilspend,elec,elecspend,pout,Espend,Mspend,Lspend,Krent,Kval,ψc_fs,ψg_fs,profit;
            # return pE,pin,y,E,gas,coal,oil,elec,Espend,profit;
        end
    #
    # 2. Energy productivity 
        # Including forward simulation
        @everywhere function Welfare_ForwardSimul_noswitch_Eprod(M::Model_Eprod,Data::DataFrame,p::Par_Eprod,τ,d_elast::Float64=p.ρ,rscale::Float64=p.η,perfcompl=false)
            @unpack N,z_fs,pm_fs,pe_fs,ψE_fs,pg_fs,pc_fs = M;
            @unpack S,Tf,T,γg,γc,γo,γe,β=p;
            τg = τ[1];
            τc = τ[2];
            τo = τ[3];
            τe = τ[4];
            p = Par_Eprod(p; ρ=d_elast, η=rscale);
            M = Model_Eprod(M; p = p);
            println("Demand elasticity = $(p.ρ)")
            # Update aggregate output price
            p,M = OutputPrice_func_Eprod(M,Data,p,τ);
            # Start static of social welfare
            pout = SharedArray{Float64}(T,S,Tf+1);
            gas = SharedArray{Float64}(N,S,Tf+1);
            coal = SharedArray{Float64}(N,S,Tf+1);
            oil = SharedArray{Float64}(N,S,Tf+1);
            elec = SharedArray{Float64}(N,S,Tf+1);
            # gasspend = SharedArray{Float64}(N,S,Tf+1);
            # coalspend = SharedArray{Float64}(N,S,Tf+1);
            # oilspend = SharedArray{Float64}(N,S,Tf+1);
            # elecspend = SharedArray{Float64}(N,S,Tf+1);
            # pE = SharedArray{Float64}(N,S,Tf+1);
            # pin = SharedArray{Float64}(N,S,Tf+1);
            y = SharedArray{Float64}(N,S,Tf+1);
            E = SharedArray{Float64}(N,S,Tf+1);
            # Lspend = SharedArray{Float64}(N,S,Tf+1);
            # Mspend = SharedArray{Float64}(N,S,Tf+1);
            # Krent = SharedArray{Float64}(N,S,Tf+1);
                δ = 0.0;
                r = 0.150838;
            # Kval = SharedArray{Float64}(N,S,Tf+1);
            # Espend = SharedArray{Float64}(N,S,Tf+1);
            profit = SharedArray{Float64}(N,S,Tf+1);
            println("      ")
            println("--------------------------------------------------------------")
            println("--------      BEGINNING FORWARD SIMULATION     ---------------")
            println("---------      Energy productivity model       ----------")
            println("---------      No Switching - No fuel prod - Tax (g,c,o,e) = $τ    ----------")
            println("--------------------------------------------------------------")
            println("      ")
            # Year of data: Get firm-level objects: output, fuel prices and productivity, fuel quantities
            for s = 1:S
                pout[:,s,1] = p.pout_struc[2:7];
            end
            for i = 1:N
                for s = 1:S
                    t = Data.year[i]-2009;
                    pm = exp(Data.logPm[i]);
                    z = exp(Data.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean;
                    pe = exp(Data.lnpelec_tilde[i]) + τe*p.egmean;
                    # energy productivity
                    ψE = exp(Data.lnfprod_E[i]);
                    # input prices indices
                    poψo = po;
                    peψe = pe;
                    if Data.combineF[i] == 12
                        pfψf = [poψo,peψe];
                        pE = (1/ψE)*pE_func(12,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 123
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc;
                        pfψf = [poψo,peψe,pcψc];
                        pE = (1/ψE)*pE_func(123,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 124
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg;
                        pfψf = [poψo,peψe,pgψg];
                        pE = (1/ψE)*pE_func(124,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 1234
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc;
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg;
                        pfψf = [poψo,peψe,pcψc,pgψg];
                        pE = (1/ψE)*pE_func(1234,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    end
                    # Output
                    y[i,s,1] = output_func_monopolistic(z,pin,p.pout_struc[t+1],t,p);
                    # profit
                    profit[i,s,1] = profit_func_monopolistic(z,pin,t,p);
                    # Energy
                    E[i,s,1] = ((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE)^p.σ);
                    # Fuel quantitiy
                    oil[i,s,1] = (p.ogmean*E[i,s,1])*((p.βo/po)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                    elec[i,s,1] = (p.egmean*E[i,s,1])*((p.βe/pe)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                    # oilspend[i,s,1] = (po/p.ogmean)*oil[i,s,1];
                    # elecspend[i,s,1] = (pe/p.egmean)*elec[i,s,1];
                    if Data.combineF[i] == 12
                        gas[i,s,1] = 0;
                        # gasspend[i,s,1]=0;
                        coal[i,s,1] = 0;
                        # coalspend[i,s,1]=0;
                    elseif Data.combineF[i] == 123
                        gas[i,s,1] = 0;
                        # gasspend[i,s,1]=0;
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                        # coalspend[i,s,1] = (pc/p.cgmean)*coal[i,s,1];
                    elseif Data.combineF[i] == 124
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                        # gasspend[i,s,1] = (pg/p.ggmean)*gas[i,s,1];
                        coal[i,s,1] = 0;
                        # coalspend[i,s,1]=0;
                    elseif Data.combineF[i] == 1234
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                        # gasspend[i,s,1] = (pg/p.ggmean)*gas[i,s,1];
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                        # coalspend[i,s,1] = (pc/p.cgmean)*coal[i,s,1];
                    end
                    # Input spending
                    # Espend[i,s,1] = pE*E[i,s,1];
                    # Mspend[i,s,1] = pm*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αm*pin[i,s,1]/pm)^p.σ);
                    # Lspend[i,s,1] = p.w[t+1]*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αl*pin[i,s,1]/p.w[t+1])^p.σ);
                    # Krent[i,s,1] = p.rk[t+1]*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,1]/p.rk[t+1])^p.σ);
                    # Kval[i,s,1] = (p.rk[t+1]/(δ+r))*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,1]/p.rk[t+1])^p.σ);
                end
            end
            # Years of forward simulation (starting one year ahead)
            @sync @distributed for j = 1:(Tf*S)
                tf,s = Tuple(CartesianIndices((Tf,S))[j]);
                # Get new aggregate price index
                Data_fs = deepcopy(Data);
                for i = 1:N
                    t = Data_fs.year[i]-2009;
                    Data_fs.lnz[i] = log(z_fs[i,s,tf]);
                    Data_fs.logPm[i] = log(pm_fs[i,s,tf]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1]; 
                    Data_fs.lnpelec_tilde[i] = log(pe_fs[i,s,tf]);
                    # Energy productivity
                    ψE = ψE_fs[i,s,tf]; Data_fs.lnfprod_E[i] = log(ψE);
                    # input prices indices
                    if Data.combineF[i] == 123
                        Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                    elseif Data.combineF[i] == 124
                        Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                    elseif Data.combineF[i] == 1234
                        Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                        Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                    end
                end
                p_fs,M = OutputPrice_func_Eprod(M,Data_fs,p,τ);
                pout[:,s,tf+1] = p_fs.pout_struc[2:7];
                # Get everything else
                for i = 1:N
                    t = Data_fs.year[i]-2009; 
                    pm = exp(Data_fs.logPm[i]);
                    z = exp(Data_fs.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean;
                    pe = exp(Data_fs.lnpelec_tilde[i]) + τe*p.egmean; 
                    pc = exp(Data_fs.lnpc_tilde[i]) + τc*p.cgmean;
                    pg = exp(Data_fs.lnpg_tilde[i]) + τg*p.ggmean;
                    # Energy productivity
                    ψE = ψE_fs[i,s,tf];
                    # input prices indices
                    poψo = po;
                    peψe = pe;
                    if Data.combineF[i] == 12
                        pfψf = [poψo,peψe];
                        pE = (1/ψE)*pE_func(12,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 123
                        pcψc = pc;
                        pfψf = [poψo,peψe,pcψc];
                        pE = (1/ψE)*pE_func(123,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 124
                        pgψg = pg;
                        pfψf = [poψo,peψe,pgψg];
                        pE = (1/ψE)*pE_func(124,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 1234
                        pcψc = pc;
                        pgψg = pg;
                        pfψf = [poψo,peψe,pcψc,pgψg];
                        pE = (1/ψE)*pE_func(1234,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    end
                    # Output
                    y[i,s,tf+1] = output_func_monopolistic(z,pin,p_fs.pout_struc[t+1],t,p_fs);
                    # profit
                    profit[i,s,tf+1] = profit_func_monopolistic(z,pin,t,p_fs);
                    # Energy
                    E[i,s,tf+1] = ((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE)^p.σ);
                    # Fuel quantitiy
                    oil[i,s,tf+1] = (p.ogmean*E[i,s,tf+1])*((p.βo/po)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                    elec[i,s,tf+1] = (p.egmean*E[i,s,tf+1])*((p.βe/pe)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                    # oilspend[i,s,tf+1] = (po/p.ogmean)*oil[i,s,tf+1];
                    # elecspend[i,s,tf+1] = (pe/p.egmean)*elec[i,s,tf+1];
                    if Data.combineF[i] == 12
                        gas[i,s,tf+1] = 0;
                        # gasspend[i,s,tf+1]=0;
                        coal[i,s,tf+1] = 0;
                        # coalspend[i,s,tf+1]=0;
                    elseif Data.combineF[i] == 123
                        gas[i,s,tf+1] = 0;
                        # gasspend[i,s,tf+1]=0;
                        coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc)^p.λ)*(ψE_fs[i,s,tf]^(p.λ-1))*(pE^p.λ);
                        # coalspend[i,s,tf+1] = (pc/p.cgmean)*coal[i,s,tf+1];
                    elseif Data.combineF[i] == 124
                        gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg)^p.λ)*(ψE_fs[i,s,tf]^(p.λ-1))*(pE^p.λ);
                        # gasspend[i,s,tf+1] = (pg/p.ggmean)*gas[i,s,tf+1];
                        coal[i,s,tf+1] = 0;
                        # coalspend[i,s,tf+1]=0;
                    elseif Data.combineF[i] == 1234
                        gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg)^p.λ)*(ψE_fs[i,s,tf]^(p.λ-1))*(pE^p.λ);
                        # gasspend[i,s,tf+1] = (pg/p.ggmean)*gas[i,s,tf+1];
                        coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc)^p.λ)*(ψE_fs[i,s,tf]^(p.λ-1))*(pE^p.λ);
                        # coalspend[i,s,tf+1] = (pc/p.cgmean)*coal[i,s,tf+1];
                    end
                    # Input spending
                    # Espend[i,s,tf+1] = pE*E[i,s,tf+1];
                    # Mspend[i,s,tf+1] = pm*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αm*pin[i,s,tf+1]/pm)^p.σ);
                    # Lspend[i,s,tf+1] = p.w[t+1]*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αl*pin[i,s,tf+1]/p.w[t+1])^p.σ);
                    # Krent[i,s,tf+1] = p.rk[t+1]*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,tf+1]/p.rk[t+1])^p.σ);
                    # Kval[i,s,tf+1] = (p.rk[t+1]/(δ+r))*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,tf+1]/p.rk[t+1])^p.σ);
                end
            end
            # Return aggregates by year forward (only look at last year in data - 2016)
            y_fs = zeros(Tf+1); E_fs = zeros(Tf+1); gas_fs = zeros(Tf+1); coal_fs = zeros(Tf+1); oil_fs = zeros(Tf+1); elec_fs = zeros(Tf+1); co2_fs = zeros(Tf+1); profit_fs = zeros(Tf+1); CS_fs = zeros(Tf+1);
            Data_year = groupby(Data,:year); Nt = size(Data_year[end])[1];
            # pE = pE[end-Nt+1:end,:,:];
            y = y[end-Nt+1:end,:,:];
            E = E[end-Nt+1:end,:,:];
            gas = gas[end-Nt+1:end,:,:];
            coal = coal[end-Nt+1:end,:,:];
            oil = oil[end-Nt+1:end,:,:];
            elec = elec[end-Nt+1:end,:,:];
            profit = profit[end-Nt+1:end,:,:];
            pout = pout[end,:,:];
            co2 = p.γg*gas .+ p.γc*coal .+ p.γo*oil .+ p.γe*elec;
            # Aggregate output (from rep consumer CES demand)
            Nfirms = Data_year[end].N[1];
            demand_shock = exp(p.d_t[end]);
            rhoterm = (p.ρ-1)/p.ρ;
            y_s_fs = zeros(p.S,p.Tf+1);
            for s = 1:p.S
                for tf = 1:(p.Tf+1)
                    if perfcompl == true
                        y_s_fs[s,tf] = demand_shock*minimum(y[:,s,tf]);
                    else
                        y_s_fs[s,tf] = ((demand_shock/Nfirms)*(sum(y[:,s,tf].^rhoterm)))^(1/rhoterm);
                    end
                end
            end
            # Consumer surplus
            CS_s_fs = zeros(p.S,p.Tf+1);
            thetaterm = p.θ/(1-p.θ)
            for tf = 1:(p.Tf+1)
                for s = 1:p.S
                    CS_s_fs[s,tf] = thetaterm*(pout[s,tf]^-thetaterm);
                end
                CS_fs[tf] = mean(CS_s_fs[:,tf]);
            end
            # Average across simulations
            for tf = 1:Tf+1
                # pE_fs[tf] = mean(pE[:,:,tf]);
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
                end
            end
            return y_fs,E_fs,gas_fs,coal_fs,oil_fs,elec_fs,co2_fs,profit_fs,CS_fs;
            #return pE,pin,y,E,gas,gasspend,coal,coalspend,oil,oilspend,elec,elecspend,pout,Espend,Mspend,Lspend,Krent,Kval,profit;
            # return pE,pin,y,E,gas,coal,oil,elec,Espend,profit;
        end
        # Including forward simulation - fixing substitution elasticity
        @everywhere function Welfare_ForwardSimul_noswitch_Eprod_fixsub(M::Model_Eprod,Data::DataFrame,p::Par_Eprod,τ,d_elast::Float64=p.ρ,rscale::Float64=p.η)
            @unpack N,z_fs,pm_fs,pe_fs,ψE_fs,pg_fs,pc_fs = M;
            @unpack S,Tf,T,γg,γc,γo,γe,β=p;
            τg = τ[1];
            τc = τ[2];
            τo = τ[3];
            τe = τ[4];
            p = Par_Eprod(p; λ=2.2235012, ρ=d_elast, η=rscale);
            M = Model_Eprod(M; p = p);
            println("Demand elasticity = $(p.ρ), fuel substitution elasticity = $(p.λ)")
            # Update aggregate output price
            p,M = OutputPrice_func_Eprod(M,Data,p,τ);
            # Start static of social welfare
            pout = SharedArray{Float64}(T,S,Tf+1);
            gas = SharedArray{Float64}(N,S,Tf+1);
            coal = SharedArray{Float64}(N,S,Tf+1);
            oil = SharedArray{Float64}(N,S,Tf+1);
            elec = SharedArray{Float64}(N,S,Tf+1);
            # gasspend = SharedArray{Float64}(N,S,Tf+1);
            # coalspend = SharedArray{Float64}(N,S,Tf+1);
            # oilspend = SharedArray{Float64}(N,S,Tf+1);
            # elecspend = SharedArray{Float64}(N,S,Tf+1);
            # pE = SharedArray{Float64}(N,S,Tf+1);
            # pin = SharedArray{Float64}(N,S,Tf+1);
            y = SharedArray{Float64}(N,S,Tf+1);
            E = SharedArray{Float64}(N,S,Tf+1);
            # Lspend = SharedArray{Float64}(N,S,Tf+1);
            # Mspend = SharedArray{Float64}(N,S,Tf+1);
            # Krent = SharedArray{Float64}(N,S,Tf+1);
                δ = 0.0;
                r = 0.150838;
            # Kval = SharedArray{Float64}(N,S,Tf+1);
            # Espend = SharedArray{Float64}(N,S,Tf+1);
            profit = SharedArray{Float64}(N,S,Tf+1);
            println("      ")
            println("--------------------------------------------------------------")
            println("--------      BEGINNING FORWARD SIMULATION     ---------------")
            println("---------      Energy productivity model       ----------")
            println("---------      No Switching - No fuel prod - Tax (g,c,o,e) = $τ    ----------")
            println("--------------------------------------------------------------")
            println("      ")
            # Year of data: Get firm-level objects: output, fuel prices and productivity, fuel quantities
            for s = 1:S
                pout[:,s,1] = p.pout_struc[2:7];
            end
            for i = 1:N
                for s = 1:S
                    t = Data.year[i]-2009;
                    pm = exp(Data.logPm[i]);
                    z = exp(Data.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean;
                    pe = exp(Data.lnpelec_tilde[i]) + τe*p.egmean;
                    # energy productivity
                    ψE = exp(Data.lnfprod_E[i]);
                    # input prices indices
                    poψo = po;
                    peψe = pe;
                    if Data.combineF[i] == 12
                        pfψf = [poψo,peψe];
                        pE = (1/ψE)*pE_func(12,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 123
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc;
                        pfψf = [poψo,peψe,pcψc];
                        pE = (1/ψE)*pE_func(123,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 124
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg;
                        pfψf = [poψo,peψe,pgψg];
                        pE = (1/ψE)*pE_func(124,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 1234
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc;
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg;
                        pfψf = [poψo,peψe,pcψc,pgψg];
                        pE = (1/ψE)*pE_func(1234,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    end
                    # Output
                    y[i,s,1] = output_func_monopolistic(z,pin,p.pout_struc[t+1],t,p);
                    # profit
                    profit[i,s,1] = profit_func_monopolistic(z,pin,t,p);
                    # Energy
                    E[i,s,1] = ((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE)^p.σ);
                    # Fuel quantitiy
                    oil[i,s,1] = (p.ogmean*E[i,s,1])*((p.βo/po)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                    elec[i,s,1] = (p.egmean*E[i,s,1])*((p.βe/pe)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                    # oilspend[i,s,1] = (po/p.ogmean)*oil[i,s,1];
                    # elecspend[i,s,1] = (pe/p.egmean)*elec[i,s,1];
                    if Data.combineF[i] == 12
                        gas[i,s,1] = 0;
                        # gasspend[i,s,1]=0;
                        coal[i,s,1] = 0;
                        # coalspend[i,s,1]=0;
                    elseif Data.combineF[i] == 123
                        gas[i,s,1] = 0;
                        # gasspend[i,s,1]=0;
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                        # coalspend[i,s,1] = (pc/p.cgmean)*coal[i,s,1];
                    elseif Data.combineF[i] == 124
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                        # gasspend[i,s,1] = (pg/p.ggmean)*gas[i,s,1];
                        coal[i,s,1] = 0;
                        # coalspend[i,s,1]=0;
                    elseif Data.combineF[i] == 1234
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                        # gasspend[i,s,1] = (pg/p.ggmean)*gas[i,s,1];
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                        # coalspend[i,s,1] = (pc/p.cgmean)*coal[i,s,1];
                    end
                    # Input spending
                    # Espend[i,s,1] = pE*E[i,s,1];
                    # Mspend[i,s,1] = pm*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αm*pin[i,s,1]/pm)^p.σ);
                    # Lspend[i,s,1] = p.w[t+1]*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αl*pin[i,s,1]/p.w[t+1])^p.σ);
                    # Krent[i,s,1] = p.rk[t+1]*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,1]/p.rk[t+1])^p.σ);
                    # Kval[i,s,1] = (p.rk[t+1]/(δ+r))*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,1]/p.rk[t+1])^p.σ);
                end
            end
            # Years of forward simulation (starting one year ahead)
            @sync @distributed for j = 1:(Tf*S)
                tf,s = Tuple(CartesianIndices((Tf,S))[j]);
                # Get new aggregate price index
                Data_fs = deepcopy(Data);
                for i = 1:N
                    t = Data_fs.year[i]-2009;
                    Data_fs.lnz[i] = log(z_fs[i,s,tf]);
                    Data_fs.logPm[i] = log(pm_fs[i,s,tf]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1]; 
                    Data_fs.lnpelec_tilde[i] = log(pe_fs[i,s,tf]);
                    # Energy productivity
                    ψE = ψE_fs[i,s,tf]; Data_fs.lnfprod_E[i] = log(ψE);
                    # input prices indices
                    if Data.combineF[i] == 123
                        Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                    elseif Data.combineF[i] == 124
                        Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                    elseif Data.combineF[i] == 1234
                        Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                        Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                    end
                end
                p_fs,M = OutputPrice_func_Eprod(M,Data_fs,p,τ);
                pout[:,s,tf+1] = p_fs.pout_struc[2:7];
                # Get everything else
                for i = 1:N
                    t = Data_fs.year[i]-2009; 
                    pm = exp(Data_fs.logPm[i]);
                    z = exp(Data_fs.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean;
                    pe = exp(Data_fs.lnpelec_tilde[i]) + τe*p.egmean; 
                    pc = exp(Data_fs.lnpc_tilde[i]) + τc*p.cgmean;
                    pg = exp(Data_fs.lnpg_tilde[i]) + τg*p.ggmean;
                    # Energy productivity
                    ψE = ψE_fs[i,s,tf];
                    # input prices indices
                    poψo = po;
                    peψe = pe;
                    if Data.combineF[i] == 12
                        pfψf = [poψo,peψe];
                        pE = (1/ψE)*pE_func(12,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 123
                        pcψc = pc;
                        pfψf = [poψo,peψe,pcψc];
                        pE = (1/ψE)*pE_func(123,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 124
                        pgψg = pg;
                        pfψf = [poψo,peψe,pgψg];
                        pE = (1/ψE)*pE_func(124,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 1234
                        pcψc = pc;
                        pgψg = pg;
                        pfψf = [poψo,peψe,pcψc,pgψg];
                        pE = (1/ψE)*pE_func(1234,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    end
                    # Output
                    y[i,s,tf+1] = output_func_monopolistic(z,pin,p_fs.pout_struc[t+1],t,p_fs);
                    # profit
                    profit[i,s,tf+1] = profit_func_monopolistic(z,pin,t,p_fs);
                    # Energy
                    E[i,s,tf+1] = ((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE)^p.σ);
                    # Fuel quantitiy
                    oil[i,s,tf+1] = (p.ogmean*E[i,s,tf+1])*((p.βo/po)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                    elec[i,s,tf+1] = (p.egmean*E[i,s,tf+1])*((p.βe/pe)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                    # oilspend[i,s,tf+1] = (po/p.ogmean)*oil[i,s,tf+1];
                    # elecspend[i,s,tf+1] = (pe/p.egmean)*elec[i,s,tf+1];
                    if Data.combineF[i] == 12
                        gas[i,s,tf+1] = 0;
                        # gasspend[i,s,tf+1]=0;
                        coal[i,s,tf+1] = 0;
                        # coalspend[i,s,tf+1]=0;
                    elseif Data.combineF[i] == 123
                        gas[i,s,tf+1] = 0;
                        # gasspend[i,s,tf+1]=0;
                        coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc)^p.λ)*(ψE_fs[i,s,tf]^(p.λ-1))*(pE^p.λ);
                        # coalspend[i,s,tf+1] = (pc/p.cgmean)*coal[i,s,tf+1];
                    elseif Data.combineF[i] == 124
                        gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg)^p.λ)*(ψE_fs[i,s,tf]^(p.λ-1))*(pE^p.λ);
                        # gasspend[i,s,tf+1] = (pg/p.ggmean)*gas[i,s,tf+1];
                        coal[i,s,tf+1] = 0;
                        # coalspend[i,s,tf+1]=0;
                    elseif Data.combineF[i] == 1234
                        gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg)^p.λ)*(ψE_fs[i,s,tf]^(p.λ-1))*(pE^p.λ);
                        # gasspend[i,s,tf+1] = (pg/p.ggmean)*gas[i,s,tf+1];
                        coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc)^p.λ)*(ψE_fs[i,s,tf]^(p.λ-1))*(pE^p.λ);
                        # coalspend[i,s,tf+1] = (pc/p.cgmean)*coal[i,s,tf+1];
                    end
                    # Input spending
                    # Espend[i,s,tf+1] = pE*E[i,s,tf+1];
                    # Mspend[i,s,tf+1] = pm*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αm*pin[i,s,tf+1]/pm)^p.σ);
                    # Lspend[i,s,tf+1] = p.w[t+1]*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αl*pin[i,s,tf+1]/p.w[t+1])^p.σ);
                    # Krent[i,s,tf+1] = p.rk[t+1]*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,tf+1]/p.rk[t+1])^p.σ);
                    # Kval[i,s,tf+1] = (p.rk[t+1]/(δ+r))*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,tf+1]/p.rk[t+1])^p.σ);
                end
            end
            # Return aggregates by year forward (only look at last year in data - 2016)
            y_fs = zeros(Tf+1); E_fs = zeros(Tf+1); gas_fs = zeros(Tf+1); coal_fs = zeros(Tf+1); oil_fs = zeros(Tf+1); elec_fs = zeros(Tf+1); co2_fs = zeros(Tf+1); profit_fs = zeros(Tf+1); CS_fs = zeros(Tf+1);
            Data_year = groupby(Data,:year); Nt = size(Data_year[end])[1];
            # pE = pE[end-Nt+1:end,:,:];
            y = y[end-Nt+1:end,:,:];
            E = E[end-Nt+1:end,:,:];
            gas = gas[end-Nt+1:end,:,:];
            coal = coal[end-Nt+1:end,:,:];
            oil = oil[end-Nt+1:end,:,:];
            elec = elec[end-Nt+1:end,:,:];
            profit = profit[end-Nt+1:end,:,:];
            pout = pout[end,:,:];
            co2 = p.γg*gas .+ p.γc*coal .+ p.γo*oil .+ p.γe*elec;
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
            CS_s_fs = zeros(p.S,p.Tf+1);
            thetaterm = p.θ/(1-p.θ)
            for tf = 1:(p.Tf+1)
                for s = 1:p.S
                    CS_s_fs[s,tf] = thetaterm*(pout[s,tf]^-thetaterm);
                end
                CS_fs[tf] = mean(CS_s_fs[:,tf]);
            end
            # Average across simulations
            for tf = 1:Tf+1
                # pE_fs[tf] = mean(pE[:,:,tf]);
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
                end
            end
            return y_fs,E_fs,gas_fs,coal_fs,oil_fs,elec_fs,co2_fs,profit_fs,CS_fs;
            #return pE,pin,y,E,gas,gasspend,coal,coalspend,oil,oilspend,elec,elecspend,pout,Espend,Mspend,Lspend,Krent,Kval,profit;
            # return pE,pin,y,E,gas,coal,oil,elec,Espend,profit;
        end
        # No forward Simulation
        @everywhere function Welfare_noswitch_Eprod(M::Model_Eprod,Data::DataFrame,p::Par_Eprod,τ,d_elast::Float64=p.ρ,rscale::Float64=p.η,perfcompl=false)
            @unpack N,z_fs,pm_fs,pe_fs,ψE_fs,pg_fs,pc_fs = M;
            @unpack S,Tf,T,γg,γc,γo,γe,β=p;
            τg = τ[1];
            τc = τ[2];
            τo = τ[3];
            τe = τ[4];
            p = Par_Eprod(p; ρ=d_elast, η=rscale);
            M = Model_Eprod(M; p =p);
            # println("Demand elasticity = $(p.ρ)")
            # Update aggregate output price
            p,M = OutputPrice_func_Eprod(M,Data,p,τ);
            # Start static of social welfare
            # ψc_fs = SharedArray{Float64}(N,S,Tf);
            # ψg_fs = SharedArray{Float64}(N,S,Tf);
            pout = SharedArray{Float64}(T);
            gas = SharedArray{Float64}(N);
            coal = SharedArray{Float64}(N);
            oil = SharedArray{Float64}(N);
            elec = SharedArray{Float64}(N);
            # gasspend = SharedArray{Float64}(N,S,Tf+1);
            # coalspend = SharedArray{Float64}(N,S,Tf+1);
            # oilspend = SharedArray{Float64}(N,S,Tf+1);
            # elecspend = SharedArray{Float64}(N,S,Tf+1);
            # pE = SharedArray{Float64}(N,S,Tf+1);
            # pin = SharedArray{Float64}(N,S,Tf+1);
            y = SharedArray{Float64}(N);
            E = SharedArray{Float64}(N);
            # Lspend = SharedArray{Float64}(N,S,Tf+1);
            # Mspend = SharedArray{Float64}(N,S,Tf+1);
            # Krent = SharedArray{Float64}(N,S,Tf+1);
                δ = 0.0;
                r = 0.150838;
            # Kval = SharedArray{Float64}(N,S,Tf+1);
            # Espend = SharedArray{Float64}(N,S,Tf+1);
            profit = SharedArray{Float64}(N);
            # println("      ")
            # println("--------------------------------------------------------------")
            # println("--------      BEGINNING FORWARD SIMULATION     ---------------")
            # println("---------      Main model                ----------")
            # println("---------      No Switching - No fuel prod - Tax (g,c,o,e) = $τ    ----------")
            # println("--------------------------------------------------------------")
            # println("      ")
            # Year of data: Get firm-level objects: output, fuel prices and productivity, fuel quantities
            pout = p.pout_struc[2:7];
            for i = 1:N
                    t = Data.year[i]-2009;
                    pm = exp(Data.logPm[i]);
                    z = exp(Data.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean;
                    pe = exp(Data.lnpelec_tilde[i]) + τe*p.egmean;
                    # energy productivity
                    ψE = exp(Data.lnfprod_E[i]);
                    # input prices indices
                    poψo = po;
                    peψe = pe;
                    if Data.combineF[i] == 12
                        pfψf = [poψo,peψe];
                        pE = (1/ψE)*pE_func(12,pfψf,p);
                        pE1 = pE_func(12,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 123
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc;
                        pfψf = [poψo,peψe,pcψc];
                        pE = (1/ψE)*pE_func(123,pfψf,p);
                        pE1 = pE_func(123,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 124
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg;
                        pfψf = [poψo,peψe,pgψg];
                        pE = (1/ψE)*pE_func(124,pfψf,p);
                        pE1 = pE_func(124,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 1234
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc;
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg;
                        pfψf = [poψo,peψe,pcψc,pgψg];
                        pE = (1/ψE)*pE_func(1234,pfψf,p);
                        pE1 = pE_func(1234,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    end
                    # Output
                    y[i] = output_func_monopolistic(z,pin,p.pout_struc[t+1],t,p);
                    # profit
                    profit[i] = profit_func_monopolistic(z,pin,t,p);
                    # Energy
                    E[i] = ((y[i]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE)^p.σ);
                    # Fuel quantitiy
                    oil[i] = (p.ogmean*E[i])*((p.βo/po)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);

                    oil_test = (p.ogmean*E[i])*((p.βo/po)^p.λ)*(pE1^p.λ)/ψE;

                    # println(oil[i],oil_test)

                    if isapprox(oil[i],oil_test) == false
                        error("Math error")
                    end

                    elec[i] = (p.egmean*E[i])*((p.βe/pe)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                    # oilspend[i,s,1] = (po/p.ogmean)*oil[i,s,1];
                    # elecspend[i,s,1] = (pe/p.egmean)*elec[i,s,1];
                    if Data.combineF[i] == 12
                        gas[i] = 0;
                        # gasspend[i,s,1]=0;
                        coal[i] = 0;
                        # coalspend[i,s,1]=0;
                    elseif Data.combineF[i] == 123
                        gas[i] = 0;
                        # gasspend[i,s,1]=0;
                        coal[i] = (p.cgmean*E[i])*((p.βc/pc)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                        # coalspend[i,s,1] = (pc/p.cgmean)*coal[i,s,1];
                    elseif Data.combineF[i] == 124
                        gas[i] = (p.ggmean*E[i])*((p.βg/pg)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                        # gasspend[i,s,1] = (pg/p.ggmean)*gas[i,s,1];
                        coal[i] = 0;
                        # coalspend[i,s,1]=0;
                    elseif Data.combineF[i] == 1234
                        gas[i] = (p.ggmean*E[i])*((p.βg/pg)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                        # gasspend[i,s,1] = (pg/p.ggmean)*gas[i,s,1];
                        coal[i] = (p.cgmean*E[i])*((p.βc/pc)^p.λ)*(ψE^(p.λ-1))*(pE^p.λ);
                        # coalspend[i,s,1] = (pc/p.cgmean)*coal[i,s,1];
                    end
                    # Input spending
                    # Espend[i,s,1] = pE*E[i,s,1];
                    # Mspend[i,s,1] = pm*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αm*pin[i,s,1]/pm)^p.σ);
                    # Lspend[i,s,1] = p.w[t+1]*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αl*pin[i,s,1]/p.w[t+1])^p.σ);
                    # Krent[i,s,1] = p.rk[t+1]*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,1]/p.rk[t+1])^p.σ);
                    # Kval[i,s,1] = (p.rk[t+1]/(δ+r))*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,1]/p.rk[t+1])^p.σ);
            end
            # Return aggregates by year forward (only look at last year in data - 2016)
            Data_year = groupby(Data,:year); Nt = size(Data_year[end])[1];
            # pE = pE[end-Nt+1:end,:,:];
            y = y[(end-Nt+1):end];
            E = E[(end-Nt+1):end];
            gas = gas[(end-Nt+1):end];
            coal = coal[(end-Nt+1):end];
            oil = oil[(end-Nt+1):end];
            elec = elec[(end-Nt+1):end];
            profit = profit[(end-Nt+1):end];
            pout = pout[end];
            co2 = p.γg*gas .+ p.γc*coal .+ p.γo*oil .+ p.γe*elec;
            # Aggregate output (from rep consumer CES demand)
            Nfirms = Data_year[end].N[1];
            demand_shock = exp(p.d_t[end]);
            rhoterm = (p.ρ-1)/p.ρ;
            if perfcompl == true
                y_fs = demand_shock*minimum(y);
            else
                y_fs = ((demand_shock/Nfirms)*(sum(y[:].^rhoterm)))^(1/rhoterm);
            end
            # Consumer surplus
            thetaterm = p.θ/(1-p.θ)
            CS_fs = thetaterm*(pout^-thetaterm);
            # Other variables
            E_fs = sum(E);
            gas_fs = sum(gas);
            coal_fs = sum(coal);
            oil_fs = sum(oil);
            elec_fs = sum(elec);
            co2_fs = sum(co2);
            profit_fs = sum(profit);
            return y_fs,E_fs,gas_fs,coal_fs,oil_fs,elec_fs,co2_fs,profit_fs,CS_fs;
            # return pE,pin,y,E,gas,gasspend,coal,coalspend,oil,oilspend,elec,elecspend,pout,Espend,Mspend,Lspend,Krent,Kval,ψc_fs,ψg_fs,profit;
            # return pE,pin,y,E,gas,coal,oil,elec,Espend,profit;
        end
    #
    # 3. No fuel productivity 1 (everyone gets average fuel prod in average model)
        @everywhere function Welfare_ForwardSimul_noswitch_nfp(M::Model_nfp,Data::DataFrame,p::Par_nfp,τ)
            @unpack N,z_fs,pm_fs,pe_fs,ψe_fs,ψo_fs,pg_fs,pc_fs = M;
            @unpack S,Tf,T,γg,γc,γo,γe,β=p;
            τg = τ[1];
            τc = τ[2];
            τo = τ[3];
            τe = τ[4];
            # Update aggregate output price
            p,M = OutputPrice_func_nfp(M,Data,p,τ);
            # Start static of social welfare
            pout = SharedArray{Float64}(T,S,Tf+1);
            gas = SharedArray{Float64}(N,S,Tf+1);
            coal = SharedArray{Float64}(N,S,Tf+1);
            oil = SharedArray{Float64}(N,S,Tf+1);
            elec = SharedArray{Float64}(N,S,Tf+1);
            # gasspend = SharedArray{Float64}(N,S,Tf+1);
            # coalspend = SharedArray{Float64}(N,S,Tf+1);
            # oilspend = SharedArray{Float64}(N,S,Tf+1);
            # elecspend = SharedArray{Float64}(N,S,Tf+1);
            # pE = SharedArray{Float64}(N,S,Tf+1);
            # pin = SharedArray{Float64}(N,S,Tf+1);
            y = SharedArray{Float64}(N,S,Tf+1);
            E = SharedArray{Float64}(N,S,Tf+1);
            # Lspend = SharedArray{Float64}(N,S,Tf+1);
            # Mspend = SharedArray{Float64}(N,S,Tf+1);
            # Krent = SharedArray{Float64}(N,S,Tf+1);
                δ = 0.0;
                r = 0.150838;
            # Kval = SharedArray{Float64}(N,S,Tf+1);
            # Espend = SharedArray{Float64}(N,S,Tf+1);
            profit = SharedArray{Float64}(N,S,Tf+1);
            println("      ")
            println("--------------------------------------------------------------")
            println("--------      BEGINNING FORWARD SIMULATION     ---------------")
            println("---------      Main model - no fuel productivity               ----------")
            println("---------      No Switching - No fuel prod - Tax (g,c,o,e) = $τ    ----------")
            println("--------------------------------------------------------------")
            println("      ")
            # Average fuel productivity for everyone
            ψe = exp(mean(Data.lnfprod_e));
            ψo = exp(mean(Data.lnfprod_o));
            ψg = exp(mean(Data.lnfprod_g[Data.gas.>0]));
            ψc = exp(mean(Data.lnfprod_c[Data.coal.>0]));
            # Year of data: Get firm-level objects: output, fuel prices and productivity, fuel quantities
            for s = 1:S
                pout[:,s,1] = p.pout_struc[2:7];
            end
            for i = 1:N
                for s = 1:S
                    t = Data.year[i]-2009;
                    pm = exp(Data.logPm[i]);
                    z = exp(Data.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean;
                    pe = exp(Data.lnpelec_tilde[i]) + τe*p.egmean;
                    # input prices indices
                    poψo = po/ψo;
                    peψe = pe/ψe;
                    if Data.combineF[i] == 12
                        pfψf = [poψo,peψe];
                        pE = pE_func(12,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 123
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc/ψc;
                        pfψf = [poψo,peψe,pcψc];
                        pE = pE_func(123,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 124
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg/ψg;
                        pfψf = [poψo,peψe,pgψg];
                        pE = pE_func(124,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 1234
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc/ψc;
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg/ψg;
                        pfψf = [poψo,peψe,pcψc,pgψg];
                        pE = pE_func(1234,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    end
                    # Output
                    y[i,s,1] = output_func_monopolistic(z,pin,p.pout_struc[t+1],t,p);
                    # profit
                    profit[i,s,1] = profit_func_monopolistic(z,pin,t,p);
                    # Energy
                    E[i,s,1] = ((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE)^p.σ);
                    # Fuel quantitiy
                    oil[i,s,1] = (p.ogmean*E[i,s,1])*((p.βo/po)^p.λ)*(ψo^(p.λ-1))*(pE^p.λ);
                    elec[i,s,1] = (p.egmean*E[i,s,1])*((p.βe/pe)^p.λ)*(ψe^(p.λ-1))*(pE^p.λ);
                    # oilspend[i,s,1] = (po/p.ogmean)*oil[i,s,1];
                    # elecspend[i,s,1] = (pe/p.egmean)*elec[i,s,1];
                    if Data.combineF[i] == 12
                        gas[i,s,1] = 0;
                        # gasspend[i,s,1]=0;
                        coal[i,s,1] = 0;
                        # coalspend[i,s,1]=0;
                    elseif Data.combineF[i] == 123
                        gas[i,s,1] = 0;
                        # gasspend[i,s,1]=0;
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc)^p.λ)*(ψc^(p.λ-1))*(pE^p.λ);
                        # coalspend[i,s,1] = (pc/p.cgmean)*coal[i,s,1];
                    elseif Data.combineF[i] == 124
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg)^p.λ)*(ψg^(p.λ-1))*(pE^p.λ);
                        # gasspend[i,s,1] = (pg/p.ggmean)*gas[i,s,1];
                        coal[i,s,1] = 0;
                        # coalspend[i,s,1]=0;
                    elseif Data.combineF[i] == 1234
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg)^p.λ)*(ψg^(p.λ-1))*(pE^p.λ);
                        # gasspend[i,s,1] = (pg/p.ggmean)*gas[i,s,1];
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc)^p.λ)*(ψc^(p.λ-1))*(pE^p.λ);
                        # coalspend[i,s,1] = (pc/p.cgmean)*coal[i,s,1];
                    end
                    # Input spending
                    # Espend[i,s,1] = pE*E[i,s,1];
                    # Mspend[i,s,1] = pm*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αm*pin[i,s,1]/pm)^p.σ);
                    # Lspend[i,s,1] = p.w[t+1]*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αl*pin[i,s,1]/p.w[t+1])^p.σ);
                    # Krent[i,s,1] = p.rk[t+1]*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,1]/p.rk[t+1])^p.σ);
                    # Kval[i,s,1] = (p.rk[t+1]/(δ+r))*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,1]/p.rk[t+1])^p.σ);
                end
            end
            # Years of forward simulation (starting one year ahead)
            @sync @distributed for j = 1:(Tf*S)
                tf,s = Tuple(CartesianIndices((Tf,S))[j]);
                # Get new aggregate price index
                Data_fs = deepcopy(Data);
                for i = 1:N
                    t = Data_fs.year[i]-2009;
                    Data_fs.lnz[i] = log(z_fs[i,s,tf]);
                    Data_fs.logPm[i] = log(pm_fs[i,s,tf]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    # po = p.po[t+1]; 
                    Data_fs.lnpelec_tilde[i] = log(pe_fs[i,s,tf]);
                    # input prices indices
                    if Data.combineF[i] == 123
                        Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                    elseif Data.combineF[i] == 124
                        Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                    elseif Data.combineF[i] == 1234
                        Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                        Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                    end
                end
                p_fs,M = OutputPrice_func_nfp(M,Data_fs,p,τ);
                pout[:,s,tf+1] = p_fs.pout_struc[2:7];
                # Get everything else
                for i = 1:N
                    t = Data_fs.year[i]-2009; 
                    pm = exp(Data_fs.logPm[i]);
                    z = exp(Data_fs.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean;
                    pe = exp(Data_fs.lnpelec_tilde[i]) + τe*p.egmean; 
                    pc = exp(Data_fs.lnpc_tilde[i]) + τc*p.cgmean;
                    pg = exp(Data_fs.lnpg_tilde[i]) + τg*p.ggmean;
                    # input prices indices
                    poψo = po/ψo;
                    peψe = pe/ψe;
                    if Data.combineF[i] == 12
                        pfψf = [poψo,peψe];
                        pE = pE_func(12,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 123
                        pcψc = pc/ψc;
                        pfψf = [poψo,peψe,pcψc];
                        pE = pE_func(123,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 124
                        pgψg = pg/ψg;
                        pfψf = [poψo,peψe,pgψg];
                        pE = pE_func(124,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 1234
                        pcψc = pc/ψc;
                        pgψg = pg/ψg;
                        pfψf = [poψo,peψe,pcψc,pgψg];
                        pE = pE_func(1234,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    end
                    # Output
                    y[i,s,tf+1] = output_func_monopolistic(z,pin,p_fs.pout_struc[t+1],t,p_fs);
                    # profit
                    profit[i,s,tf+1] = profit_func_monopolistic(z,pin,t,p_fs);
                    # Energy
                    E[i,s,tf+1] = ((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE)^p.σ);
                    # Fuel quantitiy
                    oil[i,s,tf+1] = (p.ogmean*E[i,s,tf+1])*((p.βo/po)^p.λ)*(ψo^(p.λ-1))*(pE^p.λ);
                    elec[i,s,tf+1] = (p.egmean*E[i,s,tf+1])*((p.βe/pe)^p.λ)*(ψe^(p.λ-1))*(pE^p.λ);
                    # oilspend[i,s,tf+1] = (po/p.ogmean)*oil[i,s,tf+1];
                    # elecspend[i,s,tf+1] = (pe/p.egmean)*elec[i,s,tf+1];
                    if Data.combineF[i] == 12
                        gas[i,s,tf+1] = 0;
                        # gasspend[i,s,tf+1]=0;
                        coal[i,s,tf+1] = 0;
                        # coalspend[i,s,tf+1]=0;
                    elseif Data.combineF[i] == 123
                        gas[i,s,tf+1] = 0;
                        # gasspend[i,s,tf+1]=0;
                        coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc)^p.λ)*(ψc^(p.λ-1))*(pE^p.λ);
                        # coalspend[i,s,tf+1] = (pc/p.cgmean)*coal[i,s,tf+1];
                    elseif Data.combineF[i] == 124
                        gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg)^p.λ)*(ψg^(p.λ-1))*(pE^p.λ);
                        # gasspend[i,s,tf+1] = (pg/p.ggmean)*gas[i,s,tf+1];
                        coal[i,s,tf+1] = 0;
                        # coalspend[i,s,tf+1]=0;
                    elseif Data.combineF[i] == 1234
                        gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg)^p.λ)*(ψg^(p.λ-1))*(pE^p.λ);
                        # gasspend[i,s,tf+1] = (pg/p.ggmean)*gas[i,s,tf+1];
                        coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc)^p.λ)*(ψc^(p.λ-1))*(pE^p.λ);
                        # coalspend[i,s,tf+1] = (pc/p.cgmean)*coal[i,s,tf+1];
                    end
                    # Input spending
                    # Espend[i,s,tf+1] = pE*E[i,s,tf+1];
                    # Mspend[i,s,tf+1] = pm*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αm*pin[i,s,tf+1]/pm)^p.σ);
                    # Lspend[i,s,tf+1] = p.w[t+1]*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αl*pin[i,s,tf+1]/p.w[t+1])^p.σ);
                    # Krent[i,s,tf+1] = p.rk[t+1]*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,tf+1]/p.rk[t+1])^p.σ);
                    # Kval[i,s,tf+1] = (p.rk[t+1]/(δ+r))*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,tf+1]/p.rk[t+1])^p.σ);
                end
            end
            # Return aggregates by year forward (only look at last year in data - 2016)
            y_fs = zeros(Tf+1); E_fs = zeros(Tf+1); gas_fs = zeros(Tf+1); coal_fs = zeros(Tf+1); oil_fs = zeros(Tf+1); elec_fs = zeros(Tf+1); co2_fs = zeros(Tf+1); profit_fs = zeros(Tf+1); CS_fs = zeros(Tf+1);
            Data_year = groupby(Data,:year); Nt = size(Data_year[end])[1];
            # pE = pE[end-Nt+1:end,:,:];
            y = y[end-Nt+1:end,:,:];
            E = E[end-Nt+1:end,:,:];
            gas = gas[end-Nt+1:end,:,:];
            coal = coal[end-Nt+1:end,:,:];
            oil = oil[end-Nt+1:end,:,:];
            elec = elec[end-Nt+1:end,:,:];
            profit = profit[(end-Nt+1):end,:,:];
            pout = pout[end,:,:];
            co2 = p.γg*gas .+ p.γc*coal .+ p.γo*oil .+ p.γe*elec;
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
            CS_s_fs = zeros(p.S,p.Tf+1);
            thetaterm = p.θ/(1-p.θ)
            for tf = 1:(p.Tf+1)
                for s = 1:p.S
                    CS_s_fs[s,tf] = thetaterm*(pout[s,tf]^-thetaterm);
                end
                CS_fs[tf] = mean(CS_s_fs[:,tf]);
            end
            # Average across simulations
            for tf = 1:Tf+1
                # pE_fs[tf] = mean(pE[:,:,tf]);
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
                end
            end
            return y_fs,E_fs,gas_fs,coal_fs,oil_fs,elec_fs,co2_fs,profit_fs,CS_fs;
            # return pE,pin,y,E,gas,gasspend,coal,coalspend,oil,oilspend,elec,elecspend,pout,Espend,Mspend,Lspend,Krent,Kval,profit;
            # return pE,pin,y,E,gas,coal,oil,elec,Espend,profit;
        end
    #
    # 4. No fuel productivity 2 (re-estimating model with simple CES)
        @everywhere function Welfare_ForwardSimul_noswitch_simpleCES(M::Model_simpleCES,Data::DataFrame,p::Par_simpleCES,τ)
            @unpack N,z_fs,pm_fs,pe_fs,pg_fs,pc_fs = M;
            @unpack S,Tf,T,γg,γc,γo,γe,β=p;
            τg = τ[1];
            τc = τ[2];
            τo = τ[3];
            τe = τ[4];
            # Update aggregate output price
            p,M = OutputPrice_func_simpleCES(M,Data,p,τ);
            # Start static of social welfare
            pout = SharedArray{Float64}(T,S,Tf+1);
            gas = SharedArray{Float64}(N,S,Tf+1);
            coal = SharedArray{Float64}(N,S,Tf+1);
            oil = SharedArray{Float64}(N,S,Tf+1);
            elec = SharedArray{Float64}(N,S,Tf+1);
            # gasspend = SharedArray{Float64}(N,S,Tf+1);
            # coalspend = SharedArray{Float64}(N,S,Tf+1);
            # oilspend = SharedArray{Float64}(N,S,Tf+1);
            # elecspend = SharedArray{Float64}(N,S,Tf+1);
            # pE = SharedArray{Float64}(N,S,Tf+1);
            # pin = SharedArray{Float64}(N,S,Tf+1);
            y = SharedArray{Float64}(N,S,Tf+1);
            E = SharedArray{Float64}(N,S,Tf+1);
            # Lspend = SharedArray{Float64}(N,S,Tf+1);
            # Mspend = SharedArray{Float64}(N,S,Tf+1);
            # Krent = SharedArray{Float64}(N,S,Tf+1);
                δ = 0.0;
                r = 0.150838;
            # Kval = SharedArray{Float64}(N,S,Tf+1);
            # Espend = SharedArray{Float64}(N,S,Tf+1);
            profit = SharedArray{Float64}(N,S,Tf+1);
            println("      ")
            println("--------------------------------------------------------------")
            println("--------      BEGINNING FORWARD SIMULATION     ---------------")
            println("---------      Simple CES      ----------")
            println("---------      No Switching - No fuel prod - Tax (g,c,o,e) = $τ    ----------")
            println("--------------------------------------------------------------")
            println("      ")
            # Year of data: Get firm-level objects: output, fuel prices and productivity, fuel quantities
            for s = 1:S
                pout[:,s,1] = p.pout_struc[2:7];
            end
            for i = 1:N
                for s = 1:S
                    t = Data.year[i]-2009;
                    pm = exp(Data.logPm[i]);
                    z = exp(Data.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean;
                    pe = exp(Data.lnpelec_tilde[i]) + τe*p.egmean;
                    # input prices indices
                    poψo = po;
                    peψe = pe;
                    if Data.combineF[i] == 12
                        pfψf = [poψo,peψe];
                        pE = pE_func(12,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 123
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc;
                        pfψf = [poψo,peψe,pcψc];
                        pE = pE_func(123,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 124
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg;
                        pfψf = [poψo,peψe,pgψg];
                        pE = pE_func(124,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 1234
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc;
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg;
                        pfψf = [poψo,peψe,pcψc,pgψg];
                        pE = pE_func(1234,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    end
                    # Output
                    y[i,s,1] = output_func_monopolistic(z,pin,p.pout_struc[t+1],t,p);
                    # profit
                    profit[i,s,1] = profit_func_monopolistic(z,pin,t,p);
                    # Energy
                    E[i,s,1] = ((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE)^p.σ);
                    # Fuel quantitiy
                    oil[i,s,1] = (p.ogmean*E[i,s,1])*((p.βo/po)^p.λ)*(pE^p.λ);
                    elec[i,s,1] = (p.egmean*E[i,s,1])*((p.βe/pe)^p.λ)*(pE^p.λ);
                    # oilspend[i,s,1] = (po/p.ogmean)*oil[i,s,1];
                    # elecspend[i,s,1] = (pe/p.egmean)*elec[i,s,1];
                    if Data.combineF[i] == 12
                        gas[i,s,1] = 0;
                        # gasspend[i,s,1]=0;
                        coal[i,s,1] = 0;
                        # coalspend[i,s,1]=0;
                    elseif Data.combineF[i] == 123
                        gas[i,s,1] = 0;
                        # gasspend[i,s,1]=0;
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc)^p.λ)*(pE^p.λ);
                        # coalspend[i,s,1] = (pc/p.cgmean)*coal[i,s,1];
                    elseif Data.combineF[i] == 124
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg)^p.λ)*(pE^p.λ);
                        # gasspend[i,s,1] = (pg/p.ggmean)*gas[i,s,1];
                        coal[i,s,1] = 0;
                        # coalspend[i,s,1]=0;
                    elseif Data.combineF[i] == 1234
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg)^p.λ)*(pE^p.λ);
                        # gasspend[i,s,1] = (pg/p.ggmean)*gas[i,s,1];
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc)^p.λ)*(pE^p.λ);
                        # coalspend[i,s,1] = (pc/p.cgmean)*coal[i,s,1];
                    end
                    # Input spending
                    # Espend[i,s,1] = pE*E[i,s,1];
                    # Mspend[i,s,1] = pm*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αm*pin[i,s,1]/pm)^p.σ);
                    # Lspend[i,s,1] = p.w[t+1]*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αl*pin[i,s,1]/p.w[t+1])^p.σ);
                    # Krent[i,s,1] = p.rk[t+1]*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,1]/p.rk[t+1])^p.σ);
                    # Kval[i,s,1] = (p.rk[t+1]/(δ+r))*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,1]/p.rk[t+1])^p.σ);
                end
            end
            # Years of forward simulation (starting one year ahead)
            @sync @distributed for j = 1:(Tf*S)
                tf,s = Tuple(CartesianIndices((Tf,S))[j]);
                # Get new aggregate price index
                Data_fs = deepcopy(Data);
                for i = 1:N
                    t = Data_fs.year[i]-2009;
                    Data_fs.lnz[i] = log(z_fs[i,s,tf]);
                    Data_fs.logPm[i] = log(pm_fs[i,s,tf]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1]; 
                    Data_fs.lnpelec_tilde[i] = log(pe_fs[i,s,tf]);
                    # input prices indices
                    if Data.combineF[i] == 123
                        Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                    elseif Data.combineF[i] == 124
                        Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                    elseif Data.combineF[i] == 1234
                        Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                        Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                    end
                end
                p_fs,M = OutputPrice_func_simpleCES(M,Data_fs,p,τ);
                pout[:,s,tf+1] = p_fs.pout_struc[2:7];
                # Get everything else
                for i = 1:N
                    t = Data_fs.year[i]-2009; 
                    pm = exp(Data_fs.logPm[i]);
                    z = exp(Data_fs.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean;
                    pe = exp(Data_fs.lnpelec_tilde[i]) + τe*p.egmean; 
                    pc = exp(Data_fs.lnpc_tilde[i]) + τc*p.cgmean;
                    pg = exp(Data_fs.lnpg_tilde[i]) + τg*p.ggmean;
                    # input prices indices
                    poψo = po;
                    peψe = pe;
                    if Data.combineF[i] == 12
                        pfψf = [poψo,peψe];
                        pE = pE_func(12,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 123
                        pcψc = pc;
                        pfψf = [poψo,peψe,pcψc];
                        pE = pE_func(123,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 124
                        pgψg = pg;
                        pfψf = [poψo,peψe,pgψg];
                        pE = pE_func(124,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 1234
                        pcψc = pc;
                        pgψg = pg;
                        pfψf = [poψo,peψe,pcψc,pgψg];
                        pE = pE_func(1234,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    end
                    # Output
                    y[i,s,tf+1] = output_func_monopolistic(z,pin,p_fs.pout_struc[t+1],t,p_fs);
                    # profit
                    profit[i,s,tf+1] = profit_func_monopolistic(z,pin,t,p_fs);
                    # Energy
                    E[i,s,tf+1] = ((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE)^p.σ);
                    # Fuel quantitiy
                    oil[i,s,tf+1] = (p.ogmean*E[i,s,tf+1])*((p.βo/po)^p.λ)*(pE^p.λ);
                    elec[i,s,tf+1] = (p.egmean*E[i,s,tf+1])*((p.βe/pe)^p.λ)*(pE^p.λ);
                    # oilspend[i,s,tf+1] = (po/p.ogmean)*oil[i,s,tf+1];
                    # elecspend[i,s,tf+1] = (pe/p.egmean)*elec[i,s,tf+1];
                    if Data.combineF[i] == 12
                        gas[i,s,tf+1] = 0;
                        # gasspend[i,s,tf+1]=0;
                        coal[i,s,tf+1] = 0;
                        # coalspend[i,s,tf+1]=0;
                    elseif Data.combineF[i] == 123
                        gas[i,s,tf+1] = 0;
                        # gasspend[i,s,tf+1]=0;
                        coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc)^p.λ)*(pE^p.λ);
                        # coalspend[i,s,tf+1] = (pc/p.cgmean)*coal[i,s,tf+1];
                    elseif Data.combineF[i] == 124
                        gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg)^p.λ)*(pE^p.λ);
                        # gasspend[i,s,tf+1] = (pg/p.ggmean)*gas[i,s,tf+1];
                        coal[i,s,tf+1] = 0;
                        # coalspend[i,s,tf+1]=0;
                    elseif Data.combineF[i] == 1234
                        gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg)^p.λ)*(pE^p.λ);
                        # gasspend[i,s,tf+1] = (pg/p.ggmean)*gas[i,s,tf+1];
                        coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc)^p.λ)*(pE^p.λ);
                        # coalspend[i,s,tf+1] = (pc/p.cgmean)*coal[i,s,tf+1];
                    end
                    # Input spending
                    # Espend[i,s,tf+1] = pE*E[i,s,tf+1];
                    # Mspend[i,s,tf+1] = pm*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αm*pin[i,s,tf+1]/pm)^p.σ);
                    # Lspend[i,s,tf+1] = p.w[t+1]*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αl*pin[i,s,tf+1]/p.w[t+1])^p.σ);
                    # Krent[i,s,tf+1] = p.rk[t+1]*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,tf+1]/p.rk[t+1])^p.σ);
                    # Kval[i,s,tf+1] = (p.rk[t+1]/(δ+r))*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,tf+1]/p.rk[t+1])^p.σ);
                end
            end
            # Return aggregates by year forward (only look at last year in data - 2016)
            y_fs = zeros(Tf+1); E_fs = zeros(Tf+1); gas_fs = zeros(Tf+1); coal_fs = zeros(Tf+1); oil_fs = zeros(Tf+1); elec_fs = zeros(Tf+1); co2_fs = zeros(Tf+1); profit_fs = zeros(Tf+1); CS_fs = zeros(Tf+1);
            Data_year = groupby(Data,:year); Nt = size(Data_year[end])[1];
            # pE = pE[end-Nt+1:end,:,:];
            y = y[end-Nt+1:end,:,:];
            E = E[end-Nt+1:end,:,:];
            gas = gas[end-Nt+1:end,:,:];
            coal = coal[end-Nt+1:end,:,:];
            oil = oil[end-Nt+1:end,:,:];
            elec = elec[end-Nt+1:end,:,:];
            profit = profit[end-Nt+1:end,:,:];
            pout = pout[end,:,:];
            co2 = p.γg*gas .+ p.γc*coal .+ p.γo*oil .+ p.γe*elec;
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
            CS_s_fs = zeros(p.S,p.Tf+1);
            thetaterm = p.θ/(1-p.θ)
            for tf = 1:(p.Tf+1)
                for s = 1:p.S
                    CS_s_fs[s,tf] = thetaterm*(pout[s,tf]^-thetaterm);
                end
                CS_fs[tf] = mean(CS_s_fs[:,tf]);
            end
            # Average across simulations
            for tf = 1:Tf+1
                # pE_fs[tf] = mean(pE[:,:,tf]);
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
                end
            end
            return y_fs,E_fs,gas_fs,coal_fs,oil_fs,elec_fs,co2_fs,profit_fs,CS_fs;
            # return pE,pin,y,E,gas,gasspend,coal,coalspend,oil,oilspend,elec,elecspend,pout,Espend,Mspend,Lspend,Krent,Kval,profit;
            # return pE,pin,y,E,gas,coal,oil,elec,Espend,profit;
        end
    #
    # 5. No fuel productivity 3 (re-estimating model with simple CES - keeping elasticity of substitution to main model)
        @everywhere function Welfare_ForwardSimul_noswitch_simpleCES_fixsub(M::Model_simpleCES,Data::DataFrame,p::Par_simpleCES,τ)
            @unpack N,z_fs,pm_fs,pe_fs,pg_fs,pc_fs = M;
            @unpack S,Tf,T,γg,γc,γo,γe,β=p;
            τg = τ[1];
            τc = τ[2];
            τo = τ[3];
            τe = τ[4];
            # Fix elasticity of substitution
            p = Par_simpleCES(p; σ=1.7999572,λ=2.2235012);
            # Update aggregate output price
            p,M = OutputPrice_func_simpleCES(M,Data,p,τ);
            # Start static of social welfare
            pout = SharedArray{Float64}(T,S,Tf+1);
            gas = SharedArray{Float64}(N,S,Tf+1); 
            coal = SharedArray{Float64}(N,S,Tf+1);
            oil = SharedArray{Float64}(N,S,Tf+1);
            elec = SharedArray{Float64}(N,S,Tf+1);
            # gasspend = SharedArray{Float64}(N,S,Tf+1);
            # coalspend = SharedArray{Float64}(N,S,Tf+1);
            # oilspend = SharedArray{Float64}(N,S,Tf+1);
            # elecspend = SharedArray{Float64}(N,S,Tf+1);
            # pE = SharedArray{Float64}(N,S,Tf+1);
            # pin = SharedArray{Float64}(N,S,Tf+1);
            y = SharedArray{Float64}(N,S,Tf+1);
            E = SharedArray{Float64}(N,S,Tf+1);
            # Lspend = SharedArray{Float64}(N,S,Tf+1);
            # Mspend = SharedArray{Float64}(N,S,Tf+1);
            # Krent = SharedArray{Float64}(N,S,Tf+1);
                δ = 0.0;
                r = 0.150838;
            # Kval = SharedArray{Float64}(N,S,Tf+1);
            # Espend = SharedArray{Float64}(N,S,Tf+1);
            profit = SharedArray{Float64}(N,S,Tf+1);
            CS = SharedArray{Float64}(p.T,S,Tf+1);
            println("      ")
            println("--------------------------------------------------------------")
            println("--------      BEGINNING FORWARD SIMULATION     ---------------")
            println("---------      Simple CES - Fixed substitution elasticity       ----------")
            println("---------      No Switching - No fuel prod - Tax (g,c,o,e) = $τ    ----------")
            println("--------------------------------------------------------------")
            println("      ")
            # Year of data: Get firm-level objects: output, fuel prices and productivity, fuel quantities
            for s = 1:S
                pout[:,s,1] = p.pout_struc[2:7];
            end
            for i = 1:N
                for s = 1:S
                    t = Data.year[i]-2009;
                    pm = exp(Data.logPm[i]);
                    z = exp(Data.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean;
                    pe = exp(Data.lnpelec_tilde[i]) + τe*p.egmean;
                    # input prices indices
                    poψo = po;
                    peψe = pe;
                    if Data.combineF[i] == 12
                        pfψf = [poψo,peψe];
                        pE = pE_func(12,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 123
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc;
                        pfψf = [poψo,peψe,pcψc];
                        pE = pE_func(123,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 124
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg;
                        pfψf = [poψo,peψe,pgψg];
                        pE = pE_func(124,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 1234
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean;
                        pcψc = pc;
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean;
                        pgψg = pg;
                        pfψf = [poψo,peψe,pcψc,pgψg];
                        pE = pE_func(1234,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    end
                    # Output
                    y[i,s,1] = output_func_monopolistic(z,pin,p.pout_struc[t+1],t,p);
                    # profit
                    profit[i,s,1] = profit_func_monopolistic(z,pin,t,p);
                    # Energy
                    E[i,s,1] = ((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE)^p.σ);
                    # Fuel quantitiy
                    oil[i,s,1] = (p.ogmean*E[i,s,1])*((p.βo/po)^p.λ)*(pE^p.λ);
                    elec[i,s,1] = (p.egmean*E[i,s,1])*((p.βe/pe)^p.λ)*(pE^p.λ);
                    # oilspend[i,s,1] = (po/p.ogmean)*oil[i,s,1];
                    # elecspend[i,s,1] = (pe/p.egmean)*elec[i,s,1];
                    if Data.combineF[i] == 12
                        gas[i,s,1] = 0;
                        # gasspend[i,s,1]=0;
                        coal[i,s,1] = 0;
                        # coalspend[i,s,1]=0;
                    elseif Data.combineF[i] == 123
                        gas[i,s,1] = 0;
                        # gasspend[i,s,1]=0;
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc)^p.λ)*(pE^p.λ);
                        # coalspend[i,s,1] = (pc/p.cgmean)*coal[i,s,1];
                    elseif Data.combineF[i] == 124
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg)^p.λ)*(pE^p.λ);
                        # gasspend[i,s,1] = (pg/p.ggmean)*gas[i,s,1];
                        coal[i,s,1] = 0;
                        # coalspend[i,s,1]=0;
                    elseif Data.combineF[i] == 1234
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg)^p.λ)*(pE^p.λ);
                        # gasspend[i,s,1] = (pg/p.ggmean)*gas[i,s,1];
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc)^p.λ)*(pE^p.λ);
                        # coalspend[i,s,1] = (pc/p.cgmean)*coal[i,s,1];
                    end
                    # Input spending
                    # Espend[i,s,1] = pE*E[i,s,1];
                    # Mspend[i,s,1] = pm*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αm*pin[i,s,1]/pm)^p.σ);
                    # Lspend[i,s,1] = p.w[t+1]*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αl*pin[i,s,1]/p.w[t+1])^p.σ);
                    # Krent[i,s,1] = p.rk[t+1]*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,1]/p.rk[t+1])^p.σ);
                    # Kval[i,s,1] = (p.rk[t+1]/(δ+r))*((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,1]/p.rk[t+1])^p.σ);
                end
            end
            # Years of forward simulation (starting one year ahead)
            @sync @distributed for j = 1:(Tf*S)
                tf,s = Tuple(CartesianIndices((Tf,S))[j]);
                # Get new aggregate price index
                Data_fs = deepcopy(Data);
                for i = 1:N
                    t = Data_fs.year[i]-2009;
                    Data_fs.lnz[i] = log(z_fs[i,s,tf]);
                    Data_fs.logPm[i] = log(pm_fs[i,s,tf]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    # po = p.po[t+1]; 
                    Data_fs.lnpelec_tilde[i] = log(pe_fs[i,s,tf]);
                    if Data.combineF[i] == 123
                        Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                    elseif Data.combineF[i] == 124
                        Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                    elseif Data.combineF[i] == 1234
                        Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                        Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                    end
                end
                p_fs,M = OutputPrice_func_simpleCES(M,Data_fs,p,τ);
                pout[:,s,tf+1] = p_fs.pout_struc[2:7];
                # Get everything else
                for i = 1:N
                    t = Data_fs.year[i]-2009; 
                    pm = exp(Data_fs.logPm[i]);
                    z = exp(Data_fs.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean;
                    pe = exp(Data_fs.lnpelec_tilde[i]) + τe*p.egmean; 
                    pc = exp(Data_fs.lnpc_tilde[i]) + τc*p.cgmean;
                    pg = exp(Data_fs.lnpg_tilde[i]) + τg*p.ggmean;
                    # input prices indices
                    poψo = po;
                    peψe = pe;
                    if Data.combineF[i] == 12
                        pfψf = [poψo,peψe];
                        pE = pE_func(12,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 123
                        pcψc = pc;
                        pfψf = [poψo,peψe,pcψc];
                        pE = pE_func(123,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 124
                        pgψg = pg;
                        pfψf = [poψo,peψe,pgψg];
                        pE= pE_func(124,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 1234
                        pcψc = pc;
                        pgψg = pg;
                        pfψf = [poψo,peψe,pcψc,pgψg];
                        pE= pE_func(1234,pfψf,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    end
                    # Output
                    y[i,s,tf+1] = output_func_monopolistic(z,pin,p_fs.pout_struc[t+1],t,p_fs);
                    # profit
                    profit[i,s,tf+1] = profit_func_monopolistic(z,pin,t,p_fs);
                    # Energy
                    E[i,s,tf+1] = ((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE)^p.σ);
                    # Fuel quantitiy
                    oil[i,s,tf+1] = (p.ogmean*E[i,s,tf+1])*((p.βo/po)^p.λ)*(pE^p.λ);
                    elec[i,s,tf+1] = (p.egmean*E[i,s,tf+1])*((p.βe/pe)^p.λ)*(pE^p.λ);
                    # oilspend[i,s,tf+1] = (po/p.ogmean)*oil[i,s,tf+1];
                    # elecspend[i,s,tf+1] = (pe/p.egmean)*elec[i,s,tf+1];
                    if Data.combineF[i] == 12
                        gas[i,s,tf+1] = 0;
                        # gasspend[i,s,tf+1]=0;
                        coal[i,s,tf+1] = 0;
                        # coalspend[i,s,tf+1]=0;
                    elseif Data.combineF[i] == 123
                        gas[i,s,tf+1] = 0;
                        # gasspend[i,s,tf+1]=0;
                        coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc)^p.λ)*(pE^p.λ);
                        # coalspend[i,s,tf+1] = (pc/p.cgmean)*coal[i,s,tf+1];
                    elseif Data.combineF[i] == 124
                        gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg)^p.λ)*(pE^p.λ);
                        # gasspend[i,s,tf+1] = (pg/p.ggmean)*gas[i,s,tf+1];
                        coal[i,s,tf+1] = 0;
                        # coalspend[i,s,tf+1]=0;
                    elseif Data.combineF[i] == 1234
                        gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg)^p.λ)*(pE^p.λ);
                        # gasspend[i,s,tf+1] = (pg/p.ggmean)*gas[i,s,tf+1];
                        coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc)^p.λ)*(pE^p.λ);
                        # coalspend[i,s,tf+1] = (pc/p.cgmean)*coal[i,s,tf+1];
                    end
                    # Input spending
                    # Espend[i,s,tf+1] = pE*E[i,s,tf+1];
                    # Mspend[i,s,tf+1] = pm*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αm*pin[i,s,tf+1]/pm)^p.σ);
                    # Lspend[i,s,tf+1] = p.w[t+1]*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αl*pin[i,s,tf+1]/p.w[t+1])^p.σ);
                    # Krent[i,s,tf+1] = p.rk[t+1]*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,tf+1]/p.rk[t+1])^p.σ);
                    # Kval[i,s,tf+1] = (p.rk[t+1]/(δ+r))*((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αk*pin[i,s,tf+1]/p.rk[t+1])^p.σ);
                end
            end
            # Return aggregates by year forward (only look at last year in data - 2016)
            y_fs = zeros(Tf+1); E_fs = zeros(Tf+1); gas_fs = zeros(Tf+1); coal_fs = zeros(Tf+1); oil_fs = zeros(Tf+1); elec_fs = zeros(Tf+1); co2_fs = zeros(Tf+1); profit_fs = zeros(Tf+1); CS_fs = zeros(Tf+1);
            Data_year = groupby(Data,:year); Nt = size(Data_year[end])[1];
            # pE = pE[end-Nt+1:end,:,:];
            y = y[end-Nt+1:end,:,:];
            E = E[end-Nt+1:end,:,:];
            gas = gas[end-Nt+1:end,:,:];
            coal = coal[end-Nt+1:end,:,:];
            oil = oil[end-Nt+1:end,:,:];
            elec = elec[end-Nt+1:end,:,:];
            profit = profit[(end-Nt+1):end,:,:];
            pout = pout[end,:,:];
            co2 = p.γg*gas .+ p.γc*coal .+ p.γo*oil .+ p.γe*elec;
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
            CS_s_fs = zeros(p.S,p.Tf+1);
            thetaterm = p.θ/(1-p.θ)
            for tf = 1:(p.Tf+1)
                for s = 1:p.S
                    CS_s_fs[s,tf] = thetaterm*(pout[s,tf]^-thetaterm);
                end
                CS_fs[tf] = mean(CS_s_fs[:,tf]);
            end
            # Average across simulations
            for tf = 1:Tf+1
                # pE_fs[tf] = mean(pE[:,:,tf]);
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
                end
            end
            return y_fs,E_fs,gas_fs,coal_fs,oil_fs,elec_fs,co2_fs,profit_fs,CS_fs;
            # return pE,pin,y,E,gas,gasspend,coal,coalspend,oil,oilspend,elec,elecspend,pout,Espend,Mspend,Lspend,Krent,Kval,profit;
            # return pE,pin,y,E,gas,coal,oil,elec,Espend,profit;
        end
    #
    # 6. Energy productivity with no fuel substitution
        @everywhere function Welfare_ForwardSimul_noswitch_Eprod_nofuelsub(M::Model_Eprod,Data::DataFrame,p::Par_Eprod,τ,d_elast::Float64=p.ρ,rscale::Float64=p.η)
            @unpack N,z_fs,pm_fs,pe_fs,ψE_fs,pg_fs,pc_fs = M;
            @unpack S,Tf,T,γg,γc,γo,γe,β=p;
            τg = τ[1];
            τc = τ[2];
            τo = τ[3];
            τe = τ[4];
            p = Par_Eprod(p; ρ=d_elast, η=rscale);
            M = Model_Eprod(M; p = p);
            println("Demand elasticity = $(p.ρ)")
            # Update aggregate output price
            p,M = OutputPrice_func_Eprod(M,Data,p,τ);
            # Start static of social welfare
            pout = SharedArray{Float64}(T,S,Tf+1);
            gas = SharedArray{Float64}(N,S,Tf+1);
            coal = SharedArray{Float64}(N,S,Tf+1);
            oil = SharedArray{Float64}(N,S,Tf+1);
            elec = SharedArray{Float64}(N,S,Tf+1);
            y = SharedArray{Float64}(N,S,Tf+1);
            E = SharedArray{Float64}(N,S,Tf+1);
                δ = 0.0;
                r = 0.150838;
            profit = SharedArray{Float64}(N,S,Tf+1);
            println("      ")
            println("--------------------------------------------------------------")
            println("--------      BEGINNING FORWARD SIMULATION     ---------------")
            println("---------      Energy productivity model       ----------")
            println("---------      No Switching - No fuel prod - Tax (g,c,o,e) = $τ    ----------")
            println("--------------------------------------------------------------")
            println("      ")
            # Year of data: Get firm-level objects: output, fuel prices and productivity, fuel quantities
            for s = 1:S
                pout[:,s,1] = p.pout_struc[2:7];
            end
            for i = 1:N
                for s = 1:S
                    t = Data.year[i]-2009;
                    pm = exp(Data.logPm[i]);
                    z = exp(Data.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean; po_nt = p.po[t+1];
                    pe = exp(Data.lnpelec_tilde[i]) + τe*p.egmean; pe_nt = exp(Data.lnpelec_tilde[i]);
                    # energy productivity
                    ψE = exp(Data.lnfprod_E[i]);
                    # input prices indices
                    poψo = po; poψo_nt = po_nt;
                    peψe = pe; peψe_nt = pe_nt;
                    if Data.combineF[i] == 12
                        pfψf = [poψo,peψe]; pfψf_nt = [poψo_nt,peψe_nt];
                        pE = (1/ψE)*pE_func(12,pfψf,p); pE_nt = (1/ψE)*pE_func(12,pfψf_nt,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 123
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean; pc_nt = exp(Data.lnpc_tilde[i]);
                        pcψc = pc; pcψc_nt = pc_nt;
                        pfψf = [poψo,peψe,pcψc]; pfψf_nt = [poψo_nt,peψe_nt,pcψc_nt];
                        pE = (1/ψE)*pE_func(123,pfψf,p); pE_nt = (1/ψE)*pE_func(123,pfψf_nt,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 124
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean; pg_nt = exp(Data.lnpg_tilde[i]);
                        pgψg = pg; pgψg_nt = pg_nt;
                        pfψf = [poψo,peψe,pgψg]; pfψf_nt = [poψo_nt,peψe_nt,pgψg_nt];
                        pE = (1/ψE)*pE_func(124,pfψf,p); pE_nt = (1/ψE)*pE_func(124,pfψf_nt,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 1234
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean; pc_nt = exp(Data.lnpc_tilde[i]);
                        pcψc = pc; pcψc_nt = pc_nt;
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean; pg_nt = exp(Data.lnpg_tilde[i]);
                        pgψg = pg; pgψg_nt = pg_nt;
                        pfψf = [poψo,peψe,pcψc,pgψg]; pfψf_nt = [poψo_nt,peψe_nt,pcψc_nt,pgψg_nt];
                        pE = (1/ψE)*pE_func(1234,pfψf,p); pE_nt = (1/ψE)*pE_func(1234,pfψf_nt,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    end
                    # Output
                    y[i,s,1] = output_func_monopolistic(z,pin,p.pout_struc[t+1],t,p);
                    # profit
                    profit[i,s,1] = profit_func_monopolistic(z,pin,t,p);
                    # Energy
                    E[i,s,1] = ((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE)^p.σ);
                    # Fuel quantitiy (Enegy level changes, but no substitution)
                    oil[i,s,1] = (p.ogmean*E[i,s,1])*((p.βo/po_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                    elec[i,s,1] = (p.egmean*E[i,s,1])*((p.βe/pe_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                    if Data.combineF[i] == 12
                        gas[i,s,1] = 0;
                        coal[i,s,1] = 0;
                    elseif Data.combineF[i] == 123
                        gas[i,s,1] = 0;
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                    elseif Data.combineF[i] == 124
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                        coal[i,s,1] = 0;
                    elseif Data.combineF[i] == 1234
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                    end
                end
            end
            # Years of forward simulation (starting one year ahead)
            @sync @distributed for j = 1:(Tf*S)
                tf,s = Tuple(CartesianIndices((Tf,S))[j]);
                # Get new aggregate price index
                Data_fs = deepcopy(Data);
                for i = 1:N
                    t = Data_fs.year[i]-2009;
                    Data_fs.lnz[i] = log(z_fs[i,s,tf]);
                    Data_fs.logPm[i] = log(pm_fs[i,s,tf]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1]; 
                    Data_fs.lnpelec_tilde[i] = log(pe_fs[i,s,tf]);
                    # Energy productivity
                    ψE = ψE_fs[i,s,tf]; Data_fs.lnfprod_E[i] = log(ψE);
                    # input prices indices
                    if Data.combineF[i] == 123
                        Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                    elseif Data.combineF[i] == 124
                        Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                    elseif Data.combineF[i] == 1234
                        Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                        Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                    end
                end
                p_fs,M = OutputPrice_func_Eprod(M,Data_fs,p,τ);
                pout[:,s,tf+1] = p_fs.pout_struc[2:7];
                # Get everything else
                for i = 1:N
                    t = Data_fs.year[i]-2009; 
                    pm = exp(Data_fs.logPm[i]);
                    z = exp(Data_fs.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean; po_nt = p.po[t+1];
                    pe = exp(Data_fs.lnpelec_tilde[i]) + τe*p.egmean; pe_nt = exp(Data_fs.lnpelec_tilde[i]);
                    pc = exp(Data_fs.lnpc_tilde[i]) + τc*p.cgmean; pc_nt = exp(Data_fs.lnpc_tilde[i]);
                    pg = exp(Data_fs.lnpg_tilde[i]) + τg*p.ggmean; pg_nt = exp(Data_fs.lnpg_tilde[i]);
                    # Energy productivity
                    ψE = ψE_fs[i,s,tf];
                    # input prices indices
                    poψo = po;
                    peψe = pe;
                    if Data.combineF[i] == 12
                        pfψf = [poψo,peψe]; pfψf_nt = [po_nt,pe_nt];
                        pE = (1/ψE)*pE_func(12,pfψf,p); pE_nt = (1/ψE)*pE_func(12,pfψf_nt,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 123
                        pcψc = pc; 
                        pfψf = [poψo,peψe,pcψc]; pfψf_nt = [po_nt,pe_nt,pc_nt];
                        pE = (1/ψE)*pE_func(123,pfψf,p); pE_nt = (1/ψE)*pE_func(123,pfψf_nt,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 124
                        pgψg = pg; 
                        pfψf = [poψo,peψe,pgψg]; pfψf_nt = [po_nt,pe_nt,pg_nt];
                        pE = (1/ψE)*pE_func(124,pfψf,p); pE_nt = (1/ψE)*pE_func(123,pfψf_nt,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    elseif Data.combineF[i] == 1234
                        pcψc = pc;
                        pgψg = pg;
                        pfψf = [poψo,peψe,pcψc,pgψg]; pfψf_nt = [po_nt,pe_nt,pc_nt,pg_nt];
                        pE = (1/ψE)*pE_func(1234,pfψf,p); pE_nt = (1/ψE)*pE_func(123,pfψf_nt,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p);
                    end
                    # Output
                    y[i,s,tf+1] = output_func_monopolistic(z,pin,p_fs.pout_struc[t+1],t,p_fs);
                    # profit
                    profit[i,s,tf+1] = profit_func_monopolistic(z,pin,t,p_fs);
                    # Energy
                    E[i,s,tf+1] = ((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin/pE)^p.σ);
                    # Fuel quantitiy
                    oil[i,s,tf+1] = (p.ogmean*E[i,s,tf+1])*((p.βo/po_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                    elec[i,s,tf+1] = (p.egmean*E[i,s,tf+1])*((p.βe/pe_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                    if Data.combineF[i] == 12
                        gas[i,s,tf+1] = 0;
                        coal[i,s,tf+1] = 0;
                    elseif Data.combineF[i] == 123
                        gas[i,s,tf+1] = 0;
                        coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc_nt)^p.λ)*(ψE_fs[i,s,tf]^(p.λ-1))*(pE_nt^p.λ);
                    elseif Data.combineF[i] == 124
                        gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg_nt)^p.λ)*(ψE_fs[i,s,tf]^(p.λ-1))*(pE_nt^p.λ);
                        coal[i,s,tf+1] = 0;
                    elseif Data.combineF[i] == 1234
                        gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg_nt)^p.λ)*(ψE_fs[i,s,tf]^(p.λ-1))*(pE_nt^p.λ);
                        coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc_nt)^p.λ)*(ψE_fs[i,s,tf]^(p.λ-1))*(pE_nt^p.λ);
                    end
                end
            end
            # Return aggregates by year forward (only look at last year in data - 2016)
            y_fs = zeros(Tf+1); E_fs = zeros(Tf+1); gas_fs = zeros(Tf+1); coal_fs = zeros(Tf+1); oil_fs = zeros(Tf+1); elec_fs = zeros(Tf+1); co2_fs = zeros(Tf+1); profit_fs = zeros(Tf+1); CS_fs = zeros(Tf+1);
            Data_year = groupby(Data,:year); Nt = size(Data_year[end])[1];
            # pE = pE[end-Nt+1:end,:,:];
            y = y[end-Nt+1:end,:,:];
            E = E[end-Nt+1:end,:,:];
            gas = gas[end-Nt+1:end,:,:];
            coal = coal[end-Nt+1:end,:,:];
            oil = oil[end-Nt+1:end,:,:];
            elec = elec[end-Nt+1:end,:,:];
            profit = profit[end-Nt+1:end,:,:];
            pout = pout[end,:,:];
            co2 = p.γg*gas .+ p.γc*coal .+ p.γo*oil .+ p.γe*elec;
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
            CS_s_fs = zeros(p.S,p.Tf+1);
            thetaterm = p.θ/(1-p.θ)
            for tf = 1:(p.Tf+1)
                for s = 1:p.S
                    CS_s_fs[s,tf] = thetaterm*(pout[s,tf]^-thetaterm);
                end
                CS_fs[tf] = mean(CS_s_fs[:,tf]);
            end
            # Average across simulations
            for tf = 1:Tf+1
                # pE_fs[tf] = mean(pE[:,:,tf]);
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
                end
            end
            return y_fs,E_fs,gas_fs,coal_fs,oil_fs,elec_fs,co2_fs,profit_fs,CS_fs;
            #return pE,pin,y,E,gas,gasspend,coal,coalspend,oil,oilspend,elec,elecspend,pout,Espend,Mspend,Lspend,Krent,Kval,profit;
            # return pE,pin,y,E,gas,coal,oil,elec,Espend,profit;
        end
    #
    # 7. Energy productivity with no input substitution
        @everywhere function Welfare_ForwardSimul_noswitch_Eprod_noinputsub(M::Model_Eprod,Data::DataFrame,p::Par_Eprod,τ,d_elast::Float64=p.ρ,rscale::Float64=p.η)
            @unpack N,z_fs,pm_fs,pe_fs,ψE_fs,pg_fs,pc_fs = M;
            @unpack S,Tf,T,γg,γc,γo,γe,β=p;
            τg = τ[1];
            τc = τ[2];
            τo = τ[3];
            τe = τ[4];
            p = Par_Eprod(p; ρ=d_elast, η=rscale);
            M = Model_Eprod(M; p = p);
            println("Demand elasticity = $(p.ρ)")
            # Update aggregate output price
            p,M = OutputPrice_func_Eprod(M,Data,p,τ);
            # Start static of social welfare
            pout = SharedArray{Float64}(T,S,Tf+1);
            gas = SharedArray{Float64}(N,S,Tf+1);
            coal = SharedArray{Float64}(N,S,Tf+1);
            oil = SharedArray{Float64}(N,S,Tf+1);
            elec = SharedArray{Float64}(N,S,Tf+1);
            y = SharedArray{Float64}(N,S,Tf+1);
            E = SharedArray{Float64}(N,S,Tf+1);
                δ = 0.0;
                r = 0.150838;
            profit = SharedArray{Float64}(N,S,Tf+1);
            println("      ")
            println("--------------------------------------------------------------")
            println("--------      BEGINNING FORWARD SIMULATION     ---------------")
            println("---------      Energy productivity model       ----------")
            println("---------      No Switching - No fuel prod - Tax (g,c,o,e) = $τ    ----------")
            println("--------------------------------------------------------------")
            println("      ")
            # Year of data: Get firm-level objects: output, fuel prices and productivity, fuel quantities
            for s = 1:S
                pout[:,s,1] = p.pout_struc[2:7];
            end
            for i = 1:N
                for s = 1:S
                    t = Data.year[i]-2009;
                    pm = exp(Data.logPm[i]);
                    z = exp(Data.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean; po_nt = p.po[t+1];
                    pe = exp(Data.lnpelec_tilde[i]) + τe*p.egmean; pe_nt = exp(Data.lnpelec_tilde[i]);
                    # energy productivity
                    ψE = exp(Data.lnfprod_E[i]);
                    # input prices indices
                    poψo = po; poψo_nt = po_nt;
                    peψe = pe; peψe_nt = pe_nt;
                    if Data.combineF[i] == 12
                        pfψf = [poψo,peψe]; pfψf_nt = [poψo_nt,peψe_nt];
                        pE = (1/ψE)*pE_func(12,pfψf,p); pE_nt = (1/ψE)*pE_func(12,pfψf_nt,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p); pin_nt = pinput_func(p.rk[t+1],pm,p.w[t+1],pE_nt,p);
                    elseif Data.combineF[i] == 123
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean; pc_nt = exp(Data.lnpc_tilde[i]);
                        pcψc = pc; pcψc_nt = pc_nt;
                        pfψf = [poψo,peψe,pcψc]; pfψf_nt = [poψo_nt,peψe_nt,pcψc_nt];
                        pE = (1/ψE)*pE_func(123,pfψf,p); pE_nt = (1/ψE)*pE_func(123,pfψf_nt,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p); pin_nt = pinput_func(p.rk[t+1],pm,p.w[t+1],pE_nt,p);
                    elseif Data.combineF[i] == 124
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean; pg_nt = exp(Data.lnpg_tilde[i]);
                        pgψg = pg; pgψg_nt = pg_nt;
                        pfψf = [poψo,peψe,pgψg]; pfψf_nt = [poψo_nt,peψe_nt,pgψg_nt];
                        pE = (1/ψE)*pE_func(124,pfψf,p); pE_nt = (1/ψE)*pE_func(124,pfψf_nt,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p); pin_nt = pinput_func(p.rk[t+1],pm,p.w[t+1],pE_nt,p);
                    elseif Data.combineF[i] == 1234
                        pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean; pc_nt = exp(Data.lnpc_tilde[i]);
                        pcψc = pc; pcψc_nt = pc_nt;
                        pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean; pg_nt = exp(Data.lnpg_tilde[i]);
                        pgψg = pg; pgψg_nt = pg_nt;
                        pfψf = [poψo,peψe,pcψc,pgψg]; pfψf_nt = [poψo_nt,peψe_nt,pcψc_nt,pgψg_nt];
                        pE = (1/ψE)*pE_func(1234,pfψf,p); pE_nt = (1/ψE)*pE_func(1234,pfψf_nt,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p); pin_nt = pinput_func(p.rk[t+1],pm,p.w[t+1],pE_nt,p);
                    end
                    # Output
                    y[i,s,1] = output_func_monopolistic(z,pin,p.pout_struc[t+1],t,p);
                    # profit
                    profit[i,s,1] = profit_func_monopolistic(z,pin,t,p);
                    # Energy
                    E[i,s,1] = ((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin_nt/pE_nt)^p.σ);
                    # Fuel quantitiy (Enegy level changes, but no substitution)
                    oil[i,s,1] = (p.ogmean*E[i,s,1])*((p.βo/po_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                    elec[i,s,1] = (p.egmean*E[i,s,1])*((p.βe/pe_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                    if Data.combineF[i] == 12
                        gas[i,s,1] = 0;
                        coal[i,s,1] = 0;
                    elseif Data.combineF[i] == 123
                        gas[i,s,1] = 0;
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                    elseif Data.combineF[i] == 124
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                        coal[i,s,1] = 0;
                    elseif Data.combineF[i] == 1234
                        gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                        coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                    end
                end
            end
            # Years of forward simulation (starting one year ahead)
            @sync @distributed for j = 1:(Tf*S)
                tf,s = Tuple(CartesianIndices((Tf,S))[j]);
                # Get new aggregate price index
                Data_fs = deepcopy(Data);
                for i = 1:N
                    t = Data_fs.year[i]-2009;
                    Data_fs.lnz[i] = log(z_fs[i,s,tf]);
                    Data_fs.logPm[i] = log(pm_fs[i,s,tf]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1]; 
                    Data_fs.lnpelec_tilde[i] = log(pe_fs[i,s,tf]);
                    # Energy productivity
                    ψE = ψE_fs[i,s,tf]; Data_fs.lnfprod_E[i] = log(ψE);
                    # input prices indices
                    if Data.combineF[i] == 123
                        Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                    elseif Data.combineF[i] == 124
                        Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                    elseif Data.combineF[i] == 1234
                        Data_fs.lnpc_tilde[i] = log(pc_fs[i,s,tf]);
                        Data_fs.lnpg_tilde[i] = log(pg_fs[i,s,tf]);
                    end
                end
                p_fs,M = OutputPrice_func_Eprod(M,Data_fs,p,τ);
                pout[:,s,tf+1] = p_fs.pout_struc[2:7];
                # Get everything else
                for i = 1:N
                    t = Data_fs.year[i]-2009; 
                    pm = exp(Data_fs.logPm[i]);
                    z = exp(Data_fs.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean; po_nt = p.po[t+1];
                    pe = exp(Data_fs.lnpelec_tilde[i]) + τe*p.egmean; pe_nt = exp(Data_fs.lnpelec_tilde[i]);
                    pc = exp(Data_fs.lnpc_tilde[i]) + τc*p.cgmean; pc_nt = exp(Data_fs.lnpc_tilde[i]);
                    pg = exp(Data_fs.lnpg_tilde[i]) + τg*p.ggmean; pg_nt = exp(Data_fs.lnpg_tilde[i]);
                    # Energy productivity
                    ψE = ψE_fs[i,s,tf];
                    # input prices indices
                    poψo = po;
                    peψe = pe;
                    if Data.combineF[i] == 12
                        pfψf = [poψo,peψe]; pfψf_nt = [po_nt,pe_nt];
                        pE = (1/ψE)*pE_func(12,pfψf,p); pE_nt = (1/ψE)*pE_func(12,pfψf_nt,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p); pin_nt = pinput_func(p.rk[t+1],pm,p.w[t+1],pE_nt,p);
                    elseif Data.combineF[i] == 123
                        pcψc = pc; 
                        pfψf = [poψo,peψe,pcψc]; pfψf_nt = [po_nt,pe_nt,pc_nt];
                        pE = (1/ψE)*pE_func(123,pfψf,p); pE_nt = (1/ψE)*pE_func(123,pfψf_nt,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p); pin_nt = pinput_func(p.rk[t+1],pm,p.w[t+1],pE_nt,p);
                    elseif Data.combineF[i] == 124
                        pgψg = pg; 
                        pfψf = [poψo,peψe,pgψg]; pfψf_nt = [po_nt,pe_nt,pg_nt];
                        pE = (1/ψE)*pE_func(124,pfψf,p); pE_nt = (1/ψE)*pE_func(123,pfψf_nt,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p); pin_nt = pinput_func(p.rk[t+1],pm,p.w[t+1],pE_nt,p);
                    elseif Data.combineF[i] == 1234
                        pcψc = pc;
                        pgψg = pg;
                        pfψf = [poψo,peψe,pcψc,pgψg]; pfψf_nt = [po_nt,pe_nt,pc_nt,pg_nt];
                        pE = (1/ψE)*pE_func(1234,pfψf,p); pE_nt = (1/ψE)*pE_func(123,pfψf_nt,p);
                        pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p); pin_nt = pinput_func(p.rk[t+1],pm,p.w[t+1],pE_nt,p);
                    end
                    # Output
                    y[i,s,tf+1] = output_func_monopolistic(z,pin,p_fs.pout_struc[t+1],t,p_fs);
                    # profit
                    profit[i,s,tf+1] = profit_func_monopolistic(z,pin,t,p_fs);
                    # Energy
                    E[i,s,tf+1] = ((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin_nt/pE_nt)^p.σ);
                    # Fuel quantitiy
                    oil[i,s,tf+1] = (p.ogmean*E[i,s,tf+1])*((p.βo/po_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                    elec[i,s,tf+1] = (p.egmean*E[i,s,tf+1])*((p.βe/pe_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                    if Data.combineF[i] == 12
                        gas[i,s,tf+1] = 0;
                        coal[i,s,tf+1] = 0;
                    elseif Data.combineF[i] == 123
                        gas[i,s,tf+1] = 0;
                        coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc_nt)^p.λ)*(ψE_fs[i,s,tf]^(p.λ-1))*(pE_nt^p.λ);
                    elseif Data.combineF[i] == 124
                        gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg_nt)^p.λ)*(ψE_fs[i,s,tf]^(p.λ-1))*(pE_nt^p.λ);
                        coal[i,s,tf+1] = 0;
                    elseif Data.combineF[i] == 1234
                        gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg_nt)^p.λ)*(ψE_fs[i,s,tf]^(p.λ-1))*(pE_nt^p.λ);
                        coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc_nt)^p.λ)*(ψE_fs[i,s,tf]^(p.λ-1))*(pE_nt^p.λ);
                    end
                end
            end
            # Return aggregates by year forward (only look at last year in data - 2016)
            y_fs = zeros(Tf+1); E_fs = zeros(Tf+1); gas_fs = zeros(Tf+1); coal_fs = zeros(Tf+1); oil_fs = zeros(Tf+1); elec_fs = zeros(Tf+1); co2_fs = zeros(Tf+1); profit_fs = zeros(Tf+1); CS_fs = zeros(Tf+1);
            Data_year = groupby(Data,:year); Nt = size(Data_year[end])[1];
            # pE = pE[end-Nt+1:end,:,:];
            y = y[end-Nt+1:end,:,:];
            E = E[end-Nt+1:end,:,:];
            gas = gas[end-Nt+1:end,:,:];
            coal = coal[end-Nt+1:end,:,:];
            oil = oil[end-Nt+1:end,:,:];
            elec = elec[end-Nt+1:end,:,:];
            profit = profit[end-Nt+1:end,:,:];
            pout = pout[end,:,:];
            co2 = p.γg*gas .+ p.γc*coal .+ p.γo*oil .+ p.γe*elec;
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
            CS_s_fs = zeros(p.S,p.Tf+1);
            thetaterm = p.θ/(1-p.θ)
            for tf = 1:(p.Tf+1)
                for s = 1:p.S
                    CS_s_fs[s,tf] = thetaterm*(pout[s,tf]^-thetaterm);
                end
                CS_fs[tf] = mean(CS_s_fs[:,tf]);
            end
            # Average across simulations
            for tf = 1:Tf+1
                # pE_fs[tf] = mean(pE[:,:,tf]);
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
                end
            end
            return y_fs,E_fs,gas_fs,coal_fs,oil_fs,elec_fs,co2_fs,profit_fs,CS_fs;
            #return pE,pin,y,E,gas,gasspend,coal,coalspend,oil,oilspend,elec,elecspend,pout,Espend,Mspend,Lspend,Krent,Kval,profit;
            # return pE,pin,y,E,gas,coal,oil,elec,Espend,profit;
        end
    #
    # 8. Energy productivity with no input substitution, same fuel set (all 4 fuels) and same fuel prices
        @everywhere function Welfare_ForwardSimul_noswitch_Eprod_noinputsub_sameF(M::Model_Eprod,Data::DataFrame,p::Par_Eprod,τ,d_elast::Float64=p.ρ,rscale::Float64=p.η)
            @unpack N,z_fs,pm_fs,pe_fs,ψE_fs,pg_fs,pc_fs = M;
            @unpack S,Tf,T,γg,γc,γo,γe,β=p;
            τg = τ[1];
            τc = τ[2];
            τo = τ[3];
            τe = τ[4];
            p = Par_Eprod(p; ρ=d_elast, η=rscale);
            M = Model_Eprod(M; p = p);
            println("Demand elasticity = $(p.ρ)")
            # Update aggregate output price
            Data.logPm .= mean(Data.logPm);
            Data.lnpelec_tilde .= mean(Data.lnpelec_tilde);
            Data.lnfprod_E .= mean(Data.lnfprod_E);
            Data.lnpc_tilde .= mean(skipmissing(Data.lnpc_tilde));
            Data.lnpg_tilde .= mean(skipmissing(Data.lnpg_tilde));
            Data.combineF .= 1234;
            p,M = OutputPrice_func_Eprod(M,Data,p,τ);
            # Start static of social welfare
            pout = SharedArray{Float64}(T,S,Tf+1);
            gas = SharedArray{Float64}(N,S,Tf+1);
            coal = SharedArray{Float64}(N,S,Tf+1);
            oil = SharedArray{Float64}(N,S,Tf+1);
            elec = SharedArray{Float64}(N,S,Tf+1);
            y = SharedArray{Float64}(N,S,Tf+1);
            E = SharedArray{Float64}(N,S,Tf+1);
                δ = 0.0;
                r = 0.150838;
            profit = SharedArray{Float64}(N,S,Tf+1);
            println("      ")
            println("--------------------------------------------------------------")
            println("--------      BEGINNING FORWARD SIMULATION     ---------------")
            println("---------      Energy productivity model       ----------")
            println("---------      No Switching - No fuel prod - Tax (g,c,o,e) = $τ    ----------")
            println("--------------------------------------------------------------")
            println("      ")
            # Year of data: Get firm-level objects: output, fuel prices and productivity, fuel quantities
            for s = 1:S
                pout[:,s,1] = p.pout_struc[2:7];
            end
            for i = 1:N
                for s = 1:S
                    t = Data.year[i]-2009;
                    pm = Data.logPm[i];
                    z = exp(Data.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean; po_nt = p.po[t+1];
                    pe = exp(Data.lnpelec_tilde[i]) + τe*p.egmean; pe_nt = exp(Data.lnpelec_tilde[i]);
                    # energy productivity
                    ψE = exp(Data.lnfprod_E[i]);
                    # input prices indices
                    poψo = po; poψo_nt = po_nt;
                    peψe = pe; peψe_nt = pe_nt;
                    pc = exp(Data.lnpc_tilde[i]) + τc*p.cgmean; pc_nt = exp(Data.lnpc_tilde[i]);
                    pcψc = pc; pcψc_nt = pc_nt;
                    pg = exp(Data.lnpg_tilde[i]) + τg*p.ggmean; pg_nt = exp(Data.lnpg_tilde[i]);
                    pgψg = pg; pgψg_nt = pg_nt;
                    pfψf = [poψo,peψe,pcψc,pgψg]; pfψf_nt = [poψo_nt,peψe_nt,pcψc_nt,pgψg_nt];
                    pE = (1/ψE)*pE_func(1234,pfψf,p); pE_nt = (1/ψE)*pE_func(1234,pfψf_nt,p);
                    pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p); pin_nt = pinput_func(p.rk[t+1],pm,p.w[t+1],pE_nt,p);
                    # Output
                    y[i,s,1] = output_func_monopolistic(z,pin,p.pout_struc[t+1],t,p);
                    # profit
                    profit[i,s,1] = profit_func_monopolistic(z,pin,t,p);
                    # Energy
                    E[i,s,1] = ((y[i,s,1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin_nt/pE_nt)^p.σ);
                    # Fuel quantitiy (Enegy level changes, but no substitution)
                    oil[i,s,1] = (p.ogmean*E[i,s,1])*((p.βo/po_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                    elec[i,s,1] = (p.egmean*E[i,s,1])*((p.βe/pe_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                    gas[i,s,1] = (p.ggmean*E[i,s,1])*((p.βg/pg_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                    coal[i,s,1] = (p.cgmean*E[i,s,1])*((p.βc/pc_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                end
            end
            # Years of forward simulation (starting one year ahead)
            @sync @distributed for j = 1:(Tf*S)
                tf,s = Tuple(CartesianIndices((Tf,S))[j]);
                # Get new aggregate price index
                Data_fs = deepcopy(Data);
                Data_fs.logPm .= mean(log.(pm_fs[:,s,tf]));
                Data_fs.lnpelec_tilde .= mean(log.(pe_fs[:,s,tf]));
                Data_fs.lnpc_tilde .= mean(log.(pc_fs[:,s,tf]));
                Data_fs.lnpg_tilde .= mean(log.(pg_fs[:,s,tf]));
                Data_fs.lnfprod_E .= mean(log.(ψE_fs[:,s,tf]));
                for i = 1:N
                    Data_fs.lnz[i] = log(z_fs[i,s,tf]);
                end
                Data_fs.combineF .= 1234;
                p_fs,M = OutputPrice_func_Eprod(M,Data_fs,p,τ);
                pout[:,s,tf+1] = p_fs.pout_struc[2:7];
                # Get everything else
                for i = 1:N
                    t = Data_fs.year[i]-2009; 
                    pm = exp(Data_fs.logPm[i]);
                    z = exp(Data_fs.lnz[i]);
                    # fuel prices (multiplied by geometric mean of fuel)
                    po = p.po[t+1] + τo*p.ogmean; po_nt = p.po[t+1];
                    pe = exp(Data_fs.lnpelec_tilde[i]) + τe*p.egmean; pe_nt = exp(Data_fs.lnpelec_tilde[i]);
                    pc = exp(Data_fs.lnpc_tilde[i]) + τc*p.cgmean; pc_nt = exp(Data_fs.lnpc_tilde[i]);
                    pg = exp(Data_fs.lnpg_tilde[i]) + τg*p.ggmean; pg_nt = exp(Data_fs.lnpg_tilde[i]);
                    # Energy productivity
                    ψE = exp(mean(ψE_fs[i,s,tf]));
                    # input prices indices
                    poψo = po;
                    peψe = pe;
                    pcψc = pc;
                    pgψg = pg;
                    pfψf = [poψo,peψe,pcψc,pgψg]; pfψf_nt = [po_nt,pe_nt,pc_nt,pg_nt];
                    pE = (1/ψE)*pE_func(1234,pfψf,p); pE_nt = (1/ψE)*pE_func(123,pfψf_nt,p);
                    pin = pinput_func(p.rk[t+1],pm,p.w[t+1],pE,p); pin_nt = pinput_func(p.rk[t+1],pm,p.w[t+1],pE_nt,p);
                    # Output
                    y[i,s,tf+1] = output_func_monopolistic(z,pin,p_fs.pout_struc[t+1],t,p_fs);
                    # profit
                    profit[i,s,tf+1] = profit_func_monopolistic(z,pin,t,p_fs);
                    # Energy
                    E[i,s,tf+1] = ((y[i,s,tf+1]/(p.Ygmean[t+1]*z))^(1/p.η))*((p.αe*pin_nt/pE_nt)^p.σ);
                    # Fuel quantitiy
                    oil[i,s,tf+1] = (p.ogmean*E[i,s,tf+1])*((p.βo/po_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                    elec[i,s,tf+1] = (p.egmean*E[i,s,tf+1])*((p.βe/pe_nt)^p.λ)*(ψE^(p.λ-1))*(pE_nt^p.λ);
                    gas[i,s,tf+1] = (p.ggmean*E[i,s,tf+1])*((p.βg/pg_nt)^p.λ)*(ψE_fs[i,s,tf]^(p.λ-1))*(pE_nt^p.λ);
                    coal[i,s,tf+1] = (p.cgmean*E[i,s,tf+1])*((p.βc/pc_nt)^p.λ)*(ψE_fs[i,s,tf]^(p.λ-1))*(pE_nt^p.λ);
                end
            end
            # Return aggregates by year forward (only look at last year in data - 2016)
            y_fs = zeros(Tf+1); E_fs = zeros(Tf+1); gas_fs = zeros(Tf+1); coal_fs = zeros(Tf+1); oil_fs = zeros(Tf+1); elec_fs = zeros(Tf+1); co2_fs = zeros(Tf+1); profit_fs = zeros(Tf+1); CS_fs = zeros(Tf+1);
            Data_year = groupby(Data,:year); Nt = size(Data_year[end])[1];
            # pE = pE[end-Nt+1:end,:,:];
            y = y[end-Nt+1:end,:,:];
            E = E[end-Nt+1:end,:,:];
            gas = gas[end-Nt+1:end,:,:];
            coal = coal[end-Nt+1:end,:,:];
            oil = oil[end-Nt+1:end,:,:];
            elec = elec[end-Nt+1:end,:,:];
            profit = profit[end-Nt+1:end,:,:];
            pout = pout[end,:,:];
            co2 = p.γg*gas .+ p.γc*coal .+ p.γo*oil .+ p.γe*elec;
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
            CS_s_fs = zeros(p.S,p.Tf+1);
            thetaterm = p.θ/(1-p.θ)
            for tf = 1:(p.Tf+1)
                for s = 1:p.S
                    CS_s_fs[s,tf] = thetaterm*(pout[s,tf]^-thetaterm);
                end
                CS_fs[tf] = mean(CS_s_fs[:,tf]);
            end
            # Average across simulations
            for tf = 1:Tf+1
                # pE_fs[tf] = mean(pE[:,:,tf]);
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
                end
            end
            return y_fs,E_fs,gas_fs,coal_fs,oil_fs,elec_fs,co2_fs,profit_fs,CS_fs;
            #return pE,pin,y,E,gas,gasspend,coal,coalspend,oil,oilspend,elec,elecspend,pout,Espend,Mspend,Lspend,Krent,Kval,profit;
            # return pE,pin,y,E,gas,coal,oil,elec,Espend,profit;
        end
    #
#

### Set taxes and initialize arrays
    τ = zeros(4,21);
    τ_ct = [p.γg*p.SCC_india/p.unit,p.γc*p.SCC_india/p.unit,p.γo*p.SCC_india/p.unit,p.γe*p.SCC_india/p.unit];
    tlevel = [0,0.001,0.01,0.05,0.1,0.25,0.5,0.75,1,2,2.5,5,10,25,50,100,250,500,1000,10000,100000000000]
    for tr = 1:21
        τ[:,tr] = tlevel[tr]*τ_ct;
    end
#

### Perform simulations that compare models across tax rates
    @everywhere function WelfareCompare_Compile(model::String,d_elast::Float64=p.ρ,rscale::Float64=p.η,perfcompl::Bool=false)
        y = zeros(p.Tf+1,21);  
        E = zeros(p.Tf+1,21); 
        gas = zeros(p.Tf+1,21);
        coal = zeros(p.Tf+1,21); 
        oil = zeros(p.Tf+1,21);   
        elec = zeros(p.Tf+1,21); 
        co2 = zeros(p.Tf+1,21); 
        profit = zeros(p.Tf+1,21); 
        CS = zeros(p.Tf+1,21); 
        if model == "main"
            for tr = 1:21
                y[:,tr],E[:,tr],gas[:,tr],coal[:,tr],oil[:,tr],elec[:,tr],co2[:,tr],profit[:,tr],CS[:,tr] = Welfare_ForwardSimul_noswitch(M,Data,p,τ[:,tr],d_elast,rscale,perfcompl);
            end
        elseif model == "Eprod"
            for tr = 1:21
                y[:,tr],E[:,tr],gas[:,tr],coal[:,tr],oil[:,tr],elec[:,tr],co2[:,tr],profit[:,tr],CS[:,tr] = Welfare_ForwardSimul_noswitch_Eprod(M_Eprod,Data_Eprod,p_Eprod,τ[:,tr],d_elast,rscale,perfcompl);
            end
        elseif model == "Eprod_fixsub"
            for tr = 1:21
                y[:,tr],E[:,tr],gas[:,tr],coal[:,tr],oil[:,tr],elec[:,tr],co2[:,tr],profit[:,tr],CS[:,tr] = Welfare_ForwardSimul_noswitch_Eprod_fixsub(M_Eprod,Data_Eprod,p_Eprod,τ[:,tr],d_elast,rscale);
            end
        elseif model == "nfp"
            for tr = 1:21
                y[:,tr],E[:,tr],gas[:,tr],coal[:,tr],oil[:,tr],elec[:,tr],co2[:,tr],profit[:,tr],CS[:,tr] = Welfare_ForwardSimul_noswitch_nfp(M_nfp,Data_nfp,p_nfp,τ[:,tr]);
            end
        elseif model == "simpleCES"
            for tr = 1:21
                y[:,tr],E[:,tr],gas[:,tr],coal[:,tr],oil[:,tr],elec[:,tr],co2[:,tr],profit[:,tr],CS[:,tr] = Welfare_ForwardSimul_noswitch_simpleCES(M_simpleCES,Data_simpleCES,p_simpleCES,τ[:,tr]);
            end
        elseif model == "simpleCES_fixsub"
            for tr = 1:21
                y[:,tr],E[:,tr],gas[:,tr],coal[:,tr],oil[:,tr],elec[:,tr],co2[:,tr],profit[:,tr],CS[:,tr] = Welfare_ForwardSimul_noswitch_simpleCES_fixsub(M_simpleCES,Data_simpleCES,p_simpleCES,τ[:,tr]);
            end
        elseif model == "Eprod_nofuelsub"
            for tr = 1:21
                y[:,tr],E[:,tr],gas[:,tr],coal[:,tr],oil[:,tr],elec[:,tr],co2[:,tr],profit[:,tr],CS[:,tr] = Welfare_ForwardSimul_noswitch_Eprod_nofuelsub(M_Eprod,Data_Eprod,p_Eprod,τ[:,tr],d_elast,rscale);
            end
        elseif model == "Eprod_noinputsub"
            for tr = 1:21
                y[:,tr],E[:,tr],gas[:,tr],coal[:,tr],oil[:,tr],elec[:,tr],co2[:,tr],profit[:,tr],CS[:,tr] = Welfare_ForwardSimul_noswitch_Eprod_noinputsub(M_Eprod,Data_Eprod,p_Eprod,τ[:,tr],d_elast,rscale);
            end
        elseif model == "Eprod_noinputsub_sameF"
            for tr = 1:21
                y[:,tr],E[:,tr],gas[:,tr],coal[:,tr],oil[:,tr],elec[:,tr],co2[:,tr],profit[:,tr],CS[:,tr] = Welfare_ForwardSimul_noswitch_Eprod_noinputsub_sameF(M_Eprod,Data_Eprod,p_Eprod,τ[:,tr],d_elast,rscale);
            end
        else
            error("model not found")
        end
        return Dict("y"=>y,"E"=>E,"gas"=>gas,"coal"=>coal,"oil"=>oil,"elec"=>elec,"co2"=>co2,"profit"=>profit,"CS"=>CS);
    end

    @everywhere function WelfareCompare_Compile_nofs(model::String,d_elast::Float64=p.ρ,rscale::Float64=p.η,perfcompl::Bool=false)
        y = zeros(21);  
        E = zeros(21); 
        gas = zeros(21);
        coal = zeros(21); 
        oil = zeros(21);   
        elec = zeros(21); 
        co2 = zeros(21); 
        profit = zeros(21); 
        CS = zeros(21); 
        if model == "main"
            for tr = 1:21
                y[tr],E[tr],gas[tr],coal[tr],oil[tr],elec[tr],co2[tr],profit[tr],CS[tr] = Welfare_noswitch(M,Data,p,τ[:,tr],d_elast,rscale,perfcompl);
            end
        elseif model == "Eprod"
            for tr = 1:21
                y[tr],E[tr],gas[tr],coal[tr],oil[tr],elec[tr],co2[tr],profit[tr],CS[tr] = Welfare_noswitch_Eprod(M_Eprod,Data_Eprod,p_Eprod,τ[:,tr],d_elast,rscale,perfcompl);
            end
        else
            error("model not found")
        end
        return Dict("y"=>y,"E"=>E,"gas"=>gas,"coal"=>coal,"oil"=>oil,"elec"=>elec,"co2"=>co2,"profit"=>profit,"CS"=>CS);
    end

    ### COMPARE MODELS WITH FORWARD SIMULATION ###
    # Main model
    SimulCompareModels_ns = WelfareCompare_Compile("main");
    save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_ns.jld2", "SimulCompareModels_ns", SimulCompareModels_ns);
    # Energy productivity 
    SimulCompareModels_Eprod = WelfareCompare_Compile("Eprod");
    save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_Eprod.jld2", "SimulCompareModels_Eprod", SimulCompareModels_Eprod);
    # Energy productivity with fixed fuel substitution
    SimulCompareModels_Eprod_fixsub = WelfareCompare_Compile("Eprod_fixsub");
    save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_Eprod_fixsub.jld2", "SimulCompareModels_Eprod_fixsub", SimulCompareModels_Eprod_fixsub);
    # Energy productivity without fuel substitution
    SimulCompareModels_Eprod_nofuelsub = WelfareCompare_Compile("Eprod_nofuelsub");
    save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_Eprod_nofuelsub.jld2", "SimulCompareModels_Eprod_nofuelsub", SimulCompareModels_Eprod_nofuelsub);
    # Energy productivity without input substitution
    SimulCompareModels_Eprod_noinputsub = WelfareCompare_Compile("Eprod_noinputsub");
    save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_Eprod_noinputsub.jld2", "SimulCompareModels_Eprod_noinputsub", SimulCompareModels_Eprod_noinputsub);
    # Energy productivity without input substitution and same fuel sets
    SimulCompareModels_Eprod_noinputsub_sameF = WelfareCompare_Compile("Eprod_noinputsub_sameF");
    save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_Eprod_noinputsub_sameF.jld2", "SimulCompareModels_Eprod_noinputsub_sameF", SimulCompareModels_Eprod_noinputsub_sameF);

    ### VARYING ELASTICITY OF DEMAND (RETURNS TO SCALE SET TO 1) - STATIC SIMULATION ###
        # Perfect complements
        perf_compl = true
        SimulCompareModels_ns_nfs_perfcompl = WelfareCompare_Compile_nofs("main",p.ρ,p.η,perf_compl);
        SimulCompareModels_Eprod_nfs_perfcompl = WelfareCompare_Compile_nofs("Eprod",p.ρ,p.η,perf_compl);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_ns_nfs_perfcompl.jld2", "SimulCompareModels_ns_nfs_perfcompl", SimulCompareModels_ns_nfs_perfcompl);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_Eprod_nfs_perfcompl.jld2", "SimulCompareModels_Eprod_nfs_perfcompl", SimulCompareModels_Eprod_nfs_perfcompl);
        # 3 
        d_elast = 3.0; rscale=1.0;
        SimulCompareModels_ns_nfs_delast3 = WelfareCompare_Compile_nofs("main",d_elast,rscale);
        SimulCompareModels_Eprod_nfs_delast3 = WelfareCompare_Compile_nofs("Eprod",d_elast,rscale);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_ns_nfs_delast3.jld2", "SimulCompareModels_ns_nfs_delast3", SimulCompareModels_ns_nfs_delast3);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_Eprod_nfs_delast3.jld2", "SimulCompareModels_Eprod_nfs_delast3", SimulCompareModels_Eprod_nfs_delast3);
        # 2
        d_elast = 2.0; rscale=1.0;
        SimulCompareModels_ns_nfs_delast2 = WelfareCompare_Compile_nofs("main",d_elast,rscale);
        SimulCompareModels_Eprod_nfs_delast2 = WelfareCompare_Compile_nofs("Eprod",d_elast,rscale);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_ns_nfs_delast2.jld2", "SimulCompareModels_ns_nfs_delast2", SimulCompareModels_ns_nfs_delast2);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_Eprod_nfs_delast2.jld2", "SimulCompareModels_Eprod_nfs_delast2", SimulCompareModels_Eprod_nfs_delast2);
        # 1.2
        d_elast = 1.2; rscale=1.0;
        SimulCompareModels_ns_nfs_delast1_2 = WelfareCompare_Compile_nofs("main",d_elast,rscale);
        SimulCompareModels_Eprod_nfs_delast1_2 = WelfareCompare_Compile_nofs("Eprod",d_elast,rscale);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_ns_nfs_delast1_2.jld2", "SimulCompareModels_ns_nfs_delast1_2", SimulCompareModels_ns_nfs_delast1_2);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_Eprod_nfs_delast1_2.jld2", "SimulCompareModels_Eprod_nfs_delast1_2", SimulCompareModels_Eprod_nfs_delast1_2);
        # 0.5
        d_elast = 0.5; rscale=1.0;
        SimulCompareModels_ns_nfs_delast0_5 = WelfareCompare_Compile_nofs("main",d_elast,rscale);
        SimulCompareModels_Eprod_nfs_delast0_5 = WelfareCompare_Compile_nofs("Eprod",d_elast,rscale);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_ns_nfs_delast0_5.jld2", "SimulCompareModels_ns_nfs_delast0_5", SimulCompareModels_ns_nfs_delast0_5);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_Eprod_nfs_delast0_5.jld2", "SimulCompareModels_Eprod_nfs_delast0_5", SimulCompareModels_Eprod_nfs_delast0_5);
        # 0.1
        d_elast = 0.1; rscale=1.0;
        SimulCompareModels_ns_nfs_delast0_1 = WelfareCompare_Compile_nofs("main",d_elast,rscale);
        SimulCompareModels_Eprod_nfs_delast0_1 = WelfareCompare_Compile_nofs("Eprod",d_elast,rscale);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_ns_nfs_delast0_1.jld2", "SimulCompareModels_ns_nfs_delast0_1", SimulCompareModels_ns_nfs_delast0_1);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_Eprod_nfs_delast0_1.jld2", "SimulCompareModels_Eprod_nfs_delast0_1", SimulCompareModels_Eprod_nfs_delast0_1);
        # 4
        d_elast = 4.0; rscale=1.0;
        SimulCompareModels_ns_nfs_delast4 = WelfareCompare_Compile_nofs("main",d_elast,rscale);
        SimulCompareModels_Eprod_nfs_delast4 = WelfareCompare_Compile_nofs("Eprod",d_elast,rscale);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_ns_nfs_delast4.jld2", "SimulCompareModels_ns_nfs_delast4", SimulCompareModels_ns_nfs_delast4);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_Eprod_nfs_delast4.jld2", "SimulCompareModels_Eprod_nfs_delast4", SimulCompareModels_Eprod_nfs_delast4);
        # 6
        d_elast = 6.0; rscale=1.0;
        SimulCompareModels_ns_nfs_delast6 = WelfareCompare_Compile_nofs("main",d_elast,rscale);
        SimulCompareModels_Eprod_nfs_delast6 = WelfareCompare_Compile_nofs("Eprod",d_elast,rscale);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_ns_nfs_delast6.jld2", "SimulCompareModels_ns_nfs_delast6", SimulCompareModels_ns_nfs_delast6);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_Eprod_nfs_delast6.jld2", "SimulCompareModels_Eprod_nfs_delast6", SimulCompareModels_Eprod_nfs_delast6);
        # 8
        d_elast = 8.0; rscale=1.0;
        SimulCompareModels_ns_nfs_delast8 = WelfareCompare_Compile_nofs("main",d_elast,rscale);
        SimulCompareModels_Eprod_nfs_delast8 = WelfareCompare_Compile_nofs("Eprod",d_elast,rscale);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_ns_nfs_delast8.jld2", "SimulCompareModels_ns_nfs_delast8", SimulCompareModels_ns_nfs_delast8);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_Eprod_nfs_delast8.jld2", "SimulCompareModels_Eprod_nfs_delast8", SimulCompareModels_Eprod_nfs_delast8);
        # 10
        d_elast = 10.0; rscale=1.0;
        SimulCompareModels_ns_nfs_delast10 = WelfareCompare_Compile_nofs("main",d_elast,rscale);
        SimulCompareModels_Eprod_nfs_delast10 = WelfareCompare_Compile_nofs("Eprod",d_elast,rscale);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_ns_nfs_delast10.jld2", "SimulCompareModels_ns_nfs_delast10", SimulCompareModels_ns_nfs_delast10);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_Eprod_nfs_delast10.jld2", "SimulCompareModels_Eprod_nfs_delast10", SimulCompareModels_Eprod_nfs_delast10);
        # 12
        d_elast = 12.0; rscale=1.0;
        SimulCompareModels_ns_nfs_delast12 = WelfareCompare_Compile_nofs("main",d_elast,rscale);
        SimulCompareModels_Eprod_nfs_delast12 = WelfareCompare_Compile_nofs("Eprod",d_elast,rscale);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_ns_nfs_delast12.jld2", "SimulCompareModels_ns_nfs_delast12", SimulCompareModels_ns_nfs_delast12);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_Eprod_nfs_delast12.jld2", "SimulCompareModels_Eprod_nfs_delast12", SimulCompareModels_Eprod_nfs_delast12);
        # 14
        d_elast = 14.0; rscale=1.0;
        SimulCompareModels_ns_nfs_delast14 = WelfareCompare_Compile_nofs("main",d_elast,rscale);
        SimulCompareModels_Eprod_nfs_delast14 = WelfareCompare_Compile_nofs("Eprod",d_elast,rscale);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_ns_nfs_delast14.jld2", "SimulCompareModels_ns_nfs_delast14", SimulCompareModels_ns_nfs_delast14);
        save("/project/6001227/emurrayl/DynamicDiscreteChoice/EnergyCES/Counterfactual/CompareModels/SimulCompareModels_Eprod_nfs_delast14.jld2", "SimulCompareModels_Eprod_nfs_delast14", SimulCompareModels_Eprod_nfs_delast14);
    #
