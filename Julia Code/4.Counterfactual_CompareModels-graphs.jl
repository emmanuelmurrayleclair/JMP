# Policy counterfactual of dynamic discrete choice model with real data - Comparing models
    # Create all figures that compare the trade-off between aggregate output and emission reduction
# Emmanuel Murray Leclair
# August 2023

## Home directory
computer = gethostname() ;
cd()
if computer == "LAPTOP-AKM2CA06" # Laptop
cd("C:/Users/Emmanuel/Dropbox/Prospectus_Emmanuel/ASI/Simulations/DynamicDiscreteChoice/EnergyCES") 
elseif computer == "DESKTOP-1VPTM1B" # Windows desktop
cd("D:/Users/Emmanuel/Dropbox/Prospectus_Emmanuel/ASI/Simulations/DynamicDiscreteChoice/EnergyCES") # Desktop
end

## Make auxiliary directores
Fig_Folder = "Figures"
mkpath(Fig_Folder)
Result_Folder = "Results"
mkpath(Result_Folder)

# Load packages
using SparseArrays, Interpolations, Dierckx, ForwardDiff, Optim, Roots, Parameters, Kronecker, Plots, StatsPlots, NLopt, Distributions, QuantEcon, HDF5
using CSV, DataFrames, DiscreteMarkovChains, StructTypes, StatsBase, Distributed, SharedArrays, DelimitedFiles, NLsolve, ParallelDataTransfer
using FiniteDiff, BenchmarkTools, Distances, FileIO, JLD2, PlotThemes

# Load externally written functions
include("../VFI_Toolbox.jl")

println(" ")
println("-----------------------------------------------------------------------------------------")
println("Dynamic production and fuel set choice in Julia - Steel Manufacturing Data With Selection")
println("                          Counterfactuals - Model comparison - Graphs                    ")
println("-----------------------------------------------------------------------------------------")
println(" ")

# Import simulations
    # Baseline 
    @load "Counterfactuals/CompareModels/SimulCompareModels_full.jld2" SimulCompareModels_full
    @load "Counterfactuals/CompareModels/SimulCompareModels_rswitch.jld2" SimulCompareModels_rswitch
    @load "Counterfactuals/CompareModels/SimulCompareModels_ns.jld2" SimulCompareModels_ns
    @load "Counterfactuals/CompareModels/SimulCompareModels_Eprod.jld2" SimulCompareModels_Eprod
    @load "Counterfactuals/CompareModels/SimulCompareModels_Eprod_fixsub.jld2" SimulCompareModels_Eprod_fixsub
    @load "Counterfactuals/CompareModels/SimulCompareModels_Eprod_nofuelsub.jld2" SimulCompareModels_Eprod_nofuelsub
    @load "Counterfactuals/CompareModels/SimulCompareModels_Eprod_noinputsub.jld2" SimulCompareModels_Eprod_noinputsub
    @load "Counterfactuals/CompareModels/SimulCompareModels_Eprod_noinputsub_sameF.jld2" SimulCompareModels_Eprod_noinputsub_sameF
    # Varying elasticity of demand
    @load "Counterfactuals/CompareModels/SimulCompareModels_ns_nfs_perfcompl.jld2" SimulCompareModels_ns_nfs_perfcompl
    @load "Counterfactuals/CompareModels/SimulCompareModels_Eprod_nfs_perfcompl.jld2" SimulCompareModels_Eprod_nfs_perfcompl3; SimulCompareModels_Eprod_nfs_perfcompl = SimulCompareModels_Eprod_nfs_perfcompl3;
    @load "Counterfactuals/CompareModels/SimulCompareModels_ns_nfs_delast1_2.jld2" SimulCompareModels_ns_nfs_delast1_2
    @load "Counterfactuals/CompareModels/SimulCompareModels_Eprod_nfs_delast1_2.jld2" SimulCompareModels_Eprod_nfs_delast1_2
    @load "Counterfactuals/CompareModels/SimulCompareModels_ns_nfs_delast2.jld2" SimulCompareModels_ns_nfs_delast2
    @load "Counterfactuals/CompareModels/SimulCompareModels_Eprod_nfs_delast2.jld2" SimulCompareModels_Eprod_nfs_delast2
    @load "Counterfactuals/CompareModels/SimulCompareModels_ns_nfs_delast3.jld2" SimulCompareModels_ns_nfs_delast3
    @load "Counterfactuals/CompareModels/SimulCompareModels_Eprod_nfs_delast3.jld2" SimulCompareModels_Eprod_nfs_delast3
    @load "Counterfactuals/CompareModels/SimulCompareModels_ns_nfs_delast4.jld2" SimulCompareModels_ns_nfs_delast4
    @load "Counterfactuals/CompareModels/SimulCompareModels_Eprod_nfs_delast4.jld2" SimulCompareModels_Eprod_nfs_delast4
    @load "Counterfactuals/CompareModels/SimulCompareModels_ns_nfs_delast6.jld2" SimulCompareModels_ns_nfs_delast6
    @load "Counterfactuals/CompareModels/SimulCompareModels_Eprod_nfs_delast6.jld2" SimulCompareModels_Eprod_nfs_delast6
    @load "Counterfactuals/CompareModels/SimulCompareModels_ns_nfs_delast8.jld2" SimulCompareModels_ns_nfs_delast8
    @load "Counterfactuals/CompareModels/SimulCompareModels_Eprod_nfs_delast8.jld2" SimulCompareModels_Eprod_nfs_delast8
    @load "Counterfactuals/CompareModels/SimulCompareModels_ns_nfs_delast10.jld2" SimulCompareModels_ns_nfs_delast10
    @load "Counterfactuals/CompareModels/SimulCompareModels_Eprod_nfs_delast10.jld2" SimulCompareModels_Eprod_nfs_delast10
    @load "Counterfactuals/CompareModels/SimulCompareModels_ns_nfs_delast12.jld2" SimulCompareModels_ns_nfs_delast12
    @load "Counterfactuals/CompareModels/SimulCompareModels_Eprod_nfs_delast12.jld2" SimulCompareModels_Eprod_nfs_delast12
    @load "Counterfactuals/CompareModels/SimulCompareModels_ns_nfs_delast14.jld2" SimulCompareModels_ns_nfs_delast14
    @load "Counterfactuals/CompareModels/SimulCompareModels_Eprod_nfs_delast14.jld2" SimulCompareModels_Eprod_nfs_delast14
#

# Extract variables of interest 
    # emissions 
    co2_full = SimulCompareModels_full["co2"];
    co2_rswitch = SimulCompareModels_rswitch["co2"];
    co2_ns = SimulCompareModels_ns["co2"];
    co2_Eprod = SimulCompareModels_Eprod["co2"];
    co2_Eprod_fixsub = SimulCompareModels_Eprod_fixsub["co2"];
    co2_nofuelsub = SimulCompareModels_Eprod_nofuelsub["co2"];
    co2_noinputsub = SimulCompareModels_Eprod_noinputsub["co2"];
    co2_noinputsub_sameF = SimulCompareModels_Eprod_noinputsub_sameF["co2"];
    co2_ns_perfcompl = SimulCompareModels_ns_nfs_perfcompl["co2"];
    co2_Eprod_perfcompl = SimulCompareModels_Eprod_nfs_perfcompl["co2"];
    co2_ns_delast1_2 = SimulCompareModels_ns_nfs_delast1_2["co2"];
    co2_Eprod_delast1_2 = SimulCompareModels_Eprod_nfs_delast1_2["co2"];
    co2_ns_delast2 = SimulCompareModels_ns_nfs_delast2["co2"];
    co2_Eprod_delast2 = SimulCompareModels_Eprod_nfs_delast2["co2"];
    co2_ns_delast3 = SimulCompareModels_ns_nfs_delast3["co2"];
    co2_Eprod_delast3 = SimulCompareModels_Eprod_nfs_delast3["co2"];
    co2_ns_delast4 = SimulCompareModels_ns_nfs_delast4["co2"];
    co2_Eprod_delast4 = SimulCompareModels_Eprod_nfs_delast4["co2"];
    co2_ns_delast6 = SimulCompareModels_ns_nfs_delast6["co2"];
    co2_Eprod_delast6 = SimulCompareModels_Eprod_nfs_delast6["co2"];
    co2_ns_delast8 = SimulCompareModels_ns_nfs_delast8["co2"];
    co2_Eprod_delast8 = SimulCompareModels_Eprod_nfs_delast8["co2"];
    co2_ns_delast10 = SimulCompareModels_ns_nfs_delast10["co2"];
    co2_Eprod_delast10 = SimulCompareModels_Eprod_nfs_delast10["co2"];
    co2_ns_delast12 = SimulCompareModels_ns_nfs_delast12["co2"];
    co2_Eprod_delast12 = SimulCompareModels_Eprod_nfs_delast12["co2"];
    co2_ns_delast14 = SimulCompareModels_ns_nfs_delast14["co2"];
    co2_Eprod_delast14 = SimulCompareModels_Eprod_nfs_delast14["co2"];
    # Output
    y_full = SimulCompareModels_full["y"];
    y_rswitch = SimulCompareModels_rswitch["y"];
    y_ns = SimulCompareModels_ns["y"];
    y_Eprod = SimulCompareModels_Eprod["y"];
    y_Eprod_fixsub = SimulCompareModels_Eprod_fixsub["y"];
    y_nofuelsub = SimulCompareModels_Eprod_nofuelsub["y"];
    y_noinputsub = SimulCompareModels_Eprod_noinputsub["y"];
    y_noinputsub_sameF = SimulCompareModels_Eprod_noinputsub_sameF["y"];
    y_ns_perfcompl = SimulCompareModels_ns_nfs_perfcompl["y"];
    y_Eprod_perfcompl = SimulCompareModels_Eprod_nfs_perfcompl["y"];
    y_ns_delast1_2 = SimulCompareModels_ns_nfs_delast1_2["y"];
    y_Eprod_delast1_2 = SimulCompareModels_Eprod_nfs_delast1_2["y"];
    y_ns_delast2 = SimulCompareModels_ns_nfs_delast2["y"];
    y_Eprod_delast2 = SimulCompareModels_Eprod_nfs_delast2["y"];
    y_ns_delast3 = SimulCompareModels_ns_nfs_delast3["y"];
    y_Eprod_delast3 = SimulCompareModels_Eprod_nfs_delast3["y"];
    y_ns_delast4 = SimulCompareModels_ns_nfs_delast3["y"];
    y_Eprod_delast4 = SimulCompareModels_Eprod_nfs_delast3["y"];
    y_ns_delast6 = SimulCompareModels_ns_nfs_delast6["y"];
    y_Eprod_delast6 = SimulCompareModels_Eprod_nfs_delast6["y"];
    y_ns_delast8 = SimulCompareModels_ns_nfs_delast8["y"];
    y_Eprod_delast8 = SimulCompareModels_Eprod_nfs_delast8["y"];
    y_ns_delast10 = SimulCompareModels_ns_nfs_delast10["y"];
    y_Eprod_delast10 = SimulCompareModels_Eprod_nfs_delast10["y"];
    y_ns_delast12 = SimulCompareModels_ns_nfs_delast12["y"];
    y_Eprod_delast12 = SimulCompareModels_Eprod_nfs_delast12["y"];
    y_ns_delast14 = SimulCompareModels_ns_nfs_delast14["y"];
    y_Eprod_delast14 = SimulCompareModels_Eprod_nfs_delast14["y"];
    # Profit (including fixed costs) 
    profit_full = SimulCompareModels_full["profit"] .+ SimulCompareModels_full["FC"];
    profit_rswitch = SimulCompareModels_rswitch["profit"] .+ SimulCompareModels_rswitch["FC"]; 
    # price of energy
    pE_full = SimulCompareModels_full["pE"];
    pE_rswitch = SimulCompareModels_rswitch["pE"]; 
    # Gas and coal quantities (for extensive margin)
    gas_ind_full = SimulCompareModels_full["gas_ind"];
    coal_ind_full = SimulCompareModels_full["coal_ind"];
    gas_ind_rswitch = SimulCompareModels_rswitch["gas_ind"];
    coal_ind_rswitch = SimulCompareModels_rswitch["coal_ind"];
#

# Graphs of tradeoff for forward simulation
    ntau = size(y_full,2);
    yr_fs = 1:41;
    β=0.9;
    # Construct percentage decrease in NPV of co2 emissions 
        # Initialize
        co2_perc_full = zeros(ntau);
        co2_perc_rswitch = zeros(ntau);
        co2_perc_ns = zeros(ntau);
        co2_perc_Eprod = zeros(ntau);
        co2_perc_nofuelsub = zeros(ntau);
        co2_perc_noinputsub = zeros(ntau);
        co2_perc_noinputsub_sameF = zeros(ntau);
        # Get net present value (NPV) of co2 emissions for each tax rate, then take percentage decrease relative to no tax
        for tau = 1:ntau
            for t = yr_fs
                co2_perc_full[tau] += (β^(t-1))*co2_full[t,tau]; 
                co2_perc_rswitch[tau] += (β^(t-1))*co2_rswitch[t,tau];
                co2_perc_ns[tau] += (β^(t-1))*co2_ns[t,tau]; 
                co2_perc_Eprod[tau] += (β^(t-1))*co2_Eprod[t,tau];
                co2_perc_nofuelsub[tau] += (β^(t-1))*co2_nofuelsub[t,tau];
                co2_perc_noinputsub[tau] += (β^(t-1))*co2_noinputsub[t,tau];
                co2_perc_noinputsub_sameF[tau] += (β^(t-1))*co2_noinputsub_sameF[t,tau];
            end
            if tau > 1
                co2_perc_full[tau] = ((co2_perc_full[1]-co2_perc_full[tau])/co2_perc_full[1])*100;
                co2_perc_rswitch[tau] = ((co2_perc_rswitch[1]-co2_perc_rswitch[tau])/co2_perc_rswitch[1])*100;
                co2_perc_ns[tau] = ((co2_perc_ns[1]-co2_perc_ns[tau])/co2_perc_ns[1])*100;
                co2_perc_Eprod[tau] = ((co2_perc_Eprod[1]-co2_perc_Eprod[tau])/co2_perc_Eprod[1])*100;
                co2_perc_nofuelsub[tau] = ((co2_perc_nofuelsub[1]-co2_perc_nofuelsub[tau])/co2_perc_nofuelsub[1])*100;
                co2_perc_noinputsub[tau] = ((co2_perc_noinputsub[1]-co2_perc_noinputsub[tau])/co2_perc_noinputsub[1])*100;
                co2_perc_noinputsub_sameF[tau] = ((co2_perc_noinputsub_sameF[1]-co2_perc_noinputsub_sameF[tau])/co2_perc_noinputsub_sameF[1])*100;
            end
        end
        co2_perc_full[1] = 0;
        co2_perc_rswitch[1] = 0;
        co2_perc_ns[1] = 0;
        co2_perc_Eprod[1] = 0;
        co2_perc_nofuelsub[1] = 0;
        co2_perc_noinputsub[1] = 0;
        co2_perc_noinputsub_sameF[1] = 0;
        # Compare with literature (Fowlie et al. 2016)
            co2_perc_fowlie = [0
            14.44099379
            27.48447205
            39.59627329
            49.8447205
            56.83229814
            63.35403727
            68.47826087
            72.67080745
            76.39751553
            81.05590062
            84.31677019
            86.64596273
            88.04347826];
        #
    #
    # Construct percentage of NPV of no tax output
        # Initialize
        y_perc_full = zeros(ntau);
        y_perc_rswitch = zeros(ntau);
        y_perc_ns = zeros(ntau);
        y_perc_Eprod = zeros(ntau);
        y_perc_nofuelsub = zeros(ntau);
        y_perc_noinputsub = zeros(ntau);
        y_perc_noinputsub_sameF = zeros(ntau);
        # Get net present value (NPV) of output for each tax rate, then take percentage of no tax output
		for tau = 1:ntau
            for t = yr_fs
                y_perc_full[tau] += (β^(t-1))*y_full[t,tau]; 
                y_perc_rswitch[tau] += (β^(t-1))*y_rswitch[t,tau]; 
                y_perc_ns[tau] += (β^(t-1))*y_ns[t,tau]; 
                y_perc_Eprod[tau] += (β^(t-1))*y_Eprod[t,tau]; 
                y_perc_nofuelsub[tau] += (β^(t-1))*y_nofuelsub[t,tau]; 
                y_perc_noinputsub[tau] += (β^(t-1))*y_noinputsub[t,tau]; 
                y_perc_noinputsub_sameF[tau] += (β^(t-1))*y_noinputsub_sameF[t,tau]; 
            end
            if tau > 1
                y_perc_full[tau] = (y_perc_full[tau]/y_perc_full[1])*100;
                y_perc_rswitch[tau] = (y_perc_rswitch[tau]/y_perc_rswitch[1])*100;
                y_perc_ns[tau] = (y_perc_ns[tau]/y_perc_ns[1])*100;
                y_perc_Eprod[tau] = (y_perc_Eprod[tau]/y_perc_Eprod[1])*100;
                y_perc_nofuelsub[tau] = (y_perc_nofuelsub[tau]/y_perc_nofuelsub[1])*100;
                y_perc_noinputsub[tau] = (y_perc_noinputsub[tau]/y_perc_noinputsub[1])*100;
                y_perc_noinputsub_sameF[tau] = (y_perc_noinputsub_sameF[tau]/y_perc_noinputsub_sameF[1])*100;
            end
		end
        y_perc_full[1] = 100;
        y_perc_rswitch[1] = 100;
        y_perc_ns[1] = 100;
        y_perc_Eprod[1] = 100;
        y_perc_nofuelsub[1] = 100;
        y_perc_noinputsub[1] = 100;
        y_perc_noinputsub_sameF[1] = 100;
        # Compare with literature (Fowlie et al. 2016)
            y_perc_fowlie = [100
            84.88372093
            71.51162791
            59.88372093
            50
            43.02325581
            37.20930233
            33.72093023
            30.81395349
            29.06976744
            26.74418605
            24.41860465
            22.6744186
            21.51162791];
        #
    #
    # Construct percentage of NPV of no tax profits
        # Initialize
        profit_perc_full = zeros(ntau);
        profit_perc_rswitch = zeros(ntau);
        # Get net present value (NPV) of output for each tax rate, then take percentage of no tax output
        for tau = 1:ntau
            for t = yr_fs
                profit_perc_full[tau] += (β^(t-1))*profit_full[t,tau]; 
                profit_perc_rswitch[tau] += (β^(t-1))*profit_rswitch[t,tau];
            end
            if tau > 1
                profit_perc_full[tau] = (profit_perc_full[tau]/profit_perc_full[1])*100;
                profit_perc_rswitch[tau] = (profit_perc_rswitch[tau]/profit_perc_rswitch[1])*100;
            end
        end
        profit_perc_full[1] = 100;
        profit_perc_rswitch[1] = 100;
    #
    # Make graphs
    theme(:dao)
    # Output graphs 
    # Baseline plot with only full model
        p1 = plot(y_perc_full,co2_perc_full,marker = (:circle,3),label="Full Model", gridalpha=0.2,
                xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
                legend=:bottomleft,legendfontsize=10, line=(2.5,:solid), tickfontsize=10,minorgrid=false)
        savefig(p1,"Counterfactuals/CompareModels/Graphs/TaxTradeoff_Full.pdf");
    #
    # Plot comparing full model with restricted switching
        plot(y_perc_full,co2_perc_full,marker=(:circle,3),label="Full Model",line=(2,:solid));
        plot!(y_perc_rswitch,co2_perc_rswitch,marker=(:square,3),label="No Switching",line=(2,:solid));
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Full-rswitch.pdf");
    #
    # Test with on fuel and no input substitution
        plot(y_perc_Eprod,co2_perc_Eprod,marker=(:circle,3),label="Eprod",line=(2,:solid));
        plot!(y_perc_nofuelsub,co2_perc_nofuelsub,marker=(:square,3),label="nofuelsub",line=(2,:solid));
        plot!(y_perc_noinputsub,co2_perc_noinputsub,marker=(:square,3),label="noinputsub",line=(2,:solid));
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
    #
    # Plot comparing full model with no switching
        plot(y_perc_full,co2_perc_full,marker=(:circle,3),label="Full Model",line=(2,:solid));
        plot!(y_perc_ns,co2_perc_ns,marker=(:square,3),label="No Switching",line=(2,:solid));
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Full-Ns.pdf");
    #
    # Plot comparing No switching with energy productivity
        plot(y_perc_ns,co2_perc_ns,marker=(:square,3),label="No Switching",line=(2,:solid));
        plot!(y_perc_Eprod,co2_perc_Eprod,marker=(:star,3),label="No Switching and No Fuel Productivity",line=(2,:solid));
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Ns-Eprod.pdf");
    #
    # Plot comparing Energy productivity, no switching and full models
        plot(y_perc_full,co2_perc_full,marker=(:circle,3),label="Full Model",line=(2,:solid));
        plot!(y_perc_ns,co2_perc_ns,marker=(:square,3),label="No Switching",line=(2,:solid));
        plot!(y_perc_Eprod,co2_perc_Eprod,marker=(:diamond,3),label="No Switching and No Fuel Productivity",line=(2,:solid));
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Full-Ns-Eprod.pdf");
    #
    # Plot comparing Energy productivity, restricted and full model
        plot(y_perc_full,co2_perc_full,marker=(:circle,3),label="Full Model",line=(2,:solid));
        plot!(y_perc_rswitch,co2_perc_rswitch,marker=(:square,3),label="No Switching",line=(2,:solid));
        plot!(y_perc_Eprod,co2_perc_Eprod,marker=(:diamond,3),label="No Switching and no fuel Productivity",line=(2,:solid));
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Full-rswitch-Eprod.pdf");
    #
    # Plot comparing all 4 models
        plot(y_perc_full,co2_perc_full,marker=(:circle,3),label="Full Model",line=(2,:solid));
        plot!(y_perc_rswitch,co2_perc_rswitch,marker=(:square,3),label="Restricted Switching",line=(2,:solid));
        plot!(y_perc_ns,co2_perc_ns,marker=(:diamond,3),label="No Switching",line=(2,:solid));
        plot!(y_perc_Eprod,co2_perc_Eprod,marker=(:star,3),label="No Switching and no fuel Productivity",line=(2,:solid));
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Full-rswitch-Ns-Eprod.pdf");
    #
    # Plot comparing full model, most restricted model (no switching, no input substitution) with literature
        plot(y_perc_full,co2_perc_full,label="Full Model",line=(2,:solid));
        plot!(y_perc_noinputsub,co2_perc_noinputsub,label="No Input Substitution",line=(2,:solid));
        plot!(y_perc_fowlie,co2_perc_fowlie,label="Fowlie et al. (2016)",line=(2,:solid));
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:topright,legendfontsize=8, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Full-noinputprod-fowlie.pdf");

    # Comparing full model, energy productivity and no input substitution 
        plot(y_perc_full,co2_perc_full,marker=(:circle,3),label="Full Model",line=(2,:solid));
        plot!(y_perc_Eprod,co2_perc_Eprod,marker=(:square,3),label="No Switching and No Fuel Productivity",line=(2,:solid));
        plot!(y_perc_noinputsub,co2_perc_noinputsub,marker=(:diamond,3),label="+ No Input Substitution",line=(2,:solid));
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Full-Eprod-noinputprod.pdf");
    #
    # Table that records average elasticity by model
        y_perc_decrease_full = -y_perc_full.+100;
        y_perc_decrease_ns = -y_perc_ns.+100;
        y_perc_decrease_Eprod = -y_perc_Eprod.+100;
        y_perc_decrease_noinputsub = -y_perc_noinputsub.+100;
        y_perc_decrease_fowlie = -y_perc_fowlie.+100;
        elast_full = co2_perc_full[2:end]./y_perc_decrease_full[2:end];
        elast_ns = co2_perc_ns[2:end]./y_perc_decrease_ns[2:end];
        elast_Eprod = co2_perc_Eprod[2:end]./y_perc_decrease_Eprod[2:end];
        elast_noinputsub = co2_perc_noinputsub[2:end]./y_perc_decrease_noinputsub[2:end];
        elast_fowlie = co2_perc_fowlie[2:end]./y_perc_decrease_fowlie[2:end];
        global elas_full = "$(round(mean(elast_full),digits=2))"
        global elas_ns = "$(round(mean(elast_ns),digits=2))"
        global elas_Eprod = "$(round(mean(elast_Eprod),digits=2))"
        global elas_noinputsub = "$(round(mean(elast_noinputsub),digits=2))"
        global elas_fowlie = "$(round(mean(elast_fowlie),digits=2))"
        tex_table = """
        \\begin{tabular}{@{}lc@{}}
        \\toprule
        & \\begin{tabular}[c]{@{}c@{}}Average Elasticity\\\\ \$\\frac{\\% \\Delta CO_{2e}}{\\% \\Delta Y}\$\\end{tabular} \\\\ \\midrule
        Full Model & $elas_full \\\\
        No Switching & $elas_ns \\\\
        No Switching and No Fuel Producitivty & $elas_Eprod \\\\
        No Input Substitution & $elas_noinputsub \\\\
        \\textbf{Fowlie et al. (2016)} & $elas_fowlie \\\\ \\bottomrule
        \\end{tabular}
        """
        fname = "AverageElasticity_models.tex"
        dirpath = "Counterfactuals/CompareModels/Tables"
        fpath = joinpath(dirpath,fname);
        open(fpath, "w") do file
            write(file, tex_table)
        end
    #

    # Graphs with profits instead of output
    # Plot comparing no switching and switching 
        plot(profit_perc_full,co2_perc_full,marker=(:circle,3),label="Full Model",line=(2,:solid));
        plot!(profit_perc_rswitch,co2_perc_rswitch,marker=(:square,3),label="No Switching",line=(2,:solid));
        plot!(xlabel="Aggregate Profits (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Profit_Full-rswitch.pdf");
    #
#

# Graphs of tradeoff for static simulation (comparing trade-off across values of elasticity of demand)
    # Varying elasticity of demand
        # Percentage decrease in emissions
        co2_perc_ns_perfcompl = ((co2_ns_perfcompl[1].-co2_ns_perfcompl[:])/co2_ns_perfcompl[1])*100;
        co2_perc_Eprod_perfcompl = ((co2_Eprod_perfcompl[1].-co2_Eprod_perfcompl[:])/co2_Eprod_perfcompl[1])*100;
        co2_perc_ns_delast1_2 = ((co2_ns_delast1_2[1].-co2_ns_delast1_2[:])/co2_ns_delast1_2[1])*100;
        co2_perc_Eprod_delast1_2 = ((co2_Eprod_delast1_2[1].-co2_Eprod_delast1_2[:])/co2_Eprod_delast1_2[1])*100;
        co2_perc_ns_delast2 = ((co2_ns_delast2[1].-co2_ns_delast2[:])/co2_ns_delast2[1])*100;
        co2_perc_Eprod_delast2 = ((co2_Eprod_delast2[1].-co2_Eprod_delast2[:])/co2_Eprod_delast2[1])*100;
        co2_perc_ns_delast3 = ((co2_ns_delast3[1].-co2_ns_delast3[:])/co2_ns_delast3[1])*100;
        co2_perc_Eprod_delast3 = ((co2_Eprod_delast3[1].-co2_Eprod_delast3[:])/co2_Eprod_delast3[1])*100;
        co2_perc_ns_delast4 = ((co2_ns_delast4[1].-co2_ns_delast4[:])/co2_ns_delast4[1])*100;
        co2_perc_Eprod_delast4 = ((co2_Eprod_delast4[1].-co2_Eprod_delast4[:])/co2_Eprod_delast4[1])*100;
        co2_perc_ns_delast6 = ((co2_ns_delast6[1].-co2_ns_delast6[:])/co2_ns_delast6[1])*100;
        co2_perc_Eprod_delast6 = ((co2_Eprod_delast6[1].-co2_Eprod_delast6[:])/co2_Eprod_delast6[1])*100;
        co2_perc_ns_delast8 = ((co2_ns_delast8[1].-co2_ns_delast8[:])/co2_ns_delast8[1])*100;
        co2_perc_Eprod_delast8 = ((co2_Eprod_delast8[1].-co2_Eprod_delast8[:])/co2_Eprod_delast8[1])*100;
        co2_perc_ns_delast10 = ((co2_ns_delast10[1].-co2_ns_delast10[:])/co2_ns_delast10[1])*100;
        co2_perc_Eprod_delast10 = ((co2_Eprod_delast10[1].-co2_Eprod_delast10[:])/co2_Eprod_delast10[1])*100;
        co2_perc_ns_delast12 = ((co2_ns_delast12[1].-co2_ns_delast12[:])/co2_ns_delast12[1])*100;
        co2_perc_Eprod_delast12 = ((co2_Eprod_delast12[1].-co2_Eprod_delast12[:])/co2_Eprod_delast12[1])*100;
        co2_perc_ns_delast14 = ((co2_ns_delast14[1].-co2_ns_delast14[:])/co2_ns_delast14[1])*100;
        co2_perc_Eprod_delast14 = ((co2_Eprod_delast14[1].-co2_Eprod_delast14[:])/co2_Eprod_delast14[1])*100;
        # Percentage of no tax output
        y_perc_ns_perfcompl = (y_ns_perfcompl[:]/y_ns_perfcompl[1])*100;
        y_perc_Eprod_perfcompl = (y_Eprod_perfcompl[:]/y_Eprod_perfcompl[1])*100;
        y_perc_ns_delast1_2 = (y_ns_delast1_2[:]/y_ns_delast1_2[1])*100;
        y_perc_Eprod_delast1_2 = (y_Eprod_delast1_2[:]/y_Eprod_delast1_2[1])*100;
        y_perc_ns_delast2 = (y_ns_delast2[:]/y_ns_delast2[1])*100;
        y_perc_Eprod_delast2 = (y_Eprod_delast2[:]/y_Eprod_delast2[1])*100;
        y_perc_ns_delast3 = (y_ns_delast3[:]/y_ns_delast3[1])*100;
        y_perc_Eprod_delast3 = (y_Eprod_delast3[:]/y_Eprod_delast3[1])*100;
        y_perc_ns_delast4 = (y_ns_delast4[:]/y_ns_delast4[1])*100;
        y_perc_Eprod_delast4 = (y_Eprod_delast4[:]/y_Eprod_delast4[1])*100;
        y_perc_ns_delast6 = (y_ns_delast6[:]/y_ns_delast6[1])*100;
        y_perc_Eprod_delast6 = (y_Eprod_delast6[:]/y_Eprod_delast6[1])*100;
        y_perc_ns_delast8 = (y_ns_delast8[:]/y_ns_delast8[1])*100;
        y_perc_Eprod_delast8 = (y_Eprod_delast8[:]/y_Eprod_delast8[1])*100;
        y_perc_ns_delast10 = (y_ns_delast10[:]/y_ns_delast10[1])*100;
        y_perc_Eprod_delast10 = (y_Eprod_delast10[:]/y_Eprod_delast10[1])*100;
        y_perc_ns_delast12 = (y_ns_delast12[:]/y_ns_delast12[1])*100;
        y_perc_Eprod_delast12 = (y_Eprod_delast12[:]/y_Eprod_delast12[1])*100;
        y_perc_ns_delast14 = (y_ns_delast14[:]/y_ns_delast14[1])*100;
        y_perc_Eprod_delast14 = (y_Eprod_delast14[:]/y_Eprod_delast14[1])*100;
    #
    # Create graphs (separate for each elasticity of demand)
        theme(:dao)
        # Perfect complements
        plot(y_perc_ns_perfcompl,co2_perc_ns_perfcompl,marker=(:circle,3),label="No Switching");
        plot!(y_perc_Eprod_perfcompl,co2_perc_Eprod_perfcompl,marker=(:square,3),label="No Switching and No Fuel Productivity");
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Ns-Eprod_perfcompl.pdf");
        # demand elasticity = 1.2
        plot(y_perc_ns_delast1_2,co2_perc_ns_delast1_2,marker=(:circle,3),label="No Switching");
        plot!(y_perc_Eprod_delast1_2,co2_perc_Eprod_delast1_2,marker=(:square,3),label="No Switching and No Fuel Productivity");
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Ns-Eprod_delast1_2.pdf");
        # demand elasticity = 2
        plot(y_perc_ns_delast2,co2_perc_ns_delast2,marker=(:circle,3),label="No Switching");
        plot!(y_perc_Eprod_delast2,co2_perc_Eprod_delast2,marker=(:square,3),label="No Switching and No Fuel Productivity");
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Ns-Eprod_delast2.pdf");
        # demand elasticity = 3
        plot(y_perc_ns_delast3,co2_perc_ns_delast3,marker=(:circle,3),label="No Switching");
        plot!(y_perc_Eprod_delast3,co2_perc_Eprod_delast3,marker=(:square,3),label="No Switching and No Fuel Productivity");
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Ns-Eprod_delast3.pdf");
        # demand elasticity = 4
        plot(y_perc_ns_delast4,co2_perc_ns_delast4,marker=(:circle,3),label="No Switching");
        plot!(y_perc_Eprod_delast4,co2_perc_Eprod_delast4,marker=(:square,3),label="No Switching and No Fuel Productivity");
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Ns-Eprod_delast4.pdf");
        # demand elasticity = 6
        plot(y_perc_ns_delast6,co2_perc_ns_delast6,marker=(:circle,3),label="No Switching and No Fuel Productivity",line=(2,:solid));
        plot!(y_perc_Eprod_delast6,co2_perc_Eprod_delast6,marker=(:square,3),label="No Switching and Eneryg Productivity",line=(2,:solid));
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Ns-Eprod_delast6.pdf");
        # demand elasticity = 8
        plot(y_perc_ns_delast8,co2_perc_ns_delast8,marker=(:circle,3),label="No Switching and No Fuel Productivity",line=(2,:solid));
        plot!(y_perc_Eprod_delast8,co2_perc_Eprod_delast8,marker=(:square,3),label="No Switching and Eneryg Productivity",line=(2,:solid));
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Ns-Eprod_delast8.pdf");
        # demand elasticity = 10
        plot(y_perc_ns_delast10,co2_perc_ns_delast10,marker=(:circle,3),label="No Switching and No Fuel Productivity",line=(2,:solid));
        plot!(y_perc_Eprod_delast10,co2_perc_Eprod_delast10,marker=(:square,3),label="No Switching and Eneryg Productivity",line=(2,:solid));
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Ns-Eprod_delast10.pdf");
        # demand elasticity = 12
        plot(y_perc_ns_delast12,co2_perc_ns_delast12,marker=(:circle,3),label="No Switching and No Fuel Productivity",line=(2,:solid));
        plot!(y_perc_Eprod_delast12,co2_perc_Eprod_delast12,marker=(:square,3),label="No Switching and Eneryg Productivity",line=(2,:solid));
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Ns-Eprod_delast12.pdf");
        # demand elasticity = 14
        plot(y_perc_ns_delast14,co2_perc_ns_delast14,marker=(:circle,3),label="No Switching and No Fuel Productivity",line=(2,:solid));
        plot!(y_perc_Eprod_delast14,co2_perc_Eprod_delast14,marker=(:square,3),label="No Switching and Eneryg Productivity",line=(2,:solid));
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Emission Reduction (% Relative to No Tax)",
        legend=:bottomleft,legendfontsize=10, gridalpha=0.2,tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_Ns-Eprod_delast14.pdf");
    #
    # Interpolate to combine into one graph
        # Perfect complements
        interp_ns_perfcompl = Spline1D(reverse(y_perc_ns_perfcompl),reverse(co2_perc_ns_perfcompl));
        interp_Eprod_perfcompl = Spline1D(reverse(y_perc_Eprod_perfcompl),reverse(co2_perc_Eprod_perfcompl));   
        # demand elasticity = 1.2
        interp_ns_delast1_2 = Spline1D(reverse(y_perc_ns_delast1_2),reverse(co2_perc_ns_delast1_2));
        interp_Eprod_delast1_2 = Spline1D(reverse(y_perc_Eprod_delast1_2),reverse(co2_perc_Eprod_delast1_2));
        # demand elasticity = 2
        interp_ns_delast2 = Spline1D(reverse(y_perc_ns_delast2),reverse(co2_perc_ns_delast2));
        interp_Eprod_delast2 = Spline1D(reverse(y_perc_Eprod_delast2),reverse(co2_perc_Eprod_delast2));
        # demand elasticity = 3
        interp_ns_delast3 = Spline1D(reverse(y_perc_ns_delast3),reverse(co2_perc_ns_delast3));
        interp_Eprod_delast3 = Spline1D(reverse(y_perc_Eprod_delast3),reverse(co2_perc_Eprod_delast3));
        # demand elasticity = 4
        interp_ns_delast4 = Spline1D(reverse(y_perc_ns_delast4),reverse(co2_perc_ns_delast4));
        interp_Eprod_delast4 = Spline1D(reverse(y_perc_Eprod_delast4),reverse(co2_perc_Eprod_delast4));
        # demand elasticity = 6
        interp_ns_delast6 = Spline1D(reverse(y_perc_ns_delast6),reverse(co2_perc_ns_delast6));
        interp_Eprod_delast6 = Spline1D(reverse(y_perc_Eprod_delast6),reverse(co2_perc_Eprod_delast6));
        # demand elasticity = 8
        interp_ns_delast8 = Spline1D(reverse(y_perc_ns_delast8),reverse(co2_perc_ns_delast8));
        interp_Eprod_delast8 = Spline1D(reverse(y_perc_Eprod_delast8),reverse(co2_perc_Eprod_delast8));
        # demand elasticity = 10
        interp_ns_delast10 = Spline1D(reverse(y_perc_ns_delast10),reverse(co2_perc_ns_delast10));
        interp_Eprod_delast10 = Spline1D(reverse(y_perc_Eprod_delast10),reverse(co2_perc_Eprod_delast10));
        # demand elasticity = 12
        interp_ns_delast12 = Spline1D(reverse(y_perc_ns_delast12),reverse(co2_perc_ns_delast12));
        interp_Eprod_delast12 = Spline1D(reverse(y_perc_Eprod_delast12),reverse(co2_perc_Eprod_delast12));
        # demand elasticity = 14
        interp_ns_delast14 = Spline1D(reverse(y_perc_ns_delast14),reverse(co2_perc_ns_delast14));
        interp_Eprod_delast14 = Spline1D(reverse(y_perc_Eprod_delast14),reverse(co2_perc_Eprod_delast14));
    #
    # Create one single graph
        xaxis = 67:0.1:99.9
        relco2_perfcompl = interp_ns_perfcompl(xaxis)./interp_Eprod_perfcompl(xaxis);
        relco2_delast1_2 = interp_ns_delast1_2(xaxis)./interp_Eprod_delast1_2(xaxis);
        relco2_delast2 = interp_ns_delast2(xaxis)./interp_Eprod_delast2(xaxis);
        relco2_delast3= interp_ns_delast3(xaxis)./interp_Eprod_delast3(xaxis);
        relco2_delast4 = interp_ns_delast4(xaxis)./interp_Eprod_delast4(xaxis);
        relco2_delast6 = interp_ns_delast6(xaxis)./interp_Eprod_delast6(xaxis);
        relco2_delast8 = interp_ns_delast8(xaxis)./interp_Eprod_delast8(xaxis);
        relco2_delast10 = interp_ns_delast10(xaxis)./interp_Eprod_delast10(xaxis);
        relco2_delast12 = interp_ns_delast12(xaxis)./interp_Eprod_delast12(xaxis);
        relco2_delast14 = interp_ns_delast14(xaxis)./interp_Eprod_delast14(xaxis);
        plot(xaxis,relco2_perfcompl,label="Perfect Complements",line=(2.5,:solid));
        plot!(xaxis,relco2_delast6,label="Demand Elasticity = 6",line=(2.5,:solid));
        plot!(xaxis,relco2_delast8,label="Demand Elasticity = 8",line=(2.5,:dash));
        plot!(xaxis,relco2_delast10,label="Demand Elasticity = 10",line=(2.5,:dot));
        plot!(xaxis,relco2_delast12,label="Demand Elasticity = 12",line=(2.5,:dashdot));
        plot!(xaxis,relco2_delast14,label="Demand Elasticity = 14",line=(2.5,:dashdotdot));
        plot!(xlabel="Aggregate Output (% of No Tax)", ylabel="Ratio of Emission Reduction",
            legend=:topleft,legendfontsize=8, gridalpha=0.2, tickfontsize=10,minorgrid=false)
        savefig("Counterfactuals/CompareModels/Graphs/TaxTradeoff_delast-Combine.pdf");
    #
#

# Graphs of profits, price of energy and share of plants using gas/coal between switching and no switching, across tax rates
    # Construct variables
    pE_full_avg = zeros(21);
    pE_rswitch_avg = zeros(21);
    profit_full_avg = zeros(21);
    profit_rswitch_avg = zeros(21);
    propgas_full = zeros(21);
    propgas_rswitch = zeros(21);
    propcoal_full = zeros(21);
    propcoal_rswitch = zeros(21);
    for tau = 1:21
        pE_full_avg[tau] = mean(pE_full[:,tau]);
        pE_rswitch_avg[tau] = mean(pE_rswitch[:,tau]);
        profit_full_avg[tau] = mean(profit_full[:,tau]);
        profit_rswitch_avg[tau] = mean(profit_rswitch[:,tau]);
    end
    Dgas_ind_full = !=(0).(gas_ind_full);
    Dgas_ind_rswitch = !=(0).(gas_ind_rswitch);
    Dcoal_ind_full = !=(0).(coal_ind_full);
    Dcoal_ind_rswitch = !=(0).(coal_ind_rswitch);
    for tau = 1:21
        propgas_full[tau] = mean(Dgas_ind_full[:,:,:,tau]);
        propgas_rswitch[tau] = mean(Dgas_ind_rswitch[:,:,:,tau]);
        propcoal_full[tau] = mean(Dcoal_ind_full[:,:,:,tau]);
        propcoal_rswitch[tau] = mean(Dcoal_ind_rswitch[:,:,:,tau]);
    end
    # Graph of energy prices
    xaxis1 = 100*[0,0.001,0.01,0.05,0.1,0.25,0.5,0.75,1,2,2.5,5,10,25,50,100,250,500,1000,10000,100000000000]
    plot(log.(xaxis1[1:15]),pE_full_avg[1:15], line=(2,:solid),label="Full Model");
    plot!(log.(xaxis1[1:15]),pE_rswitch_avg[1:15],line=(2,:dash),label="No Switching");
    plot!(xlabel="Level of Carbon Tax (% of average coal price)", ylabel="Average Price of Energy",
    legend=:topleft,legendfontsize=10,tickfontsize=10,gridalpha=0.2,minorgrid=false,
    xticks = (collect(-2.5:2.5:7.5),[0 1 12 150 1800]))
    savefig("Counterfactuals/CompareModels/Graphs/TaxLevel_pE_Full-rswitch.pdf");
    # Graph of profit
    plot(log.(xaxis1[1:20]),profit_full_avg[1:20],line=(2,:solid),label="Full Model");
    plot!(log.(xaxis1[1:20]),profit_rswitch_avg[1:20],line=(2,:dash),label="No Switching");
    plot!(xlabel="Level of Carbon Tax (% of average coal price)", ylabel="Average Profits",
    legend=:bottomleft,legendfontsize=10,tickfontsize=10,gridalpha=0.2,minorgrid=false,
    xticks = (collect(0:3:12),[1 20 400 8000 160000]))
    savefig("Counterfactuals/CompareModels/Graphs/TaxLevel_profit_Full-rswitch.pdf");
    # Graph of proprtion of plants using gas
    plot(log.(xaxis1[1:20]),propgas_full[1:20],line=(2,:solid),label="Full Model");
    plot!(log.(xaxis1[1:20]),propgas_rswitch[1:20],line=(2,:dash),label="No Switching");
    plot!(xlabel="Level of Carbon Tax (% of average coal price)", ylabel="Proportion of Plants using Natural",
    legend=:bottomleft,legendfontsize=10,tickfontsize=10,gridalpha=0.2,minorgrid=false,ylims=(0.15,0.25),yticks = 0.1:0.02:0.3,
    xticks = (collect(0:3:12),[1 20 400 8000 160000]))
    savefig("Counterfactuals/CompareModels/Graphs/TaxLevel_Dgas_Full-rswitch.pdf");
    # Graph of proprtion of plants using coal
    plot(log.(xaxis1[1:20]),propcoal_full[1:20],line=(2,:solid),xticks = 0,label="Full Model");
    plot!(log.(xaxis1[1:20]),propcoal_rswitch[1:20],line=(2,:dash),xticks = 0,label="No Switching");
    plot!(xlabel="Level of Carbon Tax (% of average coal price)", ylabel="Proportion of Plants using Coal",
    legend=:bottomleft,legendfontsize=10,tickfontsize=10,gridalpha=0.2,minorgrid=false,ylims=(0.2,0.5),yticks = 0.2:0.05:0.5,
    xticks = (collect(0:3:12),[1 20 400 8000 160000]))
    savefig("Counterfactuals/CompareModels/Graphs/TaxLevel_Dcoal_Full-rswitch.pdf");
#