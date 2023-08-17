# All non-CUDA julia functions required to simulate and estimate model in Murray Leclair (2023, JMP)
# Author: Emmanuel Murray Leclair 
# (Latest version): August 2023

##############################################################
#----     FUNCTIONS FOR PRODUCTION FUNCTION MODEL      ------#
##############################################################

    #-----------------------------------------
    #--------- ALL VERSIONS ------------------
    #-----------------------------------------

    ## CES Price of energy (inner pf)
        function pE_func(F,pfψf,p)
            @unpack λ,βo,βg,βc,βe = p
            # In order: oil, elec, coal, gas
            # pfψf = pf/ψf (single state variable since they always enter together)
            po=pfψf[1];
            pe=pfψf[2];
            if F == 12 
                return ( (βo^λ)*((po)^(1-λ)) + (βe^λ)*((pe)^(1-λ)) )^(1/(1-λ)); 
            elseif F == 123 
                pc = pfψf[3];
                return ( (βo^λ)*((po)^(1-λ)) + (βe^λ)*((pe)^(1-λ)) + (βc^λ)*((pc)^(1-λ)) )^(1/(1-λ)); 
            elseif F == 124
                pg = pfψf[3];
                return ( (βo^λ)*((po)^(1-λ)) + (βe^λ)*((pe)^(1-λ)) + (βg^λ)*((pg)^(1-λ)) )^(1/(1-λ)); 
            elseif F == 1234
                pc = pfψf[3];
                pg = pfψf[4];
                return ( (βo^λ)*((po)^(1-λ)) + (βe^λ)*((pe)^(1-λ)) + (βc^λ)*((pc)^(1-λ)) + (βg^λ)*((pg)^(1-λ)) )^(1/(1-λ)); 
            end
        end
    #

    ## CES input price (outer pf)
        function pinput_func(rk,pm,w,pE,p)
            @unpack σ,αk,αm,αl,αe = p
            return ((αk^σ)*(rk^(1-σ)) + (αm^σ)*(pm^(1-σ)) + (αl^σ)*(w^(1-σ)) + (αe^σ)*(pE^(1-σ)))^(1/(1-σ));
        end
    #

    #-----------------------------------------
    #--------- PERFECT COMPETITION ONLY ------
    #-----------------------------------------


    # Period profit function
        function profit_func(pinput,pout,Ygmean,z,p)
            @unpack η = p
            return (((pout*Ygmean*z)^(1/(1-η)))/(pinput^(η/(1-η))))*(η^(η/(1-η)) - η^(1/(1-η)));
        end
    #

    #-----------------------------------------
    #--------- MONOPOLISTIC COMPETITION ------
    #-----------------------------------------
    
    # Output function given aggregate output price index
        function output_func_monopolistic(z,pinput,pout_agg,t,p)
            @unpack η,ρ,θ,Ygmean,d_t,N_t = p;
            # Common functions of parameters
            y_param1 = (1+ρ*(θ-1))/(ρ*(θ-1));
            y_param2 = (η*ρ)/((1-η)*ρ + η);
            # Common terms that compose output
            yterm1 = (z*Ygmean[t+1])^(1/η);
            yterm2 = ((ρ-1)/ρ)*η;
            yterm3 = (exp(d_t[t+1])/N_t[t+1])^(1/ρ);
            yterm4 = (pout_agg^y_param1)/pinput;
            return (yterm1*yterm2*yterm3*yterm4)^(y_param2);
        end
    #
    # Output price given aggregate output price index and output
        function outprice_func_monopolistic_ind(Y,pout_agg,t,p)
            @unpack ρ,θ,Ygmean,N_t,d_t = p;
            # Common functions of parameters
            pout_param1 = (1+ρ*(θ-1))/(ρ*(θ-1));
            # Common terms that compose output
            return ((exp(d_t[t+1])/(N_t[t+1]*Y))^(1/ρ))*(pout_agg^(pout_param1));
        end 
    #
    # Period profit function
        function profit_func_monopolistic(z,pinput,t,p)
            @unpack Ygmean,η,pout_struc = p;
            pout_agg = p.pout_struc[t+1];
            # get output quantity
            Y = output_func_monopolistic(z,pinput,pout_agg,t,p);
            # Get output price
            pout = outprice_func_monopolistic_ind(Y,pout_agg,t,p);
            # Get profit
            return (pout*Y) - ((Y/(z*Ygmean[t+1]))^(1/η))*pinput;
        end
    #
#

##############################################################
#----     FUNCTIONS FOR DISTRIBUTION OF STATE VAIRABLES -----#
##############################################################

# First guess of random effects distribution parameters (from selected empirical distribution)
    function ParamRE_func(θre,M::Model)
        @unpack nc_re,ng_re = M;
        # Extract parameter values and size of grid
        μg_re = θre[1];
        μc_re = θre[2];
        σg_re = θre[3];
        σc_re = θre[4];
        # Disretize both distributions
        # Coal
        MPc_re   = QuantEcon.rouwenhorst(nc_re,0,sqrt(σc_re),μc_re)     ;  # Markov process 
        Πc_re    = MPc_re.p                                             ;  # Transition matrix
        lnc_re_grid = log.(exp.(MPc_re.state_values))                   ;  # Grid in logs
        #lnc_re_grid[3] = lnc_re_grid[3]-1;
        c_re_grid = exp.(lnc_re_grid)                                   ;  # Grid in levels
        # Gas
        MPg_re   = QuantEcon.rouwenhorst(ng_re,0,sqrt(σg_re),μg_re)     ;  # Markov process 
        Πg_re    = MPg_re.p                                             ;  # Transition matrix
        lng_re_grid = log.(exp.(MPg_re.state_values))                   ;  # Grid in logs 
        g_re_grid = exp.(lng_re_grid)                                   ;  # Grid in levels
        # Update model
        M = Model(M; μg_re = copy(μg_re), μc_re = copy(μc_re),
                    σg_re = copy(σg_re), σc_re = copy(σc_re), 
                    π_uncond = Πc_re[1,:]*Πg_re[1,:]',
                    ng_re = copy(ng_re),nc_re = copy(nc_re),
                    Πc_re=copy(Πc_re),c_re_grid=copy(c_re_grid),
                    lnc_re_grid=copy(lnc_re_grid),Πg_re=copy(Πg_re),
                    g_re_grid=copy(g_re_grid),lng_re_grid=copy(lng_re_grid));
        return M;
    end
#

## Function that generates state transition markov chains (multivariate markov chains)
    function StateTransition_func_multivariate(p::Par,M::Model)
        @unpack p,ngrid,nstate,n_c,n_g=M;
        # Discretization of VAR(1) where matrix of impact coefficient is diagonal matrix of individual AR(1) coefficients.
        b = zeros(nstate);                                                  # constant
        ρ_state = [p.ρz,p.ρ_pm,p.ρ_pe,p.ρ_ψe,p.ρ_ψo];                       # persistence of state variables in order: z,pm,pe,ψe,ψo
        B = diagm(ρ_state);                                                 # Diagonalize persistence matrix
        Cov = Array(p.p_cov);                                               # Variance/covariance matrix of shocks
        MP_state = discrete_var(b, B, Cov, ngrid,2,Even(),sqrt(ngrid-1));   # Markov chain for all state variables
        Πs = MP_state.p;                                                    # Transition matrix
        lnSgrid = mapreduce(permutedims,vcat,MP_state.state_values);        # grid in logs
        Sgrid = exp.(lnSgrid);                                              # grid in levels

        # Price/productivity of gas non-persistent process (selected)
        MP_g = QuantEcon.rouwenhorst(n_g,0,sqrt(p.σ_pgψg),0)                ;  # Markov chain
        Π_g = MP_g.p                                                        ;  # Transition matrix
        lng_grid = MP_g.state_values                                        ;  # grid in logs
        g_grid = exp.(lng_grid)                                             ;  # grid in levels
        # Price/productivity of coal non-persistent residual process (selected)
        MP_c = QuantEcon.rouwenhorst(n_c,0,sqrt(p.σ_pcψc),0)                ;  # Markov chain
        Π_c = MP_c.p                                                        ;  # Transition matrix
        lnc_grid = MP_c.state_values                                        ;  # grid in logs
        c_grid = exp.(lnc_grid)                                             ;  # grid in levels

        # Update model 
        M = Model(M; Πs=copy(Πs),lnSgrid=copy(lnSgrid),Sgrid=copy(Sgrid),
                    Π_g=copy(Π_g),lng_grid=copy(lng_grid),g_grid=copy(g_grid),
                    Π_c=copy(Π_c),lnc_grid=copy(lnc_grid),c_grid=copy(c_grid));
        return M;
    end
#
## Some functions used to draw from discrete distributions
    # Create a CDF from probabilities
    function create_discrete_cdf(weight::Array{Float64,1})
        neighbor_weight = [0.0];
        neighbor_weight = vcat(neighbor_weight,weight);
        intervals = [[sum(neighbor_weight[1:i]),sum(neighbor_weight[1:i+1])] for i in 1:length(neighbor_weight)-1]
        res_intervals = [intervals[i][2] for i in 1:length(intervals)]
        return res_intervals
    end
#
# Draw from a distribution using inverse CDF method and known uniform draw
    function sample_discrete(randraw,intervals::Array{Float64,1})
        idx = findfirst(x-> randraw <= x,intervals)
        return idx
    end
#

## Functions that find the closest point on the grid to each observed state variable from the data
    function MatchGridData_pre(M,Data::DataFrame)
        @unpack p = M;
        NT = size(Data,1);
        ## Find closest point on the grid for each state variable (In order, state variables are: z,pm,pe,ψe,ψo,c,g)
        # Keeping track of grid indices for each firm
        ind_g = convert.(Int64, zeros(NT))  ;
        ind_c = convert.(Int64, zeros(NT))  ;
        ind_gre = convert.(Int64, zeros(NT)) ;
        ind_cre = convert.(Int64, zeros(NT)) ;
        ind_s = convert.(Int64, zeros(NT))  ;
        grid_indices = DataFrame(s = ind_s, c = ind_c, g = ind_g, g_re = ind_gre, c_re = ind_cre) ;
        dist_grid_smallest = zeros(NT);
        # Find smallest grid points by euclidean distance
        dist_grid_smallest = zeros(NT);
        t = Data.year.-2009;
        z = Data.lnz .- p.μz_t[t.+1];
        pm = Data.logPm - p.μpm_t[t.+1];
        pe = Data.lnpelec_tilde - p.μpe_t[t.+1];
        ψe = Data.lnfprod_e - p.μψe_t[t.+1];
        ψo = Data.lnfprod_o - p.μψo_t[t.+1];
        state = [z pm pe ψe ψo];
        # L-2 norm between observed state variables and each combination possible on the grid
        dist_grid = pairwise(euclidean, state', M.lnSgrid', dims=2);
        grid_indices.s .= getindex.(argmin(dist_grid,dims=2),2);
        for i = 1:NT
            dist_grid_smallest[i] = dist_grid[grid_indices.s[i]]
            # Shock to price/productivity of coal
            if Data.combineF[i] == 123 || Data.combineF[i] == 1234
                grid_indices.c[i] = searchsortednearest(M.lnc_grid,Data.res_lnpc_prodc[i]) ;
            end
            # Shock to price/productivity of gas
            if Data.combineF[i] == 124 || Data.combineF[i] == 1234
                grid_indices.g[i] = searchsortednearest(M.lng_grid,Data.res_lnpg_prodg[i]) ;
            end
            # Comparative advantage for coal
            if ismissing(Data.lnfprod_c_re[i]) == false
                grid_indices.c_re[i] = searchsortednearest(M.lnc_re_grid[:],Data.lnfprod_c_re[i]) ;
            end
            # Comparative advantage for gas
            if ismissing(Data.lnfprod_g_re[i]) == false
                grid_indices.g_re[i] = searchsortednearest(M.lng_re_grid[:],Data.lnfprod_g_re[i]) ;
            end
        end
        return grid_indices, dist_grid_smallest;
    end
    function MatchGridData(M,Data::DataFrame)
        @unpack p = M;
        NT = size(Data,1);
        ## Find closest point on the grid for each state variable (In order, state variables are: z,pm,pe,ψe,ψo,c,g)
        # Keeping track of grid indices for each firm
        ind_g = convert.(Int64, zeros(NT))  ;
        ind_c = convert.(Int64, zeros(NT))  ;
        ind_gre = convert.(Int64, zeros(NT)) ;
        ind_cre = convert.(Int64, zeros(NT)) ;
        ind_s = convert.(Int64, zeros(NT))  ;
        grid_indices = DataFrame(s = ind_s, c = ind_c, g = ind_g, g_re = ind_gre, c_re = ind_cre) ;
        dist_grid_smallest = zeros(NT);
        # Find smallest grid points by euclidean distance
        dist_grid_smallest = zeros(NT);
        t = Data.year.-2009;
        z = Data.lnz .- p.μz_t[t.+1];
        pm = Data.logPm - p.μpm_t[t.+1];
        pe = Data.lnpelec_tilde - p.μpe_t[t.+1];
        ψe = Data.lnfprod_e - p.μψe_t[t.+1];
        ψo = Data.lnfprod_o - p.μψo_t[t.+1];
        state = [z pm pe ψe ψo];
        # L-2 norm between observed state variables and each combination possible on the grid
        dist_grid = pairwise(euclidean, state', M.lnSgrid', dims=2);
        grid_indices.s .= getindex.(argmin(dist_grid,dims=2),2);
        for i = 1:NT
            dist_grid_smallest[i] = dist_grid[grid_indices.s[i]]
            # Shock to price/productivity of coal
            if Data.combineF[i] == 123 || Data.combineF[i] == 1234
                grid_indices.c[i] = searchsortednearest(M.lnc_grid,Data.res_lnpc_prodc[i]) ;
            end
            # Shock to price/productivity of gas
            if Data.combineF[i] == 124 || Data.combineF[i] == 1234
                grid_indices.g[i] = searchsortednearest(M.lng_grid,Data.res_lnpg_prodg[i]) ;
            end
            # Comparative advantage for coal
            if ismissing(Data.lnfprod_c_re[i]) == false
                grid_indices.c_re[i] = searchsortednearest(p.lnc_re_grid[:],Data.lnfprod_c_re[i]) ;
            end
            # Comparative advantage for gas
            if ismissing(Data.lnfprod_g_re[i]) == false
                grid_indices.g_re[i] = searchsortednearest(p.lng_re_grid[:],Data.lnfprod_g_re[i]) ;
            end
        end
        return grid_indices, dist_grid_smallest;
    end
    # For the forward simulation
    function MatchGridData_fs(M,Data::DataFrame)
        @unpack p = M;
        NT = size(Data,1);
        ## Find closest point on the grid for each state variable (In order, state variables are: z,pm,pe,ψe,ψo,c,g)
        # Keeping track of grid indices for each firm
        ind_g = convert.(Int64, zeros(NT))  ;
        ind_c = convert.(Int64, zeros(NT))  ;
        ind_gre = convert.(Int64, zeros(NT)) ;
        ind_cre = convert.(Int64, zeros(NT)) ;
        ind_s = convert.(Int64, zeros(NT))  ;
        grid_indices = DataFrame(s = ind_s, c = ind_c, g = ind_g, g_re = ind_gre, c_re = ind_cre) ;
        dist_grid_smallest = zeros(NT);
        # Find smallest grid points by euclidean distance
        t = Data.year.-2009;
        z = Data.lnz .- p.μz_t[t.+1];
        pm = Data.logPm - p.μpm_t[t.+1];
        pe = Data.lnpelec_tilde - p.μpe_t[t.+1];
        ψe = Data.lnfprod_e - p.μψe_t[t.+1];
        ψo = Data.lnfprod_o - p.μψo_t[t.+1];
        state = [z pm pe ψe ψo];
        # L-2 norm between observed state variables and each combination possible on the grid
        dist_grid = pairwise(euclidean, state', M.lnSgrid', dims=2);
        grid_indices.s .= getindex.(argmin(dist_grid,dims=2),2);
        for i = 1:NT
            dist_grid_smallest[i] = dist_grid[grid_indices.s[i]]
            # Shock to price/productivity of coal
            if Data.combineF[i] == 123 || Data.combineF[i] == 1234
                grid_indices.c[i] = searchsortednearest(M.lnc_grid,Data.res_lnpc_prodc[i]) ;
            end
            # Shock to price/productivity of gas
            if Data.combineF[i] == 124 || Data.combineF[i] == 1234
                grid_indices.g[i] = searchsortednearest(M.lng_grid,Data.res_lnpg_prodg[i]) ;
            end
        end
        return grid_indices, dist_grid_smallest;
    end
#

## Functions that uses moments of the prices and productivity for coal and gas to discretize distribution of price separately from productivity
    function discrete_objfunc(ψgrid,p::Par,M::Model,fuel::String)
        if fuel == "g"
            @unpack lng_grid = M;
            @unpack σ_pg,σ_ψg,cov_pgψg,σ_pgψg = p;
            pgrid = ψgrid.+lng_grid;
            # Moments
            σhat_pg = var(pgrid);
            σhat_ψg = var(ψgrid);
            covhat_pgψg = cov(pgrid,ψgrid);
            σhat_pgψg = var(pgrid.-ψgrid);
            Ehat_pg = mean(pgrid);
            Ehat_ψg = mean(ψgrid);
            # Objective function
            return ( (σhat_pg-σ_pg)^2 + (σhat_ψg-σ_ψg)^2 + (covhat_pgψg-cov_pgψg)^2 + Ehat_pg^2 + Ehat_ψg^2 + (σhat_pgψg-σ_pgψg)^2);
        elseif fuel == "c"
            @unpack lnc_grid = M;
            @unpack σ_pc,σ_ψc,cov_pcψc,σ_pcψc = p;
            pgrid = ψgrid.+lnc_grid;
            # Moments
            σhat_pc = var(pgrid);
            σhat_ψc = var(ψgrid);
            covhat_pcψc = cov(pgrid,ψgrid);
            σhat_pcψc = var(pgrid.-ψgrid);
            Ehat_pc = mean(pgrid);
            Ehat_ψc = mean(ψgrid);
            # Objective function
            return ( (σhat_pc-σ_pc)^2 + (σhat_ψc-σ_ψc)^2 + (covhat_pcψc-cov_pcψc)^2 + Ehat_pc^2 + Ehat_ψc^2 + (σhat_pcψc-σ_pcψc)^2);
        end
    end
    function DiscretizeGasCoal_distr(p::Par,M::Model)
        @unpack lng_grid,lnc_grid  = M;
        # Gas 
        nparam=4;
        opt=Opt(:LN_NEWUOA,nparam);
        function objfunc_temp_g(x::Vector,g::Vector)
            if length(g) > 0
                ForwardDiff.gradient!(g, discrete_objfunc(x,p,M,"g"), x);
            end
            return discrete_objfunc(x,p,M,"g");
        end
        opt.min_objective = objfunc_temp_g;
        opt.xtol_abs=1e-6;
        opt.ftol_rel=1e-6;
        x0 = [0,0,0,0];
        (objfunc_g,lnψg_grid,ret) = NLopt.optimize(opt, x0);

        # Coal 
        nparam=4;
        opt=Opt(:LN_NEWUOA,nparam);
        function objfunc_temp_c(x::Vector,g::Vector)
            if length(g) > 0
                ForwardDiff.gradient!(g, discrete_objfunc(x,p,M,"c"), x);
            end
            return discrete_objfunc(x,p,M,"c");
        end
        opt.min_objective = objfunc_temp_c;
        opt.xtol_abs=1e-6;
        opt.ftol_rel=1e-6;
        x0 = [0,0,0,0];
        (objfunc_c,lnψc_grid,ret) = NLopt.optimize(opt, x0);

        # Get grid for both productivity and prices
        lnpg_grid = collect(lng_grid) .+ lnψg_grid;
        lnpc_grid = collect(lnc_grid) .+ lnψc_grid;

        # Update model
        M = Model(M; lnpg_grid=copy(lnpg_grid),lnpc_grid=copy(lnpc_grid),pg_grid=copy(exp.(lnpg_grid)),pc_grid=copy(exp.(lnpc_grid)),
                    lnψg_grid=copy(lnψg_grid),lnψc_grid=copy(lnψc_grid),ψg_grid=copy(exp.(lnψg_grid)),ψc_grid=copy(exp.(lnψc_grid)));
        return M;
    end
#

##############################################################
#---- STATIC SIMULATION FUNCTIONS (PRICE AND PROFITS)   -----#
##############################################################


    #-----------------------------------------
    #--------- MONOPOLISTIC COMPETITION ------
    #-----------------------------------------

### Functions that find equilibrium aggregate output price
    # With state variables from the grid
        function OutputPrice_func(M::Model,Data::DataFrame,grid_indices::DataFrame,p::Par)
            # This function does fixed point iteration to find the aggregate output price index
            @unpack T,η,ρ,σ,pout_struc,N_t = p;
            N = size(Data,1);
            Data_year = groupby(Data,:year);
            pout_struc = copy(p.pout_struc);
            # Initial guess and initial distance
            pout_agg_init = copy(p.pout_init);
            dist = 100;
            #println("Current distance = $dist");
            pout_agg_old = copy(pout_agg_init);
            pout_agg_new = copy(pout_agg_init);
            while dist > p.pout_tol 
                for t = 1:T
                    # Get predicted individual output given state variables and current guess of aggregate price index
                    NT = size(Data_year[t],1);
                    pout = Array{Float64}(undef,NT);
                    rk = p.rk[t+1];
                    w = p.w[t+1];
                    for i = 1:NT
                        ii = Data_year[t].id[i];
                        is = grid_indices.s[ii];
                            lnz_grid = M.lnSgrid[is,1];
                            lnpm_grid = M.lnSgrid[is,2];  
                            lnpe_grid = M.lnSgrid[is,3];
                            lnψe_grid = M.lnSgrid[is,4];
                            lnψo_grid = M.lnSgrid[is,5];
                        ig = grid_indices.g[ii];
                        ig_re = grid_indices.g_re[ii];
                        ic = grid_indices.c[ii];
                        ic_re = grid_indices.c_re[ii];
                        z = exp(p.μz_t[t+1] + lnz_grid);
                        pm = exp(p.μpm_t[t+1] + lnpm_grid);
                        pe = exp(p.μpe_t[t+1] + lnpe_grid);
                        ψe = exp(p.μψe_t[t+1] + lnψe_grid);
                        ψo = exp(p.μψo_t[t+1] + lnψo_grid);
                        peψe = pe/ψe;
                        poψo = (p.po[t+1])/ψo;
                        if Data.combineF[ii] == 12
                            pfψf = [poψo,peψe];
                            pE =  pE_func(12,pfψf,p);
                        elseif Data.combineF[ii] == 123
                            pc = exp(p.μpc_t[t+1] + M.lnpc_grid[ic]);
                            ψc = exp(p.μψc_t[t+1] + M.lnψc_grid[ic] + M.lnc_re_grid[ic_re]);
                            pcψc = pc/ψc;
                            pfψf = [poψo,peψe,pcψc];
                            pE = pE_func(123,pfψf,p);
                        elseif Data.combineF[ii] == 124
                            pg = exp(p.μpg_t[t+1] + M.lnpg_grid[ig]);
                            ψg = exp(p.μψg_t[t+1] + M.lnψg_grid[ig] + M.lng_re_grid[ig_re]);
                            pgψg = pg/ψg;
                            pfψf = [poψo,peψe,pgψg];
                            pE = pE_func(124,pfψf,p);
                        elseif Data.combineF[ii] == 1234
                            pc = exp(p.μpc_t[t+1] + M.lnpc_grid[ic]);
                            ψc = exp(p.μψc_t[t+1] + M.lnψc_grid[ic] + M.lnc_re_grid[ic_re]);
                            pcψc = pc/ψc;
                            pg = exp(p.μpg_t[t+1] + M.lnpg_grid[ig]);
                            ψg = exp(p.μψg_t[t+1] + M.lnψg_grid[ig] + M.lng_re_grid[ig_re]);
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
                # Check for convergence
                #println("Current distance = $(sqrt(sum((pout_agg_new-pout_agg_old).^2)))");
                dist = sqrt(sum((pout_agg_new-pout_agg_old).^2));
                # Update for new iteration
                pout_agg_old = copy(pout_agg_new);
            end
            # Update output price index to parameter struc
            p = Par(p; pout_struc = copy(pout_agg_new));
            M = Model(M; p = p);
            return p,M;
        end
    #
    # With state variables from the data 
        function OutputPrice_func_nogrid(M::Model,Data::DataFrame,p::Par,τ)
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
                        lnψe = Data.lnfprod_e[ii]; ψe = exp(lnψe);
                        lnψo = Data.lnfprod_o[ii]; ψo = exp(lnψo);
                        peψe = pe/ψe;
                        poψo = (p.po[t+1]+τo)/ψo;
                        if Data.combineF[ii] == 12
                            pfψf = [poψo,peψe];
                            pE =  pE_func(12,pfψf,p);
                        elseif Data.combineF[ii] == 123
                            pc = exp(Data.lnpc_tilde[ii]) + τc;
                            ψc = exp(Data.lnfprod_c[ii]);
                            pcψc = pc/ψc;
                            pfψf = [poψo,peψe,pcψc];
                            pE = pE_func(123,pfψf,p);
                        elseif Data.combineF[ii] == 124
                            pg = exp(Data.lnpg_tilde[ii]) + τg;
                            ψg = exp(Data.lnfprod_g[ii]);
                            pgψg = pg/ψg;
                            pfψf = [poψo,peψe,pgψg];
                            pE = pE_func(124,pfψf,p);
                        elseif Data.combineF[ii] == 1234
                            pc = exp(Data.lnpc_tilde[ii]) + τc;
                            ψc = exp(Data.lnfprod_c[ii]);
                            pcψc = pc/ψc;
                            pg = exp(Data.lnpg_tilde[ii]) + τg;
                            ψg = exp(Data.lnfprod_g[ii]);
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
            p = Par(p; pout_struc = copy(pout_agg_new));
            M = Model(M; p = p);
            return p,M;
        end
    #
#

### Functions that compute per-period profits
    # Pre-estimation: Compute Static Profits Next Period under all combination of state variables
        function StaticProfit_grid_pre(M::Model)
            @unpack p,n_c,n_g,ngrid,nstate,nc_re,ng_re = M;
            @unpack rk,w,po,pout_struc,Ygmean = p;
            # Initialize static profit under grid points (order of states: z,pm,pe,ψe,ψo)
            πgrid_oe = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                  ;
            πgrid_oge = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                 ;
            πgrid_oce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                 ;
            πgrid_ogce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                ;
            for k = 1:(p.T*(ngrid^nstate)*n_c*n_g*nc_re*ng_re)
                t,is,ic,ig,ic_re,ig_re = Tuple(CartesianIndices((p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re))[k]);
                peψe = (exp(M.lnSgrid[is,3]+p.μpe_t[t+1]))/exp(M.lnSgrid[is,4]+p.μψe_t[t+1]);
                poψo = (p.po[t+1])/exp(M.lnSgrid[is,5]+p.μψo_t[t+1]);
                pcψc = exp(p.μpcψc_t[t+1] + M.lnc_grid[ic] - M.lnc_re_grid[ic_re])  ;
                pgψg = exp(p.μpgψg_t[t+1] + M.lng_grid[ig] - M.lng_re_grid[ig_re])  ;
                pm = exp(M.lnSgrid[is,2]+p.μpm_t[t+1]); 
                z = exp(M.lnSgrid[is,1]+p.μz_t[t+1]);
                pfψf_oe = [poψo,peψe]                                                                   ;
                pfψf_oce = [poψo,peψe,pcψc]                                                             ;
                pfψf_oge = [poψo,peψe,pgψg]                                                             ;
                pfψf_ogce = [poψo,peψe,pcψc,pgψg]                                                       ;
                pE_oe = pE_func(12,pfψf_oe,p);
                pE_oce = pE_func(123,pfψf_oce,p);
                pE_oge = pE_func(124,pfψf_oge,p);
                pE_ogce = pE_func(1234,pfψf_ogce,p);
                pin_oe = pinput_func(rk[t+1],pm,w[t+1],pE_oe,p);
                pin_oce = pinput_func(rk[t+1],pm,w[t+1],pE_oce,p);
                pin_oge = pinput_func(rk[t+1],pm,w[t+1],pE_oge,p);
                pin_ogce = pinput_func(rk[t+1],pm,w[t+1],pE_ogce,p);
                πgrid_oe[t,is,ic,ig,ic_re,ig_re] = profit_func_monopolistic(z,pin_oe,t,p);
                πgrid_oge[t,is,ic,ig,ic_re,ig_re] = profit_func_monopolistic(z,pin_oge,t,p);
                πgrid_oce[t,is,ic,ig,ic_re,ig_re] = profit_func_monopolistic(z,pin_oce,t,p);
                πgrid_ogce[t,is,ic,ig,ic_re,ig_re] = profit_func_monopolistic(z,pin_ogce,t,p);
            end
            # Update model
            M=Model(M; πgrid_oe = copy(convert(Array,πgrid_oe)), πgrid_oge = copy(convert(Array,πgrid_oge)), πgrid_oce = copy(convert(Array,πgrid_oce)), πgrid_ogce = copy(convert(Array,πgrid_ogce)));
            return M;
        end
    #
    # Post-estimation  Compute Static Profits Next Period under all combination of state variables
        function StaticProfit_grid(M::Model,τ)
            @unpack p,n_c,n_g,ngrid,nstate,nc_re,ng_re = M;
            @unpack rk,w,po,pout_struc,Ygmean = p;
            τg = τ[1]*p.ggmean;
            τc = τ[2]*p.cgmean;
            τo = τ[3]*p.ogmean;
            τe = τ[4]*p.egmean;
            # Initialize static profit under grid points (order of states: z,pm,pe,ψe,ψo)
            πgrid_oe = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                  ;
            πgrid_oge = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                 ;
            πgrid_oce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                 ;
            πgrid_ogce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                ;
            for k = 1:(p.T*(ngrid^nstate)*n_c*n_g*nc_re*ng_re)
                t,is,ic,ig,ic_re,ig_re = Tuple(CartesianIndices((p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re))[k]);
                peψe = (exp(M.lnSgrid[is,3]+p.μpe_t[t+1])+τe)/exp(M.lnSgrid[is,4]+p.μψe_t[t+1]);
                poψo = (p.po[t+1]+τo)/exp(M.lnSgrid[is,5]+p.μψo_t[t+1]);
                # pcψc = exp(p.μpcψc_t[t+1] + M.lnc_grid[ic] + log(1+(τc/(exp(p.μpc_t[t+1])*M.pc_grid[ic]))))/p.c_re_grid[ic_re]  ;
                # pgψg = exp(p.μpgψg_t[t+1] + M.lng_grid[ig] + log(1+(τg/(exp(p.μpg_t[t+1])*M.pg_grid[ic]))))/p.g_re_grid[ig_re]  ;
                # pcψc = exp(p.μpcψc_t[t+1] + M.lnc_grid[ic])/p.c_re_grid[ic_re]  ;
                # pgψg = exp(p.μpgψg_t[t+1] + M.lng_grid[ig])/p.g_re_grid[ig_re]  ;
                pcψc = (exp(p.μpcψc_t[t+1] + M.lnc_grid[ic])/p.c_re_grid[ic_re]) + (τc/(exp(M.lnψc_grid[ic]+p.lnc_re_grid[ic_re])))   ;
                pgψg = (exp(p.μpgψg_t[t+1] + M.lng_grid[ig])/p.g_re_grid[ig_re]) + (τg/(exp(M.lnψg_grid[ic]+p.lng_re_grid[ig_re])))   ;
                pm = exp(M.lnSgrid[is,2]+p.μpm_t[t+1]); 
                z = exp(M.lnSgrid[is,1]+p.μz_t[t+1]);
                pfψf_oe = [poψo,peψe]                                                                   ;
                pfψf_oce = [poψo,peψe,pcψc]                                                             ;
                pfψf_oge = [poψo,peψe,pgψg]                                                             ;
                pfψf_ogce = [poψo,peψe,pcψc,pgψg]                                                       ;
                pE_oe = pE_func(12,pfψf_oe,p);
                pE_oce = pE_func(123,pfψf_oce,p);
                pE_oge = pE_func(124,pfψf_oge,p);
                pE_ogce = pE_func(1234,pfψf_ogce,p);
                pin_oe = pinput_func(rk[t+1],pm,w[t+1],pE_oe,p);
                pin_oce = pinput_func(rk[t+1],pm,w[t+1],pE_oce,p);
                pin_oge = pinput_func(rk[t+1],pm,w[t+1],pE_oge,p);
                pin_ogce = pinput_func(rk[t+1],pm,w[t+1],pE_ogce,p);
                πgrid_oe[t,is,ic,ig,ic_re,ig_re] = profit_func_monopolistic(z,pin_oe,t,p);
                πgrid_oge[t,is,ic,ig,ic_re,ig_re] = profit_func_monopolistic(z,pin_oge,t,p);
                πgrid_oce[t,is,ic,ig,ic_re,ig_re] = profit_func_monopolistic(z,pin_oce,t,p);
                πgrid_ogce[t,is,ic,ig,ic_re,ig_re] = profit_func_monopolistic(z,pin_ogce,t,p);
            end
            # Update model
            M=Model(M; πgrid_oe = copy(convert(Array,πgrid_oe)), πgrid_oge = copy(convert(Array,πgrid_oge)), πgrid_oce = copy(convert(Array,πgrid_oce)), πgrid_ogce = copy(convert(Array,πgrid_ogce)));
            return M;
        end
    #
    # Alternative profit function (sohuld be the same as StaticProfit_grid)
        function StaticProfit_grid_alt(M::Model,τ)
            @unpack p,n_c,n_g,ngrid,nstate,nc_re,ng_re = M;
            @unpack rk,w,po,pout_struc,Ygmean,η,ρ,θ,d_t,N_t = p;
            # Initialize static profit under grid points (order of states: z,pm,pe,ψe,ψo)
            πgrid_oe = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                  ;
            πgrid_oge = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                 ;
            πgrid_oce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                 ;
            πgrid_ogce = Array{Float64}(undef,p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re)                ;
            τg = τ[1]*p.ggmean;
            τc = τ[2]*p.cgmean;
            τo = τ[3]*p.ogmean;
            τe = τ[4]*p.egmean;
            for k = 1:(p.T*(ngrid^nstate)*n_c*n_g*nc_re*ng_re)
                t,is,ic,ig,ic_re,ig_re = Tuple(CartesianIndices((p.T,ngrid^nstate,n_c,n_g,nc_re,ng_re))[k]);
                peψe = exp((M.lnSgrid[is,3]+p.μpe_t[t+1])+τe)/exp(M.lnSgrid[is,4]+p.μψe_t[t+1]);
                poψo = (p.po[t+1]+τo)/exp(M.lnSgrid[is,5]+p.μψo_t[t+1]);
                pcψc = exp(p.μpcψc_t[t+1] + M.lnc_grid[ic] + log(1+(τc/(exp(p.μpc_t[t+1])*M.pc_grid[ic]))))/p.c_re_grid[ic_re]  ;
                pgψg = exp(p.μpgψg_t[t+1] + M.lng_grid[ig] + log(1+(τg/(exp(p.μpg_t[t+1])*M.pg_grid[ic]))))/p.g_re_grid[ig_re]  ;
                pm = exp(M.lnSgrid[is,2]+p.μpm_t[t+1]);
                z = exp(M.lnSgrid[is,1]+p.μz_t[t+1]);
                pfψf_oe = [poψo,peψe]                                                                   ;
                pfψf_oce = [poψo,peψe,pcψc]                                                             ;
                pfψf_oge = [poψo,peψe,pgψg]                                                             ;
                pfψf_ogce = [poψo,peψe,pcψc,pgψg]                                                       ;
                pE_oe = pE_func(12,pfψf_oe,p);
                pE_oce = pE_func(123,pfψf_oce,p);
                pE_oge = pE_func(124,pfψf_oge,p);
                pE_ogce = pE_func(1234,pfψf_ogce,p);
                pin_oe = pinput_func(rk[t+1],pm,w[t+1],pE_oe,p);
                pin_oce = pinput_func(rk[t+1],pm,w[t+1],pE_oce,p);
                pin_oge = pinput_func(rk[t+1],pm,w[t+1],pE_oge,p);
                pin_ogce = pinput_func(rk[t+1],pm,w[t+1],pE_ogce,p);
                pout_agg = p.pout_struc[t+1];
                pstruc1 = (1+ρ*(θ-1))/((θ-1)*((1-η)*ρ + η));
                pstruc2 = 1/((1-η)*ρ+η);
                pstruc3 = (ρ-1)/((1-η)*ρ+η)
                pstruc4 = (η*(1-ρ))/((1-η)*ρ+η);
                pstruc5 = (η*(ρ-1))/((1-η)*ρ+η);
                pstruc6 = ρ/((1-η)*ρ+η);
                πgrid_oe[t,is,ic,ig,ic_re,ig_re] = (pout_agg^pstruc1)*((exp(d_t[t+1])/N_t[t+1])^pstruc2)*((Ygmean[t+1]*z)^pstruc3)*(pin_oe^pstruc4)*((((ρ-1)/ρ)*η)^pstruc5 - (((ρ-1)/ρ)*η)^pstruc6);
                πgrid_oge[t,is,ic,ig,ic_re,ig_re] = (pout_agg^pstruc1)*((exp(d_t[t+1])/N_t[t+1])^pstruc2)*((Ygmean[t+1]*z)^pstruc3)*(pin_oge^pstruc4)*((((ρ-1)/ρ)*η)^pstruc5 - (((ρ-1)/ρ)*η)^pstruc6);
                πgrid_oce[t,is,ic,ig,ic_re,ig_re] = (pout_agg^pstruc1)*((exp(d_t[t+1])/N_t[t+1])^pstruc2)*((Ygmean[t+1]*z)^pstruc3)*(pin_oce^pstruc4)*((((ρ-1)/ρ)*η)^pstruc5 - (((ρ-1)/ρ)*η)^pstruc6);
                πgrid_ogce[t,is,ic,ig,ic_re,ig_re] = (pout_agg^pstruc1)*((exp(d_t[t+1])/N_t[t+1])^pstruc2)*((Ygmean[t+1]*z)^pstruc3)*(pin_ogce^pstruc4)*((((ρ-1)/ρ)*η)^pstruc5 - (((ρ-1)/ρ)*η)^pstruc6);
            end
            # Update model
            M=Model(M; πgrid_oe = copy(convert(Array,πgrid_oe)), πgrid_oge = copy(convert(Array,πgrid_oge)), πgrid_oce = copy(convert(Array,πgrid_oce)), πgrid_ogce = copy(convert(Array,πgrid_ogce)));
            return M;
        end
    #
    # Compute predicted profit under state variables in the data 
        function StaticProfit_data(M::Model,Data::DataFrame,τ)
            @unpack p = M;
            τg = τ[1]*p.ggmean;
            τc = τ[2]*p.cgmean;
            τo = τ[3]*p.ogmean;
            τe = τ[4]*p.egmean;
            # Initialize static profit under grid points (order of states: z,pm,pe,ψe,ψo)
            N = size(Data,1);
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
                    lnψe = Data.lnfprod_e[ii]; ψe = exp(lnψe);
                    lnψo = Data.lnfprod_o[ii]; ψo = exp(lnψo);
                    peψe = pe/ψe;
                    poψo = (p.po[t+1]+τo)/ψo;
                    if Data.combineF[ii] == 12
                        pfψf = [poψo,peψe];
                        pE =  pE_func(12,pfψf,p);
                    elseif Data.combineF[ii] == 123
                        pc = exp(Data.lnpc_tilde[ii])+τc;
                        ψc = exp(Data.lnfprod_c[ii]);
                        pcψc = pc/ψc;
                        pfψf = [poψo,peψe,pcψc];
                        pE = pE_func(123,pfψf,p);
                    elseif Data.combineF[ii] == 124
                        pg = exp(Data.lnpg_tilde[ii])+τg;
                        ψg = exp(Data.lnfprod_g[ii]);
                        pgψg = pg/ψg;
                        pfψf = [poψo,peψe,pgψg];
                        pE = pE_func(124,pfψf,p);
                    elseif Data.combineF[ii] == 1234
                        pc = exp(Data.lnpc_tilde[ii])+τc;
                        ψc = exp(Data.lnfprod_c[ii]);
                        pcψc = pc/ψc;
                        pg = exp(Data.lnpg_tilde[ii])+τg;
                        ψg = exp(Data.lnfprod_g[ii]);
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
    #
#

##############################################################
#----     FUNCTIONS FOR FORWARD SIMULATION              -----#
##############################################################

## Function that draws relevant state variables for all forward simulations
    # Baseline: without bounds on state variables - with switching
        function ForwardSimul_draws(p::Par,M::Model,Data::DataFrame,seed)
            Random.seed!(seed);
            @unpack N,gre_draw,cre_draw,π_cond,ng_re,nc_re = M;
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
            FcombineF_rand = GeneralizedExtremeValue(-Base.MathConstants.eulergamma,1,0)        ;  # Fixed type 1 extreme value draw for each fuel set
                FcombineF_draw = rand(FcombineF_rand,p.F_tot,N,p.S,p.Tf+1)  ;
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
                    lnψe = p.μψe_t[t+1] + p.ρ_ψe*Data.res_prode[i] + state_resdraw[8][i,s,1]; ψe[i,s,1] = exp(lnψe);
                    lnψo = p.μψo_t[t+1] + p.ρ_ψo*Data.res_prodo[i] + state_resdraw[9][i,s,1]; ψo[i,s,1] = exp(lnψo);
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
                        res_ψe = log(ψe[i,s,tf-1]) - p.μψe_t[t+1];
                            lnψe = p.μψe_t[t+1] + p.ρ_ψe*res_ψe + state_resdraw[8][i,s,tf]; ψe[i,s,tf] = exp(lnψe);
                        res_ψo = log(ψo[i,s,tf-1]) - p.μψo_t[t+1];
                            lnψo = p.μψo_t[t+1] + p.ρ_ψo*res_ψo + state_resdraw[9][i,s,tf]; ψo[i,s,tf] = exp(lnψo);
                        lnpg = p.μpg_t[t+1] + state_resdraw[2][i,s,tf]; pg[i,s,tf] = exp(lnpg);
                        lnψg[i,s,tf] = p.μψg_t[t+1] + state_resdraw[1][i,s,tf];
                        lnpc = p.μpc_t[t+1] + state_resdraw[4][i,s,tf]; pc[i,s,tf] = exp(lnpc);
                        lnψc[i,s,tf] = p.μψc_t[t+1] + state_resdraw[3][i,s,tf];
                    end
                end
            end
            gre_draw = rand(N,p.S)                                  ;  # Fixed uniform draw for gas comparative advantage
            cre_draw = rand(N,p.S)                                  ;  # Fixed uniform draw for coal comparative advantage
            #Draw from comparative advantage (conditional distribution)
            gre_fs = Array{Int64}(undef,N,S);
            cre_fs = Array{Int64}(undef,N,S);
            Nfirms = size(unique(Data.IDnum))[1];
            Data_firm = groupby(Data,:IDnum);
            for i = 1:Nfirms
                # Get pdf and cdf of comparative advantage for firm i
                pdf_gre = sum(M.π_cond[i,:,:][:,j] for j = 1:ng_re);
                pdf_cre = sum(M.π_cond[i,:,:][j,:] for j = 1:nc_re);
                pdf_gre = [sum(M.π_cond[i,:,:][:,1]),sum(M.π_cond[i,:,:][:,2]),sum(M.π_cond[i,:,:][:,3])];
                pdf_cre = [sum(M.π_cond[i,:,:][1,:]),sum(M.π_cond[i,:,:][2,:]),sum(M.π_cond[i,:,:][3,:])];
                cdf_gre = create_discrete_cdf(pdf_gre);
                cdf_cre = create_discrete_cdf(pdf_cre);
                # Draw comparative advantage
                id = Data_firm[i].id;
                for s = 1:S
                    g_fs0 = sample_discrete(gre_draw[i,s],cdf_gre);
                    c_fs0 = sample_discrete(cre_draw[i,s],cdf_cre);
                    for j = 1:size(id)[1]
                        gre_fs[id[j],s] = g_fs0;
                        cre_fs[id[j],s] = c_fs0;
                    end
                end 
            end
            # Update model
            M = Model(M; z_fs=copy(z),pm_fs=copy(pm),pe_fs=copy(pe),ψe_fs=copy(ψe),ψo_fs=copy(ψo),pg_fs=copy(pg),pc_fs=copy(pc),gre_fs=copy(gre_fs),cre_fs=copy(cre_fs),
                    lnψg_fs=copy(lnψg),lnψc_fs=copy(lnψc),gre_draw=copy(gre_draw),cre_draw=copy(cre_draw),FcombineF_draw=copy(FcombineF_draw));
            return M,state_resdraw1;
        end
    #
    # Baseline: without bounds on state variables - no switching
        function ForwardSimul_draws_noswitch(p::Par,M::Model,Data::DataFrame,seed)
            Random.seed!(seed);
            @unpack N,gre_draw,cre_draw,π_cond,ng_re,nc_re = M;
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
                    lnψe = p.μψe_t[t+1] + p.ρ_ψe*Data.res_prode[i] + state_resdraw[8][i,s,1]; ψe[i,s,1] = exp(lnψe);
                    lnψo = p.μψo_t[t+1] + p.ρ_ψo*Data.res_prodo[i] + state_resdraw[9][i,s,1]; ψo[i,s,1] = exp(lnψo);
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
                        res_ψe = log(ψe[i,s,tf-1]) - p.μψe_t[t+1];
                            lnψe = p.μψe_t[t+1] + p.ρ_ψe*res_ψe + state_resdraw[8][i,s,tf]; ψe[i,s,tf] = exp(lnψe);
                        res_ψo = log(ψo[i,s,tf-1]) - p.μψo_t[t+1];
                            lnψo = p.μψo_t[t+1] + p.ρ_ψo*res_ψo + state_resdraw[9][i,s,tf]; ψo[i,s,tf] = exp(lnψo);
                        lnpg = p.μpg_t[t+1] + state_resdraw[2][i,s,tf]; pg[i,s,tf] = exp(lnpg);
                        lnψg[i,s,tf] = p.μψg_t[t+1] + state_resdraw[1][i,s,tf];
                        lnpc = p.μpc_t[t+1] + state_resdraw[4][i,s,tf]; pc[i,s,tf] = exp(lnpc);
                        lnψc[i,s,tf] = p.μψc_t[t+1] + state_resdraw[3][i,s,tf];
                    end
                end
            end
            # Update model
            M = Model(M; z_fs=copy(z),pm_fs=copy(pm),pe_fs=copy(pe),ψe_fs=copy(ψe),ψo_fs=copy(ψo),pg_fs=copy(pg),pc_fs=copy(pc),
                    lnψg_fs=copy(lnψg),lnψc_fs=copy(lnψc));
            return M,state_resdraw1;
        end
    #
#

##############################################################
#----     FUNCTIONS FOR DYNAMIC DISCRETE CHOICE (OLD)       ------#
##############################################################

# Bellman operator (no random effects)
    function T_EVF_nore(M::Model,Κ,t)
        # This function iterates over the expected value function
        @unpack p,Wind,n_z,n_pm,n_ψe,n_pe,n_ψo,n_c,n_g = M; 
        vec_Wold_oe = vec(Wind[1,t,:,:,:,:,:,:,:])     ;
        vec_Wold_oge = vec(Wind[2,t,:,:,:,:,:,:,:])    ;
        vec_Wold_oce = vec(Wind[3,t,:,:,:,:,:,:,:])    ;
        vec_Wold_ogce = vec(Wind[4,t,:,:,:,:,:,:,:])   ;
        W_new = SharedArray{Float64}(p.F_tot,n_z,n_pm,n_ψe,n_pe,n_ψo,n_c,n_g);
        # Fixed cost parameters
        κ_g = Κ[1] ;
        κ_c = Κ[2] ;
        γ_g = Κ[3] ;
        γ_c = Κ[4] ;
        # Update value function
        @sync @distributed for k = 1:(n_z*n_pm*n_ψe*n_pe*n_ψo*n_c*n_g)
            iz,ipm,iψe,ipe,iψo,ic,ig = Tuple(CartesianIndices((n_z,n_pm,n_ψe,n_pe,n_ψo,n_c,n_g))[k]);
            # Compute transition probability (same for all values of current state variables)
            aux_1 = kronecker(M.Π_g[ig,:],M.Π_c[ic,:],M.Π_ψo[iψo,:],M.Π_pe[ipe,:],M.Π_ψe[iψe,:],M.Π_pm[ipm,:],M.Π_z[iz,:]);
            # Compute Emax
            aux_11 = p.β*sum(vec_Wold_ogce.*aux_1) ;
            aux_2_12 = aux_11 + log(exp(p.β*sum(vec_Wold_oe.*aux_1)-aux_11) +  exp(- κ_g + p.β*sum(vec_Wold_oge.*aux_1)-aux_11) + exp(- κ_c + p.β*sum(vec_Wold_oce.*aux_1)-aux_11) + exp(-κ_g - κ_c + p.β*sum(vec_Wold_ogce.*aux_1)-aux_11)) ;
            aux_2_124 = aux_11 + log(exp(γ_g+p.β*sum(vec_Wold_oe.*aux_1)-aux_11) +  exp(p.β*sum(vec_Wold_oge.*aux_1)-aux_11) + exp(γ_g - κ_c + p.β*sum(vec_Wold_oce.*aux_1)-aux_11) + exp(-κ_c + p.β*sum(vec_Wold_ogce.*aux_1)-aux_11)) ;
            aux_2_123 = aux_11 + log(exp(γ_c+p.β*sum(vec_Wold_oe.*aux_1)-aux_11) +  exp(γ_c - κ_g + p.β*sum(vec_Wold_oge.*aux_1)-aux_11) + exp(p.β*sum(vec_Wold_oce.*aux_1)-aux_11) + exp(-κ_g + p.β*sum(vec_Wold_ogce.*aux_1)-aux_11)) ;
            aux_2_1234 = aux_11 + log(exp(γ_c + γ_g + p.β*sum(vec_Wold_oe.*aux_1)-aux_11) +  exp(γ_c + p.β*sum(vec_Wold_oge.*aux_1)-aux_11) + exp(γ_g + p.β*sum(vec_Wold_oce.*aux_1)-aux_11) + exp(p.β*sum(vec_Wold_ogce.*aux_1)-aux_11)) ; 
            W_new[1,iz,ipm,iψe,ipe,iψo,ic,ig] = M.πgrid_oe[t,iz,ipm,iψe,ipe,iψo,ic,ig] + aux_2_12;#0.5772+M.πgrid_oe[t,iz,ipm,iψe,ipe,iψo,ic,ig] + aux_2_12;
            W_new[2,iz,ipm,iψe,ipe,iψo,ic,ig] = M.πgrid_oge[t,iz,ipm,iψe,ipe,iψo,ic,ig] + aux_2_124;#0.5772+M.πgrid_oge[t,iz,ipm,iψe,ipe,iψo,ic,ig] + aux_2_124;
            W_new[3,iz,ipm,iψe,ipe,iψo,ic,ig] = M.πgrid_oce[t,iz,ipm,iψe,ipe,iψo,ic,ig] + aux_2_123;#0.5772+M.πgrid_oce[t,iz,ipm,iψe,ipe,iψo,ic,ig] + aux_2_123;
            W_new[4,iz,ipm,iψe,ipe,iψo,ic,ig] = M.πgrid_ogce[t,iz,ipm,iψe,ipe,iψo,ic,ig] + aux_2_1234;#0.5772+M.πgrid_ogce[t,iz,ipm,iψe,ipe,iψo,ic,ig] + aux_2_1234;
        end
        W_new = convert(Array,W_new);
        return W_new
    end
#

# Bellman operator using multivariate markov chain (no random effects)
    function T_EVF_nore_mvariate(M::Model,Κ,Π_transition,t)
        # This function iterates over the expected value function
        @unpack p,Wind,nstate,ngrid,n_c,n_g = M; 
        vec_Wold_oe = vec(Wind[1,t,:,:,:])     ;
        vec_Wold_oge = vec(Wind[2,t,:,:,:])    ;
        vec_Wold_oce = vec(Wind[3,t,:,:,:])    ;
        vec_Wold_ogce = vec(Wind[4,t,:,:,:])   ;
        W_new = Array{Float64}(undef,p.F_tot,ngrid^nstate,n_c,n_g);
        # Fixed cost parameters
        κ_g = Κ[1] ;
        κ_c = Κ[2] ;
        γ_g = Κ[3] ;
        γ_c = Κ[4] ;
        # Update value function
        for k = 1:((ngrid^nstate)*n_c*n_g)
            is,ic,ig = Tuple(CartesianIndices((ngrid^nstate,n_c,n_g))[k]);
            # Compute transition probability (same for all values of current state variables)
            aux_1 = Π_transition[is,:];
            # Compute Emax
            aux_11 = p.β*sum(vec_Wold_ogce.*aux_1) ;
            aux_2_12 = aux_11 + log(exp(p.β*sum(vec_Wold_oe.*aux_1)-aux_11) +  exp(- κ_g + p.β*sum(vec_Wold_oge.*aux_1)-aux_11) + exp(- κ_c + p.β*sum(vec_Wold_oce.*aux_1)-aux_11) + exp(-κ_g - κ_c + p.β*sum(vec_Wold_ogce.*aux_1)-aux_11)) ;
            aux_2_124 = aux_11 + log(exp(γ_g+p.β*sum(vec_Wold_oe.*aux_1)-aux_11) +  exp(p.β*sum(vec_Wold_oge.*aux_1)-aux_11) + exp(γ_g - κ_c + p.β*sum(vec_Wold_oce.*aux_1)-aux_11) + exp(-κ_c + p.β*sum(vec_Wold_ogce.*aux_1)-aux_11)) ;
            aux_2_123 = aux_11 + log(exp(γ_c+p.β*sum(vec_Wold_oe.*aux_1)-aux_11) +  exp(γ_c - κ_g + p.β*sum(vec_Wold_oge.*aux_1)-aux_11) + exp(p.β*sum(vec_Wold_oce.*aux_1)-aux_11) + exp(-κ_g + p.β*sum(vec_Wold_ogce.*aux_1)-aux_11)) ;
            aux_2_1234 = aux_11 + log(exp(γ_c + γ_g + p.β*sum(vec_Wold_oe.*aux_1)-aux_11) +  exp(γ_c + p.β*sum(vec_Wold_oge.*aux_1)-aux_11) + exp(γ_g + p.β*sum(vec_Wold_oce.*aux_1)-aux_11) + exp(p.β*sum(vec_Wold_ogce.*aux_1)-aux_11)) ; 
            W_new[1,is,ic,ig] = M.πgrid_oe[t,is,ic,ig] + aux_2_12;#0.5772+M.πgrid_oe[t,iz,ipm,iψe,ipe,iψo,ic,ig] + aux_2_12;
            W_new[2,is,ic,ig] = M.πgrid_oge[t,is,ic,ig] + aux_2_124;#0.5772+M.πgrid_oge[t,iz,ipm,iψe,ipe,iψo,ic,ig] + aux_2_124;
            W_new[3,is,ic,ig] = M.πgrid_oce[t,is,ic,ig] + aux_2_123;#0.5772+M.πgrid_oce[t,iz,ipm,iψe,ipe,iψo,ic,ig] + aux_2_123;
            W_new[4,is,ic,ig] = M.πgrid_ogce[t,is,ic,ig] + aux_2_1234;#0.5772+M.πgrid_ogce[t,iz,ipm,iψe,ipe,iψo,ic,ig] + aux_2_1234;
        end
        W_new = convert(Array,W_new);
        return W_new
    end
#

# Bellman operator (with random effects)
    function T_EVF_re(M::Model,Κ,t,ic_re,ig_re)
        # This function iterates over the expected value function
        @unpack p,Wind,n_z,n_pm,n_ψe,n_pe,n_ψo,n_c,n_g = M 
        vec_Wold_oe = vec(Wind[1,t,:,:,:,:,:,:,:,ic_re,ig_re])     ;
        vec_Wold_oge = vec(Wind[2,t,:,:,:,:,:,:,:,ic_re,ig_re])    ;
        vec_Wold_oce = vec(Wind[3,t,:,:,:,:,:,:,:,ic_re,ig_re])    ;
        vec_Wold_ogce = vec(Wind[4,t,:,:,:,:,:,:,:,ic_re,ig_re])   ;
        W_new = SharedArray{Float64}(p.F_tot,n_z,n_pm,n_ψe,n_pe,n_ψo,n_c,n_g);
        # Fixed cost parameters
        κ_g = Κ[1] ;
        κ_c = Κ[2] ;
        γ_g = Κ[3] ;
        γ_c = Κ[4] ;
        # Update value function
        @sync @distributed for k = 1:(n_z*n_pm*n_ψe*n_pe*n_ψo*n_c*n_g)
            iz,ipm,iψe,ipe,iψo,ic,ig = Tuple(CartesianIndices((n_z,n_pm,n_ψe,n_pe,n_ψo,n_c,n_g))[k]);
            # Compute transition probability (same for all values of current state variables)
            aux_1 = kronecker(M.Π_g[ig,:],M.Π_c[ic,:],M.Π_ψo[iψo,:],M.Π_pe[ipe,:],M.Π_ψe[iψe,:],M.Π_pm[ipm,:],M.Π_z[iz,:]);
            # Compute Emax
            aux_11 = p.β*sum(vec_Wold_ogce.*aux_1) ;
            aux_2_12 = aux_11 + log(exp(p.β*sum(vec_Wold_oe.*aux_1)-aux_11) +  exp(- κ_g + p.β*sum(vec_Wold_oge.*aux_1)-aux_11) + exp(- κ_c + p.β*sum(vec_Wold_oce.*aux_1)-aux_11) + exp(-κ_g - κ_c + p.β*sum(vec_Wold_ogce.*aux_1)-aux_11)) ;
            aux_2_124 = aux_11 + log(exp(γ_g+p.β*sum(vec_Wold_oe.*aux_1)-aux_11) +  exp(p.β*sum(vec_Wold_oge.*aux_1)-aux_11) + exp(γ_g - κ_c + p.β*sum(vec_Wold_oce.*aux_1)-aux_11) + exp(-κ_c + p.β*sum(vec_Wold_ogce.*aux_1)-aux_11)) ;
            aux_2_123 = aux_11 + log(exp(γ_c+p.β*sum(vec_Wold_oe.*aux_1)-aux_11) +  exp(γ_c - κ_g + p.β*sum(vec_Wold_oge.*aux_1)-aux_11) + exp(p.β*sum(vec_Wold_oce.*aux_1)-aux_11) + exp(-κ_g + p.β*sum(vec_Wold_ogce.*aux_1)-aux_11)) ;
            aux_2_1234 = aux_11 + log(exp(γ_c + γ_g + p.β*sum(vec_Wold_oe.*aux_1)-aux_11) +  exp(γ_c + p.β*sum(vec_Wold_oge.*aux_1)-aux_11) + exp(γ_g + p.β*sum(vec_Wold_oce.*aux_1)-aux_11) + exp(p.β*sum(vec_Wold_ogce.*aux_1)-aux_11)) ; 
            W_new[1,iz,ipm,iψe,ipe,iψo,ic,ig] = M.πgrid_oe[t,iz,ipm,iψe,ipe,iψo,ic,ig,ic_re,ig_re] + aux_2_12; #0.5772+M.πgrid_oe[t,iz,ipm,iψe,ipe,iψo,ic,ig,ic_re,ig_re] + aux_2_12;
            W_new[2,iz,ipm,iψe,ipe,iψo,ic,ig] = M.πgrid_oge[t,iz,ipm,iψe,ipe,iψo,ic,ig,ic_re,ig_re] + aux_2_124; #0.5772+M.πgrid_oge[t,iz,ipm,iψe,ipe,iψo,ic,ig,ic_re,ig_re] + aux_2_124;
            W_new[3,iz,ipm,iψe,ipe,iψo,ic,ig] = M.πgrid_oce[t,iz,ipm,iψe,ipe,iψo,ic,ig,ic_re,ig_re] + aux_2_123; #0.5772+M.πgrid_oce[t,iz,ipm,iψe,ipe,iψo,ic,ig,ic_re,ig_re] + aux_2_123;
            W_new[4,iz,ipm,iψe,ipe,iψo,ic,ig] = M.πgrid_ogce[t,iz,ipm,iψe,ipe,iψo,ic,ig,ic_re,ig_re] + aux_2_1234; #0.5772+M.πgrid_ogce[t,iz,ipm,iψe,ipe,iψo,ic,ig,ic_re,ig_re] + aux_2_1234;
        end
        W_new = convert(Array,W_new);
        return W_new
    end
#

## Choice probability for estimation (with random effects, not firm by firm)
    function ChoiceProbability_func_est_re(M::Model,W,𝓕,grid_indice,Κ)
        ## Given an expected value function and fixed cost parameters for firm i, this function returns:
            # 1. Choice probability for each fuel set
        ## Inputs 
            # M:            Model
            # W:            Expected value functions
            # grid_indices: points on the grids associated with firm's state variables
            # 𝓕:            Firm's fuel set this period 
            # i:            Firm i
            # t:            Period t
        @unpack p,n_z,n_ψe,n_pe,n_ψo,n_c,n_g,ng_re,nc_re,πgrid_oe, πgrid_oge, πgrid_oce, πgrid_ogce = M
        # vgrid_oe = Array{Float64}(undef,n_firms,p.T,n_z,n_e,n_o,n_c,n_g)                       ;
        # vgrid_oge = Array{Float64}(undef,n_firms,p.T,n_z,n_e,n_o,n_c,n_g)                      ;
        # vgrid_oce = Array{Float64}(undef,n_firms,p.T,n_z,n_e,n_o,n_c,n_g)                      ;
        # vgrid_ogce = Array{Float64}(undef,n_firms,p.T,n_z,n_e,n_o,n_c,n_g)                     ;
        # Fixed cost parameters
        κ_g = Κ[1] ;
        κ_c = Κ[2] ;
        γ_g = Κ[3] ;
        γ_c = Κ[4] ;
        # Vectorized expected value functions
        vec_Wold_oe = vec(W[1,:,:,:,:,:,:,:,:])     ;
        vec_Wold_oge = vec(W[2,:,:,:,:,:,:,:,:])    ;
        vec_Wold_oce = vec(W[3,:,:,:,:,:,:,:,:])    ;
        vec_Wold_ogce = vec(W[4,:,:,:,:,:,:,:,:])   ;
        # Find the indice associated with the firm's persistent state variables
        iz = grid_indice[1];
        iψe = grid_indice[2];
        ipe = grid_indice[3];
        iψo = grid_indice[4];
        ic = grid_indice[5];
        ig = grid_indice[6];
        ic_re = grid_indice[7];
        ig_re = grid_indice[8];
        Π_cre = zeros(nc_re);
        Π_cre[ic_re] = 1; 
        Π_gre = zeros(ng_re);
        Π_gre[ig_re] = 1;
        aux_1 = kronecker(Π_gre,Π_cre,M.Π_g[ig,:],M.Π_c[ic,:],M.Π_ψo[iψo,:],M.Π_pe[ipe,:],M.Π_ψe[iψe,:],M.Π_z[iz,:]) ;
        # Compute normalized choice-specific value functions
        v_oe = p.β*sum(vec_Wold_oe.*aux_1)    ;
        v_oge = p.β*sum(vec_Wold_oge.*aux_1)   ;
        v_oce = p.β*sum(vec_Wold_oce.*aux_1)   ;
        v_ogce = p.β*sum(vec_Wold_ogce.*aux_1) ;
        if 𝓕 == 12 # From OE
            v_oge = v_oge - κ_g             ;
            v_oce = v_oce - κ_c             ;
            v_ogce = v_ogce - κ_g - κ_c     ;
        elseif 𝓕 == 124 # From OGE 
            v_oe = v_oe + γ_g               ;
            v_oce = v_oce + γ_g - κ_c       ;
            v_ogce = v_ogce - κ_c           ;
        elseif 𝓕 == 123 # From OCE
            v_oe = v_oe + γ_c               ;
            v_oge = v_oge + γ_c - κ_g       ;
            v_ogce = v_ogce - κ_g           ;
        elseif 𝓕 == 1234 # From OGCE
            v_oe = v_oe + γ_c + γ_g         ;
            v_oge = v_oge + γ_c             ;
            v_oce = v_oce + γ_g             ;
        end
        # Probability of each choice given state variables
        aux_2 = exp(v_oe-v_ogce) + exp(v_oge-v_ogce) + exp(v_oce-v_ogce) + exp(v_ogce-v_ogce) ;
        P_oe = exp.(v_oe.-v_ogce)./aux_2        ;
        P_oge = exp.(v_oge.-v_ogce)./aux_2      ;
        P_oce = exp.(v_oce.-v_ogce)./aux_2      ;
        P_ogce = exp.(v_ogce.-v_ogce)./aux_2    ;
        return P_oe, P_oge, P_oce, P_ogce
    end
#


## Choice probability
    function ChoiceProbability_func(M::Model,W,ϵF,𝓕,i,t,Κ)
        ## Given an expected value function and fixed cost parameters for firm i, this function returns:
            # 1. Choice probability for each fuel set
        ## Inputs 
            # M:            Model
            # W:            Expected value functions
            # grid_indices: points on the grids associated with firm's state variables
            # 𝓕:            Firm's fuel set this period 
            # i:            Firm i
            # t:            Period t
        @unpack p,n_z,n_e,n_o,n_c,n_g,πgrid_oe, πgrid_oge, πgrid_oce, πgrid_ogce = M
        # vgrid_oe = Array{Float64}(undef,n_firms,p.T,n_z,n_e,n_o,n_c,n_g)                       ;
        # vgrid_oge = Array{Float64}(undef,n_firms,p.T,n_z,n_e,n_o,n_c,n_g)                      ;
        # vgrid_oce = Array{Float64}(undef,n_firms,p.T,n_z,n_e,n_o,n_c,n_g)                      ;
        # vgrid_ogce = Array{Float64}(undef,n_firms,p.T,n_z,n_e,n_o,n_c,n_g)                     ;
        # Fixed cost parameters
        κ_g = Κ[1] ;
        κ_c = Κ[2] ;
        γ_g = Κ[3] ;
        γ_c = Κ[4] ;
        # Vectorized expected value functions
        vec_Wold_oe = vec(W[1,:,:,:,:,:])     ;
        vec_Wold_oge = vec(W[2,:,:,:,:,:])    ;
        vec_Wold_oce = vec(W[3,:,:,:,:,:])    ;
        vec_Wold_ogce = vec(W[4,:,:,:,:,:])   ;
        # Find the indice associated with the firm's persistent state variables
        # ig = grid_indices.g[i] ;
        # ic = grid_indices.c[i] ;
        # ie = grid_indices.e[i] ;
        # io = grid_indices.o[i] ;
        # i_z = grid_indices.z[i] ;
        aux_1 = kronecker(M.Π_z[1,:],M.Π_e[1,:],M.Π_ψo[1,:],M.Π_c[1,:],M.Π_g[1,:]) ;
        # if ig == 0 & ipc == 0 # Draw from stationary distribution for price of gas and coal if firm isn't using gas and coal
        #     aux_1 = kronecker(M.Π_z[i_z,:],M.Π0_pg,M.Π0_pc,M.Π_pe[i_pe,:],M.Π0_ψ,M.Π0_ψ,M.Π0_ψ,M.Π0_ψ)                  ;
        # elseif ig == 0 & ipc > 0 # Draw from stationary distribution for price of gas if firm isn't using gas
        #     aux_1 = kronecker(M.Π_z[i_z,:],M.Π0_pg,M.Π_pc[i_pc,:],M.Π_pe[i_pe,:],M.Π0_ψ,M.Π0_ψ,M.Π0_ψ,M.Π0_ψ)           ;
        # elseif ig > 0 & i_pc == 0 # Draw from stationary distribution for price of coal if firm isn't using coal
        #     aux_1 = kronecker(M.Π_z[i_z,:],M.Π_pg[i_pg,:],M.Π0_pc,M.Π_pe[i_pe,:],M.Π0_ψ,M.Π0_ψ,M.Π0_ψ,M.Π0_ψ)           ;
        # else
        #     aux_1 = kronecker(M.Π_z[i_z,:],M.Π_pg[i_pg,:],M.Π_pc[i_pc,:],M.Π_pe[i_pe,:],M.Π0_ψ,M.Π0_ψ,M.Π0_ψ,M.Π0_ψ)    ;
        # end
        # Compute normalized choice-specific value functions
        v_oe = p.β*sum(vec_Wold_oe.*aux_1) + ϵF[1]    ;
        v_oge = p.β*sum(vec_Wold_oge.*aux_1) + ϵF[2]   ;
        v_oce = p.β*sum(vec_Wold_oce.*aux_1) + ϵF[3]   ;
        v_ogce = p.β*sum(vec_Wold_ogce.*aux_1) + ϵF[4] ;
        if 𝓕 == 12 # From OE
            v_oge = v_oge - κ_g             ;
            v_oce = v_oce - κ_c             ;
            v_ogce = v_ogce - κ_g - κ_c     ;
        elseif 𝓕 == 124 # From OGE 
            v_oe = v_oe + γ_g               ;
            v_oce = v_oce + γ_g - κ_c       ;
            v_ogce = v_ogce - κ_c           ;
        elseif 𝓕 == 123 # From OCE
            v_oe = v_oe + γ_c               ;
            v_oge = v_oge + γ_c - κ_g       ;
            v_ogce = v_ogce - κ_g           ;
        elseif 𝓕 == 1234 # From OGCE
            v_oe = v_oe + γ_c + γ_g         ;
            v_oge = v_oge + γ_c             ;
            v_oce = v_oce + γ_g             ;
        end
        # Probability of each choice given state variables
        # aux_2 = exp(v_oe-v_ogce) + exp(v_oge-v_ogce) + exp(v_oce-v_ogce) + exp(v_ogce-v_ogce) ;
        # P_oe = exp.(v_oe.-v_ogce)./aux_2        ;
        # P_oge = exp.(v_oge.-v_ogce)./aux_2      ;
        # P_oce = exp.(v_oce.-v_ogce)./aux_2      ;
        # P_ogce = exp.(v_ogce.-v_ogce)./aux_2    ;
        # return P_oe, P_oge, P_oce, P_ogce
        return v_oe, v_oge, v_oce, v_ogce
    end
#


## Choice probability (new, not firm by firm)
    function ChoiceProbability_func_new(M::Model,W,ϵF,𝓕,grid_indice,Κ)
        ## Given an expected value function and fixed cost parameters for firm i, this function returns:
            # 1. Choice probability for each fuel set
        ## Inputs 
            # M:            Model
            # W:            Expected value functions
            # grid_indices: points on the grids associated with firm's state variables
            # 𝓕:            Firm's fuel set this period 
            # i:            Firm i
            # t:            Period t
        @unpack p,n_z,n_ψe,n_pe,n_ψo,n_c,n_g,πgrid_oe, πgrid_oge, πgrid_oce, πgrid_ogce = M
        # vgrid_oe = Array{Float64}(undef,n_firms,p.T,n_z,n_e,n_o,n_c,n_g)                       ;
        # vgrid_oge = Array{Float64}(undef,n_firms,p.T,n_z,n_e,n_o,n_c,n_g)                      ;
        # vgrid_oce = Array{Float64}(undef,n_firms,p.T,n_z,n_e,n_o,n_c,n_g)                      ;
        # vgrid_ogce = Array{Float64}(undef,n_firms,p.T,n_z,n_e,n_o,n_c,n_g)                     ;
        # Fixed cost parameters
        κ_g = Κ[1] ;
        κ_c = Κ[2] ;
        γ_g = Κ[3] ;
        γ_c = Κ[4] ;
        # Vectorized expected value functions
        vec_Wold_oe = vec(W[1,:,:,:,:,:,:])     ;
        vec_Wold_oge = vec(W[2,:,:,:,:,:,:])    ;
        vec_Wold_oce = vec(W[3,:,:,:,:,:,:])    ;
        vec_Wold_ogce = vec(W[4,:,:,:,:,:,:])   ;
        # Find the indice associated with the firm's persistent state variables
        iz = grid_indice[1];
        iψe = grid_indice[2];
        ipe = grid_indice[3];
        iψo = grid_indice[4];
        ic = grid_indice[5];
        ig = grid_indice[6];
        #aux_1 = kronecker(M.Π_z[iz,:],M.Π_ψe[iψe,:],M.Π_pe[ipe,:],M.Π_ψo[iψo,:],M.Π_c[ic,:],M.Π_g[ig,:]) ;
        aux_1 = kronecker(M.Π_g[ig,:],M.Π_c[ic,:],M.Π_ψo[iψo,:],M.Π_pe[ipe,:],M.Π_ψe[iψe,:],M.Π_z[iz,:]) ;
        # if ig == 0 & ipc == 0 # Draw from stationary distribution for price of gas and coal if firm isn't using gas and coal
        #     aux_1 = kronecker(M.Π_z[i_z,:],M.Π0_pg,M.Π0_pc,M.Π_pe[i_pe,:],M.Π0_ψ,M.Π0_ψ,M.Π0_ψ,M.Π0_ψ)                  ;
        # elseif ig == 0 & ipc > 0 # Draw from stationary distribution for price of gas if firm isn't using gas
        #     aux_1 = kronecker(M.Π_z[i_z,:],M.Π0_pg,M.Π_pc[i_pc,:],M.Π_pe[i_pe,:],M.Π0_ψ,M.Π0_ψ,M.Π0_ψ,M.Π0_ψ)           ;
        # elseif ig > 0 & i_pc == 0 # Draw from stationary distribution for price of coal if firm isn't using coal
        #     aux_1 = kronecker(M.Π_z[i_z,:],M.Π_pg[i_pg,:],M.Π0_pc,M.Π_pe[i_pe,:],M.Π0_ψ,M.Π0_ψ,M.Π0_ψ,M.Π0_ψ)           ;
        # else
        #     aux_1 = kronecker(M.Π_z[i_z,:],M.Π_pg[i_pg,:],M.Π_pc[i_pc,:],M.Π_pe[i_pe,:],M.Π0_ψ,M.Π0_ψ,M.Π0_ψ,M.Π0_ψ)    ;
        # end
        # Compute normalized choice-specific value functions
        v_oe = p.β*sum(vec_Wold_oe.*aux_1) + ϵF[1]     ;
        v_oge = p.β*sum(vec_Wold_oge.*aux_1) + ϵF[2]   ;
        v_oce = p.β*sum(vec_Wold_oce.*aux_1) + ϵF[3]   ;
        v_ogce = p.β*sum(vec_Wold_ogce.*aux_1) + ϵF[4] ;
        if 𝓕 == 12 # From OE
            v_oge = v_oge - κ_g             ;
            v_oce = v_oce - κ_c             ;
            v_ogce = v_ogce - κ_g - κ_c     ;
        elseif 𝓕 == 124 # From OGE 
            v_oe = v_oe + γ_g               ;
            v_oce = v_oce + γ_g - κ_c       ;
            v_ogce = v_ogce - κ_c           ;
        elseif 𝓕 == 123 # From OCE
            v_oe = v_oe + γ_c               ;
            v_oge = v_oge + γ_c - κ_g       ;
            v_ogce = v_ogce - κ_g           ;
        elseif 𝓕 == 1234 # From OGCE
            v_oe = v_oe + γ_c + γ_g         ;
            v_oge = v_oge + γ_c             ;
            v_oce = v_oce + γ_g             ;
        end
        # Probability of each choice given state variables
        # aux_2 = exp(v_oe-v_ogce) + exp(v_oge-v_ogce) + exp(v_oce-v_ogce) + exp(v_ogce-v_ogce) ;
        # P_oe = exp.(v_oe.-v_ogce)./aux_2        ;
        # P_oge = exp.(v_oge.-v_ogce)./aux_2      ;
        # P_oce = exp.(v_oce.-v_ogce)./aux_2      ;
        # P_ogce = exp.(v_ogce.-v_ogce)./aux_2    ;
        # return P_oe, P_oge, P_oce, P_ogce
        return v_oe, v_oge, v_oce, v_ogce
    end
#


## Choice probability (with random effects, not firm by firm)
    function ChoiceProbability_func_re(M::Model,W,ϵF,𝓕,grid_indice,Κ)
        ## Given an expected value function and fixed cost parameters for firm i, this function returns:
            # 1. Choice probability for each fuel set
        ## Inputs 
            # M:            Model
            # W:            Expected value functions
            # grid_indices: points on the grids associated with firm's state variables
            # 𝓕:            Firm's fuel set this period 
            # i:            Firm i
            # t:            Period t
        @unpack p,n_z,n_ψe,n_pe,n_ψo,n_c,n_g,ng_re,nc_re,πgrid_oe, πgrid_oge, πgrid_oce, πgrid_ogce = M
        # vgrid_oe = Array{Float64}(undef,n_firms,p.T,n_z,n_e,n_o,n_c,n_g)                       ;
        # vgrid_oge = Array{Float64}(undef,n_firms,p.T,n_z,n_e,n_o,n_c,n_g)                      ;
        # vgrid_oce = Array{Float64}(undef,n_firms,p.T,n_z,n_e,n_o,n_c,n_g)                      ;
        # vgrid_ogce = Array{Float64}(undef,n_firms,p.T,n_z,n_e,n_o,n_c,n_g)                     ;
        # Fixed cost parameters
        κ_g = Κ[1] ;
        κ_c = Κ[2] ;
        γ_g = Κ[3] ;
        γ_c = Κ[4] ;
        # Vectorized expected value functions
        vec_Wold_oe = vec(W[1,:,:,:,:,:,:,:,:])     ;
        vec_Wold_oge = vec(W[2,:,:,:,:,:,:,:,:])    ;
        vec_Wold_oce = vec(W[3,:,:,:,:,:,:,:,:])    ;
        vec_Wold_ogce = vec(W[4,:,:,:,:,:,:,:,:])   ;
        # Find the indice associated with the firm's persistent state variables
        iz = grid_indice[1];
        iψe = grid_indice[2];
        ipe = grid_indice[3];
        iψo = grid_indice[4];
        ic = grid_indice[5];
        ig = grid_indice[6];
        ic_re = grid_indice[7];
        ig_re = grid_indice[8];
        Π_cre = zeros(nc_re);
        Π_cre[ic_re] = 1; 
        Π_gre = zeros(ng_re);
        Π_gre[ig_re] = 1;
        #aux_1 = kronecker(M.Π_z[iz,:],M.Π_ψe[iψe,:],M.Π_pe[ipe,:],M.Π_ψo[iψo,:],M.Π_c[ic,:],M.Π_g[ig,:]) ;
        aux_1 = kronecker(Π_gre,Π_cre,M.Π_g[ig,:],M.Π_c[ic,:],M.Π_ψo[iψo,:],M.Π_pe[ipe,:],M.Π_ψe[iψe,:],M.Π_z[iz,:]) ;
        # if ig == 0 & ipc == 0 # Draw from stationary distribution for price of gas and coal if firm isn't using gas and coal
        #     aux_1 = kronecker(M.Π_z[i_z,:],M.Π0_pg,M.Π0_pc,M.Π_pe[i_pe,:],M.Π0_ψ,M.Π0_ψ,M.Π0_ψ,M.Π0_ψ)                  ;
        # elseif ig == 0 & ipc > 0 # Draw from stationary distribution for price of gas if firm isn't using gas
        #     aux_1 = kronecker(M.Π_z[i_z,:],M.Π0_pg,M.Π_pc[i_pc,:],M.Π_pe[i_pe,:],M.Π0_ψ,M.Π0_ψ,M.Π0_ψ,M.Π0_ψ)           ;
        # elseif ig > 0 & i_pc == 0 # Draw from stationary distribution for price of coal if firm isn't using coal
        #     aux_1 = kronecker(M.Π_z[i_z,:],M.Π_pg[i_pg,:],M.Π0_pc,M.Π_pe[i_pe,:],M.Π0_ψ,M.Π0_ψ,M.Π0_ψ,M.Π0_ψ)           ;
        # else
        #     aux_1 = kronecker(M.Π_z[i_z,:],M.Π_pg[i_pg,:],M.Π_pc[i_pc,:],M.Π_pe[i_pe,:],M.Π0_ψ,M.Π0_ψ,M.Π0_ψ,M.Π0_ψ)    ;
        # end
        # Compute normalized choice-specific value functions
        v_oe = p.β*sum(vec_Wold_oe.*aux_1) + ϵF[1]     ;
        v_oge = p.β*sum(vec_Wold_oge.*aux_1) + ϵF[2]   ;
        v_oce = p.β*sum(vec_Wold_oce.*aux_1) + ϵF[3]   ;
        v_ogce = p.β*sum(vec_Wold_ogce.*aux_1) + ϵF[4] ;
        if 𝓕 == 12 # From OE
            v_oge = v_oge - κ_g             ;
            v_oce = v_oce - κ_c             ;
            v_ogce = v_ogce - κ_g - κ_c     ;
        elseif 𝓕 == 124 # From OGE 
            v_oe = v_oe + γ_g               ;
            v_oce = v_oce + γ_g - κ_c       ;
            v_ogce = v_ogce - κ_c           ;
        elseif 𝓕 == 123 # From OCE
            v_oe = v_oe + γ_c               ;
            v_oge = v_oge + γ_c - κ_g       ;
            v_ogce = v_ogce - κ_g           ;
        elseif 𝓕 == 1234 # From OGCE
            v_oe = v_oe + γ_c + γ_g         ;
            v_oge = v_oge + γ_c             ;
            v_oce = v_oce + γ_g             ;
        end
        # Probability of each choice given state variables
        # aux_2 = exp(v_oe-v_ogce) + exp(v_oge-v_ogce) + exp(v_oce-v_ogce) + exp(v_ogce-v_ogce) ;
        # P_oe = exp.(v_oe.-v_ogce)./aux_2        ;
        # P_oge = exp.(v_oge.-v_ogce)./aux_2      ;
        # P_oce = exp.(v_oce.-v_ogce)./aux_2      ;
        # P_ogce = exp.(v_ogce.-v_ogce)./aux_2    ;
        # return P_oe, P_oge, P_oce, P_ogce
        return v_oe, v_oge, v_oce, v_ogce
    end
#