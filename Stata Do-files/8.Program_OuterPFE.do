*** CODEFILE 8 ***

**** This implements the outer production function estimation for the indian steel industry ***
global ASIpaneldir Data/Panel_Data/Clean_data

* Import data and set panel
use Data/Panel_Data/Clean_data/ASI_PanelClean-allind_b, clear
xtset IDnum year

* Keep steel manufacturing
keep if nic08_4d == 2410

************************************
*** 1. Estimate demand parameters
************************************

preserve
	gen lny_tilde = lny-ln(1/N)

	foreach vars in rho d0 d2009 d2010 d2011 d2012 d2013 d2014 d2015 d2016 theta {
		gen `vars' = .
	}
	* Estimate elasticity of demand (state-by-year variation in the price of electricity interacted with lagged electricity input share)
		levelsof nic08_4d, local(ind)
		local iter_ind = 1
		foreach j of local ind {
			ivreg lny_tilde D2009-D2016 (lnpout = zgas zcoal zoil) if nic08_4d == `j', nocons
			replace rho = -_b[lnpout] if nic08_4d == `j'
			replace d0 = 0 if nic08_4d == `j'
			forvalues yr = 2009/2016 {
				replace d`yr' = _b[D`yr'] if nic08_4d == `j'
			}
		}
		gen demand_cons = d0 if year == 2009
		forvalues yr = 2010/2016 {
			replace demand_cons = d0 + d`yr' if year == `yr'
		}

		* Get aggregate price index from elasticity of demand and observed output prices 
		egen pout_agg = total(pout^(1-rho)), by(year nic08_4d)
		replace pout_agg = (pout_agg/N)^(1/(1-rho))
		gen lnpout_agg = log(pout_agg)

		* get elasticity of substitution wrt outside good	
		capture program drop gmm_outsidegood
		program gmm_outsidegood
			version 16
			
			syntax varlist [if], at(name) rhs(varlist) lhs(varname)
			
			local m1: word 1 of `varlist'
			
			local lnpout_agg: word 1 of `lhs'
			local demand_cons: word 1 of `rhs'
			local d0: word 2 of `rhs'
			local rho: word 3 of `rhs'
			
			* Parameters
			tempname theta
			scalar `theta'=`at'[1,1]
			
			tempvar res pstruc theta_est
				
			qui gen double `pstruc' = (`demand_cons')/`lnpout_agg' `if'
			qui gen double `theta_est' = (`pstruc'-`rho'+1)/(`pstruc'-`rho') `if'
			
			* Residuals
			qui gen double `res' = `theta'-`theta_est' `if'
			* Moments	
			qui replace `m1' = `res' `if'
		end
		mat pinit = [0.1]
		levelsof nic08_4d, local(ind)
		local iter_ind = 1
		foreach j of local ind {
			gmm gmm_outsidegood if nic08_4d == `j', one nequations(1) parameters(theta) winitial(identity) lhs(lnpout_agg) rhs(demand_cons d0 rho) from(pinit) iterate(500) 
			replace theta = _b[/theta] if nic08_4d == `j'
		}
		* Separate aggregate demand shock from aggregate price index
		gen gam = demand_cons - ((1+rho*(theta-1))/(theta-1))*lnpout_agg
		gen lnypred_struc = d0 + ln(1/N) + ((1+rho*(theta-1))/(theta-1))*lnpout_agg - rho*lnpout
		gen eps_demand_struc = lny - lnypred_struc 	// Aggregate demand shock (logs)
		gen eps_demand = exp(eps_demand_struc)		// Aggregate demand shock (levels)
		
		* Keep relevant variables and merge with main data
		keep IDnum year nic08_4d lnypred_struc eps_demand_struc theta rho d0 demand_cons pout_agg lnpout_agg eps_demand gam
		tempfile demand
		save `demand'
restore

* Merge demand estimates with main data
merge 1:1 IDnum year nic08_4d using `demand'
gen theta_constraint = rho*(1-theta)
su theta_constraint
drop _merge

********************************************************************
*** 2. Perform PFE estimation
********************************************************************

* Program for PFE using NLLS
capture program drop nlpfe
program nlpfe
    version 16
    
    syntax varlist(min=5 max=5) if, at(name)
    local LogRev: word 1 of `varlist'
    local Mspend: word 2 of `varlist'
	local Lspend: word 3 of `varlist'
	local KL: word 4 of `varlist'
	local rho: word 5 of `varlist'

    // Define parameters
    tempname eta akal sig
	scalar `eta' = `at'[1,1]
    scalar `akal' = `at'[1,2]
	scalar `sig' = `at'[1,3]
	
    // Some temporary variables (functions within CES)
    tempvar kterm sigterm
	generate double `sigterm' = (`sig'-1)/`sig' `if'
	generate double `kterm' = `akal'*((`KL')^(`sigterm')) `if'

    // Now fill in dependent variable
	replace `LogRev' = ln(1/(`eta')) + ln(`rho'/(`rho'-1)) + ln(`Lspend'*(1+`kterm') + `Mspend') `if'
end
foreach vars in eta akal sig aeal amal ae am al ak {
	gen `vars' = .
}
levelsof nic08_4d, local(ind)
local iter_ind = 1
foreach j of local ind {
	nl pfe @ LogYspend Mspend_all Lspend_nominal KL rho if nic08_4d == `j', parameters(eta akal sig) initial(eta 0.8 akal 0 sig 1.5) vce(robust) iterate(1000)
	if _b[/sig] < 0 {
		nl pfe @ LogYspend Mspend_all Lspend_nominal KL rho if nic08_4d == `j', parameters(eta akal sig) initial(eta 0.8 akal 0 sig 0.5) vce(robust) iterate(1000)
	}
	replace eta = _b[/eta] if nic08_4d == `j'
	replace sig = _b[/sig] if nic08_4d == `j'
	replace akal = _b[/akal] if nic08_4d == `j'
}
replace aeal = Espend_gmean/Lspend_gmean
replace amal = Mspend_gmean/Lspend_gmean
replace al = 1/(1+aeal+amal+akal)
replace ak = akal*al
replace ae = aeal*al
replace am = amal*al

********************************************************************
*** 3. Recover heterogeneity in prices of energy and materials
********************************************************************

gen E_struc = (((Espend_nominal/Lspend_nominal)*(al/ae))^(sig/(sig-1)))*L
gen M_struc = (((Mspend_nominal/Lspend_nominal)*(al/am))^(sig/(sig-1)))*L
gen Pe = Espend_nominal/E_struc 
gen Pm = Mspend_nominal/M_struc
gen logPe = log(Pe)
gen logPm = log(Pm)

*********************************************************
*** 4. Reporting production function estimation results
*********************************************************

*** Get average output elasticities ***
gen lterm = (al*(L)^((sig-1)/sig))
gen mterm = (am*(M_struc)^((sig-1)/sig))
gen kterm = (ak*(K)^((sig-1)/sig))
gen eterm = (ae*(E_struc)^((sig-1)/sig))
gen Q = lterm+mterm+kterm+eterm
* labor
gen eps_l = eta*lterm/Q
* capital
gen eps_k = eta*kterm/Q
* intermediates
gen eps_m = eta*mterm/Q
* energy
gen eps_e = eta*eterm/Q

*** Get average revenue elasticities ***
gen reps_l = ((rho-1)/rho)*eta*lterm/Q
gen reps_k = ((rho-1)/rho)*eta*kterm/Q
gen reps_m = ((rho-1)/rho)*eta*mterm/Q
gen reps_e = ((rho-1)/rho)*eta*eterm/Q

*** Get Bootstrap standard errors and create table for parameter values (Steel) ***
{
		local nrep 499
		set seed 420
		mat param_struc_boot = J(`nrep',4,.)
		mat eps_avg = J(`nrep',4,.)
		mat reps_avg = J(`nrep',4,.)
		forvalues i=1/`nrep'{
			preserve
				keep IDnum year Lqty Kqty Lspend_nominal Espend_nominal Mspend_nominal Mspend_all LogYspend nic08_4d pout lnpout zgas zcoal zoil D2009-D2016 lny N
				bsample, strata(nic08_4d)/* Sample with replacement within each industry*/
				display _newline(2) `i' /* Display iteration number */
				* Create variables
				qui egen Espend_gmean = gmean(Espend_nominal), by(nic08_4d)
				qui egen Lspend_gmean = gmean(Lspend_nominal), by(nic08_4d)
				qui egen Mspend_gmean = gmean(Mspend_nominal), by(nic08_4d)
				qui egen Lgmean = gmean(Lqty), by(nic08_4d)
				qui gen L = Lqty/Lgmean
				qui egen Kgmean = gmean(Kqty), by(nic08_4d)
				qui gen K = Kqty/Kgmean
				qui gen KL = K/L
				/// Estimate demand
					qui gen lny_tilde = lny-ln(1/N)
					foreach vars in rho d0 d2009 d2010 d2011 d2012 d2013 d2014 d2015 d2016 theta {
						qui gen `vars' = .
					}
					* Estimate elasticity of demand (state-by-year variation in the price of electricity interacted with lagged electricity input share)
					qui ivreg lny_tilde D2009-D2016 (lnpout = zgas zcoal zoil), nocons
					qui replace rho = -_b[lnpout]
					qui replace d0 = 0 
					forvalues yr = 2009/2016 {
						qui replace d`yr' = _b[D`yr']
					}
					qui gen demand_cons = d0 if year == 2010
					forvalues yr = 2010/2016 {
						qui replace demand_cons = d0 + d`yr' if year == `yr'
					}
					* Get aggregate price index from elasticity of demand and observed output prices 
					qui egen pout_agg = total(pout^(1-rho)), by(year nic08_4d)
					qui replace pout_agg = (pout_agg/N)^(1/(1-rho))
					qui gen lnpout_agg = log(pout_agg)
					* get elasticity of substitution wrt outside good	
					mat pinit = [0.1]
					qui: gmm gmm_outsidegood, one nequations(1) parameters(theta) winitial(identity) lhs(lnpout_agg) rhs(demand_cons d0 rho) from(pinit) iterate(500) 
					qui replace theta = _b[/theta]
					qui gen rho_boot = rho
					qui gen theta_boot = theta
				///
				* Perform estimation with bootstrap sample
				foreach vars in eta akal sig aeal amal ae am al ak {
					qui gen `vars'_boot = .
				}
				qui: nl pfe @ LogYspend Mspend_all Lspend_nominal KL rho_boot, parameters(eta akal sig) initial(eta 0.8 akal 0 sig 1.5) vce(robust) iterate(1000)
				if _b[/sig] < 0 | _b[/sig] > 100 {
					qui: nl pfe @ LogYspend Mspend_all Lspend_nominal KL rho_boot, parameters(eta akal sig) initial(eta 0.8 akal 0 sig 0.5) vce(robust) iterate(1000)
				}
				qui replace eta_boot = _b[/eta]
				qui replace sig_boot = _b[/sig]
				qui replace akal_boot = _b[/akal]
				* Recover ae/al and am/al from optimality condition
				qui replace aeal_boot = Espend_gmean/Lspend_gmean
				qui replace amal_boot = Mspend_gmean/Lspend_gmean
				qui replace al_boot = 1/(1+aeal_boot+amal_boot+akal_boot)
				qui replace ak_boot = akal_boot*al_boot
				qui replace ae_boot = aeal_boot*al_boot
				qui replace am_boot = amal_boot*al_boot
				* Recover structural parameters
				* sig
				quietly: su sig_boot
				mat param_struc_boot[`i',1] = r(mean)
				* eta
				quietly: su eta_boot
				mat param_struc_boot[`i',2] = r(mean)
				* rho
				quietly: su rho_boot
				mat param_struc_boot[`i',3] = r(mean)
				* theta
				quietly: su theta_boot
				mat param_struc_boot[`i',4] = r(mean)
				* Recover average elasticities (output and revenue)
				qui gen E_struc = (((Espend_nominal/Lspend_nominal)*(al_boot/ae_boot))^(sig_boot/(sig_boot-1)))*L
				qui gen M_struc = (((Mspend_nominal/Lspend_nominal)*(al_boot/am_boot))^(sig_boot/(sig_boot-1)))*L
				qui gen lterm = (al_boot*(L)^((sig_boot-1)/sig_boot))
				qui gen mterm = (am_boot*(M_struc)^((sig_boot-1)/sig_boot))
				qui gen kterm = (ak_boot*(K)^((sig_boot-1)/sig_boot))
				qui gen eterm = (ae_boot*(E_struc)^((sig_boot-1)/sig_boot))
				qui gen Q = lterm+mterm+kterm+eterm
				* labor
				qui gen eps_l = eta_boot*lterm/Q
				qui gen reps_l = ((rho_boot-1)/rho_boot)*eta_boot*lterm/Q
				* capital
				qui gen eps_k = eta_boot*kterm/Q
				qui gen reps_k = ((rho_boot-1)/rho_boot)*eta_boot*kterm/Q
				* intermediates
				qui gen eps_m = eta_boot*mterm/Q
				qui gen reps_m = ((rho_boot-1)/rho_boot)*eta_boot*mterm/Q
				* energy
				qui gen eps_e = eta_boot*eterm/Q
				qui gen reps_e = ((rho_boot-1)/rho_boot)*eta_boot*eterm/Q
				* Output elasticities
				qui su eps_l
				mat eps_avg[`i',1] = r(mean)
				qui su eps_k
				mat eps_avg[`i',2] = r(mean)
				qui su eps_m
				mat eps_avg[`i',3] = r(mean)
				qui su eps_e
				mat eps_avg[`i',4] = r(mean)
				* Revenue elasticities
				qui su reps_l
				mat reps_avg[`i',1] = r(mean)
				qui su reps_k
				mat reps_avg[`i',2] = r(mean)
				qui su reps_m
				mat reps_avg[`i',3] = r(mean)
				qui su reps_e
				mat reps_avg[`i',4] = r(mean)
			restore
		}
		*** Table: bootstrap confidence intervals for parameter estimates and average output/revenue elasticities ***
		* Parameter estimates
		mata:
		param_boot = st_matrix("param_struc_boot")
		param_boot_lb95 = J(1,4,.)
		param_boot_ub95 = J(1,4,.)
		for (i=1;i<=4;i++) {
			param_boot_lb95[.,i] = mm_quantile(param_boot[.,i],1,0.05)
			param_boot_ub95[.,i] = mm_quantile(param_boot[.,i],1,0.95)
		}
		st_matrix("param_boot_lb",(param_boot_lb95))
		st_matrix("param_boot_ub",(param_boot_ub95))
		end
		* Average elasticities
		mata:
		eps_avg = st_matrix("eps_avg")
		eps_avg_lb = J(1,4,.)
		eps_avg_ub = J(1,4,.)
		for (i=1;i<=4;i++) {
			eps_avg_lb[.,i] = mm_quantile(eps_avg[.,i],1,0.05)
			eps_avg_ub[.,i] = mm_quantile(eps_avg[.,i],1,0.95)
		}
		st_matrix("eps_avg_lb",(eps_avg_lb))
		st_matrix("eps_avg_ub",(eps_avg_ub))
		end
		mata:
		reps_avg = st_matrix("reps_avg")
		reps_avg_lb = J(1,4,.)
		reps_avg_ub = J(1,4,.)
		for (i=1;i<=4;i++) {
			reps_avg_lb[.,i] = mm_quantile(reps_avg[.,i],1,0.05)
			reps_avg_ub[.,i] = mm_quantile(reps_avg[.,i],1,0.95)
		}
		st_matrix("reps_avg_lb",(reps_avg_lb))
		st_matrix("reps_avg_ub",(reps_avg_ub))
		end
		preserve
			file close _all
			file open PFE_results using "Output/Tables/OuterPFE/PFE_results-Steel.tex", write replace
			file write PFE_results "\begin{tabular}{@{}lllll@{}}"_n
			file write PFE_results "\toprule\hline"_n
			file write PFE_results "\multicolumn{2}{c}{\begin{tabular}[c]{@{}c@{}}Production and Demand \\ Parameters\end{tabular}} & \multicolumn{2}{c}{Average Output Elasticities} & \multicolumn{1}{c}{\begin{tabular}[c]{@{}c@{}}Average Revenue \\ Elasticities\end{tabular}} \\ \midrule"_n
			* Elasticity of substitution
			su sig
			local sig: di %3.2f r(mean)
			local sig_lb: di %4.3f param_boot_lb[1,1]
			local sig_ub: di %4.3f param_boot_ub[1,1]
			* Returns to scale
			su eta
			local eta: di %3.2f r(mean)
			local eta_lb: di %4.3f param_boot_lb[1,2]
			local eta_ub: di %4.3f param_boot_ub[1,2]
			* Elasticity of demand
			su rho
			local rho: di %3.2f r(mean)
			local rho_lb: di %4.3f param_boot_lb[1,3]
			local rho_ub: di %4.3f param_boot_ub[1,3]
			* Elasticity of outside good
			su theta
			local theta: di %3.2f r(mean)
			local theta_lb: di %4.3f param_boot_lb[1,4]
			local theta_ub: di %4.3f param_boot_ub[1,4]
			* Labor output and revenue elasticity 
			su eps_l
			local eps_l: di %4.3f r(mean)
			local eps_l_lb: di %4.3f eps_avg_lb[1,1]
			local eps_l_ub: di %4.3f eps_avg_ub[1,1]
			su reps_l
			local reps_l: di %4.3f r(mean)
			local reps_l_lb: di %4.3f reps_avg_lb[1,1]
			local reps_l_ub: di %4.3f reps_avg_ub[1,1]
			* Capital output and revenue elasticity
			su eps_k
			local eps_k: di %4.3f r(mean)
			local eps_k_lb: di %4.3f eps_avg_lb[1,2]
			local eps_k_ub: di %4.3f eps_avg_ub[1,2]
			su reps_k
			local reps_k: di %4.3f r(mean)
			local reps_k_lb: di %4.3f reps_avg_lb[1,2]
			local reps_k_ub: di %4.3f reps_avg_ub[1,2]
			* Materials output and revenue elasticity
			su eps_m
			local eps_m: di %4.3f r(mean)
			local eps_m_lb: di %4.3f eps_avg_lb[1,3]
			local eps_m_ub: di %4.3f eps_avg_ub[1,3]
			su reps_m
			local reps_m: di %4.3f r(mean)
			local reps_m_lb: di %4.3f reps_avg_lb[1,3]
			local reps_m_ub: di %4.3f reps_avg_ub[1,3]
			* Energy output and revenue elasticity
			su eps_e
			local eps_e: di %4.3f r(mean)
			local eps_e_lb: di %4.3f eps_avg_lb[1,4]
			local eps_e_ub: di %4.3f eps_avg_ub[1,4]
			su reps_e
			local reps_e: di %4.3f r(mean)
			local reps_e_lb: di %4.3f reps_avg_lb[1,4]
			local reps_e_ub: di %4.3f reps_avg_ub[1,4]
			* Number of observations
			su LogYspend
			local obs: di r(N)
			
			file write PFE_results "Elasticity of substitution $\hat\sigma$ & \begin{tabular}[c]{@{}l@{}}`sig'\\"
			file write PFE_results "\footnotesize{\color{darkgray}[`sig_lb',`sig_ub']}\end{tabular}" 
			file write PFE_results "& Labor & \begin{tabular}[c]{@{}l@{}}`eps_l'\\"
			file write PFE_results "\footnotesize{\color{darkgray}[`eps_l_lb',`eps_l_ub']}\end{tabular}"
			file write PFE_results "&\begin{tabular}[c]{@{}l@{}}`reps_l'\\"
			file write PFE_results "\footnotesize{\color{darkgray}[`reps_l_lb',`reps_l_ub']}\end{tabular}\\" _n
			
			file write PFE_results "Returns to scale $\hat\eta$ & \begin{tabular}[c]{@{}l@{}}`eta'\\"
			file write PFE_results "\footnotesize{\color{darkgray}[`eta_lb',`eta_ub']}\end{tabular}" 
			file write PFE_results "& Capital & \begin{tabular}[c]{@{}l@{}}`eps_k'\\"
			file write PFE_results "\footnotesize{\color{darkgray}[`eps_k_lb',`eps_k_ub']}\end{tabular}"
			file write PFE_results "&\begin{tabular}[c]{@{}l@{}}`reps_k'\\"
			file write PFE_results "\footnotesize{\color{darkgray}[`reps_k_lb',`reps_k_ub']}\end{tabular}\\" _n
			
			file write PFE_results "Elasticity of demand $\hat\rho$ & \begin{tabular}[c]{@{}l@{}}`rho'\\"
			file write PFE_results "\footnotesize{\color{darkgray}[`rho_lb',`rho_ub']}\end{tabular}" 
			file write PFE_results "& Materials & \begin{tabular}[c]{@{}l@{}}`eps_m'\\"
			file write PFE_results "\footnotesize{\color{darkgray}[`eps_m_lb',`eps_m_ub']}\end{tabular}"
			file write PFE_results "&\begin{tabular}[c]{@{}l@{}}`reps_m'\\"
			file write PFE_results "\footnotesize{\color{darkgray}[`reps_m_lb',`reps_m_ub']}\end{tabular}\\" _n
			
			file write PFE_results "Elasticity of outside good $\hat\theta$ & \begin{tabular}[c]{@{}l@{}}`theta'\\"
			file write PFE_results "\footnotesize{\color{darkgray}[`theta_lb',`theta_ub']}\end{tabular}" 
			file write PFE_results "& Energy & \begin{tabular}[c]{@{}l@{}}`eps_e'\\"
			file write PFE_results "\footnotesize{\color{darkgray}[`eps_e_lb',`eps_e_ub']}\end{tabular}"
			file write PFE_results "&\begin{tabular}[c]{@{}l@{}}`reps_e'\\"
			file write PFE_results "\footnotesize{\color{darkgray}[`reps_e_lb',`reps_e_ub']}\end{tabular}\\" _n
			
			file write PFE_results "\hline"_n
			file write PFE_results "Observations & `obs'\\"_n
			file write PFE_results "\hline\bottomrule"_n
			file write PFE_results "\multicolumn{5}{l}{\footnotesize Bootstrap 95\% confidence interval in bracket (499 reps)}\\"_n
			file write PFE_results "\end{tabular}"_n
			file close _all
		restore
	}
}


****************************************************************************************
*** 5. Recover  productivity (hicks-neutral) - using FOC for labor (Grieco et al. 2016)
****************************************************************************************
* Unanticipated shock to revenue (structural regression error)
gen sigterm = (sig-1)/sig
gen klterm = (ak/al)*((KL)^sigterm)
gen ln_uhat = LogYspend - log(1/eta) -log(rho/(rho-1)) - log(Lspend_nominal*(1+klterm) + Mspend_all)
gen uhat = exp(ln_uhat)
* hicks-neutral productivity
gen param_struc1 = (1-rho*(1-theta))/((1-theta)*rho)
gen param_struc2 = -(rho*(sig*(eta-1)+1) - (eta*sig))/((sig-1)*rho)
gen rhoterm = rho/(rho-1)
gen etaterm = 1/eta
gen zrho = (rhoterm*etaterm*(N^(1/rho))*(pout_agg^param_struc1)*Lspend_nominal*(Q^param_struc2))/((Ygmean^((rho-1)/(rho)))*al*(L^((sig-1)/sig))*((exp(gam))^(1/rho))) // with gam
gen z = zrho^(rho/(rho-1))
gen lnz = log(z)

****************************************
*** 6. Model Fit
****************************************

* Target rental rate of capital using profit maximization first-order conditions
gen pout_tilde = pout*Ygmean
gen w_tilde = wage*Lgmean
gen rk_tilde = p_capital*Kgmean
capture program drop gmm1
program gmm1
	version 16
	syntax varlist [if], at(name) rhs(varlist)
	
	local m1: word 1 of `varlist'
	local m2: word 2 of `varlist'
	local m3: word 3 of `varlist'
	local m4: word 4 of `varlist'
	
	local Edata: word 1 of `rhs'
	local Mdata: word 2 of `rhs'
	local Kdata: word 3 of `rhs'
	local Ldata: word 4 of `rhs'
	local ak: word 5 of `rhs'
	local al: word 6 of `rhs'
	local am: word 7 of `rhs'
	local ae: word 8 of `rhs'
	local eta: word 9 of `rhs'
	local w_tilde: word 10 of `rhs'
	local Pe: word 11 of `rhs'
	local Pm: word 12 of `rhs'
	local Yqty: word 13 of `rhs'
	local Ygmean: word 14 of `rhs'
	local rk_tilde: word 15 of `rhs'
	local pout: word 16 of `rhs'
	local sig: word 17 of `rhs'
	local rho: word 18 of `rhs'
	local theta: word 19 of `rhs'
	local N: word 20 of `rhs'
	local d0: word 21 of `rhs'
	local p_agg: word 22 of `rhs'
	local gam: word 23 of `rhs'
	
	// Define parameters 
	tempname r
	scalar `r' = `at'[1,1]
	
	// Some temporary variables 
	tempvar rk_new pinput Ypred z L K M E pterm_z1 pterm_z2 pterm_y1 pterm_y2
	gen double `rk_new' = `r'*`rk_tilde' `if'
	gen double `pinput' = ((`ak'^`sig')*(`rk_new'^(1-`sig')) + (`am'^`sig')*(`Pm'^(1-`sig')) + (`ae'^`sig')*(`Pe'^(1-`sig')) + (`al'^`sig')*(`w_tilde'^(1-`sig')))^(1/(1-`sig')) `if'
	
	gen double `pterm_z1' = -`eta'*(`rho'*(`theta'-1)+1)/(`rho'*(`theta'-1)) `if'
	gen double `pterm_z2' = ((1-`eta')*`rho' + `eta')/`rho' `if'
	gen double `pterm_y1' = (1+`rho'*(`theta'-1))/(`rho'*(`theta'-1)) `if'
	gen double `pterm_y2' = (`eta'*`rho')/((1-`eta')*`rho' + `eta') `if'
	
	gen double `z' = (1/`Ygmean')*(((`rho'/(`rho'-1))*(1/`eta')*`pinput')^`eta')*((`N'/exp(`gam'))^(`eta'/`rho'))*(`p_agg'^`pterm_z1')*(`Yqty'^`pterm_z2') `if'
	gen double `Ypred' = (((`z'*`Ygmean')^(1/`eta'))*((`rho'-1)/`rho')*`eta'*((exp(`gam')/`N')^(1/`rho'))*((`p_agg'^`pterm_y1')/`pinput'))^`pterm_y2' `if'
	gen double `L' = ((`Ypred'/(`Ygmean'*`z'))^(1/`eta'))*((`al'/`w_tilde')^`sig')*(`pinput'^`sig') `if'
	gen double `K' = ((`Ypred'/(`Ygmean'*`z'))^(1/`eta'))*((`ak'/`rk_new')^`sig')*(`pinput'^`sig') `if'
	gen double `M' = ((`Ypred'/(`Ygmean'*`z'))^(1/`eta'))*((`am'/`Pm')^`sig')*(`pinput'^`sig') `if'
	gen double `E' = ((`Ypred'/(`Ygmean'*`z'))^(1/`eta'))*((`ae'/`Pe')^`sig')*(`pinput'^`sig') `if'

	// Define moments
	replace `m1' = `E_data'-`E' `if'
	replace `m2' = `M_data'-`M' `if'
	replace `m3' = `L_data' - `L' `if'
	replace `m4' = `K_data' - `K' `if'
end
gen r = .
levelsof nic08_4d, local(ind)
local iter_ind = 1
foreach j of local ind {
	mat param_init = 0.5
	gmm gmm1 if nic08_4d == `j', one nequations(4) parameters(r) winitial(identity) rhs(E_struc M_struc K L ak al am ae eta w_tilde Pe Pm Yqty Ygmean rk_tilde pout sig rho theta N d0 pout_agg gam) from(param_init) conv_maxiter(100)
	replace r = _b[/r] if nic08_4d == `j'
}
replace rk_tilde = r*p_capital*Kgmean
gen pinput = ((ak^sig)*(rk_tilde^(1-sig)) + (am^sig)*(Pm^(1-sig)) + (ae^sig)*(Pe^(1-sig)) + (al^sig)*(w_tilde^(1-sig)))^(1/(1-sig))

* Get predicted model objects under the assumption that capital is rented flexibly
gen pterm_z1 = -eta*(rho*(theta-1)+1)/(rho*(theta-1))
gen pterm_z2 = ((1-eta)*rho + eta)/(rho)
gen pterm_y1 = (1+rho*(theta-1))/(rho*(theta-1))
gen pterm_y2 = (eta*rho)/((1-eta)*rho + eta)
gen Ypred = (((z*Ygmean)^(1/eta))*((rho-1)/rho)*eta*((exp(gam)/N)^(1/rho))*((pout_agg^pterm_y1)/pinput))^pterm_y2 // with gam
gen Lpred = ((Ypred/(Ygmean*z))^(1/eta))*((al/w_tilde)^sig)*(pinput^sig) 
gen Kpred = ((Ypred/(Ygmean*z))^(1/eta))*((ak/rk_tilde)^sig)*(pinput^sig) 
gen Mpred = ((Ypred/(Ygmean*z))^(1/eta))*((am/Pm)^sig)*(pinput^sig) 
gen Epred = ((Ypred/(Ygmean*z))^(1/eta))*((ae/Pe)^sig)*(pinput^sig) 

* Get predicted output price, revenue and profit under data and predicted by model (under assumption that capital is rented flexibly)
gen pout_pred = ((exp(gam)/(N*Ypred))^(1/rho))*(pout_agg^pterm_y1)
gen profit_emp = Yspend_nominal_new - (Mspend_all + Lspend_nominal + Kspend_nominal*r)
gen profit_pred = Ypred*pout_pred - (Mpred*Pm + Lpred*w_tilde + Epred*Pe + Kpred*rk_tilde)


***********************************************************************
*** 7. Export data for estimation of energy production function
***********************************************************************

* Save main dataset
save Data/Panel_Data/Clean_data/ASI_PostOuterEstimation-Steel, replace