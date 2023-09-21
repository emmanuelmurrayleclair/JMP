*** CODEFILE 9 ***

**** This implements estimation of the energy production function for the indian steel industry ***
* After estimation, the Markovian process is computed for all state variables, and the data is save to estimate dynamic parameters in julia
use Data/Panel_Data/Clean_data/ASI_PostOuterEstimation-Steel.dta, clear

**********************
*** 1. Prepare Data 
**********************
gen age = year-YearInitialProduction
gen pE_struc = Espend_nominal/E_struc
rename pE_struc pE
gen lnpE = log(pE)
rename E_struc E
gen lnE = ln(E)

* Rename fuel quantities
rename TotOil_mmbtu oil
rename TotGas_mmbtu gas
rename TotCoal_mmbtu coal
rename elecb_mmbtu elec

* Rename fuel prices
rename poil_mmbtu po
rename pgas_mmbtu pg
rename pcoal_mmbtu pc
rename pelecb_mmbtu pelec

* Keep firms who always use oil and electricity
keep if combineF == 12 | combineF == 123 | combineF == 124 | combineF == 1234
drop if elec == 0 | elec == .

* District identifiers
capture drop distnum
encode districtname, gen(distnum)

***************************************************************************************
*** 2. Table: relationship between  between price of energy and number of fuels
***************************************************************************************

* Table: pre-estimation relationship between price of realized energy (marginal cost) and number of fuels
foreach vars in o g c elec {
	capture drop lnp`vars'
	gen lnp`vars' = log(p`vars')
}
gen nfuel = 0
replace nfuel = 1 if combineF == 1 | combineF == 2 | combineF == 3 |  combineF == 4
replace nfuel = 2 if combineF == 12 | combineF == 13 | combineF == 14 | combineF == 23 | combineF == 24 | combineF == 34
replace nfuel = 3 if combineF == 123 | combineF == 124 | combineF == 134 | combineF == 234
replace nfuel = 4 if combineF == 1234

preserve 
eststo clear
local control1 lnpo lnpelec
local control2 lnz
eststo mdl1: quietly reg lnpE i.nfuel
eststo mdl2: quietly reg lnpE i.nfuel i.year
eststo mdl3: quietly reg lnpE i.nfuel i.year `control1'
eststo mdl4: quietly reg lnpE i.nfuel i.year lnpo lnpelec `control1' `control2'
esttab using "Output/Tables/EnergyPFE/EnergyPrice_nfuels-Steel.tex", se noconstant keep(2.nfuel 3.nfuel 4.nfuel) indicate("Year Dummies = *.year" "Controlling for fuel prices = `control1'" "Controlling for TFP = `control2'") replace
restore

*************************************************
*** 3. Estimation of CES distribution parameters 			
*************************************************

*** fuel quantity and prices normalized around geometric mean ***

* Fuel quantities
egen o_gmean = gmean(oil), by(nic08_4d)
egen g_gmean = gmean(gas), by(nic08_4d)
egen c_gmean = gmean(coal), by(nic08_4d)
egen elec_gmean = gmean(elec), by(nic08_4d)
gen o = oil/o_gmean
gen g = gas/g_gmean
gen c = coal/c_gmean
gen e = elec/elec_gmean
foreach vars in o g c e {
	gen ln`vars' = log(`vars')
}

* Fuel prices
foreach vars in o g c elec {
	replace `vars' = 0 if `vars' == .
	gen p`vars'_tilde = p`vars'*`vars'_gmean 
	replace p`vars'_tilde = 0 if `vars' == 0
	gen lnp`vars'_tilde = ln(p`vars'_tilde)
}

*** Recover distribution parameters of production function ***
egen ospend_gmean = gmean(po_tilde*o), by(nic08_4d)
egen gspend_gmean = gmean(pg_tilde*g), by(nic08_4d)
egen cspend_gmean = gmean(pc_tilde*c), by(nic08_4d)
egen espend_gmean = gmean(pelec_tilde*e), by(nic08_4d)
gen fuelspend_gmean = ospend_gmean + gspend_gmean + cspend_gmean + espend_gmean
gen bo = ospend_gmean/fuelspend_gmean
gen bg = gspend_gmean/fuelspend_gmean
gen bc = cspend_gmean/fuelspend_gmean
gen be = espend_gmean/fuelspend_gmean

*******************************************************
*** 4. Create variables for energy production function		
*******************************************************

*** Generate lagged variables ***

* Energy quantity and prices 
xtset IDnum year 
gen LlnE = L.lnE
gen LlnpE = L.lnpE

* Fuel quantity 
foreach vars in o g c e {
	gen L`vars' = L.`vars'
	gen Lln`vars' = ln(L`vars')
}

* Fuel prices 
foreach vars in o g c elec {
	gen Lp`vars'_tilde = L.p`vars'_tilde
	gen Lp`vars' = L.p`vars'
	gen Llnp`vars'_tilde = ln(Lp`vars'_tilde)
	gen Llnp`vars' = ln(Lp`vars')
}

* Generate lagged and  future fuel set choice
gen LcombineF = L.combineF
gen FcombineF = F.combineF

* Other variables (taking electricity as baseline fuel) 
gen grel = g/e
gen crel = c/e
gen orel = o/e
gen Lgrel = Lg/Le
gen Lcrel = Lc/Le
gen Lorel = Lo/Le
gen pgrel = pg_tilde/pelec_tilde
gen pcrel = pc_tilde/pelec_tilde
gen porel = po_tilde/pelec_tilde
gen Lpgrel = Lpg_tilde/Lpelec_tilde
gen Lpcrel = Lpc_tilde/Lpelec_tilde
gen Lporel = Lpo_tilde/Lpelec_tilde
gen relspend = 1 + pgrel*grel + pcrel*crel + porel*orel
gen Lrelspend = 1 + Lpgrel*Lgrel + Lpcrel*Lcrel + Lporel*Lorel
gen lnEe = lnE-lne
gen LlnEe = LlnE-Llne 
gen LlnpeE = Llnpelec_tilde-LlnpE
gen lnrelspend = ln(relspend)
gen Llnrelspend = ln(Lrelspend)

* Year dummies
forvalues t = 2010/2016 {
	capture drop D`t'
	gen D`t' = 0
	replace D`t' = 1 if year == `t'
}

*********************************************************************************
***	4. Estimation of energy production function - Blundell and Bond (System GMM)			
*********************************************************************************

* Get unrestricted parameters from Blundell and Bond (1998)
foreach vars in rho_tilde b1_tilde b2_tilde rho0 ll rho_eprod {
	gen `vars' = .
}
forvalues t = 2011/2016 {
	gen b`t' = .
}
levelsof nic08_4d, local(ind)
local iter_ind = 1
foreach j of local ind {
	xtdpdsys lnEe D2011-D2016 if nic08_4d == `j', lags(1) endogenous(lnrelspend,lags(1,.)) vce(robust)
	mat param_var_`j' = e(V)
	scalar n_`j' = e(N)
	replace rho_tilde = _b[L.lnEe] if nic08_4d == `j'
	replace b1_tilde = _b[lnrelspend] if nic08_4d == `j'
	replace b2_tilde = _b[L.lnrelspend] if nic08_4d == `j'
	replace rho0 = _b[_cons] if nic08_4d == `j'
	forvalues t = 2011/2016 {
		replace b`t' = _b[D`t'] if nic08_4d == `j'
	}
	* Test for common factors (steel: reject at 95%, fail to reject at 97% and above)
	testnl _b[L.lnrelspend] = -1*_b[lnrelspend]*_b[L.lnEe]
}
* Get structural (restricted) parameters by imposing common factors with minimum distance (GMM)
capture program drop gmm_struc
program gmm_struc
	version 16
	
	syntax varlist [if], at(name) rhs(varlist) lhs(varname)
	
	local m1: word 1 of `varlist'
	local m2: word 2 of `varlist'
	local m3: word 3 of `varlist'
	
	local b1_tilde: word 1 of `lhs'
	local b2_tilde: word 1 of `rhs'
	local rho_tilde: word 2 of `rhs'
	
	* Parameters
	tempname rho ll
	scalar `rho'=`at'[1,1] 
	scalar `ll'=`at'[1,2]
	
	* Moments
	qui replace `m1' = `b1_tilde'-(`ll'/(`ll'-1)) `if'
	qui replace `m2' = `b2_tilde'+((`rho')*(`ll'/(`ll'-1))) `if'
	qui replace `m3' = `rho_tilde'-`rho' `if'
end
mat pinit = [0.7,2.0]
levelsof nic08_4d, local(ind)
local iter_ind = 1
foreach j of local ind {
	gmm gmm_struc if nic08_4d == `j', one nequations(3) parameters(rho ll) winitial(identity) lhs(b1_tilde) rhs(b2_tilde rho_tilde) from(pinit) iterate(500)
	replace rho_eprod = _b[/rho] if nic08_4d == `j'
	replace ll = _b[/ll] if nic08_4d == `j'
}

// * Get 95% confidence intervals (delta method)
// levelsof nic08_4d, local(ind)
// foreach j of local ind {
// 	mat ci_`j' = J(2,2,.)
// 	scalar var_`j' = param_var_`j'[2,2]
// 	su b1_tilde if nic08_4d == `j'
// 	scalar se_ll_`j' = sqrt(var_`j'*((-1/((r(mean)-1)^2))^2))	
// 	scalar se_rho_`j' = sqrt(param_var_`j'[1,1]) 
// 	su ll if nic08_4d == `j'
// 	mat ci_`j'[1,1] = r(mean)-(3.291*se_ll_`j')
// 	mat ci_`j'[1,2] = r(mean)+(3.291*se_ll_`j')
// 	su rho_eprod if nic08_4d == `j'
// 	mat ci_`j'[2,1] = r(mean)-(3.291*se_rho_`j')
// 	mat ci_`j'[2,2] = r(mean)+(3.291*se_rho_`j')
// }
// * Construct table of results
// preserve
// file close _all
// file open PFE_results using "Output/Tables/EnergyPFE/PFE_estimate-Steel.tex", write replace
// *file write PFE_results "& Casting of Steel \& Iron & Cement & Basic Steel \\"_n
// file write PFE_results "\def\sym#1{\ifmmode^{#1}\else\(^{#1}\)\fi}"_n
// file write PFE_results "\caption{Estimates of Energy Production Function}"_n
// file write PFE_results "\begin{tabular}{l*{1}{c}}"_n
// file write PFE_results "\toprule\hline"_n
// file write PFE_results "& \multicolumn{1}{c}{Steel} \\"_n
// file write PFE_results "\midrule"_n
// * lamda estimate
// file write PFE_results "Elasticity of substitution $\hat\lambda$"
// su ll if nic08_4d == 2410
// local llhat: di %4.3f r(mean)
// file write PFE_results "&`llhat'\sym{***}"
// file write PFE_results "\\"_n
// * lambda standard error
// local ll_se_hat: di %4.3f se_ll_2410
// file write PFE_results "&(`ll_se_hat')"
// file write PFE_results "\\"_n
//
// * rho estimate
// file write PFE_results "Persistence of electricity productivity $\hat\rho_{\psi_e}$"
// su rho_eprod if nic08_4d == 2410
// local rhohat: di %4.3f r(mean)
// file write PFE_results "&`rhohat'\sym{***}"
// file write PFE_results "\\"_n
// * rho standard error
// local rho_se_hat: di %4.3f se_rho_2410
// file write PFE_results "&(`rho_se_hat')"
// file write PFE_results "\\"_n
// * Observations
// file write PFE_results "\midrule"_n
// file write PFE_results "Observations"
// local n_2410: di n_2410
// file write PFE_results "&`n_2410'"
// file write PFE_results "\\"_n
// file write PFE_results "\hline\bottomrule"_n
// file write PFE_results "\multicolumn{2}{l}{\footnotesize Standard errors in parentheses} \\"_n
// file write PFE_results "\multicolumn{2}{l}{\footnotesize \sym{+} \(p<0.1\), \sym{*} \(p<0.05\), \sym{**} \(p<0.01\), \sym{***} \(p<0.001\)} \\"_n
// file write PFE_results "\end{tabular}"_n
// file close _all
// restore

*******************************************************
***	5. Recover distribution of fuel productivity 
*******************************************************

* electricity productivity
gen lnfprod_e = lnEe - (ll/(ll-1))*ln(be) - (ll/(ll-1))*lnrelspend
gen fprod_e = exp(lnfprod_e)

* Relative productivity of other fuels
gen fprod_orel = (((po_tilde*be)/(pelec_tilde*bo))^(ll/(ll-1)))*((o/e)^(1/(ll-1)))
gen lnfprod_orel = ln(fprod_orel)
gen fprod_grel = (((pg_tilde*be)/(pelec_tilde*bg))^(ll/(ll-1)))*((g/e)^(1/(ll-1)))
gen lnfprod_grel = ln(fprod_grel)
gen fprod_crel = (((pc_tilde*be)/(pelec_tilde*bc))^(ll/(ll-1)))*((c/e)^(1/(ll-1)))
gen lnfprod_crel = ln(fprod_crel)

* Recover productivity of all fuels
gen fprod_g = fprod_grel*fprod_e 
gen lnfprod_g = ln(fprod_g)
gen fprod_c = fprod_crel*fprod_e
gen lnfprod_c = ln(fprod_c)
gen fprod_o = fprod_orel*fprod_e
gen lnfprod_o = ln(fprod_o)

* Recover average fuel productivity and fuel productivity relative to average productivity
egen lnfprod_avg = rowmean(lnfprod_o lnfprod_g lnfprod_c lnfprod_e)
gen fprod_avg = exp(lnfprod_avg)
gen lnfprod_grelavg = lnfprod_g-lnfprod_avg
gen fprod_grelavg = exp(lnfprod_grelavg)
gen lnfprod_crelavg = lnfprod_c-lnfprod_avg
gen fprod_crelavg = exp(lnfprod_crelavg)
gen lnfprod_erelavg = lnfprod_e-lnfprod_avg
gen fprod_erelavg = exp(lnfprod_erelavg)
gen lnfprod_orelavg = lnfprod_o-lnfprod_avg
gen fprod_orelavg = exp(lnfprod_erelavg)

* Recover predicted energy price index and input price index
gen pE_pred = ((bo^ll)*((po_tilde/fprod_o)^(1-ll)) + (be^ll)*((pelec_tilde/fprod_e)^(1-ll)))^(1/(1-ll)) if combineF == 12
replace pE_pred = ((bo^ll)*((po_tilde/fprod_o)^(1-ll)) + (be^ll)*((pelec_tilde/fprod_e)^(1-ll)) + (bc^ll)*((pc_tilde/fprod_c)^(1-ll)))^(1/(1-ll)) if combineF == 123
replace pE_pred = ((bo^ll)*((po_tilde/fprod_o)^(1-ll)) + (be^ll)*((pelec_tilde/fprod_e)^(1-ll)) + (bg^ll)*((pg_tilde/fprod_g)^(1-ll)))^(1/(1-ll)) if combineF == 124
replace pE_pred = ((bo^ll)*((po_tilde/fprod_o)^(1-ll)) + (be^ll)*((pelec_tilde/fprod_e)^(1-ll)) + (bc^ll)*((pc_tilde/fprod_c)^(1-ll)) + (bg^ll)*((pg_tilde/fprod_g)^(1-ll)))^(1/(1-ll)) if combineF == 1234

* Fuel productivity per mmBtu of fuel
gen rfprod_g = (bg^(ll/(ll-1)))/fprod_g
gen rfprod_c = (bc^(ll/(ll-1)))/fprod_c
gen rfprod_e = (be^(ll/(ll-1)))/fprod_e
gen rfprod_o = (bo^(ll/(ll-1)))/fprod_o
gen rfprod_qty_g = (bg^(ll/(ll-1)))*fprod_g/g_gmean
gen rfprod_qty_c = (bc^(ll/(ll-1)))*fprod_c/c_gmean
gen rfprod_qty_e = (be^(ll/(ll-1)))*fprod_e/elec_gmean
gen rfprod_qty_o = (bo^(ll/(ll-1)))*fprod_o/o_gmean
gen lnrfprod_qty_g = log(rfprod_qty_g)
gen lnrfprod_qty_c = log(rfprod_qty_c)
gen lnrfprod_qty_o = log(rfprod_qty_o)
gen lnrfprod_qty_e = log(rfprod_qty_e)

* Fuel Productivity per U.S. dollar invested
gen rfprod_dollar_g = rfprod_qty_g/(pg*1000000)
gen rfprod_dollar_c = rfprod_qty_c/(pc*1000000)
gen rfprod_dollar_e = rfprod_qty_e/(pelec*1000000)
gen rfprod_dollar_o = rfprod_qty_o/(po*1000000)
gen lnrfprod_dollar_g = log(rfprod_dollar_g)
gen lnrfprod_dollar_c = log(rfprod_dollar_c)
gen lnrfprod_dollar_e = log(rfprod_dollar_e)
gen lnrfprod_dollar_o = log(rfprod_dollar_o)

* Return (Revenue in dollars) per mmbtu of each fuel
gen Ecesterm = bo*((fprod_o*o)^((ll-1)/ll)) + be*((fprod_e*e)^((ll-1)/ll)) if combineF == 12
replace Ecesterm = bo*((fprod_o*o)^((ll-1)/ll)) + be*((fprod_e*e)^((ll-1)/ll)) + bc*((fprod_c*c)^((ll-1)/ll)) if combineF == 123
replace Ecesterm = bo*((fprod_o*o)^((ll-1)/ll)) + be*((fprod_e*e)^((ll-1)/ll)) + bg*((fprod_g*g)^((ll-1)/ll)) if combineF == 124
replace Ecesterm = bo*((fprod_o*o)^((ll-1)/ll)) + be*((fprod_e*e)^((ll-1)/ll)) + bc*((fprod_c*c)^((ll-1)/ll)) + bg*((fprod_g*g)^((ll-1)/ll))  if combineF == 1234
gen outprod_mmbtu_g = (Ecesterm^(1/(ll-1)))*(g^(-1/ll))*(rfprod_g^((ll-1)/ll))/g_gmean
gen outprod_mmbtu_c = (Ecesterm^(1/(ll-1)))*(c^(-1/ll))*(rfprod_c^((ll-1)/ll))/c_gmean
gen outprod_mmbtu_o = (Ecesterm^(1/(ll-1)))*(o^(-1/ll))*(rfprod_o^((ll-1)/ll))/o_gmean
gen outprod_mmbtu_e = (Ecesterm^(1/(ll-1)))*(e^(-1/ll))*(rfprod_e^((ll-1)/ll))/elec_gmean
gen Ycesterm = al*(L^((sig-1)/sig)) + ak*(K^((sig-1)/sig)) + ae*(E^((sig-1)/sig)) + am*(M_struc^((sig-1)/sig))
gen outrev_param1 = (1+(rho*(theta-1)))/((theta-1)*rho)
gen outrev_part1 = ((exp(gam)/N)^(1/rho))*(pout_agg^(outrev_param1))*((rho-1)/rho)*(Yqty^(-1/rho))*z*Ygmean*(Ycesterm^(1/(sig-1)))*be*(E^(-1/sig))
gen outrev_mmbtu_g = outrev_part1*outprod_mmbtu_g
replace outrev_mmbtu_g = . if gas == 0
gen outrev_mmbtu_c = outrev_part1*outprod_mmbtu_c
replace outrev_mmbtu_c = . if coal == 0
gen outrev_mmbtu_o = outrev_part1*outprod_mmbtu_o
gen outrev_mmbtu_e = outrev_part1*outprod_mmbtu_e
foreach f in o g c e {
	replace outrev_mmbtu_`f' = outrev_mmbtu_`f'*1000000
}

*******************************************************************
***	6. Graph: distribution of fuel productivity across fuel sets 
*******************************************************************

* Distribution of fuel productivity per mmBtu across fuel sets
// preserve 
// 	* Keep plants I use to estimate production function
// 	drop if LcombineF == . & FcombineF == .
// 	keep IDnum year combineF rfprod_dollar_o rfprod_dollar_e lnrfprod_dollar_o lnrfprod_dollar_e rfprod_qty_o rfprod_qty_e lnrfprod_qty_o lnrfprod_qty_e FcombineF ///
// 			rfprod_dollar_g rfprod_dollar_c lnrfprod_dollar_g lnrfprod_dollar_c rfprod_qty_g rfprod_qty_c lnrfprod_qty_g lnrfprod_qty_c 
//			
// 	* Demean productivity
// 	reg lnrfprod_qty_e 
// 	predict lnrfprod_avg, xb
// 	foreach f in e o g c {
// 		replace lnrfprod_qty_`f' = lnrfprod_qty_`f' - lnrfprod_avg
// 	}
// 	* Organize data for graph
// 	rename lnrfprod_qty_e lnfprod1
// 	rename lnrfprod_qty_o lnfprod2
// 	rename lnrfprod_qty_g lnfprod3
// 	rename lnrfprod_qty_c lnfprod4
// 	reshape long lnfprod, i(IDnum year) j(fuel)
// 	collapse (mean) y = lnfprod (semean) se_y = lnfprod, by(fuel combineF)
// 	gen combineF_sort = 1 if combineF == 12
// 	replace combineF_sort = 2 if combineF == 124
// 	replace combineF_sort = 3 if combineF == 123
// 	replace combineF_sort = 4 if combineF == 1234
// 	sort fuel combineF_sort 
// 	gen x = _n
// 	gen yu = y + 1.96*se_y
// 	gen yl = y - 1.96*se_y
// 	* Create figure
// 	twoway ///
// 	(rcap yl yu x, vert lcolor(gray)) /// code for 95% CI
// 	(scatter y x if fuel == 1, mcolor(cranberry) msymbol(circle) msize(5pt)) /// 
// 	(scatter y x if fuel == 2, mcolor(navy) msymbol(diamond) msize(5pt)) ///
// 	(scatter y x if fuel == 3, mcolor(cyan) msymbol(square) msize(5pt)) ///
// 	(scatter y x if fuel == 4, mcolor(olive) msymbol(triangle) msize(5pt)) ///
// 	, legend(row(1) order(2 "Electricity" 3 "Oil" 4 "Natural Gas" 5 "Coal") pos(10) ring(0) size(12pt))  ///
// 	xlabel(1 "oe" 2 "oge" 3 "oce" 4 "ogce" 5 "oe" 6 "oge" 7 "oce" 8 "ogce" 9 "oe" 10 "oge" 11 "oce" 12 "ogce" 13 "oe" 14 "oge" 15 "oce" 16 "ogce", ///
// 	angle(0) noticks labsize(11pt)) xline(4.5, lpattern(dash) lcolor(gray)) xline(8.5, lpattern(dash) lcolor(gray)) xline(12.5, lpattern(dash) lcolor(gray)) ///
// 	xtitle("Fuel set",size(12pt)) ytitle("(log) fuel productivity",size(12pt)) graphregion(color(white))
// 	graph export "output/graphs/EnergyPFE/fprod_qty_allfuel_bysetf-steel.pdf", replace
// restore

* Distribution of fuel productivity per dollar spent across fuel sets
// preserve 
// 	* Keep plants I use to estimate production function
// 	drop if LcombineF == . & FcombineF == .
// 	keep IDnum year combineF rfprod_dollar_o rfprod_dollar_e lnrfprod_dollar_o lnrfprod_dollar_e rfprod_qty_o rfprod_qty_e lnrfprod_qty_o lnrfprod_qty_e FcombineF ///
// 			rfprod_dollar_g rfprod_dollar_c lnrfprod_dollar_g lnrfprod_dollar_c rfprod_qty_g rfprod_qty_c lnrfprod_qty_g lnrfprod_qty_c 
//			
// 	reg lnrfprod_qty_e 
// 	predict lnrfprod_avg, xb
// 	foreach f in e o g c {
// 		replace lnrfprod_dollar_`f' = lnrfprod_dollar_`f' - lnrfprod_avg
// 	}
// 	* Organize data for graph
// 	rename lnrfprod_dollar_e lnfprod1
// 	rename lnrfprod_dollar_o lnfprod2
// 	rename lnrfprod_dollar_g lnfprod3
// 	rename lnrfprod_dollar_c lnfprod4
// 	reshape long lnfprod, i(IDnum year) j(fuel)
// 	collapse (mean) y = lnfprod (semean) se_y = lnfprod, by(fuel combineF)
// 	gen combineF_sort = 1 if combineF == 12
// 	replace combineF_sort = 2 if combineF == 124
// 	replace combineF_sort = 3 if combineF == 123
// 	replace combineF_sort = 4 if combineF == 1234
// 	sort fuel combineF_sort 
// 	gen x = _n
// 	gen yu = y + 1.96*se_y
// 	gen yl = y - 1.96*se_y
// 	* Create figure
// 	twoway ///
// 	(rcap yl yu x, vert lcolor(gray)) /// code for 95% CI
// 	(scatter y x if fuel == 1, mcolor(cranberry) msymbol(circle) msize(5pt)) /// 
// 	(scatter y x if fuel == 2, mcolor(navy) msymbol(diamond) msize(5pt)) ///
// 	(scatter y x if fuel == 3, mcolor(cyan) msymbol(square) msize(5pt)) ///
// 	(scatter y x if fuel == 4, mcolor(olive) msymbol(triangle) msize(5pt)) ///
// 	, legend(row(1) order(2 "Electricity" 3 "Oil" 4 "Natural Gas" 5 "Coal") pos(10) ring(0) size(12pt))  ///
// 	xlabel(1 "oe" 2 "oge" 3 "oce" 4 "ogce" 5 "oe" 6 "oge" 7 "oce" 8 "ogce" 9 "oe" 10 "oge" 11 "oce" 12 "ogce" 13 "oe" 14 "oge" 15 "oce" 16 "ogce", ///
// 	angle(0) noticks labsize(11pt)) xline(4.5, lpattern(dash) lcolor(gray)) xline(8.5, lpattern(dash) lcolor(gray)) xline(12.5, lpattern(dash) lcolor(gray)) ///
// 	xtitle("Fuel set",size(12pt)) ytitle("(log) fuel productivity",size(12pt)) graphregion(color(white))
// 	graph export "output/graphs/EnergyPFE/fprod_dollar_allfuel_bysetf-steel.pdf", replace
// restore

*******************************************************************
***	7. Table: distribution of return to one mmBtu of each fuel
*******************************************************************
// estpost tabstat outrev_mmbtu_g outrev_mmbtu_c outrev_mmbtu_o outrev_mmbtu_e, statistics(p10 p25 p50 p75 p90)
// esttab . using "output/Tables/EnergyPFE/fprod_rev-steel.tex", cells("outrev_mmbtu_g outrev_mmbtu_c outrev_mmbtu_o outrev_mmbtu_e") noobs nomtitle nonumber replace


************************************************************************************************
***	7. Graph: relationship between coal prices and coal productivity (evidence of coal grades)
************************************************************************************************

************************************************************************************************
***	7. Graph: Effect of 2014 oil/gas crash on output --> validation of main result
************************************************************************************************



preserve 

capture drop nyear
egen nyear = total(inrange(year, 2009, 2016)), by(IDnum)
keep if nyear == 8
reg LogYqty i.year
predict lny_res, res

gen Dcoal = 0
replace Dcoal = 1 if coal > 0 & gas == 0
gen Dgas = 0 
replace Dgas = 1 if gas > 0 & coal == 0
collapse (mean) lny_res, by(year Dcoal Dgas)

graph twoway (connected lny_res year if Dcoal == 1) (connected lny_res year if Dgas == 1), legend(label(1 "Coal users") label(2 "Gas users"))


restore 


************************************************************
***	8. Estimation of state transition (steel manufacturing) 			
************************************************************

* Administrative regions
gen region = 0
lab def region_lab 1 "Northern" 2 "Central" 3 "Eastern" 4 "North-East" 5 "Western" 6 "Southern", replace
replace region = 1 if StateCode == 1 | StateCode == 2 | StateCode == 3 | StateCode == 4 | StateCode == 6 | StateCode == 7 | StateCode == 8 
replace region = 2 if StateCode == 5 | StateCode == 9 | StateCode == 22 | StateCode == 23
replace region = 3 if StateCode == 10 | StateCode == 19  | StateCode == 20 | StateCode == 21
replace region = 4 if StateCode == 14 | StateCode == 16 | StateCode == 17 | StateCode == 18
replace region = 5 if StateCode == 24 | StateCode == 25 | StateCode == 26 | StateCode == 27 | StateCode == 32
replace region = 6 if StateCode == 28 | StateCode == 29 | StateCode == 30 | StateCode == 33 | StateCode == 34 | StateCode == 36
lab val region region_lab

* Define if plants are connected to natural gas pipeline
drop if distnum == .
replace Connection = 3 if Connection == .
replace Pipeline = 99 if Pipeline == .
replace Zone = 99 if Zone == .
* Keep firms that I observe at least twice subsequently
order IDnum year LcombineF combineF FcombineF, last
drop if LcombineF == . & FcombineF == .


*** Non-selected state variables (with fixed effects) ***

* Winsoring state variables at 1st and 99th percentile to estimate markovian process
preserve 
	* 1. Hicks-neutral productivity (AR(1) disturbance only)
	gen lnz11 = lnz
	su lnz11, det
	replace lnz11 = . if lnz11 < r(p1) | lnz11> r(p99) 
	reg lnz11 i.year
	forvalues t = 10/16 {
		scalar b`t'_z = _b[20`t'.year]
	}
	scalar cons_z = _b[_cons]
	predict res_lnz, res
	xtdpdsys res_lnz, lags(1) nocons two
	predict res_res_lnz, e
	scalar rho_z = _b[L1.res_lnz]
	reg res_res_lnz
	gen cons_lnz = _b[_cons]
	replace res_res_lnz = res_res_lnz-cons_lnz
	su res_res_lnz
	scalar sig_z = r(sd)^2
	mat param_z = (rho_z,sig_z,cons_z,b10_z,b11_z,b12_z,b13_z,b14_z,b15_z,b16_z,0,0)
	gen fe_lnz = 0
	
	* 2. Productivity of oil (same as before)
	gen lnfprod_o1 = lnfprod_o
	su lnfprod_o1, det
	replace lnfprod_o1 = . if lnfprod_o1 < r(p5) | lnfprod_o1 > r(p95) 
	reg lnfprod_o1 i.year
	forvalues t = 10/16 {
		scalar b`t'_o = _b[20`t'.year]
	}
	scalar cons_o = _b[_cons]
	predict res_prodo, res
	xtdpdsys res_prodo, lags(1) nocons two
	scalar rho_o = _b[L1.res_prodo]
	predict res_res_prodo, e
	su res_res_prodo
	scalar sig_o = r(sd)^2
	mat param_o = (rho_o,sig_o,cons_o,b10_o,b11_o,b12_o,b13_o,b14_o,b15_o,b16_o,0,0)
	gen fe_prodo = 0

	* 3. Price of oil (normalize with right units)
	forvalues t = 2009/2016 {
		su po_tilde if year == `t'
		scalar po_`t' = r(mean)
	}
	mat po_tilde = (po_2009,po_2010,po_2011,po_2012,po_2013,po_2014,po_2015,po_2016,0,0,0,0)

	* 4. Produtivity of electricity (same as before)
	gen lnfprod_e1 = lnfprod_e
	su lnfprod_e1, det
	replace lnfprod_e1 = . if lnfprod_e1 < r(p5) | lnfprod_e1 > r(p95) 
	reg lnfprod_e1 i.year
	forvalues t = 10/16 {
		scalar b`t'_prode = _b[20`t'.year]
	}
	scalar cons_prode = _b[_cons]
	predict res_prode, res
	xtdpdsys res_prode, lags(1) nocons two
	scalar rho_prode = _b[L1.res_prode]
	predict res_res_prode, e
	su res_res_prode
	scalar sig_prode = r(sd)^2
	gen fe_prode = 0
	
	* 5. Price of electricity (no state fixed effects)
	gen lnpelec_tilde1 = lnpelec_tilde
	su lnpelec_tilde1, det
	replace lnpelec_tilde1 = . if lnpelec_tilde1 < r(p1) | lnpelec_tilde1 > r(p99)
	reg lnpelec_tilde1 i.year
	forvalues t = 10/16 {
		scalar b`t'_pe = _b[20`t'.year]
	}
	scalar cons_pe = _b[_cons]
	predict res_pe, res
	xtdpdsys res_pe, lags(1) nocons two
	scalar rho_pe = _b[L1.res_pe]
	predict res_res_pe, e
	su res_res_pe
	scalar sig_pe = r(sd)^2
	gen fe_pe = 0

	* Joint process for shocks to price and productivity of electricity
	correlate res_res_pe res_res_prode, covariance
	scalar cov_e = r(cov_12)
	mat param_e = (rho_prode,sig_prode,cons_prode,b10_prode,b11_prode,b12_prode,b13_prode,b14_prode,b15_prode,b16_prode,cov_e,0 ///
					\rho_pe,sig_pe,cons_pe,b10_pe,b11_pe,b12_pe,b13_pe,b14_pe,b15_pe,b16_pe,cov_e,0)

	* 6. Price of materials
	sort IDnum year
	gen logPm1 = logPm
	su logPm1, det
	replace logPm1 = . if logPm1 < r(p1) | logPm1 > r(p99) 
	reg logPm1 i.year
	scalar cons_pm = _b[_cons]
	forvalues t = 10/16 {
		scalar b`t'_pm = _b[20`t'.year]
	}
	predict res_pm, res
	xtdpdsys res_pm, lags(1) nocons two
	scalar rho_pm = _b[L1.res_pm]
	predict res_res_pm, e
	su res_res_pm
	scalar sig_pm = r(sd)^2
	mat param_m = (rho_pm,sig_pm,cons_pm,b10_pm,b11_pm,b12_pm,b13_pm,b14_pm,b15_pm,b16_pm,0,0)
	gen fe_pm = 0

	mat pm = J(12,1,0)
	forvalues t = 2009/2016 {
		su Pm if year == `t'
		mat pm[`t'-2008,1] = r(mean)
	}

	* Covariance matrix between state variables
	correlate res_res_lnz res_res_pm res_res_pe res_res_prode res_res_prodo, covariance
	mat param_cov = r(C)

	* 7. wages
	mat wage = J(12,1,0)
	forvalues t = 2009/2016 {
		su w_tilde if year == `t'
		mat wage[`t'-2008,1] = r(mean)
	}
	* 8. rental rate of capital (0.05% interest rate, 40% deppreciation)
	mat rk = J(12,1,0)
	forvalues t = 2009/2016 {
		su rk_tilde if year == `t'
		mat rk[`t'-2008,1] = r(mean)
	}

	* 9. Geometric mean of output (normalization)
	mat Ygmean = J(12,1,0)
	forvalues t = 2009/2016 {
		su Ygmean if year == `t'
		mat Ygmean[`t'-2008,1] = r(mean)
	}

	*** Selected state variables (including comparative advantages for plants using gas and/or coal) ***

	** 1. Natural Gas 
	* Price
	reg lnpg_tilde i.year
	gen state_pg = 0
	scalar cons_pg = _b[_cons]
	forvalues t = 10/16 {
		scalar b`t'_pg = _b[20`t'.year]
	}
	predict res_pg, res
	su res_pg
	scalar sig_pg = r(sd)^2
	* Productivity
	gen lnfprod_g1 = lnfprod_g
	su lnfprod_g1, det
	replace lnfprod_g1 = . if lnfprod_g1 < r(p5) | lnfprod_g1 > r(p95) 
	xtreg lnfprod_g1 i.year, re
	scalar cons_prodg = _b[_cons]
	forvalues t = 10/16 {
		scalar b`t'_prodg = _b[20`t'.year]
	}
	predict lnfprod_g_re, u
	predict res_prodg, e
	su res_prodg
	scalar sig_prodg = r(sd)^2
	correlate res_prodg res_pg, cov
	scalar cov_g = r(cov_12)
	* Price over productivity
	gen lnpg_prodg = lnpg_tilde-lnfprod_g
	gen res_lnpg_prodg = res_pg-res_prodg
	* Comparative advantage (random effect)
	su lnfprod_g_re
	scalar mug_re = r(mean)
	scalar sig_g_re = r(sd)^2
	mat param_g = (sig_prodg,cons_prodg,b10_prodg,b11_prodg,b12_prodg,b13_prodg,b14_prodg,b15_prodg,b16_prodg,cov_g,mug_re,sig_g_re ///
				\sig_pg,cons_pg,b10_pg,b11_pg,b12_pg,b13_pg,b14_pg,b15_pg,b16_pg,cov_g,0,0)
				
	* Covariance matrix between gas and all non-selected state variable
	correlate res_prodg res_pg res_res_lnz res_res_pm res_res_pe res_res_prode res_res_prodo, covariance
	mat param_cov_g = r(C)

	** 2. Coal
	* Price
	reg lnpc_tilde i.year
	gen state_pc = 0
	scalar cons_pc = _b[_cons]
	forvalues t = 10/16 {
		scalar b`t'_pc = _b[20`t'.year]
	}
	predict res_pc, res
	su res_pc
	scalar sig_pc = r(sd)^2
	* Productivity
	gen lnfprod_c1 = lnfprod_c
	su lnfprod_c1, det
	replace lnfprod_c1 = . if lnfprod_c1 < r(p10) | lnfprod_c1 > r(p90) 
	xtreg lnfprod_c1 i.year, re
	scalar cons_prodc = _b[_cons]
	forvalues t = 10/16 {
		scalar b`t'_prodc = _b[20`t'.year]
	}
	predict lnfprod_c_re, u
	predict res_prodc, e
	su res_prodc
	scalar sig_prodc = r(sd)^2
	correlate res_prodc res_pc, cov
	scalar cov_c = r(cov_12)
	* Price over productivity (residual: state variable)
	gen lnpc_prodc = lnpc_tilde-lnfprod_c
	gen res_lnpc_prodc = res_pc-res_prodc
	* random effects
	su lnfprod_c_re
	scalar muc_re = r(mean)
	scalar sig_c_re = r(sd)^2
	mat param_c = (sig_prodc,cons_prodc,b10_prodc,b11_prodc,b12_prodc,b13_prodc,b14_prodc,b15_prodc,b16_prodc,cov_c,muc_re,sig_c_re ///
				\sig_pc,cons_pc,b10_pc,b11_pc,b12_pc,b13_pc,b14_pc,b15_pc,b16_pc,cov_c,0,0)			
	* Covariance matrix between coal and all non-selected state variable
	correlate res_prodc res_pc res_res_lnz res_res_pm res_res_pe res_res_prode res_res_prodo, covariance
	mat param_cov_c = r(C)

	* Covariance matrix between gas and coal 
	correlate res_prodg res_pg res_prodc res_pc, covariance
	mat param_cov_gc = r(C)
restore

*** Store starting value of all state variables for each plant ***

* 1. Hicks-neutral productivity (AR(1) disturbance only)
reg lnz i.year
predict res_lnz, res
gen fe_lnz = 0
xtdpdsys res_lnz, lags(1) nocons two
predict res_res_lnz, e
reg res_res_lnz
gen cons_lnz = _b[_cons]
replace res_res_lnz = res_res_lnz-cons_lnz

* 2. Productivity of oil (same as before)
reg lnfprod_o i.year
predict res_prodo, res
gen fe_prodo = 0
xtdpdsys res_prodo, lags(1) nocons two
predict res_res_prodo, e
reg res_res_prodo
gen cons_prodo = _b[_cons]
replace res_res_prodo = res_res_prodo - cons_prodo

* 4. Produtivity of electricity (same as before)
reg lnfprod_e i.year
predict res_prode, res
gen fe_prode = 0
xtdpdsys res_prode, lags(1) nocons two
predict res_res_prode, e
reg res_res_prode
gen cons_prode = _b[_cons]
replace res_res_prode = res_res_prode - cons_prode

* 5. Price of electricity (no state fixed effects)
reg lnpelec_tilde i.year
predict res_pe, res
gen fe_pe = 0
xtdpdsys res_pe, lags(1) nocons two
predict res_res_pe, e
reg res_res_pe
gen cons_pe = _b[_cons]
replace res_res_pe = res_res_pe - cons_pe

* 6. Price of materials (AR(1) disturbance only, may want to assume it is contant)
sort IDnum year
reg logPm i.year
predict res_pm, res
gen fe_pm = 0
xtdpdsys res_pm, lags(1) nocons two
predict res_res_pm, e
reg res_res_pm
gen cons_pm = _b[_cons]
replace res_res_pm = res_res_pm - cons_pm

*** Selected state variables (including fixed effets) ***

* 1. Natural Gas
* Price
reg lnpg_tilde i.year
gen state_pg = 0
scalar cons_pg = _b[_cons]
forvalues t = 10/16 {
	scalar b`t'_pg = _b[20`t'.year]
	*replace state_pg = state_pg - cons_pg - b`t'_pg if year == 20`t'
}
predict res_pg, res
su res_pg
* Productivity
xtreg lnfprod_g i.year, re
predict lnfprod_g_re, u
predict res_prodg, e
su res_prodg
* Price over productivity (residual: state variable)
gen lnpg_prodg = lnpg_tilde-lnfprod_g
gen res_lnpg_prodg = res_pg-res_prodg
			
* 2. Coal
* Price
reg lnpc_tilde i.year
gen state_pc = 0
scalar cons_pc = _b[_cons]
forvalues t = 10/16 {
	scalar b`t'_pc = _b[20`t'.year]
	*replace state_pc = state_pc - cons_pc - b`t'_pc if year == 20`t'
}
predict res_pc, res
* Productivity
xtreg lnfprod_c i.year, re
predict lnfprod_c_re, u
predict res_prodc, e
su res_prodc
* Price over productivity (residual: state variable)
gen lnpc_prodc = lnpc_tilde-lnfprod_c
gen res_lnpc_prodc = res_pc-res_prodc

*********************************************************************************************
***	9. Export Data for dynamic estimation (Nested Fixed point method) - Steel manufacturing 
*********************************************************************************************

* Redefine connection wrt pipeline (beware, I am losing data)
drop if distnum == .
replace Connection = 3 if Connection == .
replace Pipeline = 99 if Pipeline == .
replace Zone = 99 if Zone == .
* Keep firms that I observe at least twice subsequently
order IDnum year LcombineF combineF FcombineF, last
drop if LcombineF == . & FcombineF == .
drop if FcombineF == .

*** Aggregate price index and predicted model objects ***
* Fixed point to find price index
egen pout_agg0 = total((pout)^(1-rho)), by(year)
replace pout_agg0 = (pout_agg0/N)^(1/(1-rho))
tab pout_agg0
gen lnpout_agg0 = ln(pout_agg0)
egen dist0 = sum((pout_agg0-pout_agg)^2), by(year)
gen gam0 = demand_cons- ((1+rho*(theta-1))/(theta-1))*lnpout_agg0
gen Ypred0 = (((z*Ygmean)^(1/eta))*((rho-1)/rho)*eta*((exp(gam0)/N)^(1/rho))*((pout_agg0^pterm_y1)/pinput))^pterm_y2
gen pout_pred0 = ((exp(gam0)/(N*Ypred0))^(1/rho))*(pout_agg0^pterm_y1)
local dist = 100
local tol = 1.0e-15
while `dist' > `tol' {
	egen pout_agg1 = total((pout_pred0)^(1-rho)), by(year)
	replace pout_agg1 = (pout_agg1/N)^(1/(1-rho))
	tab pout_agg1
	egen dist1 = sum((pout_agg1-pout_agg0)^2), by(year)	
	forvalues yr = 2010/2015 {
		su dist1 if year == `yr'
		local dist1_`yr' = r(mean)
	}
	gen distt1 = `dist1_2010' + `dist1_2011' + `dist1_2012' + `dist1_2013' + `dist1_2014' + `dist1_2015'
	replace distt1 = distt1^0.5
	su distt1 
	local dist = r(mean)
	
	gen lnpout_agg1 = ln(pout_agg1)
	gen gam1 = demand_cons- ((1+rho*(theta-1))/(theta-1))*lnpout_agg1
	gen Ypred1 = (((z*Ygmean)^(1/eta))*((rho-1)/rho)*eta*((exp(gam1)/N)^(1/rho))*((pout_agg1^pterm_y1)/pinput))^pterm_y2
	gen pout_pred1 = ((exp(gam1)/(N*Ypred1))^(1/rho))*(pout_agg1^pterm_y1)
	
	replace pout_agg0 = pout_agg1
	replace lnpout_agg0 = lnpout_agg1
	replace gam0 = gam1
	replace Ypred0 = Ypred1
	replace pout_pred0 = pout_pred1
	drop pout_agg1 lnpout_agg1 gam1 Ypred1 pout_pred1 dist1 distt1
}
rename pout_agg0 pout_agg_struc 
rename lnpout_agg0 lnpout_agg_struc
rename Ypred0 Ypred_struc
rename gam0 gam_struc
rename pout_pred0 poutpred_struc
* Predicted revenue, input quantity and profit
gen Yspend_struc = Ypred_struc*poutpred_struc
gen Lpred_struc = ((Ypred_struc/(Ygmean*z))^(1/eta))*((al/w_tilde)^sig)*(pinput^sig) 
gen Kpred_struc = ((Ypred_struc/(Ygmean*z))^(1/eta))*((ak/rk_tilde)^sig)*(pinput^sig) 
gen Mpred_struc = ((Ypred_struc/(Ygmean*z))^(1/eta))*((am/Pm)^sig)*(pinput^sig) 
gen Epred_struc = ((Ypred_struc/(Ygmean*z))^(1/eta))*((ae/Pe)^sig)*(pinput^sig) 
capture drop profit_pred
gen profit_pred = Yspend_struc - (Mpred_struc*Pm + Lpred_struc*w_tilde + Epred_struc*Pe + Kpred_struc*rk_tilde)

* 9. Output price matrix 
mat pout_agg = J(12,1,0)
forvalues t = 2010/2015 {
	su pout_agg_struc if year == `t'
	mat pout_agg[`t'-2008,1] = r(mean)
}

* Predicted fuel quantities
gen coal_pred = c_gmean*Epred_struc*((bc/pc_tilde)^ll)*(pE^ll)*(fprod_c^(ll-1))
gen gas_pred = g_gmean*Epred_struc*((bg/pg_tilde)^ll)*(pE^ll)*(fprod_g^(ll-1))
gen oil_pred = o_gmean*Epred_struc*((bo/po_tilde)^ll)*(pE^ll)*(fprod_o^(ll-1))
gen elec_pred = elec_gmean*Epred_struc*((be/pelec_tilde)^ll)*(pE^ll)*(fprod_e^(ll-1))

***  Store parameter values ***
* Production function parameters 
foreach p in sig eta ak al am ae ll bo bg bc be gamma_elec gamma_gas gamma_coal gamma_oil rho theta {
	su `p'
	scalar `p' = r(mean)
}
mat param_pf = (sig,eta,ak,al,am,ae,ll,bo,bg,bc,be,0)
* Demand parameters (demand elasticity, elasticity of outisde good and year fixed effects)
forvalues yr = 2009/2016 {
	su gam_struc if year == `yr'
	scalar d`yr' = r(mean)
}
mat param_demand = (rho,theta,0,d2010,d2011,d2012,d2013,d2014,d2015,0,0,0)
* Number of plants by year
forvalues yr = 2010/2015 {
	su N if year == `yr'
	scalar N`yr' = r(mean)
}
mat Nplants = (0,N2010,N2011,N2012,N2013,N2014,N2015,0,0,0,0,0)

* Emission parameters 
mat param_emission = (gamma_elec,gamma_gas,gamma_coal,gamma_oil,0,0,0,0,0,0,0,0)
foreach p in o_gmean g_gmean c_gmean elec_gmean {
	su `p'
	scalar `p' = r(mean)
}
* Geometric mean of fuel quantities 
mat param_fgmean = (o_gmean,g_gmean,c_gmean,elec_gmean,0,0,0,0,0,0,0,0)
*	row 1: production function parameters: sig eta ak al am ae ll bo bg bc be
*	row 2: hicks-neutral z
*	row 3: productivity of electricity
*	row 4: price of electricity
*	row 5: productivity of oil
*	row 6: price of oil
*	row 7: productivity of natural gas
*	row 8: price of natural gas
*	row 9: productivity of coal
*	row 10: price of coal
*	row 11: price of materials (process)
*	row 12: price of materials (prices)
*	row 13: wages
*	row 14: rental rate of capital
*	row 15: aggregate output price (market clearing from model)
*	row 16: Ygmean
*	row 17: emission factors 
*   row 18: geometric mean of fuel quantities 
* 	row 19: demand parameters: rho, theta, d09, d10, d11, d12, d13, d14, d15, d16
* 	row 20: Number of plants by year (2010 to 2015)
mat param_all = (param_pf\param_z\param_e\param_o\po_tilde\param_g\param_c\param_m\pm'\wage'\rk'\pout_agg'\Ygmean'\param_emission\param_fgmean\param_demand\Nplants) 

mat2txt, mat(param_all) saving("Data/Dynamics/param_all.txt") replace
mat2txt, mat(param_cov) saving("Data/Dynamics/param_cov.txt") replace
mat2txt, mat(param_cov_g) saving("Data/Dynamics/param_cov_g.txt") replace
mat2txt, mat(param_cov_c) saving("Data/Dynamics/param_cov_c.txt") replace
mat2txt, mat(param_cov_gc) saving("Data/Dynamics/param_cov_gc.txt") replace
preserve

* drop first year (no initial condition) and last year (no next period fuel set)
drop if year == 2009 | year == 2016
capture gen D2010=0
replace D2010=1 if year == 2010
capture gen D2011=0
replace D2011=1 if year == 2011
capture gen D2012=0
replace D2012=1 if year == 2012
capture gen D2013=0
replace D2013=1 if year == 2013
capture gen D2014=0
replace D2014=1 if year == 2014
capture gen D2015=0
replace D2015=1 if year == 2015
gen Dr1 = 0
replace Dr1 = 1 if region == 1
gen Dr2 = 0
replace Dr2 = 1 if region == 2
gen Dr3 = 0
replace Dr3 = 1 if region == 3
gen Dr4 = 0
replace Dr4 = 1 if region == 4
gen Dr5 = 0
replace Dr5 = 1 if region == 5
gen Dr6 = 0
replace Dr6 = 1 if region == 6

* keep relevant variables
xtset IDnum year
sort year IDnum

keep IDnum year combineF FcombineF lnrfprod_qty_e fprod_g fprod_c fprod_e fprod_o lnz lnfprod_e lnpelec_tilde lnpo_tilde lnfprod_o lnfprod_g lnpg_tilde lnfprod_c lnpc_tilde logPm Pm pE res_prodg res_prodc lnpg_prodg lnpc_prodc res_lnpg_prodg res_lnpc_prodc res_lnz res_pm res_prode res_prodo res_pe D2010 D2011 D2012 D2013 D2014 D2015 Dr1 Dr2 Dr3 Dr4 Dr5 Dr6 res_res_lnz res_res_prodo res_res_prode res_res_pe res_res_pm Connection Pipeline Zone lnfprod_c_re lnfprod_g_re oil gas coal elec rk_tilde w_tilde res_pg res_pc Yqty Yspend_nominal E Espend_nominal M Mspend_nominal L Lspend_nominal K Kspend_nominal pinput pout pout_pred profit_pred profit_emp N demand_cons pout_agg_struc fe_lnz fe_prode fe_prodo fe_pm fe_pe state_pg state_pc cons_lnz cons_prode cons_prodo cons_pe cons_pm bo bg bc be g_gmean c_gmean elec_gmean o_gmean ll
* Export data to csv
export delimited using "Data/Dynamics/MainData_wPipeline-Steel.csv", replace
restore



*****************************************************************
***	8. Estimation of CCP - To know which state variable matters		
*****************************************************************

*************************************************************************************

drop if FcombineF == .
drop if year == 2009 | year == 2016

gen lnprofit = log(profit_full2)

gen lnz2 = lnz^2 
gen lnfprod_e2 = lnfprod_e^2
gen lnfprod_o2 = lnfprod_o^2
gen lnpelec_tilde2 = lnpelec_tilde^2
gen lnpE2 = lnpE^2
gen logPm2 = logPm^2

*mlogit FcombineF i.combineF lnz lnfprod_e lnfprod_o lnpelec_tilde lnpE logPm lnz2 lnfprod_e2 lnfprod_o2 lnpelec_tilde2 lnpE2 logPm2

gen lnL = log(Lqty)
gen lnK = log(Kqty)


mlogit FcombineF i.combineF lnz lnpelec_tilde lnfprod_e lnfprod_o lnpE logPm i.year i.StateCode, baseoutcome(12)











