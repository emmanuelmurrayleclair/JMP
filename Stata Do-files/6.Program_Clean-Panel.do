*** CODEFILE 6 ***

**** This file cleans the dataset to be used for production function estimation ***
* it creates and cleans all remaining variables needed for estimation


global ASIpaneldir Data/Panel_Data/Clean_data

* Import data and set panel
use Data/Panel_Data/Clean_data/ASI_PanelClean-allind_a, clear
egen IDnum = group(ID)
xtset IDnum year

* keep manufacturing industries only
drop if nic08_3d >= 351
gen nic08_5d = IndCodeReturn

******************************************************************************************************************************************
*** 0. Merge state-level electricity generation capacity by source
		* Source: Executive Summary of Power Sector - January 2008
		* Installed capacity (in MW) of power utilities by states including allocated shares in joint and central sector utilities
		* Retrived from: https://cea.nic.in/monthly-quarterly-archive-reports/?lang=en
******************************************************************************************************************************************

preserve
* Import state-level electricity generation capacity by source - 2008
import excel "Data/External/ElectricityGeneration_byfuel-State_2008.xlsx", sheet("Sheet1") firstrow clear
// gen ElecGenTot = Coal+Gas+Oil 
gen ElecGenTot = Coal+Gas+Oil+Nuclear+Hydro+Other
gen sElec_coal = Coal/ElecGenTot
gen sElec_gas = Gas/ElecGenTot
gen sElec_oil = Oil/ElecGenTot
drop Coal Gas Oil Nuclear Hydro Other

tempfile elecgen
save `elecgen'
restore 
merge m:1 StateCode using `elecgen' 
drop _merge

****************************************************************************
*** 1. Create Bartik (shift-share) Instrument to Estimate Demand Elasticity
****************************************************************************

* Average annual (log) fuel prices excluding own-state
gen lnpelec = ln(pelecb_mmbtu)
gen lnpelec_avg = .
foreach fuel in oil gas coal {
	gen lnp`fuel' = log(p`fuel'_mmbtu)
	gen lnp`fuel'_avg = .
}
levelsof StateCode, local(state)
local iter_state = 1
foreach j of local state {
	forvalues yr = 2009/2016 {
		* Oil
		qui su lnpoil if StateCode != `j' & year == `yr'
		qui replace lnpoil_avg = r(mean) if StateCode == `j' & year == `yr'
		* Gas
		qui su lnpgas if StateCode != `j' & year == `yr'
		qui replace lnpgas_avg = r(mean) if StateCode == `j' & year == `yr'
		* Coal
		qui su lnpcoal if StateCode != `j' & year == `yr'
		qui replace lnpcoal_avg = r(mean) if StateCode == `j' & year == `yr'
		* Elec
		qui su lnpelec if StateCode != `j' & year == `yr'
		qui replace lnpelec_avg = r(mean) if StateCode == `j' & year == `yr'
	}
}
* Shift-Share (Bartik) Instrument (interaction of annual fuel prices with 2008 fuel share in electricity generation capacity)
gen zoil = lnpoil_avg*sElec_oil
gen zgas = lnpgas_avg*sElec_gas
gen zcoal = lnpcoal_avg*sElec_coal

********************************************************************
*** 2. Prepare Inputs, Output and Emissions
********************************************************************

* Unit conversion factor (rupees to million USD)
local be = 1/82865100 

*** Output ***

gen Yspend_nominal = TotalOutput
gen Yspend_real = TotalOutput_defl
gen pout_index = Yspend_nominal/Yspend_real
* Revenues sums sales across all outputs
egen Yspend_nominal_new = rowtotal(GrossSaleValueOutput1 GrossSaleValueOutput2 GrossSaleValueOutput3 GrossSaleValueOutput4 GrossSaleValueOutput5 ///
								GrossSaleValueOutput6 GrossSaleValueOutput7 GrossSaleValueOutput8 GrossSaleValueOutput9 GrossSaleValueOutput10)
replace Yspend_nominal_new = Yspend_nominal_new*`be'
gen Yspend_real_new = Yspend_nominal_new/pout_index
* Output price (weighted average of all outputs sold)
forvalues p = 1/10 {
	replace GrossSaleValueOutput`p' = GrossSaleValueOutput`p'*`be'
	gen pout`p' = GrossSaleValueOutput`p'/QtySoldOutput`p'
	gen s_output`p' = GrossSaleValueOutput`p'/Yspend_nominal_new
	gen w_pout`p' = pout`p'*s_output`p'
}
egen pout = rowmean(w_pout1 w_pout2 w_pout3 w_pout4 w_pout5 w_pout6 w_pout7 w_pout8 w_pout9 w_pout10)
* Output quantity (Total sales/output price)
gen Yqty = Yspend_nominal_new/pout
keep if Yqty > 0 
// drop if Yqty == .
gen LogYqty = log(Yqty)
gen LogYspend = log(Yspend_nominal_new)
* Geometric mean of output quantity
egen Ygmean = gmean(Yqty), by(nic08_4d)
gen Y = Yqty/Ygmean
gen lnpout = ln(pout)
gen lny = ln(Yqty)

*** Inputs ***

* Geometric mean of energy spending
egen Espend_gmean = gmean(Espend_nominal), by(nic08_4d)

* Labor (deflated - rescaled around geometric mean)
gen Lspend_nominal = TotalEmoluments
gen Lqty = TotalEmoluments_defl
*gen Lqty = PersonsTotal
gen logLspend = log(Lspend_nominal)
* Geometric mean of Labor spending
egen Lgmean = gmean(Lqty), by(nic08_4d)
gen L = Lqty/Lgmean
egen Lspend_gmean = gmean(Lspend_nominal), by(nic08_4d)

* Capital (deflated - rescaled around geometric mean)
gen Kspend_nominal = Capital
gen Kqty = Capital_defl
gen logKspend = log(Kspend_nominal)
* Geometric mean of Capital
egen Kgmean = gmean(Kqty), by(nic08_4d)
gen K = Kqty/Kgmean

* Intermediates (deflated - rescaled around geometric mean)
gen Mspend_nominal = Materials
gen Mqty = Materials_defl
gen logMspend = log(Mspend_nominal)
* Geometric mean of Intermediates
egen Mgmean = gmean(Mqty), by(nic08_4d)
gen M = Mqty/Mgmean
egen Mspend_gmean = gmean(Mspend_nominal), by(nic08_4d)

* Fuel sets and number of fuels
gen F = 4
egen minusF = anycount(TotGas_mmbtu TotCoal_mmbtu TotOil_mmbtu elecb_mmbtu), values(0)
drop if minusF == 4
replace F = F-minusF
gen combineF = 0
replace combineF = 1 if TotOil > 0 & elecb_mmbtu == 0 & TotCoal == 0 & TotGas == 0 
replace combineF = 2 if TotOil == 0 & elecb_mmbtu > 0 & TotCoal == 0 & TotGas == 0 
replace combineF = 3 if TotOil == 0 & elecb_mmbtu == 0 & TotCoal > 0 & TotGas == 0 
replace combineF = 4 if TotOil == 0 & elecb_mmbtu == 0 & TotCoal == 0 & TotGas > 0 
replace combineF = 12 if TotOil > 0 & elecb_mmbtu > 0 & TotCoal == 0 & TotGas == 0 
replace combineF = 13 if TotOil > 0 & elecb_mmbtu == 0 & TotCoal > 0 & TotGas == 0 
replace combineF = 14 if TotOil > 0 & elecb_mmbtu == 0 & TotCoal == 0 & TotGas > 0 
replace combineF = 23 if TotOil == 0 & elecb_mmbtu > 0 & TotCoal > 0 & TotGas == 0 
replace combineF = 24 if TotOil == 0 & elecb_mmbtu > 0 & TotCoal == 0 & TotGas > 0 
replace combineF = 34 if TotOil == 0 & elecb_mmbtu == 0 & TotCoal > 0 & TotGas > 0 
replace combineF = 123 if TotOil > 0 & elecb_mmbtu > 0 & TotCoal > 0 & TotGas == 0 
replace combineF = 124 if TotOil > 0 & elecb_mmbtu > 0 & TotCoal == 0 & TotGas > 0 
replace combineF = 134 if TotOil > 0 & elecb_mmbtu == 0 & TotCoal > 0 & TotGas > 0 
replace combineF = 234 if TotOil == 0 & elecb_mmbtu > 0 & TotCoal > 0 & TotGas > 0 
replace combineF = 1234 if TotOil > 0 & elecb_mmbtu > 0 & TotCoal > 0 & TotGas > 0

*** Emision factors (metric ton CO2e per mmBtu) and co2 ***
	* Source (fuels):  Gupta et al. (2019) - Annexure 3
	* Source (electricity): All India Install Capacity, from Executive Summary of Power Sector (January 2016)
		* Retrieved from: https://cea.nic.in/monthly-quarterly-archive-reports/?lang=en
capture drop gamma_coal gamma_oil gamma_natgas gamma_elec
gen gamma_gas = (56.1+0.001+0.0001)/947.817
gen gamma_oil = (77.4+0.003+0.0006)/947.817 								// furnace oil   
gen gamma_coal = (93.6833 +0.001+0.0015)/947.817							// Coal: all other industries
replace gamma_coal = (95.62667+0.001+0.0015)/947.817 if nic08_4d == 2394 	// Coal: Cement
replace gamma_coal = (96.28667+0.001+0.0015)/947.817 if nic08_4d == 1701	// Coal: Pulp & paper
replace gamma_coal = (96.36+0.001+0.0015)/947.817 if nic08_4d == 2420		// Coal: Non-ferrous metals
gen gamma_elec = 0.608*gamma_coal+0.086*gamma_gas+0.0034*gamma_oil			// Electricity
* Add labels
lab var gamma_gas "Emission Factor - Natural Gas"
lab var gamma_oil "Emission Factor - Oil"
lab var gamma_elec "Emission Factor - Electricity"
lab var gamma_coal "Emission Factor - Coal"

* CO2e (metric tons)
gen co2_gas = gamma_gas*TotGas_mmbtu
gen co2_coal = gamma_coal*TotCoal_mmbtu
gen co2_elec = gamma_elec*elecb_mmbtu
gen co2_oil = gamma_oil*TotOil_mmbtu
egen co2 = rowtotal(co2_gas co2_coal co2_elec co2_oil)
* Total Energy (mmBtu)
egen Energy_mmbtu = rowtotal(TotGas_mmbtu TotOil_mmbtu TotCoal_mmbtu elecb_mmbtu)

********************************************************************************
*** 2. Clean data for production function estimation 
********************************************************************************

forvalues t = 2009/2016 {
	gen D`t' = 0
	replace D`t' = 1 if year == `t'
}
* Keep active plants
keep if Yspend_nominal != . & Lspend_nominal != . & Espend_nominal != . & Mspend_nominal != . & Kspend_nominal != .
keep if Lspend_nominal > 0 & Espend_nominal > 0 & Mspend_nominal > 0 & Kspend_nominal > 0 & Yspend_nominal > 0

* All variables used for PFE
gen KL = K/L
gen KM = K/M
gen LM = L/M
gen ME = Mspend_nominal/Espend_nominal
gen MY = Mspend_nominal/Yspend_nominal
gen EY = Espend_nominal/Yspend_nominal
gen LY = Lspend_nominal/Yspend_nominal
gen CostY = (Mspend_nominal+Espend_nominal+Lspend_nominal)/Yspend_nominal
gen Mspend_all = Espend_nominal+Mspend_nominal
* Fuel prices
replace pgas_mmbtu = TotGas/TotGas_mmbtu
drop if TotGas_mmbtu > 0 & pgas_mmbtu == 0
replace pcoal_mmbtu = TotCoal/TotCoal_mmbtu
replace pelecb_mmbtu = PurchValElecBought/elecb_mmbtu
replace poil_mmbtu = TotOil/TotOil_mmbtu

* Trim data (remove unlikely outliers by top-coding and bottom-coding the 1% tail)
gen flag1 = 0
levelsof nic08_4d, local(ind)
local iter_ind = 1
foreach j of local ind {
	foreach vars in Lspend_nominal Mspend_nominal Kspend_nominal Yspend_nominal Yspend_nominal_new MY KL CostY {
		qui su `vars' if nic08_4d == `j', det 
		qui replace flag1 = 1 if (`vars' < r(p1) & nic08_4d == `j') | (`vars' > r(p99) & nic08_4d == `j')
	}
	foreach vars in elecb_mmbtu TotOil_mmbtu TotGas_mmbtu TotOil_mmbtu {
		qui su `vars' if nic08_4d == `j' & `vars' > 0, det
		qui replace flag1 = 1 if (`vars' < r(p1) & `vars' > 0 & `vars' != . & nic08_4d == `j') | (`vars' > r(p99) & `vars' > 0 & `vars' != . & nic08_4d == `j')
	}
	foreach vars in pgas_mmbtu pcoal_mmbtu {
		qui su `vars' if nic08_4d == `j', det
		qui replace flag1 = 1 if (`vars' < r(p1) & `vars' != . & nic08_4d == `j') | (`vars' > r(p99) & `vars' != . & nic08_4d == `j')
	}
	foreach vars in KM LM {
		qui su `vars' if nic08_4d == `j', det
 		qui replace flag1 = 1 if (`vars' < r(p1) & nic08_4d == `j') | (`vars' > r(p95) & nic08_4d == `j') // I cut at the 95th percentile for KM and LM because the 99th percentile is abnormally high
	}
}
drop if flag1 == 1
drop flag1

* Recompute CES geometric mean normalization after winsoring
foreach vars in M L K Y {
	drop `vars'gmean
	drop `vars'
	egen `vars'gmean = gmean(`vars'qty), by(nic08_4d)
	gen `vars' = `vars'qty/`vars'gmean
}
drop KM LM KL
gen KL = K/L
gen KM = K/M
gen LM = L/M
bysort year nic08_4d: gen N = _N

* Save data
save Data/Panel_Data/Clean_data/ASI_PanelClean-allind_b, replace