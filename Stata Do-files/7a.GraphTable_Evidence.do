*** CODEFILE 7a ***

**** This file creates graphs/tables for reduced-form evidence section of the paper
	* Graph of fuel prices
	* Table of mixing between fuel sets
	* Table of switching between fuel sets
	* Graph of relatonship between output/worker and number of fuels

* Data directory
global ASIpaneldir Data/Panel_Data/Clean_data

* Import data and set panel
use Data/Panel_Data/Clean_data/ASI_PanelClean-allind_b, clear
xtset IDnum year


**************************** ALL INDUSTRIES ****************************

**************************************************************
* 2. GRAPH OF EVOLUTION OF ENERGY PRICES
**************************************************************

* Graph for energy prices (oil, gas, coal, electricity)
preserve
	* Show prices in rupees/mmBtu 
	local be = 82865100 
	foreach vars in pgas_mmbtu pcoal_mmbtu poil_mmbtu pelecb_mmbtu {
		replace `vars' = `vars'*`be'
	}
	collapse (median) pgas_mmbtu pcoal_mmbtu poil_mmbtu pelecb_mmbtu, by(year)
	graph twoway (connected pgas_mmbtu year, msymbol(circle)) (connected pcoal_mmbtu year, lpattern(dash) msymbol(square)) /// 
	(connected poil_mmbtu year, lpattern(shortdash) msymbol(triangle)) (connected pelecb_mmbtu year, lpattern(dash_dot) msymbol(diamond)), ///
	xlabel(2009[1]2016) ytitle("Median price (rupee per mmBtu)") xtitle("Year") graphregion(color(white)) ///
	legend(ring(0) position(10) label(1 "Natural gas") label(2 "Coal") size(med) label(3 "Oil") label(4 "Electricity")) xline(2014, lwidth(thin) lpattern(solid) lcolor(gray))
	graph export Output\Graphs\EnergyPrices_Year.pdf, replace 
restore


**************************** Selected Industries **************************

****************************************************************
* 3. SUMMARY STATISTICS - NUMBER OF PLANTS, REVENUES, EMISSIONS
****************************************************************

preserve 
	gen ind = nic08_4d
	replace ind = 9999 if nic08_4d != 2394 & nic08_4d !=  2410 & nic08_4d !=  1701 & nic08_4d != 2310
	* Average number of plants by year
	bysort ind year: gen Nplants = _N
	egen Nplants_avg = mean(Nplants), by(ind)
	* Average annual revenue (million)
	egen Yrev_annual = mean(Yspend_nominal), by(ind year)
	* Aggregate annual energy input share (spending)
	foreach vars in Espend Mspend Lspend {
		egen `vars'_tot = total(`vars'_nominal), by(ind year)
	}
	gen sE_agg = Espend_tot/(Espend_tot+Mspend_tot+Lspend_tot)
	egen sE_agg_mean = mean(sE_agg), by(ind)
	* Total annual co2 emissions
	egen co2_tot = sum(co2), by(year ind)
	egen co2_tot_avg = mean(co2_tot), by(ind)
	replace co2_tot_avg = co2_tot_avg/1000000
	* Aggregate coal fuel share
	foreach vars in TotGas_mmbtu TotCoal_mmbtu TotOil_mmbtu elecb_mmbtu {
		egen `vars'_tot = total(`vars'), by(ind year)
	}
	gen scoal_agg = TotCoal_mmbtu_tot/(TotGas_mmbtu_tot + TotCoal_mmbtu_tot + TotOil_mmbtu_tot + elecb_mmbtu_tot)
	egen scoal_agg_avg = mean(scoal_agg), by(ind)

	eststo clear
	eststo: estpost tabstat Nplants_avg Yrev_annual co2_tot_avg sE_agg_mean scoal_agg_avg, by(ind)
	esttab using "Output/Tables/SummaryStats_SelectedInd.tex", cell("Nplants_avg(fmt(0)) Yrev_annual(fmt(2)) co2_tot_avg(fmt(2)) sE_agg_mean(fmt(2)) scoal_agg_avg(fmt(2))") /// 
							collabels("Number of Plants" "Annual Revenue" "CO2 Emissions" "Energy Input Share" "Coal Fuel Share") noobs ///
							nomtitle nonumber drop("Total") tex wrap replace
restore 

**************************************************************
* 4. TABLE OF FUEL MIXING
**************************************************************

preserve 
	replace combineF = 9999 if combineF == 1 | combineF == 2 | combineF == 3 | combineF==4|combineF==13|combineF==14|combineF==23|combineF==24|combineF==34|combineF==134|combineF==234
	lab def combineF 9999 "Other", add
	lab def combineF 12 "Oil,Electricity", add
	lab def combineF 124 "Oil,Gas,Electricity", add
	lab def combineF 123 "Oil,Coal,Electricity", add
	lab def combineF 1234 "Oil,Coal,Gas,Electricity", add
	lab val combineF combineF
	keep if nic08_4d == 2410 | nic08_4d == 2394 | nic08_4d == 2431 | nic08_4d == 2310
	eststo clear
	estpost tab combineF if nic08_4d == 2410
	eststo steel
	estpost tab combineF if nic08_4d == 2431
	eststo casting
	estpost tab combineF if nic08_4d == 2394
	eststo cement
	estpost tab combineF if nic08_4d == 2310
	eststo glass
	esttab using "Output/Tables/Mixing/FuelMixing_ASI-SelectedInd.tex", cells("pct(fmt(2))")       ///
    varlabels(`e(labels)', blist(Total "{hline @width}{break}")) ///
    varwidth(20) nonumber nomtitle noobs replace	
restore

**************************************************************
* 5. TABLE OF FUEL SET SWITCHING
**************************************************************

preserve
	* Tag number of years plants are observed 
	egen nyear = total(inrange(year, 2009, 2016)), by(IDnum)
	sort IDnum year
	xtset IDnum year
	* Define adding a fuel to the mix from last to current year
	foreach fuel in TotCoal TotOil TotGas elecb {
		gen fuelswitch_to`fuel' = 0
		replace fuelswitch_to`fuel' = 1 if `fuel'_mmbtu > 0 & L.`fuel'_mmbtu == 0
		gen fuelswitch_off`fuel' = 0
		replace fuelswitch_off`fuel' = 1 if `fuel'_mmbtu == 0 & L.`fuel'_mmbtu > 0 & L.`fuel'_mmbtu != . & `fuel'_mmbtu != .
	}
	rename fuelswitch_toTotCoal fuelswitch_tocoal
	rename fuelswitch_toTotGas fuelswitch_togas
	rename fuelswitch_toTotOil fuelswitch_tooil
	rename fuelswitch_offTotCoal fuelswitch_offcoal
	rename fuelswitch_offTotGas fuelswitch_offgas
	rename fuelswitch_offTotOil fuelswitch_offoil
	* Drop first year because I don't observe switching from 2008 to 2009
	drop if year == 2009
	* Define switching to a new fuel or off an existing fuel from last to current year
	gen fuelswitch_to = 0
	replace fuelswitch_to = 1 if fuelswitch_tocoal == 1 | fuelswitch_tooil ==  1 | fuelswitch_togas == 1 | fuelswitch_toelecb == 1
	gen fuelswitch_off = 0
	replace fuelswitch_off = 1 if fuelswitch_offcoal == 1 | fuelswitch_offoil ==  1 | fuelswitch_offgas == 1 | fuelswitch_offelecb == 1
	* Tag plants that add a fuel to their mix at least once
	bysort IDnum: egen switch_to_anyyear = max(fuelswitch_to)
	bysort IDnum: egen switch_togas_anyyear = max(fuelswitch_togas)
	bysort IDnum: egen switch_tocoal_anyyear = max(fuelswitch_tocoal)
	bysort IDnum: egen switch_tooil_anyyear = max(fuelswitch_tooil)
	bysort IDnum: egen switch_toelec_anyyear = max(fuelswitch_toelecb)
	* Tag plants that drop a fuel from their mix at least once
	bysort IDnum: egen switch_off_anyyear = max(fuelswitch_off)
	bysort IDnum: egen switch_offgas_anyyear = max(fuelswitch_offgas)
	bysort IDnum: egen switch_offcoal_anyyear = max(fuelswitch_offcoal)
	bysort IDnum: egen switch_offoil_anyyear = max(fuelswitch_offoil)
	bysort IDnum: egen switch_offelec_anyyear = max(fuelswitch_offelecb)
	* Create table
	collapse (mean) switch_to_anyyear switch_off_anyyear nyear, by(IDnum)
	lab var switch_to_anyyear "Adds a New Fuel"
	lab def switch_to_anyyear 0 "No" 1 " Yes"
	lab val switch_to_anyyear switch_to_anyyear
	lab var switch_off_anyyear "Drops an Existing Fuel"
	lab def switch_off_anyyear 0 "No" 1 " Yes"
	lab val switch_off_anyyear switch_to_anyyear
	eststo clear
	eststo: estpost tabstat switch_to_anyyear switch_off_anyyear if nyear > 1, by(nyear) notot
	esttab using "Output/Tables/Switching/nswitch.tex", cell("switch_to_anyyear(fmt(2)) switch_off_anyyear(fmt(2))") /// 
		collabels("Switch to New Fuel" "Switch off Existing Fuel") noobs nomtitle nonumber replace
restore

******************************************************************************
* 6. GRAPH AND TABLE OF NUMBER OF FUELS AGAINST PRODUCTIVITY (OUTPUT/WORKER)
******************************************************************************

preserve 
	* Table: regression of log(revenue/worker) against number of fuels	
	gen prodest = Yspend_nominal/PersonsTotal
	drop if prodest == 0
	gen lnprodest = log(prodest)
	gen nfuel = 0
	replace nfuel = 1 if combineF < 12
	replace nfuel = 2 if combineF < 100 & nfuel != 1
	replace nfuel = 3 if combineF < 1000 & nfuel != 1 & nfuel != 2
	replace nfuel = 4 if combineF == 1234
	
	* Table: regression of log(revenue/worker) against number of fuels
	eststo clear
	eststo mdl1: reg lnprodest nfuel
	eststo mdl2: reg lnprodest nfuel i.nic08_4d
	eststo mdl3: reg lnprodest nfuel i.nic08_4d i.year 
	esttab using "Output/Tables/nfuel_productivity-allind.tex", se noconstant keep(nfuel) ///
		indicate("Industry Fixed Effects = *.nic08_4d" "Year Fixed Effects = *.year") replace
	
	* Graph of number of fuels against log(revenue/worker) percentiles (leaving out industry and year fixed effects)
	reg lnprodest i.nic08_4d i.year
	predict lnprodest1, res
	drop lnprodest
	rename lnprodest1 lnprodest
	
	pctile pctile_prod = lnprodest, nquantiles(100)  
	gen prod_pctile = pctile_prod[1] if lnprodest <= pctile_prod[1]
	forvalues t = 2/100 {
		replace prod_pctile = pctile_prod[`t'] if lnprodest > pctile_prod[`t'-1] & lnprodest <= pctile_prod[`t']
	}
	su lnprodest if lnprodest > pctile_prod[100]
	replace prod_pctile = r(mean) if lnprodest > pctile_prod[100]

	gen nfuel_se = nfuel 
	collapse(mean) nfuel (semean) nfuel_se, by(prod_pctile)
	
	gen nfuel_lb = nfuel - 1.96*nfuel_se
	gen nfuel_ub = nfuel + 1.96*nfuel_se
	gen prod_pctilenum = _n
	
	graph twoway (line nfuel_lb prod_pctilenum, lcolor(navy) lwidth(thin) lpattern(shortdash)) ///
			(line nfuel prod_pctilenum, lcolor(navy) lpattern(solid)) ///
			(line nfuel_ub prod_pctilenum, lcolor(navy) lwidth(thin) lpattern(shortdash)), xtitle("Percentile of log(Revenue/Worker)", size(large)) ///
			xlabel(0[10]100) ytitle("Average Number of Fuels", size(large)) legend(ring(0) position(10) size(mid) label(1 "95% Confidence Interval") label(2 "Average Number of Fuels") ///
			label(3 ""))	
	graph export "Output/Graphs/nfuel_RevenuePerWorker.pdf", replace
restore




