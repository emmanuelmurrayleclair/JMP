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
	graph twoway (connected pgas_mmbtu year, msymbol(circle) msize(medlarge)) (connected pcoal_mmbtu year, lpattern(dash) msymbol(square) msize(medlarge)) /// 
	(connected poil_mmbtu year, lpattern(shortdash) msymbol(triangle) msize(medlarge)) (connected pelecb_mmbtu year, lpattern(dash_dot) msymbol(diamond) msize(medlarge)), ///
	xlabel(2009[1]2016, labsize(med)) ytitle("Median price (rupee per mmBtu)", size(medlarge)) xtitle("Year", size(medlarge)) graphregion(color(white)) ///
	legend(pos(12) label(1 "Natural gas") label(2 "Coal") size(large) label(3 "Oil") label(4 "Electricity")) xline(2014, lwidth(thin) lpattern(solid) lcolor(gray)) ///
	ylabel(0[500]2000, labsize(med))
	graph export Output\Graphs\EnergyPrices_Year.pdf, replace 
restore


**************************** Selected Industries **************************

****************************************************************
* 3. SUMMARY STATISTICS - NUMBER OF PLANTS, REVENUES, EMISSIONS
****************************************************************

* Heavy manufacturing
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

* Steel only
preserve 
	gen ind = nic08_4d
	replace ind = 9999 if nic08_4d != 2410
	* Average number of plants by year
	bysort ind year: gen Nplants = _N
	egen Nplants_avg = mean(Nplants), by(ind)
	* Average annual revenue (million)
	egen Yrev_annual = mean(Yspend_nominal), by(ind year)
	* Average annual co2 emissions
	egen co2_avg = mean(co2), by(ind year)
	replace co2_avg = co2_avg/1000
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
	eststo: estpost tabstat Nplants_avg Yrev_annual co2_avg sE_agg_mean scoal_agg_avg, by(ind)
	esttab using "Output/Tables/SummaryStats_Steel.tex", cell("Nplants_avg(fmt(0)) Yrev_annual(fmt(2)) co2_avg(fmt(2)) sE_agg_mean(fmt(2)) scoal_agg_avg(fmt(2))") /// 
							collabels("Number of Plants" "Annual Revenue" "CO2 Emissions" "Energy Input Share" "Coal Fuel Share") noobs ///
							nomtitle nonumber drop("Total") tex wrap replace
restore 


**************************************************************
* 4. TABLE OF FUEL MIXING
**************************************************************

* Heavy industries
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

* Steel only
preserve 
	replace combineF = 9999 if combineF == 1 | combineF == 2 | combineF == 3 | combineF==4|combineF==13|combineF==14|combineF==23|combineF==24|combineF==34|combineF==134|combineF==234
	lab def combineF 9999 "Other", add
	lab def combineF 12 "Oil,Electricity", add
	lab def combineF 124 "Oil,Gas,Electricity", add
	lab def combineF 123 "Oil,Coal,Electricity", add
	lab def combineF 1234 "Oil,Coal,Gas,Electricity", add
	lab val combineF combineF
	keep if nic08_4d == 2410
	eststo clear
	estpost tab combineF if nic08_4d == 2410
	eststo steel
	esttab using "Output/Tables/Mixing/FuelMixing_ASI-Steel.tex", cells("pct(fmt(2))")       ///
    varlabels(`e(labels)', blist(Total "{hline @width}{break}")) ///
    varwidth(20) nonumber nomtitle noobs replace	
restore

**************************************************************
* 5. GRAPH/TABLE OF FUEL SET SWITCHING
**************************************************************

*** All industries
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

*** Steel only

* Graph of fuel switching (year by year)
preserve
	keep if nic08_4d == 2410
// 	keep if combineF == 12 | combineF == 123 | combineF == 124 | combineF == 1234
	* Tag number of years plants are observed 
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
	egen nyear = total(inrange(year, 2010, 2016)), by(IDnum)
	* Define switching to a new fuel or off an existing fuel from last to current year
	gen fuelswitch_to = 0
	replace fuelswitch_to = 1 if fuelswitch_tocoal == 1 | fuelswitch_tooil ==  1 | fuelswitch_togas == 1 | fuelswitch_toelecb == 1
	gen fuelswitch_off = 0
	replace fuelswitch_off = 1 if fuelswitch_offcoal == 1 | fuelswitch_offoil ==  1 | fuelswitch_offgas == 1 | fuelswitch_offelecb == 1
	* Keep plants that are observed between year t and t+1
	sort IDnum year 
	order IDnum year nyear, last
	drop if nyear == 1
	gen tag = 0
	bysort IDnum: replace tag = 1 if nyear == L1.nyear | nyear == F1.nyear
	drop if tag == 0
	* Average switching on and off by years
	collapse (mean) fuelswitch_to fuelswitch_off, by(year)
	graph twoway (connected fuelswitch_to year, msymbol(circle) msize(medlarge)) (connected fuelswitch_off year, lpattern(dash) msymbol(square) msize(medlarge)), ///
	xlabel(2010[1]2016, labsize(med)) ytitle("Share of Plants", size(medlarge)) xtitle("", size(medlarge)) graphregion(color(white)) ///
	legend(ring(0) pos(10) label(1 "Add a fuel") label(2 "Drop a fuel") size(medlarge)) ylabel(0[0.05]0.2)
	graph export Output\Graphs\Switching\Switching_Year-steel.pdf, replace 
restore

* Table of fuel switching (all years)
preserve
	keep if nic08_4d == 2410
// 	keep if combineF == 12 | combineF == 123 | combineF == 124 | combineF == 1234
	* Tag number of years plants are observed 
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
	egen nyear = total(inrange(year, 2010, 2016)), by(IDnum)
	* Define switching to a new fuel or off an existing fuel from last to current year
	gen fuelswitch_to = 0
	replace fuelswitch_to = 1 if fuelswitch_tocoal == 1 | fuelswitch_tooil ==  1 | fuelswitch_togas == 1 | fuelswitch_toelecb == 1
	gen fuelswitch_off = 0
	replace fuelswitch_off = 1 if fuelswitch_offcoal == 1 | fuelswitch_offoil ==  1 | fuelswitch_offgas == 1 | fuelswitch_offelecb == 1
	* Keep plants that are observed between year t and t+1
	sort IDnum year 
	order IDnum year nyear, last
	drop if nyear == 1
	gen tag = 0
	bysort IDnum: replace tag = 1 if nyear == L1.nyear | nyear == F1.nyear
	drop if tag == 0
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
	collapse (mean) switch_to_anyyear switch_off_anyyear, by(IDnum)
	lab var switch_to_anyyear "Adds a New Fuel"
	lab def switch_to_anyyear 0 "No" 1 " Yes"
	lab val switch_to_anyyear switch_to_anyyear
	lab var switch_off_anyyear "Drops an Existing Fuel"
	lab def switch_off_anyyear 0 "No" 1 " Yes"
	lab val switch_off_anyyear switch_to_anyyear
	eststo clear
	eststo: estpost tabstat switch_to_anyyear switch_off_anyyear
	esttab using "Output/Tables/Switching/nswitch_overall-steel.tex", cell("switch_to_anyyear(fmt(2)) switch_off_anyyear(fmt(2))") /// 
		collabels("Switch to New Fuel" "Switch off Existing Fuel") noobs nomtitle nonumber replace
restore


********************************************************
* 6. GRAPH OF AGGREGATE FUEL QUANTITIES AND FUEL SHARES
********************************************************

preserve
	keep if nic08_4d == 2410
	
	gen lngas = log(TotGas_mmbtu)
	gen lncoal = log(TotCoal_mmbtu)
	gen lnoil = log(TotOil_mmbtu)
	gen lnelec = log(elecb_mmbtu)
	
	egen fueltot = rowtotal(TotGas_mmbtu TotCoal_mmbtu TotOil_mmbtu elecb_mmbtu)
	gen s_gas = TotGas_mmbtu/fueltot
	gen s_coal = TotCoal_mmbtu/fueltot
	gen s_oil = TotOil_mmbtu/fueltot
	gen s_elec = elecb_mmbtu/fueltot
	
	collapse (mean) TotGas_mmbtu TotCoal_mmbtu TotOil_mmbtu elecb_mmbtu lngas lncoal lnoil lnelec ///
					s_gas s_coal s_oil s_elec, by(year)
	
	* Average (log) fuel quantities
	graph twoway (connected lngas year, msymbol(circle) msize(medlarge)) (connected lncoal year, lpattern(dash) msymbol(square) msize(medlarge)) /// 
	(connected lnoil year, lpattern(shortdash) msymbol(triangle) msize(medlarge)) (connected lnelec year, lpattern(dash_dot) msymbol(diamond) msize(medlarge)), ///
	xlabel(2009[1]2016, labsize(med)) ytitle("Average (log) fuel quantities", size(medlarge)) xtitle("Year", size(medlarge)) graphregion(color(white)) ///
	legend(pos(12) label(1 "Natural gas") label(2 "Coal") size(large) label(3 "Oil") label(4 "Electricity")) xline(2014, lwidth(thin) lpattern(solid) lcolor(gray)) 
	graph export Output\Graphs\FuelQuantities_Year.pdf, replace 
	* Average within-firm fuel shares
	graph twoway (connected s_gas year, msymbol(circle) msize(medlarge)) (connected s_coal year, lpattern(dash) msymbol(square) msize(medlarge)) /// 
	(connected s_oil year, lpattern(shortdash) msymbol(triangle) msize(medlarge)) (connected s_elec year, lpattern(dash_dot) msymbol(diamond) msize(medlarge)), ///
	xlabel(2009[1]2016, labsize(med)) ytitle("Average fuel shares", size(medlarge)) xtitle("Year", size(medlarge)) graphregion(color(white)) ///
	legend(pos(12) label(1 "Natural gas") label(2 "Coal") size(large) label(3 "Oil") label(4 "Electricity")) xline(2014, lwidth(thin) lpattern(solid) lcolor(gray)) 
	graph export Output\Graphs\FuelShares_Year.pdf, replace 
restore



******************************************************************************
* 6. GRAPH AND TABLE OF NUMBER OF FUELS AGAINST PRODUCTIVITY (OUTPUT/WORKER)
******************************************************************************

* All industries
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


* Steel only
preserve 
	keep if nic08_4d == 2410
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
	eststo raw: ologit nfuel lnprodest i.year
	eststo margin: margins, dydx(lnprodest) post
	est store m1
	esttab m1 using "Output/Tables/nfuel_productivity-steel.tex", se noconstant cells("b(star fmt(3)) se(par fmt(3))")  ///
		nonumber nomtitle wide replace
	
	* Graph of number of fuels against log(revenue/worker) percentiles (leaving out industry and year fixed effects)
	reg lnprodest i.nic08_4d i.year
	predict lnprodest1, res
	drop lnprodest
	rename lnprodest1 lnprodest
	
	su lnprodest, det
	gen prod_pctile = r(p1) if lnprodest <= r(p1)
	replace prod_pctile = r(p5) if lnprodest > r(p1) & lnprodest <= r(p5)
	replace prod_pctile = r(p10) if lnprodest > r(p5) & lnprodest <= r(p10)
	replace prod_pctile = r(p25) if lnprodest > r(p10) & lnprodest <= r(p25)
	replace prod_pctile = r(p50) if lnprodest > r(p25) & lnprodest <= r(p50)
	replace prod_pctile = r(p75) if lnprodest > r(p50) & lnprodest <= r(p75)
	replace prod_pctile = r(p90) if lnprodest > r(p75) & lnprodest <= r(p90)
	replace prod_pctile = r(p95) if lnprodest > r(p90) & lnprodest <= r(p95)
	replace prod_pctile = r(p99) if lnprodest > r(p95) 
	
	gen nfuel_se = nfuel 
	collapse(mean) nfuel (semean) nfuel_se, by(prod_pctile)
	
	gen nfuel_lb = nfuel - 1.96*nfuel_se
	gen nfuel_ub = nfuel + 1.96*nfuel_se
	gen prod_pctilenum = .
	replace prod_pctilenum = 1 if prod_pctile == prod_pctile[1]
	replace prod_pctilenum = 5 if prod_pctile == prod_pctile[2]
	replace prod_pctilenum = 10 if prod_pctile == prod_pctile[3]
	replace prod_pctilenum = 25 if prod_pctile == prod_pctile[4]
	replace prod_pctilenum = 50 if prod_pctile == prod_pctile[5]
	replace prod_pctilenum = 75 if prod_pctile == prod_pctile[6]
	replace prod_pctilenum = 90 if prod_pctile == prod_pctile[7]
	replace prod_pctilenum = 95 if prod_pctile == prod_pctile[8]
	replace prod_pctilenum = 99 if prod_pctile == prod_pctile[9]
	
	graph twoway (line nfuel_lb prod_pctilenum, lcolor(navy) lwidth(thin) lpattern(shortdash)) ///
			(connected nfuel prod_pctilenum, lcolor(navy) lpattern(solid) mcolor(navy)) ///
			(line nfuel_ub prod_pctilenum, lcolor(navy) lwidth(thin) lpattern(shortdash)), xtitle("Percentile of log(Revenue/Worker)", size(large)) ///
			xlabel(1 5 10 25 50 75 90 95 99) ytitle("Average Number of Fuels", size(large)) legend(ring(0) position(4) size(mid) label(1 "95% Confidence Interval") label(2 "Average Number of Fuels") ///
			label(3 ""))	
	graph export "Output/Graphs/nfuel_RevenuePerWorker-steel.pdf", replace
	
	graph twoway (line nfuel_lb prod_pctile, lcolor(navy) lwidth(thin) lpattern(shortdash)) ///
			(connected nfuel prod_pctile, lcolor(navy) lpattern(solid) mcolor(navy)) ///
			(line nfuel_ub prod_pctile, lcolor(navy) lwidth(thin) lpattern(shortdash)), xtitle("log(Revenue/Worker)", size(large)) ///
			ytitle("Average Number of Fuels", size(large)) legend(ring(0) position(4) size(mid) label(1 "95% Confidence Interval") label(2 "Average Number of Fuels") ///
			label(3 ""))	
	graph export "Output/Graphs/nfuel_RevenuePerWorker-steel.pdf", replace
restore





