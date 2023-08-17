*** CODEFILE 11 ***

**** This Appendix file provides all graphs and tables found in Appendix which are created in Stata, in order of appearance

* Import data and set panel
use Data/Panel_Data/Clean_data/ASI_PanelClean-allind_b, clear
xtset IDnum year

*********************************************************
*** 1. Further preliminary and motivating evidence
*********************************************************


**# Graph: histogram comparing price of natural gas and coal
local be = 82865100
gen lnpgas_mmbtu = log(pgas_mmbtu*`be')
gen lnpcoal_mmbtu = log(pcoal_mmbtu*`be')
gen pgas_mmbtu_inr = pgas_mmbtu*`be'
gen pcoal_mmbtu_inr = pcoal_mmbtu*`be'
twoway (hist pcoal_mmbtu_inr if pcoal_mmbtu_inr < 1000, lcolor(gs12) fcolor(gs12)) (hist pgas_mmbtu_inr if pgas_mmbtu_inr < 3000, lcolor(red) fcolor(none)), ///
legend(label(1 "Coal") label(2 "Natural Gas"))
graph export "Output/Graphs/Appendix/FuelPrices_Hist.pdf", replace

**# Table: relationship between natural gas prices and proximity to pipeline
eststo clear
eststo mdl1: reg lnpgas_mmbtu i.Zone i.Pipeline i.year
eststo mdl2: reg lnpgas_mmbtu i.Zone i.Pipeline i.year i.nic08_4d
eststo mdl3: reg lnpgas_mmbtu i.Zone i.Pipeline i.year i.nic08_4d PersonsTotal TotGas_mmbtu
esttab using "Output/Tables/Appendix/GasPrice_PipelineProximity.tex", label wide ///
	 unstack booktabs star(+ 0.1 * 0.05 ** 0.01 *** 0.001) ///
	 se title("Relationship between (log) natural gas prices and proximity to pipelines") ///
	 indicate("year dummies = *year" "industry dummies = *nic08_4d" "Pipeline dummies = *Pipeline") nocons addnote("test" )replace


**# Graph: annual electricity generation by source (all of india)
preserve
	import delimited "Data/External/ElectricityGeneration_byfuel1.csv",clear
	drop if year == .
	destring coal, replace
	gen nuclear_renewable = wind+solar+other
	graph twoway (connected coal year) (connected gas year) (connected hydro year) (connected nuclear_renewable year), ///
	legend(label(1 "Coal") label(2 "Natural Gas") label(3 "Hdro") label(4 "Nuclear and Renewables")) graphregion(color(white))
	graph export "Output/Graphs/Appendix/ElectricityGeneration_byfuel.pdf", replace
restore


**# Graphs: Comparing Canada and India 
preserve
	* Define fuel quantity shares
	gen totfuel_mmbtu = TotOil_mmbtu+TotGas_mmbtu+TotCoal_mmbtu+elecb_mmbtu
	gen oil_s = TotOil_mmbtu/totfuel_mmbtu
	gen natgas_s = TotGas_mmbtu/totfuel_mmbtu
	gen coal_s = TotCoal_mmbtu/totfuel_mmbtu
	gen elec_s = elecb_mmbtu/totfuel_mmbtu
	 * Merge with Canadian dataset (NPRI)
	gen dataset_id = 0
	rename year yr
	append using "Data/External/NPRI_Clean.dta"
	* Emission intensity per unit of energy
	* Coal
	su gamma_coal
	replace gamma_coal = r(mean)
	lab var gamma_coal "Emission Factor - Coal"
	* Oil
	su gamma_oil 
	replace gamma_oil = r(mean)
	lab var gamma_oil "Emission Factor - Oil"
	* Natural Gas
	su gamma_gas
	replace gamma_gas = r(mean) 
	lab var gamma_gas "Emission Factor - Natural Gas"
	* Electricity
	su gamma_elec
	replace gamma_elec = r(mean)
	lab var gamma_elec "Emission Factor - Electricity"
	replace elecb_mmbtu = 0 if elecb_mmbtu == .
	replace elec_s = 0 if elec_s == .
	* Pollution intensity
	gen pol_intensity = (gamma_coal*coal_s)+(gamma_oil*oil_s)+(gamma_gas*natgas_s)+(gamma_elec*elec_s)
	keep if yr >= 2009
	* Distribution of pollution intensity: cement
	twoway (hist pol_intensity if nic08_4d == 2394 & dataset_id == 0, frac lcolor(gs12) fcolor(gs12)) ///
		(hist pol_intensity if ind5d == 2394 & dataset_id == 1, frac lcolor(red) fcolor(none)), ///
		legend(ring(0) position(1) size(medlarge) label(1 "Indian plants (ASI)") label(2 "Canadian plants (NPRI)")) ///
		xtitle("CO2e per mmbtu") graphregion(color(white)) ylabel(, angle(horizontal) format(%9.1f))
		graph export "Output/Graphs/Appendix/pol_intensity_cement-CAN-IND.pdf", replace
	* Evolution of pollution intensity - heavy manufacturing industries
	keep if ind5d == 2394 | ind5d == 2410 | ind5d == 1701 | ind5d == 24202 | ind5d == 2310  | nic08_4d == 2394 | nic08_4d == 2410 | nic08_4 == 1701 | nic08_4d == 2420 | nic08_4d == 2310
	collapse (mean) pol_intensity, by(yr dataset_id)
	drop if yr > 2015
	graph twoway (connected pol_intensity yr if dataset_id == 0) (connected pol_intensity yr if dataset_id == 1), xtitle("Year") ///
	ytitle("Emission intensity of 1 mmBtu (ton CO2e)")  xlabel(2009[1]2015) legend(label(1 "Indian plants (ASI)") label(2 "Canadian plants (NPRI)") size(med))
	graph export "Output/Graphs/Appendix/pol_intensity_average-CAN-IND.pdf", replace
restore


**# Graphs: annual energy usage by fuel (all ASI plants and Steel)
set scheme burd
gen gas = TotGas_mmbtu
gen coal = TotCoal_mmbtu
gen oil = TotOil_mmbtu
gen elec = elecb_mmbtu
* All industries
preserve
	collapse (sum) gas coal oil elec, by(year)
	foreach vars in gas coal oil elec {
		replace `vars' = `vars'/1000000
	}
	* Stacked area graph
	gen fqty_1 = gas
	gen fqty_2 = fqty_1 + elec
	gen fqty_3 = fqty_2 + oil
	gen fqty_4 = fqty_3 + coal
	graph twoway (area gas year) (rarea gas fqty_2 year) (rarea fqty_2 fqty_3 year) (rarea fqty_3 fqty_4 year), ///
	legend(label(1 "Natural Gas") label(2 "Electricity") label(3 "Oil") label(4 "Coal")) xlabel(2009[1]2016) xtitle("")
	graph export "Output\Graphs\Appendix\QuantityByFuel-year.pdf", replace
	* Percent area graph
	gen zero = 0
	gen p1 = fqty_1/fqty_4
	gen p2 = (fqty_1+fqty_2)/fqty_4
	gen p3 = (fqty_1+fqty_2+fqty_3)/fqty_4
	gen p4 = 1
	graph twoway (rarea zero p1 year) (rarea p1 p2 year) (rarea p2 p3 year) (rarea p3 p4 year), ///
	legend(label(1 "Natural Gas") label(2 "Electricity") label(3 "Oil") label(4 "Coal")) xlabel(2009[1]2016) xtitle("")
	graph export "Output\Graphs\Appendix\QuantityPercentByFuel-year.pdf", replace
restore
* Steel manufacturing
preserve
	keep if nic08_4d == 2410
	collapse (sum) gas coal oil elec, by(year)
	foreach vars in gas coal oil elec {
		replace `vars' = `vars'/1000000
	}
	* Stacked area graph
	gen fqty_1 = gas
	gen fqty_2 = fqty_1 + elec
	gen fqty_3 = fqty_2 + oil
	gen fqty_4 = fqty_3 + coal
	graph twoway (area gas year) (rarea gas fqty_2 year) (rarea fqty_2 fqty_3 year) (rarea fqty_3 fqty_4 year), ///
	legend(label(1 "Natural Gas") label(2 "Electricity") label(3 "Oil") label(4 "Coal")) xlabel(2009[1]2016) xtitle("")
	graph export "Output\Graphs\Appendix\QuantityByFuel-year_steel.pdf", replace
	* Percent area graph
	gen zero = 0
	gen p1 = fqty_1/fqty_4
	gen p2 = (fqty_1+fqty_2)/fqty_4
	gen p3 = (fqty_1+fqty_2+fqty_3)/fqty_4
	gen p4 = 1
	graph twoway (rarea zero p1 year) (rarea p1 p2 year) (rarea p2 p3 year) (rarea p3 p4 year), ///
	legend(label(1 "Natural Gas") label(2 "Electricity") label(3 "Oil") label(4 "Coal")) xlabel(2009[1]2016) xtitle("")
	graph export "Output\Graphs\Appendix\QuantityPercentByFuel-year_steel.pdf", replace
restore
* Graph of emission intensity
graph bar (mean) gamma_gas gamma_elec gamma_oil gamma_coal, blabel(bar, format(%4.2f)) bargap(50) over() legend(label(1 "Natural Gas") label(2 "Electricity") label(3 "Oil") label(4 "Coal"))
graph export "Output\Graphs\Appendix\EmissionIntensity-fuel.pdf", replace
set scheme cblind1 
grstyle init
grstyle color background white
grstyle color major_grid dimgray
grstyle linewidth major_grid thin
grstyle yesno draw_major_hgrid yes
grstyle yesno grid_draw_min yes
grstyle yesno grid_draw_max yes
grstyle anglestyle vertical_tick horizontal


**# Graph: number of times plants switch
preserve 
	xtset IDnum year
	sort IDnum year
	* Define adding a fuel to the mix in current period
	foreach fuel in coal oil gas elec {
		gen fuelswitch_to`fuel' = 0
		replace fuelswitch_to`fuel' = 1 if `fuel' > 0 & L.`fuel' == 0
		gen fuelswitch_off`fuel' = 0
		replace fuelswitch_off`fuel' = 1 if `fuel' == 0 & L.`fuel' > 0 & L.`fuel' != . & `fuel' != .
	}
	
	* Drop first year because I don't observe switching for that period
	gen fuelswitch_to = 0
	replace fuelswitch_to = 1 if fuelswitch_tocoal == 1 | fuelswitch_tooil ==  1 | fuelswitch_togas == 1 | fuelswitch_toelec == 1
	gen fuelswitch_off = 0
	replace fuelswitch_off = 1 if fuelswitch_offcoal == 1 | fuelswitch_offoil ==  1 | fuelswitch_offgas == 1 | fuelswitch_offelec == 1
	gen fuelswitch_any = 0
	replace fuelswitch_any = 1 if fuelswitch_to == 1 | fuelswitch_off == 1
	* count number of times plants switch fuel sets
	bysort IDnum: egen nfuelswitch = sum(fuelswitch_any)
	bysort nfuelswitch: gen nplants_nfuelswitch = _N
	drop if nfuelswitch == 0
	graph bar (mean) nplants_nfuelswitch if nic08_4d == 2410, bargap(50) over(nfuelswitch) ytitle("Number of plants")
	graph export "Output\Graphs\Appendix\nSwitch_dist-ASI.pdf", replace
restore


**# Table: Fuel set heterogeneity and fuel set switching in U.S. pulp & paper industry (GHGRP), and number of times U.S. plants witch
preserve 
	import excel "Data/External/GHGRP_US.xlsx", sheet("FUEL_DATA") firstrow clear
	rename FacilityId plantid
	rename GeneralFuelType fuel
	rename SpecificFuelType fuel_detail
	rename ReportingYear year
	rename PrimaryNAICSCode naics
	rename FuelCO2emissionsnonbiogenic co2
	rename FuelMethaneCH4emissionsmt ch4
	rename FuelNitrousOxideN2Oemissio n2o
	egen id = concat(plantid year)
	encode id, gen(idnum)
	drop if fuel == "Biomass"
	drop if fuel == "Natural Gas "
	drop if fuel == "Other"
	encode fuel, gen(fuelnum)
	drop if fuelnum == .
	sort idnum fuelnum
	by idnum fuelnum: egen co2_new = total(co2)
	by idnum fuelnum: egen ch4_new = total(ch4)
	by idnum fuelnum: egen n2o_new = total(n2o)
	gen co2e = co2_new + ch4_new + n2o_new
	quietly by idnum fuelnum:  gen dup = cond(_N==1,0,_n)
	drop if dup > 1
	drop FacilityName UnitName fuel fuel_detail OtherFuelName BlendFuelName co2 ch4 n2o co2_new ch4_new n2o_new dup 
	reshape wide co2e, i(idnum) j(fuelnum)
	sort plantid year
	rename co2e1 coal
	rename co2e2 gas
	rename co2e3 oil
	keep if IndustryTypesectors == "Pulp and Paper"
	*** Table: Fuel set heterogeneity
	foreach vars in coal oil gas {
		replace `vars' = . if `vars' == 0
	}
	gen setF = 0
	replace setF = 1 if gas != . & coal == . & oil == .
	replace setF = 2 if gas == . & coal != . & oil == .
	replace setF = 3 if gas == . & coal == . & oil != .
	replace setF = 12 if gas != . & coal != . & oil == .
	replace setF = 13 if gas != . & coal == . & oil != .
	replace setF = 23 if gas == . & coal != . & oil != .
	replace setF = 123 if gas != . & coal != . & oil != .
	lab def setF 1 "Natural Gas", add
	lab def setF 2 "coal", add
	lab def setF 3 "Oil", add
	lab def setF 12 "Natural Gas, Coal", add
	lab def setF 13 "Natural Gas, Oil", add
	lab def setF 23 "Coal, Oil", add
	lab def setF 123 "Natural Gas, Coal, Oil", add
	lab val setF setF 
	eststo clear
	estpost tab setF
	esttab using "Output/Tables/Appendix/FuelMixing_GHGRP_US-PulpPaper.tex", cells("b(label(Frequency)) pct(fmt(2))")       ///
     varlabels(`e(labels)', blist(Total "{hline @width}{break}")) ///
     varwidth(20) nonumber nomtitle noobs replace
	* Fuel Switching
	xtset plantid year
	foreach fuel in coal gas oil {
		replace `fuel' = 0 if `fuel' == .
		gen fuelswitch_to`fuel' = 0
		bysort plantid: replace fuelswitch_to`fuel' = 1 if `fuel' > 0 & L.`fuel'== 0 & `fuel' != . & L.`fuel' != .
		gen fuelswitch_off`fuel' = 0
		bysort plantid: replace fuelswitch_off`fuel' = 1 if `fuel' == 0 & L.`fuel' > 0 & `fuel' != . & L.`fuel' != .
	}
	gen fuelswitch_to = 0
	replace fuelswitch_to = 1 if fuelswitch_tocoal == 1 | fuelswitch_tooil ==  1 | fuelswitch_togas == 1 
	gen fuelswitch_off = 0
	replace fuelswitch_off = 1 if fuelswitch_offcoal == 1 | fuelswitch_offoil ==  1 | fuelswitch_offgas == 1
	gen fuelswitch_any = 0
	replace fuelswitch_any = 1 if fuelswitch_to == 1 | fuelswitch_off == 1
	*** Histogram: number of times plant switches
	bysort plantid: egen nfuelswitch = sum(fuelswitch_any)
	bysort nfuelswitch: gen nplants_nfuelswitch = _N
	gen nplants_tot = _N
	gen prplants_nfuelswitch = nplants_nfuelswitch/nplants_tot
	graph bar (mean) prplants_nfuelswitch, bargap(50) over(nfuelswitch) ytitle("Fraction of plants")
	graph export "Output\Graphs\Appendix\nSwitch_dist-US.pdf", replace
	*** Table: fuel set switching
	* Tag plants that add a fuel to their mix at least once
	bysort plantid: egen switch_to_anyyear = max(fuelswitch_to)
	bysort plantid: egen switch_togas_anyyear = max(fuelswitch_togas)
	bysort plantid: egen switch_tocoal_anyyear = max(fuelswitch_tocoal)
	bysort plantid: egen switch_tooil_anyyear = max(fuelswitch_tooil)
	* Tag plants that drop a fuel from their mix at least once
	bysort plantid: egen switch_off_anyyear = max(fuelswitch_off)
	bysort plantid: egen switch_offgas_anyyear = max(fuelswitch_offgas)
	bysort plantid: egen switch_offcoal_anyyear = max(fuelswitch_offcoal)
	bysort plantid: egen switch_offoil_anyyear = max(fuelswitch_offoil)
	* Create table
	collapse (mean) switch_to_anyyear switch_off_anyyear, by(plantid)
	lab var switch_to_anyyear "Adds a New Fuel"
	lab def switch_to_anyyear 0 "No" 1 " Yes"
	lab val switch_to_anyyear switch_to_anyyear
	lab var switch_off_anyyear "Drops an Existing Fuel"
	lab def switch_off_anyyear 0 "No" 1 " Yes"
	lab val switch_off_anyyear switch_to_anyyear
	eststo clear
	eststo: estpost tabstat switch_to_anyyear switch_off_anyyear
	esttab using "Output/Tables/Appendix/fset_switch-US.tex", cell("switch_to_anyyear(fmt(3)) switch_off_anyyear(fmt(3))") /// 
		collabels("Switch to New Fuel" "Switch off Existing Fuel") noobs nomtitle nonumber replace
restore


**# Graph: Number of fuels by plant age, average of all ASI plants
preserve 
	grstyle init
	grstyle set plain, horizontal grid dotted

	gen age = year-YearInitialProduction
	drop if age == .
	gen nfuel = 0
	replace nfuel = 1 if combineF < 12
	replace nfuel = 2 if combineF < 123 & nfuel == 0
	replace nfuel = 3 if combineF < 1234 & nfuel == 0
	replace nfuel = 4 if combineF == 1234
	gen nfuel_se = nfuel
	collapse(mean) nfuel (semean) nfuel_se, by(age)
	
	gen nfuel_lb = nfuel - 1.96*nfuel_se
	gen nfuel_ub = nfuel + 1.96*nfuel_se
	
	graph twoway (line nfuel_lb age if age <= 100, lcolor(navy) lwidth(thin) lpattern(shortdash)) ///
		(line nfuel age if age <= 100, lcolor(navy) lpattern(solid)) ///
		(line nfuel_ub age if age <= 100, lcolor(navy) lwidth(thin) lpattern(shortdash)), ///
		xtitle("Plant age in years") xlabel(0[10]100) ytitle("Average Number of Fuels") ///
		legend(ring(0) position(10) size(mid) label(1 "95% Confidence Interval") label(2 "Average Number of Fuels") label(3 ""))
	graph export "Output/Graphs/Appendix/nfuel_age.pdf", replace
restore


**# Graph: Projected natural gas demand by sector - all of India (2012 and 2016)


**# Graphs: Fuel expenditure shares
egen fueltot = rowtotal(gas coal oil elec)
gen s_gas = gas/fueltot
gen s_coal = coal/fueltot
gen s_oil = oil/fueltot
gen s_elec = elec/fueltot
hist s_gas if s_gas > 0, fraction xtitle("Fuel Share of Natural Gas")
graph export "Output/Graphs/Appendix/sGas_dist.pdf", replace
hist s_coal if s_coal > 0, fraction xtitle("Fuel Share of Coal")
graph export "Output/Graphs/Appendix/sCoal_dist.pdf", replace
hist s_oil if s_oil > 0, fraction xtitle("Fuel Share of Oil")
graph export "Output/Graphs/Appendix/sOil_dist.pdf", replace
hist s_elec if s_elec > 0, fraction xtitle("Fuel Share of Electricity")
graph export "Output/Graphs/Appendix/sElec_dist.pdf", replace



**# Table: Output varieties from steel plants
preserve
	keep if nic08_4d == 2410
	tab npcmsOutput1, sort 
	collapse (first) pipeline PipelineName PostDate KQtyK year, by(npcmsOutput1)
	* top 6 varieties
	gen occurence = 1
	collapse (sum) occurence, by(npcmsOutput1)
	gsort -occurence
	drop if npcmsOutput1 == .
	egen totoccurence = sum(occurence)
	gen frac_occur = occurence/totoccurence
	gen perc_occur = frac_occur*100
	* THIS TABLE IS CREATED MANUALLY
	list npcmsOutput1 frac_occur in 1/6, table N sum sep(0)
restore











