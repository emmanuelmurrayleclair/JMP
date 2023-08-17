*** CODEFILE 7b ***

**** This file creates graphs/tables for spatial reduced-form evidence section of the paper
	* Map comparing natural gas pipeline network in 2009 and 2016
	* Table of regression results for natural gas pipeline expansion on natural gas takeup
	
**************************************************************
* 1. IMPORT AND ORGANIZE DATA
**************************************************************

* Import Data 
use Data/Panel_Data/Clean_data/ASI_PanelClean-allind_b, clear
xtset IDnum year
cd "Data/Spatial Data"

* Keep relevant variables
keep DistrictCodeOriginal DistrictCode districtname StateCode UnitsSameState IDnum year nic08_4d TotGas TotGas_mmbtu pgas_mmbtu TotCoal_mmbtu pcoal_mmbtu TotOil_mmbtu elecb_mmbtu Pipeline Operator Connection Zone
rename districtname distname
drop if distname == ""
* rename district names to match spatial dataset
replace distname = "Guntur" if distname == "Amaravati"
* Merge with district geolocalisation data
merge m:m distname using indb_dist
drop if _merge == 2

cd ../..

*****************************************************************
* 2. MAP COMPARING NATURAL GAS PIPELINE NETWORK IN 2009 AND 2016
*****************************************************************

* 2009
preserve
	collapse (min) Zone, by(id year)
	replace Zone = Zone + 4
	replace Zone = 1 if Zone == 8
	replace Zone = 2 if Zone == 7
	replace Zone = 3 if Zone == 6
	replace Zone = 4 if Zone == 5
	spmap Zone using "Data/Spatial Data/indcoord_dist.dta" if year == 2009, id(id) clmethod(unique) fcolor(Reds) ///
		legend(pos(7) size(2.8) order(1 "Outside" 2 "Zone 4" 3 "Zone 3" 4 "Zone 2" 5 "Zone 1")) osize(0.05 ..) ndsize(0.05 ..)
	graph export "Output/Graphs/Spatial/GraphZone-2009.pdf", as(pdf) name("Graph") replace
	graph export "Output/Graphs/Spatial/GraphZone-2009.png", as(png) name("Graph") replace
restore 
* 2016 
preserve
	collapse (min) Zone, by(id)
	replace Zone = Zone + 4
	replace Zone = 1 if Zone == 8
	replace Zone = 2 if Zone == 7
	replace Zone = 3 if Zone == 6
	replace Zone = 4 if Zone == 5
	spmap Zone using "Data/Spatial Data/indcoord_dist.dta", id(id) clmethod(unique) fcolor(Reds) ///
		legend(pos(7) size(2.8) order(1 "Outside" 2 "Zone 4" 3 "Zone 3" 4 "Zone 2" 5 "Zone 1")) osize(0.05 ..) ndsize(0.05 ..)
	graph export "Output/Graphs/Spatial/GraphZone-2016.pdf", as(pdf) name("Graph") replace
	graph export "Output/Graphs/Spatial/GraphZone-2016.png", as(png) name("Graph") replace
restore


*****************************************************************
* 3. TABLE OF REGRESSION RESULTS : EXPANSION OF PIPELINE NETWORK
*****************************************************************

* Share of plants who use natural gas 
gen Dgas = 0
replace Dgas = 1 if TotGas_mmbtu > 0
* Share of plants with have access ti pipeline
gen Dpipeline = 0
replace Dpipeline = 1 if Pipeline != .

* Indicator for adding natural gas between year t-1 and t
encode distname, gen(distnum)
xtset IDnum year
sort IDnum year
gen add_gas = 0
replace add_gas = 1 if Dgas == 1 & L.Dgas == 0
* Indicator for pipeline network expanding between t-1 and t
gen add_pipeline = 0 
replace add_pipeline = 1 if Dpipeline == 1 & L.Dpipeline == 0

* Marginal effect regression: logit of pipeline expansion on switching to natural gas
eststo clear
logit add_gas i.add_pipeline
margins, dydx(add_pipeline) post
eststo mdl1, title("No controls")
logit add_gas i.add_pipeline i.nic08_4d
margins, dydx(add_pipeline) post 
eststo mdl2, title("Industry FE") 
logit add_gas i.add_pipeline i.nic08_4d i.distnum
margins, dydx(add_pipeline) post 
eststo mdl3, title("Industry, Distridct FE") 
esttab using "Output/Tables/Spatial/Table_AddGas_AddPipeline.tex", label se replace booktabs 