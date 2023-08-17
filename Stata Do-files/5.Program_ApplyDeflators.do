*** CODEFILE 5 ***

**** This file deflates variables as needed ***
*** This code file was partially influenced by the code provided in Martin, Nataraj and Harrison (2017, AER) ***

set more off

*********************************************
*** Prepare Deflators (base year 2005)    ***
*********************************************

* Prepare WPI for output, capital, materials and energy deflators
insheet using Data/External/WPI.csv, clear
rename *, lower
foreach i of numlist 2005/2016{
	rename in`i' wpi`i'
}
reshape long wpi, i(comm_name comm_code comm_wt) j(year)
gen temp = wpi if year == 2005
by comm_code, sort: egen wpi2005 = mean(temp)
drop temp
rename wpi origwpi
gen wpi = origwpi/wpi2005
sort comm_code year
save Data/External/WPI, replace

* Prepare capital deflators
preserve
keep if comm_code == 1311000000
rename wpi wpiMachinery
keep year wpiMachinery
sort year
save Data/External/capital_deflator, replace
restore

* Prepare materials deflators
preserve
keep if comm_code == 1000000000
rename wpi wpiMaterials
keep year wpiMaterials
sort year
save Data/External/materials_deflator, replace
restore

* Prepare energy deflators
preserve
keep if comm_code == 1200000000
rename wpi wpiEnergy
keep year wpiEnergy
sort year
save Data/External/energy_deflator, replace
restore

* Prepare alternate capital deflator (Penn World Table)
insheet using Data/External/Penn.csv, clear
keep if year>=2005
gen temp = pi if year == 2005
egen pi2005 = mean(temp)
drop temp
gen p_inv=pi/pi2005
sort year
save Data/External/Penn_capital_deflator, replace

* Prepare wage deflators
insheet using Data/External/CPI.csv, clear
gen temp = cpi if year == 2005
egen cpi2005 = mean(temp)
drop temp
rename cpi origcpi
gen cpi = origcpi/cpi2005
sort year
save Data/External/CPI, replace	

* Save NIC (3-DIGIT) to wpi concordance
insheet using Data/External/NIC3toWPI.csv, clear
drop nicgroupdescription 
order nicgroup
drop if nicgroup == .
rename commodityname comm_name
rename commoditycode comm_code
rename nicgroup nic08_3d
forvalues i=2005/2016 {
	gen wpi`i'=.
	}
reshape long wpi, i(nic08_3d comm_name comm_code) j(year)
sort comm_code year
merge m:1 comm_code year using Data/External/WPI, keepusing(wpi) update
drop if _merge==2
drop _merge
sort nic08_3d year
save Data/External/NIC3toWPI, replace

**************************************************************
*** Deflating key variables and constructing price indices ***
**************************************************************

use "Data/Panel_Data/Clean_data/ASI_PanelCleanFinal.dta", clear
capture drop _merge
local be = 1/82865100 // rupee to million US dollars (MAIN VERSION even more new)
egen Espend_nominal = rowtotal(TotCoal TotOil TotGas PurchValElecBought)

* Output deflator
sort nic08_3d year
merge m:1 nic08_3d year using Data/External/NIC3toWPI, keepusing(wpi)
keep if _merge==3
drop _merge
replace TotalOutput = TotalOutput*`be'
replace Output = Output*`be'
gen Output_defl = Output/wpi
gen TotalOutput_defl = TotalOutput/wpi
gen p_output = wpi

* Materials deflator
sort year
merge m:1 year using Data/External/materials_deflator
keep if _merge==3 
drop _merge
gen Materials = Inputs-Espend_nominal
replace Materials = Materials*`be'
gen Materials_defl = Materials/wpiMaterials
gen p_materials = wpiMaterials

* Energy
replace Espend_nominal = Espend_nominal*`be'
gen logEspend = log(Espend_nominal)

* Wage deflator (use full wage data in later versions)
sort year
merge m:1 year using Data/External/CPI
keep if _merge==3
drop _merge
replace TotalEmoluments = TotalEmoluments*`be'
gen TotalEmoluments_defl = TotalEmoluments/cpi
gen wage = cpi

* Capital deflator
sort year
merge m:1 year using Data/External/capital_deflator
keep if _merge==3
drop _merge
replace Capital = Capital*`be'
gen Capital_defl = Capital/wpiMachinery
gen p_capital = wpiMachinery

* Fuel spending rescaling
foreach vars in TotGas TotCoal TotOil PurchValElecBought {
	replace `vars' = `vars'*`be'
}

save Data/Panel_Data/Clean_data/ASI_PanelClean-allind_a, replace
