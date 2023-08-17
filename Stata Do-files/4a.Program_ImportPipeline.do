*** CODEFILE 4a ***

**** This file imports data on the natural gas pipeline network and merges it with the main dataset by districts

*--------------------------------------------- Pipeline variable names: ---------------------------------------------------*
		* Pipeline: Name of pipeline
		* Operator: Operator/owner of pipeline
		* Zone: approx 250km areas from source of pipeline 
			* Zone 1: first 250km
			* Zone 2: 250km to 500km
			* Zone 3: 500km to 750km
			* Zone 4: 750km +
		* Connection: Wehther Indian district is directly or indirectly (through adjacent district) connected to pipeline
*--------------------------------------------------------------------------------------------------------------------------
	
import excel "Data/External/GasDistrNetworks.xlsx", sheet("Sheet1") firstrow clear

rename Dateofinoguration year
drop Source Extension

***************************************************************
* Remove duplicates with the same year
***************************************************************

replace year = 2008 if year < 2008
duplicates tag Districts year, gen(duptag)
encode Connection, gen(ConnectionNum)
bysort Districts year: egen BestConnection = min(ConnectionNum)
bysort Districts year: egen BestZone = min(Zone)
sort Districts year duptag
bysort Districts year: drop if ConnectionNum > BestConnection & duptag >= 1 
drop duptag
duplicates tag Districts year, gen(duptag)
bysort Districts year: drop if Zone > BestZone & duptag >= 1
drop duptag ConnectionNum BestConnection BestZone
duplicates drop Districts year, force
 
***************************************************************
* Remove duplicates with different years
***************************************************************

* Create each variable for by year and assign it to the pipeline in the previous year if it is missing
foreach vars in Pipeline Operator States Connection {
	rename `vars' `vars'string
	encode `vars'string, gen(`vars')
	drop `vars'string
}
reshape wide Pipeline Operator States Zone Connection, i(Districts) j(year)
foreach year in 2013 2015 2016 {
	foreach vars in Pipeline Operator States Zone Connection {
		gen `vars'`year' = .
	}
}
forvalues year = 2009/2016 {
	local lastyear `=`year'-1'
	foreach vars in Pipeline Operator States Zone Connection {
		replace `vars'`year' = `vars'`lastyear' if  `vars'`year' == . & `vars'`lastyear' != .
	}
}
foreach year in 2013 2015 2016 {
	foreach vars in Pipeline Operator States Zone Connection {
		lab values `vars'`year' `vars'
	}
}

* Find the best pipeline that each district has access to in a given year (priority to direct connection, then zone)
forvalues year = 2009/2016 {
	local lastyear `=`year'-1'
	foreach vars in Pipeline Operator States Zone Connection {
		replace `vars'`year' = `vars'`lastyear' if Connection`year' == 2 & Connection`lastyear' == 1
	}
}
forvalues year = 2009/2016 {
	local lastyear `=`year'-1'
	foreach vars in Pipeline Operator States Connection Zone {
		replace `vars'`year' = `vars'`lastyear' if Zone`year' > Zone`lastyear' & Connection`year' == Connection`lastyear'
	}
}
reshape long Pipeline Operator States Zone Connection, i(Districts) j(year)

***************************************************************
* Organize pipeline data to match main dataset
	* In particular, states and district names
***************************************************************
bysort Districts: egen maxState = max(States)
replace States = maxState
drop maxState
decode States, gen(StatesString)
replace StatesString = "Uttaranchal" if StatesString == "Uttarakhand"
replace States = States + 40
replace States = 35 if StatesString == "Andaman & N. Island"
label define States 35 "Andaman & N. Island", modify
replace States = 28 if StatesString == "Andhra Pradesh"
label define States 28 "Andhra Pradesh", modify
replace States = 12 if StatesString == "Arunachal Pradesh"
label define States 12 "Arunachal Pradesh", modify
replace States = 18 if StatesString == "Assam"
label define States 18 "Assam", modify
replace States = 10 if StatesString == "Bihar"
label define States 10 "Bihar", modify
replace States = 04 if StatesString == "Chandigarh(U.T.)"
label define States 04 "Chandigarh(U.T.)", modify
replace States = 22 if StatesString == "Chattisgarh"
label define States 22 "Chattisgarh", modify
replace States = 26 if StatesString == "Dadra & Nagar Haveli"
label define States 26 "Dadra & Nagar Haveli", modify
replace States = 25 if StatesString == "Daman & Diu"
label define States 25 "Daman & Diu", modify
replace States = 07 if StatesString == "Delhi"
label define States 07 "Delhi", modify
replace States = 30 if StatesString == "Goa"
label define States 30 "Goa", modify
replace States = 24 if StatesString == "Gujarat"
label define States 24 "Gujarat", modify
replace States = 06 if StatesString == "Haryana"
label define States 06 "Haryana", modify
replace States = 02 if StatesString == "Himachal Pradesh"
label define States 02 "Himachal Pradesh", modify
replace States = 01 if StatesString == "Jammu & Kashmir"
label define States 01 "Jammu & Kashmir", modify
replace States = 20 if StatesString == "Jharkhand"
label define States 20 "Jharkhand", modify
replace States = 29 if StatesString == "Karnataka"
label define States 29 "Karnataka", modify
replace States = 32 if StatesString == "Kerala"
label define States 32 "Kerala", modify
replace States = 31 if StatesString == "Lakshadweep"
label define States 31 "Lakshadweep", modify
replace States = 23 if StatesString == "Madhya Pradesh"
label define States 23 "Madhya Pradesh", modify
replace States = 27 if StatesString == "Maharashtra"
label define States 27 "Maharashtra", modify
replace States = 14 if StatesString == "Manipur"
label define States 14 "Manipur", modify
replace States = 17 if StatesString == "Meghalaya"
label define States 17 "Meghalaya", modify
replace States = 15 if StatesString == "Mizoram"
label define States 15 "Mizoram", modify
replace States = 13 if StatesString == "Nagaland"
label define States 13 "Nagaland", modify
replace States = 21 if StatesString == "Orissa"
label define States 21 "Orissa", modify
replace States = 34 if StatesString == "Pondicherry"
label define States 34 "Pondicherry", modify
replace States = 03 if StatesString == "Punjab"
label define States 03 "Punjab", modify
replace States = 08 if StatesString == "Rajasthan"
label define States 08 "Rajasthan", modify
replace States = 11 if StatesString == "Sikkim"
label define States 11 "Sikkim", modify
replace States = 33 if StatesString == "Tamil Nadu"
label define States 33 "Tamil Nadu", modify
replace States = 16 if StatesString == "Tripura"
label define States 16 "Tripura", modify
replace States = 09 if StatesString == "Uttar Pradesh"
label define States 09 "Uttar Pradesh", modify
replace States = 05 if StatesString == "Uttaranchal"
label define States 05 "Uttaranchal", modify
replace States = 19 if StatesString == "West Bengal"
label define States 19 "West Bengal", modify
replace States = 36 if StatesString == "Telegana"
label define States 36 "Telegana", modify

rename States StateCode
rename Districts districtname
replace districtname = "Bans Kantha" if districtname == "Banas Kantha"
replace districtname = "Dibrugarh" if districtname == "Dibrugahr"
replace districtname = "Navasari" if districtname == "Navsari"
replace districtname = "Panipat" if districtname == "panipat"
replace districtname = "Karbiaglong" if districtname == "Karbiaglong "
replace districtname = "Jorhat" if districtname == "Jorhat "
duplicates drop districtname year StateCode, force
lab var Pipeline "Name of Gas Pipeline"
lab var Connection "Connection to Nearest Gas Pipeline (Direct or Indirect)"
lab var Operator "Company Operating Pipeline"
drop StatesString
save "Data/Panel_Data/Clean_data/ASI_GasPipeline", replace 

***************************************************************
* Merge with main dataset by district
***************************************************************
use "Data/Panel_Data/Clean_data/ASI_PanelCleanFinal", clear
merge m:1 districtname year StateCode using Data/Panel_Data/Clean_data/ASI_GasPipeline
drop if _merge == 2
drop _merge
save "Data/Panel_Data/Clean_data/ASI_PanelCleanFinal", replace



