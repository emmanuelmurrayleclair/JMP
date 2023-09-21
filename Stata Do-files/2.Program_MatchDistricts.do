*** CODEFILE 2 ***

**** This file adds district information from the public ASI dataset
* REQUIRED: ASI_Clean.dta

* Data directory
global ASIpaneldir Data/Panel_Data/Raw_data

*************************************************************************
* Add district information form the public dataset (available 2000-2010)
*************************************************************************

* Prepare public dataset with district identifier for merging
use "Data/ASI_Clean.dta", clear
rename yr year
rename a5 IndCodeReturn
rename a9 AreaCode
rename a8 DistrictCode
rename awgt Multiple
rename a12 Open
rename b2 OrgCode
rename b3 OwnerCode
rename b6 YearInitialProduction
rename b8 MonthsOperation
rename c12_10 FixCapOpen
rename c13_10 FixCapClose
rename e6_10 PersonsTotal
rename e8_10 WagesTotal
replace f1 = . if f1 == 0
rename f1 WorkDoneByOthers
replace f5 = . if f5 == 0
rename f5 Insurance
keep if year < 2017
local matchvars year IndCodeReturn AreaCode Multiple Open OrgCode OwnerCode YearInitialProduction MonthsOperation FixCapOpen FixCapClose PersonsTotal WagesTotal WorkDoneByOthers Insurance
keep `matchvars' dsl DistrictCode
duplicates tag `matchvars', gen(dup)
drop if dup>0 & Open~=1
duplicates drop `matchvars', force
sort `matchvars'
save $ASIpaneldir/temp/ASI_CrossSection, replace

* Perpare the panel dataset for merging
use "Data/Panel_Data/Clean_data/ASI_PanelClean.dta", clear
destring YearCode, gen(year)
local matchvars year IndCodeReturn AreaCode Multiple Open OrgCode OwnerCode YearInitialProduction MonthsOperation FixCapOpen FixCapClose PersonsTotal WagesTotal WorkDoneByOthers Insurance
duplicates drop `matchvars', force
sort `matchvars'
merge 1:1 `matchvars' using $ASIpaneldir/temp/ASI_CrossSection
gen status=(Open==1)
label define statusvals 0 "Closed" 1 "Open"
label values status statusvals
tab _merge status
keep if _merge == 3
keep ID StateCode DistrictCode dsl year
sort ID
* Add districts for years after 2010 using panel IDs
reshape wide StateCode DistrictCode dsl, i(ID) j(year)
forvalues year = 2011/2016 {
	gen SourceYear`year' = .
	foreach prevyear in 2010 2009 2008 2007 2006 2005 2004 2003 2002 2001 {
		replace SourceYear`year' = `prevyear' if DistrictCode`prevyear' != . & DistrictCode`year' == 99
		replace DistrictCode`year' = DistrictCode`prevyear' if DistrictCode`prevyear' != . & DistrictCode`year' == 99
	}
}
reshape long StateCode DistrictCode SourceYear dsl, i(ID) j(year)
drop if dsl == .
replace SourceYear = year if SourceYear == . & DistrictCode != 99
sort ID SourceYear
save "Data/Panel_Data/Clean_data/ASI_District", replace
erase "Data/Panel_Data/Raw_data/temp/ASI_CrossSection.dta"


