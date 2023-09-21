*** CODEFILE 1 ***

**** This file imports the panel ASI data
*** This code file was partially influenced by the code provided in Martin, Nataraj and Harrison (2017, AER) ***


* Import ASICC codes for inputs, imports, and output products
insheet using Data/Panel_Data/Raw_data/asicc.csv, clear
sort asicc
save Data/Panel_Data/Clean_data/asicc, replace

* Data directory
global ASIpaneldir Data/Panel_Data/Raw_data

* Import ASICC codes for inputs, imports, and output products
insheet using $ASIpaneldir/asicc.csv, clear
sort asicc
save $ASIpaneldir/asicc, replace

set more off

***************************************************************
* Import data from .txt to stata then save into temporary file
***************************************************************

* Years are in the following format: e.g. 2003 is associated with 2002-2003

* A and B
set more off
foreach j in 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 {
	foreach block in A B {
		quietly: infile using $ASIpaneldir/DictionaryFiles/1999-2008/Dictionary`block'.txt, using($ASIpaneldir/`j'/OASIBL`block'`j'.txt) clear
		drop block
		local actualyear=`j'
		foreach k of varlist Multiple{
			destring `k', replace
			}
		save $ASIpaneldir/temp/`actualyear'`block', replace
		}
}
foreach j in 2009 2010 2011 2012 2013 2014 2015 2016 {
	foreach block in A B {
		quietly: infile using $ASIpaneldir/DictionaryFiles/`j'/Dictionary`block'.txt, using($ASIpaneldir/`j'/OAS`block'`j'.txt) clear
		drop block
		local actualyear=`j'
		foreach k of varlist Multiple{
			destring `k', replace
			}
		save $ASIpaneldir/temp/`actualyear'`block', replace
	}
}

* C 
foreach j in 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 {
	foreach block in C {
		quietly: infile using $ASIpaneldir/DictionaryFiles/1999-2008/Dictionary`block'.txt, using($ASIpaneldir/`j'/OASIBL`block'`j'.txt) clear
		drop block
		local actualyear=`j'
		foreach k of varlist GrossOpen Revaluation Addition Deductions GrossClose DeprOpen Depreciation DeprClose Open Close Multiple{
			replace `k'= substr(`k', indexnot(`k', "0"), .)
			destring `k', replace
			}
		save $ASIpaneldir/temp/`actualyear'`block', replace
		}
}
foreach j in 2009 2010 2011 2012 2013 2014 2015 2016 {
	foreach block in C {
		quietly: infile using $ASIpaneldir/DictionaryFiles/`j'/Dictionary`block'.txt, using($ASIpaneldir/`j'/OAS`block'`j'.txt) clear
		drop block
		local actualyear=`j'
		foreach k of varlist GrossOpen Revaluation Addition Deductions GrossClose DeprOpen Depreciation DeprClose Open Close Multiple{
			replace `k'= substr(`k', indexnot(`k', "0"), .)
			destring `k', replace
			}
		save $ASIpaneldir/temp/`actualyear'`block', replace
	}
}

* D
foreach j in 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 {
	foreach block in D {
		quietly: infile using $ASIpaneldir/DictionaryFiles/1999-2008/Dictionary`block'.txt, using($ASIpaneldir/`j'/OASIBL`block'`j'.txt) clear
		drop block
		local actualyear=`j'
		foreach k of varlist Open Close Multiple{
			replace `k'= substr(`k', indexnot(`k', "0"), .)
			destring `k', replace
			}
		save $ASIpaneldir/temp/`actualyear'`block', replace
		}
}	
foreach j in 2009 2010 2011 2012 2013 2014 2015 2016 {
	foreach block in D {
		quietly: infile using $ASIpaneldir/DictionaryFiles/`j'/Dictionary`block'.txt, using($ASIpaneldir/`j'/OAS`block'`j'.txt) clear
		drop block
		local actualyear=`j'
		foreach k of varlist Open Close Multiple{
			replace `k'= substr(`k', indexnot(`k', "0"), .)
			destring `k', replace
			}
		save $ASIpaneldir/temp/`actualyear'`block', replace
	}
}

* E
foreach j in 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008{
	foreach block in E {
		quietly: infile using $ASIpaneldir/DictionaryFiles/1999-2008/Dictionary`block'.txt, using($ASIpaneldir/`j'/OASIBL`block'`j'.txt) clear
		drop block
		local actualyear=`j'
		foreach k of varlist DaysManuf DaysNonManuf Days Persons DaysPaid Wages Bonus PF Welfare Multiple{
			replace `k'= substr(`k', indexnot(`k', "0"), .)
			destring `k', replace
			}
		save $ASIpaneldir/temp/`actualyear'`block', replace
		}
}	
foreach j in 2009 2010 2011 2012 2013 2014 2015 2016 {
	foreach block in E {
		quietly: infile using $ASIpaneldir/DictionaryFiles/`j'/Dictionary`block'.txt, using($ASIpaneldir/`j'/OAS`block'`j'.txt) clear
		drop block
		local actualyear=`j'
		foreach k of varlist DaysManuf DaysNonManuf Days Persons DaysPaid Wages Bonus PF Welfare Multiple{
			replace `k'= substr(`k', indexnot(`k', "0"), .)
			destring `k', replace
			}
		save $ASIpaneldir/temp/`actualyear'`block', replace
	}
}

* F 
foreach j in 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 {
	foreach block in F {
		quietly: infile using $ASIpaneldir/DictionaryFiles/1999-2008/Dictionary`block'.txt, using($ASIpaneldir/`j'/OASIBL`block'`j'.txt) clear
		drop block
		local actualyear=`j'
		foreach k of varlist WorkDoneByOthers RepairMaintBuilding RepairMaintPlant RepairMaintPollution RepairMaintOther OperatingExpenses NonOperatingExpenses Insurance RentAssetsPaid Expenses RentBuildingsPaid RentLandPaid InterestPaid PurchaseValueGoodsResold Multiple{
			replace `k'= substr(`k', indexnot(`k', "0"), .)
			destring `k', replace
			}
		save $ASIpaneldir/temp/`actualyear'`block', replace
		}
}	
foreach j in 2009 2010 2011 2012 2013 2014 2015 2016 {
	foreach block in F {
		quietly: infile using $ASIpaneldir/DictionaryFiles/`j'/Dictionary`block'.txt, using($ASIpaneldir/`j'/OAS`block'`j'.txt) clear
		drop block
		local actualyear=`j'
		foreach k of varlist WorkDoneByOthers RepairMaintBuilding RepairMaintOther OperatingExpenses NonOperatingExpenses Insurance RentAssetsPaid Expenses RentBuildingsPaid RentLandPaid InterestPaid PurchaseValueGoodsResold Multiple{
			replace `k'= substr(`k', indexnot(`k', "0"), .)
			destring `k', replace
			}
		save $ASIpaneldir/temp/`actualyear'`block', replace
	}
}

* G
foreach j in 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 {
	foreach block in G {
		quietly: infile using $ASIpaneldir/DictionaryFiles/1999-2008/Dictionary`block'.txt, using($ASIpaneldir/`j'/OASIBL`block'`j'.txt) clear
		drop block
		local actualyear=`j'
		foreach k of varlist IncomeServices IncreaseStockSemiFinished ElectricitySold OwnConstruction NetSaleValueGoodsResold RentAssetsReceived Receipts RentBuildingsReceived RentLandReceived InterestReceived SaleValueGoodsResold Multiple{
			replace `k'= substr(`k', indexnot(`k', "0"), .)
			destring `k', replace
			}
		save $ASIpaneldir/temp/`actualyear'`block', replace
		}
}
foreach j in 2009 2010 2011 2012 2013 2014 2015 2016 {
	foreach block in G {
		if `j' <= 2015 {
			quietly: infile using $ASIpaneldir/DictionaryFiles/`j'/Dictionary`block'.txt, using($ASIpaneldir/`j'/OAS`block'`j'.txt) clear
			drop block
			local actualyear=`j'
			foreach k of varlist IncomeServices IncreaseStockSemiFinished ElectricitySold OwnConstruction NetSaleValueGoodsResold RentAssetsReceived Receipts RentBuildingsReceived RentLandReceived InterestReceived SaleValueGoodsResold Multiple{
				replace `k'= substr(`k', indexnot(`k', "0"), .)
				destring `k', replace
				}
		}
		else {
			quietly: infile using $ASIpaneldir/DictionaryFiles/`j'/Dictionary`block'.txt, using($ASIpaneldir/`j'/OAS`block'`j'.txt) clear
			drop block
			local actualyear=`j'
			foreach k of varlist IncomeServicesManuf IncomeServicesNonManuf IncreaseStockSemiFinished ElectricitySold OwnConstruction NetSaleValueGoodsResold RentAssetsReceived RentBuildingsReceived RentLandReceived InterestReceived SaleValueGoodsResold Multiple{
				replace `k'= substr(`k', indexnot(`k', "0"), .)
				destring `k', replace	
				}
		}
		save $ASIpaneldir/temp/`actualyear'`block', replace
	}
}

* H
foreach j in 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 {
	foreach block in H {
		quietly: infile using $ASIpaneldir/DictionaryFiles/1999-2008/Dictionary`block'.txt, using($ASIpaneldir/`j'/OASIBL`block'`j'.txt) clear
		drop block
		local actualyear=`j'
		foreach k of varlist asicc UnitCode QtyCons PurchVal UnitPrice Multiple{
			replace `k'= substr(`k', indexnot(`k', "0"), .)
			destring `k', replace
			}
		save $ASIpaneldir/temp/`actualyear'`block', replace
		}
}	
foreach j in 2009 2010 2011 2012 2013 2014 2015 2016 {
	foreach block in H {
		quietly: infile using $ASIpaneldir/DictionaryFiles/`j'/Dictionary`block'.txt, using($ASIpaneldir/`j'/OAS`block'`j'.txt) clear
		drop block
		local actualyear=`j'
		if `j' <= 2010 {
			foreach k of varlist asicc UnitCode QtyCons PurchVal UnitPrice Multiple{
			replace `k'= substr(`k', indexnot(`k', "0"), .)
			destring `k', replace
			}
		}
		else {
			foreach k of varlist npcms UnitCode QtyCons PurchVal UnitPrice Multiple{
			replace `k'= substr(`k', indexnot(`k', "0"), .)
			destring `k', replace
			}
		}
		save $ASIpaneldir/temp/`actualyear'`block', replace
	}
}

* I 
foreach j in 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 {
	foreach block in I {
		quietly: infile using $ASIpaneldir/DictionaryFiles/1999-2008/Dictionary`block'.txt, using($ASIpaneldir/`j'/OASIBL`block'`j'.txt) clear
		drop block
		local actualyear=`j'
		foreach k of varlist asicc UnitCode QtyCons PurchVal UnitPrice Multiple{
			replace `k'= substr(`k', indexnot(`k', "0"), .)
			destring `k', replace
			}
		save $ASIpaneldir/temp/`actualyear'`block', replace
		}
}	
foreach j in 2009 2010 2011 2012 2013 2014 2015 2016 {
	foreach block in I {
		quietly: infile using $ASIpaneldir/DictionaryFiles/`j'/Dictionary`block'.txt, using($ASIpaneldir/`j'/OAS`block'`j'.txt) clear
		drop block
		local actualyear=`j'
		if `j' <= 2010 {
			foreach k of varlist asicc UnitCode QtyCons PurchVal UnitPrice Multiple{
			replace `k'= substr(`k', indexnot(`k', "0"), .)
			destring `k', replace
			}
		}
		else {
			foreach k of varlist npcms UnitCode QtyCons PurchVal UnitPrice Multiple{
			replace `k'= substr(`k', indexnot(`k', "0"), .)
			destring `k', replace
			}
		}
		save $ASIpaneldir/temp/`actualyear'`block', replace
	}
}

* J
foreach j in 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 {
	foreach block in J {
		quietly: infile using $ASIpaneldir/DictionaryFiles/1999-2008/Dictionary`block'.txt, using($ASIpaneldir/`j'/OASIBL`block'`j'.txt) clear
		drop block
		local actualyear=`j'
		foreach k of varlist asicc UnitCode QtyManuf QtySold GrossSaleValue ExciseDuty SalesTax OtherTaxes DistributiveExpenses NetSaleValueUnit ExFactoryValue Multiple{
			replace `k'= substr(`k', indexnot(`k', "0"), .)
			destring `k', replace
			}
		save $ASIpaneldir/temp/`actualyear'`block', replace
		}
}	
foreach j in 2009 2010 2011 2012 2013 2014 2015 2016 {
	foreach block in J {
		quietly: infile using $ASIpaneldir/DictionaryFiles/`j'/Dictionary`block'.txt, using($ASIpaneldir/`j'/OAS`block'`j'.txt) clear
		drop block
		local actualyear=`j'
		if `j' <= 2010 {
			foreach k of varlist asicc UnitCode QtyManuf QtySold GrossSaleValue ExciseDuty SalesTax OtherTaxes DistributiveExpenses NetSaleValueUnit ExFactoryValue Multiple{
			replace `k'= substr(`k', indexnot(`k', "0"), .)
			destring `k', replace
			}
		}
		else {
			foreach k of varlist npcms UnitCode QtyManuf QtySold GrossSaleValue ExciseDuty SalesTax OtherTaxes DistributiveExpenses NetSaleValueUnit ExFactoryValue Multiple{
			replace `k'= substr(`k', indexnot(`k', "0"), .)
			destring `k', replace
			}
		}
		save $ASIpaneldir/temp/`actualyear'`block', replace
	}
}


*************************************************************************
* Remove duplicates and convert long to wide
*************************************************************************

*** Blocks with only one observaton per plant/year pair
* Flag duplicates and remove the same observations in each block

* Block A
forvalues year = 1999/2016 {
	foreach block in A {
		use $ASIpaneldir/temp/`year'`block', clear
		duplicates report ID
		gen ones=1
		bysort ID: egen count=sum(ones)
		gen temp=1 if Open==1
		bysort ID: egen test=sum(temp)
		* Doesn't drop yet - generates a drop flag
		gen drop=1 if count==2 & Open!=1 & test>0 & test!=.
		drop temp test
		gen temp=1 if Days>0 & Days!=.
		bysort ID: egen test=sum(temp)
		replace drop=1 if count==2 & (Days==. | Days==0) & test>0 & test!=.
		sort ID
		by ID: gen num=_n
		replace drop=1 if num>1 & drop==.
		drop count temp test ones num
		sort ID Multiple
		save $ASIpaneldir/temp/`year'`block'_clean, replace
	}
}
* Block B: only the ID variable is duplicated from block A, so we independently decide which duplicates to drop 
forvalues year = 1999/2016 {
	foreach block in B {
		use $ASIpaneldir/temp/`year'`block', clear
		duplicates report ID
		gen ones=1
		bysort ID: egen count=sum(ones)
		gen temp=1 if MonthsOperation>0 & MonthsOperation!=.
		bysort ID: egen test=sum(temp)
		drop if count==2 & (MonthsOperation==. | MonthsOperation==0) & test>0 & test!=.
		sort ID
		by ID: gen num=_n
		drop if num>1
		drop count temp test ones num
		sort ID
		save $ASIpaneldir/temp/`year'`block'_clean, replace
	}
}

* Block F
forvalues year = 1999/2016 {
	foreach block in F {
		use $ASIpaneldir/temp/`year'`block', clear
		duplicates report ID
		* Drops any observations already flagged for dropping from block a
		sort ID Multiple
		merge ID Multiple using $ASIpaneldir/temp/`year'A_clean
		tab _merge
		drop _merge
		drop if drop==1
		drop SchemeCode-drop
		* Drops any other duplicates remaining
		gen ones=1
		bysort ID: egen count=sum(ones)
		gen temp=1 if Expenses!=0 & Expenses!=.
		bysort ID: egen test=sum(temp)
		drop if count==2 & (Expenses==. | Expenses==0) & test>0 & test!=.
		sort ID
		by ID: gen num=_n
		drop if num>1
		drop count temp test ones num
		sort ID
		save $ASIpaneldir/temp/`year'`block'_clean, replace
	}
}

* Block G
set more off
forvalues year = 1999/2016 {
	foreach block in G {
		use $ASIpaneldir/temp/`year'`block', clear
		duplicates report ID
		* Drops any observations already flagged for dropping from block a
		sort ID Multiple
		merge ID Multiple using $ASIpaneldir/temp/`year'A_clean
		tab _merge
		drop _merge
		drop if drop==1
		drop SchemeCode-drop
		* Drops any other duplicates remaining
		gen ones=1
		bysort ID: egen count=sum(ones)
		* Check if Receipts exist, and create if it doesn't
		capture confirm var Receipts
		if !_rc {
			di in red "Receipts exist"
		}
		else {
			egen Receipts = rowtotal(IncomeServicesManuf IncomeServicesNonManuf ElectricitySold OwnConstruction NetSaleValueGoodsResold RentAssetsReceived IncreaseStockSemiFinished)
		}
		gen temp=1 if Receipts!=0 & Receipts!=.
		bysort ID: egen test=sum(temp)
		drop if count==2 & (Receipts==. | Receipts==0) & test>0 & test!=.
		sort ID
		by ID: gen num=_n
		drop if num>1
		drop count temp test ones num
		sort ID
		save $ASIpaneldir/temp/`year'`block'_clean, replace
	}
}

* Goes back to block A and drops duplicates already flagged
forvalues year = 1999/2016 {
	foreach block in A {
		use $ASIpaneldir/temp/`year'`block'_clean, clear
		drop if drop==1
		drop drop
		sort ID
		save $ASIpaneldir/temp/`year'`block'_clean, replace
	}
}


*** Blocks with multiple observations per plant/year pair

* Block C: Fixed Assets

set more off
foreach year in 1999 2000 2001 {
	use $ASIpaneldir/temp/`year'C, clear
	gen FixAssetName=""
	replace FixAssetName="Land" if SI==1
	replace FixAssetName="Building" if SI==2
	replace FixAssetName="PlantMachine" if SI==3
	replace FixAssetName="Transport" if SI==4
	replace FixAssetName="Computer" if SI==5
	replace FixAssetName="OtherFixCap" if SI==6
	drop if SI==7
	replace FixAssetName="WIP" if SI==8
	replace FixAssetName="FixCap" if SI==9
	* Drop GrossOpen Revaluation Addition GrossClose DeprOpen Depreciation
	collapse (sum) Open Close Revaluation Depreciation, by(ID FixAssetName)
	drop if FixAssetName==""
	reshape wide Open Close Revaluation Depreciation, i(ID) j(FixAssetName) string
	foreach y in Open Close Revaluation Depreciation {
		foreach x in Land Building PlantMachine Transport Computer OtherFixCap WIP FixCap {
			rename `y'`x' `x'`y'
		}
		* Check sums, replace if FixCap is 0 or missing
		egen double FixCap`y'_ManSum=rsum(Land`y' Building`y' PlantMachine`y' Transport`y' Computer`y' OtherFixCap`y' WIP`y')
		replace FixCap`y'=FixCap`y'_ManSum if FixCap`y'==0 | FixCap`y'==.
		drop FixCap`y'_ManSum
	}
	order ID Land* Building* PlantMachine* Transport* Computer* OtherFixCap* WIP* FixCapOp* FixCapCl*
	sort ID
	save $ASIpaneldir/temp/`year'C_wide, replace
}
* These years include Pollution Control assets
forvalues year = 2002/2016 {	
	use $ASIpaneldir/temp/`year'C, clear
	gen FixAssetName=""
	replace FixAssetName="Land" if SI==1
	replace FixAssetName="Building" if SI==2
	replace FixAssetName="PlantMachine" if SI==3
	replace FixAssetName="Transport" if SI==4
	replace FixAssetName="Computer" if SI==5
	replace FixAssetName="Pollution" if SI==6
	replace FixAssetName="OtherFixCap" if SI==7
	drop if SI==8
	replace FixAssetName="WIP" if SI==9
	replace FixAssetName="FixCap" if SI==10
	* Drop GrossOpen Revaluation Addition Deductions GrossClose DeprOpen Depreciation DeprClose
	collapse (sum) Open Close Revaluation Depreciation, by(ID FixAssetName)
	drop if FixAssetName==""
	reshape wide Open Close Revaluation Depreciation, i(ID) j(FixAssetName) string
	foreach y in Open Close Revaluation Depreciation {
		foreach x in Land Building PlantMachine Transport Computer Pollution OtherFixCap WIP FixCap {
			rename `y'`x' `x'`y'
		}
		* Check sums, only replace if FixCap is 0 or missing
		egen double FixCap`y'_ManSum=rsum(Land`y' Building`y' PlantMachine`y' Transport`y' Computer`y' Pollution`y'  OtherFixCap`y' WIP`y')
		replace FixCap`y'=FixCap`y'_ManSum if FixCap`y'==0 | FixCap`y'==.
		drop FixCap`y'_ManSum
	}
	order ID Land* Building* PlantMachine* Transport* Computer* Pollution* OtherFixCap* WIP* FixCapOp* FixCapCl*
	sort ID
	sort ID
	save $ASIpaneldir/temp/`year'C_wide, replace
}

* Block D: Working Capital and Loans
forvalues year = 1999/2016 {
	use $ASIpaneldir/temp/`year'D, clear
	gen WorkCapName=""
	replace WorkCapName="RawMaterials" if SI==1
	replace WorkCapName="Fuels" if SI==2
	replace WorkCapName="Spares" if SI==3
	drop if SI==4
	replace WorkCapName="SemiFinishedGoods" if SI==5
	replace WorkCapName="FinishedGoods" if SI==6
	replace WorkCapName="Inventory" if SI==7
	replace WorkCapName="Cash" if SI==8
	replace WorkCapName="SundryDebtors" if SI==9
	replace WorkCapName="OtherCurrentAssets" if SI==10
	replace WorkCapName="CurrentAssets" if SI==11
	replace WorkCapName="SundryCreditors" if SI==12
	replace WorkCapName="ShortTermLoans" if SI==13
	replace WorkCapName="OtherCurrentLiabilities" if SI==14
	replace WorkCapName="CurrentLiabilities" if SI==15
	replace WorkCapName="WorkCap" if SI==16
	replace WorkCapName="Loans" if SI==17
	collapse (sum) Open Close, by(ID WorkCapName)
	drop if WorkCapName==""
	reshape wide Open Close, i(ID) j(WorkCapName) string
	foreach y in Open Close {
		foreach x in RawMaterials Fuels Spares SemiFinishedGoods FinishedGoods Inventory Cash SundryDebtors OtherCurrentAssets ///
			CurrentAssets SundryCreditors ShortTermLoans OtherCurrentLiabilities CurrentLiabilities WorkCap Loans {
			rename `y'`x' `x'`y'
			replace `x'`y'=0 if `x'`y'==.
		}
		 gen Stock`y'= RawMaterials`y' + Fuels`y' + Spares`y'
		 move Stock`y' SemiFinishedGoods`y'
	}
	sort ID
	save $ASIpaneldir/temp/`year'D_wide, replace
}

* Block E: Labor
foreach year in 1999 2000 {
	use $ASIpaneldir/temp/`year'E, clear
	gen LaborName=""	
	replace LaborName="Men" if SI==1
	replace LaborName="Women" if SI==2
	replace LaborName="Child" if SI==3
	drop if SI==4
	replace LaborName="Contractors" if SI==5
	replace LaborName="Workers" if SI==6
	replace LaborName="Superv" if SI==7
	replace LaborName="Other" if SI==8
	replace LaborName="Total" if SI==9
	collapse (sum) DaysManuf DaysNonManuf Days Persons DaysPaid Wages Bonus PF Welfare, by(ID LaborName)
	order ID LaborName
	reshape wide DaysManuf DaysNonManuf Days Persons DaysPaid Wages Bonus PF Welfare, i(ID) j(LaborName) string
	sort ID
	save $ASIpaneldir/temp/`year'E_wide, replace
}
forvalues year = 2001/2008 {
	use $ASIpaneldir/temp/`year'E, clear
	gen LaborName=""	
	replace LaborName="Men" if SI==1
	replace LaborName="Women" if SI==2
	replace LaborName="Child" if SI==3
	drop if SI==4
	replace LaborName="Contractors" if SI==5
	replace LaborName="Workers" if SI==6
	replace LaborName="Superv" if SI==7
	replace LaborName="Other" if SI==8
	replace LaborName="Unpaid" if SI==9
	replace LaborName="Total" if SI==10
	collapse (sum) DaysManuf DaysNonManuf Days DaysPaid Persons Wages Bonus PF Welfare, by(ID LaborName)
	order ID LaborName
	reshape wide DaysManuf DaysNonManuf Days DaysPaid Persons Wages Bonus PF Welfare, i(ID) j(LaborName) string
	sort ID
	save $ASIpaneldir/temp/`year'E_wide, replace
}
forvalues year = 2009/2016 {
	use $ASIpaneldir/temp/`year'E, clear
	gen LaborName=""	
	replace LaborName="Men" if SI==1
	replace LaborName="Women" if SI==2
	drop if SI==3
	replace LaborName="Contractors" if SI==4
	replace LaborName="Workers" if SI==5
	replace LaborName="Superv" if SI==6
	replace LaborName="Other" if SI==7
	replace LaborName="Unpaid" if SI==8
	replace LaborName="Total" if SI==9
	replace LaborName="TotalExtra" if SI==10
	collapse (sum) DaysManuf DaysNonManuf Days DaysPaid Persons Wages Bonus PF Welfare, by(ID LaborName)
	order ID LaborName
	reshape wide DaysManuf DaysNonManuf Days DaysPaid Persons Wages Bonus PF Welfare, i(ID) j(LaborName) string
	sort ID
	save $ASIpaneldir/temp/`year'E_wide, replace
}

* Block H: Inputs domestic
forvalues year = 2000/2003 {
	use $ASIpaneldir/temp/`year'H, clear
	sort asicc
	merge asicc using $ASIpaneldir/asicc
	drop if _merge==2
	drop _merge
	sort ID SI
	gen InputName=""
	replace InputName="Input1" if SI == 1
	replace InputName="Input2" if SI == 2
	replace InputName="Input3" if SI == 3
	replace InputName="Input4" if SI == 4
	replace InputName="Input5" if SI == 5
	replace InputName="Other" if SI == 6
	replace InputName="TotalBasicItem" if SI == 7
	replace InputName="Chemical" if SI == 8
	replace InputName="Packing" if SI == 9
	replace InputName="ElecOwn" if SI == 10
	replace InputName="ElecBought" if SI == 11
	replace InputName="Oil" if SI == 12
	replace InputName="Coal" if SI == 13
	replace InputName="OtherFuel" if SI == 14
	replace InputName="Consumable" if SI == 15
	replace InputName="TotalNonBasic" if SI == 16
	replace InputName="Total" if SI == 17
	replace InputName="UnmetElecDemand" if SI == 18
	order ID SI InputName
	drop if InputName==""
	collapse (sum) QtyCons PurchVal UnitPrice UnitCode asicc, by(ID InputName)
	reshape wide QtyCons PurchVal UnitPrice UnitCode asicc, i(ID) j(InputName) string
	save $ASIpaneldir/temp/`year'H_wide, replace
}
forvalues year = 2004/2008 {
	use $ASIpaneldir/temp/`year'H, clear
	sort asicc
	merge asicc using $ASIpaneldir/asicc
	drop if _merge==2
	drop _merge
	sort ID SI
	gen InputName=""
	replace InputName="Input1" if SI == 1
	replace InputName="Input2" if SI == 2
	replace InputName="Input3" if SI == 3
	replace InputName="Input4" if SI == 4
	replace InputName="Input5" if SI == 5
	replace InputName="Input6" if SI == 6
	replace InputName="Input7" if SI == 7
	replace InputName="Input8" if SI == 8
	replace InputName="Input9" if SI == 9
	replace InputName="Input10" if SI == 10
	replace InputName="Other" if SI == 11
	replace InputName="TotalBasicItem" if SI == 12
	replace InputName="Chemical" if SI == 13
	replace InputName="Packing" if SI == 14
	replace InputName="ElecOwn" if SI == 15
	replace InputName="ElecBought" if SI == 16
	replace InputName="Oil" if SI == 17
	replace InputName="Coal" if SI == 18
	replace InputName="OtherFuel" if SI == 19
	replace InputName="Consumable" if SI == 20
	replace InputName="TotalNonBasic" if SI == 21
	replace InputName="Total" if SI == 22
	replace InputName="UnmetElecDemand" if SI == 23
	order ID SI InputName
	drop if InputName==""
	collapse (sum) QtyCons PurchVal UnitPrice UnitCode asicc, by(ID InputName)
	reshape wide QtyCons PurchVal UnitPrice UnitCode asicc, i(ID) j(InputName) string
	save $ASIpaneldir/temp/`year'H_wide, replace
}
forvalues year = 2009/2010 {
	use $ASIpaneldir/temp/`year'H, clear
	sort asicc
	merge asicc using $ASIpaneldir/asicc
	drop if _merge==2
	drop _merge
	sort ID SI
	gen InputName=""
	replace InputName="Input1" if SI == 1
	replace InputName="Input2" if SI == 2
	replace InputName="Input3" if SI == 3
	replace InputName="Input4" if SI == 4
	replace InputName="Input5" if SI == 5
	replace InputName="Input6" if SI == 6
	replace InputName="Input7" if SI == 7
	replace InputName="Input8" if SI == 8
	replace InputName="Input9" if SI == 9
	replace InputName="Input10" if SI == 10
	replace InputName="Other" if SI == 11
	replace InputName="TotalBasicItem" if SI == 12
	replace InputName="Chemical" if SI == 13
	replace InputName="Packing" if SI == 14
	replace InputName="ElecOwn" if SI == 15
	replace InputName="ElecBought" if SI == 16
	replace InputName="Oil" if SI == 17
	replace InputName="Coal" if SI == 18
	replace InputName="Gas" if SI == 19
	replace InputName="OtherFuel" if SI == 20
	replace InputName="Consumable" if SI == 21
	replace InputName="TotalNonBasic" if SI == 22
	replace InputName="Total" if SI == 23
	replace InputName="UnmetElecDemand" if SI == 24
	order ID SI InputName
	drop if InputName==""
	collapse (sum) QtyCons PurchVal UnitPrice UnitCode asicc, by(ID InputName)
	reshape wide QtyCons PurchVal UnitPrice UnitCode asicc, i(ID) j(InputName) string
	save $ASIpaneldir/temp/`year'H_wide, replace
}
forvalues year = 2011/2016 {
	use $ASIpaneldir/temp/`year'H, clear
	sort npcms
	sort ID SI
	gen InputName=""
	replace InputName="Input1" if SI == 1
	replace InputName="Input2" if SI == 2
	replace InputName="Input3" if SI == 3
	replace InputName="Input4" if SI == 4
	replace InputName="Input5" if SI == 5
	replace InputName="Input6" if SI == 6
	replace InputName="Input7" if SI == 7
	replace InputName="Input8" if SI == 8
	replace InputName="Input9" if SI == 9
	replace InputName="Input10" if SI == 10
	replace InputName="Other" if SI == 11
	replace InputName="TotalBasicItem" if SI == 12
	replace InputName="Chemical" if SI == 13
	replace InputName="Packing" if SI == 14
	replace InputName="ElecOwn" if SI == 15
	replace InputName="ElecBought" if SI == 16
	replace InputName="Oil" if SI == 17
	replace InputName="Coal" if SI == 18
	replace InputName="Gas" if SI == 19
	replace InputName="OtherFuel" if SI == 20
	replace InputName="Consumable" if SI == 21
	replace InputName="TotalNonBasic" if SI == 22
	replace InputName="Total" if SI == 23
	replace InputName="UnmetElecDemand" if SI == 24
	order ID SI InputName
	drop if InputName==""
	collapse (sum) QtyCons PurchVal UnitPrice UnitCode npcms, by(ID InputName)
	reshape wide QtyCons PurchVal UnitPrice UnitCode npcms, i(ID) j(InputName) string
	save $ASIpaneldir/temp/`year'H_wide, replace
}

*Block I: Inputs Imported
forvalues year = 2000/2010 {
	use $ASIpaneldir/temp/`year'I, clear
	drop YearCode
	sort asicc
	merge asicc using $ASIpaneldir/asicc
	drop if _merge==2
	drop _merge
	replace QtyCons=. if (UnitCode==999 & QtyCons == 0) | (UnitCode == . & QtyCons==0)
	sort ID SI
	gen InputName=""
	replace InputName="Import1" if SI == 1
	replace InputName="Import2" if SI == 2
	replace InputName="Import3" if SI == 3
	replace InputName="Import4" if SI == 4
	replace InputName="Import5" if SI == 5
	replace InputName="ImportOther" if SI == 6
	replace InputName="ImportTotal" if SI == 7
	order ID SI asicc PurchVal QtyCons UnitPrice UnitCode Multiple
	drop if InputName==""
	collapse (sum) QtyCons PurchVal UnitPrice UnitCode asicc, by(ID InputName)
	reshape wide QtyCons PurchVal UnitPrice UnitCode asicc, i(ID) j(InputName) string
	save $ASIpaneldir/temp/`year'I_wide, replace
}
forvalues year = 2011/2016 {
	use $ASIpaneldir/temp/`year'I, clear
	drop YearCode
	sort npcms
	replace QtyCons=. if (UnitCode==999 & QtyCons == 0) | (UnitCode == . & QtyCons==0)
	sort ID SI
	gen InputName=""
	replace InputName="Import1" if SI == 1
	replace InputName="Import2" if SI == 2
	replace InputName="Import3" if SI == 3
	replace InputName="Import4" if SI == 4
	replace InputName="Import5" if SI == 5
	replace InputName="ImportOther" if SI == 6
	replace InputName="ImportTotal" if SI == 7
	order ID SI npcms PurchVal QtyCons UnitPrice UnitCode Multiple
	drop if InputName==""
	collapse (sum) QtyCons PurchVal UnitPrice UnitCode npcms, by(ID InputName)
	reshape wide QtyCons PurchVal UnitPrice UnitCode npcms, i(ID) j(InputName) string
	save $ASIpaneldir/temp/`year'I_wide, replace
}

*Block J: Output
forvalues year = 2000/2010 {
	use $ASIpaneldir/temp/`year'J, clear
	drop YearCode
	sort asicc
	merge asicc using $ASIpaneldir/asicc
	drop if _merge==2
	drop _merge
	sort ID SI
	gen OutputName=""
	replace OutputName="Output1" if SI == 1
	replace OutputName="Output2" if SI == 2
	replace OutputName="Output3" if SI == 3
	replace OutputName="Output4" if SI == 4
	replace OutputName="Output5" if SI == 5
	replace OutputName="Output6" if SI == 6
	replace OutputName="Output7" if SI == 7
	replace OutputName="Output8" if SI == 8
	replace OutputName="Output9" if SI == 9
	replace OutputName="Output10" if SI == 10
	replace OutputName="Other" if SI == 11
	replace OutputName="Total" if SI == 12
	order ID SI asicc UnitCode QtyManuf QtySold GrossSaleValue DistributiveExpenses NetSaleValueUnit ExFactoryValue
	drop if OutputName==""
	collapse (sum) asicc UnitCode QtyManuf QtySold GrossSaleValue DistributiveExpenses NetSaleValueUnit ExFactoryValue, by(ID OutputName)
	reshape wide asicc UnitCode QtyManuf QtySold GrossSaleValue DistributiveExpenses NetSaleValueUnit ExFactoryValue, i(ID) j(OutputName) string
	save $ASIpaneldir/temp/`year'J_wide, replace
}
forvalues year = 2011/2016 {
	use $ASIpaneldir/temp/`year'J, clear
	drop YearCode
	sort npcms
	sort ID SI
	gen OutputName=""
	replace OutputName="Output1" if SI == 1
	replace OutputName="Output2" if SI == 2
	replace OutputName="Output3" if SI == 3
	replace OutputName="Output4" if SI == 4
	replace OutputName="Output5" if SI == 5
	replace OutputName="Output6" if SI == 6
	replace OutputName="Output7" if SI == 7
	replace OutputName="Output8" if SI == 8
	replace OutputName="Output9" if SI == 9
	replace OutputName="Output10" if SI == 10
	replace OutputName="Other" if SI == 11
	replace OutputName="Total" if SI == 12
	order ID SI npcms UnitCode QtyManuf QtySold GrossSaleValue DistributiveExpenses NetSaleValueUnit ExFactoryValue
	drop if OutputName==""
	collapse (sum) npcms UnitCode QtyManuf QtySold GrossSaleValue DistributiveExpenses NetSaleValueUnit ExFactoryValue, by(ID OutputName)
	reshape wide npcms UnitCode QtyManuf QtySold GrossSaleValue DistributiveExpenses NetSaleValueUnit ExFactoryValue, i(ID) j(OutputName) string
	save $ASIpaneldir/temp/`year'J_wide, replace
}

*************************************************************************
* Merge into single dataset and delete temporary files
*************************************************************************

forvalues year = 2000/2016 {
	use $ASIpaneldir/temp/`year'A_clean, clear
	gen StateCode = substr(ID,6,2)
	destring StateCode, replace force
	foreach block in B F G {
		merge ID using $ASIpaneldir/temp/`year'`block'_clean
		drop _merge
		sort ID
	}
	foreach block in C D E H I J {
		merge ID using $ASIpaneldir/temp/`year'`block'_wide
		drop _merge
		sort ID
	}
	save $ASIpaneldir/temp/ASI`year', replace
}
use $ASIpaneldir/temp/ASI2000, clear
forvalues year = 2001/2016 {
	append using $ASIpaneldir/temp/ASI`year'
}
save "Data/Panel_Data/Clean_data/ASI_PanelClean", replace

forvalues year = 1999/2016{
	foreach block in A B F G {
	erase $ASIpaneldir/temp/`year'`block'.dta
	erase $ASIpaneldir/temp/`year'`block'_clean.dta
	}
}
forvalues year = 2000/2016{
	foreach block in C D E H I J {
	erase $ASIpaneldir/temp/`year'`block'.dta
	erase $ASIpaneldir/temp/`year'`block'_wide.dta
	}
}
erase $ASIpaneldir/temp/1999C.dta
erase $ASIpaneldir/temp/1999C_wide.dta
erase $ASIpaneldir/temp/1999D.dta
erase $ASIpaneldir/temp/1999D_wide.dta
erase $ASIpaneldir/temp/1999E.dta
erase $ASIpaneldir/temp/1999E_wide.dta
erase $ASIpaneldir/temp/1999H.dta
erase $ASIpaneldir/temp/1999I.dta
erase $ASIpaneldir/temp/1999J.dta
forvalues year = 2000/2016{
	erase $ASIpaneldir/temp/ASI`year'.dta
}

