*** CODEFILE 10 ***

**** This creates a figure of the distribution of fuel productivity, including fuel productivity for counterfactual fuel set ***
* NOTE: You must estimate the dynamic parameters in julia before creating this figure

* Import dataset prior to dynamic estimation
import delimited "Data/Dynamics/MainData_wPipeline-Steel.csv", clear
rename idnum IDnum
rename combinef combineF
reg lnrfprod_qty_e 
predict lnrfprod_avg, xb
keep IDnum year bo bg bc be g_gmean c_gmean elec_gmean o_gmean ll fprod_g fprod_c fprod_e fprod_o combineF lnrfprod_avg 
tempfile maindata
save `maindata'
* Import dataset with coal and gas productivity from distribution of counterfactual comparative advantages and reshape to long (after dynamic estimation)
import delimited "Data/Dynamics/PostEstimation/Data_prod.csv", clear
rename idnum IDnum
sort IDnum year
xtset IDnum year
* Merge with original dataset
merge 1:1 IDnum year using `maindata'
keep if _merge == 3
drop _merge
* Reshape to long
reshape long lnfprod_g lnfprod_c, i(IDnum year) j(simulation)
* create physical productivity measure
gen fprod_g_est = exp(lnfprod_g)
gen fprod_c_est = exp(lnfprod_c)
gen rfprod_g = (bg^(ll/(ll-1)))*exp(lnfprod_g_old)/g_gmean if combineF == 124 | combineF == 1234
gen rfprod_c = (bc^(ll/(ll-1)))*exp(lnfprod_c_old)/c_gmean if combineF == 123 | combineF == 1234
replace rfprod_g = (bg^(ll/(ll-1)))*fprod_g_est/g_gmean if combineF == 12 | combineF == 123
replace rfprod_c = (bc^(ll/(ll-1)))*fprod_c_est/c_gmean if combineF == 12 | combineF == 124
gen lnrfprod_g = log(rfprod_g)
gen lnrfprod_c = log(rfprod_c)

* Create graph of distribution by fuel set
preserve 
	* Organize data
	foreach f in g c {
		gen lnrfprod_qty_`f' = lnrfprod_`f' - lnrfprod_avg
	}
	rename lnrfprod_qty_g lnfprod1
	rename lnrfprod_qty_c lnfprod2
// 	keep if simulation == 5
	reshape long lnfprod, i(IDnum year simulation) j(fuel)
	collapse (mean) y = lnfprod (semean) se_y = lnfprod, by(fuel combineF)
	gen combineF_sort = 1 if combineF == 12
	replace combineF_sort = 2 if combineF == 124
	replace combineF_sort = 3 if combineF == 123
	replace combineF_sort = 4 if combineF == 1234
	sort fuel combineF_sort 
	gen x = _n
	gen yu = y + 1.96*se_y
	gen yl = y - 1.96*se_y
	* Create figure
	twoway ///
	(rcap yl yu x, vert lcolor(gray)) /// code for 95% CI
	(scatter y x if (fuel == 1 & combineF == 124) | (fuel == 1 & combineF == 1234), mcolor(cranberry) msymbol(circle) msize(5pt)) /// 
	(scatter y x if (fuel == 1 & combineF == 12) | (fuel == 1 & combineF == 123), mcolor(erose) msymbol(circle) msize(5pt)) /// 
	(scatter y x if (fuel == 2 & combineF == 123) | (fuel == 2 & combineF == 1234), mcolor(navy) msymbol(diamond) msize(5pt)) ///
	(scatter y x if (fuel == 2 & combineF == 12) | (fuel == 2 & combineF == 124), mcolor(eltblue) msymbol(diamond) msize(5pt)) ///
	, legend(row(2) order(2 "Natural Gas" 4 "Coal" 3 "Natural Gas (Counterfactual)" 5 "Coal (Counterfactual)") pos(10) ring(0) size(11pt)) ///
	xlabel(1 "oe" 3 "oce", labcolor(erose) angle(0) noticks labsize(11pt))  ///
	xlabel(5 "oe" 6 "oge", add custom labcolor(eltblue) angle(0) noticks labsize(11pt))  ///
	xlabel(2 "oge" 4 "ogce" 7 "oce" 8 "ogce", add custom angle(0) noticks labsize(11pt) labcolor(black)) ///
	xline(4.5, lpattern(dash) lcolor(gray)) ///
	xtitle("Fuel set",size(12pt)) ytitle("(log) fuel productivity",size(12pt)) graphregion(color(white))
	graph export "Output/Graphs/Counterfactual/fprod_qty_allfuel_allsetf-steel.pdf", replace
restore