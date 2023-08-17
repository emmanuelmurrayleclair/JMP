*** CODEFILE 3 ***

**** This file prepares the panel ASI data

* Data directory
global ASIpaneldir Data/Panel_Data/Raw_data

***********************************************************
* 1. INDUSTRY CONCONCORDANCES BETWEEN 1998, 2004, 2008
***********************************************************

* Prepare industry concordances
cd Data/Panel_Data/Raw_data
use ../Clean_data/ASI_PanelClean, clear
destring YearCode, gen(year)
drop if year == 9900
* Impute missing 4-digit industries
sort IndCodeFrame
replace IndCodeFrame = int(IndCodeReturn/10) if IndCodeFrame == 9999
* NIC04-NIC08
gen ind4d = IndCodeFrame if year > 2004
if year < 2009 & year > 2004 {
	replace ind4d = 1010 if ind4d == 1511
	replace ind4d = 1020 if ind4d == 1512
	replace ind4d = 1030 if ind4d == 1513
	replace ind4d = 1040 if ind4d == 1514
	replace ind4d = 1050 if ind4d == 1520
	replace ind4d = 1061 if ind4d == 1531
	replace ind4d = 1062 if ind4d == 1532
	replace ind4d = 1071 if ind4d == 1541
	replace ind4d = 1072 if ind4d == 1542
	replace ind4d = 1073 if ind4d == 1543
	replace ind4d = 1074 if ind4d == 1544
	replace ind4d = 1075 if (ind4d == 1512) | (ind4d == 1513) | (ind4d == 1544) | (ind4d == 1549) 
	replace ind4d = 1079 if (ind4d == 1549) | (ind4d == 2429)
	replace ind4d = 1080 if ind4d == 1533
	replace ind4d = 1101 if ind4d == 1551
	replace ind4d = 1102 if (ind4d == 0113) | (ind4d == 1552)
	replace ind4d = 1103 if ind4d == 1553
	replace ind4d = 1105 if ind4d == 1554
	replace ind4d = 1200 if ind4d == 1600
	replace ind4d = 1311 if (ind4d == 1711) | (ind4d == 1713)
	replace ind4d = 1313 if (ind4d == 1712) | (ind4d == 1714)
	replace ind4d = 1391 if ind4d == 1730
	replace ind4d = 1392 if (ind4d == 1721) | (ind4d == 1722) | (ind4d == 1725)
	replace ind4d = 1394 if ind4d == 1723
	replace ind4d = 1399 if (ind4d == 1724) | (ind4d == 1729)
	replace ind4d = 1410 if ind4d == 1810
	replace ind4d = 1420 if ind4d == 1820
	replace ind4d = 1430 if ind4d == 1730
	replace ind4d = 1511 if (ind4d == 1820) | (ind4d == 1911)
	replace ind4d = 1512 if (ind4d == 1912) | (ind4d == 3966)
	replace ind4d = 1520 if ind4d == 1920
	replace ind4d = 1610 if ind4d == 2010
	replace ind4d = 1621 if ind4d == 2021
	replace ind4d = 1622 if ind4d == 2022
	replace ind4d = 1623 if ind4d == 2023
	replace ind4d = 1629 if (ind4d == 2029) | (ind4d == 3699)
	replace ind4d = 1701 if ind4d == 2101
	replace ind4d = 1702 if ind4d == 2102
	replace ind4d = 1709 if (ind4d == 2109) | (ind4d == 3699)
	replace ind4d = 1811 if ind4d == 2221
	replace ind4d = 1812 if ind4d == 2222
	replace ind4d = 1910 if ind4d == 2310
	replace ind4d = 1920 if (ind4d == 1010)| (ind4d == 1020) | (ind4d == 2320)
	replace ind4d = 2011 if (ind4d == 2330)| (ind4d == 2411) | (ind4d == 2429)
	replace ind4d = 2012 if ind4d == 2412
	replace ind4d = 2013 if ind4d == 2413
	replace ind4d = 2021 if ind4d == 2421
	replace ind4d = 2022 if ind4d == 2422
	replace ind4d = 2023 if ind4d == 2424
	replace ind4d = 2029 if ind4d == 2429
	replace ind4d = 2030 if ind4d == 2430
	replace ind4d = 2100 if ind4d == 2423
	replace ind4d = 2211 if ind4d == 2511
	replace ind4d = 2219 if ind4d == 2519
	replace ind4d = 2220 if ind4d == 2520
	replace ind4d = 2310 if ind4d == 2610
	replace ind4d = 2391 if ind4d == 2692
	replace ind4d = 2392 if ind4d == 2693
	replace ind4d = 2393 if ind4d == 2691
	replace ind4d = 2394 if ind4d == 2694
	replace ind4d = 2395 if ind4d == 2695
	replace ind4d = 2396 if ind4d == 2696
	replace ind4d = 2399 if ind4d == 2699
	replace ind4d = 2410 if (ind4d == 2711) | (ind4d == 2712) | (ind4d == 2713) | (ind4d == 2714)| (ind4d == 2715) | (ind4d == 2716) | (ind4d == 2717)| (ind4d == 2718) | (ind4d == 2719)
	replace ind4d = 2424 if ind4d == 2720
	replace ind4d = 2431 if ind4d == 2731
	replace ind4d = 2432 if ind4d == 2732
	replace ind4d = 2511 if ind4d == 2811
	replace ind4d = 2512 if ind4d == 2812
	replace ind4d = 2513 if ind4d == 2813 
	replace ind4d = 2520 if ind4d == 2927
	replace ind4d = 2591 if ind4d == 2891
	replace ind4d = 2592 if ind4d == 2892
	replace ind4d = 2593 if (ind4d == 2893) | (ind4d == 2929)
	replace ind4d = 2599 if ind4d == 2899
	replace ind4d = 2610 if ind4d == 3210
	replace ind4d = 2620 if ind4d == 3000
	replace ind4d = 2630 if ind4d == 3220
	replace ind4d = 2640 if ind4d == 3230
	replace ind4d = 2651 if ind4d == 3312 | ind4d == 3313
	replace ind4d = 2652 if ind4d == 3330
	replace ind4d = 2660 if ind4d == 3311
	replace ind4d = 2670 if ind4d == 3312 | ind4d == 3313
	replace ind4d = 2652 if ind4d == 3330
	replace ind4d = 2660 if ind4d == 3311
	replace ind4d = 2670 if ind4d == 3312 | ind4d == 3320
	replace ind4d = 2680 if ind4d == 2429
	replace ind4d = 2710 if ind4d == 3110 | ind4d == 3120
	replace ind4d = 2720 if ind4d == 3140
	replace ind4d = 2731 if ind4d == 3130
	replace ind4d = 2733 if ind4d == 3120
	replace ind4d = 2740 if ind4d == 3250
	replace ind4d = 2750 if ind4d == 2930
	replace ind4d = 2790 if ind4d == 3120 | ind4d == 3130 | ind4d == 3150 | ind4d == 3190
	replace ind4d = 2811 if ind4d == 2911
	replace ind4d = 2812 if ind4d == 2912
	replace ind4d = 2814 if ind4d == 2913
	replace ind4d = 2815 if ind4d == 2914
	replace ind4d = 2816 if ind4d == 2915
	replace ind4d = 2817 if ind4d == 3000
	replace ind4d = 2818 if ind4d == 2922
	replace ind4d = 2819 if ind4d == 2919
	replace ind4d = 2821 if ind4d == 2921
	replace ind4d = 2822 if ind4d == 2922
	replace ind4d = 2823 if ind4d == 2923
	replace ind4d = 2824 if ind4d == 2924
	replace ind4d = 2825 if ind4d == 2925
	replace ind4d = 2826 if ind4d == 2926
	replace ind4d = 2829 if ind4d == 2929
	replace ind4d = 2910 if ind4d == 3410
	replace ind4d = 2920 if ind4d == 3420
	replace ind4d = 2930 if ind4d == 3430
	replace ind4d = 3011 if ind4d == 3511
	replace ind4d = 3012 if ind4d == 3512
	replace ind4d = 3020 if ind4d == 3520
	replace ind4d = 3030 if ind4d == 3530
	replace ind4d = 3040 if ind4d == 2927
	replace ind4d = 3091 if ind4d == 3591
	replace ind4d = 3092 if ind4d == 3592
	replace ind4d = 3099 if ind4d == 3599
	replace ind4d = 3100 if ind4d == 3610
	replace ind4d = 3211 if ind4d == 3691
	replace ind4d = 3212 if ind4d == 3699
	replace ind4d = 3220 if ind4d == 3692
	replace ind4d = 3230 if ind4d == 3693
	replace ind4d = 3240 if ind4d == 3694
	replace ind4d = 3250 if ind4d == 3311 | ind4d == 3320
	replace ind4d = 3290 if ind4d == 3699
	replace ind4d = 3311 if (ind4d == 2811) | (ind4d == 2812) | (ind4d == 2813) | (ind4d == 2892)| (ind4d == 2893) | (ind4d == 2899) | (ind4d == 2927)| (ind4d == 2929) | (ind4d == 3420)
	replace ind4d = 3312 if (ind4d == 2911) | (ind4d == 2912) | (ind4d == 2913) | (ind4d == 2914)| (ind4d == 2915) | (ind4d == 2919) | (ind4d == 2921)| (ind4d == 2922) | (ind4d == 2923) ///
		| (ind4d == 2924) | (ind4d == 2925) | (ind4d == 2926) | (ind4d == 2929) | (ind4d == 3110) | (ind4d == 3699) | (ind4d == 7250)
	replace ind4d = 3313 if (ind4d == 3220) | (ind4d == 3311) | (ind4d == 3312) | (ind4d == 3313)| (ind4d == 3320)
	replace ind4d = 3314 if (ind4d == 2520) | (ind4d == 3110) | (ind4d == 3120) | (ind4d == 3130)| (ind4d == 3140) | (ind4d == 3150) | (ind4d == 3190)| (ind4d == 3210)
	replace ind4d = 3315 if (ind4d == 3511) | (ind4d == 3512) | (ind4d == 3520) | (ind4d == 3530)| (ind4d == 3599) | (ind4d == 6303)
	replace ind4d = 3319 if (ind4d == 1721) | (ind4d == 1723) | (ind4d == 2023) | (ind4d == 2029)| (ind4d == 2519) | (ind4d == 2520) | (ind4d == 2610)| (ind4d == 2699) | (ind4d == 3311) ///
		| (ind4d == 3312) | (ind4d == 3330) | (ind4d == 3692) | (ind4d == 3694)
	replace ind4d = 3320 if (ind4d == 2813) | (ind4d == 2911) | (ind4d == 2912) | (ind4d == 2914)| (ind4d == 2915) | (ind4d == 2919) | (ind4d == 2921)| (ind4d == 2922) | (ind4d == 2923) ///
		| (ind4d == 2924) | (ind4d == 2925) | (ind4d == 2926) | (ind4d == 2929) | (ind4d == 3000) | (ind4d == 3110) | (ind4d == 3220) | (ind4d == 3311) | (ind4d == 3313)
	replace ind4d = 3510 if ind4d == 4010
	replace ind4d = 3520 if ind4d == 4020
	replace ind4d = 3530 if ind4d == 4030
	replace ind4d = 3600 if ind4d == 4100
	replace ind4d = 3700 if ind4d == 9000
	replace ind4d = 3830 if ind4d == 3710 | ind4d == 3720
	replace ind4d = 3520 if ind4d == 4020
	replace ind4d = 3900 if ind4d == 9000
	replace ind4d = 4100 if ind4d == 4520
	replace ind4d = 4311 if ind4d == 4510
	replace ind4d = 4321 if ind4d == 4530
	replace ind4d = 4322 if ind4d == 4530
	replace ind4d = 4330 if ind4d == 4540
	replace ind4d = 4390 if ind4d == 4520
	replace ind4d = 4510 if ind4d == 5010
	replace ind4d = 4520 if ind4d == 5020
	replace ind4d = 4530 if ind4d == 5030
	replace ind4d = 4540 if ind4d == 5040
	replace ind4d = 4610 if ind4d == 5110
	replace ind4d = 4620 if ind4d == 5121
	replace ind4d = 4630 if ind4d == 5122
	replace ind4d = 4641 if ind4d == 5131
	replace ind4d = 4649 if ind4d == 5139
	replace ind4d = 4651 if ind4d == 5151
	replace ind4d = 4652 if ind4d == 5152 | ind4d == 5139
	replace ind4d = 4653 if ind4d == 5159
	replace ind4d = 4661 if ind4d == 5141
	replace ind4d = 4662 if ind4d == 5142
	replace ind4d = 4663 if ind4d == 5143
	replace ind4d = 4669 if ind4d == 5149 | ind4d == 5139
	replace ind4d = 4690 if ind4d == 5190
	replace ind4d = 4711 if ind4d == 5211
	replace ind4d = 4719 if ind4d == 5219
	replace ind4d = 4721 if ind4d == 5220
	replace ind4d = 4730 if ind4d == 5050
	replace ind4d = 4741 if ind4d == 5239
	replace ind4d = 4742 if ind4d == 5233
	replace ind4d = 4751 if ind4d == 5232
	replace ind4d = 4752 if ind4d == 5234
	replace ind4d = 4753 if ind4d == 5233 | ind4d == 5239
	replace ind4d = 4772 if ind4d == 5231
	replace ind4d = 4774 if ind4d == 5240
	replace ind4d = 4781 if ind4d == 5252
	replace ind4d = 4791 if ind4d == 5251
	replace ind4d = 4799 if ind4d == 5259
	replace ind4d = 4911 if ind4d == 6010
	replace ind4d = 4921 if ind4d == 6021
	replace ind4d = 4922 if ind4d == 6022
	replace ind4d = 4923 if ind4d == 6023
	replace ind4d = 4930 if ind4d == 6030
	replace ind4d = 5011 if ind4d == 6110
	replace ind4d = 5021 if ind4d == 6120
	replace ind4d = 5110 if ind4d == 6210
	replace ind4d = 5120 if ind4d == 6210
	replace ind4d = 5210 if ind4d == 6302
	replace ind4d = 5221 if ind4d == 6303
	replace ind4d = 5224 if ind4d == 6301
	replace ind4d = 5229 if ind4d == 6309
	replace ind4d = 5310 if ind4d == 6411
	replace ind4d = 5320 if ind4d == 6412
	replace ind4d = 5610 if ind4d == 5520
	replace ind4d = 5811 if ind4d == 2211 | ind4d == 7240
	replace ind4d = 5813 if ind4d == 2212
	replace ind4d = 5819 if ind4d == 2219
	replace ind4d = 5820 if ind4d == 7221
	replace ind4d = 5911 if ind4d == 9211 | ind4d == 9213
	replace ind4d = 5914 if ind4d == 9212
	replace ind4d = 5920 if ind4d == 2213
	replace ind4d = 6010 if ind4d == 9213
	replace ind4d = 6110 if ind4d == 6420
	replace ind4d = 6201 if ind4d == 7229
	replace ind4d = 6202 if ind4d == 7210
	replace ind4d = 6209 if ind4d == 7290
	replace ind4d = 6311 if ind4d == 7230
	replace ind4d = 6312 if ind4d == 7240
	replace ind4d = 6391 if ind4d == 9220
	replace ind4d = 6399 if ind4d == 7499
	replace ind4d = 6411 if ind4d == 6511
	replace ind4d = 6419 if ind4d == 6519
	replace ind4d = 6491 if ind4d == 6591
	replace ind4d = 6492 if ind4d == 6592
	replace ind4d = 6499 if ind4d == 6599
	replace ind4d = 6511 if ind4d == 6601
	replace ind4d = 6512 if ind4d == 6603
	replace ind4d = 6530 if ind4d == 6602
	replace ind4d = 6611 if ind4d == 6711
	replace ind4d = 6612 if ind4d == 6712
	replace ind4d = 6619 if ind4d == 6719
	replace ind4d = 6621 if ind4d == 6720
	replace ind4d = 6630 if ind4d == 6712
	replace ind4d = 6810 if ind4d == 7010
	replace ind4d = 6820 if ind4d == 7020
	replace ind4d = 6910 if ind4d == 7411
	replace ind4d = 6920 if ind4d == 7412
	replace ind4d = 7010 if ind4d == 7414
	replace ind4d = 7110 if ind4d == 7421
	replace ind4d = 7120 if ind4d == 7422
	replace ind4d = 7210 if ind4d == 7310
	replace ind4d = 7220 if ind4d == 7320
	replace ind4d = 7310 if ind4d == 7430
	replace ind4d = 7320 if ind4d == 7413
	replace ind4d = 7410 if ind4d == 7499
	replace ind4d = 7420 if ind4d == 7494
	replace ind4d = 7490 if ind4d == 7414 | ind4d == 7421 | ind4d == 7492
	replace ind4d = 7500 if ind4d == 8520
	replace ind4d = 7710 if ind4d == 7111
	replace ind4d = 7721 if ind4d == 7130
	replace ind4d = 7730 if ind4d == 7111 | ind4d == 7112 | ind4d == 7113 | ind4d == 7121 | ind4d == 7122 | ind4d == 7123 | ind4d == 7129
	replace ind4d = 7740 if ind4d == 6599
	replace ind4d = 7810 if ind4d == 7491
	replace ind4d = 7911 if ind4d == 6304
	replace ind4d = 8010 if ind4d == 7492
	replace ind4d = 8110 if ind4d == 7493
	replace ind4d = 8130 if ind4d == 9000
	replace ind4d = 8211 if ind4d == 7499
	replace ind4d = 8292 if ind4d == 7495
	replace ind4d = 8411 if ind4d == 7511
	replace ind4d = 8412 if ind4d == 7512
	replace ind4d = 8413 if ind4d == 7513
	replace ind4d = 8421 if ind4d == 7521
	replace ind4d = 8422 if ind4d == 7522
	replace ind4d = 8423 if ind4d == 7523
	replace ind4d = 8430 if ind4d == 7530
	replace ind4d = 8510 if ind4d == 8010
	replace ind4d = 8521 if ind4d == 8021
	replace ind4d = 8522 if ind4d == 8022
	replace ind4d = 8530 if ind4d == 8030
	replace ind4d = 8541 if ind4d == 9241
	replace ind4d = 8542 if ind4d == 8090
	replace ind4d = 8610 if ind4d == 8511
	replace ind4d = 8620 if ind4d == 8512
	replace ind4d = 8690 if ind4d == 8519
	replace ind4d = 8720 if ind4d == 8531
	replace ind4d = 8810 if ind4d == 8532
	replace ind4d = 9000 if ind4d == 9214
	replace ind4d = 9101 if ind4d == 9231
	replace ind4d = 9102 if ind4d == 9232
	replace ind4d = 9103 if ind4d == 9233
	replace ind4d = 9200 if ind4d == 5190 | ind4d == 5259 | ind4d == 9249
	replace ind4d = 9311 if ind4d == 9241
	replace ind4d = 9312 if ind4d == 9241
	replace ind4d = 9321 if ind4d == 9249
	replace ind4d = 9329 if ind4d == 9219
	replace ind4d = 9411 if ind4d == 9111
	replace ind4d = 9412 if ind4d == 9112
	replace ind4d = 9420 if ind4d == 9120
	replace ind4d = 9491 if ind4d == 9191
	replace ind4d = 9492 if ind4d == 9192
	replace ind4d = 9499 if ind4d == 9199
	replace ind4d = 9511 if ind4d == 7250
	replace ind4d = 9512 if ind4d == 3220
	replace ind4d = 9521 if ind4d == 3230 | ind4d == 5260
	replace ind4d = 9522 if ind4d == 5260
	replace ind4d = 9523 if ind4d == 5260
	replace ind4d = 9524 if ind4d == 3610
	replace ind4d = 9529 if ind4d == 5260
	replace ind4d = 9601 if ind4d == 9301
	replace ind4d = 9602 if ind4d == 9302
	replace ind4d = 9603 if ind4d == 9303
	replace ind4d = 9609 if ind4d == 9309
	replace ind4d = 9700 if ind4d == 9500
	replace ind4d = 9810 if ind4d == 9600
	replace ind4d = 9820 if ind4d == 9700
	replace ind4d = 9900 if ind4d == 9900
}
rename ind4d nic08_4d
gen nic08_3d = int(nic08_4d/10) if nic08_4d >= 1000
replace nic08_3d = nic08_4d if nic08_4d < 1000
gen nic08_2d = int(nic08_3d/10)


***********************************************************
* 2. CLEANING COMMON TO ALL YEARS
***********************************************************

order ID year

* Rounds multipliers
destring Multiple, replace
replace Multiple=round(Multiple)

* Census
gen census = 0
replace census = 1 if Multiple==1

* Open/Closed
gen Closed=1
replace Closed=0 if Open==1
			
* Year Initial Production
recode YearInitialProduction (0=.)
replace YearInitialProduction=. if YearInitialProduction<1500
replace YearInitialProduction=. if YearInitialProduction>=year

* States
* Note: panel has time-consistent identifiers whereas cross-section did not
label define statevals 35 "Andaman & N. Island"
label define statevals 28 "Andhra Pradesh", add
label define statevals 12 "Arunachal Pradesh", add
label define statevals 18 "Assam", add
label define statevals 10 "Bihar", add
label define statevals 04 "Chandigarh(U.T.)", add
label define statevals 22 "Chattisgarh", add
label define statevals 26 "Dadra & Nagar Haveli", add
label define statevals 25 "Daman & Diu", add
label define statevals 07 "Delhi", add
label define statevals 30 "Goa", add
label define statevals 24 "Gujarat", add
label define statevals 06 "Haryana", add
label define statevals 02 "Himachal Pradesh", add
label define statevals 01 "Jammu & Kashmir", add
label define statevals 20 "Jharkhand", add
label define statevals 29 "Karnataka", add
label define statevals 32 "Kerala", add
label define statevals 31 "Lakshadweep", add
label define statevals 23 "Madhya Pradesh", add
label define statevals 27 "Maharashtra", add
label define statevals 14 "Manipur", add
label define statevals 17 "Meghalaya", add
label define statevals 15 "Mizoram", add
label define statevals 13 "Nagaland", add
label define statevals 21 "Orissa", add
label define statevals 34 "Pondicherry", add
label define statevals 03 "Punjab", add
label define statevals 08 "Rajasthan", add
label define statevals 11 "Sikkim", add
label define statevals 33 "Tamil Nadu", add
label define statevals 16 "Tripura", add
label define statevals 09 "Uttar Pradesh", add
label define statevals 05 "Uttaranchal", add
label define statevals 19 "West Bengal", add
label define statevals 36 "Telegana", add
label values StateCode statevals

* Output (rupees)
egen Output = rsum(ExFactoryValueTotal IncreaseStockSemiFinished OwnConstruction)
egen TotalOutput = rsum(ExFactoryValueTotal IncreaseStockSemiFinished IncomeServices IncomeServicesManuf IncomeServicesNonManuf ElectricitySold OwnConstruction NetSaleValueGoodsResold)

* Other Receipts (Rent and Interest are not counted)
drop SaleValueGoodsResold PurchaseValueGoodsResold
gen RentReceived = RentAssetsReceived + RentBuildingsReceived + RentLandReceived
gen RentPaid = RentAssetsPaid + RentBuildingsPaid + RentLandPaid
foreach var in InterestReceived InterestPaid RentReceived RentPaid {
	replace `var'=0 if `var'==.
}
gen NetInterest = InterestPaid - InterestReceived
label var NetInterest "Net interest paid"
gen NetRent = RentPaid - RentReceived
label var NetRent "Net rent paid for fixed assets, buildings and land"
drop InterestReceived InterestPaid RentReceived RentPaid RentAssetsReceived RentBuildingsReceived RentLandReceived ///
	RentAssetsPaid RentBuildingsPaid RentLandPaid 

* Expenditures
replace RepairMaintPollution=0 if RepairMaintPollution==.
egen RepairMaint= rsum(RepairMaintBuilding RepairMaintPlant RepairMaintPollution RepairMaintOther)
drop RepairMaintBuilding RepairMaintPlant RepairMaintOther
egen OtherExpenditures = rsum(WorkDoneByOthers RepairMaint OperatingExpenses NonOperatingExpenses Insurance)

* Inputs (rupees)
egen Materials_Domestic = rsum(PurchValTotalBasicItem PurchValChemical PurchValPacking PurchValConsumable)
egen Fuels_Domestic = rsum(PurchValOil PurchValCoal PurchValGas PurchValOtherFuel)
egen Elec_Domestic = rsum(PurchValElecOwn PurchValElecBought)
egen Inputs_Domestic = rsum(Materials_Domestic Fuels_Domestic Elec_Domestic)
gen Imports = PurchValImportTotal
egen Inputs = rsum(Inputs_Domestic Imports)
egen TotalInputs = rsum(Inputs WorkDoneByOthers RepairMaint OperatingExpenses NonOperatingExpenses Insurance)

* Capital (rupees)
foreach var in FixCapClose FixCapOpen FixCapRevaluation FixCapDepreciation StockClose StockOpen ///
	SemiFinishedGoodsClose SemiFinishedGoodsOpen FinishedGoodsClose FinishedGoodsOpen {
		replace `var' = 0 if `var'==.
	}
gen Capital = FixCapClose	
gen GrossCapFormation = (FixCapClose - FixCapOpen - FixCapRevaluation + FixCapDepreciation) + (StockClose - StockOpen) + ///
	(SemiFinishedGoodsClose- SemiFinishedGoodsOpen) + (FinishedGoodsClose - FinishedGoodsOpen)
gen GrossFixCapFormation = (FixCapClose - FixCapOpen - FixCapRevaluation + FixCapDepreciation)

* Labor (rupees)
gen Labor=PersonsTotal
egen TotalEmoluments=rsum(WagesTotal BonusTotal PFTotal WelfareTotal)

* Area, Organization, and Owner Codes
label define AreaCodevals 1 "rural" 2 "urban"
label values AreaCode AreaCodevals
replace AreaCode=. if AreaCode==0 | AreaCode==3 | AreaCode==7 		// all but 11 are 0s

label define OrgCodevals 1 "Indiv proprietorship" 2 "Joint family" 3 "Partnership" 4 "Public Ltd" 5 "Private Ltd" ///
	6 "Gov Dept Enterprise" 7 "Public Corp Special Act" 8 "Khadi" 9 "Handlooms" 10 "Co-op" 19 "Other"
label values OrgCode OrgCodevals
replace OrgCode=. if OrgCode==0 | OrgCode==11						// all but 1 are 0s

label define OwnerCodevals 1 "Central Gov" 2 "State or Local Gov" 3 "Central and Local Gov" 4 "Joint Public" 5 "Joint Private" 6 "Private"
label values OwnerCode OwnerCodevals
replace OwnerCode=. if OwnerCode==0 | OwnerCode==7 | OwnerCode==8 	// all but 7 are 0s

save temp/ASI_PanelClean_InProgress, replace

***********************************************************
* 3. DISTRICT CONCORDANCE
***********************************************************

* Prepare district concordance
insheet using "OrigData/districtconcordance.csv", clear
drop v19-v237 notes
rename v# DistrictCode#, renumber(1998)
forvalues year = 2009/2016 {
	gen DistrictCode`year' = DistrictCode2008
}
reshape long DistrictCode, i(statecode statename districtname districtnamealt rev) j(year)
rename rev DistrictCode1998rev
rename statecode StateCode
collapse (first) DistrictCode1998rev districtname districtnamealt, by(StateCode DistrictCode year)
sort StateCode DistrictCode year
rename year SourceYear
save temp/districtconcordance, replace

* Match district data with district concordance file
use ../Clean_data/ASI_District.dta, clear
sort ID
merge m:1 StateCode DistrictCode SourceYear using temp/districtconcordance, keepusing(DistrictCode1998rev districtname)
drop if _merge==2
drop _merge
save temp/ASI_District, replace
* Import districts to main dataset
use temp/ASI_PanelClean_InProgress, clear
sort ID year
merge m:1 ID year using temp/ASI_District
replace DistrictCode = 99 if DistrictCode == .
rename DistrictCode DistrictCodeOriginal
rename DistrictCode1998rev DistrictCode
replace DistrictCode = 99 if DistrictCode == .
drop if _merge == 2
drop _merge

erase temp/ASI_District.dta
erase temp/ASI_PanelClean_InProgress.dta
erase temp/districtconcordance.dta

***********************************************************
* 4. ADD EXTERNAL DATA
***********************************************************

cd ../..
**** Price of Oil *****
* Import crude oil prices - Indian basket (USD)
preserve
	clear
	import excel "External\Price_IndianOil_USDperBarrel.xlsx", sheet("Sheet1") firstrow
	rename yr year
	save "External\Price_IndianOil_USDperBarrel.dta", replace
restore
merge m:1 year using "External/Price_IndianOil_USDperBarrel.dta"
drop if _merge == 2
drop _merge
* Exchange rate to get price in rupees
preserve
	use "External/Exchange_rate.dta", clear
	capture rename yr year
	save "External/Exchange_rate.dta", replace
restore
merge m:1 year using "External/Exchange_rate.dta"
drop if _merge == 2
drop _merge
gen p_oil_bar = p_oil*exinus
* Convert price of oil from barell to gallon (rupees)
gen p_oil_gal = p_oil_bar/42
* Convert price of oil from gallon to mmbtu (rupees, source: EPA)
gen p_oil_mmbtu = p_oil/0.138

* Value of collateral needed for a loan (1/theta in model*100) - Higher value means larger collateral constraint 
gen colreq=.
lab var colreq "Collateral requirement for a loan (percentage of loan amount)"
replace colreq = 244.9 if StateCode == 27
replace colreq = 254.2 if StateCode == 29
replace colreq = 129.7 if StateCode == 28
replace colreq = 205.4 if StateCode == 19
replace colreq = 318 if StateCode == 33
replace colreq = 253.5 if StateCode == 7
replace colreq = 188.1 if StateCode ==  9
replace colreq = 246.6 if StateCode == 23
replace colreq = 327.7 if StateCode == 8
replace colreq = 314.9 if StateCode == 32
replace colreq = 129.7 if StateCode == 3
replace colreq = 420.4 if StateCode == 6
replace colreq = 287.6 if StateCode == 18
replace colreq = 281.9 if StateCode == 10
replace colreq = 55.1 if StateCode == 22
replace colreq = 266.5 if StateCode == 20
replace colreq = 326.7 if StateCode == 2
replace colreq = 265.6 if (StateCode == 12) | (StateCode == 13) | (StateCode == 14) | (StateCode== 16) | (StateCode == 17)
replace colreq = 199.2 if StateCode == 30

* Average case pendency in high courts (Daksh database)
gen k1=.
lab var k1 "Average pendency of court cases by high court (days)"
replace k1 = 1370 if StateCode == 9 // Allahabad
replace k1 = 1300 if (StateCode ==  27) // Bombay
replace k1 = 1207 if StateCode == 24 // Gujarat
replace k1 = 1102 if StateCode == 10 // Patna
replace k1 = 1025 if StateCode == 23 // Madhya Pradesh
replace k1 = 1015 if StateCode == 29 // Karnataka
replace k1 = 992 if StateCode == 7 // Delhi
replace k1 = 922 if StateCode == 8 // Rajasthan
replace k1 = 891 if StateCode == 33 // Madras
replace k1 = 866 if (StateCode == 35)  // Kolkata
replace k1 = 822 if (StateCode == 28)  // Hyderabad
replace k1 = 750 if (StateCode == 6)  // Punjab and Haryana
replace k1 = 723 if StateCode == 16 // Tripura
replace k1 = 711 if (StateCode == 32) // Kerala
replace k1 = 703 if StateCode == 20 // Jharkand
replace k1 = 679 if StateCode == 2 // Himachal Pradesh
replace k1 = 610 if StateCode == 21 // Odissa
replace k1 = 435 if (StateCode == 30) | (StateCode == 25) | (StateCode == 26) // Goa
replace k1 = 390 if StateCode == 5 // Uttarakhand
replace k1 = 314 if StateCode == 11 // Sikkim
rename k1 HighCourtPendency

* Year of High court creation (Boehm and Oberfield 2020)
gen k2=.
lab var k2 "Year of high court creation"
replace k2 = 1866 if StateCode == 9 // Allahabad
replace k2 = 1956 if (StateCode == 28) | (StateCode == 36) // Hyderabad
replace k2 = 1862 if (StateCode ==  27) // Bombay
replace k2 = 1862 if (StateCode == 35) | (StateCode == 19) // Kolkata
replace k2 = 1966 if StateCode == 7 // Delhi
replace k2 = 1948 if (StateCode == 12) | (StateCode == 18) | (StateCode == 15) | (StateCode == 13) // Gauhati
replace k2 = 1960 if StateCode == 24 // Gujarat
replace k2 = 1971 if StateCode == 2 // Himachal Pradesh
replace k2 = 1928 if StateCode == 1 // Jammu & Kashmir
replace k2 = 2000 if StateCode == 20 // Jharkand
replace k2 = 1884 if StateCode == 29 // Karnataka
replace k2 = 1956 if (StateCode == 32) | (StateCode == 31) // Kerala
replace k2 = 1936 if StateCode == 23 // Madhya Pradesh
replace k2 = 1862 if StateCode == 33 // Madras
replace k2 = 1948 if StateCode == 21 // Odissa
replace k2 = 1916 if StateCode == 10 // Patna
replace k2 = 1947 if (StateCode == 6) | (StateCode == 4) | (StateCode == 3) // Punjab and Haryana
replace k2 = 1949 if StateCode == 8 // Rajasthan 
replace k2 = 1955 if StateCode == 11 // Sikkim
replace k2 = 2000 if StateCode == 5 // Uttarakhand
replace k2 = 1982 if (StateCode == 30) |  (StateCode == 25) | (StateCode == 26) // Goa
replace k2 = 2013 if StateCode == 14 // Manipur
replace k2 = 2013 if StateCode == 17 // Meghalaya
replace k2 = 2013 if StateCode == 16 // Tripura
rename k2 HighCourtInitDate

***********************************************************
* 5. CREATE FUEL PRICES AND QUANTITIES
***********************************************************

*Keep years where we observe natural gas
keep if year >= 2009

*** Domestic Coal, Gas and Oil ***
rename PurchValCoal Coal
rename PurchValOil Oil
rename PurchValGas Gas
rename QtyConsCoal CoalQty
rename QtyConsGas GasQty
gen OilQty = Oil/p_oil_bar

* Replace missing values for 0 when plant is consuming some amount of fuel
replace Coal = 0 if Coal == .
replace Oil = 0 if Oil == .
replace Gas = 0 if Gas == .
* Get quantities in mmBtu
rename p_oil_mmbtu poil_mmbtu
gen gas_mmbtu = GasQty*0.04739 if UnitCodeGas == 9 // Kg to mmbtu
gen coal_mmbtu = CoalQty*27.78 if UnitCodeCoal == 27 // ton to mmbtu
gen oil_mmbtu = Oil/poil_mmbtu
gen pgas_mmbtu = UnitPriceGas/0.04739 
gen pcoal_mmbtu = UnitPriceCoal/27.78
replace pgas_mmbtu = . if pgas_mmbtu == 0
replace pcoal_mmbtu = . if pcoal_mmbtu == 0
* For firms where only spending is available, use price index for that year
bysort year: egen pcoal_mmbtu_index = median(pcoal_mmbtu)
bysort year: egen pgas_mmbtu_index = median(pgas_mmbtu)
replace coal_mmbtu = Coal/pcoal_mmbtu_index if coal_mmbtu == . & Coal > 0
replace coal_mmbtu = Coal/pcoal_mmbtu_index if coal_mmbtu == 0 & Coal > 0
replace gas_mmbtu = Gas/pgas_mmbtu_index if gas_mmbtu == . & Gas > 0
replace gas_mmbtu = Gas/pgas_mmbtu_index if gas_mmbtu == 0 & Gas > 0
replace gas_mmbtu = 0 if gas_mmbtu == . & (oil_mmbtu > 0 | coal_mmbtu > 0)
replace oil_mmbtu = 0 if oil_mmbtu == . & (gas_mmbtu > 0 | coal_mmbtu > 0)
replace coal_mmbtu = 0 if coal_mmbtu == . & (oil_mmbtu > 0 | coal_mmbtu > 0)

*** Imported Coal, Gas and Oil ***
gen CoalImport = 0
gen OilImport = 0
gen GasImport = 0
gen CoalImport_mmbtu = 0
gen OilImport_mmbtu = 0
gen GasImport_mmbtu = 0
* 2009-2010 (ASICC product codes)
forvalues i = 1/5 {
	gen asiccImport`i'_3d = int(int(asiccImport`i'/10)/10)
	gen pcoal_mmbtu_import`i' = 0
}
forvalues i = 1/5 {
	replace CoalImport = CoalImport + PurchValImport`i' if asiccImport`i'_3d == 231
	replace pcoal_mmbtu_import`i' = UnitPriceImport`i'/27.78 if (asiccImport`i'_3d == 231 & UnitCodeImport`i' == 27)
	replace OilImport = OilImport + PurchValImport`i' if (asiccImport`i'_3d == 232 | asiccImport`i'_3d == 233 | asiccImport`i'_3d == 234 | asiccImport`i'_3d == 239)
	replace GasImport = GasImport + PurchValImport`i' if asiccImport`i'_3d == 241
}
* 2011-2016 (NPCMS product codes)
forvalues i = 1/5 {
	gen npcmsImport`i'_4d = int(int(int(npcmsImport`i'/10)/10)/10) if npcmsImport`i' >= 1000000
	gen npcmsImport`i'_3d = int(npcmsImport`i'_4d/10)
	replace npcmsImport`i'_4d = int(int(npcmsImport`i'/10)/10) if npcmsImport`i' < 100000
}
forvalues i = 1/5 {
	replace CoalImport = CoalImport + PurchValImport`i' if npcmsImport`i'_3d == 110 | npcmsImport`i'_4d == 1203 | npcmsImport`i'_3d == 331
	replace pcoal_mmbtu_import`i' = UnitPriceImport`i'/27.78 if (npcmsImport`i'_3d == 110 & UnitCodeImport`i' ==27) | (npcmsImport`i'_4d == 1203 & UnitCodeImport`i'==27) | (npcmsImport`i'_3d == 331 & UnitCodeImport`i' == 27)
	replace OilImport = OilImport + PurchValImport`i' if npcmsImport`i'_4d == 1201 | npcmsImport`i'_3d == 333 | npcmsImport`i'_3d == 334
	replace GasImport = GasImport + PurchValImport`i' if npcmsImport`i'_4d == 1202
}
forvalues i = 1/5 {
	replace pcoal_mmbtu_import`i' = . if pcoal_mmbtu_import`i' == 0
}
egen pcoal_mmbtu_import = rmean(pcoal_mmbtu_import1 pcoal_mmbtu_import2 pcoal_mmbtu_import3 pcoal_mmbtu_import4 pcoal_mmbtu_import5)
replace CoalImport_mmbtu = CoalImport/pcoal_mmbtu_import
replace OilImport_mmbtu = OilImport/poil_mmbtu
replace GasImport_mmbtu = GasImport/pgas_mmbtu_index

*** Create prices and quantity (imported + domestic) of Coal, Gas, Oil and Electricity ***
* For firms where only spending is available, use price index for that year
bysort year: egen pcoal_mmbtu_import_index = median(pcoal_mmbtu_import)
replace CoalImport_mmbtu = CoalImport/pcoal_mmbtu_import_index if CoalImport_mmbtu == . & CoalImport > 0 
replace CoalImport_mmbtu = CoalImport/pcoal_mmbtu_import_index if CoalImport_mmbtu == 0 & CoalImport > 0 
replace CoalImport_mmbtu = 0 if CoalImport_mmbtu == . & (OilImport_mmbtu > 0 | GasImport_mmbtu > 0)
egen TotCoal = rsum(Coal CoalImport)
egen TotOil = rsum(Oil OilImport)
egen TotGas = rsum(Gas GasImport)
egen TotCoal_mmbtu = rsum(coal_mmbtu CoalImport_mmbtu)
egen TotOil_mmbtu = rsum(oil_mmbtu OilImport_mmbtu)
egen TotGas_mmbtu = rsum(gas_mmbtu GasImport_mmbtu)
* Keep plants that use some amount of fuel
replace TotCoal = 0 if TotCoal == .
replace TotOil = 0 if TotOil == .
replace TotGas = 0 if TotGas == .
replace PurchValElecBought = 0 if PurchValElecBought == .
drop if TotCoal == 0 & TotOil == 0 & TotGas == 0 & PurchValElecBought == 0
* Convert electricity units (kwh) to mBbtu and create price of electricity
gen elecb_mmbtu = QtyConsElecBought*0.003412 if UnitCodeElecBought == 28 	// Purchased electricity
gen pelecb_mmbtu = UnitPriceElecBought/0.003412 if UnitCodeElecBought == 28
gen elecommbtu = QtyConsElecOwn*0.003412 if UnitCodeElecOwn == 28			// Generated electricity

* Save dataset
cd ..
save "Data/Panel_Data/Clean_data/ASI_PanelCleanFinal", replace
erase Data/Panel_Data/Clean_data/ASI_PanelClean.dta 


