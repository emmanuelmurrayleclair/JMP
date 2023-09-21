*----------------
* Initial Set Up
*----------------
cls
clear all
version 13
set maxvar 10000
set type double
set more off

*---------------------------------------------
* Graph details
*---------------------------------------------

*ssc instal schemepack, replace
set scheme cblind1 
grstyle init
grstyle color background white
grstyle color major_grid dimgray
grstyle linewidth major_grid thin
grstyle yesno draw_major_hgrid yes
grstyle yesno grid_draw_min yes
grstyle yesno grid_draw_max yes
grstyle anglestyle vertical_tick horizontal

* File path -- PUT YOUR FILEPATH HERE

*** Emmanuel ***
* Laptop
cd "C:\Users\Emmanuel\Dropbox\JMP"
* Desktop
// cd "D:\Users\Emmanuel\Dropbox\JMP"

*---------------------------------------------------------------------------------------------------------
* All steps in order: 
*---------------------------------------------------------------------------------------------------------

* from cleaning data to estimating the production function and exporting to julia for dynamic simulation
do "Stata Do-files/1.Program_Import-Panel.do"
do "Stata Do-files/2.Program_MatchDistricts.do"
do "Stata Do-files/3.Program_Prepare-Panel.do"
do "Stata Do-files/4a.Program_ImportPipeline.do"
do "Stata Do-files/4b.Program_MatchDist-GIS.do"
do "Stata Do-files/5.Program_ApplyDeflators.do"
do "Stata Do-files/6.Program_Clean-Panel.do"
do "Stata Do-files/7a.GraphTable_Evidence.do"
do "Stata Do-files/7b.GraphTable_SpatialEvidence.do"
do "Stata Do-files/8.Program_OuterPFE.do"
do "Stata Do-files/9.Program_EnergyPFE.do"
do "Stata Do-files/11.Appendix_GraphTables.do"

** DO ONLY AFTER YOU'VE ESTIMATED DYNAMIC MODEL IN JULIA
do "Stata Do-files/10.Graphs_PostDynamics.do"

