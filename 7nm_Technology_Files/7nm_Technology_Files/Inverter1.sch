v {xschem version=3.4.5 file_version=1.2
}
G {}
K {}
V {}
S {}
E {}
N 23.75 8.75 23.75 26.25 {
lab=out}
N -6.25 -23.75 -6.25 56.25 {
lab=in}
N 23.75 88.75 23.75 102.5 {
lab=gnd!}
N 23.75 -60 23.75 -53.75 {
lab=vdd!}
N 26.25 -38.75 33.75 -38.75 {
lab=vdd!}
N 33.75 -52.5 33.75 -38.75 {
lab=vdd!}
N 23.75 -53.75 33.75 -52.5 {
lab=vdd!}
N 23.75 88.75 35 88.75 {
lab=gnd!}
N 35 72.5 35 88.75 {
lab=gnd!}
N 35 71.25 35 72.5 {
lab=gnd!}
N 26.25 71.25 35 71.25 {
lab=gnd!}
N 23.75 16.25 51.25 16.25 {
lab=out}
N -38.75 15 -6.25 15 {
lab=in}
C {devices/vdd.sym} 23.75 -60 0 0 {name=l1 lab=vdd!}
C {devices/gnd.sym} 23.75 102.5 0 0 {name=l2 lab=gnd!}
C {devices/ipin.sym} -38.75 15 0 0 {name=p2 lab=in}
C {devices/opin.sym} 51.25 16.25 0 0 {name=p3 lab=out}
C {devices/simulator_commands_shown.sym} 133.75 -81.25 0 0 {name=COMMANDS
simulator=ngspice
only_toplevel=false 
value="
* ngspice commands
.include 'asap7_TT_slvt.sp'

Vvdd vdd! 0 dc=0.7
Vgnd gnd! 0 dc=0

Vin in 0 dc=0
.dc Vin 0 0.7 0.001

.control
    run
    set xbrushwidth=3
    plot out vs in title 'Voltage Transfer Characteristic' xlabel 'Vin (V)' ylabel 'Vout (V)'
.endc
"}
C {FinFet Technology/PFFet7.sym} -80 -30 0 0 {name=Npmos1 model=BSIMCMG_osdi_P nfin=8}
C {FinFet Technology/NFFet7.sym} -80 50 0 0 {name=Nnmos model=BSIMCMG_osdi_N nfin=5}
