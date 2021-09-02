@echo off
@echo off

Setlocal enabledelayedexpansion
pushd "D:\cmu_us_slt_arctic\cmu_us_slt_arctic\wavs\DUMMY"

Set "Pattern=_mic2temp"
Set "Replace="

    for %%f in (*.wav) do (
    sox.exe "%%f" -r 22050 "converted/%%~nxf"

)

pause