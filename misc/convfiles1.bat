@echo off
@echo off

Setlocal enabledelayedexpansion
pushd "D:\cmu_us_slt_arctic\cmu_us_slt_arctic\DUMMY"

REM del "*mic1.flac" /S /Q

Set "Pattern=_mic2temp"
Set "Replace="

    for %%f in (*.wav) do (
    sox "%%f" "temp.wav" silence 1 0.1 1% reverse
    sox "temp.wav" "converted/%%~nxf" silence 1 0.1 1% reverse

)

pause