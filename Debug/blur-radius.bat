FOR /L %%i IN (2,1,10) DO (
    call "%~dp0\UnsharpMaskCUDA.exe" ghost-town-8k.ppm ghost-out-%%i.ppm %%i cpu box
)
FOR /L %%i IN (2,1,10) DO (
    call "%~dp0\UnsharpMaskCUDA.exe" ghost-town-8k.ppm ghost-out-%%i.ppm %%i gpu box
)
FOR /L %%i IN (2,1,10) DO (
    call "%~dp0\UnsharpMaskCUDA.exe" ghost-town-8k.ppm ghost-out-%%i.ppm %%i cpu gauss
)
FOR /L %%i IN (2,1,10) DO (
    call "%~dp0\UnsharpMaskCUDA.exe" ghost-town-8k.ppm ghost-out-%%i.ppm %%i gpu gauss
)
