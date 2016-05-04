FOR /L %%i IN (1,1,5) DO (
    call "%~dp0\UnsharpMaskCUDA.exe" lena.ppm lena-out.ppm 5 cpu box
)
FOR /L %%i IN (1,1,5) DO (
    call "%~dp0\UnsharpMaskCUDA.exe" lena.ppm lena-out.ppm 5 cpu gauss
)
FOR /L %%i IN (1,1,5) DO (
    call "%~dp0\UnsharpMaskCUDA.exe" lena.ppm lena-out.ppm 5 gpu box
)
FOR /L %%i IN (1,1,5) DO (
    call "%~dp0\UnsharpMaskCUDA.exe" lena.ppm lena-out.ppm 5 gpu gauss
)

FOR /L %%i IN (1,1,5) DO (
    call "%~dp0\UnsharpMaskCUDA.exe" ghost-town-8k.ppm ghost-out.ppm 5 cpu box
)
FOR /L %%i IN (1,1,5) DO (
    call "%~dp0\UnsharpMaskCUDA.exe" ghost-town-8k.ppm ghost-out.ppm 5 cpu gauss
)
FOR /L %%i IN (1,1,5) DO (
    call "%~dp0\UnsharpMaskCUDA.exe" ghost-town-8k.ppm ghost-out.ppm 5 gpu box
)
FOR /L %%i IN (1,1,5) DO (
    call "%~dp0\UnsharpMaskCUDA.exe" ghost-town-8k.ppm ghost-out.ppm 5 gpu gauss
)