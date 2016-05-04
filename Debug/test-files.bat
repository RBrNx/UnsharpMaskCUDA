FOR /L %%i IN (1,1,5) DO (
    call "%~dp0\UnsharpMaskCUDA.exe" planet.ppm planet-out-%%i.ppm 5 cpu box
)
FOR /L %%i IN (1,1,5) DO (
    call "%~dp0\UnsharpMaskCUDA.exe" planet.ppm planet-out-%%i.ppm 5 gpu box
)
FOR /L %%i IN (1,1,5) DO (
    call "%~dp0\UnsharpMaskCUDA.exe" outdoor-wall.ppm wall-out-%%i.ppm 5 cpu box
)
FOR /L %%i IN (1,1,5) DO (
    call "%~dp0\UnsharpMaskCUDA.exe" outdoor-wall.ppm wall-out-%%i.ppm 5 gpu box
)
pause
