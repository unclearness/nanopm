rem rd /s /q win_build

IF NOT EXIST win_build (
    md win_build
)

cd win_build

IF EXIST CMakeCache.txt (
    del CMakeCache.txt
)
IF EXIST "C:\\Program Files\\Microsoft Visual Studio\\2022\\" (
    cmake -G "Visual Studio 17 2022"  ..
) ELSE IF EXIST "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\" (
    cmake -G "Visual Studio 16 2019" ..
) ELSE IF EXIST "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\" (
    cmake -G "Visual Studio 15 2017 Win64" ..
)

cd ..

pause
