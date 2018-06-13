## Reimplementation of the Garg/Nayar rain removal algorithm
---

This is the source code of the reimplementation of the rain removal algorithm from Garg and Nayar as described in their paper ["Detection and Removal of Rain from Videos"](ieeexplore.ieee.org/abstract/document/1315077/). 
Please refer to the source files located at GargNayarRainRemoval/gargnayar.cpp for a full documentation of the approach. The file is commented, no worries.

### The method at a glance
![](https://bitbucket.org/aauvap/rainremoval/raw/76c840def005e450ca1c6ca6eff84ecbbf7bea42/GargNayarOverview.png)

### Download
Windows binaries are available from the [downloads section](https://bitbucket.org/aauvap/rainremoval/downloads/).

Open the file GargNayarRainRemoval.exe from the command line and see the available options.


### Compilation
OpenCV 3.x is needed in order to compile the program.

On Windows and Visual Studio, open RainRemoval.sln, configure your solution, and build from there. 
On Unix, you should compile the file gargnayar.cpp.


### Who do I talk to?
Chris H. Bahnsen at cb@create.aau.dk