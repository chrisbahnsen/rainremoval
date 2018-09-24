## Reimplementation of the Garg/Nayar rain removal algorithm
---

This is the source code of the reimplementation of the rain removal algorithm from Garg and Nayar as described in their paper ["Detection and Removal of Rain from Videos"](https://ieeexplore.ieee.org/abstract/document/1315077/). 
Please refer to the source files located at GargNayarRainRemoval/gargnayar.cpp for a full documentation of the approach. The file is commented, no worries.

### The method at a glance
![](https://bitbucket.org/aauvap/rainremoval/raw/76c840def005e450ca1c6ca6eff84ecbbf7bea42/GargNayarOverview.png)

### Download
Windows binaries are available from the [downloads section](https://bitbucket.org/aauvap/rainremoval/downloads/).

Open the file GargNayarRainRemoval.exe from the command line and see the available options.

### Other rain removal algorithms

* **Kang et al, 2012**: [Automatic single-image-based rain streaks removal via image decomposition](http://www.ee.nthu.edu.tw/cwlin/Rain_Removal/Rain_Removal.htm)
* **Kim et al, 2015**: [Video Deraining and Desnowing Using Temporal Correlation and Low-Rank Matrix Completion (currently unavailable](http://mcl.korea.ac.kr/deraining/)


### Compilation
OpenCV 3.x is needed in order to compile the program.

On Windows and Visual Studio, open RainRemoval.sln, configure your solution, and build from there. 
On Unix, you should compile the file gargnayar.cpp.


### Who do I talk to?
Chris H. Bahnsen at cb@create.aau.dk