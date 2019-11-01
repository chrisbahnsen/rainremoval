## Reimplementation of the Garg/Nayar rain removal algorithm
---

This is the source code of the reimplementation of the rain removal algorithm from Garg and Nayar as described in their paper ["Detection and Removal of Rain from Videos"](https://ieeexplore.ieee.org/abstract/document/1315077/). 
Please refer to the source files located at GargNayarRainRemoval/gargnayar.cpp for a full documentation of the approach. The file is commented, no worries.

### The method at a glance
![](https://bitbucket.org/aauvap/rainremoval/raw/76c840def005e450ca1c6ca6eff84ecbbf7bea42/GargNayarOverview.png)

### Download
Windows binaries are available from the [downloads section](https://bitbucket.org/aauvap/rainremoval/downloads/).

Open the file GargNayarRainRemoval.exe from the command line and see the available options.

### Other open-source rain removal algorithms

* **Kang et al, 2012**: Automatic Single-Image-Based Rain Streaks Removal via Image Decomposition [[code]](http://www.ee.nthu.edu.tw/cwlin/Rain_Removal/Rain_Removal.htm)
* **Huang et al, 2012**: Context-Aware Single Image Rain Removal [[code]](http://mml.citi.sinica.edu.tw/papers/RainRemovalSoucre_ICME.zip)
* **Huang et al, 2014**: Self-Learning Based Image Decomposition with Applications to Single Image Denoising [[code]](http://mml.citi.sinica.edu.tw/papers/TMM.zip)
* **Kim et al, 2015**: Video Deraining and Desnowing Using Temporal Correlation and Low-Rank Matrix Completion [[code, currently unavailable]](http://mcl.korea.ac.kr/deraining/)
* **Luo et al, 2015**: Removing rain from a single image via discriminative sparse coding [[code]](http://www.google.com/url?q=http%3A%2F%2Fwww.math.nus.edu.sg%2F~matjh%2Fdownload%2Fimage_deraining%2Frain_removal_v.1.1.zip&sa=D&sntz=1&usg=AFQjCNHtuwP1pMD67WSZDUDe7v97nk3gyA)
* **Zhang et al, 2017**: Image De-Raining Using a Conditional Generative Adversarial Network [[code]](https://github.com/hezhangsprinter/ID-CGAN/)
* **Fu et al, 2017**: Clearing the Skies: A deep network architecture for single-image rain removal [[code]](https://xueyangfu.github.io/projects/tip2017.html) [[Tensorflow implementation]](https://github.com/jinnovation/derain-net)
* **Jiang et al, 2017**: A Novel Tensor-Based Video Rain Streaks Removal Approach via Utilizing Discriminatively Intrinsic Priors [[code]](https://github.com/uestctensorgroup/FastDeRain)
* **Wei et al, 2017**: Should We Encode Rain Streaks in Video as Deterministic or Stochastic? [[code]](https://github.com/wwxjtu/RainRemoval_ICCV2017)
* **Qian et al, 2018**: Attentive Generative Adversarial Network for Raindrop Removal from A Single Image [[code]](https://github.com/rui1996/DeRaindrop)
* **Zhang et al, 2018**: Density-aware Single Image De-raining using a Multi-stream Dense Network [[code]](https://github.com/hezhangsprinter/DID-MDN)

### Compilation
OpenCV 3.x is needed in order to compile the program.

On Windows and Visual Studio, open RainRemoval.sln, configure your solution, and build from there. 
On Unix, you should compile the file gargnayar.cpp.


### Who do I talk to?
Chris H. Bahnsen at cb@create.aau.dk

### Acknowledgements
Please cite the following paper if you use our reimplementation:

```TeX
@article{bahnsen2018rain,
  title={Rain Removal in Traffic Surveillance: Does it Matter?},
  author={Bahnsen, Chris H. and Moeslund, Thomas B.},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2018},
  publisher={IEEE},
  doi={10.1109/TITS.2018.2872502},
  pages={1--18}
}
```