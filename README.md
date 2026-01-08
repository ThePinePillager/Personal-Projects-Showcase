# Personal-Projects-Showcase
A showcase for personal projects I've done.
This repository uses the MIT License, following standard open-source practice.

Simplex_Noise_v2 is my custom noise function. Instead of relying on approximate differentiation, which is slower and less accurate, I implemented exact (analytical) differentiation using sympy and a bunch of jacobians. Since this program essentially does many small operations on massive matrices, I used CuPy (which utilizes the GPU) to make it faster. 

Known bugs: 
  - There is a suspected bug in the differentiation logic when noise warping is enabled, but it's quite hard to verify. I've combed over the logic several times, and nothing looks wrong on paper, but the output is more chaotic than I suspect is should be. I'll create an approximate differentiation function for comparison.
  - The lighting engine is rather hard to work with because of MatPlotLib limitations. When the light vector is vertical, then there is no difference in lighting across the whole map, so the map appears dark.

Images and Settings:

  <img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/b47a6568-6016-45b0-96ac-e6254e9cfb14" />

