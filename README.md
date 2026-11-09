# Personal-Projects-Showcase
A showcase for personal projects I've done.
This repository uses the MIT License, following standard open-source practice.

Simplex_Noise_v2 is my custom noise function. Instead of relying on approximate differentiation, which is slower and less accurate, I implemented exact (analytical) differentiation using sympy and a bunch of jacobians. Since this program essentially does many small operations on massive matrices, I used CuPy (which utilizes the GPU) to make it faster. 

Images:

  <img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/b47a6568-6016-45b0-96ac-e6254e9cfb14" /> <img width="430" height="430" alt="image" src="https://github.com/user-attachments/assets/8599ffb6-a327-4927-9c32-41f3ac35918c" />


Settings: 

    octaves = 12, 
    bias_function = sp.sqrt(x**2 + 0.000001) * -1, 
    fBm_bias_func = sp.sqrt(x**2 + 0.00001), 
    lacunarity = 1.5, 
    persistence = 0.75, 
    warp_func_x = (sp.ln(y ** 2 + 0.000001)), 
    warp_func_y = (sp.ln(x ** 2 + 0.000001)), 
    noise_function_range = 1000, 
    light_vector = [1, 1, 1]

