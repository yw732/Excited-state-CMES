# Excited-state-CMES
## 1D excited state CMES:  
  Equations are solved with dense grids  
###  Double well:  
  * Potential has a general form of $ax^{2}+bx^{4}$  
  * While negative $a$ corresponds to a double-well potential, the single well potential with a positive $a$ can also be solved
###  Morse: 
  * Potential has a general form of $V(x)=D_{e}(e^{-2\alpha(x-x_{e})}-2e^{-\alpha(x-x_{e})})$
  * This is not a symmetric potential. The local minima of excited state CMES need to be searched before calculating frequencies.
## 2D excited state CMES:  
  Instead of dense grids with finite difference method, now the Fourier grid method is used  
### Harmonic-like potential:
  * The 2D potential has a general form of: $V(x,y)=a_{x}x^{2}+a_{y}y^{2}+b_{x}x^{4}+b_{y}y^{4}+c_{xy}xy+c_{x^{2}y}x^{2}y$
  * Unless $c_{x^{2}y}= 0$, this is also not a symmetric potential. The local minima of excited state CMES need to be searched before calculating frequencies
  * "surface_generator" file will provide 9 points for local hessian matrix calculation
