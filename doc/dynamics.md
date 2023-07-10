# 力学
通过主动应力方法分解心肌主动张力和被动应力：
$\mathbf{P}=\mathbf{P}_p+\mathbf{P}_a=\frac{\partial \Psi_p}{\partial \mathbf{F}} + \frac{\partial \Psi_a}{\partial \mathbf{F}}$
$\Psi$是能量密度，$\mathbf{F}$是形变梯度

- 被动应力：
$$\Psi_p = \Psi _{\mathrm{SNH}}=\frac{\lambda}{2}(\det\mathrm{(}\mathbf{F})-1-\frac{\mu}{\lambda})^2+\frac{\mu}{2}(\mathrm{tr(}\mathbf{F}^T\mathbf{F})-3)$$
$$\mathbf{P}_p=\lambda(\det(\mathbf{F}) - 1-\frac{\mu}{\lambda})\frac{\partial I_3}{\partial \mathbf{F}}+\mu \mathbf{F}$$
- 主动张力：
$$\Psi _a=\frac{T_a}{2}(I_{ff}-1)$$
$$\mathbf{P}_a = T_a\mathbf{F}\mathbf{f_0}\mathbf{f_0}^T$$

## 细节
$\mathbf{D_s} = \begin{bmatrix}
x_1 - x_0 & x_2 - x_0 & x_3 - x_0\\
y_1 - y_0 & y_2 - y_0 & y_3 - y_0\\
z_1 - z_0 & z_2 - z_0 & z_3 - z_0\\
\end{bmatrix}$
$设\mathbf{D_m^{-1}} = \begin{bmatrix}
d_{00} & d_{01} & d_{02}\\
d_{10} & d_{11} & d_{12}\\
d_{20} & d_{21} & d_{22}\\
\end{bmatrix}$
则，$\mathbf{F=D_sD_m^{-1}} = 
\begin{bmatrix}
d_{00}(x_1 - x_0) + d_{10}(x_2 - x_0) + d_{20}(x_3 - x_0) & 
d_{01}(x_1 - x_0) + d_{11}(x_2 - x_0) + d_{21}(x_3 - x_0) & 
d_{02}(x_1 - x_0) + d_{12}(x_2 - x_0) + d_{22}(x_3 - x_0)\\
d_{00}(y_1 - y_0) + d_{10}(y_2 - y_0) + d_{20}(y_3 - y_0) & 
d_{01}(y_1 - y_0) + d_{11}(y_2 - y_0) + d_{21}(y_3 - y_0) & 
d_{02}(y_1 - y_0) + d_{12}(y_2 - y_0) + d_{22}(y_3 - y_0)\\
d_{00}(z_1 - z_0) + d_{10}(z_2 - z_0) + d_{20}(z_3 - z_0) & 
d_{01}(z_1 - z_0) + d_{11}(z_2 - z_0) + d_{21}(z_3 - z_0) & 
d_{02}(z_1 - z_0) + d_{12}(z_2 - z_0) + d_{22}(z_3 - z_0)\\
\end{bmatrix}$

