import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt
import matplotlib

def glorot(dims):

    assert len(dims) == 2

    params = npr.randn(*dims) * np.sqrt(2/ (dims[0]+dims[1]))

    return params

def pad_to_2d(kernel, dims, mode="zeros"): 
                                
    mid_x = dims[-2] // 2                                                        
    mid_y = dims[-1] // 2                                                        
    mid_k_x = kernel.shape[-2] // 2                                              
    mid_k_y = kernel.shape[-1] // 2                                              
                                                                                
    start_x = mid_x - mid_k_x                                                   
    start_y = mid_y - mid_k_y                                                   
                                                                                
    padded = np.zeros(dims)                                                     
    padded[..., start_x:start_x + kernel.shape[-2],
            start_y:start_y + kernel.shape[-1]] = kernel             

    if mode == "zeros":
        pass
    elif mode == "circular":
        padded[...,0,0] = kernel[...,-1,-1]
        padded[...,-1,-1] = kernel[...,0,0]
        padded[...,-1,0] = kernel[...,0,-1]
        padded[...,0,-1] = kernel[...,-1,0]

                                                                                
    return padded                         
                                                        
def ft_convolve(grid, kernel):                                                  
                                             
    if np.shape(kernel) != np.shape(grid):                                     
        padded_kernel = pad_to_2d(kernel, grid.shape)                          
    else:                                                                       
        padded_kernel = kernel                                                  
                                                                                
    fourier_kernel = np.fft.fft2(np.fft.fftshift(padded_kernel, axes=(-2,-1)), axes=(-2,-1))
    fourier_grid = np.fft.fft2(np.fft.fftshift(grid, axes=(-2,-1)), axes=(-2,-1))
    fourier_product = fourier_grid * fourier_kernel 
    real_spatial_convolved = np.real(np.fft.ifft2(fourier_product, axes=(-2,-1)))
    convolved = np.fft.ifftshift(real_spatial_convolved, axes=(-2, -1))
                                                                                
    return convolved 
