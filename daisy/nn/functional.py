import numpy as np
import numpy.random as npr

def glorot(dims):

    assert len(dims) == 2

    params = npr.randn(*dims) * np.sqrt(2/ (dims[0]+dims[1]))

    return params

def ft_convolve(grid, kernel):                                                  
                                             
    if np.shape(kernel) != np.shape(grid):                                     

        diff_h  = np.shape(grid)[-2] - np.shape(kernel)[-2] 
        diff_w =  np.shape(grid)[-1] - np.shape(kernel)[-1] 
        pad_h = diff_h // 2
        pad_w = diff_w // 2

        rh, rw = diff_h % pad_h, diff_w % pad_w

        if rh:
            hp = rh
            hm = 0
        else:
            hp = 1
            hm = -1

        if rw:
            wp = rw
            wm = 0
        else:
            wp = 1
            wm = -1

        padded_kernel = np.pad(kernel, \
                ((0,0), (0,0), (pad_h+hp, pad_h+hm), (pad_w+wp, pad_w+wm)))

    else:                                                                       
        padded_kernel = kernel                                                  
                                                                                
    fourier_kernel = np.fft.fft2(np.fft.fftshift(padded_kernel, axes=(-2,-1)), axes=(-2,-1))
    fourier_grid = np.fft.fft2(np.fft.fftshift(grid, axes=(-2,-1)), axes=(-2,-1))
    fourier_product = fourier_grid * fourier_kernel 
    real_spatial_convolved = np.real(np.fft.ifft2(fourier_product, axes=(-2,-1)))
    convolved = np.fft.ifftshift(real_spatial_convolved, axes=(-2, -1))
                                                                                
    return convolved 
