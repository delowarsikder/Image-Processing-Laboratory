# edge-detection-using-sobel-filter
Here use convulation with sobel filter from scratch in opencv python .<br>
<br>
have to detect edge using sobel filter.<br>
  * sobel x derivative kernel (detect vertical edge)<br>
    -1 0 1<br>
    -2 0 2<br>
    -1 0 1<br>
  * sobel y derivative kernel (detect horizontal edge)<br>
    -1 -2 -1<br>
     0  0  0<br>
     1  2  1<br>

### input image
![This is an image](https://github.com/nahid0335/edge-detection-using-sobel-filter/blob/main/lena.jpg)

### output x derivative
![This is an image](https://github.com/nahid0335/edge-detection-using-sobel-filter/blob/main/sobel_x.PNG)

### output y derivative
![This is an image](https://github.com/nahid0335/edge-detection-using-sobel-filter/blob/main/sobel_y.PNG)

### combine of x and y derivative
![This is an image](https://github.com/nahid0335/edge-detection-using-sobel-filter/blob/main/sobel_output.PNG)
