# Pure Pytorch implementation of RGB to HSV/HSL conversion

This repository implement pytorch Tensor color space 
conversion from rgb to hsv/hsl and backwards.

## Notes 
1. The conversion process is differentiable in a natural way,
but since the mapping from RGB to HSV/HSL space is not 
continuous in some place, it may not be a good idea to 
perform back propagation.

2. The code was written to accept only image like tensors 
`tensor.shape == [B x 3 x H x W]`, but it's easy to modify
the code to accept other shapes.

3. Reference is in [here](https://www.rapidtables.com/convert/color/index.html)

## Usage

example usage can be find in `test.py`. 
Before using the function, please make sure that:
1. The input/output rbg/hsv/hsl tensors should all be normalized to 0~1 for each channel
2. The rgb format is RGB instead of BGR
2. The shape of the tensor match the requirement
 