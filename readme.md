# Weighted Least Squares Filter

This project implements Farbman, Fattal, Lischinski, and Szeliski's "Edge-Preserving Decompositions for Multi-Scale Tone and Detail Manipulation."

![Tone Manipulation](result/mesa.gif)

Image from author's original dataset and courtesy of Norman Koren, [www.normankoren.com](http://www.normankoren.com)

As illustrated above, this filter has applications in edge-preserving smoothing, HDR tone manipulation, detail enhancement, and non-photorealistic rendering.

## Usage

`python tone_manipulation.py --img_path=<img_path>`

* `img_path` - Path to an image.

An OpenCV application showing the processed image will pop up. The controls for this application are:

* Sliders to change the weighting between detail layers and the base layer decomposition.
* Sliders (as proposed in the paper) to correct exposure and saturation after weighting the decomposition layers.
* 0-3 to change the mask level of the detail mask. Masked out detail layers will not appear in the final image. `a` and `d` affect the radius of the mask cursor, and clicking the image applies the detail mask at the selected location. 

## Dependencies

* Python - Tested on version 3.7.0
* NumPy - Tested on version 1.15.0
* OpenCV - Tested on version 3.4.1
* Scipy - Tested on version 0.19.1

