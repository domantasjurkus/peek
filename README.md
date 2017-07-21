## CS Sideproject - Manufacturing Quality Assurance

The purpouse of this project is to explore how product manufacturing quality can be ensured using technologies such as computer vision or machine learning. To keep the project withing a manageable scope, the quality of Cooper Software coasters shall be inspected. After an initial investigation into the possible technologies, two methods of carrying out the project emerged.

# Option 1: Computer Vision

OpenCV is a handy library that has many tools for image manipulation and analysis.

Possible steps:

`*` Detect scale-invariant features (SIFT) of a sample control coaster and a target coaster
`*` Align both coasters
`*` Calculate edge difference
`*` Smooth differnece in case alignment is not optimal
`*` Represent edge diference in a numerical format

`+` Single test sample required
`+` Possibly generalisable to any arbitrary shape/product
`-` Alignment/rotation can be tricky/computationally expensive

OpenCV 3.0 has removed SIFT and SURF operators - those need to be added while rebuilding OpenCV.



# Option 2: Machine Learning

Tensorflow is a popular machine learning library that has a retrainable model called Inception. The final layer of this model can be adapted to recognise new sets of images.

Steps:

`1` Gather many samples of quality and non-quality coasters
`2` Train a model on the sample data (good option: Tensorflow's Inception model)
`3` Evaluate the model by passing good/bad coaster images

`+` Relatively simple implementation
`-` Many samples required
`-` Reliance on a complex model
`-` Need to provide many *bad* examples

Investigation 1:
The model was trained with 77 good and 48 bad examples with some variation in background, lighting and positioning of the coaster. Damaged costers had some parts covered by a blank piece of paper, emulating a missing shape or contour.

Running the classifier on images of unseen damaged coasters, it was able to identify 4 out of 5 correctly. The mis-identified coaster did not have any obvious difference for other damaged smaples

```
damaged/01.jpg bad (score = 0.97133) good (score = 0.02867)
damaged/02.jpg bad (score = 0.87462) good (score = 0.12538)
damaged/03.jpg bad (score = 0.83371) good (score = 0.16629)
damaged/04.jpg bad (score = 0.94451) good (score = 0.05549)
damaged/05.jpg good (score = 0.76480) bad (score = 0.23520)

```

This accuracy can be further improved with more traning samples. The model does not mis-identify good coasters that were not part of the training data.
