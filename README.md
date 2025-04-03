Processing code to: 1) classify gully presence/absence based on LUCAS survey observations within a scikit-learn workflow, 2) implement the classifier model on raster grids across Europe in a spatially chunked workflow design, and 3) automatically download photos form the LUCAS 2022 photo repository for LUCAS points of interest.

The standard implementation applies the predict probability function from the random forest classifer to approximate the probability of gully occurance at sites. This gully probability classifier is then implemented across the whole of Europe using feature arrays with a 100 meter resolution.

For licensing reasons, the spatial predictor layers to the classifier model are not shared. To reproduce this code, users should download and harmonise their own input features to a standard grid size and make the LUCAS point extractions. The code serves as a starting point for building machine learning predictions using the LUCAS observations combined with spatial features.

Note that for several reasons, such as offset between the gully location and LUCAS point location, as well as the design of the LUCAS photos module which are taken of the point and 4 cardinal directions, a large number of LUCAS photos will not contain the surveyed gully feature(s). The photos in many cases can only be used for gaining context on the land cover condition and environmental context of the gully features, rather than their direct visualisation. 

For questions and requests contact: fmatthews1381@gmail.com
