# Examples for NeuroProof [![Picture](https://raw.github.com/janelia-flyem/janelia-flyem.github.com/master/images/HHMI_Janelia_Color_Alternate_180x40.png)](http://www.janelia.org)

This directory contains examples for how to run the neuroproof
(https://github.com/janelia-flyem/NeuroProof) utilities
along with some sample data from FlyEM (http://janelia.org/team-project/fly-em)
and their efforts to reconstruct neurons from the Drosophila medulla prepared
using FIB-SEM imaging.  The h5 files and xml need to be unzipped before using.  The h5 file contain segmentation as a combination of two datasets: 'stack' and 'transforms'.  'stack' is a 3D volume (z,y,x) of supervoxel labels.  'transforms' is a mapping of supervoxels to bodies.  In some cases, 'transforms' is simply an identify and 'stack' represents the final segmentation or ground truth.

There are three EM data samples from the medulla in 'training_sample1', 'training_sample2',
and 'validation_sample'.  While any of these sample could be used as a validation stack or
training stack, in our workflow, we have used the first sample to train a boundary
classifier using Ilastik, the second sample to train our agglomeration using neuroproof, and the validation
stack to test our agglomeration and analysis algorithms in neuroproof.

## 1.  Boundary Training

The 'training_sample1' directory contains a volume that was used to train a boundary
classifier to used in subsequent parts of the segmentation flow.  To
generate a classifier, one should use the Ilastik tool from Heidelberg
(http://www.ilastik.org).  One should open up the grayscale stack,
create multiple class labels (where class label 0 should be designated as
classifying cellular boundaries).  These labels can be efficiently
trained and saved to an ILP file format.  We provide an example from
our own pipeline for comparisons 'training_sample1/results/boundary_classifier_ilastik.ilp'.

## 2.  Agglomeration Training

The 'training_sample2' directory contains a grayscale image volume and its labeled groundtruth
and is used to train an agglomeration classifier.  This agglomeration
classifier is produced by calling 'neuroproof_graph_train'.  The inputs required
is an oversegmented (watershed) volume 'oversegmented_stack_labels.h5'
and its respective groundtruth 'groundtruth.h5'.  Also,
a prediction file ('prediction.h5') that contains multiple channels of class predictions for
each label needs to be provided.  (For now channel 0 is always treated as the
boundary channel and channel 2 is the mitochondria channel IF mitochondria options
are enabled in the pipeline, otherwise the predictions can be generic.)

The oversegmented volume can be achieved by running the Ilasitik boundary
predictor over the image stack and generating a watershed.  For now, NeuroProof
does not support this operation; however, the open-source tool Gala
(https://github.com/janelia-flyem/gala) can be used to take the boundary
classification file produced by Ilastik and an image volume to produce an
initial oversegmented stack.  An example of how Gala could be called:

gala-segmentation-pipeline -I '|graymaps|/*.png' --ilp-file |boundary classifier| --enable-gen-supervoxels
                --enable-gen-agglomeration --enable-gen-pixel --seed-size 5 |output directory| --segmentation-thresholds 0.0 

The output directory will contain a label volume, as well as, a prediction file and
other data not essential for agglomeration training.

Once an over-segmentation and ground-truth labeling exists, 'neuroproof_graph_learn' will learn a supervoxel boundary classifier for agglomeration using algorithms proposed by  [Parag, et al '15](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0125825), 
[Nunez-Iglesias et al '13] (http://arxiv.org/abs/1303.6163).  For segmentation of EM images, the following command can be used to train a superpixel boundary classifier in a context-aware fashion as described in [Parag, et al '15](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0125825) provided mitochondria detection. Setting 'num_iterations' to 1 enforces a "flat" learning -- only the supervoxels in the original over-segmentation are used for learning the boundary classifier.

neuroproof_graph_learn |oversegmented_labels| |prediction| |groundtruth| --num_iterations 1

Context-aware training with mitochondria prediction can be disabled by setting 'use_mito' parameter to 0. By setting the 'num_iterations' to more than 1, one can augment the training set by agglomerating the over-segmented volume with the boundary classifier trained in the previous iteration as discussed in [Nunez-Iglesias et al '13] (http://arxiv.org/abs/1303.6163). 

neuroproof_graph_learn |oversegmented_labels| |prediction| |groundtruth|  --use_mito 0 --num_iterations 5

The output of these procedures is a superpixel boundary  classifier.  We provide example
classifiers produced from the neuroproof using these two commands -- 'mito_aware.xml'
and  'nomito_aware.h5' respectively.

It is also possible to train the boundary classifier interactively when an exhaustive segmentation groundtruth is not available using the algorithm proposed in [Parag, et.al. 14] (http://www.researchgate.net/publication/265683774_Small_Sample_Learning_of_Superpixel_Classifiers_for_EM_Segmentation). This option has been implemented for 3D data in this package. Please let us know if you wish use this feature.

## 3.  Agglomeration Procedure

Once an agglomeration classifier is produced, one can efficiently apply this
classifier to a new dataset and generate a segmentation.  This is done by
calling 'neuroproof_graph_predict'. To run this algorithm, one needs an agglomeration
classifier as produced in #2, an over-segmented volume to be segmented, and a prediction
file.  As before, this over-segmented volume and prediction file will be produced
using Gala which needs a boundary classifier (#1) and an initial image volume.
(We provide the oversegmented volume for convencience in 'validation_sample/oversegmented_stack_labels.h5',
but omit including the boundary prediction for this volume due to its large
size.  To run the following example with the validaation sample, one will need
to run gala over its original grayscales.)

neuroproof_graph_predict |oversegmented_labels| |prediction| |classifier| 

By default, this code assumes the channel 2 contains the pixewise detection confidences for mitochondria (channel 0: membrane, channel 1:cytoplasm probabilities). The agglomeration is performaed in two phases, as described in [Parag, et al '15](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0125825). To disable this mode, add the following option "--merge_mito 0" after the above command. There are several thresholds that can also be defined for the aglomeration, check the code for details.

This will produce two files: a segmented label volume and graph.json which describes
the certainty of an edge be a true edge in the graph.  This graph file can be
used to focus manual correcting efforts in the segmentation.  These outputs are
provided in the results directory.

The prediction algorithm gives the option to specify a separate classifier for
annotating certainty values on the final exported graph (--postseg-classifier-file).
For example, one can use the mito-aware classifier to separate out mitochondria
properly when doing multi-stage classification, but then use the non-mito
classifier to generate the graph file so that edge certainty of mitochondria
bodies are treated in similar way to non-mitochondria bodies.

## 4.  Segmentation and Graph Analysis

The uncertainty in the resulting graph 'validation_sample/results/graph.json'
produced in #3 can be analyzed using the following NeuroProof command:

neuroproof_graph_analyze -g 1 -b 1

This provides an estimate of how many edits will be needed to fix the graph
and the overall uncertainty of the segmentation as defined in [Plaza '12] 
(http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0044448).

To compare the segmentation to ground truth using the similarity
metric Variance of Information (VI), one can issue the following command:

neuroproof_graph_analyze_gt |segmentation| |groundtruth|

We provide a ground-truth labeling in 'validation_sample/groundtruth.h5'
and a test segmentation in 'validation_sample/results/segmentation.h5'
to allow one to test these command.

## 5. GUI stack viewer

NeuroProof provides a front-end for viewing the results of segmentation
and comparing that segmentation to ground truth.  It also provides the framework
for modifying the segmentation manually.  This manual modification depends
on the prioritization algorithm used and is a work in progress.

To launch the stack viewer:

neuroproof_stack_viwer

This will launch a GUI that allows one to load or create a new stack session
for viewing a segmentation.  If one already has a saved stack session, one could
call:

neuroproof_stack_viewer --session-name <directory of session>

To create a new session, click on File->New Session.  The GUI will prompt the user
for a list of PNG or JPG files corresponding to a stack of grayscale images.  Navigate
to a directory with a set of these images, select all of the files and click okay.
For now, this list of grayscale files must be sortable in the order in which they
should be displayed.
After this, the user will be given another prompt for the label volume (segmentation).
For instance, this label volume could be the segmentation h5 file produced by the neuroproof
predict executable.

Note: users should not try to open volumes much bigger than 500x500x500 (as in this
example).  The stack viewer loads the whole volume in memory and is meant to explore
aspects of the segmentation very efficiently.

The user will see a planar viewer of the segmentation with controls and a 3D viewer
on the left side.  Only use the 3D viewer if necessary since it will take a couple of
seconds to open.  Future improvements to the viewer might look to optimize this loading
time considerably.  Options->Shortcuts show a list of keyboard or mouse shortcuts for
the viewer.  If a user shift clicks a body, just that body will show.  Multiple bodies
can be selected as well.  These are the bodies that will appear in the 3D window if
the window is enabled.  The 3D window can be popped out and resized as desired.

If a user wants to modify the segmentation, they can click on the training mode.  This
takes the user to a sequencer that orders pairs of bodies to examine.  The training
mode can only be added if a session location has been created 'File->Save As Session'.
The training mode could be used to refine the segmentation in theory but is meant for
training a classifier.  This capability will be expanded on in the future but is just
a placeholder now.

If a user wants to view their segmentation and also a ground truth labeling, a ground
truth can be associated with the session by clicking 'File->Add GT'.  This ground
truth might have been optained using segmentation revision tools such as
[Raveler](https://openwiki.janelia.org/wiki/display/flyem/Raveler).

In this example directory, use the grayscales from validation_sample/grayscale_maps
and the segmentation from validation_sample/results/segmentation.h5 (unzip first).
For ground truth, use validation_sample/groundtruth.h5 (unzip first).










