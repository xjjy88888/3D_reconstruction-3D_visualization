# Cellular-Segmentation
Cellular Segmentation for 3D reconstruction of tomographic images using FFN by Google.
# Flood-Filling Networks

Flood-Filling Networks (FFNs) are a class of neural networks designed for
instance segmentation of complex and large shapes, particularly in volume
EM datasets of brain tissue.

For more details, see the related publications:

 * https://arxiv.org/abs/1611.00421
 * https://doi.org/10.1101/200675

This is not an official Google product.

# Installation

No installation is required. To install the necessary dependencies, run:

```shell
  pip install -r requirements.txt
```

The code has been tested on an Ubuntu 16.04.3 LTS system equipped with a
Tesla P100 GPU.

# Training

FFN networks can be trained with the `train.py` script, which expects a
TFRecord file of coordinates at which to sample data from input volumes.

## Preparing the training data

There are two scripts to generate training coordinate files for
a labeled dataset stored in HDF5 files: `compute_partitions.py` and
`build_coordinates.py`.

`compute_partitions.py` transforms the label volume into an intermediate
volume where the value of every voxel `A` corresponds to the quantized
fraction of voxels labeled identically to `A` within a subvolume of
radius `lom_radius` centered at `A`. `lom_radius` should normally be
set to `(fov_size // 2) + deltas` (where `fov_size` and `deltas` are
FFN model settings). Every such quantized fraction is called a *partition*.
Sample invocation:

```shell
  python compute_partitions.py \
    --input_volume third_party/neuroproof_examples/validation_sample/groundtruth.h5:stack \
    --output_volume third_party/neuroproof_examples/validation_sample/af.h5:af \
    --thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
    --lom_radius 24,24,24 \
    --min_size 10000
```

`build_coordinates.py` uses the partition volume from the previous step
to produce a TFRecord file of coordinates in which every partition is
represented approximately equally frequently. Sample invocation:

```shell
  python build_coordinates.py \
     --partition_volumes validation1:third_party/neuroproof_examples/validation_sample/af.h5:af \
     --coordinate_output third_party/neuroproof_examples/validation_sample/tf_record_file \
     --margin 24,24,24
```

## Sample data

We provide a sample coordinate file for the FIB-25 `validation1` volume
included in `third_party`. Due to its size, that file is hosted in
Google Cloud Storage. If you haven't used it before, you will need to
install the Google Cloud SDK and set it up with:

```shell
  gcloud auth application-default login
```

You will also need to create a local copy of the labels and image with:

```shell
  gsutil rsync -r -x ".*.gz" gs://ffn-flyem-fib25/ third_party/neuroproof_examples
```

## Running training

Once the coordinate files are ready, you can start training the FFN with:

```shell
  python train.py \
    --train_coords gs://ffn-flyem-fib25/validation_sample/fib_flyem_validation1_label_lom24_24_24_part14_wbbox_coords-*-of-00025.gz \
    --data_volumes validation1:third_party/neuroproof_examples/validation_sample/grayscale_maps.h5:raw \
    --label_volumes validation1:third_party/neuroproof_examples/validation_sample/groundtruth.h5:stack \
    --model_name convstack_3d.ConvStack3DFFNModel \
    --model_args "{\"depth\": 12, \"fov_size\": [33, 33, 33], \"deltas\": [8, 8, 8]}" \
    --image_mean 128 \
    --image_stddev 33
```

Note that both training and inference with the provided model are
computationally expensive processes. We recommend a GPU-equipped machine
for best results, particularly when using the FFN interactively in a Jupyter
notebook. Training the FFN as configured above requires a GPU with 12 GB of RAM.
You can reduce the batch size, model depth, `fov_size`, or number of features in
the convolutional layers to reduce the memory usage.

The training script is not configured for multi-GPU or distributed training.
For instructions on how to set this up, see the documentation on
[Distributed TensorFlow](https://www.tensorflow.org/deploy/distributed#replicated_training).

# Inference

We provide two examples of how to run inference with a trained FFN model.
For a non-interactive setting, you can use the `run_inference.py` script:

```shell
  python run_inference.py \
    --inference_request="$(cat configs/inference_training_sample2.pbtxt)" \
    --bounding_box 'start { x:0 y:0 z:0 } size { x:250 y:250 z:250 }'
```

which will segment the `training_sample2` volume and save the results in
the `results/fib25/training2` directory. Two files will be produced:
`seg-0_0_0.npz` and `seg-0_0_0.prob`. Both are in the `npz` format and
contain a segmentation map and quantized probability maps, respectively.
In Python, you can load the segmentation as follows:

```python
  from ffn.inference import storage
  seg, _ = storage.load_segmentation('results/fib25/training2', (0, 0, 0))
```

We provide sample segmentation results in `results/fib25/sample-training2.npz`.
For the training2 volume, segmentation takes ~7 min with a P100 GPU.

For an interactive setting, check out `ffn_inference_demo.ipynb`. This Jupyter
notebook shows how to segment a single object with an explicitly defined
seed and visualize the results while inference is running.

Both examples are configured to use a 3d convstack FFN model trained on the
`validation1` volume of the FIB-25 dataset from the FlyEM project at Janelia.



Neuroglancer is a WebGL-based viewer for volumetric data.  It is capable of displaying arbitrary (non axis-aligned) cross-sectional views of volumetric data, as well as 3-D meshes and line-segment based models (skeletons).

This is not an official Google product.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Travis CI Build Status](https://travis-ci.org/google/neuroglancer.svg?branch=master)](https://travis-ci.org/google/neuroglancer)
[![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/2npw99gr2x7kh763/branch/master?svg=true)](https://ci.appveyor.com/project/jbms/neuroglancer/branch/master)

# Examples

A live demo is hosted at <https://neuroglancer-demo.appspot.com>.  (The prior link opens the viewer without any preloaded dataset.)  Use the viewer links below to open the viewer preloaded with an example dataset.

The four-pane view consists of 3 orthogonal cross-sectional views as well as a 3-D view (with independent orientation) that displays 3-D models (if available) for the selected objects.  All four views maintain the same center position.  The orientation of the 3 cross-sectional views can also be adjusted, although they maintain a fixed orientation relative to each other.  (Try holding the shift key and either dragging with the left mouse button or pressing an arrow key.)

- Kasthuri et al., 2014.  Mouse somatosensory cortex (6x6x30 cubic nanometer resolution). <a href="https://neuroglancer-demo.appspot.com/#!{'layers':{'original-image':{'type':'image'_'source':'precomputed://gs://neuroglancer-public-data/kasthuri2011/image'_'visible':false}_'corrected-image':{'type':'image'_'source':'precomputed://gs://neuroglancer-public-data/kasthuri2011/image_color_corrected'}_'ground_truth':{'type':'segmentation'_'source':'precomputed://gs://neuroglancer-public-data/kasthuri2011/ground_truth'_'selectedAlpha':0.63_'notSelectedAlpha':0.14_'segments':['3208'_'4901'_'13'_'4965'_'4651'_'2282'_'3189'_'3758'_'15'_'4027'_'3228'_'444'_'3207'_'3224'_'3710']}}_'navigation':{'pose':{'position':{'voxelSize':[6_6_30]_'voxelCoordinates':[5523.99072265625_8538.9384765625_1198.0423583984375]}}_'zoomFactor':22.573112129999547}_'perspectiveOrientation':[-0.004047565162181854_-0.9566211104393005_-0.2268827110528946_-0.1827099621295929]_'perspectiveZoom':340.35867907175077}" target="_blank">Open viewer.</a>

  This dataset was copied from <https://neurodata.io/data/kasthuri15/> and is made available under the [Open Data Common Attribution License](http://opendatacommons.org/licenses/by/1.0/).  Paper: <a href="http://dx.doi.org/10.1016/j.cell.2015.06.054" target="_blank">Kasthuri, Narayanan, et al.  "Saturated reconstruction of a volume of neocortex." Cell 162.3 (2015): 648-661.</a>
  
- Janelia FlyEM FIB-25.  7-column Drosophila medulla (8x8x8 cubic nanometer resolution).  <a href="https://neuroglancer-demo.appspot.com/#!{'layers':{'image':{'type':'image'_'source':'precomputed://gs://neuroglancer-public-data/flyem_fib-25/image'}_'ground-truth':{'type':'segmentation'_'source':'precomputed://gs://neuroglancer-public-data/flyem_fib-25/ground_truth'_'segments':['21894'_'22060'_'158571'_'24436'_'2515']}}_'navigation':{'pose':{'position':{'voxelSize':[8_8_8]_'voxelCoordinates':[2914.500732421875_3088.243408203125_4045]}}_'zoomFactor':30.09748283999932}_'perspectiveOrientation':[0.3143535554409027_0.8142156600952148_0.4843369424343109_-0.06040262430906296]_'perspectiveZoom':443.63404517712684_'showSlices':false}" target="_blank">Open viewer.</a>

  This dataset was copied from <https://www.janelia.org/project-team/flyem/data-and-software-release>, and is made available under the [Open Data Common Attribution License](http://opendatacommons.org/licenses/by/1.0/).  Paper: <a href="http://dx.doi.org/10.1073/pnas.1509820112" target="_blank">Takemura, Shin-ya et al. "Synaptic Circuits and Their Variations within Different Columns in the Visual System of Drosophila."  Proceedings of the National Academy of Sciences of the United States of America 112.44 (2015): 13711-13716.</a>
  
- FAFB: A Complete Electron Microscopy Volume of the Brain of Adult Drosophila melanogaster. <a href="https://neuroglancer-demo.appspot.com/?#!%7B%22layers%22:%7B%22fafb_v14%22:%7B%22source%22:%22precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig%22%2C%22type%22:%22image%22%7D%2C%22fafb_v14_clahe%22:%7B%22source%22:%22precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_clahe%22%2C%22type%22:%22image%22%2C%22visible%22:false%7D%2C%22neuropil-regions-surface%22:%7B%22type%22:%22segmentation%22%2C%22mesh%22:%22precomputed://gs://neuroglancer-fafb-data/elmr-data/FAFBNP.surf/mesh%22%2C%22segments%22:%5B%221%22%2C%2210%22%2C%2211%22%2C%2212%22%2C%2213%22%2C%2214%22%2C%2215%22%2C%2216%22%2C%2217%22%2C%2218%22%2C%2219%22%2C%222%22%2C%2220%22%2C%2221%22%2C%2222%22%2C%2223%22%2C%2224%22%2C%2225%22%2C%2226%22%2C%2227%22%2C%2228%22%2C%2229%22%2C%223%22%2C%2230%22%2C%2231%22%2C%2232%22%2C%2233%22%2C%2234%22%2C%2235%22%2C%2236%22%2C%2237%22%2C%2238%22%2C%2239%22%2C%224%22%2C%2240%22%2C%2241%22%2C%2242%22%2C%2243%22%2C%2244%22%2C%2245%22%2C%2246%22%2C%2247%22%2C%2248%22%2C%2249%22%2C%225%22%2C%2250%22%2C%2251%22%2C%2252%22%2C%2253%22%2C%2254%22%2C%2255%22%2C%2256%22%2C%2257%22%2C%2258%22%2C%2259%22%2C%226%22%2C%2260%22%2C%2261%22%2C%2262%22%2C%2263%22%2C%2264%22%2C%2265%22%2C%2266%22%2C%2267%22%2C%2268%22%2C%2269%22%2C%227%22%2C%2270%22%2C%2271%22%2C%2272%22%2C%2273%22%2C%2274%22%2C%2275%22%2C%228%22%2C%229%22%5D%7D%2C%22neuropil-full-surface%22:%7B%22type%22:%22mesh%22%2C%22source%22:%22vtk://https://storage.googleapis.com/neuroglancer-fafb-data/elmr-data/FAFB.surf.vtk.gz%22%2C%22vertexAttributeSources%22:%5B%5D%2C%22shader%22:%22void%20main%28%29%20%7B%5Cn%20%20emitRGBA%28vec4%281.0%2C%200.0%2C%200.0%2C%200.5%29%29%3B%5Cn%7D%5Cn%22%2C%22visible%22:false%7D%7D%2C%22navigation%22:%7B%22pose%22:%7B%22position%22:%7B%22voxelSize%22:%5B4%2C4%2C40%5D%2C%22voxelCoordinates%22:%5B123943.625%2C73323.8828125%2C5234%5D%7D%7D%2C%22zoomFactor%22:1210.991144617663%7D%2C%22perspectiveOrientation%22:%5B-0.28037887811660767%2C-0.19049881398677826%2C-0.13574382662773132%2C-0.9309519529342651%5D%2C%22perspectiveZoom%22:21335.91710335963%2C%22layout%22:%22xy-3d%22%7D" target="_blank">Open viewer.</a>

  This dataset was copied from <https://www.temca2data.org/>, and is made available under the [CC-BY-NC 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).  The surface meshes were copied from <https://jefferis.github.io/elmr/reference/FAFB.surf.html>.
  
  Paper: <a href="https://doi.org/10.1016/j.cell.2018.06.019" target="_blank">Zhihao Zheng et al. "A Complete Electron Microscopy Volume of the Brain of Adult Drosophila melanogaster" Cell 174.3 (2018): 730-743.</a>
  
# Supported data sources

Neuroglancer itself is purely a client-side program, but it depends on data being accessible via HTTP in a suitable format.  It is designed to easily support many different data sources, and there is existing support for the following data APIs/formats:

- BOSS <https://bossdb.org/>
- DVID <https://github.com/janelia-flyem/dvid>
- Render <https://github.com/saalfeldlab/render>
- [Precomputed chunk/mesh fragments exposed over HTTP](src/neuroglancer/datasource/precomputed)
- Single NIfTI files <https://www.nitrc.org/projects/nifti>
- [Python in-memory volumes](python/README.md) (with automatic mesh generation)
- N5 <https://github.com/saalfeldlab/n5>

# Supported browsers

- Chrome >= 51
- Firefox >= 46

# Keyboard and mouse bindings

For the complete set of bindings, see
[src/neuroglancer/ui/default_input_event_bindings.ts](src/neuroglancer/default_input_event_bindings.ts),
or within Neuroglancer, press `h` or click on the button labeled `?` in the upper right corner.

- Click on a layer name to toggle its visibility.

- Double-click on a layer name to edit its properties.

- Hover over a segmentation layer name to see the current list of objects shown and to access the opacity sliders.

- Hover over an image layer name to access the opacity slider and the text editor for modifying the [rendering code](src/neuroglancer/sliceview/image_layer_rendering.md).

# Troubleshooting

- Neuroglancer doesn't appear to load properly.

  Neuroglancer requires WebGL (2.0) and the `EXT_color_buffer_float` extension.
  
  To troubleshoot, check the developer console, which is accessed by the keyboard shortcut `control-shift-i` in Firefox and Chrome.  If there is a message regarding failure to initialize WebGL, you can take the following steps:
  
  - Chrome
  
    Check `chrome://gpu` to see if your GPU is blacklisted.  There may be a flag you can enable to make it work.
    
  - Firefox

    Check `about:support`.  There may be webgl-related properties in `about:config` that you can change to make it work.  Possible settings:
    - `webgl.disable-fail-if-major-performance-caveat = true`
    - `webgl.force-enabled = true`
    - `webgl.msaa-force = true`
    
- Failure to access a data source.

  As a security measure, browsers will in many prevent a webpage from accessing the true error code associated with a failed HTTP request.  It is therefore often necessary to check the developer tools to see the true cause of any HTTP request error.

  There are several likely causes:
  
  - [Cross-origin resource sharing (CORS)](https://en.wikipedia.org/wiki/Cross-origin_resource_sharing)
  
    Neuroglancer relies on cross-origin requests to retrieve data from third-party servers.  As a security measure, if an appropriate `Access-Control-Allow-Origin` response header is not sent by the server, browsers prevent webpages from accessing any information about the response from a cross-origin request.  In order to make the data accessible to Neuroglancer, you may need to change the cross-origin request sharing (CORS) configuration of the HTTP server.
  
  - Accessing an `http://` resource from a Neuroglancer client hosted at an `https://` URL
    
    As a security measure, recent versions of Chrome and Firefox prohibit webpages hosted at `https://` URLs from issuing requests to `http://` URLs.  As a workaround, you can use a Neuroglancer client hosted at a `http://` URL, e.g. the demo client running at http://neuroglancer-demo.appspot.com, or one running on localhost.  Alternatively, you can start Chrome with the `--disable-web-security` flag, but that should be done only with extreme caution.  (Make sure to use a separate profile, and do not access any untrusted webpages when running with that flag enabled.)
    
# Multi-threaded architecture

In order to maintain a responsive UI and data display even during rapid navigation, work is split between the main UI thread (referred to as the "frontend") and a separate WebWorker thread (referred to as the "backend").  This introduces some complexity due to the fact that current browsers:
 - do not support any form of *shared* memory or standard synchronization mechanism (although they do support relatively efficient *transfers* of typed arrays between threads);
 - require that all manipulation of the DOM and the WebGL context happens on the main UI thread.

The "frontend" UI thread handles user actions and rendering, while the "backend" WebWorker thread handle all queuing, downloading, and preprocessing of data needed for rendering.

# Documentation Index

- [Image Layer Rendering](src/neuroglancer/sliceview/image_layer_rendering.md)
- [Cross-sectional view implementation architecture](src/neuroglancer/sliceview/README.md)
- [Compressed segmentation format](src/neuroglancer/sliceview/compressed_segmentation/README.md)
- [Data chunk management](src/neuroglancer/chunk_manager/)
- [On-GPU hashing](src/neuroglancer/gpu_hash/)

# Building

node.js is required to build the viewer.

1. First install NVM (node version manager) per the instructions here:

  https://github.com/creationix/nvm

2. Install a recent version of Node.js if you haven't already done so:

    `nvm install stable`
    
3. Install the dependencies required by this project:

   (From within this directory)

   `npm i`

   Also re-run this any time the dependencies listed in [package.json](package.json) may have
   changed, such as after checking out a different revision or pulling changes.

4. To run a local server for development purposes:

   `npm run dev-server`
  
   This will start a server on <http://localhost:8080>.
   
5. To run the unit test suite on Chrome:
   
   `npm test`
   
   To run only tests in files matching a given regular expression pattern:
   
   `npm test -- --pattern='<pattern>'`
   
   For example,
   
   `npm test -- --pattern='util/uint64'`

6. See [package.json](package.json) for other commands available.

# Creating a dependent project

See [examples/dependent-project](examples/dependent-project).

# Discussion Group

There is a Google Group/mailing list for discussion related to Neuroglancer:
<https://groups.google.com/forum/#!forum/neuroglancer>.

# Related Projects

- [nyroglancer](https://github.com/funkey/nyroglancer) - Jupyter notebook extension for visualizing
  Numpy arrays with Neuroglancer.
- [4Quant/neuroglancer-docker](https://github.com/4Quant/neuroglancer-docker) - Example setup for
  Docker deployment of the [Neuroglancer Python integration](python/README.md).
- [FZJ-INM1-BDA/neuroglancer-scripts](https://github.com/FZJ-INM1-BDA/neuroglancer-scripts) -
  Scripts for converting the [BigBrain](https://bigbrain.loris.ca) dataset to the
  Neuroglancer [precomputed data format](src/neuroglancer/datasource/precomputed), which may serve
  as a useful example for converting other datasets.
- [BigArrays.jl](https://github.com/seung-lab/BigArrays.jl) - Julia interface of neuroglancer precomputed data format.
- [cloudvolume](https://github.com/seung-lab/cloud-volume) - Python interface of neuroglancer precomputed data format.



# Acknowledgements
Cross-browser Testing Platform Provided by [Sauce Labs](https://saucelabs.com)

# License

Copyright 2016 Google Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>.
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
