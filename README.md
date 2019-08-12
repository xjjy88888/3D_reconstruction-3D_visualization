This repo is used to do cellular segmentation for 3D reconstruction of tomographic images using FFN by Google.  
Meanwhile, u can make a 3D visualization in your own browser based on WebGL.  
Thanks to google for the code, i put the reference link at bottom.  

# Part1: Cellular-Segmentation
    
## Flood-Filling Networks

Flood-Filling Networks (FFNs) are a class of neural networks designed for
instance segmentation of complex and large shapes, particularly in volume
EM datasets of brain tissue.

For more details, see the related publications:

 * https://arxiv.org/abs/1611.00421
 * https://doi.org/10.1101/200675


## Installation

No installation is required. To install the necessary dependencies, run:

```shell
  pip install -r requirements.txt
```

## Preparing the training data


FFN networks can be trained with the `train.py` script, which expects a
TFRecord file of coordinates at which to sample data from input volumes.
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

Note that both training and inference with the provided model are computationally expensive processes. 
You can reduce the batch size, model depth, `fov_size`, or number of features in
the convolutional layers to reduce the memory usage.


## Inference

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

```
jupyter load.ipynb
```

We provide sample segmentation results in `results/fib25/sample-training2.npz`.


For an interactive setting, check out `ffn_inference_demo.ipynb`. This Jupyter
notebook shows how to segment a single object with an explicitly defined
seed and visualize the results while inference is running.

Both examples are configured to use a 3d convstack FFN model trained on the
`validation1` volume of the FIB-25 dataset from the FlyEM project at Janelia.


# Part2: 3D-visualization
Neuroglancer is a WebGL-based viewer for volumetric data.  It is capable of displaying arbitrary (non axis-aligned) cross-sectional views of volumetric data, as well as 3-D meshes and line-segment based models (skeletons).
  
## Supported data sources

Neuroglancer itself is purely a client-side program, but it depends on data being accessible via HTTP in a suitable format.  It is designed to easily support many different data sources, and there is existing support for the following data APIs/formats:

- BOSS <https://bossdb.org/>
- DVID <https://github.com/janelia-flyem/dvid>
- Render <https://github.com/saalfeldlab/render>
- [Precomputed chunk/mesh fragments exposed over HTTP](src/neuroglancer/datasource/precomputed)
- Single NIfTI files <https://www.nitrc.org/projects/nifti>
- [Python in-memory volumes](python/README.md) (with automatic mesh generation)
- N5 <https://github.com/saalfeldlab/n5>

## Supported browsers

- Chrome >= 51
- Firefox >= 46

## Keyboard and mouse bindings

For the complete set of bindings, see
[src/neuroglancer/ui/default_input_event_bindings.ts](src/neuroglancer/default_input_event_bindings.ts),
or within Neuroglancer, press `h` or click on the button labeled `?` in the upper right corner.

- Click on a layer name to toggle its visibility.

- Double-click on a layer name to edit its properties.

- Hover over a segmentation layer name to see the current list of objects shown and to access the opacity sliders.

- Hover over an image layer name to access the opacity slider and the text editor for modifying the [rendering code](src/neuroglancer/sliceview/image_layer_rendering.md).

## Troubleshooting

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
    
## Multi-threaded architecture

In order to maintain a responsive UI and data display even during rapid navigation, work is split between the main UI thread (referred to as the "frontend") and a separate WebWorker thread (referred to as the "backend").  This introduces some complexity due to the fact that current browsers:
 - do not support any form of *shared* memory or standard synchronization mechanism (although they do support relatively efficient *transfers* of typed arrays between threads);
 - require that all manipulation of the DOM and the WebGL context happens on the main UI thread.

The "frontend" UI thread handles user actions and rendering, while the "backend" WebWorker thread handle all queuing, downloading, and preprocessing of data needed for rendering.

## Documentation Index

- [Image Layer Rendering](src/neuroglancer/sliceview/image_layer_rendering.md)
- [Cross-sectional view implementation architecture](src/neuroglancer/sliceview/README.md)
- [Compressed segmentation format](src/neuroglancer/sliceview/compressed_segmentation/README.md)
- [Data chunk management](src/neuroglancer/chunk_manager/)
- [On-GPU hashing](src/neuroglancer/gpu_hash/)

## how to show 3D visualization locally

node.js is required to build the viewer.

```
cd /nvm_nodejs
```

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


## Creating a dependent project

See [examples/dependent-project](examples/dependent-project).



## Related Projects

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





## Referennce
1. https://github.com/google/ffn/  
2. https://github.com/nvm-sh/nvm  
3. https://github.com/google/neuroglancer  
