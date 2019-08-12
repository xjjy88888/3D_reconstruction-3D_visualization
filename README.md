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
Firstly, u need to

```
cd /3D_visualization
```
  
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


## how to show 3D visualization locally

node.js is required to build the viewer.
Firstly, u need to

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
   
6. To show 3D visualization locally,   
   You need to open the botton({}) which is in your top right-hand corner.    
   Then, you could paste in the following code.  
   Demo1: Kasthuri et al., 2014. Mouse somatosensory cortex (6x6x30 cubic nanometer resolution)     
   
```
{
  "layers": [
    {
      "source": "precomputed://gs://neuroglancer-public-data/kasthuri2011/image",
      "type": "image",
      "name": "original-image",
      "visible": false
    },
    {
      "source": "precomputed://gs://neuroglancer-public-data/kasthuri2011/image_color_corrected",
      "type": "image",
      "name": "corrected-image"
    },
    {
      "source": "precomputed://gs://neuroglancer-public-data/kasthuri2011/ground_truth",
      "type": "segmentation",
      "selectedAlpha": 0.63,
      "notSelectedAlpha": 0.14,
      "segments": [
        "13",
        "15",
        "2282",
        "3189",
        "3207",
        "3208",
        "3224",
        "3228",
        "3710",
        "3758",
        "4027",
        "444",
        "4651",
        "4901",
        "4965"
      ],
      "skeletonRendering": {
        "mode2d": "lines_and_points",
        "mode3d": "lines"
      },
      "name": "ground_truth"
    }
  ],
  "navigation": {
    "pose": {
      "position": {
        "voxelSize": [
          6,
          6,
          30
        ],
        "voxelCoordinates": [
          5523.99072265625,
          8538.9384765625,
          1198.0423583984375
        ]
      }
    },
    "zoomFactor": 22.573112129999547
  },
  "perspectiveOrientation": [
    -0.0040475670248270035,
    -0.9566215872764587,
    -0.22688281536102295,
    -0.18271005153656006
  ],
  "perspectiveZoom": 340.35867907175077,
  "layout": "4panel"
}
```
   Demo2: Janelia FlyEM FIB-25. 7-column Drosophila medulla (8x8x8 cubic nanometer resolution)  
```
{
  "layers": [
    {
      "source": "precomputed://gs://neuroglancer-public-data/flyem_fib-25/image",
      "type": "image",
      "name": "image"
    },
    {
      "source": "precomputed://gs://neuroglancer-public-data/flyem_fib-25/ground_truth",
      "type": "segmentation",
      "segments": [
        "158571",
        "21894",
        "22060",
        "24436",
        "2515"
      ],
      "skeletonRendering": {
        "mode2d": "lines_and_points",
        "mode3d": "lines"
      },
      "name": "ground-truth"
    }
  ],
  "navigation": {
    "pose": {
      "position": {
        "voxelSize": [
          8,
          8,
          8
        ],
        "voxelCoordinates": [
          2914.500732421875,
          3088.243408203125,
          4041
        ]
      }
    },
    "zoomFactor": 30.09748283999932
  },
  "perspectiveOrientation": [
    0.17737820744514465,
    0.8829324841499329,
    0.14149528741836548,
    -0.41103073954582214
  ],
  "perspectiveZoom": 443.63404517712684,
  "showSlices": false,
  "layout": "4panel"
}
```
   Demo3: FAFB: A Complete Electron Microscopy Volume of the Brain of Adult Drosophila melanogaster  
```
{
  "layers": [
    {
      "source": "precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_orig",
      "type": "image",
      "name": "fafb_v14"
    },
    {
      "source": "precomputed://gs://neuroglancer-fafb-data/fafb_v14/fafb_v14_clahe",
      "type": "image",
      "name": "fafb_v14_clahe",
      "visible": false
    },
    {
      "type": "segmentation",
      "mesh": "precomputed://gs://neuroglancer-fafb-data/elmr-data/FAFBNP.surf/mesh",
      "segments": [
        "1",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "2",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "3",
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
        "4",
        "40",
        "41",
        "42",
        "43",
        "44",
        "45",
        "46",
        "47",
        "48",
        "49",
        "5",
        "50",
        "51",
        "52",
        "53",
        "54",
        "55",
        "56",
        "57",
        "58",
        "59",
        "6",
        "60",
        "61",
        "62",
        "63",
        "64",
        "65",
        "66",
        "67",
        "68",
        "69",
        "7",
        "70",
        "71",
        "72",
        "73",
        "74",
        "75",
        "8",
        "9"
      ],
      "skeletonRendering": {
        "mode2d": "lines_and_points",
        "mode3d": "lines"
      },
      "name": "neuropil-regions-surface"
    },
    {
      "type": "mesh",
      "source": "vtk://https://storage.googleapis.com/neuroglancer-fafb-data/elmr-data/FAFB.surf.vtk.gz",
      "vertexAttributeSources": [],
      "shader": "void main() {\n  emitRGBA(vec4(1.0, 0.0, 0.0, 0.5));\n}\n",
      "name": "neuropil-full-surface",
      "visible": false
    }
  ],
  "navigation": {
    "pose": {
      "position": {
        "voxelSize": [
          4,
          4,  This dataset was copied from https://www.janelia.org/project-team/flyem/data-and-software-release, and is made available under the Open Data Common Attribution License. 

          40
        ],
        "voxelCoordinates": [
          123943.625,
          73323.8828125,
          5234
        ]
      }
    },
    "zoomFactor": 1210.991144617663
  },
  "perspectiveOrientation": [
    -0.28037887811660767,
    -0.19049881398677826,
    -0.13574382662773132,
    -0.9309519529342651
  ],
  "perspectiveZoom": 21335.91710335963,
  "layout": "xy-3d"
}
```



## Creating a dependent project

See [examples/dependent-project](examples/dependent-project).


## Referennce
1. https://github.com/google/ffn/  
2. https://github.com/nvm-sh/nvm  
3. https://github.com/google/neuroglancer  
4. https://www.temca2data.org/
5. https://jefferis.github.io/elmr/reference/FAFB.surf.html
6. https://www.janelia.org/project-team/flyem/data-and-software-release
7. https://neurodata.io/data/kasthuri15/
8. https://arxiv.org/abs/1611.00421
9. https://doi.org/10.1101/200675
10. Takemura, Shin-ya et al. "Synaptic Circuits and Their Variations within Different Columns in the Visual System of Drosophila." Proceedings of the National Academy of Sciences of the United States of America 112.44 (2015): 13711-13716.
11. Kasthuri, Narayanan, et al. "Saturated reconstruction of a volume of neocortex." Cell 162.3 (2015): 648-661.
