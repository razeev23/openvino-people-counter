# Project Write-Up


## Explaining Model Selection & Custom Layers
There are lots ot pre-trained detection model availaible on different dataset such as e COCO dataset, the Kitti dataset, the Open Images dataset, the AVA v2.1 dataset and the iNaturalist Species Detection Dataset on the **tensorflow detection model zoo**. how i have tried the three different model they are 
- ssd_mobilenet_v2_coco
- faster_rcnn_inception_v2_coco
- Ssd_inception_v2_coco 
in which _faster_rcnn_inception_v2_coco_ gave the best result in terms of fast detection and errors.Intel openVINO already contains extensions for custom layers used in TensorFlow Object Detection Model Zoo.

## Downloading the faster rcnn model of TensorFlow Object Detection Modal Zoo
```
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```
## extracting the faster rcnn model
```
tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```
### navigate to the extracted model by the following command
```
cd faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```
## convert the TensorFlow model to Intermediate Representation (IR) or OpenVINO IR format with issuing the following command
```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
```
## comparing the model performance 
### Model - 1 ssd_mobilenet_v2_coco
Among the three model this model gaves the bad result in terms of detection and error while issuing the following command to convert in to the IR format.
```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_mobilenet_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```
### Model - 2 ssd_inception_v2_coco
Compare to the ssd_mobilenet_v2_coco it gives the good result but not the desirable result in terms of detection and errors by issuing the following command to convert it into the IR Representation.
```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```
### Model - 3 faster_rcnn_inception_v2_coco
Among the three this model gave the best result which is acceptable with its performance in terms of faste detection and error while issuing the following command to convert it into the IR Representation.
```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```
## comparing the model in terms of size

|model name                                                     | size     |
|---------------------------------------------------------------|----------|
|sss_mobilenet_v2_coo frozen_inference_graph.pb                 |68055KB   |
|sss_mobilenet_v2_coo frozen_inference_graph.(xml+bin)          |65807KB   |
|------------------------------------------------------------------------  |
|ssd_inception_v2_coco frozen_inference_graph.pb                |99592KB   |
|ssd_inception_v2_coco frozen_inference_graph.(xml+bin)         |97875KB   |
|--------------------------------------------------------------------------|
|faster_rcnn_inception_v2_coco frozen_inference_graph.pb        |55815KB   |    
|faster_rcnn_inception_v2_coco frozen_inference_graph.(xml+bin) |52106KB   |


## Model Use Cases
In the present situation (COVID-19 pandemic )we can use this model to check how many people are in the frame ,if more than one found then alert can be generated.

## Effects on End user needs
Various insights could be drawn on the model by testing it with different videos and analyzing the model performance on low light input videos. This would be an important factor in determining the best model for the given scenario.

## Run the application

From the main directory:

### Step 1 - Start the Mosca server

```
cd webservice/server/node-server
node ./server.js
```

You should see the following message, if successful:
```
Mosca server started.
```

### Step 2 - Start the GUI

Open new terminal and run below commands.
```
cd webservice/ui
npm run dev
```

You should see the following message in the terminal.
```
webpack: Compiled successfully
```

### Step 3 - FFmpeg Server

Open new terminal and run the below commands.
```
sudo ffserver -f ./ffmpeg/server.conf
```

### Step 4 - Run the code

Open a new terminal to run the code. 

#### Setup the environment

You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```

You should also be able to run the application with Python 3.6, although newer versions of Python will not work with the app.

#### Running on the CPU

When running Intel® Distribution of OpenVINO™ toolkit Python applications on the CPU, the CPU extension library is required. This can be found at: 

```
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/
```

*Depending on whether you are using Linux or Mac, the filename will be either `libcpu_extension_sse4.so` or `libcpu_extension.dylib`, respectively.* (The Linux filename may be different if you are using a AVX architecture)

Though by default application runs on CPU, this can also be explicitly specified by ```-d CPU``` command-line argument:

```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```
If you are in the classroom workspace, use the “Open App” button to view the output. If working locally, to see the output on a web based interface, open the link [http://0.0.0.0:3004](http://0.0.0.0:3004/) in a browser.

