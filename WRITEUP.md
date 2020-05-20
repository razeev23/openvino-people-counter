# Project Write-Up

# What are Custom Model
OpenVINO Toolkit Documentations has a list of Supported Framework Layers for DL Inference. Custom layers are layers that are not included in the list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom

## To convert a Custom Layer (Tensorflow example)
### Step 1 - Create the Custom Layer
**Generate the Extension Template Using the Model Extension Generator**
Model Extension Generator tool will automatically create templates for all the extensions needed by the Model Optimizer to convert and the Inference Engine to execute the custom layer. The extension template files will be partially replaced by Python and C++ code to implement the functionality of your custom layer as needed by the different tools. To create the four extensions for the custom layer, we run the Model Extension Generator with the following options:

- ```--mo-tf-ext``` = Generate a template for a Model Optimizer Tensorflow extractor
- ```--mo-op``` = Generate a template for a Model Optimizer custom layer operation
- ```--ie-cpu-ext``` = Generate a template for an Inference Engine CPU extension
- ```--ie-gpu-ext```` = Generate a template for an Inference Engine GPU extension
- ```--output-dir ``` = Set the output directory. Here we are using your/directory/cl_CustomLayer as the target directory to store the output from the Model Extension Generator.

To Create the four extension templates for the custom layer, run the command
``` python /opt/intel/openvino/deployment_tools/tools/extension_generator/extgen.py new --mo-tf-ext --mo-op --ie-cpu-ext --ie-gpu-ext --output_dir = your/directory   ```

The Model Extension Generator will start in interactive mode and prompt us with questions about the custom layer to be generated. Use the text between the ``[]'s`` to answer each of the Model Extension Generator questions as follows:
``
Enter layer name: 
[layer_name]

Do you want to automatically parse all parameters from the model file? (y/n)
...
[n]

Enter all parameters in the following format:
...
Enter 'q' when finished:
[q]

Do you want to change any answer (y/n) ? Default 'no'
[n]

Do you want to use the layer name as the operation name? (y/n)
[y]

Does your operation change shape? (y/n)  
[n]

Do you want to change any answer (y/n) ? Default 'no'
[n]When complete, the output text will appear similar to:
  
``
When complete, the output text will appear similar to:
``
Stub file for TensorFlow Model Optimizer extractor is in your/directory/user_mo_extensions/front/tf folder
Stub file for the Model Optimizer operation is in your/directory/user_mo_extensions/ops folder
Stub files for the Inference Engine CPU extension are in your/directory/user_ie_extensions/cpu folder
Stub files for the Inference Engine GPU extension are in your/directory/user_ie_extensions/gpu folder
``

Template files (containing source code stubs) that may need to be edited have just been created in the following locations:
- TensorFlow Model Optimizer extractor extension:
  -``your/directory/user_mo_extensions/front/tf/``
  -``CustomLayer_ext.py ``
- Model Optimizer operation extension:
  -``your/directory/user_mo_extensions/ops``
  -``CustomLayer.py``
- Inference Engine CPU extension:
  -``your/directory/user_ie_extensions/cpu``
  -``ext_CustomLayer.cpp``
  -``CMakeLists.txt``
- Inference Engine GPU extension
  -``your/directory/user_ie_extensions/gpu``
  -``CustomLayer_kernel.cl``
  -``CustomLayer_kernel.xml``
  
## Step 2 - Using Model Optimizer to Generate IR Files Containing the Custom Layer
Now use the generated extractor and operation extensions with the Model Optimizer to generate the model IR files needed by the Inference Engine. The steps covered are:
1. Edit the extractor extension template file
2. Edit the operation extension template file
3. Generate the Model IR Files

**Edit the Extractor Extension Template File**
Below is a walkthrough of the Python code for the extractor extension that appears in the file
``your/directory/user_mo_extensions/front/tf/CustomLayer_ext.py ``
 1. Using the text editor, open the extractor extension source file
  ``your/directory/user_mo_extensions/front/tf/CustomLayer_ext.py``.
 2. The class is defined with the unique name CustomLayerFrontExtractor that inherits from the base extractor FrontExtractorOp class.       The class variable op is set to the name of the layer operation and enabled is set to tell the Model Optimizer to use (True) or         exclude (False) the layer during processing.
    ```class CustomLayerFrontExtractor(FrontExtractorOp):
       op = 'CustomLayer'
       enabled = True
      ```
 3.The extract function is overridden to allow modifications while extracting parameters from layers within the input model.
    ```
      @staticmethod
      def extract(node):
      ```
 4.The layer parameters are extracted from the input model and stored in param. This is where the layer parameters in param may be          retrieved and used as needed. For a simple custom layer, the op attribute is simply set to the name of the operation extension used.
    ```
    proto_layer = node.pb
      param = proto_layer.attr
    # extracting parameters from TensorFlow layer and prepare them for IR
      attrs = {
    'op': __class__.op
      }   
     ```
      
 5. The attributes for the specific node are updated. This is where we can modify or create attributes in attrs before updating node         with the results and the enabled class variable is returned.
    ``` 
      # update the attributes of the node
      Op.get_op_class_by_name(__class__.op).update_node_stat(node, attrs)

      return __class__.enabled
      
      ```
## Step 3 - Edit the Operation Extension Template File
If the shape (i.e., dimensions) of the layer output is the same as the input shape, the generated operation extension does not need to be modified.
Below is a walkthrough of the Python code for the operation extension that appears in the file
```
your/directory/user_mo_extensions/ops/CustomLayer.py
```
1.Using the text editor, open the operation extension source file ``your/directory/user_mo_extensions/ops/CustomLayer.py``
2.The class is defined with the unique name CustomLayerOp that inherits from the base operation Op class. The class variable op is set     to 'CustomLayer', the name of the layer operation
  ```
  class CustomLayerOp(Op):
   op = 'CustomLayer'
   ```
3.The CustomLayerOp class initializer __init__ function will be called for each layer created. The initializer must initialize the super   class Op by passing the graph and attrs arguments along with a dictionary of the mandatory properties for the CustomLayer operation     layer that define the type (type), operation (op), and inference function (infer). This is where any other initialization needed by     the CustomLayerOP operation can be specified.
```
 def __init__(self, graph, attrs):
    mandatory_props = dict(
        type=__class__.op,
        op=__class__.op,
        infer=CustomLayerOp.infer            
    )
    super().__init__(graph, mandatory_props, attrs)

```  

    
4.The infer function is defined to provide the Model Optimizer information on a layer, specifically returning the shape of the layer       output for each node. Here, the layer output shape is the same as the input and the value of the helper function                         copy_shape_infer(node) is returned.
``` 
 @staticmethod
    def infer(node: Node):
    # ==========================================================
    # You should add your shape calculation implementation here
    # If a layer input shape is different to the output one
    # it means that it changes shape and you need to implement
    # it on your own. Otherwise, use copy_shape_infer(node).
    # ==========================================================
    return copy_shape_infer(node)
```
    
## Step 4 - Generate the Model IR Files
With the extensions now complete, we use the Model Optimizer to convert and optimize the example TensorFlow model into IR files that will run inference using the Inference Engine.
To create the IR files, we run the Model Optimizer for TensorFlow mo_tf.py with the following options:
- ``--input_meta_graph model.ckpt.meta ``
  -Specifies the model input file.
- ``--batch 1``
  -Explicitly sets the batch size to 1 because the example model has an input dimension of "-1"
  -TensorFlow allows "-1" as a variable indicating "to be filled in later", however the Model Optimizer requires explicit information      for the optimization process.
- ``--output "ModCustomLayer/Activation_8/softmax_output"``
  -``The full name of the final output layer of the model.``
- ``--extensions your/director/user_mo_extensions``
  -``Location of the extractor and operation extensions for the custom layer to be used by the Model Optimizer during model extraction and optimization``
- ``--output_dir your/directory/cl_ext_CustomLayer``
 -``Location to write the output IR files.``

To create the model IR files that will include the CustomLayer custom layer, we run the commands:
 ``
 cd your/directory/tf_model
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_meta_graph model.ckpt.meta --batch 1 --output "ModCustomLayer/Activation_8/softmax_output" --extensions your/directory/cl_CustomLayer/user_mo_extensions --output_dir your/directory/cl_ext_CustomLayer
``
The output will appear similar to:

```
[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: your/directory/cl_ext_CustomLayer/model.ckpt.xml
[ SUCCESS ] BIN file: your/directory/cl_ext_CustomLayer/model.ckpt.bin
[ SUCCESS ] Total execution time: x.xx seconds.
```
## Inference Engine Custom Layer Implementation for the Intel® CPU
We will now use the generated CPU extension with the Inference Engine to execute the custom layer on the CPU. The steps are:

Edit the CPU extension template files.
Compile the CPU extension library.
Execute the Model with the custom layer.
You will need to make the changes in this section to the related files.

Note that the classroom workspace only has an Intel CPU available, so we will not perform the necessary steps for GPU usage with the Inference Engine.

## Edit the CPU Extension Template Files
The generated CPU extension includes the template file ext_CustomLayer.cpp that must be edited to fill-in the functionality of the CustomLayer custom layer for execution by the Inference Engine.
We also need to edit the CMakeLists.txt file to add any header file or library dependencies required to compile the CPU extension. In the next sections, we will walk through and edit these files.
**Edit ext_CustomLayer.cpp**
Now edit the ext_CustomLayer.cpp by walking through the code and making the necessary changes for the CustomLayer custom layer along the way.
1. Using the text editor, open the CPU extension source file ``your/directory/cl_CustomLayer/user_ie_extensions/cpu/ext_CustomLayer.cpp``.

2.To implement the CustomLayer function to efficiently execute in parallel, the code will use the parallel processing supported by the Inference Engine through the use of the Intel® Threading Building Blocks library. To use the library, at the top we must include the header ie_parallel.hpp file by adding the #include line as shown below.

Before:
```
#include "ext_base.hpp"
#include <cmath>
```

After:
```
#include "ext_base.hpp"
#include "ie_parallel.hpp"
#include <cmath>
```
3. The class CustomLayerImp implements the CustomLayer custom layer and inherits from the extension layer base class ExtLayerBase.
```
class CustomLayerImpl: public ExtLayerBase {
    public:
```
4. The CustomLayerImpl constructor is passed the layer object that it is associated with to provide access to any layer parameters that    may be needed when implementing the specific instance of the custom layer.
 ```
 explicit CustomLayerImpl(const CNNLayer* layer) {
  try {
    ...
   
  ```
5. The CustomLayerImpl constructor configures the input and output data layout for the custom layer by calling addConfig(). In the          template file, the line is commented-out and we will replace it to indicate that layer uses DataConfigurator(ConfLayout::PLN) (plain    or linear) data for both input and output.

Before:

 ```
  
// addConfig({DataConfigurator(ConfLayout::PLN), DataConfigurator(ConfLayout::PLN)}, {DataConfigurator(ConfLayout::PLN)});
```
After:

```
addConfig(layer, { DataConfigurator(ConfLayout::PLN) }, { DataConfigurator(ConfLayout::PLN) });
```


6.The construct is now complete, catching and reporting certain exceptions that may have been thrown before exiting.
 ```
 } catch (InferenceEngine::details::InferenceEngineException &ex) {
    errorMsg = ex.what();
  }
 }
 ```
7. The execute method is overridden to implement the functionality of the custom layer. The inputs and outputs are the data buffers passed as Blob objects. The template file will simply return NOT_IMPLEMENTED by default. To calculate the custom layer, we will replace the execute method with the code needed to calculate the CustomLayer function in parallel using the parallel_for3d function.

Before:

```
StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
    ResponseDesc *resp) noexcept override {
    // Add here implementation for layer inference
    // Examples of implementations you can find in Inference Engine tool samples/extensions folder
    return NOT_IMPLEMENTED;
   ```
 After:
 
 ``` 
 StatusCode execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
    ResponseDesc *resp) noexcept override {
    // Add implementation for layer inference here
    // Examples of implementations are in OpenVINO samples/extensions folder

    // Get pointers to source and destination buffers
    float* src_data = inputs[0]->buffer();
    float* dst_data = outputs[0]->buffer();

    // Get the dimensions from the input (output dimensions are the same)
    SizeVector dims = inputs[0]->getTensorDesc().getDims();

    // Get dimensions:N=Batch size, C=Number of Channels, H=Height, W=Width
    int N = static_cast<int>((dims.size() > 0) ? dims[0] : 1);
    int C = static_cast<int>((dims.size() > 1) ? dims[1] : 1);
    int H = static_cast<int>((dims.size() > 2) ? dims[2] : 1);
    int W = static_cast<int>((dims.size() > 3) ? dims[3] : 1);

    // Perform (in parallel) the hyperbolic cosine given by: 
    //    CustomLayer(x) = (e^x + e^-x)/2
    parallel_for3d(N, C, H, [&](int b, int c, int h) {
    // Fill output_sequences with -1
    for (size_t ii = 0; ii < b*c; ii++) {
      dst_data[ii] = (exp(src_data[ii]) + exp(-src_data[ii]))/2;
    }
  });
return OK;
}
```


**Edit CMakeLists.txt**
Because the implementation of the CustomLayer custom layer makes use of the parallel processing supported by the Inference Engine, we need to add the Intel® Threading Building Blocks dependency to CMakeLists.txt before compiling. We will add paths to the header and library files and add the Intel® Threading Building Blocks library to the list of link libraries. We will also rename the .so.

1.Using the text editor, open the CPU extension CMake file ``your/directory/cl_CustomLayer/user_ie_extensions/cpu/CMakeLists.txt.``

2.At the top, rename the TARGET_NAME so that the compiled library is named libCustomLayer_cpu_extension.so:

Before:

```
set(TARGET_NAME "user_cpu_extension")
```

After:

```
set(TARGET_NAME "CustomLayer_cpu_extension")
```
3.Now modify the include_directories to add the header include path for the Intel® Threading Building Blocks library located in ``/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/include:``

Before:

```
include_directories (PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/common
${InferenceEngine_INCLUDE_DIRS}
)
```

After:

```
include_directories (PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/common
${InferenceEngine_INCLUDE_DIRS}
"/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/include"
)
```

4.Now add the link_directories with the path to the Intel® Threading Building Blocks library binaries at /opt/intel/openvino/deployment_tools/inference_engine/external/tbb/lib:

Before:
```
...
#enable_omp()
```

After:
```
...
link_directories(
"/opt/intel/openvino/deployment_tools/inference_engine/external/tbb/lib"
)
#enable_omp()
```
5.Finally, add the Intel® Threading Building Blocks library tbb to the list of link libraries in target_link_libraries:

Before:
```
target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${intel_omp_lib})
```
After:
```
target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${intel_omp_lib} tbb)
```

**Compile the Extension Library**
To run the custom layer on the CPU during inference, the edited extension C++ source code must be compiled to create a .so shared library used by the Inference Engine. In the following steps, we will now compile the extension C++ library.

1.First, we run the following commands to use CMake to setup for compiling:

```
cd your/directory/cl_CustomLayer/user_ie_extensions/cpu
mkdir -p build
cd build
cmake ..
```
The output will appear similar to:
```
-- Generating done
-- Build files have been written to: your/directory/cl_tutorial/cl_CustomLayer/user_ie_extensions/cpu/build
```
2.The CPU extension library is now ready to be compiled. Compile the library using the command:
```
make -j $(nproc)
```
The output will appear similar to:
```
[100%] Linking CXX shared library libCustomLayer_cpu_extension.so
[100%] Built target CustomLayer_cpu_extension
```

## Execute the Model with the Custom Layer
Using a C++ Sample
To start on a C++ sample, we first need to build the C++ samples for use with the Inference Engine:
```
cd /opt/intel/openvino/deployment_tools/inference_engine/samples/
./build_samples.sh ``

This will take a few minutes to compile all of the samples.

Next, we will try running the C++ sample without including the CustomLayer extension library to see the error describing the unsupported CustomLayer operation using the command:
``~/inference_engine_samples_build/intel64/Release/classification_sample_async -i pic.bmp -m your/directory/cl_ext_CustomLayer/model.ckpt.xml -d CPU
```

The error output will be similar to:

```
[ ERROR ] Unsupported primitive of type: CustomLayer name: ModCustomLayer/CustomLayer/CustomLayer
```
We will now run the command again, this time with the CustomLayer extension library specified using the -l your/directory/cl_CustomLayer/user_ie_extensions/cpu/build/libCustomLayer_cpu_extension.so option in the command:
```
~/inference_engine_samples_build/intel64/Release/classification_sample_async -i pic.bmp -m your/directory/cl_ext_CustomL
```

The output will appear similar to:
```
Image /directory/path/pic.bmp

|classid| probability|
|-------|------------|
|0      | 0.9308984  |  
|1      | 0.0691015  |

total inference time: xx.xxxxxxx
Average running time of one iteration: xx.xxxxxxx ms

Throughput: xx.xxxxxxx FPS

[ INFO ] Execution successful
```
 **Using a Python Sample**
 First, we will try running the Python sample without including the CustomLayer extension library to see the error describing the unsupported CustomLayer operation using the command:
 ``
 python /opt/intel/openvino/deployment_tools/inference_engine/samples/python_samples/classification_sample_async/classification_sample_async.py -i pic.bmp -m your/directory/cl_ext_CustomLayer/model.ckpt.xml -d CPU 
 ``
 
 The error output will be similar to:
 
 ```
 [ INFO ] Loading network files:
your/directory/cl_tutorial/tf_model/model.ckpt.xml
your/directory/cl_tutorial/tf_model/model.ckpt.bin
[ ERROR ] Following layers are not supported by the plugin for specified device CPU:
ModCustomLayer/CustomLayer/CustomLayer, ModCustomLayer/CustomLayer_1/CustomLayer, ModCustomLayer/CustomLayer_2/CustomLayer
[ ERROR ] Please try to specify cpu extensions library path in sample's command line parameters using -l or --cpu_extension command line argument
```
We will now run the command again, this time with the CustomLayer extension library specified using the -l your/directory/cl_CustomLayer/user_ie_extensions/cpu/build/libCustomLayer_cpu_extension.so option in the command:
```
python /opt/intel/openvino/deployment_tools/inference_engine/samples/python_samples/classification_sample_async/classification_sample_async.py -i pic.bmp -m your/directory/cl_ext_CustomLayer/model.ckpt.xml -l your/directory/cl_CustomLayer/user_ie_extensions/cpu/build/libCustomLayer_cpu_extension.so -d CPU
```
The output will appear similar to:
```
Image your/directory/cl_tutorial/OpenVINO-Custom-Layers/pics/dog.bmp

|classid| probability|
|-------|----------- |
|0      |0.9308984   |
|1      |0.0691015   |
```
## Why you might need to handle custom layers?
Some of the potential reasons for handling custom layers are:

-You might want to run some experimental layer on top of what already exists in the list of supported layer.
-The Layers you're trying to run uses unsupported input/ output shapes or formats.
-You're trying to run a framework out of the support frameworks like Tensorflow, ONNX, Caffe.

## Explaining Model Selection
There are lots ot pre-trained detection model availaible on different dataset such as e COCO dataset, the Kitti dataset, the Open Images dataset, the AVA v2.1 dataset and the iNaturalist Species Detection Dataset on the **tensorflow detec
Template files (containing source code stubs) that may need to be edited have just been created in the following locations:
- TensorFlow Model Optimizer extractor extension:
  -``your/directory/user_mo_extensions/front/tf/``
  -``CustomLayer_ext.py ``
- Model Optimizer operation extension:
  -``CustomLayer.py``
  -``CustomLayer.pytion model zoo**. how i have tried the three different model they are 
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


Comparison
Comparing the two models i.e. ssd_inception_v2_coco and faster_rcnn_inception_v2_coco in terms of latency and memory, several insights were drawn. It could be clearly seen that the Latency (microseconds) and Memory (Mb) decreases in case of OpenVINO as compared to plain Tensorflow model which is very useful in case of OpenVINO applications.

|Model/Framework	|Latency (microseconds) |	Memory (Mb)|
------------------|-----------------------|------------|
|ssd_inception_v2_coco (plain TF)	|222|	538|
|ssd_inception_v2_coco (OpenVINO)	|155	|329|
|faster_rcnn_inception_v2_coco (plain TF)	|1281	|562|
|faster_rcnn_inception_v2_coco (OpenVINO)	|889	|281|
**Differences in Edge and Cloud computing**
Edge Computing is regarded as ideal for operations with extreme latency concerns. Thus, medium scale companies that have budget limitations can use edge computing to save financial resources. Cloud Computing is more suitable for projects and organizations which deal with massive data storage


## Model Use Cases
In the present situation (COVID-19 pandemic )we can use this model to check how many people are in the frame ,if more than one found then alert can be generated.

## Effects on End user needs
Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:

Lighting - Perhaps no other aspects of a computer vision model has consistently caused delays, false detections and detection failures than lighting. In an ideal scenario, lighting would maximize the contrast on features of interest, and would in turn make it easier for a model to detect separate instances of object, in our case person. Since most of the use cases of a people counter application rely on a static CCTV camera, it is critical to have a proper lighting in the area it aims to cover or it may cause false detections or no-detection at all.
Model Accuracy - The model needs to be highly accurate if deployed in a mass scale as it may cause false detections or no-detection which can produce misleading data, and in case of retail applications may cause the company or business to lose money.
Image Size/Focal Length - It is critical to have a sharp and high resolution image as an input to our model to make it easy for it to perform segmentation easily and keep the features of interest detectable.

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

