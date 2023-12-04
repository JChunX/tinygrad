# control.net

![](./img/cn_canny_in_out.png)

![](./img/cn_scribble_in_out.png)

By Zhiyu Lei, Gene Liu, and Jason Xie

This is an implementation of [ControlNet](https://github.com/lllyasviel/ControlNet) in the browser powered by WebGPU. ControlNet is an extension of stable diffusion models where an additional user constraint is used to influence the final result. The original ControlNet introduces model weights for a variety of user constraint forms, such as through Canny edge maps, depth/normal maps, human poses, or even scribbles.

This project makes use of the tinygrad deep learning framework and its machine learning compilation capabilities to compile a custom tinygrad ControlNet implementation to WebGPU compute shaders. A simple frontend enables users to input prompts and constraints to access the constructed compute pipelines.

Note that the generated compute shaders are not necessarily optimized. The implementation here demonstrates various compiler and shader level optimizations that improve model performance on the GPU.

## WebGPU Model Porting

To construct a functional ControlNet model using WebGPU, the following general steps are taken:

### tinygrad Implementation and MLC

The original ControlNet implementation is written in PyTorch. A corresponding tinygrad implementation is written to make use of tinygrad's MLC capabilities.

Machine learning compilation (MLC) allows for machine learning models written in higher level languages such as Python to be converted to equivalent representations in other languages and frameworks. In this case, we use the abstract syntax tree of a tinygrad model and convert each component into a separate kernel. The kernels are combined to form full modules through WebGPU compute pipelines.

![](./img/tg_mlc.png)

### WebGPU Model Weight Loading

The above step produces a set of WebGPU compute pipelines, with each compute shader requiring the matching weight buffer of the model to be loaded as input. These weight buffers are extracted from huggingface safe tensor files. As these weights are quite large(~5.5 GB) and exceed the v8 JavaScript engine heap size limit, the weights are directly loaded into GPU buffers in chunks.

### Frontend

A simple frontend allows for useres to interact with the ControlNet model. [OpenCV.js](https://docs.opencv.org/3.4/d5/d10/tutorial_js_root.html) is used to process user input images and convert them to the appropriate format for the model.

## Optimizations

TBD

## Performance Analysis

TBD

## Usage

TBD

## References

[ControlNet](https://github.com/lllyasviel/ControlNet)

[Web LLM](https://github.com/mlc-ai/web-llm)

[Web Stable Diffusion](https://github.com/mlc-ai/web-stable-diffusion)
