import os
import struct
import re

import argparse
from collections import namedtuple
from pathlib import Path
from typing import Any, List, NamedTuple

import numpy as np

from tinygrad.ops import Device
from tinygrad.nn.state import load_state_dict, torch_load, get_state_dict, safe_save, safe_load_metadata
from tinygrad.tensor import Tensor
from extra.export_model import jit_model, compile_net

from examples.stable_diffusion import (
  StableDiffusion,
  download_file,
)
from examples.controlnet import (
  ControlNetUNetModel, 
  ControlNetModel, 
  ControlNetStableDiffusion
)


class Step(NamedTuple):
  name: str = ""
  input: List[Tensor] = []
  forward: Any = None
  
def convert_f32_to_f16(input_file, output_file):
  with open(input_file, 'rb') as f:
    metadata_length_bytes = f.read(8)
    metadata_length = int.from_bytes(metadata_length_bytes, byteorder='little', signed=False)
    metadata_json_bytes = f.read(metadata_length)
    float32_values = np.fromfile(f, dtype=np.float32)

  float16_values = float32_values.astype(np.float16)

  with open(output_file, 'wb') as f:
    f.write(metadata_length_bytes)
    f.write(metadata_json_bytes)
    float16_values.tofile(f)
    
# look into render bundles

FILENAME_DIFFUSION = Path(__file__).parents[3] / "weights/sd-v1-4.ckpt"

def get_controlnet_url_file(variant):
  if variant == "canny":
    return ("https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/diffusion_pytorch_model.bin", 
            os.path.join(str(Path(__file__).parents[3]), "weights/sd-controlnet-canny.bin"))
  elif variant == "scribble":
    return ("https://huggingface.co/lllyasviel/sd-controlnet-scribble/resolve/main/diffusion_pytorch_model.bin", 
            os.path.join(str(Path(__file__).parents[3]), "weights/sd-controlnet-scribble.bin"))
  else:
    raise ValueError(f"Unknown variant: {variant}")
  
def replace_float_literals(kernel_code):
  # Define the regex pattern for floating point literals
  float_literal_pattern = r"""
      /0[fh]/ |                           # Match 0 followed by 'f' or 'h'
      /[1-9][0-9]*[fh]/ |                 # Match a non-zero digit followed by any digits and 'f' or 'h'
      /[0-9]*\.[0-9]+([eE][+-]?[0-9]+)?[fh]?/ |   # Match floating point literals with optional exponent
      /[0-9]+\.[0-9]*([eE][+-]?[0-9]+)?[fh]?/ |   # Match floating point literals with optional exponent
      /[0-9]+[eE][+-]?[0-9]+[fh]?/              # Match literals with exponent
  """

  # Replace 'f' with 'h' in the matched literals
  replaced_code = re.sub(r'(\d+(\.\d+)?(e[+-]?\d+)?)[f]', r'\1h', kernel_code, flags=re.IGNORECASE|re.VERBOSE)

  return replaced_code

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Compile ControlNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--remoteweights', action='store_true', help="Use safetensors from Huggingface, or from local")
  parser.add_argument('--weights_dir', type=str, default=os.path.dirname(__file__), help="Path to weights directory")
  parser.add_argument('--variant', type=str, default='canny', help="Variant of ControlNet to use")
  parser.add_argument('--dtype', type=str, default='f32', help="Data type to use for weights")
  args = parser.parse_args()
  Device.DEFAULT = "WEBGPU"
  
  Tensor.no_grad = True
  print(FILENAME_DIFFUSION)
  download_file(
  'https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt', FILENAME_DIFFUSION)
  state_dict_diffusion = torch_load(FILENAME_DIFFUSION)["state_dict"]
  diffusion_model = StableDiffusion()
  diffusion_model.model = namedtuple("DiffusionModel", ["diffusion_model"])(
  diffusion_model=ControlNetUNetModel())
  load_state_dict(diffusion_model, state_dict_diffusion, strict=False)
  
  controlnet_url, controlnet_filename = get_controlnet_url_file(args.variant)
  
  download_file(controlnet_url, controlnet_filename)
  state_dict_controlnet = torch_load(controlnet_filename)
  controlnet = ControlNetModel(cross_attention_dim=768)
  load_state_dict(controlnet, state_dict_controlnet, strict=False)

  model = ControlNetStableDiffusion(diffusion_model, controlnet)
  
  sub_steps = [
    Step(name = "textModel", 
       input = [Tensor.randn(1, 77)], 
       forward = model.cond_stage_model.transformer.text_model), 
    Step(name = "diffusor",
       input = [
            Tensor.randn(1, 3, 512, 512),
            Tensor.randn(1, 77, 768),
            Tensor.randn(1, 77, 768),
            Tensor.randn(1, 4, 64, 64),
            Tensor.rand(1), 
            Tensor.randn(1), 
            Tensor.randn(1), 
            Tensor.randn(1),
            Tensor.randn(1)
        ],
       forward = model),
    Step(name = "decoder", input = [Tensor.randn(1,4,64,64)], forward = model.decode)
  ]
  
  prg = ""
  
  def ensure_nonzero_state(state_dict):
    # for each k, v in state_dict, call v.numpy() until v is not all zeros.
    for k, v in state_dict.items():
      print("k: ", k)
      while np.all(v.numpy() == 0):
        print(f"Warning: {k} is all zeros.")
        print("Retrying...")
        v.numpy()
    return state_dict
  
  def compile_step(model, step: Step):
    run, special_names = jit_model(step, *step.input)
    functions, statements, bufs, _ = compile_net(run, special_names)
    
    state = get_state_dict(model)
    weights = {id(x.lazydata.realized): name for name, x in state.items()}
    
    
    if args.dtype == 'f16':
      kernel_code = '\n\n'.join([f"const {key} = `enanble f16;\n{code.replace(key, 'main')}`;" for key, code in functions.items()])
      kernel_code = kernel_code.replace('f32', 'f16')
      kernel_code = replace_float_literals(kernel_code)
    else:
      kernel_code = '\n\n'.join([f"const {key} = `{code.replace(key, 'main')}`;" for key, code in functions.items()])
      
    kernel_names = ', '.join([name for (name, _, _, _) in statements])
    
    # Creates bind groups and compute pass
    nskip = 2
    kernel_calls = [f"addComputePass(device, passEncoder, piplines[{i}], [{', '.join(args)}], {global_size});" for i, (_name, args, global_size, _local_size) in enumerate(statements) ]
    # every 3rd kernel call, we inject passEncoder.end() and passEncoder = commandEncoder.beginComputePass()
    kernel_calls = [f"passEncoder.end();\n    passEncoder = commandEncoder.beginComputePass();\n    {kernel_call}" if (i % nskip == 0 and i > 0 and i < len(kernel_calls)-1) else kernel_call for i , kernel_call in enumerate(kernel_calls)]
    # add one last end() call
    kernel_calls.append("passEncoder.end();")
    kernel_calls = '\n    '.join(kernel_calls)
    
    # Allocate GPU IO buffers
    bufs =  '\n  '.join([f"const {name} = " + (f"createEmptyBuf(device, {size});" if _key not in weights else f"await createWeightBuf(device, {size}, filename, metadata['{weights[_key]}'], metadataLength)") + ";"  for name,(size,dtype,_key) in bufs.items()])
    
    # Allocate Stage buffer for input
    gpu_write_bufs =  '\n  '.join([f"const gpuWriteBuffer{i} = device.createBuffer({{size:input{i}.size+input{i}.size%4, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE }});" for i,(_,value) in enumerate(special_names.items()) if "output" not in value])
    
    if args.dtype == 'f16':
      input_writer = '\n  '.join([f"await gpuWriteBuffer{i}.mapAsync(GPUMapMode.WRITE);\n  new Uint16Array(gpuWriteBuffer{i}.getMappedRange()).set(" + f'data{i});' + f"\n  gpuWriteBuffer{i}.unmap();\ncommandEncoder.copyBufferToBuffer(gpuWriteBuffer{i}, 0, input{i}, 0, gpuWriteBuffer{i}.size);"  for i,(_,value) in enumerate(special_names.items()) if value != "output0"])
      result_writer = """const resultBuffer = new Uint16Array(gpuReadBuffer.size/2);
        resultBuffer.set(new Uint16Array(gpuReadBuffer.getMappedRange()));"""
    else:
      input_writer = '\n  '.join([f"await gpuWriteBuffer{i}.mapAsync(GPUMapMode.WRITE);\n  new Float32Array(gpuWriteBuffer{i}.getMappedRange()).set(" + f'data{i});' + f"\n  gpuWriteBuffer{i}.unmap();\ncommandEncoder.copyBufferToBuffer(gpuWriteBuffer{i}, 0, input{i}, 0, gpuWriteBuffer{i}.size);"  for i,(_,value) in enumerate(special_names.items()) if value != "output0"])
      result_writer = """const resultBuffer = new Float32Array(gpuReadBuffer.size/4);
        resultBuffer.set(new Float32Array(gpuReadBuffer.getMappedRange()));"""
    
    return f"""\n  var {step.name} = function() {{
  
    {kernel_code}

    return {{
    "setup": async (device, filename, metadata, metadataLength) => {{
      {bufs}
      
      {gpu_write_bufs}
      const gpuReadBuffer = device.createBuffer({{ size: output0.size+output0.size%4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }});

      const kernels = [{kernel_names}];
      const piplines = await Promise.all(kernels.map(name => device.createComputePipelineAsync({{layout: "auto", compute: {{ module: device.createShaderModule({{ code: name }}), entryPoint: "main" }}}})));

      return async ({",".join([f'data{i}' for i,(k,v) in enumerate(special_names.items()) if v != "output0"])}) => {{
        const commandEncoder = device.createCommandEncoder();

        {input_writer}
        
        var passEncoder = commandEncoder.beginComputePass();

        {kernel_calls}
        commandEncoder.copyBufferToBuffer(output0, 0, gpuReadBuffer, 0, output0.size);
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        await gpuReadBuffer.mapAsync(GPUMapMode.READ);
        {result_writer}
        gpuReadBuffer.unmap();
        return resultBuffer;
      }}
    }} 
    }}
  }}
  """

  base_url = "."
  if args.remoteweights:
    base_url = "https://huggingface.co/jchun/tinygrad-sd-controlnet-f16/resolve/main/"
  if not os.path.exists(args.weights_dir):
    os.makedirs(args.weights_dir)
    
  fp32_offsets = []
    
  for step in sub_steps:
    prg += compile_step(model, step)
    if step.name == "diffusor":
      state = get_state_dict(model)
      ensure_nonzero_state(state)
      f32_dir = os.path.join(args.weights_dir, "net_{}.safetensors".format(args.variant))
      safe_save(state, f32_dir)
      if args.dtype == "f16":
        f16_dir = os.path.join(args.weights_dir, "net_f16_{}.safetensors".format(args.variant))
        if not os.path.exists(f16_dir):
          convert_f32_to_f16(f32_dir, f16_dir)
          
  if args.dtype == "f16":
    # patch metatdataoffsets for f16 weights
    readWeightsFromFileFunction = """const readWeightsFromFile = async (filename, tensorMetaData, metadataLength) => {{
      const dataOffsets = tensorMetaData.data_offsets;
      const data = await readRangeFromFile(
        filename, 
        dataOffsets[0]/2 + 8 + metadataLength,
        dataOffsets[1]/2 + 8 + metadataLength
      );
      return data;
    }};"""
  else:
    readWeightsFromFileFunction = """const readWeightsFromFile = async (filename, tensorMetaData, metadataLength) => {{
      const dataOffsets = tensorMetaData.data_offsets;
      const data = await readRangeFromFile(
        filename, 
        dataOffsets[0] + 8 + metadataLength,
        dataOffsets[1] + 8 + metadataLength
      );
      return data;
    }};"""
    
  prekernel = f"""
    window.MODEL_BASE_URL= "{base_url}";
    const getTensorMetadata = (safetensorBuffer) => {{
      const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
      const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
      return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {{...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}}]));
    }};

  const createEmptyBuf = (device, size) => {{
      return device.createBuffer({{size: size+size%4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }});
  }};
  
  const readRangeFromFile = async (file, start, end) => {{
    // read bytes from start to end from file, exclusive of end
    console.log("Fetching safetensors from: " + file + " start: " + start + " end: " + end);

    try {{
        const response = await fetch(file, {{
            headers: {{
                "Range": 'bytes='.concat(start, '-', end-1)
            }}
        }});
        if (!response.ok && response.status !== 206) {{
            throw new Error('Unexpected response status ${{response.status}}');
        }}
        const data = await response.arrayBuffer();
        const result = new Uint8Array(data);

        // Return or process the result as needed
        return result;
    }}
    catch (e) {{
        console.log("Error fetching safetensors: " + e);
    }}
    return [];
  }};
  
  {readWeightsFromFileFunction}

  const createWeightBuf = async (device, size, filename, tensorMetaData, metadataLength) => {{
    
    let data = await readWeightsFromFile(filename, tensorMetaData, metadataLength);

    const buf = device.createBuffer({{ mappedAtCreation: true, size: size+size%4, usage: GPUBufferUsage.STORAGE }});
    new Uint8Array(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
  }};

  const addComputePass = (device, passEncoder, pipeline, bufs, workgroup) => {{
    const bindGroup = device.createBindGroup({{layout: pipeline.getBindGroupLayout(0), entries: bufs.map((buffer, index) => ({{ binding: index, resource: {{ buffer }} }}))}});
    // const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...workgroup);
    // passEncoder.end();
  }};"""

  output_js = f"net_{args.dtype}.js"
  with open(os.path.join(os.path.dirname(__file__), output_js), "w") as text_file:
    text_file.write(prekernel + prg)
