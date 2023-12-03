import os
import struct

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

def split_safetensor(fn):
  _, json_len, metadata = safe_load_metadata(fn)
  # text_model_offset = 3772703308
  chunk_size = 536870912

  for k in metadata:
    # safetensor is in fp16
    # if (metadata[k]["data_offsets"][0] < text_model_offset):
    metadata[k]["data_offsets"][0] = int(metadata[k]["data_offsets"][0]/2)
    metadata[k]["data_offsets"][1] = int(metadata[k]["data_offsets"][1]/2)
  
  last_offset = 0
  part_end_offsets = []

  for k in metadata:
    offset = metadata[k]['data_offsets'][0]

    # if offset == text_model_offset:
    #   break

    part_offset = offset - last_offset
    
    if (part_offset >= chunk_size):
      part_end_offsets.append(8+json_len+offset)
      last_offset = offset

  # text_model_start = int(text_model_offset/2)
  net_bytes = bytes(open(fn, 'rb').read())
  # part_end_offsets.append(text_model_start+8+json_len)
  cur_pos = 0
  
  for i, end_pos in enumerate(part_end_offsets):
    with open(f'./net_part{i}.safetensors', "wb+") as f:
      if i < len(part_end_offsets)-1:
        f.write(net_bytes[cur_pos:end_pos])
      else:
        f.write(net_bytes[cur_pos:])
      cur_pos = end_pos

  # with open(f'./net_textmodel.safetensors', "wb+") as f:
  #   f.write(net_bytes[text_model_start+8+json_len:])
  
  return part_end_offsets

# look into render bundles

FILENAME_DIFFUSION = Path(__file__).parents[3] / "weights/sd-v1-4.ckpt"

def get_controlnet_url_file(variant):
  if variant == "canny":
    return ("https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/diffusion_pytorch_model.bin", 
            Path(__file__).parents[3] + "/sd-controlnet-canny.bin")
  elif variant == "scribble":
    return ("https://huggingface.co/lllyasviel/sd-controlnet-scribble/resolve/main/diffusion_pytorch_model.bin", 
            Path(__file__).parents[3] + "/sd-controlnet-scribble.bin")
  else:
    raise ValueError(f"Unknown variant: {variant}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Compile ControlNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--remoteweights', action='store_true', help="Use safetensors from Huggingface, or from local")
  parser.add_argument('--weights_dir', type=str, default=os.path.dirname(__file__), help="Path to weights directory")
  parser.add_argument('--variant', type=str, default='canny', help="Variant of ControlNet to use")
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
    
    kernel_code = '\n\n'.join([f"const {key} = `{code.replace(key, 'main')}`;" for key, code in functions.items()])
    kernel_names = ', '.join([name for (name, _, _, _) in statements])
    
    # Creates bind groups and compute pass
    kernel_calls = '\n    '.join([f"addComputePass(device, commandEncoder, piplines[{i}], [{', '.join(args)}], {global_size});" for i, (_name, args, global_size, _local_size) in enumerate(statements) ])
    
    # Allocate GPU IO buffers
    bufs =  '\n  '.join([f"const {name} = " + (f"createEmptyBuf(device, {size});" if _key not in weights else f"await createWeightBuf(device, {size}, filename, metadata['{weights[_key]}'], metadataLength)") + ";"  for name,(size,dtype,_key) in bufs.items()])
    
    # Allocate Stage buffer for input
    gpu_write_bufs =  '\n  '.join([f"const gpuWriteBuffer{i} = device.createBuffer({{size:input{i}.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE }});" for i,(_,value) in enumerate(special_names.items()) if "output" not in value])
    
    # Write input data to GPU
    input_writer = '\n  '.join([f"await gpuWriteBuffer{i}.mapAsync(GPUMapMode.WRITE);\n  new Float32Array(gpuWriteBuffer{i}.getMappedRange()).set(" + f'data{i});' + f"\n  gpuWriteBuffer{i}.unmap();\ncommandEncoder.copyBufferToBuffer(gpuWriteBuffer{i}, 0, input{i}, 0, gpuWriteBuffer{i}.size);"  for i,(_,value) in enumerate(special_names.items()) if value != "output0"])
    return f"""\n  var {step.name} = function() {{
  
    {kernel_code}

    return {{
    "setup": async (device, filename, metadata, metadataLength) => {{
      {bufs}
      
      {gpu_write_bufs}
      const gpuReadBuffer = device.createBuffer({{ size: output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }});

      const kernels = [{kernel_names}];
      const piplines = await Promise.all(kernels.map(name => device.createComputePipelineAsync({{layout: "auto", compute: {{ module: device.createShaderModule({{ code: name }}), entryPoint: "main" }}}})));

      return async ({",".join([f'data{i}' for i,(k,v) in enumerate(special_names.items()) if v != "output0"])}) => {{
        const commandEncoder = device.createCommandEncoder();

        {input_writer}

        {kernel_calls}
        commandEncoder.copyBufferToBuffer(output0, 0, gpuReadBuffer, 0, output0.size);
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        await gpuReadBuffer.mapAsync(GPUMapMode.READ);
        const resultBuffer = new Float32Array(gpuReadBuffer.size/4);
        resultBuffer.set(new Float32Array(gpuReadBuffer.getMappedRange()));
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
      safe_save(state, os.path.join(args.weights_dir, "net_{}.safetensors".format(args.variant)))
    
  prekernel = f"""
    window.MODEL_BASE_URL= "{base_url}";
    const getTensorMetadata = (safetensorBuffer) => {{
      const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
      const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
      return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {{...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}}]));
    }};

  const getTensorBuffer = (safetensorParts, tensorMetadata, key) => {{
    // Dynamically compute start offsets in bytes
    let partStartOffsets = [];
    let totalLength = 0;

    for (let part of safetensorParts) {{
        partStartOffsets.push(totalLength);
        totalLength += part.length; // part.length is already in bytes
    }}

    let selectedPart = 0;
    let correctedOffsets = tensorMetadata.data_offsets;

    for (let i = 0; i < partStartOffsets.length; i++) {{
        let start = partStartOffsets[i];
        let prevOffset = (i === 0) ? 0 : partStartOffsets[i - 1];

        if (tensorMetadata.data_offsets[0] < start) {{
            selectedPart = i;
            correctedOffsets = [correctedOffsets[0] - prevOffset, correctedOffsets[1] - prevOffset];
            break;
        }}
    }}

    let allZero = true;
    let out = safetensorParts[selectedPart].subarray(...correctedOffsets);

    for (let i = 0; i < out.length; i++) {{
        if (out[i] !== 0) {{
            allZero = false;
            break;
        }}
    }}

    if (allZero) {{
        console.log("Error: weight '" + key + "' is all zero.");
    }}

    return out;
  }};


  const getWeight = (safetensors, key) => {{
    let uint8Data = getTensorBuffer(safetensors, getTensorMetadata(safetensors[0])[key], key);
    return new Float32Array(uint8Data.buffer, uint8Data.byteOffset, uint8Data.byteLength / Float32Array.BYTES_PER_ELEMENT);
  }}

  const createEmptyBuf = (device, size) => {{
      return device.createBuffer({{size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }});
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
  
  const readWeightsFromFile = async (filename, tensorMetaData, metadataLength) => {{
    const dataOffsets = tensorMetaData.data_offsets;
    const data = await readRangeFromFile(
      filename, 
      dataOffsets[0] + 8 + metadataLength,
      dataOffsets[1] + 8 + metadataLength
    );
    return data;
  }};

  const createWeightBuf = async (device, size, filename, tensorMetaData, metadataLength) => {{
    
    let data = await readWeightsFromFile(filename, tensorMetaData, metadataLength);

    const buf = device.createBuffer({{ mappedAtCreation: true, size, usage: GPUBufferUsage.STORAGE }});
    new Uint8Array(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
  }};

  const addComputePass = (device, commandEncoder, pipeline, bufs, workgroup) => {{
    const bindGroup = device.createBindGroup({{layout: pipeline.getBindGroupLayout(0), entries: bufs.map((buffer, index) => ({{ binding: index, resource: {{ buffer }} }}))}});
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(...workgroup);
    passEncoder.end();
  }};"""

  with open(os.path.join(os.path.dirname(__file__), "net.js"), "w") as text_file:
    text_file.write(prekernel + prg)
