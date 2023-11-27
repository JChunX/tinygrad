import os
import argparse
from collections import namedtuple
from pathlib import Path
from typing import Any, List, NamedTuple

from tinygrad.ops import Device
from tinygrad.nn.state import load_state_dict, torch_load, get_state_dict, safe_save
from tinygrad.tensor import Tensor
from extra.export_model import jit_model, compile_net

from examples.webgpu.stable_diffusion.compile import (
  convert_f32_to_f16,
  split_safetensor
)
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


FILENAME_DIFFUSION = Path(__file__).parents[3] / "weights/sd-v1-4.ckpt"
FILENAME_CONTROLNET = Path(__file__).parents[3] / "weights/sd-controlnet-canny.bin"

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Compile ControlNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--remoteweights', action='store_true', help="Use safetensors from Huggingface, or from local")
  parser.add_argument('--weights_dir', type=str, default=os.path.dirname(__file__), help="Path to weights directory")
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
  download_file(
  'https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/diffusion_pytorch_model.bin', FILENAME_CONTROLNET)
  state_dict_controlnet = torch_load(FILENAME_CONTROLNET)
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
    bufs =  '\n  '.join([f"const {name} = " + (f"createEmptyBuf(device, {size});" if _key not in weights else f"createWeightBuf(device, {size}, getTensorBuffer(safetensor, metadata['{weights[_key]}'], '{weights[_key]}'))") + ";"  for name,(size,dtype,_key) in bufs.items()])
    
    # Allocate Stage buffer for input
    gpu_write_bufs =  '\n  '.join([f"const gpuWriteBuffer{i} = device.createBuffer({{size:input{i}.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE }});" for i,(_,value) in enumerate(special_names.items()) if "output" not in value])
    
    # Write input data to GPU
    input_writer = '\n  '.join([f"await gpuWriteBuffer{i}.mapAsync(GPUMapMode.WRITE);\n  new Float32Array(gpuWriteBuffer{i}.getMappedRange()).set(" + f'data{i});' + f"\n  gpuWriteBuffer{i}.unmap();\ncommandEncoder.copyBufferToBuffer(gpuWriteBuffer{i}, 0, input{i}, 0, gpuWriteBuffer{i}.size);"  for i,(_,value) in enumerate(special_names.items()) if value != "output0"])
    return f"""\n  var {step.name} = function() {{
  
    {kernel_code}

    return {{
    "setup": async (device, safetensor) => {{
      const metadata = getTensorMetadata(safetensor[0]);

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

  base_url = ""
  if not os.path.exists(args.weights_dir):
    os.makedirs(args.weights_dir)
    
  for step in sub_steps:
    prg += compile_step(model, step)
    
    if step.name == "diffusor":
      if args.remoteweights:
        base_url = "https://huggingface.co/jchun/tinygrad-sd-controlnet-f16/resolve/main/"
      else:
        state = get_state_dict(model)
        safetensor_path = os.path.join(args.weights_dir, "net.safetensors")
        safetensor_conv_path = safetensor_path.replace(".safetensors", "_conv.safetensors")
        if not os.path.exists(os.path.join(args.weights_dir, "net_part0.safetensors")):
          if not os.path.exists(safetensor_path):
            print("Saving safetensors to: ", safetensor_path)
            safe_save(state, safetensor_path)
          if not os.path.exists(safetensor_conv_path):
            print("Converting safetensors to f16...")
            convert_f32_to_f16(safetensor_path,
                            safetensor_conv_path)
          print("Splitting safetensors...")
          split_safetensor(safetensor_conv_path)
        os.remove(safetensor_path)
        os.remove(safetensor_conv_path)
        
        base_url = "."
    
  prekernel = f"""
    window.MODEL_BASE_URL= "{base_url}";
    const getTensorMetadata = (safetensorBuffer) => {{
      const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
      const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
      return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {{...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}}]));
    }};

  const getTensorBuffer = (safetensorParts, tensorMetadata, key) => {{
    let selectedPart = 0;
    let counter = 0;
    let partStartOffsets = [1131408336, 2227518416, 3308987856, 4265298864];
    let correctedOffsets = tensorMetadata.data_offsets;
    let prev_offset = 0;

    for (let start of partStartOffsets) {{
      prev_offset = (counter == 0) ? 0 : partStartOffsets[counter-1];

      if (tensorMetadata.data_offsets[0] < start) {{
        selectedPart = counter;
        correctedOffsets = [correctedOffsets[0]-prev_offset, correctedOffsets[1]-prev_offset];
        break;
      }}

      counter++;
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

    return safetensorParts[selectedPart].subarray(...correctedOffsets);
  }}

  const getWeight = (safetensors, key) => {{
    let uint8Data = getTensorBuffer(safetensors, getTensorMetadata(safetensors[0])[key], key);
    return new Float32Array(uint8Data.buffer, uint8Data.byteOffset, uint8Data.byteLength / Float32Array.BYTES_PER_ELEMENT);
  }}

  const createEmptyBuf = (device, size) => {{
      return device.createBuffer({{size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }});
  }};

  const createWeightBuf = (device, size, data) => {{
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
