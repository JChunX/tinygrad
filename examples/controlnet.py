
from collections import namedtuple
from pathlib import Path
from typing import Optional, Tuple, Union

from examples.stable_diffusion import (
  UNetModel,
  StableDiffusion,
  ResBlock,
  SpatialTransformer,
  timestep_embedding,
)
from tinygrad.nn import GroupNorm, Linear, Conv2d
from tinygrad.tensor import Tensor


def zero_module(module):
  module.weight = Tensor.zeros_like(module.weight)
  module.bias = Tensor.zeros_like(module.bias)
  return module


class ControlNetUNetModel(UNetModel):
  '''
  Tinygrad SD controlled UNet.
  see 
  https://github.com/lllyasviel/ControlNet/blob/main/cldm/cldm.py &
  https://github.com/tinygrad/tinygrad/blob/master/examples/stable_diffusion.py
  '''

  def __call__(self, x, timesteps=None, context=None, control=None, only_mid_control=False):
    # (stable_diffusion.py) TODO: real time embedding
    t_emb = timestep_embedding(timesteps, 320)
    emb = t_emb.sequential(self.time_embed)

    def run(x, bb):
      if isinstance(bb, ResBlock):
        x = bb(x, emb)
      elif isinstance(bb, SpatialTransformer):
        x = bb(x, context)
      else:
        x = bb(x)
      return x

    saved_inputs = []

    for b in self.input_blocks:
      for bb in b:
        x = run(x, bb)
      saved_inputs.append(x)

    for bb in self.middle_block:
      x = run(x, bb)

    if control is not None:  # middle control
      x += control.pop()

    for b in self.output_blocks:
      if only_mid_control or control is None:
        x = x.cat(saved_inputs.pop(), dim=1)
      else:
        x = x.cat(saved_inputs.pop() + control.pop(), dim=1)
      for bb in b:
        x = run(x, bb)

    return x.sequential(self.out)


class ControlNetConditioningEmbedding:
  '''
  Converts conditioning image into 64x64 feature maps
  See Huggingface diffusers/src/diffusers/models/controlnet.py
  '''

  def __init__(self,
         conditioning_embedding_channels,
         conditioning_channels=3,
         block_out_channels=(16, 32, 96, 256),
         ):
    self.conv_in = Conv2d(conditioning_channels,
                block_out_channels[0], 3, padding=1)
    self.blocks = []

    for i in range(len(block_out_channels) - 1):
      channel_in = block_out_channels[i]
      channel_out = block_out_channels[i+1]
      self.blocks.append(Conv2d(channel_in, channel_in, 3, padding=1))
      self.blocks.append(
        Conv2d(channel_in, channel_out, 3, padding=1, stride=2))

    self.conv_out = zero_module(
      Conv2d(block_out_channels[-1],
           conditioning_embedding_channels, 3, padding=1)
    )

  def __call__(self, conditioning):
    embedding = self.conv_in(conditioning)
    embedding = Tensor.silu(embedding)

    for block in self.blocks:
      embedding = block(embedding)
      embedding = Tensor.silu(embedding)

    embedding = self.conv_out(embedding)

    return embedding


class ControlNetResBlock:
  def __init__(self, channels, emb_channels, out_channels):

    self.norm1 = GroupNorm(32, channels)
    self.conv1 = Conv2d(channels, out_channels, 3, padding=1)

    self.time_emb_proj = Linear(emb_channels, out_channels)

    self.norm2 = GroupNorm(32, out_channels)
    self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1)

    self.conv_shortcut = Conv2d(
      channels, out_channels, 1) if channels != out_channels else lambda x: x

  def __call__(self, x, emb):

    h = self.conv1(Tensor.silu(self.norm1(x)))
    emb_out = self.time_emb_proj(Tensor.silu(emb))
    h = h + emb_out.reshape(*emb_out.shape, 1, 1)
    h = self.conv2(Tensor.silu(self.norm2(h)))
    ret = self.conv_shortcut(x) + h

    return ret


class ControlNetDownsample:
  def __init__(self, channels):
    self.conv = Conv2d(channels, channels, 3, stride=2, padding=1)

  def __call__(self, x):
    return self.conv(x)


class ControlNetModel:
  '''
  Tinygrad SD controlnet.
  see ControlNetModel in diffusers/models/controlnet.py
  '''

  def __init__(
    self,
    in_channels: int = 4,
    conditioning_channels: int = 3,
    down_block_types: Tuple[str] = (
      "CrossAttnDownBlock2D",
      "CrossAttnDownBlock2D",
      "CrossAttnDownBlock2D",
      "DownBlock2D",
    ),
    only_cross_attention: Union[bool, Tuple[bool]] = False,
    block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
    layers_per_block: int = 2,
    cross_attention_dim: int = 1280,
    transformer_layers_per_block: Union[int, Tuple[int]] = 1,
    attention_head_dim: Union[int, Tuple[int]] = 8,
    num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
    conditioning_embedding_out_channels: Optional[Tuple[int]] = (
      16, 32, 96, 256),
  ):

    num_attention_heads = num_attention_heads or attention_head_dim
    self.block_out_channels = block_out_channels

    # Check inputs
    if len(block_out_channels) != len(down_block_types):
      raise ValueError(
        f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
      )

    if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
      raise ValueError(
        f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
      )

    if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
      raise ValueError(
        f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
      )

    if isinstance(transformer_layers_per_block, int):
      transformer_layers_per_block = [
        transformer_layers_per_block] * len(down_block_types)

    # input
    conv_in_kernel = 3
    conv_in_padding = (conv_in_kernel - 1) // 2
    self.conv_in = Conv2d(
      in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
    )

    # time
    time_embed_dim = block_out_channels[0] * 4

    self.time_embedding = namedtuple("TimeEmbedding", ["linear_1", "linear_2"])(Linear(
      block_out_channels[0], time_embed_dim), Linear(time_embed_dim, time_embed_dim))

    self.encoder_hid_proj = None
    self.class_embedding = None

    # control net conditioning embedding
    self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
      conditioning_embedding_channels=block_out_channels[0],
      block_out_channels=conditioning_embedding_out_channels,
      conditioning_channels=conditioning_channels,
    )

    self.down_blocks = []
    self.controlnet_down_blocks = []  # zero convs

    if isinstance(only_cross_attention, bool):
      only_cross_attention = [
        only_cross_attention] * len(down_block_types)

    if isinstance(attention_head_dim, int):
      attention_head_dim = (attention_head_dim,) * len(down_block_types)

    if isinstance(num_attention_heads, int):
      num_attention_heads = (num_attention_heads,) * \
        len(down_block_types)

    # down
    output_channel = block_out_channels[0]

    controlnet_block = Conv2d(
      output_channel, output_channel, kernel_size=1)
    controlnet_block = zero_module(controlnet_block)
    self.controlnet_down_blocks.append(controlnet_block)

    # see huggingface diffusers/models/controlnet.py, line 780
    for i, down_block_type in enumerate(down_block_types):
      input_channel = output_channel
      output_channel = block_out_channels[i]
      is_final_block = i == len(block_out_channels) - 1

      down_block = get_down_block(
        down_block_type,
        num_layers=layers_per_block,
        in_channels=input_channel,
        out_channels=output_channel,
        temb_channels=time_embed_dim,
        add_downsample=not is_final_block,
        transformer_layers_per_block=transformer_layers_per_block[i],
        num_attention_heads=num_attention_heads[i],
        cross_attention_dim=cross_attention_dim,
        attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel
      )
      self.down_blocks.append(down_block)

      for _ in range(layers_per_block):
        # zero conv for each layer in block
        controlnet_block = Conv2d(
          output_channel, output_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

      if not is_final_block:
        # zero conv for ControlNetDownsample block
        controlnet_block = Conv2d(
          output_channel, output_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

    # mid
    mid_block_channel = block_out_channels[-1]

    controlnet_block = Conv2d(
      mid_block_channel, mid_block_channel, kernel_size=1)
    controlnet_block = zero_module(controlnet_block)
    self.controlnet_mid_block = controlnet_block

    self.mid_block = UNetMidBlock2DCrossAttn(
      in_channels=mid_block_channel,
      temb_channels=time_embed_dim,
      transformer_layers_per_block=transformer_layers_per_block[-1],
      num_attention_heads=num_attention_heads[-1],
      cross_attention_dim=cross_attention_dim
    )

  def __call__(self, sample, control, timesteps, encoder_hidden_states, conditioning_scale=1.0):
    '''
    Args:
    - sample: 4x64x64 image latent
    - control: image conditioning
    - timesteps: timestep of the current iteration
    - encoder_hidden_states: prompt encoding
    '''
    t_emb = timestep_embedding(timesteps, self.block_out_channels[0])
    emb = self.time_embedding.linear_2(
      Tensor.silu(self.time_embedding.linear_1(t_emb)))

    sample = self.conv_in(sample)
    sample += self.controlnet_cond_embedding(control)

    down_outputs = (sample,)

    for downsample_block in self.down_blocks:
      if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
        sample, res_samples = downsample_block(
          hidden_states=sample,
          temb=emb,
          encoder_hidden_states=encoder_hidden_states
        )
      else:
        sample, res_samples = downsample_block(
          hidden_states=sample, temb=emb)
      down_outputs += res_samples

    # 4. mid
    if self.mid_block is not None:
      sample = self.mid_block(
        sample,
        emb,
        encoder_hidden_states=encoder_hidden_states
      )

    # 5. zero convs
    outputs = []

    for sample, zero_conv in zip(down_outputs, self.controlnet_down_blocks):
      outputs.append(zero_conv(sample))

    mid_block_res_sample = self.controlnet_mid_block(sample)
    outputs.append(mid_block_res_sample)

    # 6. scaling
    outputs = [x * conditioning_scale for x in outputs]

    return outputs


def get_down_block(
  down_block_type: str,
  num_layers: int,
  in_channels: int,
  out_channels: int,
  temb_channels: int,
  add_downsample: bool,
  transformer_layers_per_block: int = 1,
  num_attention_heads: Optional[int] = None,
  cross_attention_dim: Optional[int] = None,
  attention_head_dim: Optional[int] = None,
):
  # If attn head dim is not defined, we default it to the number of heads
  if attention_head_dim is None:
    print(
      f"It is recommended to provide `attention_head_dim` when calling `get_down_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
    )
    attention_head_dim = num_attention_heads

  down_block_type = down_block_type[7:] if down_block_type.startswith(
    "UNetRes") else down_block_type
  if down_block_type == "DownBlock2D":
    return DownBlock2D(
      in_channels=in_channels,
      out_channels=out_channels,
      num_layers=num_layers,
      temb_channels=temb_channels,
      add_downsample=add_downsample,
    )
  elif down_block_type == "CrossAttnDownBlock2D":
    if cross_attention_dim is None:
      raise ValueError(
        "cross_attention_dim must be specified for CrossAttnDownBlock2D")
    return CrossAttnDownBlock2D(
      in_channels=in_channels,
      out_channels=out_channels,
      temb_channels=temb_channels,
      num_layers=num_layers,
      transformer_layers_per_block=transformer_layers_per_block,
      num_attention_heads=num_attention_heads,
      cross_attention_dim=cross_attention_dim,
      add_downsample=add_downsample,
    )
  raise ValueError(f"{down_block_type} does not exist.")


class DownBlock2D:
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    num_layers: int = 1,
    temb_channels: int = 512,
    add_downsample: bool = True,
  ):
    super().__init__()
    resnets = []

    for i in range(num_layers):
      in_channels = in_channels if i == 0 else out_channels
      resnets.append(
        ControlNetResBlock(
          channels=in_channels,
          emb_channels=temb_channels,
          out_channels=out_channels
        )
      )

    self.resnets = resnets

    if add_downsample:
      self.downsamplers = [
        ControlNetDownsample(out_channels)
      ]
    else:
      self.downsamplers = None

  def __call__(self, hidden_states, temb=None):
    output_states = ()

    for resnet in self.resnets:
      hidden_states = resnet(hidden_states, temb)
      output_states = output_states + (hidden_states,)

    if self.downsamplers is not None:
      for downsampler in self.downsamplers:
        hidden_states = downsampler(hidden_states)

      output_states = output_states + (hidden_states,)

    return hidden_states, output_states


class CrossAttnDownBlock2D:
  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    temb_channels: int = 512,
    num_layers: int = 1,
    transformer_layers_per_block: Union[int, Tuple[int]] = 1,
    num_attention_heads: int = 1,
    cross_attention_dim: int = 768,
    add_downsample: bool = True,
  ):
    super().__init__()
    resnets = []
    attentions = []

    self.has_cross_attention = True
    self.num_attention_heads = num_attention_heads
    if isinstance(transformer_layers_per_block, int):
      transformer_layers_per_block = [
        transformer_layers_per_block] * num_layers

    for i in range(num_layers):
      in_channels = in_channels if i == 0 else out_channels
      resnets.append(
        ControlNetResBlock(
          channels=in_channels,
          emb_channels=temb_channels,
          out_channels=out_channels,
        )
      )
      attentions.append(
        SpatialTransformer(
          out_channels,
          cross_attention_dim,
          num_attention_heads,
          out_channels // num_attention_heads,
          num_layers=transformer_layers_per_block[i],
        )
      )

    self.attentions = attentions
    self.resnets = resnets

    if add_downsample:
      self.downsamplers = [ControlNetDownsample(out_channels)]
    else:
      self.downsamplers = None

  def __call__(
    self,
    hidden_states,
    temb=None,
    encoder_hidden_states=None,
  ):
    output_states = ()

    blocks = list(zip(self.resnets, self.attentions))

    for _, (resnet, attn) in enumerate(blocks):

      hidden_states = resnet(hidden_states, temb)
      hidden_states = attn(
        hidden_states,
        context=encoder_hidden_states,
      )

      output_states = output_states + (hidden_states,)

    if self.downsamplers is not None:
      for downsampler in self.downsamplers:
        hidden_states = downsampler(hidden_states,)

      output_states = output_states + (hidden_states,)

    return hidden_states, output_states


class UNetMidBlock2DCrossAttn:
  def __init__(
    self,
    in_channels: int,
    temb_channels: int,
    num_layers: int = 1,
    transformer_layers_per_block: Union[int, Tuple[int]] = 1,
    num_attention_heads: int = 1,
    cross_attention_dim: int = 1280,
  ):
    super().__init__()

    self.has_cross_attention = True
    self.num_attention_heads = num_attention_heads

    # support for variable transformer layers per block
    if isinstance(transformer_layers_per_block, int):
      transformer_layers_per_block = [
        transformer_layers_per_block] * num_layers

    # there is always at least one resnet
    resnets = [
      ControlNetResBlock(
        channels=in_channels,
        emb_channels=temb_channels,
        out_channels=in_channels,
      )
    ]
    attentions = []

    for _ in range(num_layers):
      attentions.append(
        SpatialTransformer(
          in_channels,
          cross_attention_dim,
          num_attention_heads,
          in_channels // num_attention_heads,
        )
      )

      resnets.append(
        ControlNetResBlock(
          channels=in_channels,
          emb_channels=temb_channels,
          out_channels=in_channels,
        )
      )

    self.attentions = attentions
    self.resnets = resnets

  def __call__(
    self,
    hidden_states,
    temb=None,
    encoder_hidden_states=None,
  ):

    hidden_states = self.resnets[0](hidden_states, temb)
    for attn, resnet in zip(self.attentions, self.resnets[1:]):
      hidden_states = attn(
        hidden_states,
        context=encoder_hidden_states,
      )
      hidden_states = resnet(hidden_states, temb)

    return hidden_states


class ControlNetStableDiffusion(StableDiffusion):

  def __init__(self, diffusion_model, controlnet):
    super().__init__()
    self.alphas_cumprod = diffusion_model.alphas_cumprod
    self.model = diffusion_model.model
    self.first_stage_model = diffusion_model.first_stage_model
    self.cond_stage_model = diffusion_model.cond_stage_model

    self.controlnet = controlnet

  def get_model_output(
      self,
      control,
      unconditional_context,
      context,
      latent,
      timestep,
      unconditional_guidance_scale,
      conditioning_scale):

    context = unconditional_context.cat(context, dim=0)

    control = self.controlnet(
      latent,
      control,
      timestep,
      context,
      conditioning_scale
    )

    latents = self.model.diffusion_model(latent.expand(
      2, *latent.shape[1:]), timestep, context, control)
    unconditional_latent, latent = latents[0:1], latents[1:2]

    e_t = unconditional_latent + unconditional_guidance_scale * \
      (latent - unconditional_latent)

    return e_t

  def __call__(
      self,
      control,
      unconditional_context,
      context,
      latent,
      timestep,
      alphas,
      alphas_prev,
      guidance,
      conditioning_scale):
    '''
    Args:
    - unconditional_context: CLIP embedding of the unconditional prompt
    - context: CLIP embedding of the prompt
    - latent: 4x64x64 image latent tensor
    - timestep: timestep of the current iteration
    - guidance: scale of unconditional guidance
    - conditioning_scale: scale of image conditioning
    '''

    e_t = self.get_model_output(
      control, unconditional_context, context, latent, timestep, guidance, conditioning_scale)
    x_prev, _ = self.get_x_prev_and_pred_x0(
      latent, e_t, alphas, alphas_prev)

    return x_prev.realize()
  
FILENAME_DIFFUSION = Path(__file__).parents[1] / "weights/sd-v1-4.ckpt"
FILENAME_CONTROLNET = Path(__file__).parents[1] / "weights/sd-controlnet-canny.bin"

if __name__ == "__main__":
  from examples.stable_diffusion import *
  from PIL import Image

  import numpy as np
  import cv2
  
  parser = argparse.ArgumentParser(description='Run ControlNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--steps', type=int, default=5, help='Number of steps in diffusion')
  parser.add_argument('--control_image', type=str, default="examples/webgpu/controlnet/vermeer.png", help='Image (512x512) to use as control')
  parser.add_argument('--prompt', type=str, default="a photo of a cat")
  parser.add_argument('--out', type=str, default=Path(tempfile.gettempdir()) / "rendered.png", help="Output filename")
  parser.add_argument('--noshow', action='store_true', help="Don't show the image")
  parser.add_argument('--fp16', action='store_true', help="Cast the weights to float16")
  parser.add_argument('--timing', action='store_true', help="Print timing per step")
  parser.add_argument('--seed', type=int, help="Set the random latent seed")
  parser.add_argument('--guidance', type=float, default=7.5, help="Prompt strength")
  parser.add_argument('--conditioning_scale', type=float, default=1.0, help="Conditioning strength")
  parser.add_argument('--canny_high', type=int, default=200, help="Canny high threshold")
  parser.add_argument('--canny_low', type=int, default=100, help="Canny low threshold")
  args = parser.parse_args()
  
  Tensor.no_grad = True

  # Create control image
  image = Image.open(args.control_image)
  image = np.array(image)

  image = cv2.Canny(image, args.canny_low, args.canny_high)
  image = image[:, :, None]
  image = np.concatenate([image, image, image], axis=2)
  canny_image = Image.fromarray(image)
  canny_condition = np.array(canny_image).transpose(2, 0, 1).astype(np.float32) / 255.0
  canny_condition = Tensor(canny_condition).unsqueeze(0)

  # Load model
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
  
  if args.fp16:
    for l in get_state_dict(model).values():
      l.assign(l.cast(dtypes.float16).realize())

  # CLIP encoding
  tokenizer = ClipTokenizer()
  prompt = Tensor([tokenizer.encode(args.prompt)])
  context = model.cond_stage_model.transformer.text_model(prompt).realize()
  print("got CLIP context", context.shape)

  prompt = Tensor([tokenizer.encode("")])
  unconditional_context = model.cond_stage_model.transformer.text_model(
    prompt).realize()
  print("got unconditional CLIP context", unconditional_context.shape)

  del model.cond_stage_model

  timesteps = list(range(1, 1000, 1000//args.steps))
  print(f"running for {timesteps} timesteps")
  alphas = model.alphas_cumprod[Tensor(timesteps)]
  alphas_prev = Tensor([1.0]).cat(alphas[:-1])

  if args.seed is not None: Tensor._seed = args.seed
  latent = Tensor.randn(1, 4, 64, 64)

  @TinyJit
  def run(model, *x): return model(*x).realize()

  with Context(BEAM=getenv("LATEBEAM")):
    for index, timestep in (t := tqdm(list(enumerate(timesteps))[::-1])):
      GlobalCounters.reset()
      t.set_description("%3d %3d" % (index, timestep))
      with Timing("step in ", enabled=args.timing, on_exit=lambda _: f", using {GlobalCounters.mem_used/1e9:.2f} GB"):
        tid = Tensor([index])
        print(canny_condition.shape)
        latent = run(model, canny_condition, unconditional_context, context, latent, Tensor(
          [timestep]), alphas[tid], alphas_prev[tid], Tensor([args.guidance]), Tensor([args.conditioning_scale]))
        if args.timing: Device[Device.DEFAULT].synchronize()
    del run

  while (not latent.numpy().any() or (np.isnan(latent.numpy()).any())):
    print("retry latent memory read: ", latent.numpy().max())
  # upsample latent space to image with autoencoder
  x = model.decode(latent)
  print(x.shape)

  # save image
  im = Image.fromarray(x.numpy().astype(np.uint8, copy=False))
  print(f"saving {args.out}")
  im.save(args.out)
  # Open image.
  if not args.noshow: im.show()