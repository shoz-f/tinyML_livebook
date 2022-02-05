# Monocular Depth Estimation by MiDaS v2.1

## 0.Original work

Intelligent Systems Lab Org:<br>
"Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer"

- https://arxiv.org/abs/1907.01341v3
- https://github.com/isl-org/MiDaS

***Thanks a lot!!!***

---

## Implementation for Elixir/Nerves using TflInterp

## 1.Helper module
Create the module to assist with tasks such as downloading a model.

```elixir
defmodule Model do
  @model_file "midas_opt.tflite"

  @wearhouse "https://github.com/isl-org/MiDaS/releases/download/v2_1/model_opt.tflite"
  @local "/data/#{@model_file}"

  def file() do
    @local
  end

  def get() do
    Req.get!(@wearhouse).body
    |> then(fn x -> File.write(@local, x) end)
  end

  def rm() do
    File.rm(@local)
  end

  def exists?() do
    File.exists?(@local)
  end
end
```

Get the tflite model from `@wearhouse` and store it in `@local`.

```elixir
Model.get()
```

## 2.Defining the inference module: Midas
```elixir
defmodule Midas do
  #use TflInterp, model: Model.file()
  use TflInterp

  @midas_shape {256, 256}

  def apply(img) do
    # preprocess
    bin =
      img
      |> CImg.resize(@midas_shape)
      |> CImg.to_binary(range: {-2.0, 2.0})

    # prediction
    outputs =
      __MODULE__
      |> TflInterp.set_input_tensor(0, bin)
      |> TflInterp.invoke()
      |> TflInterp.get_output_tensor(0)
      |> Nx.from_binary({:f, 32})
      |> Nx.reshape({256, 256})

    # postprocess
    [min, max] =
      [Nx.window_min(outputs, {256, 256}), Nx.window_max(outputs, {256, 256})]
      |> Enum.map(&Nx.squeeze/1)
      |> Enum.map(&Nx.to_number/1)

    _result =
      outputs
      |> Nx.subtract(min)
      |> Nx.divide(max - min)
      |> Nx.to_binary()
      |> CImg.from_binary(256, 256, 1, 1, "<f4")
  end
end
```

Launch `Midas`.

```elixir
Midas.start_link(model: Model.file())
```

Displays the properties of the `Midas` model.

```elixir
TflInterp.info(Midas)
```

## 3.Let's try it


```elixir
img = Picam.next_frame()

Kino.render(Kino.Image.new(img, :jpeg))

CImg.from_binary(img)
|> Midas.apply()
|> CImg.resize({320, 240})
|> CImg.color_mapping(:jet)
|> CImg.to_binary(:png)
|> Kino.Image.new(:png)
```
## 4.TIL ;-)

#### Date: Feb. 5, 2022 / Nerves-livebook rpi3

It takes a long time to quantize the depth image in  post-processing, 

The heatmap scale (256) is narrow, so you may not see the depth details.

&#9633;

#### License
Copyright 2022 Shozo Fukuda.
Apache License Version 2.0
