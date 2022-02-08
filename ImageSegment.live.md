# Semantic Image Segmentation by DeepLab3

## 0.Original work
Liang-Chieh Chen and Yukun Zhu "Semantic Image Segmentation with DeepLab in TensorFlow"
* https://www.tensorflow.org/lite/examples/segmentation/overview
* https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html

JoonBeom Park "TFLite Segmentation Python"
* https://github.com/joonb14/TFLiteSegmentation

***Thanks a lot!!!***

---

## 1.Helper module

Create the module to assist with tasks such as downloading a model.

```elixir
defmodule Helper do
  @model_file "lite-model_deeplabv3_1_metadata_2.tflite"

  @wearhouse "https://github.com/joonb14/TFLiteSegmentation/raw/main/#{@model_file}"
  @local "/data/#{@model_file}"

  def model() do
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
Helper.get()
```

## Implementation for Elixir/Nerves using TflInterp

## 2.Defining the inference module: DeepLab3

* Pre-processing:<br>
  Resize the input image to the size of `@deeplab3_shape` and create a Float32 binary sequence normalized to the range {-1.0, 1.0}.

* Post-processing:<br>

```elixir
defmodule DeepLab3 do
  # use TflInterp, model: Model.file()
  use TflInterp

  @deeplab3_shape {257, 257}

  def apply(jpeg) do
    # preprocess
    bin =
      CImg.from_binary(jpeg)
      |> CImg.resize(@deeplab3_shape)
      |> CImg.to_binary(range: {-1.0, 1.0})

    # prediction
    outputs =
      __MODULE__
      |> TflInterp.set_input_tensor(0, bin)
      |> TflInterp.invoke()
      |> TflInterp.get_output_tensor(0)
      |> Nx.from_binary({:f, 32})
      |> Nx.reshape({257, 257, :auto})

    # postprocess
    _result =
      outputs
      |> Nx.argmax(axis: 2)
      |> Nx.as_type({:u, 8})
      |> Nx.to_binary()
      |> CImg.from_binary(257, 257, 1, 1, "<u1")
      |> CImg.color_mapping(:lines)
  end
end
```

Launch `DeepLab3`.

```elixir
DeepLab3.start_link(model: Helper.model())
```

Displays the properties of the `DeepLab3` model.

```elixir
TflInterp.info(DeepLab3)
```

## 3.Let's try it

```elixir
img = Picam.next_frame()

Kino.render(Kino.Image.new(img, :jpeg))

DeepLab3.apply(img)
|> CImg.resize({320, 240})
|> CImg.to_binary(:png)
|> Kino.Image.new(:png)
```

## 4.TIL ;-)

#### Date: Feb. 8, 2022 / Nerves-livebook rpi3
Total processing time is about 7.8 seconds, excluding camera shooting.
Of that time, the DeepLab3 inference - TflInterp.invoke(DeepLab3) - takes about 280 micro seconds,
and the post-processing Nx.argmax takes about 5.8 seconds.

&#9633;

#### License

Copyright 2022 Shozo Fukuda.
Apache License Version 2.0
