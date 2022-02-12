# Artistic Style

## 0.Original work

"Fast Style Transfer for Arbitrary Styles"

* https://github.com/magenta/magenta/tree/main/magenta/models/arbitrary_image_stylization
* https://www.tensorflow.org/lite/examples/style_transfer/overview

From their performance bench mark:

model name | size | device | NNAPI | CPU 
---------|-----|-------|------|----
Style prediction model(int8) | 2.8 Mb | Pixel 3 (Android 10)| 142ms | 14ms*
|| Pixel 4 (Android 10)| 5.2ms | 6.7ms*
|| iPhone XS (iOS 12.4.1)||10.7ms**	
Style transform model(int8)| 0.2 Mb | Pixel 3 (Android 10)||540ms*
|| Pixel 4 (Android 10)  || 405ms*
||iPhone XS (iOS 12.4.1) || 251ms**

<br />

Leon A. Gatys, Alexander S. Ecker, Matthias Bethge "A Neural Algorithm of Artistic Style"

* https://arxiv.org/abs/1508.06576

***Thanks a lot!!!***

---

## 1.Helper module

The module to assist with tasks such as downloading a model.

```elixir
defmodule Helper do
  @url_predict "https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite"
  @url_transform "https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite"

  @local_predict "/data/style_predict.tflite"
  @local_transform "/data/style_transform.tflite"

  def model(:predict) do
    @local_predict
  end

  def model(:transform) do
    @local_transform
  end

  def get() do
    [
      {@url_predict, @local_predict},
      {@url_transform, @local_transform}
    ]
    |> Enum.each(&get/1)
  end
  
  defp get({url, local}) do
    Req.get!(url).body
    |> then(fn x -> File.write(local, x) end)
  end

  def rm() do
    [@local_predict, @local_transform]
    |> Enum.each(&File.rm/1)
  end

  def exists?() do
    [@local_predict, @local_transform]
    |> Enum.all?(&File.exists?/1)
  end
end
```

Get the tflite models from `@url_*` and store them in `@local_*`.

```elixir
Helper.get()
```

## Implementation for Elixir/Nerves using TflInterp

## 2.Defining the DNN modules: StylePredict

* Pre-processing:<br>
 Resize the input image to the size of `@predict_shape` and create a Float32 binary sequence normalized to the range {0.0, 1.0}.

* Post-processing:<br>
 Kepp the output tensor as a style of the image.

```elixir
defmodule StylePredict do
  # use TflInterp, model: Model.file()
  use TflInterp

  @predict_shape {256, 256}

  def get_style(jpeg) do
    # preprocess
    bin =
      CImg.from_binary(jpeg)
      |> CImg.resize(@predict_shape)
      |> CImg.to_binary()

    # prediction
    _outputs =
      __MODULE__
      |> TflInterp.set_input_tensor(0, bin)
      |> TflInterp.invoke()
      |> TflInterp.get_output_tensor(0)
  end
end
```

Launch `StylePredict`.

```elixir
StylePredict.start_link(model: Helper.model(:predict))
```

Displays the properties of the `StylePredict` model.

```elixir
TflInterp.info(StylePredict)
```

## 3.Defining the DNN modules: StyleTransform

* Pre-processing:<br>
  Resize the input image to the size of `@transform_shape` and create a Float32 binary sequence normalized to the range {0.0, 1.0}.

* Post-processing:<br>
  Convert the output tensor to an @transform_shape image.

```elixir
defmodule StyleTransform do
  # use TflInterp, model: Model.file()
  use TflInterp

  @transform_shape {384, 384}

  def apply_style(jpeg, style) do
    # preprocess
    bin =
      CImg.from_binary(jpeg)
      |> CImg.resize(@transform_shape)
      |> CImg.to_binary()

    # prediction
    outputs =
      __MODULE__
      |> TflInterp.set_input_tensor(0, bin)
      |> TflInterp.set_input_tensor(1, style)
      |> TflInterp.invoke()
      |> TflInterp.get_output_tensor(0)
    
    # postprocess
    CImg.from_binary(outputs, 384, 384, 1, 3, "<f4")
  end
end
```

Launch `StyleTransform`.

```elixir
StyleTransform.start_link(model: Helper.model(:transform))
```

Displays the properties of the `StyleTransform` model.

```elixir
TflInterp.info(StyleTransform)
```

## 4.Let's try it

```elixir
style =
  Req.get!("https://github.com/shoz-f/tinyML_livebook/releases/download/model/helleborus.jpg").body
  |> tap(&Kino.render(Kino.Image.new(&1, :png)))
  |> StylePredict.get_style()

Picam.next_frame()
|> tap(&Kino.render(Kino.Image.new(&1, :jpeg)))
|> StyleTransform.apply_style(style)
|> CImg.resize({320, 240})
|> CImg.to_binary(:jpeg)
|> Kino.Image.new(:jpeg)
```

## 5.TIL ;-)

#### Date: Feb. 10, 2022 / Nerves-livebook rpi3

Total processing time is about 4.4 seconds, including downloading the style image and camera shooting.
Of that time, the StylePredict inference - TflInterp.invoke(StylePredict) - takes about 78 micro seconds,
and the StyleTransform inference - TflInterp.invoke(StyleTransform) - takes about 1.9 seconds.
That's about a quarter of the performance of the Pixel 3, which has a 2.5GHz + 1.6GHz, 64-bit octa-core CPU.
I think it's fighting a pretty good fight ;-)

&#9633;

#### License

Copyright 2022 Shozo Fukuda.
Apache License Version 2.0
