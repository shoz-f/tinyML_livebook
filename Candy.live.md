# Fast Style Transfer: Candy

## 0.Original work

Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian<br>
"YOLOX: Exceeding YOLO Series in 2021"

* https://arxiv.org/abs/2107.08430
* https://github.com/Megvii-BaseDetection/YOLOX

#### > A technical article on YOLOX in Japanese

@koshian2 "実装から見るYOLOX：2021年のYOLOシリーズを超えて"
* https://qiita.com/koshian2/items/af032cb102f48e789e66

***Thanks a lot!!!***

---

## Implementation for Elixir/Nerves using TflInterp

## 1.Helper module

Create the module to assist with tasks such as downloading a model.

```elixir
defmodule Helper do
  @model_file "candy.tflite"

  @wearhouse "https://github.com/shoz-f/tinyML_livebook/releases/download/model/"
  @local "/data/"

  def model() do
    @local <> @model_file
  end

  def get() do
    Req.get!(@wearhouse <> @model_file).body
    |> then(fn x -> File.write(model(), x) end)
  end

  def rm(:model), do: File.rm(model())

  def rm() do
    rm(:model)
  end

  def exists?(:model), do: File.exists?(model())

  def exists?() do
    exists?(:model) && exists?(:label)
  end
end
```

Get the tflite model and the coco lable from `@wearhouse` and store it in `@local`.

```elixir
Helper.get()
```

## 2.Defining the inference module: YoloX

* Model<br>
Standard Model: YOLOX-s 640 converted from Pytorch model.

* Pre-processing:<br>
Resize the input image to the size of `@yolox_shape` and create a Float32 binary sequence normalized to the range {0.0, 255.0}, NCHW, BGR.

* Post-processing:<br>
Split the output tensor f32[8400][85] into class scores and bounding boxes
and sieve the inference results by the score value threshold and NMS.

```elixir
defmodule Candy do
  # use TflInterp, model: Helper.model(), label: Helper.label()
  use TflInterp

  @candy_shape {224, 224}

  def apply(jpeg) do
    img = CImg.from_binary(jpeg)

    # preprocess
    bin = 
      img
      |> CImg.resize(@candy_shape)
      |> CImg.to_binary([{:range, {0.0, 255.0}}, :nchw, :bgr])

    # prediction
    outputs =
      __MODULE__
      |> TflInterp.set_input_tensor(0, bin)
      |> TflInterp.invoke()
      |> TflInterp.get_output_tensor(0)
      |> CImg.from_binary(224, 224, 1, 3, [{:dtype, "<f4"}, {:range, {0.0, 255.0}}, :nchw])
  end
end
```

Launch `Candy`.

```elixir
Candy.start_link(model: Helper.model(), "none")
```

Displays the properties of the `YoloX` model.

```elixir
TflInterp.info(Candy)
```

## 3.Let's try it

In one shot.

```elixir
alias CImg.Builder

draw_object = fn builder, {name, boxes} ->
  Enum.reduce(boxes, builder, fn [_score | box], canvas ->
    [x0, y0, x1, y1] = Enum.map(box, &round(&1))
    CImg.draw_rect(canvas, x0, y0, x1, y1, {255, 0, 0})
    |> CImg.draw_text(x0, y0 - 16, name, 16, :red)
  end)
end

jpeg = Picam.next_frame()

with {:ok, res} <- YoloX.apply(jpeg) do
  # draw result box
  Enum.reduce(Map.to_list(res), Builder.from_binary(jpeg), &draw_object.(&2, &1))
  |> Builder.runit()
else
  _ -> CImg.from_binary(jpeg)
end
|> CImg.resize({640, 480})
|> CImg.to_binary(:jpeg)
|> Kino.Image.new(:jpeg)
```

## 4.TIL ;-)

#### Date: Feb. 6, 2022 / Nerves-livebook rpi3

TflInterp.non_max_suppression_multi_class hangs up. 
Oh well, I forgot that ARM is strict about word alignment.
Solved this problem by adjusting the i/f structure of non_max_suppression_multi_class() to 32-bit word alignment.
This will be fixed in the next version 0.1.4.

Total processing time is about 13.7 seconds, excluding camera shooting.
Of that time, the YoloX inference - TflInterp.invoke(YoloX) - takes about 5.8 seconds,
and the post-processing YoloX.extract_scores/1 takes about 5.2 seconds.
YoloX.extract_scores/1 seems to be taking a long time to calculate Nx.tensor f32[8400][80].

The model I tried this time was too heavy for the Raspberry Pi, so I'll try a smaller model, tiny or nano, next.

&#9633;
