# Object detection by YOLOX

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
  @model_file "yolox_s.tflite"
  @label_file "coco.label"

  @wearhouse "https://github.com/shoz-f/tinyML_livebook/releases/download/model/"
  @local "/data/"

  def model() do
    @local <> @model_file
  end

  def label() do
    @local <> @label_file
  end

  def get() do
    Req.get!(@wearhouse <> @model_file).body
    |> then(fn x -> File.write(model(), x) end)

    Req.get!(@wearhouse <> @label_file).body
    |> then(fn x -> File.write(label(), x) end)
  end

  def rm(:model), do: File.rm(model())
  def rm(:label), do: File.rm(label())

  def rm() do
    rm(:model)
    rm(:label)
  end

  def exists?(:model), do: File.exists?(model())
  def exists?(:label), do: File.exists?(label())

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

* Model
Standard Model: YOLOX-s 640 converted from Pytorch model.

* Pre-processing:<br>
Resize the input image to the size of `@yolox_shape` and create a Float32 binary sequence normalized to the range {0.0, 255.0}, NCHW, BGR.

* Post-processing:<br>
Split the output tensor f32[8400][85] into class scores and bounding boxes
and sieve the inference results by the score value threshold and NMS.

```elixir
defmodule YoloX do
  # use TflInterp, model: Helper.model(), label: Helper.label()
  use TflInterp

  @yolox_shape {640, 640}

  def apply(jpeg) do
    img = CImg.from_binary(jpeg)

    # preprocess
    bin = 
      img
      |> CImg.resize(@yolox_shape, :ul, 114)
      |> CImg.to_binary([{:range, {0.0, 255.0}}, :nchw, :bgr])

    # prediction
    outputs =
      __MODULE__
      |> TflInterp.set_input_tensor(0, bin)
      |> TflInterp.invoke()
      |> TflInterp.get_output_tensor(0)
      |> Nx.from_binary({:f, 32}) |> Nx.reshape({:auto, 85})

    # postprocess
    boxes  = extract_boxes(outputs, scale(img))
    scores = extract_scores(outputs)

    TflInterp.non_max_suppression_multi_class(__MODULE__,
      Nx.shape(scores), Nx.to_binary(boxes), Nx.to_binary(scores)
    )
  end

  defp extract_boxes(tensor, scale) do
    {grid, strides} = grid_strides(@yolox_shape, [8, 16, 32])

    [
      Nx.add(Nx.slice_axis(tensor, 0, 2, 1), grid),
      Nx.exp(Nx.slice_axis(tensor, 2, 2, 1))
    ]
    |> Nx.concatenate(axis: 1) |> Nx.multiply(strides) |> Nx.multiply(scale)
  end

  defp grid_strides({wsize, hsize}, block) do
    reso = Enum.map(block, fn x -> {div(hsize, x), div(wsize, x), x} end)

    {
      Enum.map(reso, &grid/1)    |> Nx.concatenate(axis: 0),
      Enum.map(reso, &strides/1) |> Nx.concatenate(axis: 0)
    }
  end

  defp grid({hsize, wsize, _}) do
    xv = Nx.iota({wsize}) |> Nx.tile([hsize, 1])
    yv = Nx.iota({hsize}) |> Nx.tile([wsize, 1]) |> Nx.transpose()
    Nx.stack([xv, yv], axis: 2) |> Nx.reshape({:auto, 2})
  end
  
  defp strides({hsize, wsize, stride}) do
    Nx.tensor(stride) |> Nx.tile([hsize*wsize, 1])
  end

  defp extract_scores(tensor) do
    Nx.multiply(Nx.slice_axis(tensor, 4, 1, 1), Nx.slice_axis(tensor, 5, 80, 1))
  end
  
  defp scale(img) do
    {w, h, _, _}   = CImg.shape(img)
    {wsize, hsize} = @yolox_shape
    max(w/wsize, h/hsize)
  end
end
```

Launch `YoloX`.

```elixir
YoloX.start_link(model: Helper.model(), label: Helper.label())
```

Displays the properties of the `YoloX` model.

```elixir
TflInterp.info(YoloX)
```

## 3.Let's try it

In one shot.

```elixir
alias CImg.Builder

draw_object = fn builder, {_name, boxes} ->
  Enum.reduce(boxes, builder, fn [_score | box], canvas ->
    [x0, y0, x1, y1] = Enum.map(box, &round(&1))
    CImg.draw_rect(canvas, x0, y0, x1, y1, {255, 0, 0})
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
