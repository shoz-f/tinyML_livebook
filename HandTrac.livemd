# Hand Tracking by SSD MobileNet

## 0.Original work

Victor Dibia's "Real-time Hand-Detection using Neural Networks (SSD) on Tensorflow"

 https://github.com/victordibia/handtracking

From his github:
> Both examples above were run on a macbook pro CPU (i7, 2.5GHz, 16GB). Some fps numbers are:
>
> | FPS | Image Size | Device                         | Comments                                                                |
> |----|----------|----------------------------|-----------------------------------------------------|
> |  21 | 320 * 240  | Macbook pro (i7, 2.5GHz, 16GB) | Run without visualizing results                     |
> |  16 | 320 * 240  | Macbook pro (i7, 2.5GHz, 16GB) | Run while visualizing results (image above)         |
> |  11 | 640 * 480  | Macbook pro (i7, 2.5GHz, 16GB) | Run while visualizing results (image above)         |

---
Shubham Panchal's "Hand Detection using TFLite in Android"

https://github.com/shubham0204/Hand_Detection_TFLite_Android


***Thanks Victor and Shubham!!!***

---

## Implementation for Elixir/Nerves using TflInterp

## 1.Helper module
Create the module to assist with tasks such as downloading a model.

```elixir
defmodule Model do
  @model_file "hand_trac.tflite"

  @wearhouse "https://github.com/shoz-f/tinyML_livebook/releases/download/model/#{@model_file}"
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

## 2.Defining the inference module: HandTrac

- Pre-processing:<br>
Resize the input image to the size of `@handtrack_shape` and create a Float32 binary sequence normalized to the range {-1.0, 1.0}.

- Post-processing:<br>
Extract the BBOXes with scores that exceed the threshold `@threshold` from the inference results.

```elixir
defmodule HandTrac do
  # TflInterp, model: Model.file()
  use TflInterp

  @handtrack_shape {300, 300}
  @threshold 0.9

  alias CImg.Builder

  def apply(jpeg) do
    # preprocess
    bin =
      CImg.from_binary(jpeg)
      |> CImg.resize(@handtrack_shape)
      |> CImg.to_binary(range: {-1.0, 1.0})

    # prediction
    __MODULE__
    |> TflInterp.set_input_tensor(0, bin)
    |> TflInterp.invoke()

    [bboxes, scores] =
      for i <- [0, 2] do
        TflInterp.get_output_tensor(__MODULE__, i)
        |> Nx.from_binary({:f, 32})
        |> Nx.reshape({10, :auto})
      end

    # postprocess
    index =
      Nx.to_flat_list(scores)
      |> Enum.with_index()
      |> (&(for {score, index} <- &1, score >= @threshold do index end)).()

    unless index == [] do
      bboxes
      |> Nx.take(Nx.tensor(index))
      |> Nx.to_batched_list(1)
    else
      []
    end
  end

  def draw_result(results, jpeg) do
    Enum.reduce(results, Builder.from_binary(jpeg), fn box, canvas ->
      [y1, x1, y2, x2] = Nx.to_flat_list(box)
      CImg.draw_rect(canvas, x1, y1, x2, y2, {255, 0, 0})
    end)
    |> Builder.runit()
    |> CImg.resize({640, 480})
    |> CImg.to_binary(:jpeg)
  end
end
```

Launch `HandTrac`.

```elixir
HandTrac.start_link(model: Model.file())
```

Displays the properties of the `HandTrac` model.

```elixir
TflInterp.info(HandTrac)
```

## 3.Let's try it

In one shot.
```elixir
img = Picam.next_frame()

HandTrac.apply(img)
|> HandTrac.draw_result(img)
|> Kino.Image.new(:jpeg)
```

In continuous shooting.
```elixir
Kino.animate(10, 0, fn i ->
  img = Picam.next_frame()

  res =
    HandTrac.apply(img)
    |> HandTrac.draw_result(img)
    |> Kino.Image.new(:jpeg)

  {:cont, res, i + 1}
  #:halt
end)
```

## 4.TIL ;-)

#### Date: Feb. 3, 2022 / Nerves-livebook rpi3

It is still not at a practical level.
So far, there seems to be no problem in the post-processing, which is computationally inexpensive.
However, the pre-processing and inference sections, which process large amounts of data, take a lot of processing time.

Problems to be solved:<br>
- Both the inference time(about 3 FPS) and the input data transfer time are too time-consuming.

&#9633;

#### License
Copyright 2022 Shozo Fukuda.
Apache License Version 2.0
