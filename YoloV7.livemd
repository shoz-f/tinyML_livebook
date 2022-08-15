# Object detection by YOLOv7

```elixir
Mix.install([
  {:nx, "~> 0.2.1"},
  {:kino, "~> 0.6.2"},
  {:onnx_interp, github: "shoz-f/onnx_interp"},
  {:cimg, github: "shoz-f/cimg_ex"}
])
```

## 0.Original work

Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao<br>
"YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"

* https://arxiv.org/abs/2207.02696
* https://github.com/WongKinYiu/yolov7

Ibai Gorordo (@ibai_gorordo)<br>
"YOLOv7 ONNX Converson(Google Colab)"

* https://colab.research.google.com/drive/1733xwaETLhAJguRKDqhjgPWf7a2NaOro?usp=sharing

***Thanks a lot!!!***

---

## Implementation with OnnxInterp in Elixir

### 1.Prepare the onnx model

Use Ibai's jupyter notebook (URL above) to get the converted YOLOv7 onnx model from the Pytorch model.
You put the model into the livebook home directory.  And also you download the coco.label file and put it in the livebook directory.

```shell
> cd livebook
> cp down-load-directory/yolov7.onnx .
> wget https://raw.githubusercontent.com/shoz-f/onnx_interp/main/demo_yolo7/coco.label
```

### 2.Defining the inference module: DemoYolo7

* Model<br>
  Standard Model: YOLOv7.onnx converted from Pytorch model.

* Pre-processing:<br>
  Resize the input image to the size of `@yolo7_shape` and create a Float32 binary sequence normalized to the range {0.0, 1.0}, NCHW.

* Post-processing:<br>
  Split the output tensor f32[18900][85] into class scores and bounding boxes
and sieve the inference results by the score value threshold and NMS.

```elixir
defmodule DemoYolo7 do
  # use OnnxInterp, model: Helper.model(), label: Helper.label()
  use OnnxInterp, model: "./yolov7.onnx", label: "./coco.label"

  @yolo7_shape {640, 480}

  def apply(img) do
    # preprocess
    bin = img
      |> CImg.resize(@yolo7_shape)
      |> CImg.to_binary([{:range, {0.0, 1.0}}, :nchw])

    # prediction
    outputs =
      __MODULE__
      |> OnnxInterp.set_input_tensor(0, bin)
      |> OnnxInterp.invoke()
      |> OnnxInterp.get_output_tensor(0)
      |> Nx.from_binary({:f, 32}) |> Nx.reshape({:auto, 85})

    # postprocess
    boxes  = extract_boxes(outputs, scale(img))
    scores = extract_scores(outputs)

    OnnxInterp.non_max_suppression_multi_class(__MODULE__,
      Nx.shape(scores), Nx.to_binary(boxes), Nx.to_binary(scores)
    )
  end

  defp extract_boxes(tensor, scale) do
    Nx.slice_along_axis(tensor, 0, 4, axis: 1) |> Nx.multiply(scale)
  end

  defp extract_scores(tensor) do
    Nx.multiply(Nx.slice_along_axis(tensor, 4, 1, axis: 1), Nx.slice_along_axis(tensor, 5, 80, axis: 1))
  end
  
  defp scale(img) do
    {w, h, _, _}   = CImg.shape(img)
    {wsize, hsize} = @yolo7_shape
    max(w/wsize, h/hsize)
  end
end
```

Launch `DemoYolo7`.

```elixir
DemoYolo7.start_link([])
```

Displays the properties of the `YOLOv7` model.

```elixir
OnnxInterp.info(DemoYolo7)
```

## 3.Let's try it

```elixir
draw_object = fn builder, {name, boxes} ->
  Enum.reduce(boxes, builder, fn [_score | box], canvas ->
    [x0, y0, x1, y1] = Enum.map(box, &round(&1))

    CImg.draw_rect(canvas, x0, y0, x1, y1, {255, 0, 0})
    |> CImg.draw_text(x0, y0 - 16, name, 16, :red)
  end)
end

img = CImg.load("dog.jpg")

with {:ok, res} <- DemoYolo7.apply(img) do
  # draw result box
  Enum.reduce(Map.to_list(res), CImg.builder(img), &draw_object.(&2, &1))
  |> CImg.run()
else
  _ -> img
end
|> CImg.resize({640, 480})
|> CImg.display_kino(:jpeg)
```

## 4.TIL ;-)

&#9633;
