import { Tensor, InferenceSession } from "onnxjs";
const session = new InferenceSession();
// use the following in an async method
const url = "./model.onnx";
await session.loadModel(url);
// creating an array of input Tensors is the easiest way. For other options see the API documentation
const inputs = [
    new Tensor(new Float32Array([1.0, 2.0, 3.0, 4.0]), "float32", [2, 2]),
  ];
// run this in an async method:
const outputMap = await session.run(inputs);
const outputTensor = outputMap.values().next().value;