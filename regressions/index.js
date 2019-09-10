require("@tensorflow/tfjs-node");
const loadCSV = require("./load-csv");
const LinearRegression = require("./linear-regression");
const plot = require("node-remote-plot");

const { features, labels, testFeatures, testLabels } = loadCSV("./cars.csv", {
  shuffle: true,
  splitTest: 50,
  dataColumns: ["horsepower", "weight", "displacement"],
  labelColumns: ["mpg"]
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10
});

regression.train();

// console.log(
//   "Updated M is ",
//   regression.weights.get(1, 0),
//   "Updated B is",
//   regression.weights.get(0, 0)
// );

// plot({
//   x: regression.mseHistory.reverse(),
//   xLabel: "Iteration #",
//   yLabel: "Mean Squared Error"
// });

//console.log(regression.mseHistory);

const r2 = regression.test(testFeatures, testLabels);

console.log("R2 is", r2);

regression
  .predict([[130, 1.752, 12], [88, 1.065, 14.5], [170, 2.327, 13]])
  .print();
