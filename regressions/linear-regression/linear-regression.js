const tf = require("@tensorflow/tfjs");
const _ = require("lodash");

//gradientDescent() ： Run one iteration of GD and update m and b
//train: run GD until we get good values for m and b
//test: Use test data set to evaluate accuracy of our calculated m and b
//predict: Make a prediction using our calculated m and b

//We reuse variance and mean of features for test features, we reapplying the mean and variance of features onto test features

class LinearRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.mseHistory = [];
    this.bHistory = [];

    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options
    );

    this.weights = tf.zeros([this.features.shape[1], 1]);
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights);
    const differences = currentGuesses.sub(labels);

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    //Find the number of batches
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize
    );

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * this.options.batchSize;
        const { batchSize } = this.options;

        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1]
        );

        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1]);
        this.gradientDescent(featureSlice, labelSlice);
      }
      this.bHistory.push(this.weights.get(0, 0));
      this.recordMSE();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations).matMul(this.weights);
  }

  //Calculate the accuracy of the model by determining coeeficients of determination
  test(testFeatures, testLabels) {
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);

    const predictions = testFeatures.matMul(this.weights);

    const res = testLabels
      .sub(predictions)
      .pow(2)
      .sum() //this return a tensor
      .get(); // this return an integer

    const tot = testLabels
      .sub(testLabels.mean())
      .pow(2)
      .sum()
      .get();

    return 1 - res / tot;
  }

  processFeatures(features) {
    features = tf.tensor(features);

    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }

    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  // standardize(features) {
  //   const { mean, variance } = tf.moments(features, 0);
  //   this.mean = mean;
  //   this.variance = variance;

  //   return features.sub(mean).div(variance.pow(0.5));
  // }

  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);
    this.mean = mean;
    this.variance = variance.add(1e-7);

    return features.sub(mean).div(variance.add(1e-7));
  }

  recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .get();

    this.mseHistory.unshift(mse);
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) {
      return;
    }

    if (this.mseHistory[0] > this.mseHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LinearRegression;

// gradientDescent() {
//   const currentGuessesForMPG = this.features.map(row => {
//     return this.m * row[0] + this.b;
//   });

//   const bSlope =
//     (_.sum(
//       currentGuessesForMPG.map((guess, i) => {
//         return guess - this.labels[i][0];
//       })
//     ) *
//       2) /
//     this.features.length;

//   const mSlope =
//     (_.sum(
//       currentGuessesForMPG.map((guess, i) => {
//         return -1 * this.features[i][0] * (this.labels[i][0] - guess);
//       })
//     ) *
//       2) /
//     this.features.length;

//   this.m = this.m - mSlope * this.options.learningRate;
//   this.b = this.b - bSlope * this.options.learningRate;
// }
