const outputs = [];

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

// function runAnalysis() {
//   // Write code here to analyze stuff
//   const testSetSize = 100;
//   const [testSet, trainingSet] = splitDataset(outputs, testSetSize);
//   _.range(1, 20).forEach(k => {
//     const accuracy = _.chain(testSet)
//       .filter(
//         testPoint => knn(trainingSet, _.initial(testPoint), k) === testPoint[3]
//       )
//       .size()
//       .divide(testSetSize)
//       .value();

//     console.log("k value is ", k, " Accuracy is: ", accuracy);
//   });
// }

function distance(pointA, pointB) {
  return (
    _.chain(pointA)
      .zip(pointB)
      .map(([a, b]) => (a - b) ** 2)
      .sum()
      .value() ** 0.5
  );
}

function runAnalysis() {
  const testSetSize = 50;
  //We fix the value of k for feature selection to select the most relevant subset of feature, feature that has higher predictive power
  const k = 10;

  _.range(0, 3).forEach(feature => {
    //feature == 0, feature == 1 , feature == 2
    const data = _.map(outputs, row => [row[feature], _.last(row)]);
    const [testSet, trainingSet] = splitDataset(minMax(data, 1), testSetSize);
    const accuracy = _.chain(testSet)
      .filter(testPoint => {
        return knn(trainingSet, _.initial(testPoint), k) == _.last(testPoint);
      })
      .size()
      .divide(testSetSize)
      .value();
    console.log("For feature of", feature, "accuracy is", accuracy);
  });
}

function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data);

  const testSet = _.slice(shuffled, 0, testCount);

  const trainingSet = _.slice(shuffled, testCount);

  return [testSet, trainingSet];
}

function knn(data, point, k) {
  return _.chain(data)
    .map(row => {
      return [distance(_.initial(row), point), _.last(row)];
    })
    .sortBy(row => row[0])
    .slice(0, k)
    .countBy(row => row[1])
    .toPairs()
    .sortBy(row => row[1])
    .last()
    .first()
    .parseInt()
    .value();
}

function minMax(data, featureCount) {
  const clonedData = _.cloneDeep(data);

  for (let i = 0; i < featureCount; i++) {
    const column = clonedData.map(row => row[i]);
    const min = _.min(column);
    const max = _.max(column);

    for (let j = 0; j < clonedData.length; j++) {
      clonedData[j][i] = (clonedData[j][i] - min) / (max - min);
    }
  }

  return clonedData;
}
