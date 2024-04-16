using DataMining.Lab.Classifiers;
using DataMining.Lab.Utility;

var dataList = DataReader.ReadDataJson("ClassificationData.json");

var classifiers = new List<IClassifier>()
{
    new OneRule(),
    new NaiveBayes(),
    new DecisionTree(),
    new KNearestNeighbors(1),
    new KNearestNeighbors(4)
};

foreach (var classifier in classifiers)
{
    Console.WriteLine(classifier.GetName());
    foreach (var data in dataList)
    {
        classifier.Train(data);

        Console.WriteLine(classifier.Classify(data));
    }
}
