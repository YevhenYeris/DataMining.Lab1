using DataMining.Lab.Models;

namespace DataMining.Lab.Classifiers;

public interface IClassifier
{
    void Train(ClassificationData data);

    int Classify(ClassificationData data);

    string GetName();
}
