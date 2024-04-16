using DataMining.Lab.Models;

namespace DataMining.Lab.Classifiers;

public class OneRule : IClassifier
{
    // The best feature index to use for classification
    private int bestFeatureIndex;
    // The best rule (dictates the predicted class for each feature value)
    private Dictionary<int, int> bestRule;

    // Constructor that initializes the classifier
    public OneRule()
    {
        bestFeatureIndex = -1;
        bestRule = new Dictionary<int, int>();
    }

    // Method to train the classifier and find the best feature and rule
    public void Train(ClassificationData data)
    {
        // Initialize variables to keep track of the best rule and lowest error rate
        int lowestErrorRate = int.MaxValue;
        bestFeatureIndex = -1;

        // Get the number of features
        int numFeatures = data.TrainingData.First().Features.Count();

        // Iterate through each feature
        for (int i = 0; i < numFeatures; i++)
        {
            // Dictionary to count the classes for each feature value
            var rule = new Dictionary<int, Dictionary<int, int>>();

            // Iterate through the training data
            foreach (var dataTuple in data.TrainingData)
            {
                int featureValue = dataTuple.Features.ElementAt(i);
                int label = dataTuple.Label ?? 0;

                if (!rule.ContainsKey(featureValue))
                {
                    rule[featureValue] = new Dictionary<int, int>();
                }

                if (!rule[featureValue].ContainsKey(label))
                {
                    rule[featureValue][label] = 0;
                }

                // Increment the count for the class of the feature value
                rule[featureValue][label]++;
            }

            // Calculate the error rate for this feature
            int errorRate = 0;
            var ruleForFeature = new Dictionary<int, int>();

            // For each feature value, find the most common class
            foreach (var kvp in rule)
            {
                int featureValue = kvp.Key;
                var classCounts = kvp.Value;

                // Find the class with the highest count
                var mostCommonClass = classCounts.Aggregate((a, b) => a.Value > b.Value ? a : b).Key;

                // Store the most common class for the feature value
                ruleForFeature[featureValue] = mostCommonClass;

                // Calculate the error rate
                errorRate += data.TrainingData.Count(dt => dt.Features.ElementAt(i) == featureValue && dt.Label != mostCommonClass);
            }

            // Update the best rule and feature index if this rule has a lower error rate
            if (errorRate < lowestErrorRate)
            {
                lowestErrorRate = errorRate;
                bestFeatureIndex = i;
                bestRule = ruleForFeature;
            }
        }
    }

    // Method to classify the test data using the trained classifier
    public int Classify(ClassificationData data)
    {
        if (bestFeatureIndex == -1 || bestRule.Count == 0)
        {
            throw new InvalidOperationException("The classifier has not been trained.");
        }

        // Get the test data feature value for the best feature index
        int testFeatureValue = data.TestData.Features.ElementAt(bestFeatureIndex);

        // Classify the test data using the best rule
        if (bestRule.ContainsKey(testFeatureValue))
        {
            return bestRule[testFeatureValue];
        }

        // If the test data feature value is not in the best rule, return the most common class
        return bestRule.Values.GroupBy(v => v).OrderByDescending(g => g.Count()).First().Key;
    }

    public string GetName()
    {
        return "1-Rule";
    }
}
