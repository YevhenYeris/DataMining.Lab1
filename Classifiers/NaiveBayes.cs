using DataMining.Lab.Models;

namespace DataMining.Lab.Classifiers;

public class NaiveBayes : IClassifier
{
    // Dictionaries to store class priors and feature conditional probabilities
    private Dictionary<int, double> classPriors;
    private Dictionary<int, Dictionary<int, Dictionary<int, double>>> featureConditionals;

    // Constructor to initialize dictionaries
    public NaiveBayes()
    {
        classPriors = new Dictionary<int, double>();
        featureConditionals = new Dictionary<int, Dictionary<int, Dictionary<int, double>>>();
    }

    // Method to train the classifier on the provided training data
    public void Train(ClassificationData data)
    {
        // Initialize dictionaries
        classPriors.Clear();
        featureConditionals.Clear();

        // Count class occurrences and feature occurrences given each class
        var classCounts = new Dictionary<int, int>();
        var featureCounts = new Dictionary<int, Dictionary<int, Dictionary<int, int>>>();

        // Iterate through the training data
        foreach (var dataTuple in data.TrainingData)
        {
            int label = dataTuple.Label ?? 0;

            // Update class counts
            if (!classCounts.ContainsKey(label))
            {
                classCounts[label] = 0;
            }
            classCounts[label]++;

            // Iterate through each feature
            int featureIndex = 0;
            foreach (var featureValue in dataTuple.Features)
            {
                // Update feature counts given class
                if (!featureCounts.ContainsKey(label))
                {
                    featureCounts[label] = new Dictionary<int, Dictionary<int, int>>();
                }

                if (!featureCounts[label].ContainsKey(featureIndex))
                {
                    featureCounts[label][featureIndex] = new Dictionary<int, int>();
                }

                if (!featureCounts[label][featureIndex].ContainsKey(featureValue))
                {
                    featureCounts[label][featureIndex][featureValue] = 0;
                }

                featureCounts[label][featureIndex][featureValue]++;
                featureIndex++;
            }
        }

        // Calculate class priors
        int totalSamples = data.TrainingData.Count();
        foreach (var kvp in classCounts)
        {
            int label = kvp.Key;
            int count = kvp.Value;
            classPriors[label] = (double)count / totalSamples;
        }

        // Calculate conditional probabilities for each feature given each class
        int numFeatures = data.TrainingData.First().Features.Count();
        foreach (var kvp in featureCounts)
        {
            int label = kvp.Key;
            var features = kvp.Value;

            if (!featureConditionals.ContainsKey(label))
            {
                featureConditionals[label] = new Dictionary<int, Dictionary<int, double>>();
            }

            foreach (var featureKvp in features)
            {
                int featureIndex = featureKvp.Key;
                var featureValues = featureKvp.Value;

                if (!featureConditionals[label].ContainsKey(featureIndex))
                {
                    featureConditionals[label][featureIndex] = new Dictionary<int, double>();
                }

                int totalFeatureCount = featureValues.Values.Sum();
                foreach (var valueKvp in featureValues)
                {
                    int featureValue = valueKvp.Key;
                    int count = valueKvp.Value;

                    // Calculate conditional probability for the feature value given the class
                    featureConditionals[label][featureIndex][featureValue] = (double)count / totalFeatureCount;
                }
            }
        }
    }

    // Method to classify the test data using the trained classifier
    public int Classify(ClassificationData data)
    {
        // Ensure classifier has been trained
        if (classPriors.Count == 0 || featureConditionals.Count == 0)
        {
            throw new InvalidOperationException("The classifier has not been trained.");
        }

        // Dictionary to store the posterior probabilities for each class
        var posteriors = new Dictionary<int, double>();

        // Calculate the posterior probability for each class
        foreach (var kvp in classPriors)
        {
            int label = kvp.Key;
            double priorProbability = kvp.Value;

            // Initialize the posterior with the prior probability
            double posteriorProbability = Math.Log(priorProbability);

            // Multiply conditional probabilities for each feature
            int featureIndex = 0;
            foreach (var featureValue in data.TestData.Features)
            {
                if (featureConditionals[label].ContainsKey(featureIndex) &&
                    featureConditionals[label][featureIndex].ContainsKey(featureValue))
                {
                    // Multiply the conditional probability
                    double conditionalProbability = featureConditionals[label][featureIndex][featureValue];
                    posteriorProbability += Math.Log(conditionalProbability);
                }
                else
                {
                    // Apply Laplace smoothing if the feature value is not in the dictionary
                    // This is done to avoid multiplying by zero
                    posteriorProbability += Math.Log(1.0 / (classPriors.Count + 1));
                }

                featureIndex++;
            }

            // Store the posterior probability for the current class
            posteriors[label] = posteriorProbability;
        }

        // Return the class with the highest posterior probability
        return posteriors.Aggregate((a, b) => a.Value > b.Value ? a : b).Key;
    }

    public string GetName()
    {
        return "Naive Bayes";
    }
}
