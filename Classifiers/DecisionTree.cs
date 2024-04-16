using DataMining.Lab.Models;

namespace DataMining.Lab.Classifiers;

public class DecisionTree : IClassifier
{
    private DecisionTreeNode root; // Root node of the decision tree

    // Train the decision tree using the provided training data
    public void Train(ClassificationData data)
    {
        // Train the decision tree using the training data
        root = BuildTree(data.TrainingData.ToList(), new HashSet<int>());
    }

    // Classify the test data using the trained decision tree
    public int Classify(ClassificationData data)
    {
        // Ensure the decision tree has been trained
        if (root == null)
        {
            throw new InvalidOperationException("The classifier has not been trained.");
        }

        // Traverse the decision tree to classify the test data
        return root.Classify(data.TestData.Features.ToList());
    }

    public string GetName()
    {
        return "Decision Tree";
    }

    // Build the decision tree recursively
    private DecisionTreeNode BuildTree(List<DataTuple> data, HashSet<int> usedFeatures)
    {
        // Check if data is empty or all labels are the same
        if (data.Count == 0 || data.All(d => d.Label == data[0].Label))
        {
            return new DecisionTreeNode
            {
                IsLeaf = true,
                Label = 1
            };
        }

        // Calculate the best split feature using a criterion such as Gini impurity
        int bestFeatureIndex = FindBestSplitFeature(data, usedFeatures);

        // If no feature provides a good split, create a leaf node
        if (bestFeatureIndex == -1)
        {
            return new DecisionTreeNode
            {
                IsLeaf = true,
                Label = data.GroupBy(d => d.Label).OrderByDescending(g => g.Count()).First().Key
            };
        }

        // Split the data based on the best feature
        var (leftData, rightData) = SplitData(data, bestFeatureIndex);

        // Create a decision node with the best feature and its split
        var node = new DecisionTreeNode
        {
            FeatureIndex = bestFeatureIndex,
            LeftChild = BuildTree(leftData, new HashSet<int>(usedFeatures) { bestFeatureIndex }),
            RightChild = BuildTree(rightData, new HashSet<int>(usedFeatures) { bestFeatureIndex })
        };

        return node;
    }

    // Find the best split feature using a criterion such as Gini impurity
    private int FindBestSplitFeature(List<DataTuple> data, HashSet<int> usedFeatures)
    {
        int numFeatures = data.First().Features.Count();
        double bestCriterion = double.MaxValue;
        int bestFeatureIndex = -1;

        // Iterate through each feature
        for (int i = 0; i < numFeatures; i++)
        {
            // Skip already used features
            if (usedFeatures.Contains(i))
            {
                continue;
            }

            // Calculate the criterion for the feature
            double criterion = CalculateGiniImpurity(data, i);

            // Update the best feature index if the criterion is better
            if (criterion < bestCriterion)
            {
                bestCriterion = criterion;
                bestFeatureIndex = i;
            }
        }

        return bestFeatureIndex;
    }

    // Calculate the Gini impurity for a specific feature
    private double CalculateGiniImpurity(List<DataTuple> data, int featureIndex)
    {
        var featureValues = data.GroupBy(d => d.Features.ElementAt(featureIndex));
        double impurity = 0;

        // Calculate the impurity for each group of feature values
        foreach (var group in featureValues)
        {
            double groupSize = group.Count();
            double groupImpurity = 1;

            // Calculate the Gini impurity for the group
            var labelCounts = group.GroupBy(d => d.Label);
            foreach (var labelGroup in labelCounts)
            {
                double labelProbability = labelGroup.Count() / groupSize;
                groupImpurity -= labelProbability * labelProbability;
            }

            impurity += (groupSize / data.Count) * groupImpurity;
        }

        return impurity;
    }

    // Split the data into two subsets based on the feature index
    private (List<DataTuple> leftData, List<DataTuple> rightData) SplitData(List<DataTuple> data, int featureIndex)
    {
        var leftData = new List<DataTuple>();
        var rightData = new List<DataTuple>();

        // Iterate through the data and split it based on the feature value
        foreach (var dataTuple in data)
        {
            int featureValue = dataTuple.Features.ElementAt(featureIndex);

            // Split the data into left and right based on the feature value
            if (featureValue == 0)
            {
                leftData.Add(dataTuple);
            }
            else
            {
                rightData.Add(dataTuple);
            }
        }

        return (leftData, rightData);
    }
}

// Decision tree node class
public class DecisionTreeNode
{
    public bool IsLeaf { get; set; } // Indicates whether the node is a leaf node
    public int? Label { get; set; } // The class label (for leaf nodes)
    public int FeatureIndex { get; set; } // The feature index (for decision nodes)
    public DecisionTreeNode LeftChild { get; set; } // Left child node
    public DecisionTreeNode RightChild { get; set; } // Right child node

    // Classify the features based on the decision tree node
    public int Classify(List<int> features)
    {
        // If the node is a leaf node, return the label
        if (IsLeaf)
        {
            return Label ?? 0;
        }

        // Decide which child node to traverse based on the feature value
        int featureValue = features.ElementAt(FeatureIndex);
        return featureValue == 0 ? LeftChild.Classify(features) : RightChild.Classify(features);
    }
}
