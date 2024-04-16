using DataMining.Lab.Models;

namespace DataMining.Lab.Classifiers;

public class KNearestNeighbors : IClassifier
{
    private int k; // The number of nearest neighbors to consider
    private List<DataTuple> trainingData; // The training data

    // Constructor to initialize the k value
    public KNearestNeighbors(int k)
    {
        this.k = k;
    }

    // Train the classifier using the provided training data
    public void Train(ClassificationData data)
    {
        trainingData = data.TrainingData.ToList();
    }

    // Classify the test data using the trained classifier
    public int Classify(ClassificationData data)
    {
        // Ensure the classifier has been trained
        if (trainingData == null || trainingData.Count == 0)
        {
            throw new InvalidOperationException("The classifier has not been trained.");
        }

        // Calculate the k nearest neighbors of the test instance
        var neighbors = FindKNearestNeighbors(data.TestData);

        // Determine the majority class of the k nearest neighbors
        var majorityClass = DetermineMajorityClass(neighbors);

        // Return the majority class as the classification result
        return majorityClass;
    }

    public string GetName()
    {
        return $"{k} Nearest Neighbors";
    }

    // Find the k nearest neighbors of the test instance
    private List<DataTuple> FindKNearestNeighbors(DataTuple testInstance)
    {
        // Create a list of tuples containing the distance and the data tuple
        var distances = new List<(double distance, DataTuple dataTuple)>();

        // Calculate the Euclidean distance between the test instance and each training data instance
        foreach (var trainingInstance in trainingData)
        {
            double distance = CalculateEuclideanDistance(testInstance.Features, trainingInstance.Features);
            distances.Add((distance, trainingInstance));
        }

        // Sort the list based on the distance in ascending order
        distances.Sort((a, b) => a.distance.CompareTo(b.distance));

        // Select the first k elements from the sorted list
        return distances.Take(k).Select(tuple => tuple.dataTuple).ToList();
    }

    // Calculate the Euclidean distance between two feature vectors
    private double CalculateEuclideanDistance(IEnumerable<int> features1, IEnumerable<int> features2)
    {
        double sumOfSquares = 0;
        var iterator1 = features1.GetEnumerator();
        var iterator2 = features2.GetEnumerator();

        // Calculate the sum of squares of the differences between each feature
        while (iterator1.MoveNext() && iterator2.MoveNext())
        {
            double diff = iterator1.Current - iterator2.Current;
            sumOfSquares += diff * diff;
        }

        // Return the square root of the sum of squares
        return Math.Sqrt(sumOfSquares);
    }

    // Determine the majority class of the k nearest neighbors
    private int DetermineMajorityClass(List<DataTuple> neighbors)
    {
        // Create a dictionary to count the occurrence of each class in the neighbors
        var classCounts = new Dictionary<int, int>();

        // Count the occurrence of each class in the neighbors
        foreach (var neighbor in neighbors)
        {
            int label = neighbor.Label ?? 0;
            if (!classCounts.ContainsKey(label))
            {
                classCounts[label] = 0;
            }
            classCounts[label]++;
        }

        // Find the class with the maximum count
        return classCounts.Aggregate((a, b) => a.Value > b.Value ? a : b).Key;
    }
}
