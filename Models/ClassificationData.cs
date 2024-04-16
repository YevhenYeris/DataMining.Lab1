using System.Text.Json.Serialization;

namespace DataMining.Lab.Models;

public class ClassificationData
{
    [JsonPropertyName("training_data")]
    public required IEnumerable<DataTuple> TrainingData { get; set; }

    [JsonPropertyName("test_data")]
    public required DataTuple TestData { get; set; }
}
