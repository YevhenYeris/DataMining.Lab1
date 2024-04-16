using System.Text.Json.Serialization;

namespace DataMining.Lab.Models;

public class DataTuple
{
    [JsonPropertyName("Q")]
    public required IEnumerable<int> Features { get; set; }

    [JsonPropertyName("S")]
    public int? Label { get; set; }
}
