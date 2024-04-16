using System.Text.Json;
using DataMining.Lab.Models;

namespace DataMining.Lab.Utility;

public static class DataReader
{
    public static IEnumerable<ClassificationData> ReadDataJson(string pathToJson)
    {
        pathToJson = Path.GetFullPath(pathToJson);

        var json = File.ReadAllText(pathToJson);

        return JsonSerializer.Deserialize<IEnumerable<ClassificationData>>(json) ?? Enumerable.Empty<ClassificationData>();
    }
}
