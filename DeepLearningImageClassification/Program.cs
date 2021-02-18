using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

using Microsoft.ML;
using Microsoft.ML.Vision;

using DeepLearningImageClassification.Models;

namespace DeepLearningImageClassification
{
    class Program
    {
        private static string basePath = "/Users/luisbeltran/Projects/comunidadai/";
        private static string dataPath = basePath + "preprocessed/index.csv";
        private static string imagesPath = basePath + "preprocessed/image";
        private static string testImagesPath = basePath + "car_test";
        private static string modelPath = basePath + "ImageMLModelv2.zip";
        private static string logPath = basePath + "log.txt";

        static IEnumerable<CarImage> LoadData(string path, string images)
        {
            Func<string, string> getfullPath = (localPath) =>
                Path.Combine(images, localPath.Split('/')[1]);

            var data = File.ReadAllLines(path)
                .Skip(1)
                .Select(line =>
                {
                    var columns = line.Split(',');
                    return new CarImage()
                    {
                        ImagePath = getfullPath(columns[0]),
                        DamageClass = columns[1],
                        Subset = columns[2]
                    };
                });

            return data;
        }

        static void PrintMessage(string message)
        {
            var messageTime = $"{DateTime.Now.ToString("hh.mm.ss.ffffff")} -> {message}...";
            Console.WriteLine(messageTime);

            using (var sw = File.AppendText(logPath))
            {
                sw.WriteLine(messageTime);
            }
        }

        static void Main(string[] args)
        {
            var context = new MLContext();
            
            PrintMessage("Loading data");
            var data = LoadData(dataPath, imagesPath);

            var trainingImages = data.Where(x => x.Subset == "T");
            var validationImages = data.Where(x => x.Subset == "V");

            var trainingImagesDataView = context.Data.LoadFromEnumerable(trainingImages);
            var validationImagesDataView = context.Data.LoadFromEnumerable(validationImages);

            var loadPipeline = context.Transforms.LoadRawImageBytes(
                outputColumnName: "ImageBytes",
                imageFolder: null,
                inputColumnName: "ImagePath");

            var trainingOptions = new ImageClassificationTrainer.Options
            {
                FeatureColumnName = "ImageBytes",
                LabelColumnName = "EncodedLabel",
                WorkspacePath = "workspace",
                Arch = ImageClassificationTrainer.Architecture.InceptionV3,
                ReuseTrainSetBottleneckCachedValues = true,
                MetricsCallback = (metrics) => Console.WriteLine(metrics.ToString())
            };

            var trainingPipeline = context.Transforms
                .Conversion.MapValueToKey(outputColumnName: "EncodedLabel",
                                        inputColumnName: "DamageClass")
                    .Append(context.MulticlassClassification.Trainers.ImageClassification(trainingOptions))
                    .Append(context.Transforms.Conversion
                        .MapKeyToValue(outputColumnName: "PredictedLabel",
                                        inputColumnName: "PredictedLabel"));

            var fullPipeline = loadPipeline.Append(trainingPipeline);
            PrintMessage("Training starts.");

            var model = fullPipeline.Fit(trainingImagesDataView);

            PrintMessage("Training completed.");

            PrintMessage("Validating model.");
            var predictionsDataView = model.Transform(validationImagesDataView);

            Console.WriteLine("Metrics:");
            var evaluation = context.MulticlassClassification.Evaluate(
                predictionsDataView,
                labelColumnName: "EncodedLabel");

            Console.WriteLine($"  * Macro Accuracy: {evaluation.MacroAccuracy}");
            PrintMessage("Validation completed.");

            PrintMessage("Image classification #1.");
            var predictions = context.Data.CreateEnumerable<CarImagePrediction>(
                predictionsDataView,
                reuseRowObject: true);

            foreach (var item in predictions)
            {
                var image = Path.GetFileName(item.ImagePath);
                PrintMessage($"* Image: {image} | Actual damage: {item.DamageClass} | Predicted: {item.PredictedLabel}");
            }

            PrintMessage("Saving model");
            context.Model.Save(model, trainingImagesDataView.Schema, modelPath);

            PrintMessage("Model saved");
            */
            PrintMessage("Classifying several images");
            ConsumingModel();
        }

        private static void ConsumingModel()
        {
            DataViewSchema dataViewSchema;
            var context = new MLContext();

            PrintMessage("Loading model");
            var model = context.Model.Load(modelPath, out dataViewSchema);

            var data = new List<CarImage>();

            for (int i = 1; i < 5; i++)
                data.Add(new CarImage() {
                    ImagePath = $"{testImagesPath}/car{i}.jpg"
                });

            var imagesDataView = context.Data.LoadFromEnumerable(data);
            PrintMessage("Model loaded");

            ClassifyImages(context, imagesDataView, model);
        }

        private static void ClassifyImages(MLContext context, IDataView data, ITransformer model)
        {
            var predictionEngine = context.Model
                .CreatePredictionEngine<CarImage, CarImagePrediction>(model);
            var images = context.Data.CreateEnumerable<CarImage>(data, reuseRowObject: true);

            PrintMessage("Classifying images");

            foreach (var item in images)
            {
                var prediction = predictionEngine.Predict(item);

                var image = Path.GetFileName(prediction.ImagePath);
                PrintMessage($"* Image: {image} | Predicted damage: {prediction.PredictedLabel}");
            }
        }
    }
}