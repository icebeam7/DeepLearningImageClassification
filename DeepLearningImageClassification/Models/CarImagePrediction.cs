using System;

namespace DeepLearningImageClassification.Models
{
    public class CarImagePrediction
    {
        public string ImagePath { get; set; }

        public string DamageClass { get; set; }

        public string PredictedLabel { get; set; }
    }
}
