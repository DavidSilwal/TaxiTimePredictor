using System;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MLdotNET
{
    class Program
    {
        const string DataPath     = @".\Data\train.csv";
        const string TestDataPath = @".\Data\test.csv";
        const string ModelPath    = @".\Models\Model.zip";

        public static async Task Main()
        {
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = await Train();
            Evaluate(model);

            var prediction = model.Predict(TestTrips.Trip1);

            Console.WriteLine("Predicted fare: {0}, actual fare: 29.5", prediction.fare_amount);

            Console.ReadLine();

        }
        public static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> Train()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader<TaxiTrip>(DataPath, useHeader: true, separator: ","));
            pipeline.Add(new ColumnCopier(("fare_amount", "Label")));

            pipeline.Add(new CategoricalOneHotVectorizer("vendor_id",
                                                         "rate_code",
                                                         "payment_type"));

            pipeline.Add(new ColumnConcatenator("Features",
                                                "vendor_id",
                                                "rate_code",
                                                "passenger_count",
                                                "trip_distance",
                                                "payment_type"));

            pipeline.Add(new FastTreeRegressor());

            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();

            await model.WriteAsync(ModelPath);

            return model;
        }

        public static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var testData = new TextLoader<TaxiTrip>(TestDataPath, useHeader: true, separator: ",");

            var               evaluator = new RegressionEvaluator();
            RegressionMetrics metrics   = evaluator.Evaluate(model, testData);
            // Rms should be around 2.795276
            Console.WriteLine("Rms=" + metrics.Rms);

            Console.WriteLine("RSquared = " + metrics.RSquared);
        }

        static class TestTrips
        {
            internal static readonly TaxiTrip Trip1 = new TaxiTrip
                                                      {
                                                          vendor_id       = "VTS",
                                                          rate_code       = "1",
                                                          passenger_count = 1,
                                                          trip_distance   = 10.33f,
                                                          payment_type    = "CSH",
                                                          fare_amount     = 0 // predict it. actual = 29.5
                                                      };
        }
        
    }
}
