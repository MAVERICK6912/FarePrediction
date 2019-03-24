using System;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

namespace MachineLearningAlpha
{
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _fareModelPath= Path.Combine(Environment.CurrentDirectory, "Data", "model_taxi_fare.zip");
        static readonly string _distanceModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "model_taxi_distance.zip");
        static readonly string _timeModelPath = Path.Combine(Environment.CurrentDirectory, "Data", "model_taxi_time.zip");



        static void Main(string[] args) {
            MLContext mLContext = new MLContext(seed: 0);
            var model = TrainForFare(mLContext, _trainDataPath);
            GetUserChoice(mLContext,model);
            Console.ReadLine();
        }

        private static void GetUserChoice(MLContext mLContext,ITransformer model)
        {
            Console.WriteLine("Enter 1. for model evaluation.");
            Console.WriteLine("Enter 2. for testing model with user input.");
            Console.WriteLine("Enter 3. for testing model using predefined values.");
            Console.WriteLine("Enter 4. to train model for distance prediction.");
            Console.WriteLine("Enter 5. to train model for journey time prediction.");
            Console.WriteLine("Enter 6. to exit.");
            int userChoice = int.Parse(Console.ReadLine());
            if (userChoice==1)
            {
                Evaluate(mLContext, model,1);
            }
            else if (userChoice==2)
            {
                getPredictionForUserInput(mLContext,model);
            }
            else if (userChoice==3)
            {
                TestSinglePrediction(mLContext);
            }
            else if (userChoice==4)
            {
                GetUserChoice(mLContext, TrainForDistance(mLContext, _trainDataPath));
            }
            else if (userChoice==5)
            {
                GetUserChoice(mLContext,TrainForTime(mLContext, _trainDataPath));
            }
            else if (userChoice==6)
            {
                Console.WriteLine("Closing connections.....");
                Console.WriteLine("Connections closed.");
                Console.WriteLine("Bye!!");
                Console.WriteLine("Press enter to exit.");
                Console.ReadLine();
                System.Environment.Exit(0);
            }
            else
            {
                Console.WriteLine("Invalid choice.");
                GetUserChoice(mLContext, model);
            }
        }

        private static void getPredictionForUserInput(MLContext mLContext, ITransformer model)
        {
            Console.WriteLine("Press 1. to get prediction based on distance to travel");
            Console.WriteLine("Press 2. to get prediction based on time of travel");
            Console.WriteLine("Press 3. to get prediction based on number of passengers.");
            int choice = int.Parse(Console.ReadLine());
            if (choice==1)
            {
                Console.WriteLine("Enter distance you want to travel.");
                float dist = float.Parse(Console.ReadLine());
                TestSinglePrediction(mLContext, new TaxiTrip()
                {
                    PassengerCount=1,
                    PaymentType="CRD",
                    RateCode="1",
                    TripAmount=0,
                    TripDistance=dist,
                    TripTime=1440,      //might have to predict this too for better results.
                    VendorID="VTS"
                });
            }
            else if (choice==2)
            {
                Console.WriteLine("Enter time in minutes.");
                float sec = float.Parse(Console.ReadLine())*60;
                TestSinglePrediction(mLContext, new TaxiTrip()
                {
                    PassengerCount = 1,
                    PaymentType = "CRD",
                    RateCode = "1",
                    TripAmount = 0,
                    TripDistance = 5,       //might have to predict this too for better results.
                    TripTime = sec,      
                    VendorID = "VTS"
                });
            }
            else if (choice==3)
            {
                Console.WriteLine("Enter number of passengers");
                float passengers = float.Parse(Console.ReadLine());
                TestSinglePrediction(mLContext, new TaxiTrip()
                {
                    PassengerCount = passengers,
                    PaymentType = "CRD",
                    RateCode = "1",
                    TripAmount = 0,
                    TripDistance = 5,
                    TripTime = 1440,      //might have to predict this too for better results.
                    VendorID = "VTS"
                });
            }
            else
            {
                Console.WriteLine("Invalid input");
                GetUserChoice(mLContext,model);
            }
        }

        private static ITransformer TrainForFare(MLContext mLContext, string trainDataPath)
        {
            IDataView dataView = mLContext.Data.LoadFromTextFile<TaxiTrip>(trainDataPath, hasHeader: true, separatorChar: ',');
            var pipeline = mLContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "TripAmount")
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName:"VendorIDEncoded",inputColumnName: "VendorID"))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName:"RateCodeEncoded",inputColumnName:"RateCode"))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName:"PaymentTypeEncoded", inputColumnName:"PaymentType"))
                .Append(mLContext.Transforms.Concatenate("Features","VendorIDEncoded","RateCodeEncoded","PassengerCount","TripTime","TripDistance","PaymentTypeEncoded"))
                .Append(mLContext.Regression.Trainers.FastTree());
            Console.WriteLine("Training model for fare prediction........");
            var model = pipeline.Fit(dataView);
            Console.WriteLine("Model trained.");
            SaveModelAsFile(mLContext, model,1);
            return model;
        }
        private static ITransformer TrainForDistance(MLContext mLContext, string trainDataPath)
        {
            IDataView dataView = mLContext.Data.LoadFromTextFile<DistanceTimeTrainer>(trainDataPath, separatorChar: ',', hasHeader: true);
            var pipeline = mLContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "TripDistance")
                .Append(mLContext.Transforms.Concatenate("Features", "TripTime"))
                .Append(mLContext.Regression.Trainers.FastTree());
            Console.WriteLine("Training model for distance prediction........");
            var model = pipeline.Fit(dataView);
            Console.WriteLine("Model trained.");
            SaveModelAsFile(mLContext, model,2);
            Evaluate(mLContext, model,2);

            return model;
        }
        private static ITransformer TrainForTime(MLContext mLContext, string trainDataPath)
        {
            IDataView dataView = mLContext.Data.LoadFromTextFile<DistanceTimeTrainer>(trainDataPath, separatorChar: ',', hasHeader: true);
            var pipeline = mLContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "TripTime")
                .Append(mLContext.Transforms.Concatenate("Features", "TripDistance"))
                .Append(mLContext.Regression.Trainers.FastTree());
            Console.WriteLine("Training model for time prediction........");
            var model = pipeline.Fit(dataView);
            Console.WriteLine("Model trained.");
            SaveModelAsFile(mLContext, model, 3);
            Evaluate(mLContext, model,3);
            return model;
        }

        private static void SaveModelAsFile(MLContext mLContext, ITransformer model,int trainBit)
        {
            string modelPath="";
            if (trainBit==1)
            {
                modelPath = modelPath+_fareModelPath;
            }
            else if (trainBit==2)
            {
                modelPath = modelPath + _distanceModelPath;
            }
            else if (trainBit==3)
            {
                modelPath = modelPath + _timeModelPath;
            }
            using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mLContext.Model.Save(model, fileStream);            
            Console.WriteLine("Model saved to: {0}",modelPath);            
        }

        private static void Evaluate(MLContext mLContext, ITransformer model, int trainBit)
        {
            if (trainBit==1)
            {
                IDataView dataView = mLContext.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: ',');
                var predictions = model.Transform(dataView);
                var metrics = mLContext.Regression.Evaluate(predictions, "Label", "Score");
                Console.WriteLine();
                Console.WriteLine($"*************************************************");
                Console.WriteLine($"*       Model quality metrics evaluation         ");
                Console.WriteLine($"*------------------------------------------------");
                Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}" + ". Closer to 1 the better the model has trained");
                Console.WriteLine($"*       RMS loss:      {metrics.Rms:#.##}" + ". The lesser it is the better the model has trained");
                Console.WriteLine($"        Actual loss: {metrics.L1}");
                Console.WriteLine("__________________________________________________");
                GetUserChoice(mLContext, model);
            }
            else
            {
                IDataView dataView = mLContext.Data.LoadFromTextFile<DistanceTimeTrainer>(_testDataPath, hasHeader: true, separatorChar: ',');
                var predictions = model.Transform(dataView);
                var metrics = mLContext.Regression.Evaluate(predictions, "Label", "Score");
                Console.WriteLine();
                Console.WriteLine($"*************************************************");
                Console.WriteLine($"*       Model quality metrics evaluation         ");
                Console.WriteLine($"*------------------------------------------------");
                Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}" + ". Closer to 1 the better the model has trained");
                Console.WriteLine($"*       RMS loss:      {metrics.Rms:#.##}" + ". The lesser it is the better the model has trained");
                Console.WriteLine($"        Actual loss: {metrics.L1}");
                Console.WriteLine("__________________________________________________");
                GetUserChoice(mLContext, model);
            }
            
        }

        private static void TestSinglePrediction(MLContext mLContext)
        {
            ITransformer loadedModel;
            using (var stream=new FileStream(_fareModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mLContext.Model.Load(stream);                
            }
            var predictionFunction = loadedModel.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(mLContext);
            var taxiTripSample = new TaxiTrip() {
                VendorID="VTS",
                RateCode="1",
                PassengerCount=2,
                TripTime=1140,
                TripDistance=3.75f,
                PaymentType="CRD",
                TripAmount=0
            };
            var prediction = predictionFunction.Predict(taxiTripSample);

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
            GetUserChoice(mLContext, loadedModel);
        }

        private static void TestSinglePrediction(MLContext mLContext, TaxiTrip taxiTrip)
        {
            ITransformer loadedModel;
            using (var stream = new FileStream(_fareModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mLContext.Model.Load(stream);
            }
            var prediction = loadedModel.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(mLContext).Predict(taxiTrip);
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}.");
            Console.WriteLine($"**********************************************************************");
            GetUserChoice(mLContext,loadedModel);
        }
    }
}
