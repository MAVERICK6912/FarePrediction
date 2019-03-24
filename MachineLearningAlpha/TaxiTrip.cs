using System;
using System.Collections.Generic;
//importing ML.Data
using Microsoft.ML.Data;

namespace MachineLearningAlpha
{
    class TaxiTrip
    {
        [LoadColumn(0)]
        public string VendorID;

        [LoadColumn(1)]
        public string RateCode;

        [LoadColumn(2)]
        public float PassengerCount;

        [LoadColumn(3)]
        public float TripTime;

        [LoadColumn(4)]
        public float TripDistance;

        [LoadColumn(5)]
        public string PaymentType;

        [LoadColumn(6)]
        public float TripAmount;
    }

    class DistanceTimeTrainer
    {
        [LoadColumn(3)]
        public float TripTime;
        [LoadColumn(4)]
        public float TripDistance;
    }

    class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
    class DistancePrediction
    {
        [ColumnName("Score")]
        public double Distance;
    }
    class TimePrediction
    {
        [ColumnName("Score")]
        public float Seconds;
    }
}
