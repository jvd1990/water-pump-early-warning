Can a water pump failure be detected before it happens?

I explored this question using 220,320 minute-level observations from 52 anonymized sensors and seven recorded failures in the Kaggle Pump Sensor Data dataset.

I built a Python workflow covering data quality checks, sensor analysis, past-only preprocessing, trailing-window features and a Random Forest classifier targeting failures within the next 24 hours.

To evaluate realistically, I used chronological training, validation and test periods, with a 24-hour purge before split boundaries. I assessed precision, recall, precision-recall performance, failure-event coverage and false-warning duration.

The most useful finding was a limitation: the validation-selected threshold covered both test failures, but it also generated continuous alerts. Its 100% recall therefore did not translate into a useful maintenance policy. An always-negative baseline achieved 96.5% accuracy while detecting no failures—a clear example of why accuracy alone can mislead.

This project strengthened my understanding of time-series leakage, rare-event evaluation and the connection between machine learning metrics and maintenance decisions. With only seven failures, more data and operational context are essential before deployment.

Code, figures and a transparent discussion of results: https://github.com/jvd1990/water-pump-early-warning

#PredictiveMaintenance #MechanicalEngineering #Python #MachineLearning #DataAnalysis
