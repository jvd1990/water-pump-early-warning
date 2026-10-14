# Water Pump Early Warning — Engineering Notes

## 1. Data understanding

The dataset contains 220,320 rows and 55 columns. `timestamp` initially loads as a string/object and is converted using `pd.to_datetime`. Status counts are NORMAL: 205,836; RECOVERING: 14,477; BROKEN: 7. Failure moments represent approximately 0.00318% of all rows. Only 7 independent events provide very limited statistical evidence.

## 2. Cleaning

`sensor_15` is entirely missing and is removed using a threshold of more than 50% missingness in training. `Unnamed: 0` is a row index and is excluded from model inputs. `sensor_50` has approximately 34.96% missingness overall and is retained under this rule. Missing percentages for every sensor are saved in `missing_percent.csv`.

`ffill(limit=60)` carries the latest past reading forward for at most 60 minutes. Remaining gaps use that sensor's training median. This avoids future information; backward filling or two-sided interpolation would be inappropriate for live warnings. The 60-minute limit is a project assumption, and median filling across long gaps may hide real behavior.

## 3. Engineering analysis

The five leading candidates based on standardized mean shifts before the four training failures are `sensor_28`, `sensor_22`, `sensor_33`, `sensor_20`, and `sensor_21`. For each failure, the preceding 24 hours are compared with the earlier seven-day-to-48-hour baseline, using NORMAL minutes only.

This descriptive ranking does not establish causality or consistent precursors across all failures. Unknown sensor identities and units prevent claims about pressure or bearing temperature. Maintenance engineers should compare these channels with equipment information, load, and repair history. Correlation also does not establish causation.

## 4. Target and chronological split

For each healthy minute, the target asks whether a BROKEN event occurs within the next 24 hours. Failure moments and repair periods are excluded from early-warning modeling. The final 24 hours are excluded because their future is unobserved.

April–May form training, June validation, and July–August testing. The 24 hours before split boundaries are purged so label horizons cannot reach into the next partition. Random shuffling could distribute highly similar minutes preceding the same failure across training and test, producing optimistic results.

## 5. Model and evaluation

A Random Forest uses current sensor readings, trailing 60-minute means, and differences from 60 minutes earlier. `machine_status` is not a model input. The alert threshold is selected using validation F1 only.

Always predicting no upcoming failure achieves 96.50% test accuracy but warns of no failures. At the selected threshold of 0.05, the model alerts at every eligible test minute: recall is 100%, but precision is only 3.50%. Both events are covered, with 79,451 false-warning minutes. This warning policy is therefore unsuitable for practical use. The apparent 24-hour lead time results from continuous alerts and is not evidence of operational success.

Missing failures threatens water supply, while continuous warnings undermine operator trust. Event coverage, lead time, and false-warning burden must be assessed together. Actual costs and team capacity determine the preferred tradeoff. Only two test failures are insufficient to establish reliability.
