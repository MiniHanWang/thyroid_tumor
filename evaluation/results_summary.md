# Results Summary

- Sample size: 995
- Malignant proportion: 0.653
- LN metastasis proportion: 0.310

## Table 1 Highlights (first 20 rows)

              variable                 group_0                 group_1           test      p_value
                   age    52.00 [39.00, 61.00]    43.00 [36.00, 53.00] Mann-Whitney U 1.481813e-10
             weight_kg    60.00 [54.00, 67.88]    62.00 [55.00, 70.38] Mann-Whitney U 7.585623e-03
             height_cm 160.00 [156.00, 165.00] 160.00 [156.25, 167.00] Mann-Whitney U 9.358015e-02
                   bmi    23.30 [21.42, 25.00]    23.70 [21.50, 26.27] Mann-Whitney U 9.099042e-03
                 sex=0             237 (79.5%)             423 (75.3%)                         NaN
                 sex=1              61 (20.5%)             139 (24.7%)                         NaN
         sex (overall)                                                     Chi-square 1.856848e-01
         smoking_bin=0             265 (88.9%)             491 (87.4%)                         NaN
         smoking_bin=1              33 (11.1%)              71 (12.6%)                         NaN
 smoking_bin (overall)                                                     Chi-square 5.770936e-01
        drinking_bin=0             264 (88.6%)             481 (85.6%)                         NaN
        drinking_bin=1              34 (11.4%)              81 (14.4%)                         NaN
drinking_bin (overall)                                                     Chi-square 2.600938e-01
        hypertension=0             237 (79.5%)             471 (83.8%)                         NaN
        hypertension=1              61 (20.5%)              91 (16.2%)                         NaN
hypertension (overall)                                                     Chi-square 1.412957e-01
            diabetes=0             272 (91.3%)             514 (91.5%)                         NaN
            diabetes=1               26 (8.7%)               48 (8.5%)                         NaN
    diabetes (overall)                                                     Chi-square 1.000000e+00
           hepatitis=0             284 (95.3%)             527 (93.8%)                         NaN

## Significant Variables (Univariate, malignant)

           variable       OR  CI95_low  CI95_high      p_value   n
nodular_goiter_flag 0.354993  0.265752   0.474200 2.367440e-12 860
   thyroiditis_flag 4.346834  2.866360   6.591972 4.625189e-12 860
                age 0.961489  0.950151   0.972963 8.660028e-11 860
                bmi 1.065331  1.021638   1.110892 3.057086e-03 860

## Significant Variables (Multivariable, malignant)

            variable       OR  CI95_low  CI95_high      p_value   n
    thyroiditis_flag 4.153775  2.691963   6.409392 1.236367e-10 860
 nodular_goiter_flag 0.447171  0.324467   0.616278 8.750252e-07 860
                 age 0.966820  0.953442   0.980386 2.071136e-06 860
                 bmi 1.063891  1.013509   1.116777 1.234471e-02 860
other_cancer_history 2.561338  1.067548   6.145348 3.517068e-02 860

## Best Prediction Model

- LogisticRegression (AUC=0.708, Accuracy=0.669)
