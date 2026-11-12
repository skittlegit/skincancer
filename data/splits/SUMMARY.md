# Split summary

```
label_cancer    cancer  no label  not_cancer    All
split                                              
ak_side              0       703           0    703
ddi_test           171         0         485    656
dermacon_other       0         0        5379   5379
dermacon_test       19         0          46     65
excluded             5       922          19    946
midas_test         241         0         173    414
midas_train        321         0         231    552
midas_val          240         0         174    414
test               478         0        2217   2695
train             2220         0       10342  12562
val                478         0        2217   2695
All               4173      1625       21283  27081
```

## By type of lesion

```
label_fine        ak   bcc  benign  mel  non_tumour  other_benign_tumour  rare_cancer   scc  unlabelled    All
split                                                                                                         
ak_side          703     0       0    0           0                    0            0     0           0    703
ddi_test           4    49     255   21           0                    0           40    53         234    656
dermacon_other     0     0       0    0           0                    0            0     0        5379   5379
dermacon_test      0    10      46    2           0                    0            0     7           0     65
excluded         329     0       2    0          13                    2            4     1         595    946
midas_test         0   121     173   47           0                    0            0    73           0    414
midas_train        0   161     231   62           0                    0            0    98           0    552
midas_val          0   120     174   46           0                    0            0    74           0    414
test               0   217     158   94        1810                  249           51   116           0   2695
train              0  1007     733  437        8447                 1162          236   540           0  12562
val                0   217     158   94        1810                  249           51   116           0   2695
All             1036  1902    1930  803       12080                 1662          382  1078        6208  27081
```

## train / val / test by dataset

```
label_cancer          cancer  not_cancer    All
split dataset                                  
test  fitzpatrick17k     314        2145   2459
      pad_ufes_20        164          72    236
train fitzpatrick17k    1459       10007  11466
      pad_ufes_20        761         335   1096
val   fitzpatrick17k     314        2145   2459
      pad_ufes_20        164          72    236
All                     3176       14776  17952
```
