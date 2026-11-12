# Split summary

```
label_5class      ak   bcc  benign  exclude  exclude_rare_cancer  mel  other   scc    All
split                                                                                    
ak_side          311     0       0        0                    0    0      0     0    311
ddi_test           4    49     255      234                   40   21      0    53    656
dermacon_other     0     0       0     5379                    0    0      0     0   5379
dermacon_test      0    10      46        0                    0    2      0     7     65
excluded         721    13     368     2257                  342   27    401    46   4175
midas_test         0   402     578        0                    0  155      0   245   1380
pad_dark_test      0    25      26        0                    0    4      0     5     60
test               0   211     100        0                    0   89   1752   109   2261
train              0   981     458        0                    0  416   8175   503  10533
val                0   211      99        0                    0   89   1752   110   2261
All             1036  1902    1930     7870                  382  803  12080  1078  27081
```

## train / val / test by skin type

```
label_5class       bcc  benign  mel  other  scc    All
split fst_bucket                                      
test  I-III        194      83   73   1172   89   1611
      IV            12      10    9    317   11    359
      V-VI           5       7    7    263    9    291
train I-III        904     386  341   5469  411   7511
      IV            54      43   42   1479   49   1667
      V-VI          23      29   33   1227   43   1355
val   I-III        194      83   73   1172   89   1611
      IV            12       9    9    317   11    358
      V-VI           5       7    7    263   10    292
All               1403     657  594  11679  722  15055
```
