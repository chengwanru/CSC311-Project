# Ensemble Results (Seed Sensitivity)

## Protocol A: 60/20/20 (Train=60%, Val=20%, Test=20%)

Rule 1: **Majority vote (LR / NB / RF)**  
Rule 2: **LR high-confidence gate → LR, else Majority**


| seed | Majority acc | Majority macro-F1 | LR gate→maj acc | LR gate→maj macro-F1 |
| ---- | ------------ | ----------------- | --------------- | -------------------- |
| 1    | 0.8926       | 0.8919            | 0.8957          | 0.8949               |
| 7    | 0.9202       | 0.9202            | 0.9294          | 0.9295               |
| 13   | 0.9419       | 0.9410            | 0.9266          | 0.9257               |
| 21   | 0.9465       | 0.9458            | 0.9465          | 0.9458               |
| 42   | 0.9235       | 0.9230            | 0.9266          | 0.9262               |
| 84   | 0.8896       | 0.8887            | 0.8896          | 0.8887               |


**Overall (60/20/20)**


| method      | acc max | acc min | acc range | macro-F1 max | macro-F1 min | macro-F1 range |
| ----------- | ------- | ------- | --------- | ------------ | ------------ | -------------- |
| Majority    | 0.9465  | 0.8896  | 0.0569    | 0.9458       | 0.8887       | 0.0571         |
| LR gate→maj | 0.9465  | 0.8896  | 0.0569    | 0.9458       | 0.8887       | 0.0571         |


---

## Protocol B: 80/20 (Refit on Train+Val=80% → Test=20%)

Rule 1: **Majority vote (LR / NB / RF)**  
Rule 2: **LR high-confidence gate → LR, else Majority**


| seed | Majority acc | Majority macro-F1 | LR gate→maj acc | LR gate→maj macro-F1 |
| ---- | ------------ | ----------------- | --------------- | -------------------- |
| 1    | 0.8926       | 0.8920            | 0.8957          | 0.8952               |
| 7    | 0.9387       | 0.9386            | 0.9387          | 0.9387               |
| 13   | 0.9419       | 0.9409            | 0.9266          | 0.9260               |
| 21   | 0.9465       | 0.9459            | 0.9497          | 0.9491               |
| 42   | 0.9511       | 0.9509            | 0.9480          | 0.9479               |
| 84   | 0.9110       | 0.9103            | 0.9141          | 0.9135               |


**Overall (80/20)**


| method      | acc max | acc min | acc range | macro-F1 max | macro-F1 min | macro-F1 range |
| ----------- | ------- | ------- | --------- | ------------ | ------------ | -------------- |
| Majority    | 0.9511  | 0.8926  | 0.0585    | 0.9509       | 0.8920       | 0.0589         |
| LR gate→maj | 0.9497  | 0.8957  | 0.0540    | 0.9491       | 0.8952       | 0.0539         |


