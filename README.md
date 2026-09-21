# Web Traffic Forecasting with a Bidirectional LSTM

An early practice project (March 2024) on the Kaggle competition [Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting): predict the daily page views of about 145,000 Wikipedia articles, scored by SMAPE. The repository holds the data pipeline and model definition for a direct multi-horizon LSTM forecaster in PyTorch.

**Status.** Exploratory and incomplete. It contains no inference or submission code, and the training script needs the fixes listed under [Known issues](#known-issues) before it runs on a current PyTorch. A late submission made with this model scored about 42 SMAPE; that score cannot be reproduced from the code in this repository.

## Approach

| Item | Detail |
|---|---|
| Data | `train_1.csv` from Kaggle (daily views, 2015-07-01 to 2016-12-31); a random sample of 5,000 pages is used |
| Target | `log1p(views)`; missing values filled with the per-date median across pages, then 0 |
| Input | 128 consecutive days × 4 features: log views plus label-encoded project, access type and agent (parsed from the page name) |
| Output | The next 64 days in one forward pass (direct multi-horizon, no decoder) |
| Model | Per-step MLP 4 → 64 → 128, six-layer bidirectional LSTM (hidden 128), linear layers 256 → 1 and 128 → 64; about 2.3M parameters |
| Training | MSE on log views, Adam (learning rate 4e-3, weight decay 0.005), 40 epochs, batch size 64; one random window per page per epoch |
| Validation | 80/20 split by page (4,000 / 1,000 pages) |

## Repository layout

| File | Purpose |
|---|---|
| `data_preprocess.py` | Configuration, page-name parsing, sampling, `TrainDataset_sequence` |
| `model.py` | `Pred_Sequence_Model` |
| `train.py` | Training and validation loops, SMAPE helper |
| `run_neural_network.py` | Entry point; place `train_1.csv.zip` from Kaggle in the working directory first |

## Known issues

- `train.py` was adapted from a classification project: labels are cast to integers before the MSE loss (which raises a dtype error on current PyTorch), validation computes an arg-max "accuracy", and the checkpoint is saved on that accuracy instead of validation loss.
- The SMAPE helper is never called and its denominator lacks parentheses (`|y| + |ŷ|/2` instead of `(|y| + |ŷ|)/2`).
- Validation is split by page, not by time, so it does not measure forecasting into an unseen period as the competition does.
- The page sample is unseeded, and the output layer ties the sequence length to the LSTM width (both 128).

## License

MIT, see [LICENSE](LICENSE).
