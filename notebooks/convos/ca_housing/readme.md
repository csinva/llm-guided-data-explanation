- settings
  - flipped_coefs: flips the signs of 2 coefficients (correct)
  - gt_vs_subsampled: compares gt model, trained on all data, to the model trained on the subsampled data (wrong, but coef differences are minor)
  - 2_subsampled_models: returns the 2 best models based on a small, subsampled training set (wrong, but coef differences are minor)
- hyperparameters
  - all models use gpt-3.5-turbo
  - correct model always comes first