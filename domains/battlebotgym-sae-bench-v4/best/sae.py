# SAE Architecture — agents edit this file
#
# The default engine uses sae_lens's built-in BatchTopKTrainingSAE.
# To use a custom architecture:
#   1. Define your SAE class and config class here
#   2. Set sae_class: YourClassName in config.yaml
#   3. Your config class needs a from_dict(cfg, total_steps) classmethod
#
# The encoder/decoder interface:
#   - forward(x) -> (sae_out, feature_acts, loss, loss_dict)
#   - encode(x) -> feature_acts
#   - decode(feature_acts) -> sae_out
#
# Useful imports:
#   from sae_lens.saes.sae import TrainingSAE, TrainingSAEConfig
#   from sae_lens.saes.batchtopk_sae import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig
