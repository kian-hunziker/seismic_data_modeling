optimizer = {
    "adam":    "torch.optim.Adam",
    "adamw":   "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd":     "torch.optim.SGD",
    "lamb":    "src.utils.optim.lamb.JITLamb",
}

scheduler = {
    "constant":        "transformers.get_constant_schedule",
    "plateau":         "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step":            "torch.optim.lr_scheduler.StepLR",
    "multistep":       "torch.optim.lr_scheduler.MultiStepLR",
    "cosine":          "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup":   "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup":   "transformers.get_cosine_schedule_with_warmup",
}

model = {
    "sashimi": "models.sashimi.sashimi_standalone.Sashimi",
    "conv_net": "models.simple_test_models.ConvNet",
    "mamba-sashimi": "models.sashimi.sashimi_mamba.MambaSashimi",
    "pure-mamba": "models.pure_mamba.PureMamba",
    "lstm": "models.lstm_baseline.LSTMSequenceModel",
    "identity": "models.simple_test_models.IdentityModel"
}

dataset = {
    "mnist": "dataloaders.MNISTdataloader.MNISTdataset",
    "sine": "dataloaders.simple_waveform.SineWaveLightningDataset",
    "costarica-small": "dataloaders.costa_rica_small.CostaRicaSmallLighting",
    "costarica-long-seq": "dataloaders.costa_rica_quantized.CostaRicaQuantizedLightning",
    "costarica-bpe": "dataloaders.costa_rica_bpe.CostaRicaBPELightning",
    "costarica-enc-dec": "dataloaders.costa_rica_quantized.CostaRicaEncDecLightning",
    "ethz-auto-reg": "dataloaders.seisbench_auto_reg.SeisBenchAutoReg",
    "ethz-phase-pick": "dataloaders.seisbench_auto_reg.SeisBenchPhasePick",
}