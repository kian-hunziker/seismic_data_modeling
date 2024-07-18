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
    "conv_net": "models.simple_test_models.ConvNet"
}

dataset = {
    "mnist": "dataloaders.MNISTdataloader.MNISTdataset",
    "sine": "dataloaders.simple_waveform.SineWaveLightningDataset",
    "costarica-small": "dataloaders.costa_rica_small.CostaRicaSmallLighting"
}