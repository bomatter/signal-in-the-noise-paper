import importlib

import torch


def create_model(
    model_name,
    image_size: tuple[int, int, int],
    checkpoint: str = None,
    device: str = "cpu",
    compile: bool = False,  # whether to use torch.compile
    **model_kwargs,  # model-specific kwargs forwarded to the model module's create_model
):
    try:
        model_module = importlib.import_module(f"sitn.models.{model_name.lower()}")
    except ModuleNotFoundError:
        raise ValueError(f"Model '{model_name}' is not available. Add a module in sitn/models/")

    model = model_module.create_model(image_size=image_size, **model_kwargs)

    # Move model to device
    model.to(device)

    # Restore checkpoint if provided
    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=device)
        # Remove "_orig_mod" prefix, which is added when a torch.compile-wrapped model is saved
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)

    # Compile model if requested
    if compile:
        model = torch.compile(model)

    return model
