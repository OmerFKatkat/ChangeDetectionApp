from collections import OrderedDict

import torch


MODEL_STATE_KEYS = (
    'model_G_state_dict',
    'model_state_dict',
    'state_dict',
    'net_G_state_dict',
    'model',
    'net_G',
)


def _write(message, logger=None):
    if logger is not None:
        logger.write(message)
    else:
        print(message, end='')


def _is_weights_only_error(error):
    message = str(error)
    return 'weights_only' in message or 'Weights only load failed' in message


def torch_load_checkpoint(path, map_location, logger=None):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as error:
        if not _is_weights_only_error(error):
            raise
        _write(
            'Default torch.load failed because of weights_only checkpoint loading. '
            'Retrying with weights_only=False for this trusted local checkpoint.\n',
            logger=logger,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def _looks_like_state_dict(value):
    return (
        isinstance(value, dict)
        and len(value) > 0
        and all(isinstance(key, str) for key in value.keys())
        and any(torch.is_tensor(item) for item in value.values())
    )


def extract_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in MODEL_STATE_KEYS:
            value = checkpoint.get(key)
            if _looks_like_state_dict(value):
                return value, key

        if _looks_like_state_dict(checkpoint):
            return checkpoint, 'raw_state_dict'

    raise KeyError(
        'Could not find model weights in checkpoint. Expected one of: %s'
        % ', '.join(MODEL_STATE_KEYS)
    )


def load_model_state_dict(model, state_dict, strict=True):
    try:
        return model.load_state_dict(state_dict, strict=strict), 'as-is'
    except RuntimeError:
        model_keys = list(model.state_dict().keys())
        checkpoint_keys = list(state_dict.keys())

        if checkpoint_keys and model_keys:
            checkpoint_has_module = all(key.startswith('module.') for key in checkpoint_keys)
            model_has_module = all(key.startswith('module.') for key in model_keys)

            if checkpoint_has_module and not model_has_module:
                stripped_state_dict = OrderedDict(
                    (key.replace('module.', '', 1), value)
                    for key, value in state_dict.items()
                )
                return model.load_state_dict(stripped_state_dict, strict=strict), 'stripped module. prefix'

            if model_has_module and not checkpoint_has_module:
                prefixed_state_dict = OrderedDict(
                    (key if key.startswith('module.') else 'module.' + key, value)
                    for key, value in state_dict.items()
                )
                return model.load_state_dict(prefixed_state_dict, strict=strict), 'added module. prefix'

        raise
