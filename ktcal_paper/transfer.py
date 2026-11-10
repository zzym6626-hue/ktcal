"""Warm-start PaperSTNet from a sequence-trained models.STNet checkpoint.

Both implementations share the same ITD convolutions and the same 2048-to-128
bidirectional GRU; only the parameter names and the coordinate head differ.
The head is never transferred, so this is an initialization, not a claim that
the two architectures are identical.
"""
import torch

BLOCK_MAP = (('block1', 'itd.0'), ('block2', 'itd.1'), ('block3', 'itd.2'))
MODULE_MAP = (('spatial.0', 'layers.0'), ('spatial.1', 'layers.1'),
              ('temporal.0', 'layers.3'), ('temporal.1', 'layers.4'))
SHARED_MAP = (('layernorm.', 'norm.'), ('gru.', 'gru.'))


def map_sequence_state(state, variant='full'):
    """Translate a models.STNet state dict into PaperSTNet parameter names."""
    if variant != 'full':
        raise ValueError('Sequence transfer requires the full decoupled variant')
    if 'model_state_dict' in state and not any(k.startswith('block1') for k in state):
        state = state['model_state_dict']
    mapped = {}
    for source, target in BLOCK_MAP:
        for from_module, to_module in MODULE_MAP:
            prefix = f'{source}.{from_module}.'
            for key, value in state.items():
                if key.startswith(prefix):
                    mapped[f'{target}.{to_module}.' + key[len(prefix):]] = value
    for prefix, target in SHARED_MAP:
        for key, value in state.items():
            if key.startswith(prefix):
                mapped[target + key[len(prefix):]] = value
    return mapped


def apply_sequence_state(model, state, variant='full'):
    """Load compatible weights and report what stayed randomly initialized."""
    mapped = map_sequence_state(state, variant)
    if not mapped:
        raise ValueError('Source checkpoint has no transferable sequence weights')
    target = model.state_dict()
    missing = sorted(key for key in mapped if key not in target)
    mismatch = {key: (tuple(value.shape), tuple(target[key].shape))
                for key, value in mapped.items()
                if key in target and tuple(value.shape) != tuple(target[key].shape)}
    if missing or mismatch:
        raise ValueError(f'Incompatible sequence transfer: missing={missing} mismatch={mismatch}')
    model.load_state_dict({key: value.to(target[key].dtype) for key, value in mapped.items()},
                          strict=False)
    return {'mapped_parameters': len(mapped),
            'untouched_parameters': sorted(key for key in target if key not in mapped)}


def load_initialization(model, checkpoint):
    """Load either a PaperSTNet checkpoint or a models.STNet sequence checkpoint."""
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if 'model_state' in state:
        model.load_state_dict(state['model_state'])
        return {'source': 'PaperSTNet', 'untouched_parameters': []}
    report = apply_sequence_state(model, state, getattr(model, 'variant', 'full'))
    return {'source': 'models.STNet', **report}
