import torch

checkpoint = {
    'model': raw_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'config': config,
    'tokens': tokens_trained,
}
torch.save(checkpoint, 'emergency_ckpt.pt')