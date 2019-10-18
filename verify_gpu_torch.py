import torch
def verify_gpu():
    # Verify GPU
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())
    print('Memory:{:0.1f} GB'.format(torch.cuda.get_device_properties(0).total_memory/1000000000.0))