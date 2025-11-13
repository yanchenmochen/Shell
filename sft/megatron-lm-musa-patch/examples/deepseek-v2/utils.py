import os
import time
import torch

def get_torch_profiler(device, profile = False):
    def trace_handler(prof):
        rank = 0
        curr_trace_dir_name = "iteration_" + str(prof.step_num)
        curr_trace_dir = os.path.join('./profiler_test', curr_trace_dir_name)
        if not os.path.exists(curr_trace_dir):
            os.makedirs(curr_trace_dir, exist_ok=True)
        curr_trace_path = os.path.join(curr_trace_dir, f"rank{rank}.{int(time.time()*1000)}.pt.trace.json")
        print(f"Dumping profiler traces at step {prof.step_num} to {curr_trace_path}")
        begin = time.monotonic()
        prof.export_chrome_trace(curr_trace_path)
        print(
            f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds"
    )
    if not profile:
        return None
    profiler =  torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MUSA if device=='musa' else torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=2, warmup=3, active=5, skip_first=2,repeat=100),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=False,
            with_stack=True,
            with_modules=True,
        )
    return profiler

def sync_device(device):
    torch.musa.synchronize() if device=='musa' else torch.cuda.synchronize()

def get_system_name():
    system = None
    device = None
    MAX_TFLOPS = None
    
    try:
        system = torch.cuda.get_device_name(0)
        device = 'cuda'
        if 'A100' in system:
            MAX_TFLOPS = 312
            system_name = 'a100'
        elif 'H100' in system:
            MAX_TFLOPS = 1979
            system_name = 'h100'
        else:
            raise ValueError("Unsupported device")  
    except:
        pass

    try:
        print(f"System detected: {system}")
        system = torch.musa.get_device_name(0)
        
        device = 'musa'
        if 'S5000' in system:
            MAX_TFLOPS = 401
            system_name = 's5000_mpx28'
        else:
            raise ValueError("Unsupported device")
    except:
        pass
    # assert system is not None and device is not None and MAX_TFLOPS is not None, "Unsupported device"   
    if os.environ.get('MAX_TFLOPS', None) is not None:
        MAX_TFLOPS = float(os.environ.get('MAX_TFLOPS'))
        
    print(f'System: {system}, Device: {device}, Max TFLOPS: {MAX_TFLOPS}')
    return system, device, MAX_TFLOPS