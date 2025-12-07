import torch
import triton
import triton.language as tl
from torch.autograd import Function

# from https://github.com/sustcsonglin/pytorch_linear_rnn

@triton.jit
def fwd_sequential_scan_complex(
    v_real,
    v_imag,
    decay_real,
    decay_imag,
    hidden_real,
    hidden_imag,                
    B,
    L,
    C, 
    BLOCK_M: tl.constexpr,
):
    
    offset_b = tl.program_id(0)
    
    if offset_b >= B:
        return

    offset_n = tl.program_id(1)
    ptr = tl.arange(0, BLOCK_M) + offset_b * L * C + offset_n * BLOCK_M        
    h_real = tl.zeros([BLOCK_M,], dtype=tl.float32)
    h_imag = tl.zeros([BLOCK_M,], dtype=tl.float32)

    for _ in range(L):        
        x_real = tl.load(v_real + ptr).to(tl.float32)                
        x_imag = tl.load(v_imag + ptr).to(tl.float32)
        
        f_real = tl.load(decay_real + ptr).to(tl.float32) 
        f_imag = tl.load(decay_imag + ptr).to(tl.float32) 
        
        h_real_new = h_real * f_real - h_imag * f_imag + x_real
        h_imag_new = h_real * f_imag + h_imag * f_real + x_imag 
                
        tl.store(hidden_real + ptr, h_real_new.to(hidden_real.dtype.element_ty))
        tl.store(hidden_imag + ptr, h_imag_new.to(hidden_imag.dtype.element_ty))
        h_real = h_real_new
        h_imag = h_imag_new
        ptr += C


@triton.jit
def bwd_sequential_scan_complex(

    grad_output_real,
    grad_output_imag,

    v_real,
    v_imag,

    f_real,
    f_imag,
        
    hidden_real,
    hidden_imag,

    B,
    L,
    C, 
    BLOCK_M: tl.constexpr,
):

    offset_b = tl.program_id(0)
    
    if offset_b >= B:
        return

    offset_n = tl.program_id(1)    

    ptr = tl.arange(0, BLOCK_M) + offset_b * L * C + (L-1) * C + offset_n * BLOCK_M

    grad_h_real = tl.zeros([BLOCK_M,], dtype=tl.float32)
    grad_h_imag = tl.zeros([BLOCK_M,], dtype=tl.float32)

    for time_step in range(L-1, -1, -1):
        grad_real = tl.load(grad_output_real + ptr).to(tl.float32)            
        grad_imag = tl.load(grad_output_imag + ptr).to(tl.float32)          
        
        grad_h_real += grad_real
        grad_h_imag += grad_imag
        
        decay_real = tl.load(f_real + ptr).to(tl.float32)   
        decay_imag = tl.load(f_imag + ptr).to(tl.float32)   

        h_real = tl.load(hidden_real + ptr - C, mask= ptr >= (offset_b * L * C + C), other=0.0).to(tl.float32)
        h_imag = tl.load(hidden_imag + ptr - C, mask= ptr >= (offset_b * L * C + C), other=0.0).to(tl.float32)
                
        grad_f_real = (grad_h_real * h_real + grad_h_imag * h_imag) 
        grad_f_imag = (grad_h_imag * h_real - grad_h_real * h_imag) 

        tl.store(f_real + ptr, grad_f_real.to(f_real.dtype.element_ty))                
        tl.store(f_imag + ptr, grad_f_imag.to(f_real.dtype.element_ty))                

        tl.store(v_real + ptr, grad_h_real.to(v_real.dtype.element_ty))
        tl.store(v_imag + ptr, grad_h_imag.to(v_real.dtype.element_ty))

        grad_h_real_new = grad_h_real * decay_real + grad_h_imag * decay_imag 
        grad_h_imag_new = grad_h_imag * decay_real - grad_h_real * decay_imag
        
        grad_h_real = grad_h_real_new
        grad_h_imag = grad_h_imag_new
        
        ptr -= C        



class TritonSequentialScan_Complex(Function):
    @staticmethod
    def forward(ctx, v_real, v_imag, f_real, f_imag):
        B,L,C = v_real.shape
        num_warps = 8
        assert C % 256 == 0, 'Hidden dimension must be multiple of 256'
        v_real = v_real.contiguous()
        v_imag = v_imag.contiguous()
        f_real = f_real.contiguous()
        f_imag = f_imag.contiguous()

        hidden_real = torch.zeros_like(v_real).contiguous()
        hidden_imag = torch.zeros_like(v_imag).contiguous()
        
        with torch.cuda.device(v_real.device):
            fwd_sequential_scan_complex[(B, int(C/256))](
                v_real,
                v_imag,
                f_real,
                f_imag,
                hidden_real,
                hidden_imag,
                B,
                L,
                C, 
                BLOCK_M=256,
                num_warps=num_warps
            )

            ctx.save_for_backward(v_real, v_imag, f_real, f_imag, hidden_real, hidden_imag)    
        return hidden_real, hidden_imag
            
    @staticmethod
    def backward(ctx, grad_output_real, grad_output_imag):
        
        v_real, v_imag, f_real, f_imag, hidden_real, hidden_imag = ctx.saved_tensors 
        B, L, C = v_real.shape
        
        num_warps = 8

        with torch.cuda.device(v_real.device):
            bwd_sequential_scan_complex[(B,  int(C/256))](
                grad_output_real, 
                grad_output_imag,

                v_real, 
                v_imag,
                f_real,
                f_imag, 
                hidden_real, 
                hidden_imag,
                
                B,
                L,
                C, 
                BLOCK_M=256,
                num_warps=num_warps
            )
        return v_real, v_imag, f_real, f_imag

def complex_scan(v_real, v_imag, f_real, f_imag):
    C = v_real.size(2)
    pad_amount = 0
    if C % 256 != 0:
        pad_amount = 256 - (C % 256)
        padding = torch.zeros((1,1,1), device=v_real.device, dtype=v_real.dtype) \
            .expand(v_real.size(0), v_real.size(1), pad_amount)
        v_real = torch.cat((v_real, padding), dim=2)
        v_imag = torch.cat((v_imag, padding), dim=2)
        f_real = torch.cat((f_real, padding), dim=2)
        f_imag = torch.cat((f_imag, padding), dim=2)
    h_real, h_imag = TritonSequentialScan_Complex.apply(v_real, v_imag, f_real, f_imag)
    return h_real[:, :, :C], h_imag[:, :, :C]