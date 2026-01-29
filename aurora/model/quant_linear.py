import gc
import math
from socket import PACKET_BROADCAST
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from quantize.matmul_had import *
#from aurora.model.quantizer import UniformActivationQuantizer

class RoundSTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, qmax, qmin):
        return x.round().clamp(qmin, qmax)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

def round_ste(x: torch.Tensor, qmax, qmin):
    """Implement Straight-Through Estimator for rounding operation."""
    return RoundSTEFunction.apply(x, qmax, qmin)

class QuantLinearTorch(nn.Module):

    def __init__(self, qparams, aparams, infeatures, outfeatures, bias, symmetric, real_pack=True):
        super().__init__()
        self.infeatures = infeatures
        self.outfeatures = outfeatures

        self.n_bits = qparams.bits  
        self.group_size = qparams.group_size  
        self.real_pack = real_pack
        self.symmetric = symmetric
        
        if real_pack:
            self.register_buffer('qweight', torch.zeros((infeatures // (32//self.n_bits[0]), outfeatures), dtype=torch.int32))
        else:
            self.qweight = nn.Parameter(torch.zeros((infeatures, outfeatures), dtype=torch.half))
        
        self.scales = nn.Parameter(torch.zeros((infeatures//list(self.group_size.values())[0], outfeatures), dtype=torch.half))
        
        # For symmetric quantization, zeros may not be needed, but created for consistency.
        self.zeros = nn.Parameter(torch.zeros((infeatures//list(self.group_size.values())[0], outfeatures), dtype=torch.half))

        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.half))
        else:
            self.bias = None

        self.act_quantizer = None#UniformActivationQuantizer(aparams)
        self.register_buffer("qmax_list", torch.ones(1)*15)
        self.register_buffer("enable_qat", torch.zeros(1)>0)

    def reinitialize(self):
        
        assert self.real_pack is False, "reinitialization is only available when real_pack is False"
        
        qscale_post = self.scales.view(self.scales.size(0), -1).half()
        qzeros_post = self.zeros.view(self.zeros.size(0), -1).half() if self.zeros is not None else None

        ratio = list(self.group_size.values())[0]
        scale = qscale_post.unsqueeze(1).repeat(1, ratio, 1).view(-1, self.qweight.size(-1))
        zeros = qzeros_post.unsqueeze(1).repeat(1, ratio, 1).view(-1, self.qweight.size(-1)) if qzeros_post is not None else None

        # Dequantize weights.
        weights = self.qweight.mul(scale)
        if zeros is not None and not self.symmetric:
            weights = weights - zeros
        
        self.weights = nn.Parameter(weights)
        del self.qweight, self.scales, self.zeros
        
        self.enable_qat[0] = True

    def pack(self, linear, scale_post, zeros_post, bits_row_end_index=None, qmax_list=None):
        """Pack a linear layer with quantization parameters.
        
        Args:
            linear: The linear layer to be quantized (nn.Linear).
            scale_post: List of scale tensors for each bit width group.
            zeros_post: List of zero point tensors for each bit width group.
            bits_row_end_index: Indices indicating where each bit width group ends.
            qmax_list: List of maximum quantization values for each bit width.
        """
        
        if linear.bias is not None:
            self.bias.data = linear.bias.data.clone()

        if linear.act_quantizer is not None:
            self.act_quantizer = linear.act_quantizer
        else:
            self.act_quantizer = None
                  
        tensors = []
        quantized = []
        self.register_buffer("qmax_list", torch.Tensor(qmax_list))

        weight = linear.weight.data

        # Independent quantization statistic per bits.
        for i in range(len(bits_row_end_index)):
            tensors.append(weight[:, :bits_row_end_index[i]])

        # Quantize & pack the tensors per bits.
        for i, (tensor_per_bits, scale_per_bits, zeros_per_bits) in enumerate(zip(tensors, scale_post, zeros_post)):
            
            scale_per_bits = scale_per_bits.cuda()
            zeros_per_bits = zeros_per_bits.cuda()

            tensor_shape_per_bits = tensor_per_bits.size()
            tensor_per_bits = tensor_per_bits.reshape(-1, list(self.group_size.values())[i]).cuda()
            
            if self.n_bits[i] > 1:
                if self.symmetric:
                    # 对称量化：先量化，再加零点并clamp
                    tensor_per_bits = torch.round(tensor_per_bits / scale_per_bits)
                    qmax_val = self.qmax_list[i].item()  # 获取标量值
                    qmin = -(qmax_val // 2) if qmax_val % 2 == 1 else -(qmax_val // 2 - 1)
                    qmax = qmax_val // 2
                    tensor_per_bits = tensor_per_bits.clamp(qmin, qmax)
                    # 将对称范围映射到[0, qmax_list[i]]
                    tensor_per_bits = tensor_per_bits + (qmax_val // 2)
                else:
                    # 非对称量化：原来的逻辑
                    tensor_per_bits = tensor_per_bits + zeros_per_bits
                    tensor_per_bits = torch.round(tensor_per_bits / scale_per_bits)
                    tensor_per_bits = tensor_per_bits.clamp(0, self.qmax_list[i].item())
            else:
                tensor_per_bits = tensor_per_bits + zeros_per_bits
                tensor_per_bits = tensor_per_bits.sign().add(1).div(2).clamp(0, 1)
            
            tensor_per_bits = tensor_per_bits.reshape(tensor_shape_per_bits).t().cpu().numpy().astype(
                np.uint32)  # (outfeatures, infeatures) -> (infeatures, outfeatures)
            quantized_tensor_per_bits = np.zeros(           
                (tensor_shape_per_bits[1] // 32 * math.ceil(self.n_bits[i]), tensor_shape_per_bits[0]), dtype=np.uint32)
            
            j = 0
            row = 0
            while row < quantized_tensor_per_bits.shape[0]:
                if self.n_bits[i] in [1, 1.5, 2, 4, 8]:
                    for k in range(j, j + (32 // math.ceil(self.n_bits[i]))):
                        quantized_tensor_per_bits[row] |= tensor_per_bits[k] << (math.ceil(self.n_bits[i]) * (k - j))
                    j += 32 // math.ceil(self.n_bits[i])
                    row += 1
                else:
                    raise NotImplementedError("Only 1, 1.5, 2, 4, 8 bits are supported.")

            if self.real_pack:
                # Real bit packing.
                quantized_tensor_per_bits = quantized_tensor_per_bits.astype(np.int32)
                quantized.append(torch.from_numpy(quantized_tensor_per_bits))
            else:
                # Store in FP16 format.
                quantized.append(torch.from_numpy(tensor_per_bits.astype(np.float16)))

        qweight = torch.cat(quantized, dim=0)
        self.qweight.data = qweight.data

        # Process scales and zeros - support symmetric and asymmetric quantization.
        qscale_list = []
        qzeros_list = []
        for i, (scale_post_per_bits, zeros_post_per_bits) in enumerate(zip(scale_post, zeros_post)):

            scale_post_per_bits = scale_post_per_bits.reshape(self.outfeatures, -1).t().contiguous()
            qscale_list.append(scale_post_per_bits.half().cpu())

            # For symmetric quantization, zero points are usually 0 or the midpoint of the quantization range.
            if self.symmetric:
                # For symmetric quantization, store midpoint as zero point.
                qmax_val = self.qmax_list[i].item()
                symmetric_zeros = torch.full_like(zeros_post_per_bits, qmax_val // 2)
                zeros_post_per_bits = symmetric_zeros.reshape(self.outfeatures, -1).t().contiguous()
                qzeros_list.append(zeros_post_per_bits.half().cpu())
            else:
                # Asymmetric quantization original logic.
                zeros_post_per_bits = zeros_post_per_bits.reshape(self.outfeatures, -1).t().contiguous()
                qzeros_list.append(zeros_post_per_bits.half().cpu())
                
        self.scales.data = torch.cat(qscale_list, dim=0).half()
        self.zeros.data = torch.cat(qzeros_list, dim=0).half()

        linear = linear.cpu()
        
        del linear, scale_post, zeros_post, weight
        torch.cuda.empty_cache()
        gc.collect()
    
    def dynamic_quant(self, x: torch.Tensor):
        quant_scale = x.abs().max(dim=-1, keepdim=True)[0].div(127.0).to(torch.float32)
        x = (x / quant_scale).round().clamp(-128, 127).to(torch.int8)
        return x, quant_scale

    def forward(self, x):
        # Original implementation for non-QQQ format
        if self.act_quantizer is not None:
            x = self.act_quantizer(x)
        
        if not self.enable_qat:
            # Move tensors to the same device as input
            self.qweight = self.qweight.to(x.device)
            self.scales = self.scales.to(x.device)
            self.zeros = self.zeros.to(x.device)

            # Prepare quantization parameters
            qweight = self.qweight
            qscale_post = self.scales.view(self.scales.size(0), -1).to(x.dtype)
            qzeros_post = self.zeros.view(self.zeros.size(0), -1).to(x.dtype) if self.zeros is not None else None
        
            # Apply scales and zero points
            ratio = list(self.group_size.values())[0]
            scale = qscale_post.unsqueeze(1).repeat(1, ratio, 1).view(-1, qweight.size(-1))
            zeros = qzeros_post.unsqueeze(1).repeat(1, ratio, 1).view(-1, qweight.size(-1)) if qzeros_post is not None else None

            if self.real_pack:
                # Unpack weights for real packed format
                with torch.no_grad():
                    bits = math.ceil(self.n_bits[0])
                    wf = (torch.arange(0, 32, bits, dtype=torch.int32)
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .to(qweight.device))
                    
                    weight_unpack = (torch.bitwise_right_shift(
                        qweight.unsqueeze(1).expand(-1, 32 // bits, -1),
                        wf
                    ).to(torch.int16 if bits == 8 else torch.int8)
                    .view(-1, qweight.size(-1)))
                    
                    torch.bitwise_and(weight_unpack, self.qmax_list[0].to(torch.int16), out=weight_unpack)
                    weights = weight_unpack
            else:
                weights = qweight.round().detach()

            # 根据量化类型进行反量化
            if self.symmetric:
                # 对称量化：将[0, qmax]映射回对称范围，然后乘以scale
                qmax_val = self.qmax_list[0].item()
                # 将量化值从[0, qmax]映射回对称范围
                weights = weights.to(x.dtype) - (qmax_val // 2)
                weights = weights.mul(scale)
            else:
                # 非对称量化：原来的逻辑
                weights = weights.mul(scale)
                if zeros is not None:
                    weights = weights - zeros

        else:
            # QAT模式
            weights = self.weights.t().contiguous()
            weights_shape = weights.size()
            weights = weights.view(-1, list(self.group_size.values())[0])
            
            if self.symmetric:
                # 对称量化的QAT
                qmax_val = self.qmax_list[0].item()
                scale_per_bits = weights.abs().amax(-1, keepdim=True).div(qmax_val // 2).clamp(1e-8, 1e8)
                qmax = qmax_val // 2
                qmin = -(qmax_val // 2)
                
                tensor_quant = round_ste(weights / scale_per_bits, qmax=qmax, qmin=qmin)
                tensor_quant = tensor_quant.mul(scale_per_bits)
            else:
                # 非对称量化的QAT - 原来的逻辑
                xmin = weights.amin(-1, keepdim=True)
                xmax = weights.amax(-1, keepdim=True)
                qmax = self.qmax_list[0].item()
                scale_per_bits = ((xmax - xmin) / qmax).clamp(1e-8, 1e8)
                zeros_per_bits = -xmin

                tensor_quant = weights + zeros_per_bits
                qmin = 0

                tensor_quant = round_ste((tensor_quant) / scale_per_bits, qmax=qmax, qmin=qmin)
                tensor_quant = tensor_quant.mul(scale_per_bits)
                tensor_quant = tensor_quant.sub(zeros_per_bits) 
            
            weights = tensor_quant.reshape(weights_shape).t().contiguous().to(x.dtype)
        
        # Compute matrix multiplication
        out = torch.matmul(x, weights)

        out = (out + self.bias) if self.bias is not None else out
                
        return out
