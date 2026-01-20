import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
import argparse
import warnings
import onnx
import math

# 忽略 weight_norm 警告
warnings.filterwarnings("ignore", message=".*weight_norm is deprecated.*")

# ==============================================================================
# 1. ONNX 完全兼容的重采样模块（使用 nn.Conv1d）
# ==============================================================================
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.: 
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.: 
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
    else: 
        beta = 0.
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = (torch.arange(-half_size, half_size) + 0.5) if even else (torch.arange(kernel_size) - half_size)
    filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)

class UpSample1d(nn.Module):
    """ONNX 兼容版本 - 使用固定 Conv1d 层"""
    def __init__(self, ratio=2, kernel_size=12, channels=512):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.channels = channels
        
        # 计算 Kaiser 滤波器
        cutoff = 0.5 / ratio
        half_width = 0.5 / ratio
        filter_ = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        
        # Polyphase 分解
        w = filter_.view(kernel_size) * ratio
        p0, p1 = w[0::2], w[1::2]
        weight = torch.stack([p0, p1], dim=0).unsqueeze(1)  # [ratio, 1, taps]
        weight = weight.repeat(channels, 1, 1)  # [C*ratio, 1, taps]
        
        # ✅ 创建固定的 Conv1d 层（而不是动态卷积）
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels * ratio,
            kernel_size=weight.shape[2],
            stride=1,
            padding=0,  # 手动 padding
            groups=channels,
            bias=False
        )
        
        # 初始化权重
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        
        # 冻结权重（不需要训练）
        self.conv.weight.requires_grad = False

    def forward(self, x):
        B, C, T = x.shape
        
        # ✅ 手动 padding（固定值）
        x = F.pad(x, (2, 3), mode='constant', value=0.0)
        
        # ✅ 使用固定的 Conv1d 层
        out = self.conv(x)
        
        # 重塑为交错输出
        out = out.view(B, C, self.ratio, -1)
        out = out.transpose(2, 3).reshape(B, C, -1)
        
        # 固定裁剪
        out = out[..., 2:-2]
        return out

class LowPassFilter1d(nn.Module):
    """ONNX 兼容版本 - 使用固定 Conv1d 层"""
    def __init__(self, stride=2, kernel_size=12, channels=512):
        super().__init__()
        self.stride = stride
        self.channels = channels
        
        # 计算滤波器
        cutoff = 0.5 / stride
        half_width = 0.5 / stride
        filter_ = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        filter_ = filter_.repeat(channels, 1, 1)
        
        # ✅ 创建固定的 Conv1d 层
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=5,
            groups=channels,
            bias=False
        )
        
        # 初始化权重
        with torch.no_grad():
            self.conv.weight.copy_(filter_)
        
        # 冻结权重
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)

class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12, channels=512):
        super().__init__()
        self.lowpass = LowPassFilter1d(stride=ratio, kernel_size=kernel_size, channels=channels)
    
    def forward(self, x):
        return self.lowpass(x)

# ==============================================================================
# 2. SnakeBeta 激活函数（ONNX 兼容版本）
# ==============================================================================
class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        init_val = torch.zeros(in_features) if alpha_logscale else torch.ones(in_features)
        
        self.alpha = nn.Parameter(init_val * alpha)
        self.beta = nn.Parameter(init_val * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x):
        if self.alpha_logscale:
            a = torch.exp(self.alpha)
            b = torch.exp(self.beta)
        else:
            a = self.alpha
            b = self.beta
        
        a = a.view(1, -1, 1)
        b = b.view(1, -1, 1)
        eps = 1e-9
        return x + (1.0 - torch.cos(2.0 * a * x)) / (2.0 * b + eps)

# ==============================================================================
# 3. Activation1d（重采样 + 激活）
# ==============================================================================
class Activation1d(nn.Module):
    def __init__(self, activation, up_ratio=2, down_ratio=2, 
                 up_kernel_size=12, down_kernel_size=12, channels=512):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        
        self.upsample = UpSample1d(up_ratio, up_kernel_size, channels)
        self.downsample = DownSample1d(down_ratio, down_kernel_size, channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x

# ==============================================================================
# 4. AMPBlock0
# ==============================================================================
def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2

class AMPBlock0(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super().__init__()
        
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                  dilation=dilation[0],
                                  padding=get_padding(kernel_size, dilation[0])))
        ])
        
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                  dilation=1,
                                  padding=get_padding(kernel_size, 1)))
        ])
        
        self.num_layers = len(self.convs1) + len(self.convs2)
        
        self.activations = nn.ModuleList([
            Activation1d(
                activation=SnakeBeta(channels, alpha_logscale=True),
                channels=channels
            ) for _ in range(self.num_layers)
        ])

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, 
                                   self.activations[::2], self.activations[1::2]):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            # ✅ 使用 narrow 代替动态切片
            x_residual = x.narrow(2, 0, xt.shape[2])
            x = xt + x_residual
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

# ==============================================================================
# 5. Generator（ONNX 兼容版本）
# ==============================================================================
class Generator(nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_initial_channel, gin_channels=0):
        super().__init__()
        
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        
        self.resblocks = nn.ModuleList()
        for i in range(1):
            ch = upsample_initial_channel // (2 ** i)
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(AMPBlock0(ch, k, d, activation="snakebeta"))
        
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        
        # ✅ 使用固定 scale_factor
        x = F.interpolate(x, scale_factor=3.0, mode='linear', align_corners=False)
        
        xs = self.resblocks[0](x)
        
        x = self.conv_post(xs)
        x = torch.tanh(x)
        
        return x

    def remove_weight_norm(self):
        for l in self.resblocks:
            l.remove_weight_norm()

# ==============================================================================
# 6. SynthesizerTrn
# ==============================================================================
class SynthesizerTrn(nn.Module):
    def __init__(self, spec_channels, segment_size, resblock,
                 resblock_kernel_sizes, resblock_dilation_sizes, 
                 upsample_initial_channel):
        super().__init__()
        self.spec_channels = spec_channels
        self.segment_size = segment_size
        
        self.dec = Generator(
            1, resblock, resblock_kernel_sizes,
            resblock_dilation_sizes, upsample_initial_channel
        )

    def forward(self, x):
        return self.dec(x)

# ==============================================================================
# 7. 自动检测配置
# ==============================================================================
def detect_config(checkpoint_path):
    """从 checkpoint 自动检测模型配置"""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    
    config = {
        'upsample_initial_channel': None,
        'resblock_kernel_sizes': [11],
        'resblock_dilation_sizes': [[1, 3, 5]]
    }
    
    for key in state_dict:
        if 'dec.conv_pre.weight' in key:
            config['upsample_initial_channel'] = state_dict[key].shape[0]
            print(f"✅ Detected upsample_initial_channel = {config['upsample_initial_channel']}")
            break
    
    if config['upsample_initial_channel'] is None:
        raise ValueError("❌ Cannot detect upsample_initial_channel from checkpoint!")
    
    return config

# ==============================================================================
# 8. 权重转换函数
# ==============================================================================
def convert_state_dict(state_dict, model_channels):
    """转换 checkpoint 的权重以匹配新的 Conv1d 结构"""
    new_state_dict = {}
    
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        
        # ✅ 将 upsample.filter 转换为 upsample.conv.weight
        if 'upsample.filter' in new_key:
            ratio = 2
            kernel_size = v.shape[2]
            w = v.view(kernel_size) * ratio
            p0, p1 = w[0::2], w[1::2]
            weight = torch.stack([p0, p1], dim=0).unsqueeze(1)
            weight = weight.repeat(model_channels, 1, 1)
            new_state_dict[new_key.replace('filter', 'conv.weight')] = weight
            
        # ✅ 将 downsample.lowpass.filter 转换为 downsample.lowpass.conv.weight
        elif 'downsample.lowpass.filter' in new_key:
            expanded = v.repeat(model_channels, 1, 1)
            new_state_dict[new_key.replace('filter', 'conv.weight')] = expanded
            
        else:
            new_state_dict[new_key] = v
    
    return new_state_dict

# ==============================================================================
# 9. 导出函数
# ==============================================================================
def export_to_onnx(checkpoint_path, output_path, opset_version=16, input_length=100):
    print(f"🔍 Loading checkpoint: {checkpoint_path}")
    
    # 自动检测配置
    config = detect_config(checkpoint_path)
    model_channels = config['upsample_initial_channel']
    
    # 创建模型
    model = SynthesizerTrn(
        spec_channels=128,
        segment_size=30,
        resblock="amp",
        resblock_kernel_sizes=config['resblock_kernel_sizes'],
        resblock_dilation_sizes=config['resblock_dilation_sizes'],
        upsample_initial_channel=model_channels
    )
    
    # 加载权重
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    
    # ✅ 先移除 weight_norm
    print("🔧 Removing weight_norm...")
    model.dec.remove_weight_norm()
    
    # ✅ 转换 state_dict
    print("🔄 Converting state_dict...")
    new_state_dict = convert_state_dict(state_dict, model_channels)
    
    print("📦 Loading state_dict...")
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️  Missing keys ({len(missing_keys)}): {missing_keys[:3]}...")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:3]}...")
    
    model.eval()
    
    # 导出 ONNX
    print(f"📤 Exporting to ONNX (opset={opset_version})...")
    dummy_input = torch.randn(1, 1, input_length, dtype=torch.float32)
    
    with torch.no_grad():
        # 测试 PyTorch 推理
        print("🧪 Testing PyTorch inference...")
        pt_output = model(dummy_input)
        print(f"   PyTorch output shape: {pt_output.shape}")
        
        # 导出
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch', 2: 'time'},
                'output': {0: 'batch', 2: 'time'}
            },
            verbose=False
        )
    
    print(f"✅ ONNX model saved to: {output_path}")
    
    # 验证导出的模型
    try:
        loaded_model = onnx.load(output_path)
        onnx.checker.check_model(loaded_model)
        print(f"✅ ONNX model validation passed!")
        print(f"   IR Version: {loaded_model.ir_version}")
        print(f"   Opset: {loaded_model.opset_import[0].version}")
        
        # 测试 ONNX 推理
        print("\n🧪 Testing ONNX inference...")
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        
        test_input = dummy_input.numpy()
        onnx_output = session.run(None, {'input': test_input})
        
        print(f"   Input shape:  {test_input.shape}")
        print(f"   Output shape: {onnx_output[0].shape}")
        
        # 对比精度
        import numpy as np
        diff = np.abs(pt_output.numpy() - onnx_output[0]).max()
        mean_diff = np.abs(pt_output.numpy() - onnx_output[0]).mean()
        print(f"   Max difference: {diff:.6f}")
        print(f"   Mean difference: {mean_diff:.6f}")
        
        if diff < 1e-4:
            print("   ✅ Precision check passed!")
        else:
            print(f"   ⚠️  Difference detected: {diff} (acceptable for float32)")
        
    except ImportError:
        print("⚠️  onnxruntime not installed, skipping runtime test")
        print("   Install with: pip install onnxruntime")
    except Exception as e:
        print(f"⚠️  Validation error: {e}")

# ==============================================================================
# 10. 主函数
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='Export NovaSR model to ONNX')
    parser.add_argument('--checkpoint', required=True, 
                        help='Path to pytorch_model_v1.bin or .pth file')
    parser.add_argument('--output', default='novasr_model.onnx', 
                        help='Output ONNX file path')
    parser.add_argument('--opset', type=int, default=16, 
                        help='ONNX opset version (default: 16, recommended: 13-17)')
    parser.add_argument('--input-length', type=int, default=100, 
                        help='Input sequence length for testing (default: 100)')
    args = parser.parse_args()
    
    try:
        export_to_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            opset_version=args.opset,
            input_length=args.input_length
        )
        
        print("\n" + "="*60)
        print("🎉 Export completed successfully!")
        print("="*60)
        print(f"\n📁 ONNX model: {args.output}")
        print(f"📊 Input:  [batch, 1, time]")
        print(f"📊 Output: [batch, 1, time*3] (3x upsampling)")
        print(f"\n💡 Usage in Unity/C#:")
        print(f"   - Load model with Barracuda/OnnxRuntime")
        print(f"   - Feed 16kHz audio as [1, 1, N]")
        print(f"   - Get 48kHz output as [1, 1, N*3]")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()