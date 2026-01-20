import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
import numpy as np
import onnxruntime as ort
import argparse
from pathlib import Path
import warnings
import math

warnings.filterwarnings("ignore")

# ==============================================================================
# 完整的模型定义（从 export_onnx.py 复制）
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
    def __init__(self, ratio=2, kernel_size=12, channels=512):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.channels = channels
        
        cutoff = 0.5 / ratio
        half_width = 0.5 / ratio
        filter_ = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        
        w = filter_.view(kernel_size) * ratio
        p0, p1 = w[0::2], w[1::2]
        weight = torch.stack([p0, p1], dim=0).unsqueeze(1)
        weight = weight.repeat(channels, 1, 1)
        
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels * ratio,
            kernel_size=weight.shape[2],
            stride=1,
            padding=0,
            groups=channels,
            bias=False
        )
        
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        self.conv.weight.requires_grad = False

    def forward(self, x):
        x = F.pad(x, (2, 3), mode='constant', value=0.0)
        out = self.conv(x)
        out = out.view(x.shape[0], x.shape[1], self.ratio, -1)
        out = out.transpose(2, 3).reshape(x.shape[0], x.shape[1], -1)
        out = out[..., 2:-2]
        return out

class LowPassFilter1d(nn.Module):
    def __init__(self, stride=2, kernel_size=12, channels=512):
        super().__init__()
        self.stride = stride
        self.channels = channels
        
        cutoff = 0.5 / stride
        half_width = 0.5 / stride
        filter_ = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        filter_ = filter_.repeat(channels, 1, 1)
        
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=5,
            groups=channels,
            bias=False
        )
        
        with torch.no_grad():
            self.conv.weight.copy_(filter_)
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)

class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12, channels=512):
        super().__init__()
        self.lowpass = LowPassFilter1d(stride=ratio, kernel_size=kernel_size, channels=channels)
    
    def forward(self, x):
        return self.lowpass(x)

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
            x_residual = x.narrow(2, 0, xt.shape[2])
            x = xt + x_residual
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

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
        x = F.interpolate(x, scale_factor=3.0, mode='linear', align_corners=False)
        xs = self.resblocks[0](x)
        x = self.conv_post(xs)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for l in self.resblocks:
            l.remove_weight_norm()

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
# 权重转换函数
# ==============================================================================

def convert_state_dict(state_dict, model_channels):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        if 'upsample.filter' in new_key:
            ratio = 2
            kernel_size = v.shape[2]
            w = v.view(kernel_size) * ratio
            p0, p1 = w[0::2], w[1::2]
            weight = torch.stack([p0, p1], dim=0).unsqueeze(1)
            weight = weight.repeat(model_channels, 1, 1)
            new_state_dict[new_key.replace('filter', 'conv.weight')] = weight
        elif 'downsample.lowpass.filter' in new_key:
            expanded = v.repeat(model_channels, 1, 1)
            new_state_dict[new_key.replace('filter', 'conv.weight')] = expanded
        else:
            new_state_dict[new_key] = v
    return new_state_dict

# ==============================================================================
# 加载 PyTorch 模型
# ==============================================================================

def load_pytorch_model(checkpoint_path):
    print("\n📦 加载原始 PyTorch 模型...")
    
    # 检测配置
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    
    model_channels = None
    for key in state_dict:
        if 'dec.conv_pre.weight' in key:
            model_channels = state_dict[key].shape[0]
            break
    
    if model_channels is None:
        raise ValueError("Cannot detect model channels")
    
    print(f"   检测到通道数: {model_channels}")
    
    # 创建模型
    model = SynthesizerTrn(
        spec_channels=128,
        segment_size=30,
        resblock="amp",
        resblock_kernel_sizes=[11],
        resblock_dilation_sizes=[[1, 3, 5]],
        upsample_initial_channel=model_channels
    )
    
    # 加载权重
    model.dec.remove_weight_norm()
    new_state_dict = convert_state_dict(state_dict, model_channels)
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    print("   ✅ PyTorch 模型加载成功")
    return model

# ==============================================================================
# 对比测试
# ==============================================================================

def compare_models(checkpoint_path, onnx_path, test_length=16000, save_outputs=False):
    print("="*70)
    print("🔍 对比 PyTorch 模型 vs ONNX 模型")
    print("="*70)
    
    # 1. 加载 PyTorch 模型
    try:
        pt_model = load_pytorch_model(checkpoint_path)
    except Exception as e:
        print(f"❌ 加载 PyTorch 模型失败: {e}")
        return False
    
    # 2. 加载 ONNX 模型
    print("\n📦 加载 ONNX 模型...")
    try:
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        print(f"   ✅ ONNX 模型加载成功")
        print(f"   输入: {input_name}")
        print(f"   输出: {output_name}")
    except Exception as e:
        print(f"❌ 加载 ONNX 模型失败: {e}")
        return False
    
    # 3. 创建测试数据
    print(f"\n🧪 创建测试数据 (长度={test_length})...")
    
    test_cases = []
    
    # 测试1: 静音
    silent_np = np.zeros((1, 1, test_length), dtype=np.float32)
    silent_torch = torch.from_numpy(silent_np)
    test_cases.append(("静音", silent_torch, silent_np))
    
    # 测试2: 正弦波 440Hz
    t = np.linspace(0, test_length/16000, test_length)
    sine_wave = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sine_np = sine_wave.reshape(1, 1, -1)
    sine_torch = torch.from_numpy(sine_np)
    test_cases.append(("正弦波 (440 Hz)", sine_torch, sine_np))
    
    # 测试3: 白噪声
    noise_np = (np.random.randn(1, 1, test_length) * 0.1).astype(np.float32)
    noise_torch = torch.from_numpy(noise_np)
    test_cases.append(("白噪声", noise_torch, noise_np))
    
    # 测试4: 脉冲
    impulse_np = np.zeros((1, 1, test_length), dtype=np.float32)
    impulse_np[0, 0, test_length//2] = 1.0
    impulse_torch = torch.from_numpy(impulse_np)
    test_cases.append(("脉冲", impulse_torch, impulse_np))
    
    # 4. 对比推理结果
    print("\n📊 对比推理结果...\n")
    print("-" * 70)
    
    all_passed = True
    max_differences = []
    
    for test_name, torch_input, np_input in test_cases:
        print(f"\n测试: {test_name}")
        print("-" * 70)
        
        try:
            # PyTorch 推理
            with torch.no_grad():
                pt_output = pt_model(torch_input).numpy()
            
            # ONNX 推理
            onnx_output = ort_session.run([output_name], {input_name: np_input})[0]
            
            # 计算差异
            abs_diff = np.abs(pt_output - onnx_output)
            max_diff = abs_diff.max()
            mean_diff = abs_diff.mean()
            
            # 计算统计信息
            pt_rms = np.sqrt(np.mean(pt_output**2))
            onnx_rms = np.sqrt(np.mean(onnx_output**2))
            pt_max = np.abs(pt_output).max()
            onnx_max = np.abs(onnx_output).max()
            
            # 检查异常值
            pt_has_nan = np.isnan(pt_output).any()
            onnx_has_nan = np.isnan(onnx_output).any()
            pt_has_inf = np.isinf(pt_output).any()
            onnx_has_inf = np.isinf(onnx_output).any()
            
            # 判断是否通过
            passed = (max_diff < 1e-3 and 
                     not pt_has_nan and not onnx_has_nan and 
                     not pt_has_inf and not onnx_has_inf)
            
            status = "✅" if passed else "❌"
            all_passed = all_passed and passed
            max_differences.append(max_diff)
            
            print(f"{status} 形状对比:")
            print(f"   PyTorch:  {pt_output.shape}")
            print(f"   ONNX:     {onnx_output.shape}")
            
            print(f"\n{status} 数值对比:")
            print(f"   最大差异:  {max_diff:.9f}")
            print(f"   平均差异:  {mean_diff:.9f}")
            print(f"   相对误差:  {(max_diff / (pt_max + 1e-9)):.6f}")
            
            print(f"\n{status} 统计信息:")
            print(f"   PyTorch  - RMS: {pt_rms:.6f}, Max: {pt_max:.6f}, NaN: {pt_has_nan}, Inf: {pt_has_inf}")
            print(f"   ONNX     - RMS: {onnx_rms:.6f}, Max: {onnx_max:.6f}, NaN: {onnx_has_nan}, Inf: {onnx_has_inf}")
            
            if not passed:
                print(f"\n⚠️  差异较大或检测到异常值！")
                if max_diff >= 1e-3:
                    print(f"   - 最大差异 {max_diff:.6f} 超过阈值 0.001")
                if pt_has_nan or onnx_has_nan:
                    print(f"   - 检测到 NaN 值")
                if pt_has_inf or onnx_has_inf:
                    print(f"   - 检测到 Inf 值")
            
            # 保存输出（可选）
            if save_outputs:
                np.save(f'pt_output_{test_name.replace(" ", "_")}.npy', pt_output)
                np.save(f'onnx_output_{test_name.replace(" ", "_")}.npy', onnx_output)
                print(f"\n💾 已保存输出到文件")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # 5. 最终总结
    print("\n" + "="*70)
    print("📊 总结")
    print("="*70)
    
    if all_passed:
        print("✅ 所有测试通过！ONNX 模型与 PyTorch 模型输出一致")
        print(f"\n📈 最大差异范围: {min(max_differences):.9f} - {max(max_differences):.9f}")
        print("\n💡 如果 Unity 中效果仍然不好，可能的原因:")
        print("   1. 输入音频采样率不是 16kHz")
        print("   2. 立体声到单声道转换有误")
        print("   3. 输出 AudioClip 的采样率未设置为 48kHz")
        print("   4. 音频数据被意外修改（如裁剪、归一化等）")
    else:
        print("❌ 部分测试未通过！ONNX 模型可能有问题")
        print(f"\n📈 最大差异: {max(max_differences):.9f}")
        print("\n🔧 建议:")
        print("   1. 重新导出模型，使用更低的 opset:")
        print("      python export_onnx.py --checkpoint pytorch_model_v2.bin --opset 13")
        print("   2. 检查导出脚本中的模型定义是否正确")
        print("   3. 尝试使用 torch.onnx.export 的 verbose=True 查看详细信息")
    
    print("="*70)
    
    return all_passed

# ==============================================================================
# 主函数
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='对比验证 ONNX 模型与 PyTorch 模型')
    parser.add_argument('--checkpoint', required=True, help='PyTorch checkpoint 路径')
    parser.add_argument('--onnx', required=True, help='ONNX 模型路径')
    parser.add_argument('--test-length', type=int, default=16000, 
                       help='测试音频长度（默认 16000 = 1秒@16kHz）')
    parser.add_argument('--save-outputs', action='store_true',
                       help='保存输出到 .npy 文件')
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint 不存在: {args.checkpoint}")
        return
    
    if not Path(args.onnx).exists():
        print(f"❌ ONNX 模型不存在: {args.onnx}")
        return
    
    compare_models(args.checkpoint, args.onnx, args.test_length, args.save_outputs)

if __name__ == '__main__':
    main()