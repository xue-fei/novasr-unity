import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
import argparse
import warnings
import onnx 

# 忽略 weight_norm 警告
warnings.filterwarnings("ignore", message=".*weight_norm is deprecated.*")

# ==============================================================================
# 1. ONNX 兼容的 SnakeBeta（无 TorchScript）
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
# 2. 简化重采样（ONNX 友好）
# ==============================================================================
class UpSample1d(nn.Module):
    def __init__(self, ratio=2, **kwargs):
        super().__init__()
        self.ratio = ratio

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.ratio, mode='linear', align_corners=False)

class DownSample1d(nn.Module):
    def __init__(self, ratio=2, **kwargs):
        super().__init__()
        self.ratio = ratio
        self.pool = nn.AvgPool1d(kernel_size=ratio, stride=ratio, padding=0)

    def forward(self, x):
        return self.pool(x)

class Activation1d(nn.Module):
    def __init__(self, activation, up_ratio=2, down_ratio=2, **kwargs):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio)
        self.downsample = DownSample1d(down_ratio)

    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x

# ==============================================================================
# 3. AMPBlock0 + Generator
# ==============================================================================
def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2

class AMPBlock0(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                                  padding=get_padding(kernel_size, dilation[0])))
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                                  padding=get_padding(kernel_size, 1)))
        ])
        self.activations = nn.ModuleList([
            Activation1d(SnakeBeta(channels, alpha_logscale=True)),
            Activation1d(SnakeBeta(channels, alpha_logscale=True))
        ])

    def forward(self, x):
        xt = self.activations[0](x)
        xt = self.convs1[0](xt)
        xt = self.activations[1](xt)
        xt = self.convs2[0](xt)
        x = xt + x[:, :, :xt.shape[2]]
        return x

    def remove_weight_norm(self):
        for l in self.convs1: remove_weight_norm(l)
        for l in self.convs2: remove_weight_norm(l)

class Generator(nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_initial_channel, gin_channels=0):
        super().__init__()
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        ch = upsample_initial_channel
        self.resblocks = nn.ModuleList()
        for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
            self.resblocks.append(AMPBlock0(ch, k, d))
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        x = F.interpolate(x, scale_factor=3, mode='linear', align_corners=False)  # ✅ 固定上采样
        x = self.resblocks[0](x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for block in self.resblocks:
            block.remove_weight_norm()

class SynthesizerTrn(nn.Module):
    def __init__(self, spec_channels, segment_size, resblock,
                 resblock_kernel_sizes, resblock_dilation_sizes, upsample_initial_channel):
        super().__init__()
        self.dec = Generator(1, resblock, resblock_kernel_sizes,
                            resblock_dilation_sizes, upsample_initial_channel)

    def forward(self, x):
        return self.dec(x)

# ==============================================================================
# 4. 自动检测配置
# ==============================================================================
def detect_upsample_initial_channel(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    for key in state_dict:
        if 'conv_pre.weight' in key:
            channel = state_dict[key].shape[0]
            print(f"✅ Detected upsample_initial_channel = {channel}")
            return channel
    raise ValueError("❌ Cannot detect model config from checkpoint!")

# ==============================================================================
# 5. 导出函数
# ==============================================================================
def export_to_onnx(checkpoint_path, output_path, opset_version=17, input_length=100):
    # 自动检测通道数
    upsample_initial_channel = detect_upsample_initial_channel(checkpoint_path)

    # 创建模型
    model = SynthesizerTrn(
        spec_channels=1,
        segment_size=8192,
        resblock="amp",
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3, 5]],
        upsample_initial_channel=upsample_initial_channel
    )

    # 加载权重
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)

    # 移除 weight norm
    model.dec.remove_weight_norm()
    model.eval()

    # 导出
    dummy_input = torch.randn(1, 1, input_length)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'time'},
            'output': {0: 'batch', 2: 'time'}
        }
    )
    print(f"✅ ONNX model saved to: {output_path}")
    try:
        loaded_model = onnx.load(output_path)
        print(f"IR Version: {loaded_model.ir_version}")
    except Exception as e:
        print(f"⚠️ 无法验证模型: {e}")

# ==============================================================================
# 6. 主函数
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to .bin or .pth model')
    parser.add_argument('--output', default='model.onnx', help='Output ONNX path')
    parser.add_argument('--opset', type=int, default=16, help='ONNX opset version (default: 16)')
    parser.add_argument('--input-length', type=int, default=100, help='Input length for dummy input')
    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        input_length=args.input_length
    )

if __name__ == '__main__':
    main()