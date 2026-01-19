import torch
import torch.onnx
import argparse
import json
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm, remove_weight_norm
import math

# ============================================================================
# 模型定义部分
# ============================================================================

# --- Utility Functions ---
def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

# --- Activation Functions ---
@torch.jit.script
def snake_fast_inference(x: torch.Tensor, a: torch.Tensor, inv_2b: torch.Tensor) -> torch.Tensor:
    return x + (1.0 - torch.cos(2.0 * a * x)) * inv_2b

class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        init_val = torch.zeros(in_features) if alpha_logscale else torch.ones(in_features)
        
        self.alpha = nn.Parameter(init_val * alpha)
        self.beta = nn.Parameter(init_val * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        
        self.register_buffer('a_eff', torch.ones(1, in_features, 1), persistent=False)
        self.register_buffer('inv_2b', torch.ones(1, in_features, 1), persistent=False)
        self._is_prepared = False
    
    def prepare(self):
        with torch.no_grad():
            a = (torch.exp(self.alpha) if self.alpha_logscale else self.alpha).view(1, -1, 1)
            b = (torch.exp(self.beta) if self.alpha_logscale else self.beta).view(1, -1, 1)
            self.a_eff.copy_(a)
            self.inv_2b.copy_(1.0 / (2.0 * b + 1e-9))
        self._is_prepared = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_prepared and not self.training:
            self.prepare()
        if not self.training:
            return snake_fast_inference(x, self.a_eff, self.inv_2b)
        
        a = (torch.exp(self.alpha) if self.alpha_logscale else self.alpha).view(1, -1, 1)
        b = (torch.exp(self.beta) if self.alpha_logscale else self.beta).view(1, -1, 1)
        return x + (1.0 - torch.cos(2.0 * a * x)) / (2.0 * b + 1e-9)

# --- Resample Modules ---
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

@torch.jit.script
def _polyphase_upsample_fused(x: torch.Tensor, weight: torch.Tensor, ratio: int):
    x = F.pad(x, (2, 3))
    out = F.conv1d(x, weight, groups=x.shape[1], stride=1)
    B, C_out, L = out.shape
    C = x.shape[1]
    out = out.view(B, C, ratio, L).transpose(2, 3).reshape(B, C, -1)
    return out[..., 2:-2]

class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12, channels=512):
        super().__init__()
        self.ratio = ratio
        self.channels = channels
        self.kernel_size = kernel_size
        self.register_buffer("filter", torch.zeros(1, 1, 12))
        self.register_buffer("f_fast", torch.zeros(channels * ratio, 1, 6), persistent=False)
        self._prepared = False
    
    def prepare(self):
        with torch.no_grad():
            w = self.filter * float(self.ratio)
            w = w.view(self.kernel_size)
            p0, p1 = w[0::2], w[1::2]
            fast_w = torch.stack([p0, p1], dim=0).unsqueeze(0).expand(self.channels, -1, -1)
            fast_w = fast_w.reshape(self.channels * self.ratio, 1, 6)
            self.f_fast.copy_(fast_w)
        self._prepared = True
    
    def forward(self, x: torch.Tensor):
        if not self._prepared and not self.training: 
            self.prepare()
        return _polyphase_upsample_fused(x, self.f_fast[:x.shape[1]*self.ratio], self.ratio)

class LowPassFilter1d(nn.Module):
    def __init__(self, stride=1, kernel_size=12, channels=512):
        super().__init__()
        self.stride = stride
        self.channels = channels
        self.kernel_size = kernel_size
        self.register_buffer("filter", torch.zeros(1, 1, 12))
        self.register_buffer("f_opt", torch.zeros(channels, 1, 12), persistent=False)
        self._prepared = False
    
    def prepare(self):
        with torch.no_grad():
            self.f_opt.copy_(self.filter.expand(self.channels, -1, -1))
        self._prepared = True
    
    def forward(self, x: torch.Tensor):
        if not self._prepared and not self.training: 
            self.prepare()
        C = x.shape[1]
        return F.conv1d(x, self.f_opt[:C], stride=self.stride, padding=5, groups=C)

class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=12, channels=512):
        super().__init__()
        self.lowpass = LowPassFilter1d(ratio, kernel_size, channels)
    
    def forward(self, x):
        return self.lowpass(x)

# --- Activation1d Module ---
class Activation1d(nn.Module):
    def __init__(self, activation, up_ratio=2, down_ratio=2, 
                 up_kernel_size=12, down_kernel_size=12):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x

# --- AMPBlock0 ---
class AMPBlock0(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super(AMPBlock0, self).__init__()
        
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0])))
        ])
        self.convs1.apply(init_weights)
        
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
        
        self.num_layers = len(self.convs1) + len(self.convs2)
        
        self.activations = nn.ModuleList([
            Activation1d(activation=SnakeBeta(channels, alpha_logscale=True))
            for _ in range(self.num_layers)
        ])
    
    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, 
                                 self.activations[::2], self.activations[1::2]):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x[:, :, :xt.shape[2]]
        return x
    
    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

# --- Generator ---
class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, 
                 resblock_dilation_sizes, upsample_initial_channel, gin_channels=0):
        super(Generator, self).__init__()
        
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = AMPBlock0
        
        self.resblocks = nn.ModuleList()
        for i in range(1):
            ch = upsample_initial_channel//(2**(i))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d, activation="snakebeta"))
        
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
    
    def forward(self, x, g=None):
        x = self.conv_pre(x)
        x = F.interpolate(x, int(x.shape[-1] * 3), mode='linear')
        xs = self.resblocks[0](x)
        x = self.conv_post(xs)
        x = torch.tanh(x)
        return x
    
    def remove_weight_norm(self):
        for l in self.resblocks:
            l.remove_weight_norm()

# --- SynthesizerTrn ---
class SynthesizerTrn(nn.Module):
    def __init__(self, spec_channels, segment_size, resblock, 
                 resblock_kernel_sizes, resblock_dilation_sizes, 
                 upsample_initial_channel):
        super().__init__()
        self.spec_channels = spec_channels
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_initial_channel = upsample_initial_channel
        self.segment_size = segment_size
        
        self.dec = Generator(1, resblock, resblock_kernel_sizes, 
                            resblock_dilation_sizes, upsample_initial_channel)
    
    def forward(self, x):
        y = self.dec(x)
        return y
    
    @torch.no_grad()
    def infer(self, x, max_len=None):
        o = self.dec(x[:,:,:max_len])
        return o

# ============================================================================
# 转换功能部分
# ============================================================================

def detect_model_config(checkpoint_path):
    """从checkpoint中检测模型配置"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 检测upsample_initial_channel
    for key in state_dict.keys():
        if 'dec.conv_pre.weight' in key:
            upsample_initial_channel = state_dict[key].shape[0]
            print(f"检测到 upsample_initial_channel: {upsample_initial_channel}")
            return upsample_initial_channel
    
    return 512  # 默认值


def load_checkpoint(checkpoint_path, model):
    """加载模型权重"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    return model


def prepare_model_for_export(model):
    """准备模型用于导出"""
    model.eval()
    
    for module in model.modules():
        if hasattr(module, 'prepare'):
            module.prepare()
    
    return model


def export_to_onnx(
    checkpoint_path,
    output_path,
    config_path=None,
    opset_version=15,
    input_length=None
):
    """将SynthesizerTrn模型导出为ONNX格式"""
    
    # 首先检测模型配置
    print("检测模型配置...")
    detected_channels = detect_model_config(checkpoint_path)
    
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "spec_channels": 513,
            "segment_size": 32,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_initial_channel": detected_channels
        }
    
    print(f"使用配置: upsample_initial_channel={config['upsample_initial_channel']}")
    print("创建模型...")
    model = SynthesizerTrn(
        spec_channels=config["spec_channels"],
        segment_size=config["segment_size"],
        resblock=config["resblock"],
        resblock_kernel_sizes=config["resblock_kernel_sizes"],
        resblock_dilation_sizes=config["resblock_dilation_sizes"],
        upsample_initial_channel=config["upsample_initial_channel"]
    )
    
    print(f"加载权重: {checkpoint_path}")
    model = load_checkpoint(checkpoint_path, model)
    
    print("准备模型用于导出...")
    model = prepare_model_for_export(model)
    
    if input_length is None:
        input_length = 100
    
    dummy_input = torch.randn(1, 1, input_length)
    
    dynamic_axes = {
        'input': {2: 'length'},
        'output': {2: 'length'}
    }
    
    print(f"开始导出ONNX模型到: {output_path}")
    print(f"输入形状: {dummy_input.shape}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print("✓ ONNX导出完成!")
    
    try:
        import onnx
        import onnxruntime as ort
        
        print("\n验证ONNX模型...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX模型验证通过!")
        
        print("\n测试ONNX推理...")
        ort_session = ort.InferenceSession(output_path)
        
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        with torch.no_grad():
            torch_output = model(dummy_input)
        
        diff = torch.abs(torch.from_numpy(ort_outputs[0]) - torch_output).max()
        print(f"PyTorch vs ONNX最大差异: {diff.item():.6f}")
        
        if diff < 1e-4:
            print("✓ 推理结果匹配!")
        else:
            print("⚠ 推理结果存在差异，但可能在可接受范围内")
            
    except ImportError:
        print("\n未安装onnx或onnxruntime，跳过验证")
        print("安装命令: pip install onnx onnxruntime")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='将PyTorch模型转换为ONNX格式')
    parser.add_argument('--checkpoint', '-c', required=True, help='模型checkpoint路径')
    parser.add_argument('--output', '-o', default='model.onnx', help='输出ONNX文件路径')
    parser.add_argument('--config', help='配置文件路径(JSON格式)')
    parser.add_argument('--opset', type=int, default=15, help='ONNX opset版本')
    parser.add_argument('--input-length', type=int, default=100, 
                       help='示例输入长度')
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=str(output_path),
        config_path=args.config,
        opset_version=args.opset,
        input_length=args.input_length
    )


if __name__ == '__main__':
    main()