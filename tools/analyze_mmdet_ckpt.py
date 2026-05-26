"""分析 MaskRCNN checkpoint 结构"""
import torch

ckpt_path = "weights/mmdet/TB_R50_20260520.pth"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt.get("state_dict", ckpt)

print("=" * 70)
print(f"TB_R50_20260520.pth 完整结构  (共 {len(sd)} 个键)")
print("=" * 70)

# 按前缀分组
from collections import OrderedDict
groups = OrderedDict()
for k in sd:
    prefix = k.split(".")[0]
    groups.setdefault(prefix, []).append(k)

for prefix, keys in sorted(groups.items()):
    params = sum(sd[k].numel() for k in keys)
    print(f"\n{'─'*60}")
    print(f"  [{prefix}]  {len(keys)} 个张量, 总参数: {params:>10,d} ({params/1e6:.1f}M)")
    
    subgroups = OrderedDict()
    for k in keys:
        sub = ".".join(k.split(".")[:2])
        subgroups.setdefault(sub, []).append(k)
    
    for sub, subkeys in sorted(subgroups.items()):
        sub_params = sum(sd[k].numel() for k in subkeys)
        shape_sample = list(sd[subkeys[0]].shape)
        print(f"      {sub:<45s} {str(shape_sample):>20s}  {sub_params:>8,d}")

print(f"\n{'='*70}")
print("总结: 这个 checkpoint 是 MaskRCNN (R18 backbone + FPN + ROI Head)")
print("="*70)

# backbone 输出通道
import torchvision.models as tv
model = tv.resnet18()
print("\nBackbone 各层输出通道 (ResNet18):")
for name, module in model.named_modules():
    if name in ("conv1",) or (name.startswith("layer") and name.count(".") == 0):
        if hasattr(module, "conv1"):  # BasicBlock
            out_ch = module.conv1.out_channels
            print(f"  {name:<12s} -> {out_ch} channels")
        elif hasattr(module, "out_channels"):
            print(f"  {name:<12s} -> {module.out_channels} channels")
