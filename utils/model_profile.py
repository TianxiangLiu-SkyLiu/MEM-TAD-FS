import torch
import torch.nn as nn


def _format_count(value):
    value = float(value)
    if value >= 1e12:
        return "{:.3f}T".format(value / 1e12)
    if value >= 1e9:
        return "{:.3f}B".format(value / 1e9)
    if value >= 1e6:
        return "{:.3f}M".format(value / 1e6)
    if value >= 1e3:
        return "{:.3f}K".format(value / 1e3)
    return "{:.0f}".format(value)


def _resolve_feature_size(cfg, model):
    feature_size = cfg.get("feature_size", None)
    if isinstance(feature_size, (list, tuple)) and len(feature_size) == 4:
        c, t, h, w = [int(x) for x in feature_size]
        return c, t, h, w

    if hasattr(model, "feature_channels") and hasattr(model, "feature_t") and hasattr(model, "feature_h") and hasattr(model, "feature_w"):
        return int(model.feature_channels), int(model.feature_t), int(model.feature_h), int(model.feature_w)

    raise ValueError("Cannot resolve feature_size from cfg/model; expected cfg['feature_size'] = [C, T, H, W]")


def profile_model_complexity(model, cfg, device):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    profile = {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "flops": None,
        "input_shape": None,
        "error": "",
    }

    try:
        c, t, h, w = _resolve_feature_size(cfg, model)
        dummy = torch.zeros((1, c, t, h, w), dtype=torch.float32, device=device)
        profile["input_shape"] = tuple(dummy.shape)

        hook_handles = []
        flops_acc = {"value": 0.0}

        def _conv_flops(module, _inp, out):
            if not torch.is_tensor(out):
                return
            out_shape = out.shape
            if len(out_shape) == 5:
                n, cout, od, oh, ow = out_shape
                kernel_mul = module.kernel_size[0] * module.kernel_size[1] * module.kernel_size[2]
                cin_group = module.in_channels // module.groups
                macs = float(n) * float(cout) * float(od) * float(oh) * float(ow) * float(cin_group) * float(kernel_mul)
            elif len(out_shape) == 4:
                n, cout, oh, ow = out_shape
                kernel_mul = module.kernel_size[0] * module.kernel_size[1]
                cin_group = module.in_channels // module.groups
                macs = float(n) * float(cout) * float(oh) * float(ow) * float(cin_group) * float(kernel_mul)
            elif len(out_shape) == 3:
                n, cout, ow = out_shape
                kernel_mul = module.kernel_size[0]
                cin_group = module.in_channels // module.groups
                macs = float(n) * float(cout) * float(ow) * float(cin_group) * float(kernel_mul)
            else:
                return
            flops_acc["value"] += 2.0 * macs

        def _linear_flops(module, _inp, out):
            if not torch.is_tensor(out):
                return
            out_elems = float(out.numel())
            macs = out_elems * float(module.in_features)
            flops_acc["value"] += 2.0 * macs

        for m in model.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                hook_handles.append(m.register_forward_hook(_conv_flops))
            elif isinstance(m, nn.Linear):
                hook_handles.append(m.register_forward_hook(_linear_flops))

        was_training = model.training
        model.eval()
        try:
            if hasattr(model, "reset_memory"):
                model.reset_memory()
            with torch.no_grad():
                _ = model(dummy, decode=True)
            if hasattr(model, "reset_memory"):
                model.reset_memory()
        finally:
            if was_training:
                model.train()
            for handle in hook_handles:
                handle.remove()

        profile["flops"] = int(flops_acc["value"])
    except Exception as exc:
        profile["error"] = str(exc)

    return profile


def print_model_complexity(profile, prefix="Model"):
    total_params = profile.get("total_params", 0)
    trainable_params = profile.get("trainable_params", 0)
    print(
        "[{}] Params: total={} ({}) | trainable={} ({})".format(
            prefix,
            total_params,
            _format_count(total_params),
            trainable_params,
            _format_count(trainable_params),
        )
    )

    if profile.get("flops", None) is not None:
        print(
            "[{}] FLOPs (single forward, decode=True, input={}): {} ({})".format(
                prefix,
                profile.get("input_shape", "N/A"),
                profile["flops"],
                _format_count(profile["flops"]),
            )
        )
    else:
        print(
            "[{}] FLOPs: unavailable ({})".format(
                prefix,
                profile.get("error", "unknown reason"),
            )
        )
