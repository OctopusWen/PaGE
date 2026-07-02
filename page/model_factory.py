from page.page_model import PaGE
from page.backbone import DinoV3HFBackbone
from page.fusion import SimpleBackboneWrapper



# =========================
#  model factory
# =========================
def get_page_model(model_name) -> tuple[PaGE, list]:
    factory = {
        # CrossGaze (Final)
        "page_vits_inout": page_vits_inout,
        "page_vits_inout_student": page_vits_inout_student,
        "page_vits_inout_finetune": page_vits_inout_finetune,
        "page_vitsplus_inout": page_vitsplus_inout,
        "page_vitsplus_inout_student": page_vitsplus_inout_student,
        "page_vitsplus_inout_finetune": page_vitsplus_inout_finetune,
        "page_vitb_inout": page_vitb_inout,
        "page_vitb_inout_student": page_vitb_inout_student,
        "page_vitb_inout_finetune": page_vitb_inout_finetune,
        "page_vitl_inout": page_vitl_inout,
        "page_vitl_inout_student": page_vitl_inout_student,
        "page_vitl_inout_finetune": page_vitl_inout_finetune,
        "page_vithplus_inout": page_vithplus_inout,
        "page_vithplus_inout_finetune": page_vithplus_inout_finetune,
    }
    assert model_name in factory.keys(), f"invalid model name: {model_name}"
    return factory[model_name]()


def page_vits_inout():
    scene_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vits16",
        in_size = (512, 512)
    ))
    head_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vits16",
        in_size = (256, 256)
    ))

    scene_transforms = scene_branch.get_transforms()
    head_transforms = head_branch.get_transforms()
    model = PaGE(scene_branch, head_branch, n_scene_self_attn_layers=1, n_head_self_attn_layers=1, n_scene_head_interaction_layers=5,
                      n_reg_tokens=4, inout=True, dim=256)
    return model, scene_transforms, head_transforms


def page_vits_inout_student():
    scene_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vits16",
        in_size = (512, 512)
    ), freeze_backbone=False)
    head_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vits16",
        in_size = (256, 256)
    ), freeze_backbone=False)

    scene_transforms = scene_branch.get_transforms()
    head_transforms = head_branch.get_transforms()
    model = PaGE(scene_branch, head_branch, n_scene_self_attn_layers=1, n_head_self_attn_layers=1, n_scene_head_interaction_layers=5,
                      n_reg_tokens=4, inout=True, dim=256)
    return model, scene_transforms, head_transforms


def page_vits_inout_finetune():
    scene_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vits16",
        in_size = (512, 512)
    ), freeze_backbone=False)
    head_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vits16",
        in_size = (256, 256)
    ), freeze_backbone=False)

    scene_transforms = scene_branch.get_transforms()
    head_transforms = head_branch.get_transforms()
    model = PaGE(scene_branch, head_branch, n_scene_self_attn_layers=1, n_head_self_attn_layers=1, n_scene_head_interaction_layers=5,
                      n_reg_tokens=4, inout=True, dim=256, dino_feature_dropout=0.5)
    return model, scene_transforms, head_transforms


def page_vitsplus_inout():
    scene_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vits16plus",
        in_size = (512, 512)
    ))
    head_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vits16plus",
        in_size = (256, 256)
    ))

    scene_transforms = scene_branch.get_transforms()
    head_transforms = head_branch.get_transforms()
    model = PaGE(scene_branch, head_branch, n_scene_self_attn_layers=1, n_head_self_attn_layers=1, n_scene_head_interaction_layers=5,
                      n_reg_tokens=4, inout=True, dim=256)
    return model, scene_transforms, head_transforms


def page_vitsplus_inout_student():
    scene_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vits16plus",
        in_size = (512, 512)
    ), freeze_backbone=False)
    head_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vits16plus",
        in_size = (256, 256)
    ), freeze_backbone=False)

    scene_transforms = scene_branch.get_transforms()
    head_transforms = head_branch.get_transforms()
    model = PaGE(scene_branch, head_branch, n_scene_self_attn_layers=1, n_head_self_attn_layers=1, n_scene_head_interaction_layers=5,
                      n_reg_tokens=4, inout=True, dim=256)
    return model, scene_transforms, head_transforms


def page_vitsplus_inout_finetune():
    scene_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vits16plus",
        in_size = (512, 512)
    ), freeze_backbone=False)
    head_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vits16plus",
        in_size = (256, 256)
    ), freeze_backbone=False)

    scene_transforms = scene_branch.get_transforms()
    head_transforms = head_branch.get_transforms()
    model = PaGE(scene_branch, head_branch, n_scene_self_attn_layers=1, n_head_self_attn_layers=1, n_scene_head_interaction_layers=5,
                      n_reg_tokens=4, inout=True, dim=256, dino_feature_dropout=0.5)
    return model, scene_transforms, head_transforms


def page_vitb_inout():
    scene_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vitb16",
        in_size = (512, 512)
    ))
    head_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vitb16",
        in_size = (256, 256)
    ))

    scene_transforms = scene_branch.get_transforms()
    head_transforms = head_branch.get_transforms()
    model = PaGE(scene_branch, head_branch, n_scene_self_attn_layers=1, n_head_self_attn_layers=1, n_scene_head_interaction_layers=5,
                      n_reg_tokens=4, inout=True, dim=256)
    return model, scene_transforms, head_transforms


def page_vitb_inout_student():
    scene_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vitb16",
        in_size = (512, 512)
    ), freeze_backbone=False)
    head_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vitb16",
        in_size = (256, 256)
    ), freeze_backbone=False)

    scene_transforms = scene_branch.get_transforms()
    head_transforms = head_branch.get_transforms()
    model = PaGE(scene_branch, head_branch, n_scene_self_attn_layers=1, n_head_self_attn_layers=1, n_scene_head_interaction_layers=5,
                      n_reg_tokens=4, inout=True, dim=256)
    return model, scene_transforms, head_transforms


def page_vitb_inout_finetune():
    scene_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vitb16",
        in_size = (512, 512)
    ), freeze_backbone=False)
    head_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vitb16",
        in_size = (256, 256)
    ), freeze_backbone=False)

    scene_transforms = scene_branch.get_transforms()
    head_transforms = head_branch.get_transforms()
    model = PaGE(scene_branch, head_branch, n_scene_self_attn_layers=1, n_head_self_attn_layers=1, n_scene_head_interaction_layers=5,
                      n_reg_tokens=4, inout=True, dim=256, dino_feature_dropout=0.5)
    return model, scene_transforms, head_transforms


def page_vitl_inout():
    scene_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vitl16",
        in_size = (512, 512)
    ))
    head_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vitl16",
        in_size = (256, 256)
    ))

    scene_transforms = scene_branch.get_transforms()
    head_transforms = head_branch.get_transforms()
    model = PaGE(scene_branch, head_branch, n_scene_self_attn_layers=1, n_head_self_attn_layers=1, n_scene_head_interaction_layers=5,
                      n_reg_tokens=4, inout=True, dim=256)
    return model, scene_transforms, head_transforms


def page_vitl_inout_student():
    scene_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vitl16",
        in_size = (512, 512)
    ), freeze_backbone=False)
    head_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vitl16",
        in_size = (256, 256)
    ), freeze_backbone=False)

    scene_transforms = scene_branch.get_transforms()
    head_transforms = head_branch.get_transforms()
    model = PaGE(scene_branch, head_branch, n_scene_self_attn_layers=1, n_head_self_attn_layers=1, n_scene_head_interaction_layers=5,
                      n_reg_tokens=4, inout=True, dim=256)
    return model, scene_transforms, head_transforms


def page_vitl_inout_finetune():
    scene_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vitl16",
        in_size = (512, 512)
    ), freeze_backbone=False)
    head_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vitl16",
        in_size = (256, 256)
    ), freeze_backbone=False)

    scene_transforms = scene_branch.get_transforms()
    head_transforms = head_branch.get_transforms()
    model = PaGE(scene_branch, head_branch, n_scene_self_attn_layers=1, n_head_self_attn_layers=1, n_scene_head_interaction_layers=5,
                      n_reg_tokens=4, inout=True, dim=256, dino_feature_dropout=0.5)
    return model, scene_transforms, head_transforms


def page_vithplus_inout():
    scene_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vith16plus",
        in_size = (512, 512)
    ))
    head_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vith16plus",
        in_size = (256, 256)
    ))

    scene_transforms = scene_branch.get_transforms()
    head_transforms = head_branch.get_transforms()
    model = PaGE(scene_branch, head_branch, n_scene_self_attn_layers=1, n_head_self_attn_layers=1, n_scene_head_interaction_layers=5,
                      n_reg_tokens=4, inout=True, dim=256)
    return model, scene_transforms, head_transforms


def page_vithplus_inout_finetune():
    scene_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vith16plus",
        in_size = (512, 512)
    ), freeze_backbone=False)
    head_branch = SimpleBackboneWrapper(DinoV3HFBackbone(
        model_path = "/root/data/dinov3_model/DINOv3/dinov3_vith16plus",
        in_size = (256, 256)
    ), freeze_backbone=False)

    scene_transforms = scene_branch.get_transforms()
    head_transforms = head_branch.get_transforms()
    model = PaGE(scene_branch, head_branch, n_scene_self_attn_layers=1, n_head_self_attn_layers=1, n_scene_head_interaction_layers=5,
                      n_reg_tokens=4, inout=True, dim=256, dino_feature_dropout=0.5)
    return model, scene_transforms, head_transforms