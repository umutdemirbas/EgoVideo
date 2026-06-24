from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")

def build_model(cfg):
    name = cfg.MODEL.NAME
    model = MODEL_REGISTRY.get(name)(cfg)
    return model


# from fvcore.common.registry import Registry

# MODEL_REGISTRY = Registry("MODEL")

# # Import models to register them
# from .egovideo_latent_sta import StillFastLatentSTA

# def build_model(cfg):
#     name = cfg.MODEL.NAME
    
#     # Handle EgoVideo+LAS models
#     if name == "stillfast_latent_sta":
#         return StillFastLatentSTA(cfg)
    
#     # Try registry
#     if name in MODEL_REGISTRY:
#         model = MODEL_REGISTRY.get(name)(cfg)
#     else:
#         raise ValueError(f"Model {name} not found in registry or known models")
    
#     return model