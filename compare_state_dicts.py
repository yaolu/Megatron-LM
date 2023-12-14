import torch
import megatron

#main = torch.load("/lustre/fsw/sa/wdykas/code/debug-outputs/flamingo-2b-ocr-clip336-unfrozen-bf16-localDDP-llamamerge/iter_0002000/mp_rank_00/model_optim_rng.pt")
#original = torch.load("/lustre/fsw/sa/wdykas/code/debug-outputs/flamingo-2b-ocr-clip336-unfrozen-bf16-localDDP-llamamerge/iter_0056281/mp_rank_00/model_optim_rng.pt")
main = torch.load("/lustre/fsw/sa/wdykas/code/debug-outputs/flamingo-2b-ocr-clip336-unfrozen-bf16-localDDP/iter_0002000/mp_rank_00/model_optim_rng.pt")
original = torch.load("/lustre/fsw/sa/wdykas/code/debug-outputs/flamingo-2b-ocr-clip336-unfrozen-bf16-localDDP/iter_0026000/mp_rank_00/model_optim_rng.pt")


def validate_state_dicts_og(model_state_dict_1, model_state_dict_2):
    if len(model_state_dict_1) != len(model_state_dict_2):
        logger.info(
            f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}"
        )
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()
        }

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {
            k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()
        }

    for ((k_1, v_1), (k_2, v_2)) in zip(
        model_state_dict_1.items(), model_state_dict_2.items()
    ):
        print(v_1.shape)
        print(v_2)
        if k_1 != k_2:
            logger.info(f"Key mismatch: {k_1} vs {k_2}")
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            logger.info(f"Tensor mismatch: {v_1} vs {v_2}")
            return False
    return True

def validate_state_dicts(model_state_dict_1, model_state_dict_2):

    #ict_keys(['embedding', 'encoder', 'output_layer'])
    #dict_keys(['_affine'])

    encoder_1 = model_state_dict_1['model']['language_model']['encoder']
    affine_1 = model_state_dict_1['model']['vision_model']['_affine']
    encoder_2 = model_state_dict_2['model']['language_model']['encoder']
    affine_2 = model_state_dict_2['model']['vision_model']['_affine']

    for key in encoder_2.keys():
        enc1_val = encoder_1[key]
        enc2_val = encoder_2[key]
        print(f"{key}:  {torch.allclose(enc1_val, enc2_val)}")
    for key in affine_2.keys():
        aff1_val = affine_1[key]
        aff2_val = affine_2[key]
        print(f"{key}:  {torch.allclose(aff1_val, aff2_val)}")


    
validate_state_dicts(main,original)