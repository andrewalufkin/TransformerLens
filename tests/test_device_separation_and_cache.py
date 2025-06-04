import time

import pytest
import torch

from transformer_lens.HookedTransformer import HookedTransformer


@pytest.fixture
def gpt2_medium_on_1_device():
    model = HookedTransformer.from_pretrained(
        "gpt2-medium", fold_ln=False, n_devices=1, device="cpu"
    )
    return model


def test_device_separation_and_cache(gpt2_medium_on_1_device):
    model_1_device = gpt2_medium_on_1_device
    model_n_devices = HookedTransformer.from_pretrained(
        "gpt2-medium", fold_ln=False, n_devices=1, device="cpu"
    )

    model_description_text = """## Loading Models
    HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. 
    See my explainer for documentation of all supported models, and this table for hyper-parameters and the name used to load them. 
    Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 
    For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"""

    # run model on single device
    start_time_1_device = time.time()
    loss_1_device = model_1_device(model_description_text, return_type="loss")
    elapsed_time_1_device = time.time() - start_time_1_device

    # get model on n_devices
    start_time_n_devices = time.time()
    loss_n_devices = model_n_devices(model_description_text, return_type="loss")
    elapsed_time_n_devices = time.time() - start_time_n_devices

    gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
    gpt2_tokens = model_1_device.to_tokens(gpt2_text)

    gpt2_logits_1_device, gpt2_cache_1_device = model_1_device.run_with_cache(
        gpt2_tokens, remove_batch_dim=True
    )
    gpt2_logits_n_devices, gpt2_cache_n_devices = model_n_devices.run_with_cache(
        gpt2_tokens, remove_batch_dim=True
    )

    # Make sure the tensors in cache remain on their respective devices
    for i in range(model_n_devices.cfg.n_layers):
        cache_device = gpt2_cache_n_devices[f"blocks.{i}.mlp.hook_post"].device
        assert cache_device.type == "cpu"  # Since we're running on CPU

    assert torch.allclose(gpt2_logits_1_device.to("cpu"), gpt2_logits_n_devices.to("cpu"))
    for key in gpt2_cache_1_device.keys():
        assert torch.allclose(
            gpt2_cache_1_device[key].to("cpu"), gpt2_cache_n_devices[key].to("cpu")
        )

    # Log device information to before.txt
    with open("before.txt", "w") as f:
        for name, param in model_n_devices.named_parameters():
            f.write(f"{name}: {param.device}\n")

    print(
        f"Model loss (1 device): {loss_1_device}, Model loss (1 device): {loss_n_devices}, Time taken (1 device): {elapsed_time_1_device:.4f} seconds, Time taken (1 device): {elapsed_time_n_devices:.4f} seconds"
    )
