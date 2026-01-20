import torch
import time
import nodes
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview
from .shared import PAUSE_STATE

class Signaler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}
    RETURN_TYPES = ("PAUSE_BUS",)
    RETURN_NAMES = ("signal_bus",)
    FUNCTION = "do_nothing"
    CATEGORY = "Pause/Control"
    def do_nothing(self):
        return ("SIGNAL_BUS_ACTIVE",)

class PSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "signal_bus": ("PAUSE_BUS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Pause/Sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, signal_bus=None):
        
        latent_samples = latent_image["samples"]
        disable_noise = False 

        if disable_noise:
            noise = torch.zeros(latent_samples.size(), dtype=latent_samples.dtype, layout=latent_samples.layout, device="cpu")
        else:
            batch_inds = latent_image.get("batch_index") if "batch_index" in latent_image else None
            noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent_image:
            noise_mask = latent_image["noise_mask"]

        preview_callback = latent_preview.prepare_callback(model, steps, latent_image)

        # --- REFINED CALLBACK WITH BETTER LOGGING ---
        def intercept_callback(step, x0, x, total_steps):
            # 'step' is 0-indexed (0, 1, 2...). 
            # So step=0 is actually the 1st step.
            human_step = step + 1 

            # 1. Update the UI Preview first (Show the result of the step we just finished)
            if preview_callback:
                preview_callback(step, x0, x, total_steps)

            # 2. Check Bridge
            command = PAUSE_STATE.get("command")
            if command == "PAUSE":
                print(f"\n[PSampler] 🛑 PAUSE TRIGGERED.")
                print(f"[PSampler] ✅ Finished Step {human_step}/{total_steps}.")
                print(f"[PSampler] ⏸️ Holding execution... (Next up: Step {human_step + 1})")
                
                # Wait Loop
                while PAUSE_STATE.get("command") == "PAUSE":
                    time.sleep(0.1)
                
                print(f"[PSampler] ▶️ RESUME SIGNAL RECEIVED.")
                print(f"[PSampler] 🚀 Proceeding to calculate Step {human_step + 1}...\n")

        # Execute
        try:
            samples = comfy.sample.sample(
                model, noise, steps, cfg, sampler_name, scheduler, 
                positive, negative, latent_samples,
                denoise=denoise, 
                disable_noise=disable_noise, 
                start_step=None, 
                last_step=None, 
                force_full_denoise=False, 
                noise_mask=noise_mask, 
                callback=intercept_callback, 
                seed=seed
            )
        except comfy.model_management.InterruptProcessingException:
            return (latent_image,)

        out = latent_image.copy()
        out['samples'] = samples
        return (out,)
