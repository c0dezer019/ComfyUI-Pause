import torch
import comfy.sample
import comfy.samplers
import comfy.k_diffusion.sampling
from .utils.injector import HistoryInjector
from .utils.storage import SnapshotManager
from .utils.cryo import CryoStateManager

class PSamplerAdvanced:
    COMMANDS = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"], ), # CRITICAL for resumes
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"], ),
                "snapshot_interval": ("INT", {"default": 10, "min": 1}),
                "storage_limit_mb": ("INT", {"default": 2048, "min": 100}),
            },
            "optional": {"snapshot_path": ("STRING", {"forceInput": True})}
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"
    CATEGORY = "sampling"

    def execute(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, 
                latent_image, start_at_step, end_at_step, return_with_leftover_noise, 
                snapshot_interval, storage_limit_mb, snapshot_path=None):
        
        # 1. Setup Session Identity
        run_id = f"ps_adv_{noise_seed}"
        storage = SnapshotManager(run_id, "./input/snapshots", storage_limit_mb)
        
        # 2. Logic for "Thawing" (Resurrection)
        current_latent = latent_image.copy()
        actual_start_step = start_at_step
        force_noise = add_noise

        if snapshot_path:
            # We load the full Cryo-State (Latent + RNG + Momentum)
            state = CryoStateManager.load_state(snapshot_path)
            current_latent["samples"] = state["latent"]
            
            # The snapshot tells us exactly where we were. 
            # We ignore the widget's 'start_at_step' and use the file's metadata.
            actual_start_step = int(state.get_metadata().get("step", start_at_step))
            
            # If we are resuming a noisy latent, we MUST disable 'add_noise' 
            # to prevent the sampler from double-layering the initial noise.
            force_noise = "disable" 
            print(f"❄️ PSampler Advanced: Resuming at step {actual_start_step}. Initial noise suppressed.")
        
        self.injection_done = False

        # 3. The Callback (The Math Interceptor)
        def callback(step, x0, x, total_steps):
            abs_step = actual_start_step + step + 1
            node_id = str(id(self))
            cmd = self.COMMANDS.get(node_id, "RUN")

            # Command Polling (Pause/Exit)
            if cmd == "PAUSE":
                while self.COMMANDS.get(node_id) == "PAUSE":
                    import time
                    time.sleep(0.1)
            elif cmd == "EXIT":
                raise InterruptedError("PSampler Advanced Exit")

            # Piecemeal Momentum Persistence
            # By saving 'x - x0', we capture the current trajectory for 2M/3M samplers.
            if abs_step % snapshot_interval == 0 or cmd == "SNAPSHOT":
                derivative = x - x0 
                path = storage.run_dir / f"step_{abs_step}.safetensors"
                storage.enforce_limit(200 * 1024 * 1024) # Slightly larger buffer for Advanced
                
                CryoStateManager.save_state(path, x, abs_step, {"seed": noise_seed}, derivative)
                storage.register(path, abs_step)
                
                if cmd == "SNAPSHOT": self.COMMANDS[node_id] = "RUN"

            if not self.injection_done and hasattr(self, 'loaded_history'):
                 # We need access to the sampler instance. 
                 # In standard Comfy callbacks, 'self' refers to our Node, not the Sampler.
                 # This is the tricky part: Comfy hides the 'k_diffusion_sampler' instance.
                 pass

        # 4. Advanced Execution
        try:
            return (comfy.sample.sample(model, noise_seed, steps, cfg, sampler_name, scheduler, 
                                        positive, negative, current_latent,
                                        start_step=actual_start_step, 
                                        last_step=end_at_step, 
                                        force_full_denoise=(return_with_leftover_noise == "disable"), 
                                        callback=callback, 
                                        disable_noise=(force_noise == "disable"), 
                                        seed=noise_seed),)
        except InterruptedError:
            return (current_latent,)

    def sample_custom(self, model, noise_seed, steps, cfg, sampler_name, scheduler,
                      positive, negative, latent_image, start_step, end_step,
                      force_full_denoise, callback, run_dir):

        # 1. Prepare the Standard Comfy Setup
        # We borrow ComfyUI's standard setup code to get the sigmas and models ready
        device = comfy.model_management.get_torch_device()
        comfy_sampler = comfy.samplers.KSampler(
            model, steps=steps, device=device, sampler=sampler_name,
            scheduler=scheduler, denoise=1.0, model_options={}
        )

        # Calculate Sigmas (The Noise Schedule)
        sigmas = comfy_sampler.sigmas

        # 2. The Injection Phase (The "Secret Sauce")
        # If we are resuming (start_step > 0), we try to load history
        extra_args = {}
        if start_step > 0 and "dpmpp" in sampler_name:
            # We determine depth based on sampler name
            depth = 3 if "3m" in sampler_name else 2

            # Load the history tensors
            history = HistoryInjector.gather_history(run_dir, start_step, depth)

            if history:
                # CRITICAL: This is where we inject.
                # k-diffusion samplers often look for 'old_denoised' in extra_args
                # Note: This depends on specific k-diffusion implementation details.
                # For DPM++ 2M, it usually maintains this internally.
                # Since we can't easily inject into a running function, we might
                # have to use a "Wrapper" callback to seed it on the first step.

                # Strategy: We attach it to the model options or a special container
                # that our modified callback can read to "stuff" the buffer
                # right before the math starts.
                extra_args["injected_history"] = history

        # 3. Execution using comfy.samplers.sample
        # We use the lower-level function that allows more control
        try:
            samples = comfy.samplers.sample(
                model, noise_seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent_image,
                start_step=start_step, last_step=end_step,
                force_full_denoise=force_full_denoise,
                callback=callback,
                disable_noise=(start_step > 0),
                seed=noise_seed
            )
        except InterruptedError:
            return latent_image

        return samples
