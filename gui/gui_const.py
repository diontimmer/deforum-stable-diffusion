from threading import Thread
from sys import settrace



# Const
sampler_list = [
    "klms", 
    "dpm2",
    "dpm2_ancestral",
    "heun",
    "euler",
    "euler_ancestral",
    "plms", 
    "ddim", 
    "dpm_fast", 
    "dpm_adaptive", 
    "dpmpp_2s_a", 
    "dpmpp_2m"
    ]

gradient_wrt_list = ["x", "x0_pred"]

gradient_add_list = ["cond", "uncond", "both"]

decode_method_list = ["autoencoder", "linear"]

grad_threshold_list = ["dynamic", "static", "mean", "schedule"]     

clip_list = ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']

seed_type_list = ["iter", "fixed", "random"]

bit_depth_list = [8, 16, 32]

anim_type_list = ['None', '2D', '3D', 'Video Input', 'Interpolation']

border_type_list = ['wrap', 'replicate']

color_coherence_list = ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB']

diffusion_cadence_list = ['1', '2', '3', '4', '5', '6', '7', '8']

padding_mode_list = ['border', 'reflection', 'zeros']

sampling_mode_list = ['bicubic', 'bilinear', 'nearest']

hybrid_video_motion_list = ['None', 'Optical Flow', 'Perspective', 'Affine']

hybrid_video_flow_method_list = ['Farneback', 'DenseRLOF', 'SF']

hybrid_video_comp_mask_type_list = ['None', 'Depth', 'Video Depth', 'Blend', 'Difference']

hybrid_video_comp_mask_equalize_list = ['None', 'Before', 'After', 'Both']

model_map = {
    "v2-1_768-ema-pruned.ckpt": {
        'sha256': 'ad2a33c361c1f593c4a1fb32ea81afce2b5bb7d1983c6b94793a26a3b54b08a0',
        'url': 'https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.ckpt',
        'requires_login': False,
        },
    "v2-1_512-ema-pruned.ckpt": {
        'sha256': '88ecb782561455673c4b78d05093494b9c539fc6bfc08f3a9a4a0dd7b0b10f36',
        'url': 'https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt',
        'requires_login': False,
        },
    "768-v-ema.ckpt": {
        'sha256': 'bfcaf0755797b0c30eb00a3787e8b423eb1f5decd8de76c4d824ac2dd27e139f',
        'url': 'https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.ckpt',
        'requires_login': False,
        },
    "512-base-ema.ckpt": {
        'sha256': 'd635794c1fedfdfa261e065370bea59c651fc9bfa65dc6d67ad29e11869a1824',
        'url': 'https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/512-base-ema.ckpt',
        'requires_login': False,
        },
    "v1-5-pruned.ckpt": {
        'sha256': 'e1441589a6f3c5a53f5f54d0975a18a7feb7cdf0b0dee276dfc3331ae376a053',
        'url': 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt',
        'requires_login': False,
        },
    "v1-5-pruned-emaonly.ckpt": {
        'sha256': 'cc6cb27103417325ff94f52b7a5d2dde45a7515b25c255d8e396c90014281516',
        'url': 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt',
        'requires_login': False,
        },
    "sd-v1-4-full-ema.ckpt": {
        'sha256': '14749efc0ae8ef0329391ad4436feb781b402f4fece4883c7ad8d10556d8a36a',
        'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-2-original/blob/main/sd-v1-4-full-ema.ckpt',
        'requires_login': True,
        },
    "sd-v1-4.ckpt": {
        'sha256': 'fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556',
        'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt',
        'requires_login': True,
        },
    "sd-v1-3-full-ema.ckpt": {
        'sha256': '54632c6e8a36eecae65e36cb0595fab314e1a1545a65209f24fde221a8d4b2ca',
        'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-3-original/blob/main/sd-v1-3-full-ema.ckpt',
        'requires_login': True,
        },
    "sd-v1-3.ckpt": {
        'sha256': '2cff93af4dcc07c3e03110205988ff98481e86539c51a8098d4f2236e41f7f2f',
        'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-3-original/resolve/main/sd-v1-3.ckpt',
        'requires_login': True,
        },
    "sd-v1-2-full-ema.ckpt": {
        'sha256': 'bc5086a904d7b9d13d2a7bccf38f089824755be7261c7399d92e555e1e9ac69a',
        'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-2-original/blob/main/sd-v1-2-full-ema.ckpt',
        'requires_login': True,
        },
    "sd-v1-2.ckpt": {
        'sha256': '3b87d30facd5bafca1cbed71cfb86648aad75d1c264663c0cc78c7aea8daec0d',
        'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-2-original/resolve/main/sd-v1-2.ckpt',
        'requires_login': True,
        },
    "sd-v1-1-full-ema.ckpt": {
        'sha256': 'efdeb5dc418a025d9a8cc0a8617e106c69044bc2925abecc8a254b2910d69829',
        'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-1-original/resolve/main/sd-v1-1-full-ema.ckpt',
        'requires_login': True,
        },
    "sd-v1-1.ckpt": {
        'sha256': '86cd1d3ccb044d7ba8db743d717c9bac603c4043508ad2571383f954390f3cea',
        'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-1-original/resolve/main/sd-v1-1.ckpt',
        'requires_login': True,
        },
    "robo-diffusion-v1.ckpt": {
        'sha256': '244dbe0dcb55c761bde9c2ac0e9b46cc9705ebfe5f1f3a7cc46251573ea14e16',
        'url': 'https://huggingface.co/nousr/robo-diffusion/resolve/main/models/robo-diffusion-v1.ckpt',
        'requires_login': False,
        },
    "wd-v1-3-float16.ckpt": {
        'sha256': '4afab9126057859b34d13d6207d90221d0b017b7580469ea70cee37757a29edd',
        'url': 'https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-float16.ckpt',
        'requires_login': False,
        },
}


class KThread(Thread):

    """A subclass of threading.Thread, with a kill() method."""
    def __init__(self, *args, **keywords):
        Thread.__init__(self, *args, **keywords)
        self.killed = False

    def start(self):
        """Start the thread."""
        self.__run_backup = self.run
        self.run = self.__run     
        Thread.start(self)

    def __run(self):
        """Hacked run function, which installs the trace."""
        settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

    def globaltrace(self, frame, why, arg):
        if why == 'call':
            return self.localtrace
        else:
            return None

    def localtrace(self, frame, why, arg):
        if self.killed:
            if why == 'line':
                raise SystemExit()
            return self.localtrace
            
    def kill(self):
        self.killed = True