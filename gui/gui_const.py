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
    "Protogen_V2.2.ckpt": {
        'sha256': 'bb725eaf2ed90092e68b892a1d6262f538131a7ec6a736e50ae534be6b5bd7b1',
        'url': "https://huggingface.co/darkstorm2150/Protogen_v2.2_Official_Release/resolve/main/Protogen_V2.2.ckpt",
        'requires_login': False,
    },
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
        'url':'https://huggingface.co/CompVis/stable-diffusion-v-1-1-original/resolve/main/sd-v1-1-full-ema.ckpt',
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

LOADING_GIF_B64 = b'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'


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