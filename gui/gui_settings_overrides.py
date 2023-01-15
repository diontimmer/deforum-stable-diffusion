import os
from gui.gui_const import *

def Root(modelpath='', model_config_override='', outputpath='outputs'):
    models_path = "models"
    configs_path = "configs"
    output_path = outputpath
    mount_google_drive = False
    models_path_gdrive = "/content/drive/MyDrive/AI/models"
    output_path_gdrive = "/content/drive/MyDrive/AI/StableDiffusion"
    model_config = model_config_override
    # get filename from modelpath
    model_checkpoint = "custom" if os.path.basename(modelpath) not in model_map else os.path.basename(modelpath)
    custom_config_path = ""
    custom_checkpoint_path = modelpath
    half_precision = True
    return locals()


def DeforumAnimArgs(overrides):

    #@markdown ####**Animation:**
    animation_mode = overrides['animation_mode']
    max_frames = int(overrides['max_frames'])
    border = overrides['border']

    #@markdown ####**Motion Parameters:**
    angle = overrides['angle']
    zoom = overrides['zoom']
    translation_x = overrides['translation_x']
    translation_y = overrides['translation_y']
    translation_z = overrides['translation_z']
    rotation_3d_x = overrides['rotation_3d_x']
    rotation_3d_y = overrides['rotation_3d_y']
    rotation_3d_z = overrides['rotation_3d_z']
    flip_2d_perspective = overrides['flip_2d_perspective']
    perspective_flip_theta = overrides['perspective_flip_theta']
    perspective_flip_phi = overrides['perspective_flip_phi']
    perspective_flip_gamma = overrides['perspective_flip_gamma']
    perspective_flip_fv = overrides['perspective_flip_fv']
    noise_schedule = overrides['noise_schedule']
    strength_schedule = overrides['strength_schedule']
    contrast_schedule = overrides['contrast_schedule']
    hybrid_video_comp_alpha_schedule = overrides['hybrid_video_comp_alpha_schedule']
    hybrid_video_comp_mask_blend_alpha_schedule = overrides['hybrid_video_comp_mask_blend_alpha_schedule']
    hybrid_video_comp_mask_contrast_schedule = overrides['hybrid_video_comp_mask_contrast_schedule']
    hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule = overrides['hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule']
    hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule = overrides['hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule']

    #@markdown ####**Unsharp mask (anti-blur) Parameters:**
    kernel_schedule = overrides['kernel_schedule']
    sigma_schedule = overrides['sigma_schedule']
    amount_schedule = overrides['amount_schedule']
    threshold_schedule = overrides['threshold_schedule']

    #@markdown ####**Coherence:**
    color_coherence = overrides['color_coherence']
    diffusion_cadence = int(overrides['diffusion_cadence'])

    #@markdown ####**3D Depth Warping:**
    use_depth_warping = overrides['use_depth_warping']
    midas_weight = float(overrides['midas_weight'])
    near_plane = int(overrides['near_plane'])
    far_plane = int(overrides['far_plane'])
    fov = float(overrides['fov'])
    padding_mode = overrides['padding_mode']
    sampling_mode = overrides['sampling_mode']
    save_depth_maps = overrides['save_depth_maps'] #b

    #@markdown ####**Video Input:**
    video_init_path = overrides['video_init_path']
    extract_nth_frame = int(overrides['extract_nth_frame'])
    overwrite_extracted_frames = overrides['overwrite_extracted_frames']
    use_mask_video = overrides['use_mask_video']
    video_mask_path = overrides['video_mask_path']

    #@markdown ####**Hybrid Video for 2D/3D Animation Mode:**
    hybrid_video_generate_inputframes = overrides['hybrid_video_generate_inputframes']
    hybrid_video_use_first_frame_as_init_image = overrides['hybrid_video_use_first_frame_as_init_image']
    hybrid_video_motion = overrides['hybrid_video_motion']
    hybrid_video_flow_method = overrides['hybrid_video_flow_method']
    hybrid_video_composite = overrides['hybrid_video_composite']
    hybrid_video_comp_mask_type = overrides['hybrid_video_comp_mask_type']
    hybrid_video_comp_mask_inverse = overrides['hybrid_video_comp_mask_inverse']
    hybrid_video_comp_mask_equalize = overrides['hybrid_video_comp_mask_equalize']
    hybrid_video_comp_mask_auto_contrast = overrides['hybrid_video_comp_mask_auto_contrast']
    hybrid_video_comp_save_extra_frames = overrides['hybrid_video_comp_save_extra_frames']
    hybrid_video_use_video_as_mse_image = overrides['hybrid_video_use_video_as_mse_image']

    #@markdown ####**Interpolation:**
    interpolate_key_frames = overrides['interpolate_key_frames']
    interpolate_x_frames = int(overrides['interpolate_x_frames'])
    
    #@markdown ####**Resume Animation:**
    resume_from_timestring = overrides['resume_from_timestring']
    resume_timestring = overrides['resume_timestring']

    return locals()


def DeforumArgs(overrides):
    W = int(overrides['W'])
    H = int(overrides['H'])
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64
    bit_depth_output = overrides['bit_depth_output']

    # **Sampling Settings**
    seed = int(overrides['seed'])
    sampler = overrides['sampler']  # 'euler_ancestral'  # ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_2m"]
    steps = int(overrides['steps'])
    scale = int(overrides['scale'])
    ddim_eta = float(overrides['ddim_eta'])
    dynamic_threshold = None
    static_threshold = None   
    save_samples = overrides['save_samples']
    save_settings = overrides['save_settings']
    display_samples = True
    save_sample_per_step = overrides['save_sample_per_step']
    show_sample_per_step = overrides['show_sample_per_step']

    # **Prompt Settings**

    prompt_weighting = overrides['prompt_weighting'] 
    normalize_prompt_weights = overrides['normalize_prompt_weights'] 
    log_weighted_subprompts = overrides['log_weighted_subprompts'] 

    # **Batch Settings**

    n_batch = int(overrides['batch_size'])
    batch_name = overrides['batch_name']
    filename_format = "{seed}_{index}_{prompt}.png"
    seed_behavior = overrides['seed_behavior']
    seed_iter_N = int(overrides['seed_iter_N']) #@param {type:'integer'}
    make_grid = overrides['make_grid']
    grid_rows = int(overrides['grid_rows'])
    outdir = overrides['outdir']

    # **Init Settings**

    use_init = overrides['use_init'] 
    strength = float(overrides['strength']) 
    strength_0_no_init = overrides['strength_0_no_init']  # Set the strength to 0 automatically when no init image is used
    init_image = overrides['init_image'] 
    # Whiter areas of the mask are areas that change more
    use_mask = overrides['use_mask'] 
    use_alpha_as_mask = overrides['use_alpha_as_mask']   # use the alpha channel of the init image as the mask
    mask_file = overrides['mask_file'] 
    invert_mask = overrides['invert_mask'] 
    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
    mask_brightness_adjust = float(overrides['mask_brightness_adjust'])  
    mask_contrast_adjust = float(overrides['mask_contrast_adjust'])  
    # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
    overlay_mask = overrides['overlay_mask'] 
    # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
    mask_overlay_blur = float(overrides['mask_overlay_blur'])

    # **Exposure/Contrast Conditional Settings**

    mean_scale = int(overrides['mean_scale']) 
    var_scale = int(overrides['var_scale']) 
    exposure_scale = int(overrides['exposure_scale']) 
    exposure_target = float(overrides['exposure_target'])

    # **Color Match Conditional Settings**

    colormatch_scale = int(overrides['colormatch_scale']) 
    colormatch_image = overrides['colormatch_image']
    colormatch_n_colors = int(overrides['colormatch_n_colors']) 
    ignore_sat_weight = int(overrides['ignore_sat_weight']) 

    # **CLIP\Aesthetics Conditional Settings**

    clip_name = overrides['clip_name']
    clip_scale = int(overrides['clip_scale']) 
    aesthetics_scale = int(overrides['aesthetics_scale']) 
    cutn = int(overrides['cutn']) 
    cut_pow = float(overrides['cut_pow'])

    # **Other Conditional Settings**

    init_mse_scale = int(overrides['init_mse_scale']) 
    init_mse_image = overrides['init_mse_image']
    blue_scale = int(overrides['blue_scale'])

    # **Conditional Gradient Settings**

    gradient_wrt = overrides['gradient_wrt']  # ["x", "x0_pred"]
    gradient_add_to = overrides['gradient_add_to']  # ["cond", "uncond", "both"]
    decode_method = overrides['decode_method']  # ["autoencoder","linear"]
    grad_threshold_type = overrides['grad_threshold_type']  # ["dynamic", "static", "mean", "schedule"]
    clamp_grad_threshold = float(overrides['clamp_grad_threshold']) 
    clamp_start = float(overrides['clamp_start'])
    clamp_stop = float(overrides['clamp_stop'])
    grad_inject_timing = [int(x) for x in overrides['grad_inject_timing'].strip('[]').split(',')]

    # **Speed vs VRAM Settings**

    cond_uncond_sync = overrides['cond_uncond_sync'] 
    n_samples = 1  # doesnt do anything
    precision = 'autocast' 
    C = 4
    f = 8
    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_sample_raw = None
    mask_sample = None
    init_c = None
    seed_internal = 0
    return locals()
