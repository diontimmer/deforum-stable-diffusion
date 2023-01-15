import PySimpleGUI as sg
import subprocess, time, gc, sys, os, random, torch  # noqa: E401
from types import SimpleNamespace
from helpers.save_images import get_output_folder
import time
import io
import sys
import trace
import pickle
import clip
import gui.gui_interface as gui
from gui.gui_const import *
from base64 import b64encode

sys.path.extend(['src'])

app_log = False

from helpers.render import render_animation, render_input_video, render_image_batch, render_interpolation
from helpers.model_load import make_linear_decode, load_model, get_model_output_paths
from helpers.aesthetics import load_aesthetics_model

# ****************************************************************************
# *                                  helpers                                 *
# ****************************************************************************

sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

# run your script

# get the captured output and error
output_vals = sys.stdout.getvalue()
error_vals = sys.stderr.getvalue()

# reset redirect
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__


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
    outdir = get_output_folder(root.output_path, batch_name)

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


def load_root_model(modelname, modelconfig, outputpath):
    set_ready(False)
    global root
    try:
        del root
        print('Removed root model.')
    except NameError:
        pass
    global loaded_model_path
    loaded_model_path = f'models/{modelname}'
    # if not os.path.isfile(loaded_model_path):
    #     print(f'Could not find model {modelname} at {loaded_model_path}, please try again.')
    #     return False
    print(f'Loading model {modelname} at {loaded_model_path}...')
    root = Root(modelpath=loaded_model_path, model_config_override=modelconfig, outputpath=outputpath)
    root = SimpleNamespace(**root)
    root.models_path, root.output_path = get_model_output_paths(root)
    root.model, root.device = load_model(root, load_on_run_all=True, check_sha256=True)
    set_ready(True)
    return


def loadargs(args):
    args = SimpleNamespace(**args)
    args.timestring = time.strftime('%Y%m%d%H%M%S')
    if (args.clip_scale > 0) or (args.aesthetics_scale > 0):
        root.clip_model = clip.load(args.clip_name, jit=False)[0].eval().requires_grad_(False).to(root.device)
    if (args.aesthetics_scale > 0):
        root.aesthetics_model = load_aesthetics_model(args, root)
    args.strength = max(0.0, min(1.0, args.strength))
    if args.seed == -1:
        args.seed = random.randint(0, 2**32 - 1)
    if not args.use_init:
        args.init_image = None
    if args.sampler != 'ddim':
        args.ddim_eta = 0
    return args


def loadanimargs(anim_args, args):
    anim_args = SimpleNamespace(**anim_args)
    if anim_args.animation_mode == 'None':
        anim_args.max_frames = 1
    elif anim_args.animation_mode == 'Video Input':
        args.use_init = True
    return anim_args, args


def do_render(args):
    prompts = values['-PROMPTS-'].split('\n')
    prompts = [x.split(': ', 1)[-1] for x in prompts]
    suffix = values['-SUFFIX-']
    args = loadargs(args)
    set_ready(False)
    gc.collect()
    torch.cuda.empty_cache()
    if suffix != '':
        prompts = [prompt + ', ' + suffix for prompt in prompts]
    # DISPLAY IMAGE IN deforum/helpers.render.py render_image_batch
    render_image_batch(args, prompts, root)
    set_ready(True)
    return


def do_video_render(args, anim_args):
    prompts = values['-PROMPTS-'].split('\n')
    for prompt in prompts:
        if not prompt[0].isdigit():
            print('Please note the keyframes in your animation prompts.')
            set_ready(True)
            return
    suffix = values['-SUFFIX-']
    prompt_dict = {}
    for prompt in prompts:
        key, value = prompt.split(': ')
        prompt_dict[int(key)] = value + ', ' + suffix
    args = loadargs(args)
    anim_args_result = loadanimargs(anim_args, args)
    anim_args = anim_args_result[0]
    args = anim_args_result[1]
    set_ready(False)
    gc.collect()
    torch.cuda.empty_cache()  
    match anim_args.animation_mode:
        case '2D' | '3D':
            render_animation(args, anim_args, prompt_dict, root)
        case 'Video Input':
            render_input_video(args, anim_args, prompt_dict, root)
        case 'Interpolation':
            render_interpolation(args, anim_args, prompt_dict, root)
    create_video(args, anim_args)
    if values['-REMOVE_FRAMES_AFTER-']:
        for file in os.listdir(args.outdir):
            if file.endswith(".png"):
                os.remove(os.path.join(args.outdir, file))        
    set_ready(True)
    return


def getmodels():
    modelvals = []
    for file in os.listdir("models"):
        modelvals.append(file)
    if 'v1-5-pruned-emaonly.ckpt' not in modelvals:
        modelvals.append('v1-5-pruned-emaonly.ckpt')
    return modelvals


def getconfigs():
    configvals = []
    for file in os.listdir("configs"):
        configvals.append(file)
    return configvals


def getargs(values, tab):
    match tab:
        case "general":
            args = {
                'W': values['-WIDTH-'], 
                'H': values['-HEIGHT-'],
                'bit_depth_output': values['-BIT_DEPTH_OUTPUT-'],

                # sampling
                'seed': values['-SEED-'],
                'sampler': values['-SAMPLER-'], 
                'steps': values['-SAMPLER_STEPS-'], 
                'scale': values['-SAMPLER_SCALE-'], 
                'ddim_eta': values['-DDIM_ETA-'],
                'save_samples': values['-SAVE_SAMPLES-'], 
                'save_settings': values['-SAVE_SETTINGS-'],
                'save_sample_per_step': values['-SAVE_SAMPLE_PER_STEP-'], 
                'show_sample_per_step': values['-SHOW_SAMPLE_PER_STEP-'],  

                # prompt
                'prompt_weighting': (values['-PROMPT_WEIGHTING-']), 
                'normalize_prompt_weights': values['-NORMALIZE_PROMPT_WEIGHTS-'],
                'log_weighted_subprompts': values['-LOG_WEIGHTED_SUBPROMPTS-'],      

                # batch
                'batch_size': (values['-BATCH_SIZE-']), 
                'batch_name': values['-BATCH_NAME-'],
                'seed_behavior': values['-SEED_BEHAVIOR-'],
                'seed_iter_N': values['-SEED_ITER_N-'],
                'make_grid': values['-MAKE_GRID-'],
                'grid_rows': values['-GRID_ROWS-'], 

                # init
                'use_init': values['-USE_INIT-'], 
                'strength': values['-STRENGTH-'],  
                'strength_0_no_init': values['-STRENGTH_0_NO_INIT-'], 
                'init_image': values['-INIT_IMAGE-'], 
                'use_mask': values['-USE_MASK-'],  
                'use_alpha_as_mask': values['-USE_ALPHA_AS_MASK-'], 
                'mask_file': values['-MASK_FILE-'], 
                'invert_mask': values['-INVERT_MASK-'], 
                'mask_brightness_adjust': values['-MASK_BRIGHTNESS_ADJUST-'], 
                'mask_contrast_adjust': values['-MASK_CONTRAST_ADJUST-'], 
                'overlay_mask': values['-OVERLAY_MASK-'], 
                'mask_overlay_blur': values['-MASK_OVERLAY_BLUR-'],

                # exposure
                'mean_scale': values['-MEAN_SCALE-'], 
                'var_scale': values['-VAR_SCALE-'],  
                'exposure_scale': values['-EXPOSURE_SCALE-'], 
                'exposure_target': values['-EXPOSURE_TARGET-'],

                # color match
                'colormatch_scale': values['-COLORMATCH_SCALE-'], 
                'colormatch_image': values['-COLORMATCH_IMAGE-'],  
                'colormatch_n_colors': values['-COLORMATCH_N_COLORS-'], 
                'ignore_sat_weight': values['-IGNORE_SAT_WEIGHT-'],

                # clip aesthetic
                'clip_name': values['-CLIP_NAME-'], 
                'clip_scale': values['-CLIP_SCALE-'],  
                'aesthetics_scale': values['-AESTHETICS_SCALE-'], 
                'cutn': values['-CUTN-'], 
                'cut_pow': values['-CUT_POW-'],

                # other conditional
                'init_mse_scale': values['-INIT_MSE_SCALE-'], 
                'init_mse_image': values['-INIT_MSE_IMAGE-'],  
                'blue_scale': values['-BLUE_SCALE-'], 
                # conditional gradient
                'gradient_wrt': values['-GRADIENT_WRT-'],  
                'gradient_add_to': values['-GRADIENT_ADD_TO-'], 
                'decode_method': values['-DECODE_METHOD-'], 
                'grad_threshold_type': values['-GRAD_THRESHOLD_TYPE-'], 
                'clamp_grad_threshold': values['-CLAMP_GRAD_THRESHOLD-'], 
                'clamp_start': values['-CLAMP_START-'], 
                'clamp_stop': values['-CLAMP_STOP-'], 
                'grad_inject_timing': values['-GRAD_INJECT_TIMING-'],

                # speed vs vram
                'cond_uncond_sync': values['-COND_UNCOND_SYNC-'],  
                }
        case "animation":
            args = {
                # animation
                'animation_mode': values['-ANIMATION_MODE-'], 
                'max_frames': values['-MAX_FRAMES-'], 
                'border': values['-BORDER-'],

                # motion
                'angle': values['-ANGLE-'],
                'zoom': values['-ZOOM-'],
                'translation_x': values['-TRANSLATION_X-'],
                'translation_y': values['-TRANSLATION_Y-'],
                'translation_z': values['-TRANSLATION_Z-'],
                'rotation_3d_x': values['-ROTATION_3D_X-'],
                'rotation_3d_y': values['-ROTATION_3D_Y-'],
                'rotation_3d_z': values['-ROTATION_3D_Z-'],
                'flip_2d_perspective': values['-FLIP_2D_PERSPECTIVE-'],
                'perspective_flip_theta': values['-PERSPECTIVE_FLIP_THETA-'],
                'perspective_flip_phi': values['-PERSPECTIVE_FLIP_PHI-'],
                'perspective_flip_gamma': values['-PERSPECTIVE_FLIP_GAMMA-'],
                'perspective_flip_fv': values['-PERSPECTIVE_FLIP_FV-'],
                'noise_schedule': values['-NOISE_SCHEDULE-'],
                'strength_schedule': values['-STRENGTH_SCHEDULE-'],
                'contrast_schedule': values['-CONTRAST_SCHEDULE-'],
                'hybrid_video_comp_alpha_schedule': values['-HYBRID_VIDEO_COMP_ALPHA_SCHEDULE-'],
                'hybrid_video_comp_mask_blend_alpha_schedule': values['-HYBRID_VIDEO_COMP_MASK_BLEND_ALPHA_SCHEDULE-'],
                'hybrid_video_comp_mask_contrast_schedule': values['-HYBRID_VIDEO_COMP_MASK_CONTRAST_SCHEDULE-'],
                'hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule': values['-HYBRID_VIDEO_COMP_MASK_AUTO_CONTRAST_CUTOFF_HIGH_SCHEDULE-'],
                'hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule': values['-HYBRID_VIDEO_COMP_MASK_AUTO_CONTRAST_CUTOFF_LOW_SCHEDULE-'],
                
                # anti blur
                'kernel_schedule': values['-KERNEL_SCHEDULE-'],
                'sigma_schedule': values['-SIGMA_SCHEDULE-'],
                'amount_schedule': values['-AMOUNT_SCHEDULE-'],
                'threshold_schedule': values['-THRESHOLD_SCHEDULE-'],
                
                # coherence 
                'color_coherence': values['-COLOR_COHERENCE-'], 
                'diffusion_cadence': values['-DIFFUSION_CADENCE-'],
                
                # depth warping 
                'use_depth_warping': values['-USE_DEPTH_WARPING-'], 
                'midas_weight': values['-MIDAS_WEIGHT-'],
                'near_plane': values['-NEAR_PLANE-'], 
                'far_plane': values['-FAR_PLANE-'],  
                'fov': values['-FOV-'], 
                'padding_mode': values['-PADDING_MODE-'], 
                'sampling_mode': values['-SAMPLING_MODE-'],
                'save_depth_maps': values['-SAVE_DEPTH_MAPS-'],
             
                # video input 
                'video_init_path': values['-VIDEO_INIT_PATH-'], 
                'extract_nth_frame': values['-EXTRACT_NTH_FRAME-'], 
                'overwrite_extracted_frames': values['-OVERWRITE_EXTRACTED_FRAMES-'], 
                'use_mask_video': values['-USE_MASK_VIDEO-'], 
                'video_mask_path': values['-VIDEO_MASK_PATH-'],

                # hybrid
                'hybrid_video_generate_inputframes': values['-HYBRID_VIDEO_GENERATE_INPUTFRAMES-'],
                'hybrid_video_use_first_frame_as_init_image': values['-HYBRID_VIDEO_USE_FIRST_FRAME_AS_INIT_IMAGE-'],
                'hybrid_video_motion': values['-HYBRID_VIDEO_MOTION-'],
                'hybrid_video_flow_method': values['-HYBRID_VIDEO_FLOW_METHOD-'],
                'hybrid_video_composite': values['-HYBRID_VIDEO_COMPOSITE-'],
                'hybrid_video_comp_mask_type': values['-HYBRID_VIDEO_COMP_MASK_TYPE-'],
                'hybrid_video_comp_mask_inverse': values['-HYBRID_VIDEO_COMP_MASK_INVERSE-'],
                'hybrid_video_comp_mask_equalize': values['-HYBRID_VIDEO_COMP_MASK_EQUALIZE-'],
                'hybrid_video_comp_mask_auto_contrast': values['-HYBRID_VIDEO_COMP_MASK_AUTO_CONTRAST-'],
                'hybrid_video_comp_save_extra_frames': values['-HYBRID_VIDEO_COMP_SAVE_EXTRA_FRAMES-'],
                'hybrid_video_use_video_as_mse_image': values['-HYBRID_VIDEO_USE_VIDEO_AS_MSE_IMAGE-'],   

                # interpolation 
                'interpolate_key_frames': values['-INTERPOLATE_KEY_FRAMES-'], 
                'interpolate_x_frames': values['-INTERPOLATE_X_FRAMES-'],

                # resume 
                'resume_from_timestring': values['-RESUME_FROM_TIMESTRING-'], 
                'resume_timestring': values['-RESUME_TIMESTRING-'],          
            }
    return args


def set_ready(ready):
    global disabled
    disabled = not ready
    window['-RENDER-'].update(disabled=disabled)
    window['-MODEL-'].update(disabled=disabled)
    window['-MODEL_CONFIG-'].update(disabled=disabled)
    if disabled:
        window['-MENUBAR-'].update(menu_definition=[['!File', ['Open::-OPEN-', 'Save::-SAVE-']]])
    else:
        print('READY!')
        window['-MENUBAR-'].update(menu_definition=[['File', ['Open::-OPEN-', 'Save::-SAVE-']]])


# ****************************************************************************
# *                               setup window                               *
# ****************************************************************************

sg.theme('DarkGrey7')

# GENERAL Options Layout
opt_gen_1 = sg.Frame(title='Main Options', layout=[
    [sg.Text('Model: '), sg.Combo(getmodels(), key='-MODEL-', default_value='v1-5-pruned-emaonly.ckpt', enable_events=True)],
    [sg.Text('Model Config: '), sg.Combo(getconfigs(), key='-MODEL_CONFIG-', default_value='v1-inference.yaml', enable_events=True)],
    [sg.Text('Batch Name: '), sg.Input('Testing', key='-BATCH_NAME-', size=(20, 1))],    
    [sg.Text('Batch Size: '), sg.Input('1', key='-BATCH_SIZE-', size=(5, 1))],
    [sg.Text('Seed: '), sg.Input('-1', key='-SEED-', size=(10, 1)), sg.Button('Random', key='-RANDOM_SEED-')],
    [sg.Text('Seed Iteration Number: '), sg.Input('1', key='-SEED_ITER_N-', size=(5, 1))],
    [sg.Text('Seed Behavior: '), sg.Combo(seed_type_list, key='-SEED_BEHAVIOR-', default_value='iter')],
    [sg.Text('Sampler: '), sg.Combo(sampler_list, key='-SAMPLER-', default_value='euler_ancestral')],
    [sg.Text('Sampler Steps: '), sg.Input('80', key='-SAMPLER_STEPS-', size=(5, 1))],
    [sg.Text('Sampler Scale: '), sg.Input('7', key='-SAMPLER_SCALE-', size=(5, 1))],
    [sg.Text('DDIM Eta: '), sg.Input('0.0', key='-DDIM_ETA-', size=(5, 1))],
    [sg.Text('Width: '), sg.Input('512', key='-WIDTH-', size=(5, 1))],
    [sg.Text('Height: '), sg.Input('512', key='-HEIGHT-', size=(5, 1))],
    [sg.Text('Bit Depth Output: '), sg.Combo(bit_depth_list, key='-BIT_DEPTH_OUTPUT-', default_value='8')],
    [sg.Checkbox('Make Grid', key='-MAKE_GRID-', )],
    [sg.Text('Grid Rows: '), sg.Input('2', key='-GRID_ROWS-', size=(5, 1))],           
    ], vertical_alignment='top')

opt_gen_init = sg.Frame(title='Init Options', layout=[
    [sg.Checkbox('Use Init', key='-USE_INIT-', )],
    [sg.Text('Strength: '), sg.Input('0.0', key='-STRENGTH-', size=(5, 1))],
    [sg.Checkbox('Strength 0 if no init: ', key='-STRENGTH_0_NO_INIT-', default=True), ],
    [sg.Text('Init Image: '), sg.Input(key='-INIT_IMAGE-', size=(50, 1)), sg.FileBrowse(file_types=(("Image Files", "*.png *.jpg *.jpeg"),))],       
    ], vertical_alignment='top')

opt_gen_mask = sg.Frame(title='Mask Options', layout=[
    [sg.Checkbox('Use Mask', key='-USE_MASK-'), ],
    [sg.Checkbox('Use Alpha As Mask', key='-USE_ALPHA_AS_MASK-'), ],
    [sg.Text('Mask File: '), sg.Input(key='-MASK_FILE-', size=(50, 1)), sg.FileBrowse(file_types=(("Image Files", "*.png *.jpg *.jpeg"),))],
    [sg.Checkbox('Invert Mask', key='-INVERT_MASK-'), ],
    [sg.Text('Mask Brightness Adjust: '), sg.Input('1.0', key='-MASK_BRIGHTNESS_ADJUST-', size=(5, 1))],
    [sg.Text('Mask Contrast Adjust: '), sg.Input('1.0', key='-MASK_CONTRAST_ADJUST-', size=(5, 1))],
    [sg.Checkbox('Overlay Mask', key='-OVERLAY_MASK-', default=True), ],
    [sg.Text('Mask Overlay Blur: '), sg.Input('5', key='-MASK_OVERLAY_BLUR-', size=(5, 1))],       
    ], vertical_alignment='top')

opt_gen_display = sg.Frame(title='Save/Display Options', layout=[
    [sg.Checkbox('Save Samples', key='-SAVE_SAMPLES-', default=True), ],
    [sg.Checkbox('Save Settings', key='-SAVE_SETTINGS-', default=True), ],   
    [sg.Checkbox('Save Sample Per Step', key='-SAVE_SAMPLE_PER_STEP-'), ],   
    [sg.Checkbox('Show Sample Per Step', key='-SHOW_SAMPLE_PER_STEP-'), ],     
    ], vertical_alignment='top')

opt_gen_prompt = sg.Frame(title='Prompt Weight Settings', layout=[
    [sg.Checkbox('Prompt Weighting', key='-PROMPT_WEIGHTING-', default=True), ],
    [sg.Checkbox('Normalize Prompt Weights', key='-NORMALIZE_PROMPT_WEIGHTS-', default=True), ],   
    [sg.Checkbox('Log Weighted Subprompts', key='-LOG_WEIGHTED_SUBPROMPTS-'), ],       
    ], vertical_alignment='top')

opt_gen_exposure = sg.Frame(title='Exposure/Contrast Settings', layout=[
    [sg.Text('Mean Scale: '), sg.Input('0', key='-MEAN_SCALE-', size=(5, 1))],
    [sg.Text('Var Scale: '), sg.Input('0', key='-VAR_SCALE-', size=(5, 1))],  
    [sg.Text('Exposure Scale: '), sg.Input('0', key='-EXPOSURE_SCALE-', size=(5, 1))],
    [sg.Text('Exposure Target: '), sg.Input('0.5', key='-EXPOSURE_TARGET-', size=(5, 1))],     
    ], vertical_alignment='top')

opt_gen_colormatch = sg.Frame(title='Colormatch Settings', layout=[
    [sg.Text('Colormatch Scale: '), sg.Input('0', key='-COLORMATCH_SCALE-', size=(5, 1))],
    [sg.Text('Colormatch Image: '), sg.Input('https://www.saasdesign.io/wp-content/uploads/2021/02/palette-3-min-980x588.png', key='-COLORMATCH_IMAGE-', size=(50, 1)), sg.FileBrowse(file_types=(("Image Files", "*.png *.jpg *.jpeg"),))], 
    [sg.Text('Colormatch N Colors: '), sg.Input('4', key='-COLORMATCH_N_COLORS-', size=(5, 1))],
    [sg.Text('Ignore Sat Weight: '), sg.Input('0', key='-IGNORE_SAT_WEIGHT-', size=(5, 1))],     
    ], vertical_alignment='top')

opt_gen_clip = sg.Frame(title='Clip/Aesthetics Settings', layout=[
    [sg.Text('Clip Name: '), sg.Combo(clip_list, key='-CLIP_NAME-', default_value='ViT-L/14')],
    [sg.Text('Clip Scale: '), sg.Input('0', key='-CLIP_SCALE-', size=(5, 1))],  
    [sg.Text('Aesthetics Scale: '), sg.Input('0', key='-AESTHETICS_SCALE-', size=(5, 1))],
    [sg.Text('Cut N: '), sg.Input('1', key='-CUTN-', size=(5, 1))],
    [sg.Text('Cut Power: '), sg.Input('0.0001', key='-CUT_POW-', size=(10, 1))],          
    ], vertical_alignment='top')

opt_gen_other = sg.Frame(title='Other Conditional Settings', layout=[
    [sg.Text('Init MSE Scale: '), sg.Input('0', key='-INIT_MSE_SCALE-', size=(5, 1))],   
    [sg.Text('Init MSE Image: '), sg.Input('https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg', key='-INIT_MSE_IMAGE-', size=(50, 1)), sg.FileBrowse(file_types=(("Image Files", "*.png *.jpg *.jpeg"),))], 
    [sg.Text('Blue Scale: '), sg.Input('0', key='-BLUE_SCALE-', size=(5, 1))],      
    ], vertical_alignment='top')

opt_gen_gradient = sg.Frame(title='Conditional Gradient Settings', layout=[
    [sg.Text('Gradient WRT: '), sg.Combo(gradient_wrt_list, key='-GRADIENT_WRT-', default_value='x0_pred')],
    [sg.Text('Gradient Add To: '), sg.Combo(gradient_add_list, key='-GRADIENT_ADD_TO-', default_value='both')],
    [sg.Text('Decode Method: '), sg.Combo(decode_method_list, key='-DECODE_METHOD-', default_value='linear')],
    [sg.Text('Threshold Type: '), sg.Combo(grad_threshold_list, key='-GRAD_THRESHOLD_TYPE-', default_value='dynamic')],
    [sg.Text('Clamp Gradient Threshold: '), sg.Input('0.2', key='-CLAMP_GRAD_THRESHOLD-', size=(5, 1))],
    [sg.Text('Clamp Start: '), sg.Input('0.2', key='-CLAMP_START-', size=(5, 1))],
    [sg.Text('Clamp Stop: '), sg.Input('0.01', key='-CLAMP_STOP-', size=(5, 1))],
    [sg.Text('Gradient Inject Timing: '), sg.Input('[1, 2, 3, 4, 5, 6, 7, 8, 9]', key='-GRAD_INJECT_TIMING-', size=(20, 1))],    
    ], vertical_alignment='top')

opt_gen_speed = sg.Frame(title='Conditional Gradient Settings', layout=[
    [sg.Checkbox('Cond Uncond Sync', key='-COND_UNCOND_SYNC-', default=True), ], 
    ], vertical_alignment='top')

opt_gen_tab = sg.Tab('General', [[opt_gen_1, opt_gen_display, opt_gen_prompt]], key='-GENERAL_TAB-')

opt_init_mask_tab = sg.Tab('Init/Mask', [[opt_gen_init, opt_gen_mask]], key='-INIT_MASK_TAB-')

opt_exposure_colormatch_tab = sg.Tab('Exposure/ColorMatch', [[opt_gen_exposure, opt_gen_colormatch]], key='-EXPOSURE_COLORMATCH_TAB-')

opt_clip_gradient_tab = sg.Tab('Clip/Gradient', [[opt_gen_clip], [opt_gen_gradient]], key='-CLIP_GRADIENT_TAB-')

opt_speed_other_tab = sg.Tab('Speed/Other', [[opt_gen_speed], [opt_gen_other]], key='-SPEED_OTHER_TAB-')

# Anim Options Layout
opt_anim_1 = sg.Frame(title='General Animation', layout=[
    [sg.Text('Animation Mode: '), sg.Combo(anim_type_list, key='-ANIMATION_MODE-', default_value='None')],
    [sg.Text('Max Frames: '), sg.Input('1000', key='-MAX_FRAMES-', size=(5, 1))],
    [sg.Text('Border: '), sg.Combo(border_type_list, key='-BORDER-', default_value='replicate')],
    [sg.Text('Color Coherence: '), sg.Combo(color_coherence_list, key='-COLOR_COHERENCE-', default_value='Match Frame 0 LAB')],
    [sg.Text('Diffusion Cadence: '), sg.Input('1', key='-DIFFUSION_CADENCE-', size=(5, 1))],
    [sg.Text('FPS: '), sg.Input('12', key='-FPS-', size=(5, 1))],
    [sg.Checkbox('Make GIF', key='-MAKE_GIF-', )],
    [sg.Checkbox('Resume From Timestring', key='-RESUME_FROM_TIMESTRING-')],
    [sg.Text('Resume Timestring: '), sg.Input('20220829210106', key='-RESUME_TIMESTRING-', size=(20, 1))],
    [sg.Checkbox('Remove Frames After Export', key='-REMOVE_FRAMES_AFTER-', )],
    ], vertical_alignment='top')

opt_anim_2 = sg.Frame(title='Depth Warping', layout=[
    [sg.Checkbox('Use Depth Warping', key='-USE_DEPTH_WARPING-', default=True)],
    [sg.Text('Midas Weight: '), sg.Input('0.3', key='-MIDAS_WEIGHT-', size=(5, 1))],
    [sg.Text('Near Plane: '), sg.Input('200', key='-NEAR_PLANE-', size=(10, 1))],
    [sg.Text('Far Plane: '), sg.Input('10000', key='-FAR_PLANE-', size=(10, 1))],
    [sg.Text('FOV: '), sg.Input('40', key='-FOV-', size=(5, 1))],
    [sg.Text('Padding Mode: '), sg.Combo(padding_mode_list, key='-PADDING_MODE-', default_value='border')],
    [sg.Text('Sampling Mode: '), sg.Combo(sampling_mode_list, key='-SAMPLING_MODE-', default_value='bicubic')],
    [sg.Checkbox('Save Depth Maps', key='-SAVE_DEPTH_MAPS-', )],
    ], vertical_alignment='top')

opt_anim_3 = sg.Frame(title='Video Input Settings', layout=[
    [sg.Text('Video Init Path: '), sg.Input(key='-VIDEO_INIT_PATH-', size=(50, 1)), sg.FileBrowse(file_types=(("Video Files", "*.mp4"),))],
    [sg.Text('Extract Nth Frame: '), sg.Input('1', key='-EXTRACT_NTH_FRAME-', size=(5, 1))],
    [sg.Checkbox('Overwrite Extracted Frames', key='-OVERWRITE_EXTRACTED_FRAMES-')],
    [sg.Checkbox('Use Mask Video', key='-USE_MASK_VIDEO-')],
    [sg.Text('Video Mask Path: '), sg.Input(key='-VIDEO_MASK_PATH-', size=(50, 1)), sg.FileBrowse(file_types=(("Video Files", "*.mp4"),))],
    ], vertical_alignment='top')

opt_anim_4 = sg.Frame(title='Interpolation Settings', layout=[
    [sg.Checkbox('Interpolate Key Frames', key='-INTERPOLATE_KEY_FRAMES-')],
    [sg.Text('Interpolate X Frames: '), sg.Input('1', key='-INTERPOLATE_X_FRAMES-', size=(5, 1))],
    ], vertical_alignment='top')

opt_anim_5 = sg.Frame(title='Hybrid Video', layout=[
    [sg.Checkbox('HV Generate Input Frames', key='-HYBRID_VIDEO_GENERATE_INPUTFRAMES-')],
    [sg.Checkbox('HV Use First Frame As Init', key='-HYBRID_VIDEO_USE_FIRST_FRAME_AS_INIT_IMAGE-', default=True)],
    [sg.Text('HV Motion: '), sg.Combo(hybrid_video_motion_list, key='-HYBRID_VIDEO_MOTION-', default_value='None')],
    [sg.Text('HV Flow Method: '), sg.Combo(hybrid_video_flow_method_list, key='-HYBRID_VIDEO_FLOW_METHOD-', default_value='Farneback')],
    [sg.Checkbox('HV Composite', key='-HYBRID_VIDEO_COMPOSITE-')],
    [sg.Text('HV Comp Mask Type: '), sg.Combo(hybrid_video_comp_mask_type_list, key='-HYBRID_VIDEO_COMP_MASK_TYPE-', default_value='None')],
    [sg.Checkbox('HV Comp Mask Inverse', key='-HYBRID_VIDEO_COMP_MASK_INVERSE-')],
    [sg.Text('HV Comp Mask Equalize: '), sg.Combo(hybrid_video_motion_list, key='-HYBRID_VIDEO_COMP_MASK_EQUALIZE-', default_value='None')],
    [sg.Checkbox('HV Comp Mask Auto Contrast', key='-HYBRID_VIDEO_COMP_MASK_AUTO_CONTRAST-')],
    [sg.Checkbox('HV Comp Mask Extra Frames', key='-HYBRID_VIDEO_COMP_SAVE_EXTRA_FRAMES-')],
    [sg.Checkbox('HV Use Video As MSE Image', key='-HYBRID_VIDEO_USE_VIDEO_AS_MSE_IMAGE-')],

    ], vertical_alignment='top')

opt_anim_tab = sg.Tab('Animation', [[opt_anim_1, opt_anim_2, opt_anim_4, opt_anim_3, opt_anim_5]], key='-ANIMATION_TAB-', )

# Motion Options Layout


def motion_opt(displaytext, defaulttext):
    motionopt = [sg.Text(f'{displaytext}: '), sg.Input(defaulttext, key=f'-{displaytext.upper().replace(" ", "_")}-', size=(200, 1))]
    return motionopt


opt_motion = sg.Column([
    [sg.Frame('Motion', [
        motion_opt('Angle', "0:(0)"),
        motion_opt('Zoom', "0:(1.04)"),
        motion_opt('Translation X', "0:(10*sin(2*3.14*t/10))"),
        motion_opt('Translation Y', "0:(0)"),
        motion_opt('Translation Z', "0:(10)"),
        motion_opt('Rotation 3D X', "0:(0)"),
        motion_opt('Rotation 3D Y', "0:(0)"),
        motion_opt('Rotation 3D Z', "0:(0)"),
        [sg.Checkbox('Flip 2D Perspective', key='-FLIP_2D_PERSPECTIVE-')],
        motion_opt('Perspective Flip Theta', "0:(0)"),
        motion_opt('Perspective Flip Phi', "0:(t%15)"),
        motion_opt('Perspective Flip Gamma', "0:(10)"),
        motion_opt('Perspective Flip FV', "0:(53)"),
        motion_opt('Noise Schedule', "0:(0.02)"),
        motion_opt('Strength Schedule', "0:(0.65)"),
        motion_opt('Contrast Schedule', "0:(1.0)"),
        # hybrid
        motion_opt('Hybrid Video Comp Alpha Schedule', "0:(1)"),
        motion_opt('Hybrid Video Comp Mask Blend Alpha Schedule', "0:(0.5)"),
        motion_opt('Hybrid Video Comp Mask Contrast Schedule', "0:(1)"),
        motion_opt('Hybrid Video Comp Mask Auto Contrast Cutoff High Schedule', "0:(100)"),
        motion_opt('Hybrid Video Comp Mask Auto Contrast Cutoff Low Schedule', "0:(0)"),
        ], expand_x=True)],
    [sg.Frame('Unsharp mask (anti-blur)', [
        motion_opt('Kernel Schedule', "0:(5)"),
        motion_opt('Sigma Schedule', "0:(1.0)"),
        motion_opt('Amount Schedule', "0:(0.2)"),
        motion_opt('Threshold Schedule', "0:(0.0)"),
        ], expand_x=True)],

], scrollable=True, expand_x=True, expand_y=True, vertical_scroll_only=True)

opt_motion_tab = sg.Tab('Anim Motion', [[opt_motion]], key='-MOTION_TAB-')

tab_layout = sg.TabGroup([
    [opt_gen_tab], 
    [opt_init_mask_tab], 
    [opt_exposure_colormatch_tab], 
    [opt_clip_gradient_tab], 
    [opt_speed_other_tab], 
    [opt_anim_tab], 
    [opt_motion_tab]], 
    key='-TABGROUP-', expand_x=True)

menu_def = [['File', ['Open::-OPEN-', 'Save::-SAVE-']]]

log_ml = sg.Multiline(disabled=True, expand_x=True, expand_y=True, reroute_stdout=app_log, reroute_stderr=app_log, reroute_cprint=app_log, autoscroll=True, auto_refresh=True, key='-LOG-')

prompt_box = sg.Column([
    [sg.Text('Prompts: (Separated by new line) '), sg.Text('Suffix: '), sg.Input('', key='-SUFFIX-', expand_x=True)],
    [sg.Multiline(expand_x=True, expand_y=False, key='-PROMPTS-', size=(0,20))],
    [sg.Text('Output Path: '), sg.Input(f'{os.path.dirname(os.path.abspath(__file__))}\\output', key='-OUTPUT_PATH-', size=(80, 1)), sg.FileBrowse()],
    [sg.Button('Render', key='-RENDER-'), sg.Button('Reload Model', key='-RELOAD-'), sg.Button('Cancel', key='-CANCEL-')],
    [log_ml],
    ], vertical_alignment='top', expand_x=True, expand_y=True)



current_image = sg.Column([[
    sg.Image(key='-IMAGE-', size=(768, 768), background_color="#2e3238"), 
    ]], expand_x=False)

layout = [
    [sg.Menubar(menu_def, key='-MENUBAR-')],
    [current_image, prompt_box], 
    [tab_layout],
    ]
  
window = sg.Window('d̷̨̗͎̲̟̤̀͆̿͒͆̈́̕e̵̦̓̍̉́̆͂f̵̨͖͙͉͇͊͑͠o̶̹̤͉̼̹͍͇͋̈́r̴̖̾̂͌̆ū̶̳̟͈͕͌̎͑̒͐̏͜m̶̻̭͎͇͔͎̜͐͒̈̓̽', layout, resizable=True, finalize=True, size=(720, 920), font=("Calibri", 11), enable_close_attempted_event=True, icon='gui/favicon.ico')
gui.guiwindow = window

# ****************************************************************************
# *                                event loop                                *
# ****************************************************************************


def save_settings(values, file):
    settings = {}
    settings['model'] = values['-MODEL-']
    settings['model_config'] = values['-MODEL_CONFIG-']
    settings['prompts'] = values['-PROMPTS-'].split('\n')
    settings['suffix'] = values['-SUFFIX-']
    settings['fps'] = values['-FPS-']
    settings['make_gif'] = values['-MAKE_GIF-']
    settings['output_path'] = values['-OUTPUT_PATH-']
    settings['remove_frames_after'] = values['-REMOVE_FRAMES_AFTER-']
    try:
        settings['args'] = DeforumArgs(getargs(values, "general"))
        settings['anim_args'] = DeforumAnimArgs(getargs(values, "animation"))
        with open(file, 'wb') as f:
            pickle.dump(settings, f)
            return
        if file != 'saved_settings.pickle':
            print(f'Successfully saved to {file}!')
    except NameError:
        return


def load_settings(file):
    try:
        with open(file, 'rb') as f:
            settings = pickle.load(f)
    except FileNotFoundError:
        settings = {}
    if settings != {}:
        #main
        loadedprompts = '\n'.join(settings['prompts'])
        window['-PROMPTS-'].update(value=loadedprompts)
        window['-SUFFIX-'].update(value=settings['suffix'])
        window['-MODEL-'].update(value=settings['model'])
        window['-MODEL_CONFIG-'].update(value=settings['model_config'])
        window['-FPS-'].update(value=settings['fps'])
        window['-MAKE_GIF-'].update(value=settings['make_gif'])
        window['-OUTPUT_PATH-'].update(value=settings['output_path'])

        # general
        window['-WIDTH-'].update(value=settings['args']['W'])
        window['-HEIGHT-'].update(value=settings['args']['H'])
        window['-BIT_DEPTH_OUTPUT-'].update(value=settings['args']['bit_depth_output'])

        # sampling
        window['-SEED-'].update(value=settings['args']['seed'])
        window['-SAMPLER-'].update(value=settings['args']['sampler'])
        window['-SAMPLER_STEPS-'].update(value=settings['args']['steps'])
        window['-SAMPLER_SCALE-'].update(value=settings['args']['scale'])
        window['-DDIM_ETA-'].update(value=settings['args']['ddim_eta'])
        window['-SAVE_SAMPLES-'].update(value=settings['args']['save_samples'])
        window['-SAVE_SETTINGS-'].update(value=settings['args']['save_settings'])
        window['-SAVE_SAMPLE_PER_STEP-'].update(value=settings['args']['save_sample_per_step'])
        window['-SHOW_SAMPLE_PER_STEP-']  .update(value=settings['args']['show_sample_per_step'])

        # batch
        window['-BATCH_SIZE-'].update(value=settings['args']['n_batch'])
        window['-BATCH_NAME-'].update(value=settings['args']['batch_name'])
        window['-SEED_BEHAVIOR-'].update(value=settings['args']['seed_behavior'])
        window['-SEED_ITER_N-'].update(value=settings['args']['seed_iter_N'])
        window['-MAKE_GRID-'].update(value=settings['args']['make_grid'])
        window['-GRID_ROWS-'].update(value=settings['args']['grid_rows'])

        # init
        window['-USE_INIT-'].update(value=settings['args']['use_init'])
        window['-STRENGTH-'].update(value=settings['args']['strength'])
        window['-STRENGTH_0_NO_INIT-'].update(value=settings['args']['strength_0_no_init'])
        window['-INIT_IMAGE-'].update(value=settings['args']['init_image'])
        window['-USE_MASK-'].update(value=settings['args']['use_mask'])
        window['-USE_ALPHA_AS_MASK-'].update(value=settings['args']['use_alpha_as_mask'])
        window['-MASK_FILE-'].update(value=settings['args']['mask_file'])
        window['-INVERT_MASK-'].update(value=settings['args']['invert_mask'])
        window['-MASK_BRIGHTNESS_ADJUST-'].update(value=settings['args']['mask_brightness_adjust'])
        window['-MASK_CONTRAST_ADJUST-'].update(value=settings['args']['mask_contrast_adjust'])
        window['-OVERLAY_MASK-'].update(value=settings['args']['overlay_mask'])
        window['-MASK_OVERLAY_BLUR-'].update(value=settings['args']['mask_overlay_blur'])

        # exposure
        window['-MEAN_SCALE-'].update(value=settings['args']['mean_scale'])
        window['-VAR_SCALE-'].update(value=settings['args']['var_scale'])
        window['-EXPOSURE_SCALE-'].update(value=settings['args']['exposure_scale'])
        window['-EXPOSURE_TARGET-'].update(value=settings['args']['exposure_target'])
        
        # color match
        window['-COLORMATCH_SCALE-'].update(value=settings['args']['colormatch_scale'])
        window['-COLORMATCH_IMAGE-'].update(value=settings['args']['colormatch_image'])
        window['-COLORMATCH_N_COLORS-'].update(value=settings['args']['colormatch_n_colors'])
        window['-IGNORE_SAT_WEIGHT-'].update(value=settings['args']['ignore_sat_weight'])

        # clip aesthetic
        window['-CLIP_NAME-'].update(value=settings['args']['clip_name'])
        window['-CLIP_SCALE-'].update(value=settings['args']['clip_scale'])
        window['-AESTHETICS_SCALE-'].update(value=settings['args']['aesthetics_scale'])
        window['-CUTN-'].update(value=settings['args']['cutn'])
        window['-CUT_POW-'].update(value=settings['args']['cut_pow'])

        # other conditional                       
        window['-INIT_MSE_SCALE-'].update(value=settings['args']['init_mse_scale'])
        window['-INIT_MSE_IMAGE-'].update(value=settings['args']['init_mse_image'])
        window['-BLUE_SCALE-'].update(value=settings['args']['blue_scale'])

        # conditional gradient
        window['-GRADIENT_WRT-'].update(value=settings['args']['gradient_wrt'])
        window['-GRADIENT_ADD_TO-'].update(value=settings['args']['gradient_add_to'])
        window['-DECODE_METHOD-'].update(value=settings['args']['decode_method'])
        window['-GRAD_THRESHOLD_TYPE-'].update(value=settings['args']['grad_threshold_type'])
        window['-CLAMP_GRAD_THRESHOLD-'].update(value=settings['args']['clamp_grad_threshold'])
        window['-CLAMP_START-'].update(value=settings['args']['clamp_start'])
        window['-CLAMP_STOP-'].update(value=settings['args']['clamp_stop'])
        window['-GRAD_INJECT_TIMING-'].update(value=f'[{", ".join(str(x) for x in settings["args"]["grad_inject_timing"])}]')

        # speed vs vram
        window['-COND_UNCOND_SYNC-'].update(value=settings['args']['cond_uncond_sync'])

        #anim
        window['-ANIMATION_MODE-'].update(value=settings['anim_args']['animation_mode'])
        window['-MAX_FRAMES-'].update(value=settings['anim_args']['max_frames'])
        window['-BORDER-'].update(value=settings['anim_args']['border'])
        window['-REMOVE_FRAMES_AFTER-'].update(value=settings['remove_frames_after'])

        #motion
        window['-ANGLE-'].update(value=settings['anim_args']['angle'])
        window['-ZOOM-'].update(value=settings['anim_args']['zoom'])
        window['-TRANSLATION_X-'].update(value=settings['anim_args']['translation_x'])
        window['-TRANSLATION_Y-'].update(value=settings['anim_args']['translation_y'])
        window['-TRANSLATION_Z-'].update(value=settings['anim_args']['translation_z'])
        window['-ROTATION_3D_X-'].update(value=settings['anim_args']['rotation_3d_x'])
        window['-ROTATION_3D_Y-'].update(value=settings['anim_args']['rotation_3d_y'])
        window['-ROTATION_3D_Z-'].update(value=settings['anim_args']['rotation_3d_z'])
        window['-FLIP_2D_PERSPECTIVE-'].update(value=settings['anim_args']['flip_2d_perspective'])
        window['-PERSPECTIVE_FLIP_THETA-'].update(value=settings['anim_args']['perspective_flip_theta'])
        window['-PERSPECTIVE_FLIP_PHI-'].update(value=settings['anim_args']['perspective_flip_phi'])
        window['-PERSPECTIVE_FLIP_GAMMA-'].update(value=settings['anim_args']['perspective_flip_gamma'])
        window['-PERSPECTIVE_FLIP_FV-'].update(value=settings['anim_args']['perspective_flip_fv'])
        window['-NOISE_SCHEDULE-'].update(value=settings['anim_args']['noise_schedule'])
        window['-STRENGTH_SCHEDULE-'].update(value=settings['anim_args']['strength_schedule'])
        window['-CONTRAST_SCHEDULE-'].update(value=settings['anim_args']['contrast_schedule'])
        window['-HYBRID_VIDEO_COMP_ALPHA_SCHEDULE-'].update(value=settings['anim_args']['hybrid_video_comp_alpha_schedule'])
        window['-HYBRID_VIDEO_COMP_MASK_BLEND_ALPHA_SCHEDULE-'].update(value=settings['anim_args']['hybrid_video_comp_mask_blend_alpha_schedule'])
        window['-HYBRID_VIDEO_COMP_MASK_CONTRAST_SCHEDULE-'].update(value=settings['anim_args']['hybrid_video_comp_mask_contrast_schedule'])
        window['-HYBRID_VIDEO_COMP_MASK_AUTO_CONTRAST_CUTOFF_HIGH_SCHEDULE-'].update(value=settings['anim_args']['hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule'])
        window['-HYBRID_VIDEO_COMP_MASK_AUTO_CONTRAST_CUTOFF_LOW_SCHEDULE-'].update(value=settings['anim_args']['hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule'])

        # anti blur
        window['-KERNEL_SCHEDULE-'].update(value=settings['anim_args']['kernel_schedule'])
        window['-SIGMA_SCHEDULE-'].update(value=settings['anim_args']['sigma_schedule'])
        window['-AMOUNT_SCHEDULE-'].update(value=settings['anim_args']['amount_schedule'])
        window['-THRESHOLD_SCHEDULE-'].update(value=settings['anim_args']['threshold_schedule'])       
        
        # coherence
        window['-COLOR_COHERENCE-'].update(value=settings['anim_args']['color_coherence']) 
        window['-DIFFUSION_CADENCE-'].update(value=settings['anim_args']['diffusion_cadence'])

        # depth warping
        window['-USE_DEPTH_WARPING-'].update(value=settings['anim_args']['use_depth_warping'])
        window['-MIDAS_WEIGHT-'].update(value=settings['anim_args']['midas_weight'])
        window['-FOV-'].update(value=settings['anim_args']['fov'])
        window['-PADDING_MODE-'].update(value=settings['anim_args']['padding_mode'])
        window['-SAMPLING_MODE-'].update(value=settings['anim_args']['sampling_mode'])
        window['-SAVE_DEPTH_MAPS-'].update(value=settings['anim_args']['save_depth_maps'])

        # video input
        window['-VIDEO_INIT_PATH-'].update(value=settings['anim_args']['video_init_path'])
        window['-EXTRACT_NTH_FRAME-'].update(value=settings['anim_args']['extract_nth_frame'])
        window['-OVERWRITE_EXTRACTED_FRAMES-'].update(value=settings['anim_args']['overwrite_extracted_frames'])
        window['-USE_MASK_VIDEO-'].update(value=settings['anim_args']['use_mask_video'])
        window['-VIDEO_MASK_PATH-'].update(value=settings['anim_args']['video_mask_path'])

        # hybrid
        window['-HYBRID_VIDEO_GENERATE_INPUTFRAMES-'].update(value=settings['anim_args']['hybrid_video_generate_inputframes'])
        window['-HYBRID_VIDEO_USE_FIRST_FRAME_AS_INIT_IMAGE-'].update(value=settings['anim_args']['hybrid_video_use_first_frame_as_init_image'])
        window['-HYBRID_VIDEO_MOTION-'].update(value=settings['anim_args']['hybrid_video_motion'])
        window['-HYBRID_VIDEO_FLOW_METHOD-'].update(value=settings['anim_args']['hybrid_video_flow_method'])
        window['-HYBRID_VIDEO_COMPOSITE-'].update(value=settings['anim_args']['hybrid_video_composite'])
        window['-HYBRID_VIDEO_COMP_MASK_TYPE-'].update(value=settings['anim_args']['hybrid_video_comp_mask_type'])
        window['-HYBRID_VIDEO_COMP_MASK_INVERSE-'].update(value=settings['anim_args']['hybrid_video_comp_mask_inverse'])
        window['-HYBRID_VIDEO_COMP_MASK_EQUALIZE-'].update(value=settings['anim_args']['hybrid_video_comp_mask_equalize'])
        window['-HYBRID_VIDEO_COMP_MASK_AUTO_CONTRAST-'].update(value=settings['anim_args']['hybrid_video_comp_mask_auto_contrast'])
        window['-HYBRID_VIDEO_COMP_SAVE_EXTRA_FRAMES-'].update(value=settings['anim_args']['hybrid_video_comp_save_extra_frames'])        
        window['-HYBRID_VIDEO_USE_VIDEO_AS_MSE_IMAGE-'].update(value=settings['anim_args']['hybrid_video_use_video_as_mse_image'])

        # interpolation
        window['-INTERPOLATE_KEY_FRAMES-'].update(value=settings['anim_args']['interpolate_key_frames'])
        window['-INTERPOLATE_X_FRAMES-'].update(value=settings['anim_args']['interpolate_x_frames'])

        # resume
        window['-RESUME_FROM_TIMESTRING-'].update(value=settings['anim_args']['resume_from_timestring'])
        window['-RESUME_TIMESTRING-'].update(value=settings['anim_args']['resume_timestring'])

        if file != 'saved_settings.pickle':
            print(f'Successfully saved to {file}!')
        return settings
    else:
        return settings

def create_video(args, anim_args):
    bitdepth_extension = "exr" if args.bit_depth_output == 32 else "png"
    image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.{bitdepth_extension}")
    mp4_path = os.path.join(args.outdir, f"{args.timestring}.mp4")
    max_frames = str(anim_args.max_frames)
    print(f"{image_path} -> {mp4_path}")

    # make video
    cmd = [
        'ffmpeg',
        '-y',
        '-vcodec', 'png',
        '-r', values['-FPS-'],
        '-start_number', str(0),
        '-i', image_path,
        '-frames:v', max_frames,
        '-c:v', 'libx264',
        '-vf',
        f'fps={values["-FPS-"]}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'veryfast',
        '-pattern_type', 'sequence',
        mp4_path
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)
    if values['-MAKE_GIF-']:
        gif_path = os.path.splitext(mp4_path)[0]+'.gif'
        cmd_gif = [
            'ffmpeg',
            '-y',
            '-i', mp4_path,
            '-r', values['-FPS-'],
            gif_path
        ]
        subprocess.Popen(cmd_gif, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# load older settings
open_file_name = ''
current_settings = load_settings('saved_settings.pickle')
if current_settings == {}:
    initmodel = 'v1-5-pruned-emaonly.ckpt'
    initconfig = 'v1-inference.yaml'
    window['-BATCH_NAME-'].update(value='Testing')
else:
    initmodel = current_settings['model']
    initconfig = current_settings['model_config']

sub_p_res = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'], stdout=subprocess.PIPE).stdout.decode('utf-8')
print(f"{sub_p_res[:-1]}")

KThread(target=load_root_model, args=(initmodel, initconfig, f'{os.path.dirname(os.path.abspath(__file__))}/output'), daemon=True).start()

renderprocess = None

while True:
    event, values = window.read()
    window['-LOG-'].Update(output_vals)
    window['-LOG-'].Update(error_vals)
    if event == '-RENDER-':
        if values['-ANIMATION_MODE-'] == 'None':
            args = DeforumArgs(getargs(values, "general"))
            save_settings(values, 'saved_settings.pickle')
            renderprocess = KThread(target=do_render, args=(args,), daemon=True)
            renderprocess.start()
        else:
            args = DeforumArgs(getargs(values, "general"))
            anim_args = DeforumAnimArgs(getargs(values, "animation"))
            save_settings(values, 'saved_settings.pickle')
            renderprocess = KThread(target=do_video_render, args=(args, anim_args,), daemon=True)
            renderprocess.start()

    if event == '-RELOAD-':
        KThread(target=load_root_model, args=(values['-MODEL-'], values['-MODEL_CONFIG-'], values['-OUTPUT_PATH-']), daemon=True).start()
        save_settings(values, 'saved_settings.pickle')

    if event == '-CANCEL-':
        if renderprocess is not None:
            renderprocess.kill()
            set_ready(True)
            print('Process Canceled!')
    if event == '-RANDOM_SEED-':
        window['-SEED-'].update(value=random.randint(0, 2**32 - 1))
    if event == 'Open::-OPEN-':
        if sg.running_mac():
            open_file_name = sg.tk.filedialog.askopenfilename(initialdir=values['-OUTPUT_PATH-'], defaultextension='.deforum')  # show the 'get files' dialog box
        else:
            open_file_name = sg.tk.filedialog.askopenfilename(filetypes=[("Deforum File", "*.deforum")], initialdir=values['-OUTPUT_PATH-'], defaultextension='.deforum')  # show the 'get files' dialog box
        load_settings(open_file_name)
    if event == 'Save::-SAVE-':
        if sg.running_mac():
            save_file_name = sg.tk.filedialog.asksaveasfilename(defaultextension='.deforum', initialdir=values['-OUTPUT_PATH-'])
        else:
            save_file_name = sg.tk.filedialog.asksaveasfilename(filetypes=[("Deforum File", "*.deforum")], defaultextension='.deforum', initialdir=values['-OUTPUT_PATH-'])
        save_settings(values, save_file_name)
    if event in (sg.WINDOW_CLOSE_ATTEMPTED_EVENT, 'Exit'):
        save_settings(values, 'saved_settings.pickle')
        if open_file_name:
            save_settings(values, open_file_name)
        break

window.close()
