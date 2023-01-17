from gui.gui_const import *
import gui.gui_interface as gui
from helpers.save_images import get_output_folder
from types import SimpleNamespace
from gui.gui_settings_overrides import Root
from helpers.aesthetics import load_aesthetics_model
import time
import random
import os
import subprocess
import sys
import gc
import torch
import re
import cv2

sys.path.extend(['src'])
from helpers.render import render_animation, render_input_video, render_image_batch, render_interpolation
from helpers.model_load import load_model, get_model_output_paths

def extract_percentage(output):
    if output:
        last_percentage = None
        for match in re.finditer(r'\d+%', output):
            last_percentage = int(match.group(0).strip('%'))
        return last_percentage
    else:
        return None


def do_render(values, args):
    negative_prompts = []
    prompts = []
    # get box
    box_prompts = values['-PROMPTS-'].split('\n')

    # clean empty
    box_prompts = [x for x in box_prompts if x]

    # get negatives
    for prompt in box_prompts:
        # before keyframe

        if prompt[0] == '-':
            prompt = prompt.replace(prompt[0], '')
            prompt = prompt.split(': ')[-1]
            negative_prompts.append(prompt)
        # after keyframe

        elif prompt.split(': ')[-1][0] == '-':
            prompt = prompt.split(': ')[-1]
            #cut first letter off prompt
            prompt = prompt.replace(prompt[0], '')
            negative_prompts.append(prompt)

        # get positive
        else:
            prompts.append(prompt.split(': ')[-1])

    # get negative dict
    negative_prompts_dict = {f'{i}: ': x for i, x in enumerate(negative_prompts)}
    if negative_prompts_dict == {}:
        negative_prompts_dict = {0: ''}

    # get positive dict
    prompts = [prompt + ', ' + values['-SUFFIX-'] for prompt in prompts]
    prompts_dict = {f'{i}: ': x for i, x in enumerate(prompts)}

    args = process_args(args)
    gui.set_ready(False)
    gc.collect()
    torch.cuda.empty_cache()
    # DISPLAY IMAGE IN deforum/helpers.render.py render_image_batch
    gui.gui_print('Rendering batch images, verbose will be in the external console window.', text_color='yellow')
    sys.stderr = gui.reroute_stderr
    render_image_batch(gui.root, args, prompts_dict, negative_prompts_dict)
    sys.stderr = sys.__stderr__
    gui.clean_err_io()
    gui.set_ready(True)
    return


def do_video_render(values, args, anim_args):
    prompts = values['-PROMPTS-'].split('\n')
    # clear empty
    prompts = [x for x in prompts if x]
    # check for keyframes
    for prompt in prompts:
        if not prompt.replace('-', '')[0].isdigit():
            gui.gui_print('Please note the keyframes in your animation prompts.', text_color='red')
            gui.set_ready(True)
            return
    prompts_dict = {}
    negative_prompts_dict = {}
    for prompt in prompts:
        # check for neg before keyframe
        if prompt[0] == '-':
            prompt = prompt.replace(prompt[0], '')
            keyframe = prompt.split(': ')[0]
            prompt_text = prompt.split(': ')[1]
            negative_prompts_dict[int(keyframe)] = prompt_text
        else:
            keyframe = prompt.split(': ')[0]
            prompt_text = prompt.split(': ')[1]

            # check for neg before prompt text
            if prompt_text[0] == '-':
                prompt_text = prompt_text.replace(prompt_text[0], '')
                negative_prompts_dict[int(keyframe)] = prompt_text

            else:  # handle as positive prompt
                prompts_dict[int(keyframe)] = prompt_text + ', ' + values['-SUFFIX-']
    if negative_prompts_dict == {}:
        negative_prompts_dict = {0: ''}
    args = process_args(args)
    anim_args_result = process_anim_args(anim_args, args)
    anim_args = anim_args_result[0]
    args = anim_args_result[1]
    gui.set_ready(False)
    gc.collect()
    torch.cuda.empty_cache()
    gui.gui_print('Rendering animation, verbose will be in the external console window.', text_color='yellow')
    sys.stderr = gui.reroute_stderr
    match anim_args.animation_mode:
        case '2D' | '3D':
            render_animation(gui.root, anim_args, args, prompts_dict, negative_prompts_dict)
        case 'Video Input':
            render_input_video(gui.root, anim_args, args, prompts_dict, negative_prompts_dict)
        case 'Interpolation':
            render_interpolation(gui.root, anim_args, args, prompts_dict, negative_prompts_dict)
    sys.stderr = sys.__stderr__
    create_video(args, anim_args, values["-FPS-"], values["-MAKE_GIF-"], values['-PATROL_CYCLE-'])
    gui.clean_err_io()  
    if values['-REMOVE_FRAMES_AFTER-']:
        for file in os.listdir(args.outdir):
            if file.endswith(".png"):
                os.remove(os.path.join(args.outdir, file))        
    gui.set_ready(True)
    return


def load_root_model(modelname, modelconfig, outputpath):
    gui.set_ready(False)
    if gui.root is not None:
        try:
            del gui.root
            gui.gui_print('Removed root model.', text_color='orange')
        except NameError:
            pass
    model_folder = gui.guiwindow['-MODELS_PATH-'].get()
    loaded_model_path = f'{model_folder}/{modelname}'
    gui.gui_print(f'Loading model {modelname} with config {modelconfig}...', text_color='yellow')
    root = Root(modelpath=loaded_model_path, model_config_override=modelconfig, outputpath=outputpath, modelpaths=model_folder)
    root = SimpleNamespace(**root)
    root.models_path, root.output_path = get_model_output_paths(root)
    try:
        root.model, root.device = load_model(root, load_on_run_all=True, check_sha256=True)
    except RuntimeError:
        gui.gui_print('Error loading model, did you load the correct config?', text_color='red')
        gui.set_ready(False, override_loading=False)
        return
    gui.root = root
    gui.gui_print('Model successfully loaded!', text_color='lightgreen')
    gui.set_ready(True)
    return


def create_video(args, anim_args, fps, make_gif, patrol_cycle):
    bitdepth_extension = "exr" if args.bit_depth_output == 32 else "png"
    image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.{bitdepth_extension}")
    mp4_path = os.path.join(args.outdir, f"{args.timestring}.mp4")
    max_frames = str(anim_args.max_frames)
    gui.gui_print(f'Creating video from frames at {mp4_path} with {fps} fps..', text_color='yellow')

    # make video
    cmd = [
        'ffmpeg',
        '-y',
        '-vcodec', 'png',
        '-r', fps,
        '-start_number', str(0),
        '-i', image_path,
        '-frames:v', max_frames,
        '-c:v', 'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'veryfast',
        '-pattern_type', 'sequence',
        mp4_path
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        gui.gui_print(stderr)
        raise RuntimeError(stderr)
    if patrol_cycle:
        gui.gui_print('Creating Patrol Cycle..', text_color='yellow')
        # Load the video
        cap = cv2.VideoCapture(mp4_path)
        # Get the video frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        # Reverse the frames
        reversed_frames = frames[::-1]

        # Append the original frames to the reversed frames
        frames = frames + reversed_frames
        # Create a video writer object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(mp4_path.replace('.mp4', '_patrol_cycle.mp4'), fourcc, float(fps), (args.W, args.H), isColor=True)
        # Write the frames to the output video
        for frame in frames:
            out.write(frame)
        # Release the video writer and capture objects
        out.release()
        cap.release()
    if make_gif:
        gui.gui_print('Creating GIF..', text_color='yellow')
        gif_path = os.path.splitext(mp4_path)[0]+'.gif'
        cmd_gif = [
            'ffmpeg',
            '-y',
            '-i', mp4_path,
            '-r', fps,
            gif_path
        ]
        subprocess.Popen(cmd_gif, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    gui.gui_print('Video creation complete!', text_color='lightgreen')


def getmodels():
    modelvals = list(model_map.keys())
    # check if directory is a directory
    loadedpath = os.path.normpath(gui.guiwindow['-MODELS_PATH-'].get())
    if os.path.isdir(loadedpath):
        for file in os.listdir(loadedpath):
            if file not in modelvals and file.endswith('.ckpt'):
                modelvals.append(file)
    return modelvals


def getconfigs():
    configvals = []
    for file in os.listdir("configs"):
        configvals.append(file)
    return configvals


def process_args(args):
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


def process_anim_args(anim_args, args):
    anim_args = SimpleNamespace(**anim_args)
    if anim_args.animation_mode == 'None':
        anim_args.max_frames = 1
    elif anim_args.animation_mode == 'Video Input':
        args.use_init = True
    return anim_args, args


def get_args_from_gui(values, tab):
    match tab:
        case "general":
            args = {
                'W': values['-WIDTH-'], 
                'H': values['-HEIGHT-'],
                'bit_depth_output': values['-BIT_DEPTH_OUTPUT-'],
                'outdir': get_output_folder(values['-OUTPUT_PATH-'], values['-BATCH_NAME-']),

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


def print_gpu():
    gui.gui_print('Getting GPU Info...', text_color='yellow')
    sub_p_res = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'], stdout=subprocess.PIPE).stdout.decode('utf-8')
    gui.gui_print(f"{sub_p_res[:-1]}", text_color='lightgreen')