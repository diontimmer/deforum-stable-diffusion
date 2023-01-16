from gui.gui_const import *
import gui.gui_interface as gui
import pickle
from gui.gui_settings_overrides import DeforumArgs, DeforumAnimArgs
from gui.gui_helpers import *
import sys
import os

def save_settings(values, file):
    settings = {}
    settings['model'] = values['-MODEL-']
    settings['model_config'] = values['-MODEL_CONFIG-']
    settings['prompts'] = values['-PROMPTS-'].split('\n')
    settings['suffix'] = values['-SUFFIX-']
    settings['fps'] = values['-FPS-']
    settings['make_gif'] = values['-MAKE_GIF-']
    settings['models_path'] = values['-MODELS_PATH-']
    settings['output_path'] = values['-OUTPUT_PATH-']
    settings['remove_frames_after'] = values['-REMOVE_FRAMES_AFTER-']
    try:
        settings['args'] = DeforumArgs(get_args_from_gui(values, "general"))
        settings['anim_args'] = DeforumAnimArgs(get_args_from_gui(values, "animation"))
        with open(file, 'wb') as f:
            pickle.dump(settings, f)
            return
        if file != 'saved_settings.pickle':
            gui.gui_print(f'Successfully saved to {file}!')
    except NameError:
        return


def load_settings(file):
    try:
        with open(file, 'rb') as f:
            settings = pickle.load(f)
    except FileNotFoundError:
        gui.guiwindow['-MODEL-'].update(value='Protogen_V2.2.ckpt', values=getmodels())
        settings = {}
    if settings != {}:
        #main
        loadedprompts = '\n'.join(settings['prompts'])
        gui.guiwindow['-MODELS_PATH-'].update(value=settings['models_path'])
        gui.guiwindow['-OUTPUT_PATH-'].update(value=settings['output_path'])
        gui.guiwindow['-PROMPTS-'].update(value=loadedprompts)
        gui.guiwindow['-SUFFIX-'].update(value=settings['suffix'])
        gui.guiwindow['-MODEL-'].update(value=settings['model'], values=getmodels())
        gui.guiwindow['-MODEL_CONFIG-'].update(value=settings['model_config'])
        gui.guiwindow['-FPS-'].update(value=settings['fps'])
        gui.guiwindow['-MAKE_GIF-'].update(value=settings['make_gif'])

        # general
        gui.guiwindow['-WIDTH-'].update(value=settings['args']['W'])
        gui.guiwindow['-HEIGHT-'].update(value=settings['args']['H'])
        gui.guiwindow['-BIT_DEPTH_OUTPUT-'].update(value=settings['args']['bit_depth_output'])

        # sampling
        gui.guiwindow['-SEED-'].update(value=settings['args']['seed'])
        gui.guiwindow['-SAMPLER-'].update(value=settings['args']['sampler'])
        gui.guiwindow['-SAMPLER_STEPS-'].update(value=settings['args']['steps'])
        gui.guiwindow['-SAMPLER_SCALE-'].update(value=settings['args']['scale'])
        gui.guiwindow['-DDIM_ETA-'].update(value=settings['args']['ddim_eta'])
        gui.guiwindow['-SAVE_SAMPLES-'].update(value=settings['args']['save_samples'])
        gui.guiwindow['-SAVE_SETTINGS-'].update(value=settings['args']['save_settings'])
        gui.guiwindow['-SAVE_SAMPLE_PER_STEP-'].update(value=settings['args']['save_sample_per_step'])
        gui.guiwindow['-SHOW_SAMPLE_PER_STEP-']  .update(value=settings['args']['show_sample_per_step'])

        # batch
        gui.guiwindow['-BATCH_SIZE-'].update(value=settings['args']['n_batch'])
        gui.guiwindow['-BATCH_NAME-'].update(value=settings['args']['batch_name'])
        gui.guiwindow['-SEED_BEHAVIOR-'].update(value=settings['args']['seed_behavior'])
        gui.guiwindow['-SEED_ITER_N-'].update(value=settings['args']['seed_iter_N'])
        gui.guiwindow['-MAKE_GRID-'].update(value=settings['args']['make_grid'])
        gui.guiwindow['-GRID_ROWS-'].update(value=settings['args']['grid_rows'])

        # init
        gui.guiwindow['-USE_INIT-'].update(value=settings['args']['use_init'])
        gui.guiwindow['-STRENGTH-'].update(value=settings['args']['strength'])
        gui.guiwindow['-STRENGTH_0_NO_INIT-'].update(value=settings['args']['strength_0_no_init'])
        gui.guiwindow['-INIT_IMAGE-'].update(value=settings['args']['init_image'])
        gui.guiwindow['-USE_MASK-'].update(value=settings['args']['use_mask'])
        gui.guiwindow['-USE_ALPHA_AS_MASK-'].update(value=settings['args']['use_alpha_as_mask'])
        gui.guiwindow['-MASK_FILE-'].update(value=settings['args']['mask_file'])
        gui.guiwindow['-INVERT_MASK-'].update(value=settings['args']['invert_mask'])
        gui.guiwindow['-MASK_BRIGHTNESS_ADJUST-'].update(value=settings['args']['mask_brightness_adjust'])
        gui.guiwindow['-MASK_CONTRAST_ADJUST-'].update(value=settings['args']['mask_contrast_adjust'])
        gui.guiwindow['-OVERLAY_MASK-'].update(value=settings['args']['overlay_mask'])
        gui.guiwindow['-MASK_OVERLAY_BLUR-'].update(value=settings['args']['mask_overlay_blur'])

        # exposure
        gui.guiwindow['-MEAN_SCALE-'].update(value=settings['args']['mean_scale'])
        gui.guiwindow['-VAR_SCALE-'].update(value=settings['args']['var_scale'])
        gui.guiwindow['-EXPOSURE_SCALE-'].update(value=settings['args']['exposure_scale'])
        gui.guiwindow['-EXPOSURE_TARGET-'].update(value=settings['args']['exposure_target'])
        
        # color match
        gui.guiwindow['-COLORMATCH_SCALE-'].update(value=settings['args']['colormatch_scale'])
        gui.guiwindow['-COLORMATCH_IMAGE-'].update(value=settings['args']['colormatch_image'])
        gui.guiwindow['-COLORMATCH_N_COLORS-'].update(value=settings['args']['colormatch_n_colors'])
        gui.guiwindow['-IGNORE_SAT_WEIGHT-'].update(value=settings['args']['ignore_sat_weight'])

        # clip aesthetic
        gui.guiwindow['-CLIP_NAME-'].update(value=settings['args']['clip_name'])
        gui.guiwindow['-CLIP_SCALE-'].update(value=settings['args']['clip_scale'])
        gui.guiwindow['-AESTHETICS_SCALE-'].update(value=settings['args']['aesthetics_scale'])
        gui.guiwindow['-CUTN-'].update(value=settings['args']['cutn'])
        gui.guiwindow['-CUT_POW-'].update(value=settings['args']['cut_pow'])

        # other conditional                       
        gui.guiwindow['-INIT_MSE_SCALE-'].update(value=settings['args']['init_mse_scale'])
        gui.guiwindow['-INIT_MSE_IMAGE-'].update(value=settings['args']['init_mse_image'])
        gui.guiwindow['-BLUE_SCALE-'].update(value=settings['args']['blue_scale'])

        # conditional gradient
        gui.guiwindow['-GRADIENT_WRT-'].update(value=settings['args']['gradient_wrt'])
        gui.guiwindow['-GRADIENT_ADD_TO-'].update(value=settings['args']['gradient_add_to'])
        gui.guiwindow['-DECODE_METHOD-'].update(value=settings['args']['decode_method'])
        gui.guiwindow['-GRAD_THRESHOLD_TYPE-'].update(value=settings['args']['grad_threshold_type'])
        gui.guiwindow['-CLAMP_GRAD_THRESHOLD-'].update(value=settings['args']['clamp_grad_threshold'])
        gui.guiwindow['-CLAMP_START-'].update(value=settings['args']['clamp_start'])
        gui.guiwindow['-CLAMP_STOP-'].update(value=settings['args']['clamp_stop'])
        gui.guiwindow['-GRAD_INJECT_TIMING-'].update(value=f'[{", ".join(str(x) for x in settings["args"]["grad_inject_timing"])}]')

        # speed vs vram
        gui.guiwindow['-COND_UNCOND_SYNC-'].update(value=settings['args']['cond_uncond_sync'])

        #anim
        gui.guiwindow['-ANIMATION_MODE-'].update(value=settings['anim_args']['animation_mode'])
        gui.guiwindow['-MAX_FRAMES-'].update(value=settings['anim_args']['max_frames'])
        gui.guiwindow['-BORDER-'].update(value=settings['anim_args']['border'])
        gui.guiwindow['-REMOVE_FRAMES_AFTER-'].update(value=settings['remove_frames_after'])

        #motion
        gui.guiwindow['-ANGLE-'].update(value=settings['anim_args']['angle'])
        gui.guiwindow['-ZOOM-'].update(value=settings['anim_args']['zoom'])
        gui.guiwindow['-TRANSLATION_X-'].update(value=settings['anim_args']['translation_x'])
        gui.guiwindow['-TRANSLATION_Y-'].update(value=settings['anim_args']['translation_y'])
        gui.guiwindow['-TRANSLATION_Z-'].update(value=settings['anim_args']['translation_z'])
        gui.guiwindow['-ROTATION_3D_X-'].update(value=settings['anim_args']['rotation_3d_x'])
        gui.guiwindow['-ROTATION_3D_Y-'].update(value=settings['anim_args']['rotation_3d_y'])
        gui.guiwindow['-ROTATION_3D_Z-'].update(value=settings['anim_args']['rotation_3d_z'])
        gui.guiwindow['-FLIP_2D_PERSPECTIVE-'].update(value=settings['anim_args']['flip_2d_perspective'])
        gui.guiwindow['-PERSPECTIVE_FLIP_THETA-'].update(value=settings['anim_args']['perspective_flip_theta'])
        gui.guiwindow['-PERSPECTIVE_FLIP_PHI-'].update(value=settings['anim_args']['perspective_flip_phi'])
        gui.guiwindow['-PERSPECTIVE_FLIP_GAMMA-'].update(value=settings['anim_args']['perspective_flip_gamma'])
        gui.guiwindow['-PERSPECTIVE_FLIP_FV-'].update(value=settings['anim_args']['perspective_flip_fv'])
        gui.guiwindow['-NOISE_SCHEDULE-'].update(value=settings['anim_args']['noise_schedule'])
        gui.guiwindow['-STRENGTH_SCHEDULE-'].update(value=settings['anim_args']['strength_schedule'])
        gui.guiwindow['-CONTRAST_SCHEDULE-'].update(value=settings['anim_args']['contrast_schedule'])
        gui.guiwindow['-HYBRID_VIDEO_COMP_ALPHA_SCHEDULE-'].update(value=settings['anim_args']['hybrid_video_comp_alpha_schedule'])
        gui.guiwindow['-HYBRID_VIDEO_COMP_MASK_BLEND_ALPHA_SCHEDULE-'].update(value=settings['anim_args']['hybrid_video_comp_mask_blend_alpha_schedule'])
        gui.guiwindow['-HYBRID_VIDEO_COMP_MASK_CONTRAST_SCHEDULE-'].update(value=settings['anim_args']['hybrid_video_comp_mask_contrast_schedule'])
        gui.guiwindow['-HYBRID_VIDEO_COMP_MASK_AUTO_CONTRAST_CUTOFF_HIGH_SCHEDULE-'].update(value=settings['anim_args']['hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule'])
        gui.guiwindow['-HYBRID_VIDEO_COMP_MASK_AUTO_CONTRAST_CUTOFF_LOW_SCHEDULE-'].update(value=settings['anim_args']['hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule'])

        # anti blur
        gui.guiwindow['-KERNEL_SCHEDULE-'].update(value=settings['anim_args']['kernel_schedule'])
        gui.guiwindow['-SIGMA_SCHEDULE-'].update(value=settings['anim_args']['sigma_schedule'])
        gui.guiwindow['-AMOUNT_SCHEDULE-'].update(value=settings['anim_args']['amount_schedule'])
        gui.guiwindow['-THRESHOLD_SCHEDULE-'].update(value=settings['anim_args']['threshold_schedule'])       
        
        # coherence
        gui.guiwindow['-COLOR_COHERENCE-'].update(value=settings['anim_args']['color_coherence']) 
        gui.guiwindow['-DIFFUSION_CADENCE-'].update(value=settings['anim_args']['diffusion_cadence'])

        # depth warping
        gui.guiwindow['-USE_DEPTH_WARPING-'].update(value=settings['anim_args']['use_depth_warping'])
        gui.guiwindow['-MIDAS_WEIGHT-'].update(value=settings['anim_args']['midas_weight'])
        gui.guiwindow['-FOV-'].update(value=settings['anim_args']['fov'])
        gui.guiwindow['-PADDING_MODE-'].update(value=settings['anim_args']['padding_mode'])
        gui.guiwindow['-SAMPLING_MODE-'].update(value=settings['anim_args']['sampling_mode'])
        gui.guiwindow['-SAVE_DEPTH_MAPS-'].update(value=settings['anim_args']['save_depth_maps'])

        # video input
        gui.guiwindow['-VIDEO_INIT_PATH-'].update(value=settings['anim_args']['video_init_path'])
        gui.guiwindow['-EXTRACT_NTH_FRAME-'].update(value=settings['anim_args']['extract_nth_frame'])
        gui.guiwindow['-OVERWRITE_EXTRACTED_FRAMES-'].update(value=settings['anim_args']['overwrite_extracted_frames'])
        gui.guiwindow['-USE_MASK_VIDEO-'].update(value=settings['anim_args']['use_mask_video'])
        gui.guiwindow['-VIDEO_MASK_PATH-'].update(value=settings['anim_args']['video_mask_path'])

        # hybrid
        gui.guiwindow['-HYBRID_VIDEO_GENERATE_INPUTFRAMES-'].update(value=settings['anim_args']['hybrid_video_generate_inputframes'])
        gui.guiwindow['-HYBRID_VIDEO_USE_FIRST_FRAME_AS_INIT_IMAGE-'].update(value=settings['anim_args']['hybrid_video_use_first_frame_as_init_image'])
        gui.guiwindow['-HYBRID_VIDEO_MOTION-'].update(value=settings['anim_args']['hybrid_video_motion'])
        gui.guiwindow['-HYBRID_VIDEO_FLOW_METHOD-'].update(value=settings['anim_args']['hybrid_video_flow_method'])
        gui.guiwindow['-HYBRID_VIDEO_COMPOSITE-'].update(value=settings['anim_args']['hybrid_video_composite'])
        gui.guiwindow['-HYBRID_VIDEO_COMP_MASK_TYPE-'].update(value=settings['anim_args']['hybrid_video_comp_mask_type'])
        gui.guiwindow['-HYBRID_VIDEO_COMP_MASK_INVERSE-'].update(value=settings['anim_args']['hybrid_video_comp_mask_inverse'])
        gui.guiwindow['-HYBRID_VIDEO_COMP_MASK_EQUALIZE-'].update(value=settings['anim_args']['hybrid_video_comp_mask_equalize'])
        gui.guiwindow['-HYBRID_VIDEO_COMP_MASK_AUTO_CONTRAST-'].update(value=settings['anim_args']['hybrid_video_comp_mask_auto_contrast'])
        gui.guiwindow['-HYBRID_VIDEO_COMP_SAVE_EXTRA_FRAMES-'].update(value=settings['anim_args']['hybrid_video_comp_save_extra_frames'])        
        gui.guiwindow['-HYBRID_VIDEO_USE_VIDEO_AS_MSE_IMAGE-'].update(value=settings['anim_args']['hybrid_video_use_video_as_mse_image'])

        # interpolation
        gui.guiwindow['-INTERPOLATE_KEY_FRAMES-'].update(value=settings['anim_args']['interpolate_key_frames'])
        gui.guiwindow['-INTERPOLATE_X_FRAMES-'].update(value=settings['anim_args']['interpolate_x_frames'])

        # resume
        gui.guiwindow['-RESUME_FROM_TIMESTRING-'].update(value=settings['anim_args']['resume_from_timestring'])
        gui.guiwindow['-RESUME_TIMESTRING-'].update(value=settings['anim_args']['resume_timestring'])

        if file != 'saved_settings.pickle':
            gui.gui_print(f'Successfully loaded {file}!')
        return settings
    else:
        return settings