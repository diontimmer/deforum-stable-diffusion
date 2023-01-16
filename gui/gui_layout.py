import PySimpleGUI as sg
from gui.gui_const import *
from gui.gui_helpers import *
import gui.gui_interface as gui

app_log = bool(gui.get_config_value('log_in_app'))

# ****************************************************************************
# *                               setup window                               *
# ****************************************************************************

sg.theme(gui.get_config_value('theme'))

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

log_ml = sg.Multiline(disabled=True, expand_x=True, expand_y=True, reroute_stdout=app_log, autoscroll=True, auto_refresh=True, key='-LOG-')

loading_gif_img = sg.Image(background_color=sg.theme_background_color(), key='-LOADINGGIF-')

prog_bar = sg.ProgressBar(100, orientation='h', expand_x=True, key='-PROGRESS-', size=(35,35))

prompt_box = sg.Column([
    [sg.Text('Prompts: (Separated by new line) '), sg.Text('Suffix: '), sg.Input('', key='-SUFFIX-', expand_x=True)],
    [sg.Multiline(expand_x=True, expand_y=False, key='-PROMPTS-', size=(0,20))],
    [sg.Text('Output Path: '), sg.Input(f'{os.path.dirname(os.path.abspath(__file__))}\\output', key='-OUTPUT_PATH-', size=(80, 1)), sg.FileBrowse()],
    [sg.Button('Render', key='-RENDER-'), sg.Button('Load Model', key='-RELOAD-'), sg.Button('Cancel', key='-CANCEL-'), loading_gif_img],
    [log_ml],
    [prog_bar],
    ], vertical_alignment='top', expand_x=True, expand_y=True)

current_image = sg.Column([[
    sg.Image(key='-IMAGE-', size=(768, 768), background_color="#2e3238"), 
    ]], expand_x=False)

gui_layout = [
    [sg.Menubar(menu_def, key='-MENUBAR-')],
    [current_image, prompt_box], 
    [tab_layout],
    ]