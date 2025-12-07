from harp.asura.asura import ASURA
from harp.asura.modules import MLP
import torch
import numpy as np
import argparse
import sounddevice
try:
    import board
    import busio
    from as5600 import AS5600
    from adafruit_tca9548a import TCA9548A
    MODE = 'box'
except ImportError as e:
    import dearpygui.dearpygui as dpg
    from scipy.io.wavfile import write as write_wav
    MODE = 'pc'
import json
import os
import time
import multiprocessing as mp
from matplotlib.pyplot import get_cmap
from harp.modeling.util import make_mel_filterbank

I2C_MUX_ADDR_0 = 0x70
I2C_MUX_ADDR_1 = 0x71

NUM_KNOBS = 16

SPECTROGRAM_WIDTH = 1024
SPECTROGRAM_HEIGHT = 256

def control_routine_pc(control_array, spectrogram, wake_event, gen_event, reset_event, listen_event, save_event, close_event):
    dpg.create_context()

    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0, category=dpg.mvThemeCat_Core)

    with dpg.theme() as panel_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 15, 10, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 7, category=dpg.mvThemeCat_Core)

    dpg.bind_theme(global_theme)

    dpg.create_viewport(title='ASURA', width=1024, height=512)
    dpg.setup_dearpygui()

    spec_cmap = get_cmap('magma')

    local_spec_buffer = np.zeros(SPECTROGRAM_WIDTH * SPECTROGRAM_HEIGHT * 4, dtype=np.float32)
    with dpg.texture_registry():
        dpg.add_raw_texture(width=SPECTROGRAM_WIDTH, height=SPECTROGRAM_HEIGHT, default_value=local_spec_buffer, format=dpg.mvFormat_Float_rgba, tag="spectrogram")

    COLS = 4
    ROWS = (len(control_array) + COLS - 1)//COLS
    def knob_callback(knob, val, i):
        with control_array.get_lock():
            control_array[i] = val
    def start_stop_callback(b, a, d):
        if gen_event.is_set():
            gen_event.clear()
            wake_event.clear()
            dpg.set_item_label('start', 'Start')
        else:
            gen_event.set()
            wake_event.set()
            dpg.set_item_label('start', 'Stop')
    def reset_callback(b, a, d):
        reset_event.set()
        wake_event.set()
    def listen_callback(b, a, d):
        listen_event.set()
        wake_event.set()
    def save_callback(b, a, d):
        save_event.set()
        wake_event.set()
    with dpg.window(label='Controls', tag='primary'):
        with dpg.drawlist(width=SPECTROGRAM_WIDTH, height=SPECTROGRAM_HEIGHT):
            dpg.draw_image('spectrogram', (0, 0), (SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT), uv_min=(0, 1), uv_max=(1, 0))
        with dpg.child_window() as panel:
            with dpg.group(horizontal=True):
                with dpg.table(header_row=False, width=256):
                    for _ in range(COLS):
                        dpg.add_table_column()
                    for row_idx in range(ROWS):
                        with dpg.table_row():
                            for col_idx in range(min(len(control_array) - row_idx*COLS, COLS)):
                                dpg.add_knob_float(
                                    user_data=row_idx*COLS + col_idx, callback=knob_callback,
                                    default_value=0.0, min_value=-1.0, max_value=1.0
                                )
                with dpg.group():
                    dpg.add_button(label='Start', tag='start', callback=start_stop_callback)
                    dpg.add_button(label='Reset', tag='reset', callback=reset_callback)
                    dpg.add_button(label='Listen', tag='listen', callback=listen_callback)
                    dpg.add_button(label='Save', tag='save', callback=save_callback)

    dpg.bind_item_theme(panel, panel_theme)

    dpg.set_primary_window('primary', True)
    dpg.show_viewport()
    while dpg.is_dearpygui_running():
        with spectrogram.get_lock():
            local_spec_buffer[:] = spec_cmap(spectrogram.get_obj()).ravel()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()
    close_event.set()
    wake_event.set()

def control_routine_box(control_array):
    print('Initializing knob encoders...')
    i2c0 = busio.I2C(board.SCL, board.SDA)
    i2c1 = busio.I2C(board.SCL_1, board.SDA_1)
    tca0 = TCA9548A(i2c0, address=I2C_MUX_ADDR_0)
    tca1 = TCA9548A(i2c1, address=I2C_MUX_ADDR_1)
    sensors = []
    time.sleep(0.1)
    for chan in range(8):
        try:
            sensors.append(AS5600(tca0[chan]))
        except Exception as e:
            pass
        try:
            sensors.append(AS5600(tca1[chan]))
        except Exception as e:
            pass
        time.sleep(0.1)
    print('Done')

def main(checkpoint_path, cmap_path, device, steps, eps_ctx, ema_decoders=tuple()):
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    print('Loading model')
    model = ASURA.from_checkpoint(checkpoint_path, device=device)
    del model.c_v_net
    del model.c_v_net_ema
    del model.c_u_net
    del model.c_y_net
    del model.token_encoder
    del model.seq_encoder
    del model.seq_encoder_ema
    del model.seq_predictor
    del model.decoder_v_m
    del model.decoder_v_phi
    del model.decoder_v_m_ema
    del model.decoder_v_phi_ema
    if 'm' in ema_decoders:
        del model.decoder_u_m
        del model.decoder_y_m
    else:
        del model.decoder_u_m_ema
        del model.decoder_y_m_ema
    if 'phi' in ema_decoders:
        del model.decoder_u_phi
        del model.decoder_y_phi
    else:
        del model.decoder_u_phi_ema
        del model.decoder_y_phi_ema


    with open(os.path.join(cmap_path, 'config.json')) as cmap_cfg_file:
        cmap_cfg = json.load(cmap_cfg_file)['model']
    control_map = MLP(model.d_token, d_h=cmap_cfg['d_h'], layers=cmap_cfg.get('layers', 2), act_fn=cmap_cfg.get('act_fn', 'silu'), device=device)
    print('Loading control map')
    cmap_sd = torch.load(os.path.join(cmap_path, 'control_map.pt'), map_location=device)
    cmap_sd = {k.removeprefix('y_net_ema.'): v for k, v in cmap_sd.items() if k.startswith('y_net_ema.')}
    control_map.load_state_dict(cmap_sd)

    control_array = mp.Array('f', NUM_KNOBS)
    control_array_bufview = np.frombuffer(control_array.get_obj(), dtype=np.float32)
    spectrogram = mp.Array('f', SPECTROGRAM_WIDTH*SPECTROGRAM_HEIGHT)
    spectrogram_bufview = np.frombuffer(spectrogram.get_obj(), dtype=np.float32).reshape(SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH)
    spectrogram_ring = np.zeros((SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH), dtype=np.float32)
    spectrogram_ring_frame_idx = 0
    spec_fb = make_mel_filterbank(44100, model.frame_bins*2, SPECTROGRAM_HEIGHT)[:, :model.frame_bins].transpose(0, 1).to(device=device)

    wake_event = mp.Event()
    gen_event = mp.Event()
    reset_event = mp.Event()
    listen_event = mp.Event()
    save_event = mp.Event()
    close_event = mp.Event()

    save_path = os.path.join('outputs', 'play')
    save_count = 0

    control_axes = torch.randn((NUM_KNOBS, model.d_token), device=device)
    control_axes /= torch.linalg.norm(control_axes, ord=2, dim=-1, keepdim=True)

    if MODE == 'box':
        control_proc = mp.Process(target=control_routine_box, args=(control_array,), daemon=True)
    elif MODE == 'pc':
        control_proc = mp.Process(target=control_routine_pc, args=(
            control_array,
            spectrogram,
            wake_event, gen_event, reset_event, listen_event, save_event, close_event
        ), daemon=True)

    control_proc.start()

    # NUM_BUFFERED_CHUNKS = 4

    # chunks = [np.zeros((model.frame_hop, 2), dtype=np.float32) for _ in range(NUM_BUFFERED_CHUNKS)]
    # read_chunk = 0
    # write_chunk = 0

    sample_encs = torch.empty((1, model.length_tokens, model.d_token), device=device)
    x_m_frames = model.sample_prior((1, model.length_frames, model.frame_bins, model.d_signal), device=device)
    x_phi_frames = model.sample_prior((1, model.length_frames, model.frame_bins, model.d_signal_enc), device=device)
    y_m_frames = x_m_frames.clone()
    y_phi_frames = x_phi_frames.clone()

    prev_m_frames = model.sample_prior((1, model.decoder_lookback_frames, model.frame_bins, model.d_signal), device=device)
    prev_phi_frames = model.sample_prior((1, model.decoder_lookback_frames, model.frame_bins, model.d_signal_enc), device=device)
    prev_m_frames = list(prev_m_frames.unbind(dim=1))
    prev_phi_frames = list(prev_phi_frames.unbind(dim=1))

    c_ctx = torch.zeros((1, model.frames_per_token, model.d_token), device=device)
    c_lat = torch.zeros((1, model.d_token), device=device)

    # def fill_from_buffer(outdata: np.ndarray, frames: int, time, status):
    #     nonlocal read_chunk
    #     if read_chunk == write_chunk:
    #         outdata.fill(0.0)
    #     else:
    #         outdata[:] = chunks[read_chunk]
    #         read_chunk = (read_chunk + 1) % len(chunks)

    # out_stream = sounddevice.OutputStream(samplerate=44100.0, blocksize=model.frame_hop, channels=2, callback=fill_from_buffer)
    # out_stream.start()

    i = 0

    elapsed_samples = []

    while True:
        wake_event.wait()

        if close_event.is_set():
            break

        if listen_event.is_set():
            # render audio and call sounddevice.play
            signal = np.clip(model.frames_to_signal(y_phi_frames[:, :i])[0].cpu().numpy(), a_min=-1.0, a_max=1.0)
            print(signal.shape)
            sounddevice.play(signal, 44100)
            listen_event.clear()

        if reset_event.is_set():
            i = 0
            spectrogram_ring[:] = 0.0
            spectrogram_ring_frame_idx = 0
            with spectrogram.get_lock():
                spectrogram_bufview[:] = 0.0

            x_m_frames = model.sample_prior((1, model.length_frames, model.frame_bins, model.d_signal), device=device)
            x_phi_frames = model.sample_prior((1, model.length_frames, model.frame_bins, model.d_signal_enc), device=device)
            y_m_frames = x_m_frames.clone()
            y_phi_frames = x_phi_frames.clone()
            prev_m_frames = model.sample_prior((1, model.decoder_lookback_frames, model.frame_bins, model.d_signal), device=device)
            prev_phi_frames = model.sample_prior((1, model.decoder_lookback_frames, model.frame_bins, model.d_signal_enc), device=device)
            prev_m_frames = list(prev_m_frames.unbind(dim=1))
            prev_phi_frames = list(prev_phi_frames.unbind(dim=1))
            reset_event.clear()

        if save_event.is_set():
            signal = (np.clip(model.frames_to_signal(y_phi_frames[:, :i])[0].cpu().numpy(), a_min=-1.0, a_max=1.0) * 32767).astype(np.int16)
            os.makedirs(save_path, exist_ok=True)
            while os.path.exists(os.path.join(save_path, f"{save_count}.wav")):
                save_count += 1
            write_wav(os.path.join(save_path, f"{save_count}.wav"), 44100, signal)
            save_event.clear()

        if not gen_event.is_set():
            wake_event.clear()
            continue

        t0 = time.perf_counter_ns()

        control_vector = torch.from_numpy(control_array_bufview).to(device=device)[None] @ control_axes
        c_lat = control_vector + control_map(control_vector)

        model.sample_single_frame(i,
            x_m_frames, x_phi_frames,
            y_m_frames, y_phi_frames,
            prev_m_frames, prev_phi_frames,
            sample_encs, c_ctx, c_lat,
            steps=steps, eps_ctx=eps_ctx,
            ema_decoders=ema_decoders
        )
        spectrogram_ring[:, spectrogram_ring_frame_idx] = (x_m_frames[0, i, :, 0] @ spec_fb).cpu().numpy()
        spectrogram_ring_frame_idx = (spectrogram_ring_frame_idx + 1) % SPECTROGRAM_WIDTH
        i += 1
        if i == model.length_frames:
            i -= model.frames_per_token
            sample_encs = torch.roll(sample_encs, -1, 1)
            x_m_frames = torch.cat((
                x_m_frames[:, model.frames_per_token:],
                model.sample_prior((1, model.frames_per_token, model.frame_bins, model.d_signal), device=device)
            ), dim=1)
            x_phi_frames = torch.cat((
                x_phi_frames[:, model.frames_per_token:],
                model.sample_prior((1, model.frames_per_token, model.frame_bins, model.d_signal_enc), device=device)
            ), dim=1)
            y_m_frames = torch.cat((
                y_m_frames[:, model.frames_per_token:],
                model.sample_prior((1, model.frames_per_token, model.frame_bins, model.d_signal), device=device)
            ), dim=1)
            y_phi_frames = torch.cat((
                y_phi_frames[:, model.frames_per_token:],
                model.sample_prior((1, model.frames_per_token, model.frame_bins, model.d_signal_enc), device=device)
            ), dim=1)
        with spectrogram.get_lock():
            spectrogram_bufview[:, 0:SPECTROGRAM_WIDTH-spectrogram_ring_frame_idx] = spectrogram_ring[:, spectrogram_ring_frame_idx:]
            spectrogram_bufview[:, SPECTROGRAM_WIDTH-spectrogram_ring_frame_idx:] = spectrogram_ring[:, :spectrogram_ring_frame_idx]
        t1 = time.perf_counter_ns()
        elapsed = t1 - t0
        elapsed_samples.append(elapsed)
        if len(elapsed_samples) == 100:
            print(100/(sum(elapsed_samples) * 1e-9), 'frames/sec')
            elapsed_samples.clear()

if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument('checkpoint_path', type=str)
    argp.add_argument('cmap_path', type=str)
    argp.add_argument('-s', '--steps', type=int, default=1)
    argp.add_argument('-e', '--eps', type=float, default=0.02)
    argp.add_argument('--use_ema_m', action='store_true')
    argp.add_argument('--use_ema_phi', action='store_true')
    args = argp.parse_args()
    device = torch.device('cuda', 0) if torch.cuda.is_available() \
        else torch.device('mps') if torch.backends.mps.is_available() \
        else torch.device('cpu')
    ema_decoders = []
    if args.use_ema_m:
        ema_decoders.append('m')
    if args.use_ema_phi:
        ema_decoders.append('phi')
    main(args.checkpoint_path, args.cmap_path, device, args.steps, args.eps, ema_decoders=ema_decoders)
