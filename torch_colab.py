import argparse
import torch
import torchaudio
import os
import gc
import glob

# --- MATHEMATICAL LOGIC (UNTOUCHED) ---

def frequency_blend_phases(phase1, phase2, freq_bins, low_cutoff=500, high_cutoff=5000, base_factor=0.25, scale_factor=1.85):
    if phase1.shape != phase2.shape:
        raise ValueError("phase1 and phase2 must have the same shape.")
    if len(freq_bins) != phase1.shape[0]:
        raise ValueError("freq_bins must have the same length as the number of frequency bins.")
    
    blended_phase = torch.zeros_like(phase1)
    blend_factors = torch.zeros_like(freq_bins)
    blend_factors[freq_bins < low_cutoff] = base_factor
    blend_factors[freq_bins > high_cutoff] = base_factor + scale_factor
    in_range_mask = (freq_bins >= low_cutoff) & (freq_bins <= high_cutoff)
    blend_factors[in_range_mask] = base_factor + scale_factor * (
        (freq_bins[in_range_mask] - low_cutoff) / (high_cutoff - low_cutoff)
    )

    for i in range(phase1.shape[0]):
        blended_phase[i, :] = (1 - blend_factors[i]) * phase1[i, :] + blend_factors[i] * phase2[i, :]
    
    blended_phase = torch.remainder(blended_phase + torch.pi, 2 * torch.pi) - torch.pi
    return blended_phase

def transfer_magnitude_phase(source_file, target_file, transfer_magnitude=True, transfer_phase=True, low_cutoff=500, high_cutoff=5000, scale_factor=1.85, output_32bit=False, output_folder=None):
    target_basename = os.path.basename(target_file)
    target_name, target_ext = os.path.splitext(target_basename)
    
    # Cleaning name for output
    suffixes_to_strip = ["_other", "_vocals", "_instrumental", "_Other", "_Vocals", "_Instrumental", "_invert", "_instr"]
    clean_name = target_name
    for s in suffixes_to_strip:
        clean_name = clean_name.replace(s, "")
    clean_name = clean_name.strip()
    
    output_file = os.path.join(output_folder, f"{clean_name} (Fixed Instrumental){target_ext}")

    print(f"Phase Fixing: {target_basename}...")
    source_waveform, source_sr = torchaudio.load(source_file)
    target_waveform, target_sr = torchaudio.load(target_file)

    if source_sr != target_sr:
        raise ValueError("Sample rates must match.")

    n_fft, hop_length = 2048, 512
    window = torch.hann_window(n_fft)

    source_stfts = torch.stft(source_waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, pad_mode="reflect")
    target_stfts = torch.stft(target_waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, pad_mode="reflect")
    freqs = torch.linspace(0, source_sr // 2, steps=n_fft // 2 + 1)

    modified_stfts = []
    for source_stft, target_stft in zip(source_stfts, target_stfts):
        source_mag, source_phs = torch.abs(source_stft), torch.angle(source_stft)
        target_mag, target_phs = torch.abs(target_stft), torch.angle(target_stft)

        modified_stft = target_stft.clone()
        if transfer_magnitude:
            modified_stft = source_mag * torch.exp(1j * torch.angle(modified_stft))
        if transfer_phase:
            blended_phase = frequency_blend_phases(target_phs, source_phs, freqs, low_cutoff, high_cutoff, scale_factor)
            modified_stft = torch.abs(modified_stft) * torch.exp(1j * blended_phase)
        modified_stfts.append(modified_stft)

    modified_waveform = torch.istft(torch.stack(modified_stfts), n_fft=n_fft, hop_length=hop_length, window=window, length=source_waveform.size(1))
    
    torchaudio.save(output_file, modified_waveform, target_sr, encoding="PCM_S", bits_per_sample=32 if output_32bit else 16)
    print(f"Saved: {os.path.basename(output_file)}")

# --- RECURSIVE FILE MATCHING (CLEAN ASCII) ---

def get_clean_core(filename):
    name = os.path.splitext(filename)[0]
    # List of suffixes to ignore for matching
    test_suffixes = ["_other", "_vocals", "_instrumental", "_Other", "_Vocals", "_Instrumental", "_invert", "_instr"]
    for s in sorted(test_suffixes, key=len, reverse=True):
        if name.endswith(s):
            name = name[:-len(s)]
    return name.strip()

def process_files(base_folder, unwa_folder, output_folder, low_cutoff, high_cutoff, scale_factor, output_32bit):
    def get_all_audio_files(path):
        found_paths = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(('.flac', '.wav')):
                    found_paths.append(os.path.join(root, f))
        return found_paths

    all_unwa = get_all_audio_files(unwa_folder)
    all_base = get_all_audio_files(base_folder)

    print(f"Files detected - Base: {len(all_base)}, Unwa: {len(all_unwa)}")

    for u_path in all_unwa:
        u_basename = os.path.basename(u_path)
        u_core = get_clean_core(u_basename)
        
        # Searching for match in base_folder
        match_path = None
        for b_path in all_base:
            b_basename = os.path.basename(b_path)
            if get_clean_core(b_basename) == u_core:
                match_path = b_path
                break
        
        if match_path:
            transfer_magnitude_phase(
                source_file=match_path,
                target_file=u_path,
                transfer_magnitude=False,
                transfer_phase=True,
                low_cutoff=low_cutoff,
                high_cutoff=high_cutoff,
                scale_factor=scale_factor,
                output_32bit=output_32bit,
                output_folder=output_folder
            )
        else:
            print(f"Warning: No match for {u_basename}")
        
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", required=True)
    parser.add_argument("--unwa_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--low_cutoff", type=int, default=500)
    parser.add_argument("--high_cutoff", type=int, default=5000)
    parser.add_argument("--scale_factor", type=float, default=1.85)
    parser.add_argument("--output_32bit", action="store_true")
    args = parser.parse_args()

    process_files(
        base_folder=args.base_folder,
        unwa_folder=args.unwa_folder,
        output_folder=args.output_folder,
        low_cutoff=args.low_cutoff,
        high_cutoff=args.high_cutoff,
        scale_factor=args.scale_factor,
        output_32bit=args.output_32bit
    )