import numpy as np
import torch
from typing import Optional

from base_models.lpaps.lpaps import LPAPS
from base_models.lpaps.meta_clap_consistency import CLAPTextConsistencyMetric

def compute_lpaps_with_windows(aud1: torch.Tensor, aud1_sr: int, aud2: torch.Tensor, aud2_sr: int, model: LPAPS,
                               windows_size1: Optional[int] = None, windows_size2: Optional[int] = None,
                               overlap: float = 0.1, method: str = 'mean', device: str = 'cuda:0') -> float:
    """Calculate the LPAPS score for the given audio files, windowed. If windows_size1 or windows_size2 is None, it will default to 10 seconds.

    :param torch.Tensor aud1: The first audio file to compute LPAPS for
    :param int aud1_sr: The sample rate of the first audio file
    :param torch.Tensor aud2: The second audio file to compute LPAPS for
    :param int aud2_sr: The sample rate of the second audio file
    :param LPAPS model: The LPAPS model to use
    :param Optional[int] windows_size1: Window size in seconds for the first audio file. Defaults to 10 seconds (None)
    :param Optional[int] windows_size2: Window size in seconds for the second audio file. Defaults to 10 seconds (None)
    :param float overlap: The overlap factor of the windows, defaults to 0.1
    :param str method: method to use to combine scores, defaults to 'mean', choices=['mean', 'median', 'max', 'min']
    :param _type_ device: Torch device to use, defaults to 'cuda:0'
    :raises ValueError: Using an unknown method
    :return float: The combined LPAPS score
    """

    if windows_size1 is None:
        windows_size1 = int(aud1_sr * 10)
    if windows_size2 is None:
        windows_size2 = int(aud2_sr * 10)

    scores = []
    for i, j in zip(range(0, aud1.shape[-1], int(windows_size1 * (1 - overlap))),
                    range(0, aud2.shape[-1], int(windows_size2 * (1 - overlap)))):
        window1 = aud1[:, i:i + windows_size1]
        window2 = aud2[:, j:j + windows_size2]
        scores.append(model(window1.unsqueeze(0).to(device), window2.unsqueeze(0).to(device),
                      torch.tensor([aud1_sr], device=device),
                      torch.tensor([aud2_sr], device=device)).item())

    if method == 'mean':
        func = np.mean
    elif method == 'median':
        func = np.median
    elif method == 'max':
        func = np.max
    elif method == 'min':
        func = np.min
    else:
        raise ValueError(f'Unknown method: {method}')
    return func(scores)


def compute_clap_with_windows(aud: torch.Tensor, aud_sr: int, prompt: str, model: CLAPTextConsistencyMetric,
                              windows_size: Optional[int] = None, overlap: float = 0.1,
                              method: str = 'mean', device: torch.device = 'cuda:0') -> float:
    """Calculate the CLAP score for the given audio file and prompt, windowed. If windows_size is None, it will default to 10 seconds.

    :param torch.Tensor aud: The audio file to compute CLAP for
    :param int aud_sr: The sample rate of the audio file
    :param str prompt: The prompt to compute CLAP relative to
    :param CLAPTextConsistencyMetric model: The CLAP model to use
    :param Optional[int] windows_size: Window size in seconds. Defaults to 10 seconds (None)
    :param float overlap: The overlap factor of the windows, defaults to 0.1
    :param str method: method to use to combine scores, defaults to 'mean', choices=['mean', 'median', 'max', 'min']
    :param _type_ device: Torch device to use, defaults to 'cuda:0'
    :raises ValueError: Using an unknown method
    :return float: The combined CLAP score
    """
    if windows_size is None:
        windows_size = int(aud_sr * 10)
    scores = []
    for i in range(0, aud.shape[-1], int(windows_size * (1 - overlap))):
        window = aud[:, i:i + windows_size]
        model.update(window.unsqueeze(0).to(device), [prompt], torch.tensor([aud_sr], device=device))
        scores.append(model.compute())
        model.reset()
    if method == 'mean':
        func = np.mean
    elif method == 'median':
        func = np.median
    elif method == 'max':
        func = np.max
    elif method == 'min':
        func = np.min
    else:
        raise ValueError(f'Unknown method: {method}')
    return func(scores)


def calc_clap_win(clap_model: CLAPTextConsistencyMetric, aud: torch.Tensor, sr: int, target_prompt: str,
                  win_length: int, method: str, overlap: float, device: torch.device) -> float:
    """Calculate the CLAP score between an audio file and a prompt, with optional windowing

    :param CLAPTextConsistencyMetric clap_model: An initialized CLAP model to use
    :param torch.Tensor aud: The audio file to compute CLAP for
    :param int sr: The sample rate of the audio file
    :param str target_prompt: The prompt to compute CLAP relative to
    :param int win_length: The length of the window in seconds
    :param str method: The method to use to combine the scores, between 'mean', 'median', 'max', 'min'
    :param float overlap: The overlap fraction between windows in the range [0, 1]
    :param torch.device device: torch device to use
    :return float: The CLAP score
    """
    # `laion_clap` auto-places its underlying model on CUDA when CUDA is visible,
    # which mismatches the caller-requested device. Sync the metric to it.
    clap_model.to(device)
    if win_length is None:
        clap_model.update(aud.unsqueeze(0).to(device), [target_prompt], torch.tensor([sr], device=device))
        tmp = clap_model.compute()
        clap_model.reset()
        return tmp
    else:
        return compute_clap_with_windows(
            aud, sr, target_prompt, clap_model, device=device,
            windows_size=win_length * sr, overlap=overlap, method=method)


def calc_lpaps_win(lpaps_model: LPAPS, aud1: torch.Tensor, aud2: torch.Tensor, sr1: int, sr2: int,
                   win_length: int, method: str, overlap: float, device: torch.device) -> float:
    """Calculate the LPAPS score between two audio files, with optional windowing

    :param LPAPS lpaps_model: An initialized LPAPS model to use
    :param torch.Tensor aud1: First audio file
    :param torch.Tensor aud2: Second audio file
    :param int sr1: Sample rate of the first audio file
    :param int sr2: Sample rate of the second audio file
    :param int win_length: The length of the window in seconds
    :param str method: The method to use to combine the scores, between 'mean', 'median', 'max', 'min'
    :param float overlap: The overlap fraction between windows in the range [0, 1]
    :param torch.device device: torch device to use
    :return float: The LPAPS score
    """
    if win_length is None:
        return lpaps_model(aud1.unsqueeze(0).to(device),
                           aud2.unsqueeze(0).to(device),
                           torch.tensor([sr1], device=device),
                           torch.tensor([sr2], device=device)).item()
    else:
        return compute_lpaps_with_windows(aud1, sr1, aud2, sr2, lpaps_model,
                                          windows_size1=win_length * sr1,
                                          windows_size2=win_length * sr2,
                                          overlap=overlap, method=method, device=device)
