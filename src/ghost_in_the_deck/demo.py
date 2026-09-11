"""Deterministic, locally synthesized demonstration music (no downloaded media)."""
from pathlib import Path

import numpy as np
import soundfile as sf


def create_demo(directory: Path, *, seconds: float = 50, performance: bool = False) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    sr = 22050
    t = np.arange(round(seconds*sr))/sr
    for i, bpm in enumerate((120, 110, 126) if performance else (120, 118, 122)):
        interval = 60/bpm
        phase = np.remainder(t, interval)
        kick = .42*np.sin(2*np.pi*(55*phase + 3*(1-np.exp(-phase*30))))*np.exp(-phase*24)
        hat = .04*np.sin(2*np.pi*6500*t)*np.exp(-np.remainder(t,interval/2)*100)
        root = (110, 130.8128, 164.8138)[i]
        pulse = (.5-.5*np.cos(2*np.pi*t/(interval*16)))
        tone = .07*np.sin(2*np.pi*root*t)*pulse
        music = kick + hat + tone
        if performance and i != 1:
            # Purpose-built audible build for performance review; the middle
            # song stays steady. Standard --demo remains byte-identical.
            music *= .08+.92*np.clip((t-8)/10,0,1)
        fade = np.minimum(1, np.minimum(t/.1, (seconds-t)/.1))
        sf.write(directory/f'demo-{i+1}.wav', np.column_stack((music*fade, music*fade)), sr)
    return directory
