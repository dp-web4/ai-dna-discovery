#!/usr/bin/env python3
"""
Save voice conversations to SSD for analysis and training
"""

import os
import wave
import json
import datetime
from pathlib import Path

CONVERSATION_DIR = Path("/mnt/sprout-data/conversations")

def setup_conversation_storage():
    """Set up conversation storage structure"""
    today = datetime.date.today()
    date_dir = CONVERSATION_DIR / today.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    return date_dir

def save_audio_segment(audio_data, sample_rate=16000):
    """Save audio segment with timestamp"""
    date_dir = setup_conversation_storage()
    timestamp = datetime.datetime.now().strftime("%H-%M-%S-%f")[:-3]
    
    # Save audio
    audio_path = date_dir / f"audio_{timestamp}.wav"
    with wave.open(str(audio_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    return audio_path

def save_transcription(text, audio_path, confidence=None):
    """Save transcription with metadata"""
    date_dir = audio_path.parent
    timestamp = audio_path.stem.replace("audio_", "")
    
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "audio_file": audio_path.name,
        "transcription": text,
        "confidence": confidence,
        "device": "AIRHUG Bluetooth",
        "model": "whisper-base",
        "gpu": True
    }
    
    # Save metadata
    meta_path = date_dir / f"transcript_{timestamp}.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Append to daily log
    log_path = date_dir / "conversation_log.txt"
    with open(log_path, 'a') as f:
        f.write(f"[{metadata['timestamp']}] {text}\n")
    
    return meta_path

def get_storage_stats():
    """Get storage statistics"""
    import shutil
    total, used, free = shutil.disk_usage("/mnt/sprout-data")
    
    stats = {
        "total_gb": total // (2**30),
        "used_gb": used // (2**30),
        "free_gb": free // (2**30),
        "conversations": len(list(CONVERSATION_DIR.rglob("*.wav"))),
        "transcriptions": len(list(CONVERSATION_DIR.rglob("*.json")))
    }
    
    return stats

if __name__ == "__main__":
    print("=== Conversation Storage Setup ===\n")
    
    # Set up storage
    date_dir = setup_conversation_storage()
    print(f"✅ Storage directory: {date_dir}")
    
    # Show stats
    stats = get_storage_stats()
    print(f"\n📊 Storage Statistics:")
    print(f"   Total: {stats['total_gb']}GB")
    print(f"   Used: {stats['used_gb']}GB")
    print(f"   Free: {stats['free_gb']}GB")
    print(f"   Conversations: {stats['conversations']}")
    print(f"   Transcriptions: {stats['transcriptions']}")
    
    print(f"\n✅ Ready to save conversations to SSD!")