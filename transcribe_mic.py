import os
import pyaudio
import wave
import tempfile
import openai
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox
from dotenv import load_dotenv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import struct
from matplotlib.animation import FuncAnimation
import datetime
import shutil

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sales Call Transcription")
        self.root.geometry("700x580")  # Further reduced height
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Audio recording parameters
        self.chunk = 1024
        self.sample_rate = 16000
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        
        # Recording state
        self.is_recording = False
        self.frames = []
        self.stream = None
        self.temp_file = None
        self.temp_audio_path = None
        
        # Speaker names
        self.speaker1 = "Daniel"
        self.speaker2 = "Speaker 2"
        
        # Audio visualization data - use multiples of chunk size
        self.audio_data = np.zeros(4096)  # 4 * chunk
        self.line = None
        self.animation = None
        
        # Create UI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Main control frame for buttons and waveform
        main_control_frame = tk.Frame(self.root)
        main_control_frame.pack(pady=10, fill=tk.X)
        
        # Left frame for buttons
        button_frame = tk.Frame(main_control_frame)
        button_frame.pack(side=tk.LEFT, padx=10)
        
        # Start button
        self.start_button = tk.Button(
            button_frame, 
            text="Start Recording", 
            command=self.start_recording,
            bg="#4CAF50",
            fg="black",
            height=2,
            width=15
        )
        self.start_button.pack(side=tk.TOP, pady=2)
        
        # Stop button (disabled initially)
        self.stop_button = tk.Button(
            button_frame, 
            text="Stop Recording", 
            command=self.stop_recording,
            bg="#F44336",
            fg="black",
            height=2,
            width=15,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.TOP, pady=2)
        
        # Right frame for waveform - match button dimensions
        waveform_frame = tk.Frame(main_control_frame)
        waveform_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        # Get button dimensions to match waveform size
        button_width = 15 * 8  # Approximate character width in pixels
        button_height = 2 * 20  # Approximate line height in pixels
        
        # Create a matplotlib figure for the waveform - sized to match buttons
        self.fig, self.ax = plt.subplots(figsize=(button_width/72, button_height/72))
        
        # Remove all axes elements
        self.ax.set_ylim(-32768, 32768)  # 16-bit audio range
        self.ax.set_xlim(0, 4096)
        self.ax.set_facecolor('#F0F0F0')
        
        # Remove all axis ticks, labels, and frame
        self.ax.set_xticks([]) 
        self.ax.set_yticks([])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        # Remove axis borders
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        
        # Create the line for the waveform
        self.line, = self.ax.plot([], [], lw=1.5, color='#007ACC')
        
        # Tight layout to remove extra space
        self.fig.tight_layout(pad=0)
        
        # Embed the matplotlib figure in tkinter - set size to match buttons
        self.canvas = FigureCanvasTkAgg(self.fig, master=waveform_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.config(width=button_width, height=button_height*2 + 4)  # Match two buttons + padding
        canvas_widget.pack()
        
        # Status label below controls
        self.status_label = tk.Label(self.root, text="Ready to record...")
        self.status_label.pack(pady=5)
        
        # Transcript display
        transcript_frame = tk.Frame(self.root)
        transcript_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Editable title for the transcript
        title_frame = tk.Frame(transcript_frame)
        title_frame.pack(fill=tk.X, pady=(0, 5))
        
        title_label = tk.Label(title_frame, text="Call Title:")
        title_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.title_entry = tk.Entry(title_frame, width=50)
        self.title_entry.insert(0, "Call with Mark from GMA")  # Default title
        self.title_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Transcript content
        self.transcript_text = scrolledtext.ScrolledText(
            transcript_frame, 
            wrap=tk.WORD,
            font=("Arial", 12)
        )
        self.transcript_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for different speakers
        self.transcript_text.tag_configure("speaker1", foreground="#0066CC")
        self.transcript_text.tag_configure("speaker2", foreground="#CC6600")
        
        # Save button at the bottom
        save_frame = tk.Frame(self.root)
        save_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.save_button = tk.Button(
            save_frame, 
            text="Save Recording", 
            command=self.save_recording,
            bg="#3498DB",
            fg="black",
            height=1,
            width=15
        )
        self.save_button.pack(side=tk.RIGHT)
        
    def update_plot(self, frame):
        """Update function for the audio visualization"""
        if self.is_recording and hasattr(self, 'audio_data'):
            self.line.set_data(range(len(self.audio_data)), self.audio_data)
        return self.line,
        
    def start_recording(self):
        """Start recording audio from microphone"""
        if not self.is_recording:
            self.is_recording = True
            self.frames = []
            self.transcript_text.delete(1.0, tk.END)
            self.audio_data = np.zeros(4096)
            
            # Update UI
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Update status with current title
            current_title = self.title_entry.get() or "Untitled Call"
            self.status_label.config(text=f"Recording: {current_title}...")
            
            # Start the animation for audio visualization
            self.animation = FuncAnimation(
                self.fig, self.update_plot, interval=30, 
                blit=True, cache_frame_data=False
            )
            
            # Open audio stream
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self.audio_callback
            )
            
            # Start the stream
            self.stream.start_stream()
            
            # Start a timer to periodically process audio chunks
            self.root.after(8000, self.process_audio_chunk)  # Increased to 8 seconds for better speaker detection
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        self.frames.append(in_data)
        
        # Update the audio visualization data
        try:
            # Convert the binary data to numpy array
            data_int = np.frombuffer(in_data, dtype=np.int16)
            
            # Ensure data_int is exactly chunk size
            data_len = min(len(data_int), self.chunk)
            
            # Shift existing data to the left by chunk size
            self.audio_data = np.roll(self.audio_data, -data_len)
            
            # Put new data at the end, making sure sizes match
            self.audio_data[-data_len:] = data_int[:data_len]
            
            # Redraw the canvas
            self.canvas.draw_idle()
        except Exception as e:
            print(f"Error updating audio visualization: {str(e)}")
            
        return (in_data, pyaudio.paContinue)
    
    def process_audio_chunk(self):
        """Process recorded audio chunks periodically"""
        if self.is_recording and len(self.frames) > 0:
            # Create a temporary copy of frames to process while recording continues
            frames_to_process = self.frames.copy()
            
            # Process in a separate thread to keep UI responsive
            threading.Thread(
                target=self.transcribe_frames,
                args=(frames_to_process,),
                daemon=True
            ).start()
            
            # Schedule next processing
            self.root.after(8000, self.process_audio_chunk)  # Increased to 8 seconds
    
    def transcribe_frames(self, frames):
        """Transcribe the given audio frames with speaker diarization"""
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        self.temp_audio_path = temp_file.name
        
        # Save frames to WAV file
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
        
        # Transcribe the audio with speaker diarization
        try:
            with open(temp_file.name, "rb") as file:
                # Use Whisper to transcribe - regular mode without verbose_json
                # Set language to German
                transcription = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=file,
                    language="de"
                )
            
            # Instead of using verbose_json, we'll use our own segmentation approach
            processed_text = self.simple_speaker_segmentation(transcription.text)
            
            # Update transcript in the UI thread
            self.update_transcript_in_ui(processed_text)
        
        except Exception as e:
            error_message = f"Error: {str(e)}"
            self.update_transcript_in_ui(error_message)
        
        finally:
            # We no longer delete the temp file here to allow for saving it later
            pass
    
    def update_transcript_in_ui(self, text):
        """Schedule the transcript update on the UI thread"""
        self.root.after(0, lambda: self.update_transcript(text))
    
    def simple_speaker_segmentation(self, text):
        """
        Apply a simple algorithm to segment text between two speakers.
        This is a heuristic approach based on sentence endings and typical dialog patterns.
        """
        if not text:
            return "No transcription available"
        
        # Split the text into sentences
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            # Consider sentence boundaries
            if char in ['.', '!', '?'] and current_sentence.strip():
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # Add any remaining text as a sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Assign speakers to sentences
        result = []
        current_speaker = self.speaker1  # Use Daniel instead of Speaker 1
        
        for i, sentence in enumerate(sentences):
            # Every other sentence, switch speakers (simple alternating pattern)
            if i > 0 and (i % 2 == 0 or len(sentence) > 30):  # Switch after every 2 sentences or long utterances
                current_speaker = self.speaker2 if current_speaker == self.speaker1 else self.speaker1
            
            # Add the sentence with speaker tag
            result.append(f"\n{current_speaker}: {sentence}")
        
        return "".join(result).strip()
    
    def update_transcript(self, text):
        """Update the transcript text box with speaker-differentiated text"""
        self.transcript_text.delete(1.0, tk.END)
        
        # Add title as first line if available
        current_title = self.title_entry.get()
        if current_title:
            self.transcript_text.insert(tk.END, f"--- {current_title} ---\n\n", "title")
        
        # Split text by speaker markers
        if f"{self.speaker1}:" in text and f"{self.speaker2}:" in text:
            # Find all speaker markers and their positions
            s1_positions = [i for i in range(len(text)) if text.startswith(f"{self.speaker1}:", i)]
            s2_positions = [i for i in range(len(text)) if text.startswith(f"{self.speaker2}:", i)]
            
            # Combine and sort all positions
            all_positions = [(pos, "speaker1") for pos in s1_positions] + [(pos, "speaker2") for pos in s2_positions]
            all_positions.sort()
            
            # Insert each part with the appropriate tag
            for i, (pos, tag) in enumerate(all_positions):
                # Find the end of this speaker's text
                if i < len(all_positions) - 1:
                    next_pos = all_positions[i+1][0]
                    speaker_text = text[pos:next_pos]
                else:
                    speaker_text = text[pos:]
                
                self.transcript_text.insert(tk.END, speaker_text, tag)
        else:
            # If no speaker differentiation, just insert the text
            self.transcript_text.insert(tk.END, text)
    
    def stop_recording(self):
        """Stop recording and process final audio"""
        if self.is_recording:
            self.is_recording = False
            
            # Stop the animation
            if self.animation:
                self.animation.event_source.stop()
            
            # Stop and close the stream
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            # Update UI
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Processing final audio...")
            
            # Process the complete recording in a separate thread
            if self.frames:
                threading.Thread(
                    target=self.process_final_recording,
                    daemon=True
                ).start()
    
    def process_final_recording(self):
        """Process the complete recording with speaker diarization"""
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        self.temp_audio_path = temp_file.name
        
        # Save the recorded audio as a WAV file
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
        
        # Transcribe the audio with speaker diarization
        try:
            with open(temp_file.name, "rb") as file:
                # Use Whisper to transcribe - regular mode without verbose_json
                # Set language to German
                transcription = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=file,
                    language="de"
                )
            
            # Process using our simple segmentation
            processed_text = self.simple_speaker_segmentation(transcription.text)
            
            # Update transcript text and status on the UI thread
            self.root.after(0, lambda: self.update_transcript(processed_text))
            
            current_title = self.title_entry.get() or "Untitled Call"
            self.root.after(0, lambda: self.status_label.config(text=f"Transcription complete: {current_title}"))
        
        except Exception as e:
            error_message = f"Error: {str(e)}"
            self.root.after(0, lambda: self.update_transcript(error_message))
            self.root.after(0, lambda: self.status_label.config(text="Error during transcription"))
        
        # We no longer delete the temp file here to allow for saving it later
    
    def save_recording(self):
        """Save the recording and transcript to Call_data directory"""
        if not self.temp_audio_path or not os.path.exists(self.temp_audio_path):
            messagebox.showerror("Error", "No recording available to save")
            return
            
        # Get the current transcript text
        transcript_text = self.transcript_text.get(1.0, tk.END).strip()
        if not transcript_text:
            messagebox.showerror("Error", "No transcript available to save")
            return
            
        # Create a unique directory name with timestamp
        call_title = self.title_entry.get() or "Untitled_Call"
        safe_title = "".join(c if c.isalnum() or c in [' ', '-', '_'] else '_' for c in call_title)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{safe_title}_{timestamp}"
        
        # Create directory path
        call_data_dir = os.path.join("Call_data", dir_name)
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(call_data_dir, exist_ok=True)
            
            # Copy audio file
            audio_dest = os.path.join(call_data_dir, "audio.wav")
            shutil.copy2(self.temp_audio_path, audio_dest)
            
            # Save transcript to text file
            transcript_path = os.path.join(call_data_dir, "transcript.txt")
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
                
            self.status_label.config(text=f"Saved to {call_data_dir}")
            messagebox.showinfo("Success", f"Recording and transcript saved to {call_data_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def on_closing(self):
        """Clean up resources when closing the app"""
        if self.is_recording and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        # Clean up temporary audio file if it exists
        if self.temp_audio_path and os.path.exists(self.temp_audio_path):
            try:
                os.unlink(self.temp_audio_path)
            except:
                pass
                
        self.audio.terminate()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main() 